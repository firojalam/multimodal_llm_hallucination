"""
Task-dependent QA prompt module.
Expose: make_messages(example: dict, **kwargs) -> List[{role, content}]

This version expects dataset items with language-prefixed fields:
- open-ended:  {en_question | msa_question | arz_question}
- multiple-choice: {<pfx>_question, <pfx>_options}
- true/false:  {<pfx>_question}  (used as a statement)

Select the prefix with kwarg "lang_prefix" (en|msa|arz). If absent, defaults to "en".
"""

import json
from typing import Dict, List, Optional


# ----- Audio-only question templates (short & strict) -----

# ----- Audio-only question templates (short & strict) -----

SYSTEM_OPEN_ENDED = (
    "You are a culturally grounded, locality-aware expert QA assistant. "
    "The user's question is provided as an attached audio clip. "
    "Answer ONLY using the audio content; do not use outside knowledge. "
    "Provide ONE short, factual answer with no explanation. "
    "Use the language specified in the input's `Language:` field. "
    "OUTPUT FORMAT (valid single-line JSON, no markdown or extra text): "
    '{{"answer":"<text>"}} '
    "Rules: do not add fields; escape internal double quotes; keep it concise; "
    "do not mention or quote any file names, URLs, or base64 data; do not include transcripts."
)

USER_OPEN_ENDED = (
    "Language: {language}\n"
    "The user's question is provided as an attached audio clip.\n\n"
    'Respond ONLY with a single-line JSON object: {{"answer":"<text>"}}'
)

SYSTEM_MCQ = (
    "You are a culturally grounded, locality-aware expert QA assistant for single-answer multiple-choice questions. "
    "The user's question is provided as an attached audio clip. "
    "Answer using ONLY the audio content. "
    "Select exactly ONE option and return its text EXACTLY as shown and its 0-based index. "
    "OUTPUT FORMAT (single-line JSON, no markdown or extra text): "
    '{{"answer":"<option_text>","answer_index":<int>}} '
    "Rules: the `answer` must match the option string verbatim (including case and punctuation); "
    "`answer_index` must match its 0-based position; no explanations or extra fields; "
    "do not mention or quote any file names, URLs, or base64 data; do not include transcripts."
)

USER_MCQ = (
    "Language: {language}\n"
    "Options (0-based JSON array): {options_json}\n"
    "The user's question is provided as an attached audio clip.\n\n"
    'Respond ONLY with: {{"answer":"<option_text>","answer_index":<int>}}'
)

SYSTEM_TRUE_FALSE = (
    "You are a culturally grounded, locality-aware expert QA assistant for True/False statements. "
    "The statement to evaluate is provided as an attached audio clip. "
    "Judge using ONLY the audio content. "
    "OUTPUT FORMAT (single-line JSON, no markdown or extra text): "
    '{{"answer":"<True/False>"}} '
    "Rules: return EXACTLY one of these tokens in English: True or False; "
    "no explanations; no additional fields; "
    "do not mention or quote any file names, URLs, or base64 data; do not include transcripts."
)

USER_TRUE_FALSE = (
    "Language: {language}\n"
    "The statement to evaluate is provided as an attached audio clip.\n\n"
    'Respond ONLY with a single-line JSON object: {{"answer":"<True/False>"}}'
)


# ----- Helpers -----
def _lang_code_from_prefix(pfx: str) -> str:
    LANG_MAP = {
        "en": "English",
        "msa": "Modern Standard Arabic",
        "ajp": "Levantine Arabic ",
        "arz": "Egyptian Arabic",
    }
    return LANG_MAP.get(pfx, "English")


def _pick(ex: Dict, pfx: str, base: str, *, required: bool = True):
    key = f"{pfx}_{base}"
    if key in ex and ex[key] is not None:
        return ex[key]
    # Fallbacks (some datasets may include non-prefixed copies)
    if base in ex and ex[base] is not None:
        return ex[base]
    if required:
        raise KeyError(f"Missing field: {key}")
    return None


# ----- Per-type builders -----
def make_messages_open_ended(
    ex: Dict, *, audio_data: Optional[str] = None, **kw
) -> List[Dict]:
    lang_prefix = kw.get("lang_prefix", "en")
    lang = _lang_code_from_prefix(lang_prefix)
    # question = _pick(ex, lang_prefix, "question")  # en_question / msa_question / arz_question
    return [
        {"role": "system", "content": SYSTEM_OPEN_ENDED},
        {
            "role": "user",
            #  "content": USER_OPEN_ENDED.format(language=lang, question=question)
            "content": [
                {
                    "type": "text",
                    "text": USER_OPEN_ENDED.format(language=lang),
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_data, "format": "wav"},
                },
            ],
        },
    ]


def make_messages_multiple_choice(
    ex: Dict, *, audio_data: Optional[str] = None, **kw
) -> List[Dict]:
    lang_prefix = kw.get("lang_prefix", "en")
    lang = _lang_code_from_prefix(lang_prefix)
    # question = _pick(ex, lang_prefix, "question")
    options = ex.get(f"{lang_prefix}_options") or ex.get("options")
    if not isinstance(options, list) or not options:
        raise KeyError(f"Missing or empty options list for prefix '{lang_prefix}'.")
    return [
        {"role": "system", "content": SYSTEM_MCQ},
        {
            "role": "user",
            #  "content": USER_MCQ.format(language=lang, question=question, options_json=json.dumps(options, ensure_ascii=False))
            "content": [
                {
                    "type": "text",
                    "text": USER_MCQ.format(
                        language=lang,
                        options_json=json.dumps(options, ensure_ascii=False),
                    ),
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_data, "format": "wav"},
                },
            ],
        },
    ]


def make_messages_true_false(
    ex: Dict, *, audio_data: Optional[str] = None, **kw
) -> List[Dict]:
    lang_prefix = kw.get("lang_prefix", "en")
    lang = _lang_code_from_prefix(lang_prefix)
    # statement = _pick(ex, lang_prefix, "question")  # treat *_question as a T/F statement
    return [
        {"role": "system", "content": SYSTEM_TRUE_FALSE},
        {
            "role": "user",
            #  "content": USER_TRUE_FALSE.format(language=lang, statement=statement)
            "content": [
                {
                    "type": "text",
                    "text": USER_TRUE_FALSE.format(language=lang),
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_data, "format": "wav"},
                },
            ],
        },
    ]


# ----- Dispatcher -----
PROMPT_REGISTRY = {
    "open_ended": make_messages_open_ended,
    "open-ended": make_messages_open_ended,
    "openended": make_messages_open_ended,
    "multiple_choice": make_messages_multiple_choice,
    "multiple-choice": make_messages_multiple_choice,
    "mcq": make_messages_multiple_choice,
    "true_false": make_messages_true_false,
    "true-false": make_messages_true_false,
    "tf": make_messages_true_false,
}


def make_messages(
    example: Dict, audio_data: Optional[str] = None, **kwargs
) -> List[Dict]:
    # Determine QA type
    qtype = (
        (example.get("type") or example.get("qa_type") or "open-ended")
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    fn = PROMPT_REGISTRY.get(qtype)
    return fn(example, audio_data=audio_data, **kwargs)
