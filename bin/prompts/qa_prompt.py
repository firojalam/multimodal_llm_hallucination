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
from typing import Dict, List

# ----- Per-type SYSTEM/USER templates (short & strict) -----
SYSTEM_OPEN_ENDED = (
    "You are a culturally grounded, locality-aware expert QA assistant. Provide ONE short, factual answer with no explanation. "
    "Use the language specified in the input's `Language:` field. "
    "OUTPUT FORMAT (must be valid single-line JSON, no markdown or extra text): "
    '{{"answer":"<text>"}} '
    "Rules: do not add fields; escape internal double quotes; keep it concise."
)

USER_OPEN_ENDED = (
    "Language: {language}\n"
    "Question: {question}\n\n"
    'Respond ONLY with a single-line JSON object: {{"answer":"<text>"}}'
)

SYSTEM_MCQ = (
    "You are a culturally grounded, locality-aware expert QA assistant for single-answer multiple-choice questions. "
    "Use the language specified in the input's `Language:` field. "
    "Select exactly ONE option: return its text EXACTLY as shown and its 0-based index. "
    "OUTPUT FORMAT (valid single-line JSON, no markdown or extra text): "
    '{{"answer":"<option_text>","answer_index":<int>}} '
    "Rules: the `answer` must match the option string verbatim (including case/punctuation); "
    "`answer_index` must correspond to its 0-based position in the provided list; "
    "no explanations or additional fields."
)

USER_MCQ = (
    "Language: {language}\n"
    "Question: {question}\n"
    "Options (0-based JSON array): {options_json}\n\n"
    'Respond ONLY with: {{"answer":"<option_text>","answer_index":<int>}}'
)

SYSTEM_TRUE_FALSE = (
    "You are a culturally grounded, locality-aware expert QA assistant for True/False statements. "
    "Use the language specified in the input's `Language:` field. "
    "Return EXACT tokens only—English: True or False"
    "OUTPUT FORMAT (valid single-line JSON, no markdown or extra text): "
    '{{"answer":"<True/False token>"}} '
    "Rules: no explanations; no additional fields; match the exact token for the specified language."
)

USER_TRUE_FALSE = (
    "Language: {language}\n"
    "Statement: {statement}\n\n"
    'Respond ONLY with a single-line JSON object: {{"answer":"<True/False token>"}}'
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
def make_messages_open_ended(ex: Dict, **kw) -> List[Dict]:
    lang_prefix = kw.get("lang_prefix", "en")
    lang = _lang_code_from_prefix(lang_prefix)
    question = _pick(
        ex, lang_prefix, "question"
    )  # en_question / msa_question / arz_question
    return [
        {"role": "system", "content": SYSTEM_OPEN_ENDED},
        {
            "role": "user",
            "content": USER_OPEN_ENDED.format(language=lang, question=question),
        },
    ]


def make_messages_multiple_choice(ex: Dict, **kw) -> List[Dict]:
    lang_prefix = kw.get("lang_prefix", "en")
    lang = _lang_code_from_prefix(lang_prefix)
    question = _pick(ex, lang_prefix, "question")
    options = ex.get(f"{lang_prefix}_options") or ex.get("options")
    if not isinstance(options, list) or not options:
        raise KeyError(f"Missing or empty options list for prefix '{lang_prefix}'.")
    return [
        {"role": "system", "content": SYSTEM_MCQ},
        {
            "role": "user",
            "content": USER_MCQ.format(
                language=lang,
                question=question,
                options_json=json.dumps(options, ensure_ascii=False),
            ),
        },
    ]


def make_messages_true_false(ex: Dict, **kw) -> List[Dict]:
    lang_prefix = kw.get("lang_prefix", "en")
    lang = _lang_code_from_prefix(lang_prefix)

    statement = _pick(
        ex, lang_prefix, "question"
    )  # treat *_question as a T/F statement
    return [
        {"role": "system", "content": SYSTEM_TRUE_FALSE},
        {
            "role": "user",
            "content": USER_TRUE_FALSE.format(language=lang, statement=statement),
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


def make_messages(example: Dict, **kwargs) -> List[Dict]:
    # Determine QA type
    qtype = (
        (example.get("type") or example.get("qa_type") or "open-ended")
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    fn = PROMPT_REGISTRY.get(qtype)
    return fn(example, **kwargs)
