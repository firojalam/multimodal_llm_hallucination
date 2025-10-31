"""
Task-dependent QA prompt module.
Expose: make_messages(example: dict, **kwargs) -> List[{role, content}]

This version expects dataset items with language-prefixed fields:
- open-ended:  {en_question | msa_question | arz_question}
- multiple-choice: {<pfx>_question, <pfx>_options}
- true/false:  {<pfx>_question}  (used as a statement)

Select the prefix with kwarg "lang_prefix" (en|msa|arz). If absent, defaults to "en".
"""

import base64
import json
from typing import Dict, List, Optional


# ----- Per-type SYSTEM/USER templates (short & strict) -----
# ----- Vision-aware, transport-agnostic templates -----

SYSTEM_OPEN_ENDED = (
    "You are a culturally grounded, locality-aware expert VISION QA assistant. "
    "Exactly one image is attached to this message. "
    "Answer ONLY using what is visible in the image plus the question text; do not use outside knowledge. "
    "Provide ONE short, factual answer with no explanation. "
    "Use the language specified in the input's `Language:` field. "
    "OUTPUT FORMAT (valid single-line JSON, no markdown or extra text): "
    '{{"answer":"<text>"}} '
    "Rules: do not add fields; escape internal double quotes; keep it concise; "
    "do not mention or quote any URLs, file names, or base64 data."
)

USER_OPEN_ENDED = (
    "Language: {language}\n"
    "Question: {question}\n\n"
    "An image is attached with this message.\n"
    'Respond ONLY with a single-line JSON object: {{"answer":"<text>"}}'
)

SYSTEM_MCQ = (
    "You are a culturally grounded, locality-aware expert VISION QA assistant for single-answer multiple-choice questions. "
    "Exactly one image is attached to this message. "
    "Use ONLY visible evidence from the image plus the question text. "
    "Select exactly ONE option and return its text EXACTLY as shown and its 0-based index. "
    "OUTPUT FORMAT (single-line JSON, no markdown or extra text): "
    '{{"answer":"<option_text>","answer_index":<int>}} '
    "Rules: the `answer` must match the option string verbatim (case/punctuation); "
    "`answer_index` must match its 0-based position; no explanations or extra fields; "
    "do not mention or quote any URLs, file names, or base64 data."
)

USER_MCQ = (
    "Language: {language}\n"
    "Question: {question}\n"
    "Options (0-based JSON array): {options_json}\n\n"
    "An image is attached with this message.\n"
    'Respond ONLY with: {{"answer":"<option_text>","answer_index":<int>}}'
)

SYSTEM_TRUE_FALSE = (
    "You are a culturally grounded, locality-aware expert VISION QA assistant for True/False statements. "
    "Exactly one image is attached to this message. "
    "Judge the statement using ONLY visible evidence from the image plus the statement text. "
    "OUTPUT FORMAT (single-line JSON, no markdown or extra text): "
    '{{"answer":"<True/False>"}} '
    "Rules: return EXACTLY one of these tokens in English: True or False; "
    "no explanations; no additional fields; "
    "do not mention or quote any URLs, file names, or base64 data."
)

SYSTEM_TRUE_FALSE = """
You are an image-grounded validator for a single True/False statement, designed for hallucination detection.

### TASK: Image-Grounded Claim Validation
1.  **Judge Claim**: Classify the provided statement against the image pixels as **ENTAILED**, **CONTRADICTED**, or **NEI** (Not Enough Information).
2.  **Sole Source of Truth**: The **image is the ONLY source of truth**. **EXTERNAL/WORLD KNOWLEDGE IS FORBIDDEN**.
3.  **Prioritize Pixels**: Use the `Original Rationale` and `Auxiliary Image Description` only as context/hints; if they conflict with the pixels, **TRUST THE PIXELS**.

### DECISION DEFINITIONS
* **ENTAILED**: Clear and unambiguous visual evidence **supports** the claim.
* **CONTRADICTED**: Clear and unambiguous visual evidence **conflicts** with the claim.
* **NEI**: The claim **cannot be decided** from the visible pixels (due to occlusion, low resolution, invisible attributes, or requirement for external facts). **Be conservative and prefer NEI** when in doubt.

### OUTPUT FORMAT (STRICT JSON)
**Return STRICT JSON ONLY. NO extra text, NO markdown, NO comments, NO preamble.**
{
  "decidability_label": "ENTAILED" | "CONTRADICTED" | "NEI",
  "confidence": <float 0.00-1.00>,
  "label_mismatch": true | false,
  "pixel_evidence": "<**MAX 20 words**; literal description of visual proof>",
  "notes": "<**MAX 30 words**; brief notes on ambiguities or edge cases>",
  "rewrite_suggestion": "<optional: minimal rewrite to make the claim pixel-decidable (if NEI) or minimally contrastive (if needed). Use \"\" if not applicable.>"
}

### RULES & MISMATCH CHECK
* **Original Label Meaning**: The `Original Label` of **True** means the statement was originally asserted to be **ENTAILED** by the image. **False** means it was asserted to be **CONTRADICTED**.
* **Literal and Conservative**: Decisions must be literal interpretations of the visual data.
* **Mismatch Check**: If your final `decidability_label` (ENTAILED/CONTRADICTED/NEI) conflicts with the provided `Original Label` (True/False), you **MUST** set `"label_mismatch": true`.
* **Final Output**: **Do not output anything outside the JSON object.**
"""


# USER_TRUE_FALSE = (
#     "Language: {language}\n"
#     "Statement: {statement}\n\n"
#     "An image is attached with this message.\n"
#     'Respond ONLY with a single-line JSON object: {{"answer":"<True/False>"}}'
# )

USER_TRUE_FALSE = """

[IMAGE ATTACHED SEPARATELY]

### ORIGINAL PROVIDED DATA (FOR MISMATCH CHECK)
**Question/Statement**: {statement}             
**Original Answer**: {answer}         
**Original Rationale**: {answer_rationale} 

### IMAGE CONTEXT
**Auxiliary Image Description**: {image_description}  # optional

* **DECISION SOURCE**: Judge the statement **ONLY** from the pixels you see in the attached image.
* **CONSERVATISM**: If the statement is undecidable from the visual evidence, the only correct output is **NEI**.

"""


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
    ex: Dict, *, image_data: Optional[str] = None, **kw
) -> List[Dict]:
    lang_prefix = kw.get("lang_prefix", "en")
    lang = _lang_code_from_prefix(lang_prefix)
    question = _pick(
        ex, lang_prefix, "question"
    )  # en_question / msa_question / arz_question
    return [
        {"role": "system", "content": SYSTEM_OPEN_ENDED},
        {
            "role": "user",
            #  "content": USER_OPEN_ENDED.format(language=lang, question=question)
            "content": [
                {
                    "type": "text",
                    "text": USER_OPEN_ENDED.format(language=lang, question=question),
                },
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        },
    ]


def make_messages_multiple_choice(
    ex: Dict, *, image_data: Optional[str] = None, **kw
) -> List[Dict]:
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
            #  "content": USER_MCQ.format(language=lang, question=question, options_json=json.dumps(options, ensure_ascii=False))
            "content": [
                {
                    "type": "text",
                    "text": USER_MCQ.format(
                        language=lang,
                        question=question,
                        options_json=json.dumps(options, ensure_ascii=False),
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        },
    ]


def make_messages_true_false(
    ex: Dict, *, image_data: Optional[str] = None, **kw
) -> List[Dict]:
    lang_prefix = kw.get("lang_prefix", "en")
    lang = _lang_code_from_prefix(lang_prefix)
    # lang =language_map(lang_prefix)

    statement = _pick(
        ex, lang_prefix, "question"
    )  # treat *_question as a T/F statement
    answer = _pick(ex, lang_prefix, "answer")
    answer_rationale = _pick(ex, lang_prefix, "rationale")
    image_description = ex.get(f"{lang_prefix}_image_description", "")
    return [
        {"role": "system", "content": SYSTEM_TRUE_FALSE},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_TRUE_FALSE.format(
                        statement=statement,
                        answer=answer,
                        answer_rationale=answer_rationale,
                        image_description=image_description,
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data}},
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
    example: Dict, image_data: Optional[str] = None, **kwargs
) -> List[Dict]:
    qtype = (
        (example.get("type") or example.get("qa_type") or "open-ended")
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    fn = PROMPT_REGISTRY.get(qtype)
    return fn(example, image_data=image_data, **kwargs)
