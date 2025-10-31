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


SYSTEM_OPEN_ENDED = """You are a benchmark item designer for IMAGE-GROUNDED open-ended QA.

GOAL
Given an image and optional seed questions, produce K pixel-decidable OE items. 
- If a seed is UNGROUNDED (answerable without pixels or not visible), REWRITE it into a pixel-decidable question.
- Each final item targets exactly ONE hallucination type:
  {object, attribute, count, spatial, ocr_script, scene, action, culture-visual}.
  ('culture-visual' = visible motifs/garments/script/architecture only; NEVER claim country/ethnicity.)
- Keep answers as short spans (single word/short NP) or integers. If truly impossible to ground, drop the seed.

DECIDABILITY & RULES
- Use PIXELS ONLY. No world knowledge. If text is too small/illegible, do not create ocr_script items.
- One fact per question. No compound (“and/or”).
- Prefer absence/negatives as questions only if clearly decidable (e.g., “What vehicle is inside the circle?” → NEI avoided by rewriting to visible object).
- Keep language concise and neutral.

OUTPUT (STRICT JSON ONLY; no extra text):
{
  "q_id": "<string>",
  "coverage": {
    "types_used": ["object","attribute", "..."],
    "dropped_seeds": [{"seed_q":"<text>","reason":"ungrounded|ambiguous|illegible"}]
  },
  "items": [
    {
      "status": "kept|rewritten|new",
      "type": "<object|attribute|count|spatial|ocr_script|scene|action|culture-visual>",
      "question": "<pixel-decidable OE question>",
      "answer": "<short answer>",
      "evidence": {"reason":"<=20 words","boxes": [[x1,y1,x2,y2]]},  // boxes optional
      "difficulty": "e|m|h"
    }
  ]
}

CONDUCT
- Pixels over priors. Favor fewer high-quality items over forced coverage.
- If a seed is ungrounded (e.g., asks country, brand, unreadable text), rewrite to a VISIBLE property (type above).
- Return ONLY the JSON object described."""


USER_OPEN_ENDED = """q_id: {q_id}
[IMAGE ATTACHED SEPARATELY]

K: {K}                             # max items to return (you may return fewer)
seed_qa_optional:
{seed_qa_json}                     # list like [{"q":"..."}] or "null"

notes:
- Use pixels only; rewrite ungrounded seeds to visible properties targeting {object, attribute, count, spatial, ocr_script, scene, action, culture-visual}.
- Keep answers short (single word/NP or integer). Omit ocr_script if text is not legible.
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


# def make_messages_true_false(
#     true_false_qa: Dict, *, image_data: Optional[str] = None, **kw
# ) -> List[Dict]:
#     lang_prefix = kw.get("lang_prefix", "en")
#     statement_0 = _pick(
#         true_false_qa[0], lang_prefix, "question"
#     )  # treat *_question as a T/F statement
#     answer_0 = _pick(
#         true_false_qa[0], lang_prefix, "answer"
#     )
#     statement_1 = _pick(
#         true_false_qa[1], lang_prefix, "question"
#     )  # treat *_question as a T/F statement
#     answer_1 = _pick(
#         true_false_qa[1], lang_prefix, "answer"
#     )


#     seed_qa={}
#     if(answer_0.lower()=="true"):
#         seed_qa_0={
#             "question":statement_0,
#             "answer":answer_0
#         }
#         seed_qa["true"]=seed_qa_0
#     if(answer_1.lower()=="false"):
#         seed_qa_1={
#             "question":statement_1,
#             "answer":answer_1
#         }
#         seed_qa["false"]=seed_qa_1

#     seed_qa_json = json.dumps(seed_qa, ensure_ascii=False, indent=2)

#     return [
#         {"role": "system", "content": SYSTEM_TRUE_FALSE},
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": USER_TRUE_FALSE.format(K=4,seed_pair_json=seed_qa_json),
#                 },
#                 {"type": "image_url", "image_url": {"url": image_data}},
#             ],
#         },
#     ]


# ----- Dispatcher -----
PROMPT_REGISTRY = {
    "open_ended": make_messages_open_ended,
    "open-ended": make_messages_open_ended,
    "openended": make_messages_open_ended,
    # "multiple_choice": make_messages_multiple_choice,
    # "multiple-choice": make_messages_multiple_choice,
    # "mcq": make_messages_multiple_choice,
    # "true_false": make_messages_true_false,
    # "true-false": make_messages_true_false,
    # "tf": make_messages_true_false,
}


def make_messages(
    example: Dict, image_data: Optional[str] = None, **kwargs
) -> List[Dict]:
    qtype = "open_ended"
    fn = PROMPT_REGISTRY.get(qtype)
    return fn(example, image_data=image_data, **kwargs)
