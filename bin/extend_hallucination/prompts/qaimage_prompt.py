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


SYSTEM_TRUE_FALSE = """
You are a expert in creating pixel-decidable, hallucination-inducing True/False pairs for an image.

GOAL
Produce up to K NEW pairs drawn ONLY from the types below. Enforce diversity, anti-paraphrase, and anti-duplicate rules. Each pair must be minimal-contrast and pixel-decidable (no external knowledge).

ALLOWED TYPES
object, attribute, count, spatial, ocr_script, scene, culture-visual
('culture-visual' = visible motifs/garments/script/architecture only.)

STRICT TYPE COMPLIANCE
- Use ONLY types listed in ALLOWED TYPES.
- Aim to cover multiple distinct types when possible.
- If a type is not pixel-decidable (e.g., illegible text for ocr_script), DO NOT fabricate a pair; add it to unmet_types with a short reason.

DIFFICULTY
Choose lvl ∈ {e, m, h} by inspecting the image:
- e (easy): large/clear anchors; coarse spatial; counts {0–1}
- m (medium): mixed sizes/occlusion; add left/right; counts {0–2}
- h (hard): smaller cues/finer spatial or subtle attributes (still avoid NEI)
Use harder settings only when still clearly pixel-decidable.

MICRO-HINTS
Build a compact hint dict h (do not over-explain):
- anchors: 2–4 salient anchors you will target (e.g., ["road","terraces"])
- forbidden: risky objects to avoid if not visible (e.g., ["river","snow"])
- tgt_counts: an integer range to prefer (e.g., [0,2]) if counting is viable
- relation: allowed spatial relations actually supported by the scene (e.g., ["below","left_of"])
- bias: short priors to tease (e.g., ["assume_water","assume_car_on_road"])
- scr: boolean; true only if text is large/legible enough for ocr_script
These hints guide generation but MUST NOT override pixel-decidability.

NON-DUPLICATION & ANTI-PARAPHRASE
- No near-duplicates across outputs; no trivial paraphrases.
- Ensure visual novelty: vary anchors/relations/attributes/counts.
- When K ≥ 3, cover ≥ min(3, number_of_decidable_types) distinct types.
- Reject/replace any pair that conflicts with novelty checks (approximate by judgment):
  * trigram_overlap > 0.50 OR
  * normalized_edit_distance < 0.25 OR
  * semantic_similarity > 0.70.

PIXEL DECIDABILITY
- EXACTLY one claim per pair is ENTAILED by pixels (TRUE) and the other is CONTRADICTED (FALSE).
- ≤ 18 words/claim. One fact only. Avoid “all/always/none” unless trivially visible.
- If undecidable (tiny/occluded/illegible), omit the pair and mark that type unmet.

OUTPUT (STRICT JSON ONLY; no extra text):
{
  "q_id": "<string>",
  "meta": {
    "difficulty_level": "e|m|h",
    "h": {"anchors": [...], "forbidden": [...], "tgt_counts": [lo,hi], "relation": [...], "bias": [...], "script": true|false}
  },
  "coverage": {
    "requested_types": ["object","attribute","count","spatial","ocr_script","scene","culture-visual"],
    "met_types": ["<subset>"],
    "unmet_types": [{"type":"<t>","reason":"<short>"}]
  },
  "items": [
    {
      "type": "<one of ALLOWED TYPES>",
      "claim_true": "<entailed, <=18 words>",
      "claim_false": "<contradicted, <=18 words>",
      "minimal_change_note": "<what changed>",
      "anchors": ["<key visible objects/regions>"],
      "novelty": {
        "trigram_overlap_max": 0.00-1.00,
        "edit_distance_min": 0.00-1.00,
        "semantic_similarity_max": 0.00-1.00
      },
      "pixel_evidence_true": "<=15 words>",
      "pixel_evidence_false": "<=15 words>"
    }
  ]
}

CONDUCT
- Pixels over assumptions; prefer fewer pairs to low-novelty or non-decidable items.
- Respect your auto-inferred lvl and h, but never sacrifice pixel-decidability.
- Return ONLY the JSON object described above.
"""


USER_TRUE_FALSE = """

[IMAGE ATTACHED SEPARATELY]

K: {K}
seed_pair_json_optional:
{seed_pair_json}  

notes:
- Auto-select difficulty (lvl) and micro-hints (h) internally based on the image.
- Enforce novelty/anti-paraphrase, cover multiple types when possible, and omit any NEI-prone pair.
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


def make_messages_true_false(
    true_false_qa: Dict, *, image_data: Optional[str] = None, **kw
) -> List[Dict]:
    lang_prefix = kw.get("lang_prefix", "en")
    statement_0 = _pick(
        true_false_qa[0], lang_prefix, "question"
    )  # treat *_question as a T/F statement
    answer_0 = _pick(true_false_qa[0], lang_prefix, "answer")
    statement_1 = _pick(
        true_false_qa[1], lang_prefix, "question"
    )  # treat *_question as a T/F statement
    answer_1 = _pick(true_false_qa[1], lang_prefix, "answer")

    seed_qa = {}
    if answer_0.lower() == "true":
        seed_qa_0 = {"question": statement_0, "answer": answer_0}
        seed_qa["true"] = seed_qa_0
    if answer_1.lower() == "false":
        seed_qa_1 = {"question": statement_1, "answer": answer_1}
        seed_qa["false"] = seed_qa_1

    seed_qa_json = json.dumps(seed_qa, ensure_ascii=False, indent=2)

    return [
        {"role": "system", "content": SYSTEM_TRUE_FALSE},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_TRUE_FALSE.format(K=4, seed_pair_json=seed_qa_json),
                },
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        },
    ]


# ----- Dispatcher -----
PROMPT_REGISTRY = {
    # "open_ended": make_messages_open_ended,
    # "open-ended": make_messages_open_ended,
    # "openended": make_messages_open_ended,
    # "multiple_choice": make_messages_multiple_choice,
    # "multiple-choice": make_messages_multiple_choice,
    # "mcq": make_messages_multiple_choice,
    "true_false": make_messages_true_false,
    "true-false": make_messages_true_false,
    "tf": make_messages_true_false,
}


def make_messages(
    example: Dict, image_data: Optional[str] = None, **kwargs
) -> List[Dict]:
    qtype = "true_false"
    fn = PROMPT_REGISTRY.get(qtype)
    return fn(example, image_data=image_data, **kwargs)
