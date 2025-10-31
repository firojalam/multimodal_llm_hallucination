#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate hallucination (fabrication vs omission) pairs from OASIS.

Usage:
  # 1) Generate pairs, rule-based only (fast, no API calls)
  python generate_hallu_pairs.py \
      --input oasis.jsonl \
      --output pairs.jsonl

  # 2) Generate pairs with LLM refinement and multilingual output
  #    (requires OPENAI_API_KEY, or Azure settings)
  python generate_hallu_pairs.py \
      --input oasis.jsonl \
      --output pairs.jsonl \
      --use_llm \
      --languages en msa arz ajp

  # 3) Score predictions (predictions.jsonl must contain one line per claim_id)
  python generate_hallu_pairs.py score \
      --pairs pairs.jsonl \
      --predictions predictions.jsonl \
      --report report.json

Notes:
- We default to EN only. Add --languages msa arz ajp to try LLM-backed Arabic variants.
- Azure/OpenAI: set env vars below; the script will auto-detect.
"""

import argparse
import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ----------------------------
# Helpers
# ----------------------------


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s: Optional[str]) -> str:
    return (s or "").strip()


def any_in(txt: str, keywords: Iterable[str]) -> bool:
    t = txt.lower()
    return any(k in t for k in keywords)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def slugify(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"\W+", "_", s.strip().lower())
    return s[:maxlen].strip("_") or "item"


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ----------------------------
# Atom schema
# ----------------------------


@dataclass
class Atoms:
    scene_desert: bool = False  # True if desert scene
    sand_dunes: bool = False  # True if dunes present
    vegetation_absent: bool = False  # True if barren
    sky_partly_cloudy: bool = False
    sky_clear: bool = False  # if clearly stated
    lighting_warm_orange: bool = False  # warm/orange glow
    lighting_cool_blue: bool = False  # cool light
    ocr_texts: List[str] = None  # e.g., ["alamy"]
    # Optional free-text for provenance/debugging
    evidence_fields: List[str] = None

    def __post_init__(self):
        if self.ocr_texts is None:
            self.ocr_texts = []
        if self.evidence_fields is None:
            self.evidence_fields = []


# ----------------------------
# Rule-based atom extraction
# ----------------------------

# English + Arabic cue lists (minimal, extend as needed)
CUES = {
    "desert": ["desert", "sand dune", "sand dunes", "dune", "صحر", "صحرا", "كثبان"],
    "dunes": ["sand dune", "sand dunes", "dune", "كثبان", "تلال رمل"],
    "vegetation_absent": [
        "barren",
        "no vegetation",
        "absence of vegetation",
        "قاحل",
        "بدون نبات",
        "مفيش زرع",
        "لا يوجد نبات",
    ],
    "partly_cloudy": [
        "partly cloudy",
        "some clouds",
        "شوية سحاب",
        "غيوم جزئيا",
        "غائمة جزئيا",
    ],
    "clear_sky": ["clear sky", "no clouds", "سماء صافية", "بدون سحب"],
    "warm_orange": [
        "warm orange",
        "orange glow",
        "warm glow",
        "وهج برتقالي",
        "لون برتقالي",
        "ضوء دافئ",
    ],
    "cool_blue": ["cool bluish", "bluish light", "cold blue", "زرقة باردة", "ضوء أزرق"],
}


def extract_atoms_rule_based(rec: Dict[str, Any]) -> Atoms:
    atoms = Atoms()
    evidence = []

    # Collect candidate texts
    descs = []
    idm = rec.get("image_desc_meta", {})
    for k, v in idm.items():
        if k.endswith("_description") or k.endswith("_reason"):
            descs.append(norm(v))

    # Add QA rationales and questions (they often restate the facts)
    qa = rec.get("QA_meta", {})
    for tf in qa.get("true_false", []) or []:
        for field in [
            "en_question",
            "en_rationale",
            "msa_question",
            "msa_rationale",
            "arz_question",
            "arz_rationale",
            "ajp_question",
            "ajp_rationale",
        ]:
            val = norm(tf.get(field))
            if val:
                descs.append(val)
    for oe in qa.get("open-ended", []) or []:
        for field in [
            "en_answer",
            "en_rationale",
            "msa_answer",
            "msa_rationale",
            "arz_answer",
            "arz_rationale",
            "ajp_answer",
            "ajp_rationale",
        ]:
            val = norm(oe.get(field))
            if val:
                descs.append(val)

    # OCR direct
    ocr = norm(safe_get(rec, "image_desc_meta", "en_extracted_text", default=""))
    if ocr:
        atoms.ocr_texts.append(ocr)
        evidence.append("image_desc_meta.en_extracted_text")

    # Plain text aggregation
    blob = "\n".join([d for d in descs if d])
    blob_l = blob.lower()

    # Scene / dunes
    if any_in(blob_l, CUES["desert"]):
        atoms.scene_desert = True
        evidence.append("desc:desert")
    if any_in(blob_l, CUES["dunes"]):
        atoms.sand_dunes = True
        evidence.append("desc:dunes")

    # Vegetation
    if any_in(blob_l, CUES["vegetation_absent"]):
        atoms.vegetation_absent = True
        evidence.append("desc:vegetation_absent")

    # Sky
    if any_in(blob_l, CUES["partly_cloudy"]):
        atoms.sky_partly_cloudy = True
        evidence.append("desc:partly_cloudy")
    elif any_in(blob_l, CUES["clear_sky"]):
        atoms.sky_clear = True
        evidence.append("desc:clear_sky")

    # Lighting
    if any_in(blob_l, CUES["warm_orange"]):
        atoms.lighting_warm_orange = True
        evidence.append("desc:warm_orange")
    elif any_in(blob_l, CUES["cool_blue"]):
        atoms.lighting_cool_blue = True
        evidence.append("desc:cool_blue")

    atoms.evidence_fields = evidence
    return atoms


# ----------------------------
# Optional LLM refinement
# ----------------------------


def has_openai() -> bool:
    # OpenAI (Responses) or Azure OpenAI settings
    return bool(
        os.getenv("OPENAI_API_KEY")
        or (os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))
    )


def openai_client():
    """
    Returns a callable: (messages)-> str JSON for atoms
    Supports both OpenAI and Azure OpenAI (Responses API preferred; fallback to ChatCompletions).
    """
    # Lazy import to avoid hard dependency
    try:
        from openai import AzureOpenAI, OpenAI
    except Exception:
        return None

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    if azure_endpoint and azure_key:
        client = AzureOpenAI(
            api_key=azure_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=azure_endpoint,
        )

        def call(messages: List[Dict[str, str]]) -> str:
            # Try Responses API
            try:
                resp = client.responses.create(
                    model=azure_deployment,
                    input=[
                        {"role": m["role"], "content": m["content"]} for m in messages
                    ],
                    temperature=0.2,
                )
                return resp.output_text
            except Exception:
                # Fallback to chat.completions
                resp = client.chat.completions.create(
                    model=azure_deployment,
                    messages=messages,
                    temperature=0.2,
                )
                return resp.choices[0].message.content

        return call

    # Regular OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def call(messages: List[Dict[str, str]]) -> str:
        # Prefer Responses API
        try:
            resp = client.responses.create(
                model=default_model,
                input=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=0.2,
            )
            return resp.output_text
        except Exception:
            # Fallback to chat.completions
            resp = client.chat.completions.create(
                model=default_model,
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content

    return call


LLM_ATOM_SYSTEM = """You extract VISUAL atoms that are answerable from the image itself.
Return STRICT JSON with:
{
 "scene_desert": bool,
 "sand_dunes": bool,
 "vegetation_absent": bool,
 "sky_partly_cloudy": bool,
 "sky_clear": bool,
 "lighting_warm_orange": bool,
 "lighting_cool_blue": bool,
 "ocr_texts": [strings],
 "notes": "short rationale"
}
Only mark attributes that are visually inferable from the image text evidence. Keep OCR only if explicitly present in text.
"""


def refine_atoms_with_llm(rec: Dict[str, Any], base_atoms: Atoms) -> Atoms:
    client = openai_client()
    if client is None:
        return base_atoms

    # Pack the “evidence text” we already used
    idm = rec.get("image_desc_meta", {})
    qa = rec.get("QA_meta", {})
    texts = []
    for k, v in idm.items():
        if k.endswith("_description") or k.endswith("_reason"):
            if v:
                texts.append(f"{k}: {v}")
    for tf in qa.get("true_false", []) or []:
        for field in [
            "en_question",
            "en_rationale",
            "msa_question",
            "msa_rationale",
            "arz_question",
            "arz_rationale",
            "ajp_question",
            "ajp_rationale",
        ]:
            val = tf.get(field)
            if val:
                texts.append(f"{field}: {val}")
    for oe in qa.get("open-ended", []) or []:
        for field in [
            "en_answer",
            "en_rationale",
            "msa_answer",
            "msa_rationale",
            "arz_answer",
            "arz_rationale",
            "ajp_answer",
            "ajp_rationale",
        ]:
            val = oe.get(field)
            if val:
                texts.append(f"{field}: {val}")
    ocr = safe_get(rec, "image_desc_meta", "en_extracted_text")
    if ocr:
        texts.append(f"ocr: {ocr}")

    user_blob = "\n".join(texts)[:8000]

    messages = [
        {"role": "system", "content": LLM_ATOM_SYSTEM},
        {"role": "user", "content": user_blob},
    ]

    try:
        out = client(messages)
        # Attempt JSON parse
        data = json.loads(out.strip())
    except Exception:
        # If parsing fails, keep base atoms
        return base_atoms

    # Merge (prefer LLM when it asserts True/False explicitly; keep OCR merged)
    merged = Atoms(
        scene_desert=bool(data.get("scene_desert", base_atoms.scene_desert)),
        sand_dunes=bool(data.get("sand_dunes", base_atoms.sand_dunes)),
        vegetation_absent=bool(
            data.get("vegetation_absent", base_atoms.vegetation_absent)
        ),
        sky_partly_cloudy=bool(
            data.get("sky_partly_cloudy", base_atoms.sky_partly_cloudy)
        ),
        sky_clear=bool(data.get("sky_clear", base_atoms.sky_clear)),
        lighting_warm_orange=bool(
            data.get("lighting_warm_orange", base_atoms.lighting_warm_orange)
        ),
        lighting_cool_blue=bool(
            data.get("lighting_cool_blue", base_atoms.lighting_cool_blue)
        ),
        ocr_texts=list(
            set((base_atoms.ocr_texts or []) + (data.get("ocr_texts") or []))
        ),
        evidence_fields=list(set((base_atoms.evidence_fields or []) + ["llm_refined"])),
    )
    return merged


# ----------------------------
# Pair generation
# ----------------------------


def make_pair(
    record: Dict[str, Any],
    family: str,
    qminus_en: str,
    qplus_en: str,
    atoms_used: List[str],
    difficulty: str = "easy",
    languages: List[str] = None,
    llm_translate=None,
) -> Dict[str, Any]:
    """Create a paired item with English text; optionally add MSA/ARZ/AJP via LLM."""
    if languages is None:
        languages = ["en"]

    image_id = (
        record.get("image_id")
        or record.get("q_id")
        or slugify(record.get("image_url", ""))
    )
    pair_id = gen_id("pair")

    out = {
        "pair_id": pair_id,
        "image_id": image_id,
        "image_path": record.get("image_path"),
        "image_url": record.get("image_url"),
        "country": record.get("country"),
        "category": record.get("category"),
        "subcategory": record.get("subcategory"),
        "topic": record.get("topic"),
        "family": family,
        "difficulty": difficulty,
        "anchors": {"supports_atom": atoms_used, "evidence_fields": []},
        "modalities": ["TEXT", "TEXT+IMAGE", "SPEECH", "SPEECH+IMAGE"],
        "q_minus": {"gold": False, "en": qminus_en, "foil_strategy": ""},
        "q_plus": {"gold": True, "en": qplus_en},
    }

    # Add multilingual (optional)
    if any(lang in ("msa", "arz", "ajp") for lang in languages) and llm_translate:
        to_translate = {"qminus": qminus_en, "qplus": qplus_en}
        trs = llm_translate(
            to_translate, target_langs=[l for l in languages if l != "en"]
        )
        for lang_code, (qm, qp) in trs.items():
            out["q_minus"][lang_code] = qm
            out["q_plus"][lang_code] = qp

    return out


LLM_TRANSLATE_SYS = """You translate short evaluation statements about images.
- Keep meaning and polarity EXACT (true/false orientation must remain intact).
- Output STRICT JSON: { "msa": {"qminus": "...", "qplus": "..."}, "arz": {...}, "ajp": {...} }
Use natural, simple phrasing for each variety.
"""


def build_translator():
    client = openai_client()
    if client is None:
        return None

    def translate(
        batch: Dict[str, str], target_langs: List[str]
    ) -> Dict[str, Tuple[str, str]]:
        # batch = {"qminus": "...", "qplus": "..."}
        prompt = {
            "qminus": batch["qminus"],
            "qplus": batch["qplus"],
            "target_langs": target_langs,
        }
        messages = [
            {"role": "system", "content": LLM_TRANSLATE_SYS},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        try:
            out = client(messages)
            data = json.loads(out.strip())
        except Exception:
            # If anything fails, fall back to English only
            return {}
        results = {}
        for lang in target_langs:
            lang_obj = data.get(lang, {})
            qm = lang_obj.get("qminus")
            qp = lang_obj.get("qplus")
            if qm and qp:
                results[lang] = (qm, qp)
        return results

    return translate


def generate_pairs_for_record(
    rec: Dict[str, Any], atoms: Atoms, languages: List[str], llm_translate
) -> List[Dict[str, Any]]:
    pairs = []

    # 1) Scene identity (desert vs forest) — easy
    if atoms.scene_desert or atoms.sand_dunes:
        qminus = (
            "The image shows a dense forest with tall trees and abundant vegetation."
        )
        qplus = "The image shows a desert landscape with sand dunes illuminated by sunlight."
        p = make_pair(
            rec,
            "scene_identity",
            qminus,
            qplus,
            atoms_used=[
                "scene=desert",
                "structure=sand_dunes" if atoms.sand_dunes else "scene=desert",
            ],
            difficulty="easy",
            languages=languages,
            llm_translate=llm_translate,
        )
        p["q_minus"]["foil_strategy"] = "category_contrast"
        p["anchors"]["evidence_fields"] = atoms.evidence_fields
        pairs.append(p)

    # 2) Terrain feature (dunes vs rocky mountains) — easy
    if atoms.sand_dunes:
        qminus = "The landscape is dominated by rocky mountains rather than sand."
        qplus = "The landscape is dominated by undulating sand dunes."
        p = make_pair(
            rec,
            "terrain_feature",
            qminus,
            qplus,
            atoms_used=["structure=sand_dunes"],
            difficulty="easy",
            languages=languages,
            llm_translate=llm_translate,
        )
        p["q_minus"]["foil_strategy"] = "feature_swap"
        p["anchors"]["evidence_fields"] = atoms.evidence_fields
        pairs.append(p)

    # 3) Vegetation presence (absent vs lush) — easy
    if atoms.vegetation_absent:
        qminus = "There is lush green vegetation covering much of the terrain."
        qplus = "The terrain appears barren with no visible vegetation."
        p = make_pair(
            rec,
            "vegetation_presence",
            qminus,
            qplus,
            atoms_used=["vegetation=absent"],
            difficulty="easy",
            languages=languages,
            llm_translate=llm_translate,
        )
        p["q_minus"]["foil_strategy"] = "presence_flip"
        p["anchors"]["evidence_fields"] = atoms.evidence_fields
        pairs.append(p)

    # 4) Sky / weather — medium
    if atoms.sky_partly_cloudy:
        qminus = "The sky is completely clear with no clouds."
        qplus = "The sky is partly cloudy with blue tones."
        p = make_pair(
            rec,
            "sky_condition",
            qminus,
            qplus,
            atoms_used=["sky=partly_cloudy"],
            difficulty="medium",
            languages=languages,
            llm_translate=llm_translate,
        )
        p["q_minus"]["foil_strategy"] = "condition_flip"
        p["anchors"]["evidence_fields"] = atoms.evidence_fields
        pairs.append(p)
    elif atoms.sky_clear:
        qminus = "The sky is overcast with many clouds."
        qplus = "The sky looks clear with no visible clouds."
        p = make_pair(
            rec,
            "sky_condition",
            qminus,
            qplus,
            atoms_used=["sky=clear"],
            difficulty="medium",
            languages=languages,
            llm_translate=llm_translate,
        )
        p["q_minus"]["foil_strategy"] = "condition_flip"
        p["anchors"]["evidence_fields"] = atoms.evidence_fields
        pairs.append(p)

    # 5) Lighting / color — medium
    if atoms.lighting_warm_orange:
        qminus = "The dunes are lit with a cool bluish light."
        qplus = "Sunlight gives the dunes a warm orange glow."
        p = make_pair(
            rec,
            "lighting_color",
            qminus,
            qplus,
            atoms_used=["lighting=warm_orange"],
            difficulty="medium",
            languages=languages,
            llm_translate=llm_translate,
        )
        p["q_minus"]["foil_strategy"] = "attribute_flip"
        p["anchors"]["evidence_fields"] = atoms.evidence_fields
        pairs.append(p)
    elif atoms.lighting_cool_blue:
        qminus = "Sunlight gives the dunes a warm orange glow."
        qplus = "The dunes are lit with a cool bluish light."
        p = make_pair(
            rec,
            "lighting_color",
            qminus,
            qplus,
            atoms_used=["lighting=cool_blue"],
            difficulty="medium",
            languages=languages,
            llm_translate=llm_translate,
        )
        p["q_minus"]["foil_strategy"] = "attribute_flip"
        p["anchors"]["evidence_fields"] = atoms.evidence_fields
        pairs.append(p)

    # 6) OCR text — hard (because small visual evidence)
    if atoms.ocr_texts:
        # Use only first for simplicity; you can loop for all
        token = atoms.ocr_texts[0]
        qminus = f'The image contains the word "Getty" in the frame.'
        qplus = f'The image contains the word "{token}" in the frame.'
        p = make_pair(
            rec,
            "ocr_text",
            qminus,
            qplus,
            atoms_used=[f'ocr_text="{token}"'],
            difficulty="hard",
            languages=languages,
            llm_translate=llm_translate,
        )
        p["q_minus"]["foil_strategy"] = "text_corruption"
        p["anchors"]["evidence_fields"] = (atoms.evidence_fields or []) + ["ocr"]
        pairs.append(p)

    return pairs


# ----------------------------
# Scorer
# ----------------------------


def score_predictions(pairs_path: str, preds_path: str) -> Dict[str, Any]:
    # predictions.jsonl expects: {"claim_id": "...", "pred": true/false}
    # Our pairs file emits claims as pair_id+":minus" and pair_id+":plus"
    # Here, we will reconstruct per-pair metrics.
    preds = {}
    for rec in read_jsonl(preds_path):
        cid = rec.get("claim_id")
        prd = rec.get("pred")
        preds[cid] = bool(prd)

    fabrications = 0
    omissions = 0
    safe = 0
    helpful = 0
    well = 0
    total_pairs = 0

    # Optional: slice-wise tallies
    per_family = {}

    for p in read_jsonl(pairs_path):
        pid = p["pair_id"]
        fam = p.get("family", "_")
        qminus_id = pid + ":minus"
        qplus_id = pid + ":plus"
        # Default to False if missing pred (conservative)
        pred_minus = preds.get(qminus_id, False)
        pred_plus = preds.get(qplus_id, False)

        # Golds are fixed: minus=False, plus=True
        fabricated = 1 if pred_minus is True else 0
        omitted = 1 if pred_plus is False else 0

        pair_safe = 1 - fabricated
        pair_help = 1 - omitted
        pair_well = 1 if (pair_safe == 1 and pair_help == 1) else 0

        fabrications += fabricated
        omissions += omitted
        safe += pair_safe
        helpful += pair_help
        well += pair_well
        total_pairs += 1

        bucket = per_family.setdefault(
            fam, {"pairs": 0, "fabrications": 0, "omissions": 0, "well": 0}
        )
        bucket["pairs"] += 1
        bucket["fabrications"] += fabricated
        bucket["omissions"] += omitted
        bucket["well"] += pair_well

    def ratio(a, b):
        return 0.0 if b == 0 else float(a) / float(b)

    report = {
        "total_pairs": total_pairs,
        "fabrication_rate": ratio(fabrications, total_pairs),
        "omission_rate": ratio(omissions, total_pairs),
        "safe_rate": ratio(safe, total_pairs),
        "helpful_rate": ratio(helpful, total_pairs),
        "well_grounded_rate": ratio(well, total_pairs),
        "by_family": {
            fam: {
                "pairs": v["pairs"],
                "fabrication_rate": ratio(v["fabrications"], v["pairs"]),
                "omission_rate": ratio(v["omissions"], v["pairs"]),
                "well_grounded_rate": ratio(v["well"], v["pairs"]),
            }
            for fam, v in per_family.items()
        },
    }
    return report


# ----------------------------
# Main CLI
# ----------------------------


def main_generate(args):
    languages = args.languages or ["en"]
    use_llm = bool(args.use_llm)
    translator = build_translator() if use_llm else None

    # If user asked for LLM but credentials are missing, we continue rule-based and warn.
    if use_llm and not has_openai():
        print(
            "[WARN] --use_llm set but no OpenAI/Azure credentials found; continuing rule-based only."
        )
        use_llm = False
        translator = None

    out_records = []
    for rec in read_jsonl(args.input):
        # 1) rule-based atoms
        atoms = extract_atoms_rule_based(rec)
        # 2) optional LLM refinement (hybrid)
        if use_llm:
            atoms = refine_atoms_with_llm(rec, atoms)

        # 3) generate pairs
        pairs = generate_pairs_for_record(rec, atoms, languages, translator)

        # 4) flatten to per-claim lines (each pair becomes TWO claim lines if you want)
        #    You can choose between (a) emit pairs only, or (b) emit both pair & claims.
        #    Here, we emit PAIRS (one line per pair), plus include derived claim_ids
        #    for convenience in later scoring.
        for p in pairs:
            pid = p["pair_id"]
            p["claim_ids"] = {"minus": pid + ":minus", "plus": pid + ":plus"}
            out_records.append(p)

    write_jsonl(args.output, out_records)
    print(f"[OK] Wrote {len(out_records)} pairs to {args.output}")
    # Also write a small README-ish manifest
    manifest = {
        "task": "fabrication_vs_omission",
        "gold_policy": {"q_minus": False, "q_plus": True},
        "modalities": ["TEXT", "TEXT+IMAGE", "SPEECH", "SPEECH+IMAGE"],
        "metrics": ["fabrication", "omission", "safe", "helpful", "well_grounded"],
        "notes": "In TEXT-only settings without image, you may score unanswerable separately.",
    }
    with open(args.output + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote manifest to {args.output}.manifest.json")


def main_score(args):
    report = score_predictions(args.pairs, args.predictions)
    out_path = args.report or "report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote report to {out_path}")


def build_argparser():
    p = argparse.ArgumentParser(description="Generate hallucination pairs from OASIS")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="Generate pairs (default)")
    g.add_argument("--input", required=True, help="Path to OASIS JSONL")
    g.add_argument("--output", required=True, help="Output pairs JSONL")
    g.add_argument(
        "--use_llm", action="store_true", help="Refine atoms & translate via LLM"
    )
    g.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Languages to include: en msa arz ajp",
    )
    g.set_defaults(func=main_generate)

    s = sub.add_parser("score", help="Score predictions")
    s.add_argument("--pairs", required=True, help="Pairs JSONL")
    s.add_argument(
        "--predictions", required=True, help="Predictions JSONL with claim_id,pred"
    )
    s.add_argument("--report", required=False, help="Output report JSON")
    s.set_defaults(func=main_score)

    # default to generate if command omitted
    p.add_argument("--input", help="(shortcut) input JSONL (defaults to generate)")
    p.add_argument("--output", help="(shortcut) output JSONL (defaults to generate)")
    p.add_argument("--use_llm", action="store_true", help="(shortcut) for generate")
    p.add_argument("--languages", nargs="+", help="(shortcut) for generate")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    # Convenience: allow "python script.py --input ... --output ..." without subcmd
    if args.cmd is None:
        if not args.input or not args.output:
            parser.print_help()
            exit(1)
        args.cmd = "generate"
        args.func = main_generate

    # Normalize languages
    if getattr(args, "languages", None):
        args.languages = [l.lower() for l in args.languages]

    # Run
    args.func(args)
