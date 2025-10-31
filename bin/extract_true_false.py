#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

LANG_PREFIX_RE = re.compile(r"^[A-Za-z]{2,3}[_-]")  # e.g., en_, msa_, ar_, fr-, etc.


def parse_args():
    ap = argparse.ArgumentParser(
        description="Filter JSONL to English-only TF QAs, normalize en_answer_pred, optionally compute stats."
    )
    ap.add_argument(
        "--input", type=Path, required=True, help="Path to input .jsonl file"
    )
    ap.add_argument(
        "--output", type=Path, required=True, help="Path to output .jsonl file"
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help=(
            "If set, compute stats (overall decidability_label + CONTRADICTED breakdown "
            "+ en_answer counts + CONTRADICTED & en_answer==False + confusion matrix) "
            "and save TSV next to output."
        ),
    )
    return ap.parse_args()


def normalize_en_answer_pred(qa: Dict[str, Any]) -> Dict[str, Any]:
    """Convert qa['en_answer_pred'] from JSON string to dict (if needed)."""
    key = "en_answer_pred"
    if key in qa and isinstance(qa[key], str):
        s = qa[key].strip()
        if s:
            try:
                qa[key] = json.loads(s)
            except json.JSONDecodeError:
                # Keep the raw string if not valid JSON
                pass
    return qa


def is_non_english_lang_key(k: str) -> bool:
    """
    Returns True if k looks like a language-prefixed key that is NOT 'en_'.
    Drop: 'msa_*', 'ar_*', etc. Keep: 'en_*' and neutral keys.
    """
    if k.startswith("en_"):
        return False
    return bool(LANG_PREFIX_RE.match(k))


def filter_qa_to_english(qa: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep 'en_*' and neutral keys; drop other language-prefixed keys.
    Normalize en_answer_pred to a dict if it was a string.
    """
    kept = {}
    for k, v in qa.items():
        if k.startswith("en_") or not is_non_english_lang_key(k):
            kept[k] = v
    kept = normalize_en_answer_pred(kept)
    # Optionally remove extra metadata; comment these out if you want to keep them.
    kept.pop("cognitive_focus", None)
    kept.pop("semantic_focus", None)
    return kept


def filter_true_false_list(tf_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter every TF QA to English + neutral keys and normalize en_answer_pred."""
    return [filter_qa_to_english(tf) for tf in (tf_list or [])]


def build_output_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """Single output record per input item; preserve QA_meta.true_false as a filtered list."""
    qa_meta = item.get("QA_meta") or {}
    tf_list = qa_meta.get("true_false") or []
    tf_list_filtered = filter_true_false_list(tf_list)

    return {
        "image_id": item.get("image_id"),
        "image_path": item.get("image_path"),
        "image_url": item.get("image_url"),
        "QA_meta": {"true_false": tf_list_filtered},
    }


def collect_label_stats(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Count ALL decidability_label values from en_answer_pred across items."""
    counts: Dict[str, int] = {}
    for item in items:
        qa_meta = item.get("QA_meta") or {}
        for tf in qa_meta.get("true_false", []) or []:
            pred = tf.get("en_answer_pred")
            if not isinstance(pred, dict):
                continue
            label = pred.get("decidability_label")
            if label is None:
                continue
            norm_label = str(label).upper()
            counts[norm_label] = counts.get(norm_label, 0) + 1
    return counts


def _is_false_value(val: Any) -> bool:
    """True if val represents False (boolean False or string 'false' case-insensitive)."""
    if isinstance(val, bool):
        return val is False
    if isinstance(val, str):
        return val.strip().lower() == "false"
    return False


def collect_contradicted_stats(
    items: Iterable[Dict[str, Any]]
) -> Tuple[int, int, int, int]:
    """
    Return:
      (total_contradicted,
       contradicted_label_mismatch_true,
       contradicted_label_mismatch_false,
       contradicted_and_en_answer_false)
    """
    total_contra = 0
    contra_mismatch_true = 0
    contra_mismatch_false = 0
    contra_en_answer_false = 0

    for item in items:
        qa_meta = item.get("QA_meta") or {}
        for tf in qa_meta.get("true_false", []) or []:
            pred = tf.get("en_answer_pred")
            if not isinstance(pred, dict):
                continue
            label = str(pred.get("decidability_label", "")).upper()
            if label == "CONTRADICTED":
                total_contra += 1
                lm = pred.get("label_mismatch")
                if lm is True:
                    contra_mismatch_true += 1
                elif lm is False:
                    contra_mismatch_false += 1
                if _is_false_value(tf.get("en_answer")):
                    contra_en_answer_false += 1

    return (
        total_contra,
        contra_mismatch_true,
        contra_mismatch_false,
        contra_en_answer_false,
    )


def collect_en_answer_stats(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Count values of 'en_answer' across TF QAs (e.g., True/False), normalized to UPPER (or 'NONE')."""
    counts: Dict[str, int] = {}
    for item in items:
        qa_meta = item.get("QA_meta") or {}
        for tf in qa_meta.get("true_false", []) or []:
            ans = tf.get("en_answer")
            norm_upper = "NONE" if ans is None else str(ans).strip().upper()
            counts[norm_upper] = counts.get(norm_upper, 0) + 1
    return counts


def collect_confusion_matrix(
    items: Iterable[Dict[str, Any]]
) -> Tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    Build confusion matrix between decidability_label (rows) and en_answer (cols).
    - decidability_label: from en_answer_pred.decidability_label, uppercased; rows missing label are skipped.
    - en_answer: uppercased string; 'NONE' if missing.
    Returns (row_labels, col_answers, matrix_dict[row][col] = count)
    """
    matrix: Dict[str, Dict[str, int]] = {}
    row_labels_set: Set[str] = set()
    col_answers_set: Set[str] = set()

    for item in items:
        qa_meta = item.get("QA_meta") or {}
        for tf in qa_meta.get("true_false", []) or []:
            pred = tf.get("en_answer_pred")
            if not isinstance(pred, dict):
                continue
            label = pred.get("decidability_label")
            if label is None:
                continue
            row = str(label).upper()
            ans = tf.get("en_answer")
            col = "NONE" if ans is None else str(ans).strip().upper()

            row_labels_set.add(row)
            col_answers_set.add(col)
            matrix.setdefault(row, {})
            matrix[row][col] = matrix[row].get(col, 0) + 1

    rows = sorted(row_labels_set)
    cols = sorted(col_answers_set)
    return rows, cols, matrix


def save_stats_tsv(
    stats_path: Path,
    label_counts: Dict[str, int],
    contra_totals: Tuple[int, int, int, int],
    en_answer_counts: Dict[str, int],
    confusion_rows: List[str],
    confusion_cols: List[str],
    confusion: Dict[str, Dict[str, int]],
) -> None:
    """
    Write stats TSV with headers:
      metric\tcount
    Then append a confusion matrix in wide format:
      decidability_label\t<EN_ANSWER_COL1>\t<EN_ANSWER_COL2>...
    """
    total_contra, contra_true, contra_false, contra_ans_false = contra_totals
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        # Distribution blocks
        f.write("metric\tcount\n")
        for label in sorted(label_counts.keys()):
            f.write(f"decidability_label:{label}\t{label_counts[label]}\n")
        f.write(f"CONTRADICTED_total\t{total_contra}\n")
        f.write(f"CONTRADICTED_label_mismatch_true\t{contra_true}\n")
        f.write(f"CONTRADICTED_label_mismatch_false\t{contra_false}\n")
        f.write(f"CONTRADICTED_en_answer_FALSE\t{contra_ans_false}\n")
        for val in sorted(en_answer_counts.keys()):
            f.write(f"en_answer:{val}\t{en_answer_counts[val]}\n")

        # Blank line, then confusion matrix (wide TSV)
        f.write("\n")
        header = ["decidability_label"] + confusion_cols
        f.write("\t".join(header) + "\n")
        for row in confusion_rows:
            row_vals = [row] + [
                str(confusion.get(row, {}).get(col, 0)) for col in confusion_cols
            ]
            f.write("\t".join(row_vals) + "\n")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_records: List[Dict[str, Any]] = []
    total_in = 0

    # 1) Read input and build filtered output records (one per input line)
    with args.input.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed

            out_rec = build_output_record(item)
            output_records.append(out_rec)

    # 2) Write output JSONL (one record per input item)
    with args.output.open("w", encoding="utf-8") as fout:
        for rec in output_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3) Optionally compute + save stats
    if args.stats:
        label_counts = collect_label_stats(output_records)
        contra_totals = collect_contradicted_stats(output_records)
        en_answer_counts = collect_en_answer_stats(output_records)
        rows, cols, cmatrix = collect_confusion_matrix(output_records)
        # Same directory as output; file name with _stat.txt suffix
        stats_path = args.output.with_name(args.output.stem + "_stat.txt")
        save_stats_tsv(
            stats_path,
            label_counts,
            contra_totals,
            en_answer_counts,
            rows,
            cols,
            cmatrix,
        )
        print(f"[stats] Saved to: {stats_path}")

    print(
        f"Processed {total_in} input lines; wrote {len(output_records)} output lines to {args.output}"
    )


if __name__ == "__main__":
    main()
