#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from typing import Any, Dict


def extract_rows(obj: Dict[str, Any]):
    image_id = obj.get("image_id", "")
    en_description = ""
    if isinstance(obj.get("image_desc_meta"), dict):
        en_description = obj["image_desc_meta"].get("en_description", "") or ""
    qa_open = []
    qam = obj.get("QA_meta") or {}
    # Some files might use "open-ended" key; fall back to "open-ended" only
    qa_open = qam.get("open-ended") or []
    # Ensure list
    if qa_open is None:
        qa_open = []
    rows = []
    for item in qa_open:
        if not isinstance(item, dict):
            continue
        en_question = item.get("en_question", "") or ""
        en_answer = item.get("en_answer", "") or ""
        en_question_hallucination = item.get("en_question_hallucination", "") or ""
        rows.append(
            (
                image_id,
                en_description,
                en_question,
                en_answer,
                en_question_hallucination,
            )
        )
    # If no open-ended entries, emit one row with empty QA fields to preserve image-level info
    if not rows:
        rows.append((image_id, en_description, "", "", ""))
    return rows


def parse_jsonl_to_tsv(infile: str, outfile: str):
    with open(infile, "r", encoding="utf-8") as inf, open(
        outfile, "w", encoding="utf-8", newline=""
    ) as outf:
        writer = csv.writer(outf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "image_id",
                "en_description",
                "en_question",
                "en_answer",
                "en_question_hallucination",
            ]
        )
        line_no = 0
        for line in inf:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: skipping invalid JSON at line {line_no}: {e}",
                    file=sys.stderr,
                )
                continue
            rows = extract_rows(obj)
            for r in rows:
                # ensure strings and strip newlines/tabs to keep TSV valid
                safe = [
                    (
                        s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
                        if isinstance(s, str)
                        else str(s)
                    )
                    for s in r
                ]
                writer.writerow(safe)


def main():
    p = argparse.ArgumentParser(
        description="Parse hallucination jsonl and write selected fields to TSV"
    )
    p.add_argument("-i", "--input", required=True, help="Input jsonl file")
    p.add_argument("-o", "--output", required=True, help="Output tsv file")
    args = p.parse_args()
    parse_jsonl_to_tsv(args.input, args.output)


if __name__ == "__main__":
    main()
