# ======================================
# bin/llm_judge/fetch_and_merge_results.py
# ======================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch and merge results for Azure OpenAI Batch.

Tweaks for your setup:
- **No question_id_field arg**. We parse question_id from custom_id (e.g., "{id}::q=open-ended::lang=msa").
- **Group by base id** and merge **all four answers back into a single original item** under QA_meta.
- Write predictions to the correct per-language key: {lang_prefix}_answer_pred
  and, for MCQ, also {lang_prefix}_answer_pred_index if present.

Usage example:
  python bin/llm_judge/fetch_and_merge_results.py \
    --batch_file  cached_dir/batch_tracking.json \
    --env_file    envs/tanbih-azure-gpt4.1_batch.env \
    --output_dir  cached_dir/ \
    --output_file cached_dir/merged.jsonl \
    --output_error_file cached_dir/merged_errors.jsonl \
    --retrieve True \
    --original_file cached_dir/data.jsonl \
    --id_field id \
    --custom_id_sep "::q" \
    --lang_prefix msa
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_env(env_path: str):
    load_dotenv(dotenv_path=env_path, override=True)
    api_base = os.environ["AZURE_API_URL"].rstrip("/")
    api_key = os.environ["AZURE_API_KEY"]
    api_version = os.environ["AZURE_API_VERSION"]
    return api_key, api_base, api_version


# ---- Minimal inline manager ----
class AzureOpenAIBatchManager:
    def __init__(
        self, api_key: str, api_endpoint: str, api_version: str, batch_file_name: str
    ):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            azure_endpoint=api_endpoint, api_key=api_key, api_version=api_version
        )
        self.batch_file_name = batch_file_name

    def retrieve_all_submitted_batches(self, batch_output_dir: str):
        os.makedirs(batch_output_dir, exist_ok=True)
        outputs, errors = [], []
        if not os.path.exists(self.batch_file_name):
            logging.warning("No tracking file found; nothing to retrieve.")
            return outputs, errors
        with open(self.batch_file_name, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    batch_id, _ = line.strip().split(",", 1)
                except ValueError:
                    continue
                try:
                    b = self.client.batches.retrieve(batch_id)
                except Exception as e:
                    logging.error(f"Failed to retrieve batch {batch_id}: {e}")
                    continue
                status = getattr(b, "status", None)
                logging.info(f"Batch {batch_id} status={status}")
                if status != "completed":
                    continue
                ofid = getattr(b, "output_file_id", None)
                logging.info(f"Batch file id: {ofid}.")
                if ofid:
                    try:
                        content = self.client.files.content(ofid).text
                        out_path = os.path.join(
                            batch_output_dir, f"batch_output_{batch_id}.jsonl"
                        )
                        with open(out_path, "w", encoding="utf-8") as w:
                            w.write(content)
                        outputs.append(out_path)
                        logging.info(f"Wrote output -> {out_path}")
                    except Exception as e:
                        logging.error(f"Failed to download output for {batch_id}: {e}")
                efid = getattr(b, "error_file_id", None)
                if efid:
                    try:
                        econtent = self.client.files.content(efid).text
                        err_path = os.path.join(
                            batch_output_dir, f"batch_output_{batch_id}_error.jsonl"
                        )
                        with open(err_path, "w", encoding="utf-8") as w:
                            w.write(econtent)
                        errors.append(err_path)
                        logging.info(f"Wrote error file -> {err_path}")
                    except Exception as e:
                        logging.error(
                            f"Failed to download error file for {batch_id}: {e}"
                        )
        return outputs, errors


# -------- Parsing & merging --------


def parse_response_file(jsonl_path: str) -> Dict[str, Dict]:
    """Parse batch output JSONL into {custom_id: {response_raw, model}}."""
    results: Dict[str, Dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("custom_id")
                if not cid:
                    continue
                resp = rec.get("response", {})
                body = resp.get("body", {})
                model = body.get("model")
                content = None
                try:
                    content = body["choices"][0]["message"]["content"]
                except Exception:
                    pass
                if content is None:
                    continue
                results[cid] = {"response_raw": content, "model": model}
            except Exception as e:
                logging.error(f"Failed to parse line in {jsonl_path}: {e}")
    return results


def _strip_fences(text: str) -> str:
    # if text is None:
    # return ""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    text = re.sub(r"^```(?:json)?\n|\n```$", "", text)
    return text.strip()


def parse_answer_payload(raw: str) -> Optional[Dict[str, Any]]:
    # print(f"raw content: {type(raw)}")
    # if not raw:
    # return None
    txt = _strip_fences(raw)
    try:
        json_obj = json.loads(txt)
        return json_obj
    except Exception:
        logging.info("Answer is not valid JSON; storing raw content.")
        return None


_QID_RE = re.compile(r"^(?P<base>.+?)::q(?:=)?(?P<qid>[^:]+)(?:::.+)?$")


def split_custom_id(custom_id: str, default_sep: str = "::q") -> Tuple[str, str]:
    """Robustly parse custom_id to (base_id, question_id).
    Accepts formats like "<id>::q=<qid>" or "<id>::q<qid>" and ignores trailing decorations.
    """
    m = _QID_RE.match(custom_id)
    if m:
        return m.group("base"), m.group("qid")
    # Fallback: split once on the provided sep
    if default_sep in custom_id:
        base, qid = custom_id.split(default_sep, 1)
        if "::" in qid:
            qid = qid.split("::", 1)[0]
        return base, qid
    return custom_id, ""


def _qa_path_for_qid(record: Dict[str, Any], qid: str) -> Optional[Tuple[str, int]]:
    """Map a question_id label to (section, index) in QA_meta."""
    qa = record.get("QA_meta") or {}
    if qid == "open-ended":
        return ("open-ended", 0) if (qa.get("open-ended") or []) else None
    if qid == "multiple-choice":
        return ("multiple-choice", 0) if (qa.get("multiple-choice") or []) else None
    if qid.startswith("true_false_"):
        try:
            i = int(qid.split("_", 2)[2])
        except Exception:
            return None
        tf_list = qa.get("true_false") or []
        if 0 <= i < len(tf_list):
            return ("true_false", i)
    return None


def inject_predicted(
    record: Dict[str, Any],
    qid: str,
    lang_prefix: str,
    payload: Optional[Dict[str, Any]],
    raw_text: str,
):
    path = _qa_path_for_qid(record, qid)
    if not path:
        logging.warning(
            f"No QA_meta slot for question_id='{qid}' in record id={record.get('id')}"
        )
        return
    section, idx = path
    qa = record.setdefault("QA_meta", {})
    lst = qa.get(section)
    if not isinstance(lst, list) or not (0 <= idx < len(lst)):
        logging.warning(
            f"Malformed QA_meta for section='{section}' id={record.get('id')}"
        )
        return
    item = lst[idx]
    key_pred = f"{lang_prefix}_answer_pred"

    if payload is None:
        item[key_pred] = raw_text
        return

    ans = payload.get("answer")
    # write main prediction
    item[key_pred] = ans if ans is not None else raw_text

    # For MCQ, also propagate index if provided
    if section == "multiple-choice" and "answer_index" in payload:
        item[f"{lang_prefix}_answer_pred_index"] = payload.get("answer_index")


# ------------------------
# CLI
# ------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fetch batch results and merge into original QA_meta (single item per id)."
    )
    parser.add_argument(
        "--batch_file",
        required=True,
        help="Tracking file with lines: <batch_id>,<local_batch_jsonl>",
    )
    parser.add_argument("--env_file", required=True, help="Path to .env")
    parser.add_argument(
        "--output_dir", required=True, help="Dir to store fetched outputs"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Merged results JSONL (one line per base id)",
    )
    parser.add_argument(
        "--output_error_file", required=True, help="Original rows that errored (JSONL)"
    )
    parser.add_argument(
        "--retrieve",
        default="True",
        help="True/False to actively fetch from API; else process local",
    )
    parser.add_argument(
        "--original_file", required=True, help="Original input JSONL (contains QA_meta)"
    )
    parser.add_argument("--id_field", default="image_id")
    parser.add_argument(
        "--custom_id_sep",
        default="::q",
        help="Separator used between base id and question_id in custom_id (regex also accepts ::q=<qid>)",
    )
    parser.add_argument(
        "--lang_prefix",
        choices=["en", "msa", "arz"],
        required=True,
        help="Which language prefix to write *_answer_pred under",
    )

    args = parser.parse_args()
    configure_logging()
    retrieve_flag = str(args.retrieve).lower() == "true"
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.dirname(args.output_file) or "."
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 1) Retrieve batch outputs
    file_list, err_file_list = [], []
    if retrieve_flag:
        api_key, api_base, api_version = load_env(args.env_file)
        manager = AzureOpenAIBatchManager(
            api_key=api_key,
            api_endpoint=api_base,
            api_version=api_version,
            batch_file_name=args.batch_file,
        )
        file_list, err_file_list = manager.retrieve_all_submitted_batches(
            batch_output_dir=args.output_dir
        )
        logging.info(
            f"Retrieved {len(file_list)} result files, {len(err_file_list)} error files."
        )
    else:
        with open(args.batch_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                batch_id, _ = line.strip().split(",", 1)
                fpath = os.path.join(args.output_dir, f"batch_output_{batch_id}.jsonl")
                if os.path.exists(fpath):
                    file_list.append(fpath)
                err_path = os.path.join(
                    args.output_dir, f"batch_output_{batch_id}_error.jsonl"
                )
                if os.path.exists(err_path):
                    err_file_list.append(err_path)

    # 2) Index originals by id
    base_index: Dict[str, Dict[str, Any]] = {}
    with open(args.original_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                base_id = ex.get(args.id_field)

                if base_id is not None:
                    base_index[str(base_id)] = ex
            except Exception as e:
                logging.error(f"Error reading original line: {e}")

    # 3) Parse all outputs and merge by id into QA_meta
    updated_ids = set()
    for p in file_list:
        if not os.path.exists(p):
            continue
        collected = parse_response_file(p)
        for cid, payload in collected.items():
            base_id, qid = split_custom_id(cid, args.custom_id_sep)
            # logging.info(f"Processing base_id={base_id}, question_id={qid}")

            rec = base_index.get(str(base_id))
            if not rec:
                logging.warning(
                    f"No original record for id='{base_id}' (custom_id={cid})"
                )
                continue

            parsed = parse_answer_payload(payload.get("response_raw"))
            # parsed=json.loads(parsed)
            inject_predicted(
                rec, qid, args.lang_prefix, parsed, payload.get("response_raw", "")
            )
            updated_ids.add(str(base_id))

    # 4) Write merged (only those updated) as one JSONL per base id
    with open(args.output_file, "w", encoding="utf-8") as w:
        for bid in updated_ids:
            w.write(json.dumps(base_index[bid], ensure_ascii=False) + "\n")
    logging.info(f"Wrote {len(updated_ids)} merged records -> {args.output_file}")

    # 5) Error subset
    def collect_error_custom_ids(err_file: str) -> List[str]:
        ids: List[str] = []
        if not os.path.exists(err_file):
            return ids
        with open(err_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj.get("custom_id")
                    if isinstance(cid, list) and cid:
                        ids.append(cid[0])
                    elif isinstance(cid, str):
                        ids.append(cid)
                except Exception:
                    pass
        return ids

    error_custom_ids: List[str] = []
    for ep in err_file_list:
        error_custom_ids.extend(collect_error_custom_ids(ep))

    targets = set(
        split_custom_id(cid, args.custom_id_sep)[0] for cid in error_custom_ids
    )
    with open(args.output_error_file, "w", encoding="utf-8") as w:
        for bid, ex in base_index.items():
            if bid in targets:
                w.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logging.info(f"Wrote error subset -> {args.output_error_file}")


if __name__ == "__main__":
    main()
