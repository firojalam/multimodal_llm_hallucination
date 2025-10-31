# ===============================
# bin/llm_judge/batch_submit.py
# ===============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch submit script for Azure OpenAI Batch.

Changes:
- For each input record, create one batch task per QA item:
  * open-ended  -> question_id = "open-ended"
  * multiple-choice -> question_id = "multiple-choice"
  * true_false -> question_id = f"true_false_{idx}"
- Derive question_id from QA_meta keys (no CLI arg for question_id).
- Forward --lang_prefix (en|msa|arz) to the prompt module.
- custom_id_format can reference {id}, {question_id}, {lang_prefix}.

Input JSONL: one line per *example* (with a nested QA_meta dict).
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv


# ------------------------
# Logging
# ------------------------
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


# ------------------------
# Env loader
# ------------------------
def load_env(env_path: str):
    load_dotenv(dotenv_path=env_path, override=True)
    api_base = os.environ["AZURE_API_URL"].rstrip("/")
    api_key = os.environ["AZURE_API_KEY"]
    api_version = os.environ["AZURE_API_VERSION"]
    engine = os.environ.get("AZURE_ENGINE_NAME", "gpt-4.1-global-batch")
    return api_key, api_base, api_version, engine


# ------------------------
# Prompt module loader
# ------------------------
def load_prompt_module(prompt_name: str, prompt_path: Optional[str]) -> Any:
    """
    Expect a module exposing: make_messages(example: dict, **kwargs) -> List[dict]
    """
    if prompt_path:
        prompt_path = os.path.abspath(prompt_path)
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        spec = importlib.util.spec_from_file_location("user_prompt_module", prompt_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    here = os.path.dirname(os.path.abspath(__file__))
    guess = os.path.join(here, "prompts", f"{prompt_name}_prompt.py")
    if not os.path.exists(guess):
        raise FileNotFoundError(
            f"No prompt file for '{prompt_name}'. Expected at: {guess}\n"
            f"Or pass --prompt_path /path/to/custom_prompt.py"
        )
    spec = importlib.util.spec_from_file_location("user_prompt_module", guess)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# ------------------------
# Batch builder
# ------------------------
class LLMBatchBuilder:
    def __init__(
        self,
        input_jsonl: str,
        output_dir: str,
        model_name: str = "gpt-4.1-global-batch",
        endpoint: str = "/chat/completions",
        max_completion_tokens: int = 512,
        batch_file_size_limit: int = 190 * 1024 * 1024,  # ~190MB per file
        id_field: str = "id",
        custom_id_format: str = "{id}::q={question_id}::lang={lang_prefix}",
    ):
        self.input_jsonl = input_jsonl
        self.output_dir = output_dir
        self.model_name = model_name
        self.endpoint = endpoint
        self.max_completion_tokens = max_completion_tokens
        self.batch_file_size_limit = batch_file_size_limit
        self.id_field = id_field
        self.custom_id_format = custom_id_format
        self.IMAGE_PARAM_DICT = {
            "APPLY_ENCODING": True,
            "SCALING": 0.5,
            "MINIMUM_SIZE": 1200,
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def _calc_bytes(self, obj: Union[str, dict, list]) -> int:
        text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        return len(text.encode("utf-8"))

    def _save_batch(self, tasks: List[dict], idx: int) -> str:
        path = os.path.join(self.output_dir, f"batch_{idx}.jsonl")
        with open(path, "w", encoding="utf-8") as w:
            for t in tasks:
                w.write(json.dumps(t, ensure_ascii=False) + "\n")
        logging.info(f"Saved batch {idx} -> {path}")
        return path

    def _render_custom_id(
        self, base_id: str, question_id: str, lang_prefix: Optional[str]
    ) -> str:
        ctx = {
            "id": base_id,
            "question_id": question_id,
            "lang_prefix": lang_prefix or "",
        }
        try:
            return self.custom_id_format.format_map(ctx)
        except Exception:
            return f"{base_id}::{question_id}"

    def _iter_qa_items(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand one record into a list of items:
          - For open-ended: one item per element in QA_meta["open-ended"]
          - For multiple-choice: per element in QA_meta["multiple-choice"]
          - For true_false: per element with question_id true_false_{i}
        Each yielded dict contains:
          { "base_id", "qa_type", "question_id", "payload" }
        """
        base_id = rec.get(self.id_field)
        qa = rec.get("QA_meta") or {}
        out: List[Dict[str, Any]] = []

        # True/false (may have multiple)
        for i, item in enumerate(qa.get("true_false", []) or []):
            out.append(
                {
                    "base_id": base_id,
                    "qa_type": "true_false",
                    "question_id": f"true_false_{i}",
                    "payload": item,
                }
            )
        return out

    def encode_image_base64(self, file_path, scaling=1.0, minimum_size=None):
        """Encodes the image as a base64 string."""
        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Scaling or resizing logic can be added here if necessary.
        # This implementation assumes scaling logic is outside scope of this base64 encoding.
        return encoded_image

    def encode_wav(self, wav_path):
        """Encodes a WAV file to Base64 format."""
        with open(wav_path, "rb") as wav_file:
            return base64.b64encode(wav_file.read()).decode("utf-8")

    def create_batches(
        self,
        make_messages: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        lang_prefix: Optional[str] = None,
        modality="text",
    ) -> List[str]:
        prompt_kwargs = dict(prompt_kwargs or {})
        if lang_prefix:
            prompt_kwargs.setdefault("lang_prefix", lang_prefix)
            # back-compat alias
            prompt_kwargs.setdefault("prefer_lang", lang_prefix)

        batch_idx, current_size = 1, 0
        current_tasks: List[dict] = []
        created_files: List[str] = []

        total_records = 0
        total_tasks = 0

        with open(self.input_jsonl, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                total_records += 1

                image_data = None
                audio_data = None
                # Determine QA type
                if modality in {"text_image", "speech_image"}:
                    image_path = rec.get("image_path")
                    base64_image = self.encode_image_base64(
                        image_path,
                        scaling=self.IMAGE_PARAM_DICT["SCALING"],
                        minimum_size=self.IMAGE_PARAM_DICT["MINIMUM_SIZE"],
                    )
                    image_data = f"data:image/jpeg;base64,{base64_image}"
                # if modality in {"speech"}:
                # prompt_kwargs["audio_data"]=audio_data
                qa = rec.get("QA_meta") or {}
                messages = make_messages(
                    qa["true_false"],
                    image_data=image_data,
                    **prompt_kwargs,
                )
                base_id = rec.get(self.id_field)
                cid = self._render_custom_id(
                    base_id=base_id,
                    question_id=f"true_false_{0}",  # str(it["question_id"]),
                    lang_prefix=lang_prefix,
                )
                task = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": self.endpoint,
                    "body": {
                        "model": self.model_name,
                        "messages": messages,
                        "max_completion_tokens": self.max_completion_tokens,
                        "temperature": 0.0,
                    },
                }

                task_size = self._calc_bytes(task)
                if (
                    current_size
                    and current_size + task_size > self.batch_file_size_limit
                ):
                    created_files.append(self._save_batch(current_tasks, batch_idx))
                    current_tasks, current_size = [], 0
                    batch_idx += 1

                current_tasks.append(task)
                current_size += task_size
                total_tasks += 1

        if current_tasks:
            created_files.append(self._save_batch(current_tasks, batch_idx))

        logging.info(
            f"Created {len(created_files)} batch file(s) from {total_records} records; "
            f"total tasks generated: {total_tasks}"
        )
        return created_files


# ------------------------
# Azure OpenAI Batch Manager
# ------------------------
class AzureOpenAIBatchManager:
    def __init__(
        self, api_key: str, api_endpoint: str, api_version: str, batch_file_name: str
    ):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            azure_endpoint=api_endpoint, api_key=api_key, api_version=api_version
        )
        self.batch_file_name = batch_file_name

    def _append_tracking(self, batch_id: str, local_file: str):
        if os.path.dirname(self.batch_file_name):
            os.makedirs(os.path.dirname(self.batch_file_name), exist_ok=True)
        with open(self.batch_file_name, "a", encoding="utf-8") as w:
            w.write(f"{batch_id},{os.path.abspath(local_file)}\n")
        logging.info(f"Tracked batch {batch_id} -> {self.batch_file_name}")

    def submit_batch_jsonl(
        self,
        batch_jsonl: str,
        completion_window: str = "24h",
        endpoint: str = "/chat/completions",
    ) -> str:
        with open(batch_jsonl, "rb") as fh:
            fobj = self.client.files.create(file=fh, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=fobj.id,
            endpoint=endpoint,
            completion_window=completion_window,
        )
        self._append_tracking(batch.id, batch_jsonl)
        logging.info(
            f"Submitted {batch_jsonl} -> batch_id={batch.id} status={batch.status}"
        )
        return batch.id

    def submit_all_batches_in_directory(
        self, directory: str, verbose: bool = True
    ) -> List[str]:
        import glob

        submitted = []
        for path in sorted(glob.glob(os.path.join(directory, "batch_*.jsonl"))):
            try:
                bid = self.submit_batch_jsonl(path)
                submitted.append(bid)
            except Exception as e:
                logging.error(f"Failed to submit {path}: {e}")
        if verbose:
            logging.info(f"Submitted {len(submitted)} batch file(s).")
        return submitted


# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Submit Azure OpenAI Batch with one task per QA item."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL (one example per line, contains QA_meta)",
    )
    parser.add_argument("--env_file", required=True, help="Path to .env")
    parser.add_argument(
        "--output_dir", required=True, help="Where to write batch JSONL files"
    )
    parser.add_argument(
        "--batch_file", required=True, help="Tracking file for batch IDs"
    )

    parser.add_argument(
        "--prompt", default="qa", help="Prompt name (qa|qaimage|qaaudio|qaaudioimage)"
    )
    parser.add_argument(
        "--prompt_path", default=None, help="Optional explicit path to prompt module"
    )
    parser.add_argument(
        "--prompt_kwargs",
        default=None,
        help="Optional JSON string to pass into make_messages",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model for body.model (overrides AZURE_ENGINE_NAME)",
    )
    parser.add_argument("--max_tokens", type=int, default=2000)

    parser.add_argument(
        "--id_field", default="image_id", help="Field name for base record id"
    )
    parser.add_argument(
        "--custom_id_format",
        default="{id}::q={question_id}::lang={lang_prefix}",
        help="Python .format_map template for custom_id",
    )

    parser.add_argument(
        "--lang_prefix",
        choices=["en", "msa", "ajp", "arz"],
        required=True,
        help="Select language prefix for fields (en_* | msa_* | arz_*)",
    )
    parser.add_argument(
        "--modality",
        choices=["text", "text_image", "speech", "speech_image"],
        required=True,
        default="text",
        help="Select one of the four modalities (text | text_image | speech |speech_image)",
    )

    args = parser.parse_args()
    configure_logging()

    api_key, api_base, api_version, env_model = load_env(args.env_file)
    model_name = args.model or env_model

    # Load prompt factory
    prompt_mod = load_prompt_module(args.prompt, args.prompt_path)
    if not hasattr(prompt_mod, "make_messages"):
        raise AttributeError(
            "Prompt module must define make_messages(example: dict, **kwargs) -> List[dict]"
        )
    prompt_kwargs = json.loads(args.prompt_kwargs) if args.prompt_kwargs else {}

    # Ensure output directory exists
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    # Build batch files
    builder = LLMBatchBuilder(
        input_jsonl=args.input,
        output_dir=args.output_dir,
        model_name=model_name,
        max_completion_tokens=args.max_tokens,
        id_field=args.id_field,
        custom_id_format=args.custom_id_format,
    )
    builder.create_batches(
        make_messages=prompt_mod.make_messages,
        prompt_kwargs=prompt_kwargs,
        lang_prefix=args.lang_prefix,
        modality=args.modality,
    )

    # Submit all
    manager = AzureOpenAIBatchManager(
        api_key=api_key,
        api_endpoint=api_base,
        api_version=api_version,
        batch_file_name=args.batch_file,
    )
    manager.submit_all_batches_in_directory(args.output_dir, verbose=True)


if __name__ == "__main__":
    main()
