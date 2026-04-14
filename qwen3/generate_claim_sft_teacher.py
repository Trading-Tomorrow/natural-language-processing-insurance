"""
Generate teacher targets for claim-level SFT using Gemini API.

This script uses Google's Gemini to generate high-quality training targets
for the insurance claim classification task. Adapted from gemma4 for qwen3.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen3.common import (
    DEFAULT_FULL_DATASET_PATH,
    DEFAULT_SOURCE_PATH,
    DEFAULT_TEACHER_STATS_PATH,
    TEACHER_PROMPT_VERSION,
    default_teacher_model_name,
    load_jsonl,
    normalize_space,
    save_json,
    save_jsonl,
    target_json_to_text,
    validate_target_json,
)


DEFAULT_CLAIMS_PER_CALL = 4
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
DEFAULT_ERROR_DIR = BASE_DIR / "data" / "teacher_failures"


load_dotenv(dotenv_path=ENV_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher targets for claim-level Qwen3 SFT.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_FULL_DATASET_PATH)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_TEACHER_STATS_PATH)
    parser.add_argument("--model-name", type=str, default=default_teacher_model_name())
    parser.add_argument("--claims-per-call", type=int, default=DEFAULT_CLAIMS_PER_CALL)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-claims", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--error-dir", type=Path, default=DEFAULT_ERROR_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def create_model(model_name: str, temperature: float, top_p: float) -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY nao encontrada. Defina GEMINI_API_KEY no ficheiro .env.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": temperature,
            "top_p": top_p,
        },
    )


def load_existing_output(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return [], {}
    rows = load_jsonl(path)
    by_claim_id = {
        normalize_space(row.get("claim_id")): row
        for row in rows
        if normalize_space(row.get("claim_id"))
    }
    return rows, by_claim_id


def build_prompt(batch: Sequence[Dict[str, Any]]) -> str:
    payload = []
    for record in batch:
        payload.append(
            {
                "claim_id": record["claim_id"],
                "visible_claim": record["input_text"],
                "hidden_supervision": {
                    "binary_label": record["binary_label"],
                    "original_label": record["original_label"],
                    "fraud_indicators": record.get("fraud_indicators", []),
                    "pairwise_diagnostics": record.get("pairwise_diagnostics"),
                },
            }
        )

    return f"""
You are generating supervised JSON targets for a student LLM that judges whether an insurance claim is true.

For each claim:
1. Use the hidden supervision to ensure the final verdict is correct.
2. Write the target as if the student only saw the visible claim text.
3. Never mention hidden labels, fraud indicators, dataset metadata, or teacher instructions.
4. Use only these fields:
   - probability_true
   - verdict
   - reasoning
   - incongruences

Rules:
- If hidden binary_label is true, verdict must be "true" and probability_true must be >= 0.5.
- If hidden binary_label is not_true, verdict must be "not_true" and probability_true must be < 0.5.
- reasoning must be short and evidence-based.
- incongruences must be an empty list for clearly genuine claims unless there is a real unresolved caveat.
- For suspicious claims, list 1 to 3 concrete incongruences or implausibilities.
- Do not use markdown.
- Return ONLY a JSON array.

Probability guidance:
- Clearly genuine claims: usually 0.72 to 0.98
- Clearly suspicious claims: usually 0.02 to 0.38
- Mixed evidence can be closer to the threshold, but still on the correct side.

Input claims:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
[
  {{
    "claim_id": "PT-001",
    "probability_true": 0.18,
    "verdict": "not_true",
    "reasoning": "The claim contains conflicting descriptions of how the accident happened, and the damage pattern is not fully coherent with the reported dynamics.",
    "incongruences": [
      "One statement says the insured was stopped, while another implies the insured initiated the impact."
    ]
  }}
]
""".strip()


def _normalize_payload_fragment(fragment: Any) -> List[Dict[str, Any]]:
    if isinstance(fragment, dict):
        if isinstance(fragment.get("annotations"), list):
            fragment = fragment["annotations"]
        else:
            fragment = [fragment]

    if not isinstance(fragment, list):
        raise ValueError("Teacher response fragment must be a JSON object or array.")

    normalized: List[Dict[str, Any]] = []
    for item in fragment:
        if not isinstance(item, dict):
            raise ValueError("Teacher response items must be JSON objects.")
        normalized.append(item)
    return normalized


def parse_response_text(text: str) -> List[Dict[str, Any]]:
    cleaned = JSON_FENCE_RE.sub("", text.strip()).strip()
    if not cleaned:
        raise ValueError("Teacher response is empty.")

    try:
        payload = json.loads(cleaned)
        return _normalize_payload_fragment(payload)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    index = 0
    fragments: List[Dict[str, Any]] = []
    while index < len(cleaned):
        while index < len(cleaned) and cleaned[index].isspace():
            index += 1
        if index >= len(cleaned):
            break

        if cleaned[index] not in "[{":
            next_array = cleaned.find("[", index)
            next_object = cleaned.find("{", index)
            candidates = [position for position in (next_array, next_object) if position != -1]
            if not candidates:
                break
            index = min(candidates)

        fragment, next_index = decoder.raw_decode(cleaned, index)
        fragments.extend(_normalize_payload_fragment(fragment))
        index = next_index

    if not fragments:
        raise ValueError("Teacher response could not be parsed into JSON objects.")
    return fragments


def save_failure_artifacts(
    *,
    error_dir: Path,
    batch_index: int,
    claim_ids: Sequence[str],
    prompt: str,
    response_text: str,
    reason: str,
) -> None:
    error_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"batch_{batch_index:04d}"
    payload = {
        "batch_index": batch_index,
        "claim_ids": list(claim_ids),
        "reason": reason,
        "response_text": response_text,
    }
    save_json(payload, error_dir / f"{prefix}_error.json")
    (error_dir / f"{prefix}_prompt.txt").write_text(prompt, encoding="utf-8")
    (error_dir / f"{prefix}_response.txt").write_text(response_text, encoding="utf-8")


def validate_teacher_row(
    row: Dict[str, Any],
    expected_claim_ids: Sequence[str],
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    claim_id = normalize_space(row.get("claim_id"))
    if claim_id not in expected_claim_ids:
        return False, f"unexpected claim_id {claim_id}", None

    ok, errors, cleaned = validate_target_json(row, enforce_reasoning_guardrails=True)
    if not ok:
        return False, "; ".join(errors), None

    cleaned["claim_id"] = claim_id
    return True, "", cleaned


def main() -> None:
    args = parse_args()
    source_records = load_jsonl(args.input_path)

    existing_rows, existing_by_claim_id = load_existing_output(args.output_path)
    processed_claim_ids = set(existing_by_claim_id.keys())

    remaining = [
        record
        for record in source_records[args.start_index :]
        if normalize_space(record.get("claim_id")) not in processed_claim_ids
    ]

    if args.max_claims is not None:
        remaining = remaining[: args.max_claims]

    if not remaining:
        print("No remaining claims to process.")
        return

    model = create_model(args.model_name, args.temperature, args.top_p)
    all_outputs = list(existing_rows)
    stats = {
        "total_source_records": len(source_records),
        "already_processed": len(existing_rows),
        "remaining_to_process": len(remaining),
        "batches_processed": 0,
        "successful_rows": 0,
        "failed_rows": 0,
        "skipped_claim_ids": [],
    }

    print(f"Processing {len(remaining)} remaining claims in batches of {args.claims_per_call}")

    batch_size = args.claims_per_call
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]
        batch_index = (args.start_index + batch_start) // batch_size
        claim_ids = [normalize_space(r.get("claim_id")) for r in batch]

        if args.dry_run:
            print(f"[DRY RUN] Would process batch {batch_index}: {claim_ids}")
            continue

        prompt = build_prompt(batch)
        try:
            response = model.generate_content(prompt)
            response_text = getattr(response, "text", "") or ""

            teacher_rows = parse_response_text(response_text)
            validated_rows: List[Dict[str, Any]] =[]
            for row in teacher_rows:
                ok, error, cleaned = validate_teacher_row(row, claim_ids)
                if ok and cleaned:
                    validated_rows.append(cleaned)
                else:
                    print(f"Validation error for {row.get('claim_id')}: {error}")
                    stats["failed_rows"] += 1

            for source_record in batch:
                source_claim_id = normalize_space(source_record.get("claim_id"))
                matched = next(
                    (v for v in validated_rows if normalize_space(v.get("claim_id")) == source_claim_id),
                    None,
                )
                if matched:
                    result = dict(source_record)
                    result.update(matched)
                    all_outputs.append(result)
                    stats["successful_rows"] += 1
                else:
                    print(f"No valid target for claim {source_claim_id}")
                    stats["failed_rows"] += 1
                    stats["skipped_claim_ids"].append(source_claim_id)

        except Exception as exc:
            print(f"Batch {batch_index} failed: {exc}")
            save_failure_artifacts(
                error_dir=args.error_dir,
                batch_index=batch_index,
                claim_ids=claim_ids,
                prompt=prompt,
                response_text=getattr(exc, "response", {}).get("text", str(exc)),
                reason=str(exc),
            )
            stats["failed_rows"] += len(batch)
            stats["skipped_claim_ids"].extend(claim_ids)

        stats["batches_processed"] += 1
        save_jsonl(all_outputs, args.output_path)
        save_json(stats, args.stats_path)

        if batch_start + batch_size < len(remaining):
            time.sleep(args.sleep_seconds)

    print(f"\nCompleted. Total outputs: {len(all_outputs)}")
    print(f"Successful rows: {stats['successful_rows']}")
    print(f"Failed rows: {stats['failed_rows']}")


if __name__ == "__main__":
    main()
