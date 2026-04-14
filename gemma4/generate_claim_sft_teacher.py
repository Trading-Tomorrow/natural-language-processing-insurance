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

from gemma4.common import (
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
    parser = argparse.ArgumentParser(description="Generate teacher targets for claim-level Gemma 4 SFT.")
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
        return False, f"unexpected claim_id: {claim_id}", None

    ok, errors, cleaned = validate_target_json(row, enforce_reasoning_guardrails=True)
    if not ok:
        return False, "; ".join(errors), None
    return True, "ok", {"claim_id": claim_id, **cleaned}


def build_batches(records: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [list(records[index : index + batch_size]) for index in range(0, len(records), batch_size)]


def merge_teacher_output(source_record: Dict[str, Any], teacher_row: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    merged = dict(source_record)
    target_payload = {
        "probability_true": teacher_row["probability_true"],
        "verdict": teacher_row["verdict"],
        "reasoning": teacher_row["reasoning"],
        "incongruences": teacher_row["incongruences"],
    }
    merged["target_json"] = target_payload
    merged["target_text"] = target_json_to_text(target_payload)
    merged["teacher_model"] = model_name
    merged["teacher_prompt_version"] = TEACHER_PROMPT_VERSION
    return merged


def main() -> None:
    args = parse_args()
    source_records = load_jsonl(args.input_path)
    existing_records, existing_by_claim_id = load_existing_output(args.output_path)

    selected_records = source_records[args.start_index :]
    if args.max_claims is not None:
        selected_records = selected_records[: args.max_claims]

    pending_records = [
        record for record in selected_records
        if normalize_space(record.get("claim_id")) not in existing_by_claim_id
    ]

    print(f"Input source claims: {len(source_records)}")
    print(f"Selected claims: {len(selected_records)}")
    print(f"Already annotated: {len(selected_records) - len(pending_records)}")
    print(f"Pending claims: {len(pending_records)}")

    if args.dry_run:
        batches = build_batches(pending_records, args.claims_per_call)
        preview = build_prompt(batches[0]) if batches else "(no pending claims)"
        print(f"\nDry run only. Prepared {len(batches)} batches.")
        print("\n=== PROMPT PREVIEW ===")
        print(preview[:5000])
        return

    if not pending_records:
        print("No pending claims to annotate.")
        return

    model = create_model(args.model_name, args.temperature, args.top_p)
    output_records = list(existing_records)
    output_by_claim_id = dict(existing_by_claim_id)
    batches = build_batches(pending_records, args.claims_per_call)
    rejected_batches = 0

    for batch_index, batch in enumerate(batches, start=1):
        expected_claim_ids = [normalize_space(record["claim_id"]) for record in batch]
        prompt = build_prompt(batch)
        response_text = ""
        try:
            response = model.generate_content(prompt)
            response_text = response.text or ""
            annotations = parse_response_text(response_text)
        except Exception as exc:
            rejected_batches += 1
            save_failure_artifacts(
                error_dir=args.error_dir,
                batch_index=batch_index,
                claim_ids=expected_claim_ids,
                prompt=prompt,
                response_text=response_text,
                reason=str(exc),
            )
            print(f"[batch {batch_index}/{len(batches)}] rejected: {exc}")
            time.sleep(args.sleep_seconds)
            continue

        if len(annotations) != len(batch):
            rejected_batches += 1
            save_failure_artifacts(
                error_dir=args.error_dir,
                batch_index=batch_index,
                claim_ids=expected_claim_ids,
                prompt=prompt,
                response_text=response_text,
                reason=f"expected {len(batch)} rows, got {len(annotations)}",
            )
            print(f"[batch {batch_index}/{len(batches)}] rejected: expected {len(batch)} rows, got {len(annotations)}")
            time.sleep(args.sleep_seconds)
            continue

        validated: Dict[str, Dict[str, Any]] = {}
        batch_valid = True
        for item in annotations:
            ok, message, cleaned = validate_teacher_row(item, expected_claim_ids)
            if not ok or cleaned is None:
                batch_valid = False
                save_failure_artifacts(
                    error_dir=args.error_dir,
                    batch_index=batch_index,
                    claim_ids=expected_claim_ids,
                    prompt=prompt,
                    response_text=response_text,
                    reason=message,
                )
                print(f"[batch {batch_index}/{len(batches)}] rejected: {message}")
                break
            validated[cleaned["claim_id"]] = cleaned

        if not batch_valid:
            rejected_batches += 1
            time.sleep(args.sleep_seconds)
            continue

        for source_record in batch:
            claim_id = normalize_space(source_record["claim_id"])
            merged = merge_teacher_output(source_record, validated[claim_id], args.model_name)
            output_by_claim_id[claim_id] = merged

        output_records = list(output_by_claim_id.values())
        output_records.sort(key=lambda row: row["claim_id"])
        save_jsonl(output_records, args.output_path)
        print(
            f"[batch {batch_index}/{len(batches)}] saved "
            f"{len(output_records)} annotated claims total"
        )
        time.sleep(args.sleep_seconds)

    final_records = list(output_by_claim_id.values())
    stats = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "model_name": args.model_name,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "selected_claims": len(selected_records),
        "annotated_claims": len(final_records),
        "rejected_batches": rejected_batches,
        "error_dir": str(args.error_dir),
        "binary_label_distribution": {
            "true": sum(1 for row in final_records if row.get("binary_label") == "true"),
            "not_true": sum(1 for row in final_records if row.get("binary_label") == "not_true"),
        },
        "original_label_distribution": {
            label: sum(1 for row in final_records if row.get("original_label") == label)
            for label in sorted({row.get("original_label") for row in final_records})
        },
    }
    save_json(stats, args.stats_path)
    print(f"Saved teacher stats to: {args.stats_path}")


if __name__ == "__main__":
    main()
