#!/usr/bin/env python3
"""Build a style-robust MLX-LM dataset with prompt jitter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen3.common import SYSTEM_PROMPT


SYSTEM_VARIANTS = [
    SYSTEM_PROMPT,
    (
        "You are an insurance claim consistency analyst. "
        "Analyze the claim and return ONLY valid JSON with: probability_true (0-1), "
        "verdict (true or not_true), reasoning (brief), incongruences (list)."
    ),
    (
        "You are an insurance claim consistency analyst. "
        "Think carefully, then output ONLY valid JSON. "
        "Required keys: probability_true, verdict, reasoning, incongruences."
    ),
]

USER_PREFIXES = [
    "",
    "Case file:\n",
    "Claim details:\n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build style-robust MLX data.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-variants", type=int, default=4)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_variants() -> List[Tuple[str, str]]:
    variants: List[Tuple[str, str]] = []
    for sys_prompt in SYSTEM_VARIANTS:
        for prefix in USER_PREFIXES:
            variants.append((sys_prompt, prefix))
    return variants


def jitter_record(
    record: Dict[str, Any], variants: List[Tuple[str, str]], max_variants: int
) -> List[Dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return []

    base_claim_id = record.get("claim_id", "unknown")
    output: List[Dict[str, Any]] = []

    for idx, (sys_prompt, prefix) in enumerate(variants[:max_variants], start=1):
        new_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                content = sys_prompt
            elif role == "user":
                content = f"{prefix}{content}"
            new_messages.append({"role": role, "content": content})

        new_record = dict(record)
        new_record["claim_id"] = f"{base_claim_id}::v{idx}"
        new_record["messages"] = new_messages
        new_record["variant_id"] = idx
        output.append(new_record)

    return output


def process_split(
    input_path: Path, output_path: Path, max_variants: int, jitter: bool
) -> int:
    records = load_jsonl(input_path)
    variants = build_variants()
    output_records: List[Dict[str, Any]] = []

    for record in records:
        if jitter:
            output_records.extend(jitter_record(record, variants, max_variants))
        else:
            output_records.append(record)

    save_jsonl(output_records, output_path)
    return len(output_records)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    train_in = input_dir / "train.jsonl"
    val_in = input_dir / "valid.jsonl"
    test_in = input_dir / "test.jsonl"

    train_out = output_dir / "train.jsonl"
    val_out = output_dir / "valid.jsonl"
    test_out = output_dir / "test.jsonl"

    train_count = process_split(train_in, train_out, args.max_variants, jitter=True)
    val_count = process_split(val_in, val_out, args.max_variants, jitter=True)
    test_count = process_split(test_in, test_out, args.max_variants, jitter=False)

    print(f"Train records: {train_count}")
    print(f"Valid records: {val_count}")
    print(f"Test records:  {test_count}")


if __name__ == "__main__":
    main()
