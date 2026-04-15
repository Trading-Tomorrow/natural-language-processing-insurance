#!/usr/bin/env python3
"""Build mixed-generator SFT dataset with balanced split.

Uses Gemini teacher labels from existing claim_sft_full.jsonl and
template targets for ChatGPT/Grok synthetic claims.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen3.common import (
    collapse_claim_label,
    serialize_claim_for_student,
    target_json_to_text,
)


SYSTEM_PROMPT = (
    "You are an insurance claim consistency analyst. "
    "You receive one structured accident claim with detected damages and party statements. "
    "Return valid JSON only. "
    "Estimate the probability that the claim is true, choose verdict=true or verdict=not_true, "
    "explain the decision briefly, and list the main incongruences if they exist."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed-generator SFT dataset.")
    parser.add_argument("--gemini-jsonl", type=Path, required=True)
    parser.add_argument("--chatgpt-json", type=Path, required=True)
    parser.add_argument("--grok-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def build_template_target(binary_label: str) -> Dict[str, Any]:
    if binary_label == "true":
        return {
            "probability_true": 0.85,
            "verdict": "true",
            "reasoning": "The narratives are coherent and the described damage aligns with the reported incident.",
            "incongruences": [],
        }
    return {
        "probability_true": 0.15,
        "verdict": "not_true",
        "reasoning": "The claim contains inconsistencies or implausible elements compared to the reported damage.",
        "incongruences": ["Statement and damage description are not fully consistent."],
    }


def to_chat_record(
    claim: Dict[str, Any], *, generator: str, use_template: bool
) -> Dict[str, Any]:
    claim_id = str(claim.get("claim_id", "")).strip()
    input_text = claim.get("input_text")
    if not input_text:
        input_text = serialize_claim_for_student(claim)

    if "binary_label" in claim:
        binary_label = claim.get("binary_label")
    else:
        binary_label = collapse_claim_label(str(claim.get("ground_truth_label")))

    if use_template:
        target = build_template_target(binary_label)
        target_text = target_json_to_text(target)
    else:
        target = claim.get("target_json")
        target_text = claim.get("target_text")
        if target is None or target_text is None:
            raise ValueError(f"Missing target fields for {claim_id}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": target_text},
    ]

    return {
        "claim_id": claim_id,
        "messages": messages,
        "binary_label": binary_label,
        "original_label": claim.get("original_label", claim.get("ground_truth_label")),
        "generator": generator,
    }


def split_by_generator(
    rows: List[Dict[str, Any]], seed: int, train_ratio: float, val_ratio: float
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_gen: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_gen.setdefault(row["generator"], []).append(row)

    splits = {"train": [], "val": [], "test": []}
    for gen, gen_rows in by_gen.items():
        rng.shuffle(gen_rows)
        n = len(gen_rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_rows = gen_rows[:n_train]
        val_rows = gen_rows[n_train : n_train + n_val]
        test_rows = gen_rows[n_train + n_val :]
        splits["train"].extend(train_rows)
        splits["val"].extend(val_rows)
        splits["test"].extend(test_rows)

    rng.shuffle(splits["train"])
    rng.shuffle(splits["val"])
    rng.shuffle(splits["test"])
    return splits


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    gemini_rows = load_jsonl(args.gemini_jsonl)
    chatgpt_rows = load_json(args.chatgpt_json)
    grok_rows = load_json(args.grok_json)

    all_rows: List[Dict[str, Any]] = []
    for row in gemini_rows:
        all_rows.append(to_chat_record(row, generator="gemini", use_template=False))
    for row in chatgpt_rows:
        all_rows.append(to_chat_record(row, generator="chatgpt", use_template=True))
    for row in grok_rows:
        all_rows.append(to_chat_record(row, generator="grok", use_template=True))

    splits = split_by_generator(all_rows, args.seed, args.train_ratio, args.val_ratio)

    save_jsonl(splits["train"], output_dir / "train.jsonl")
    save_jsonl(splits["val"], output_dir / "valid.jsonl")
    save_jsonl(splits["test"], output_dir / "test.jsonl")

    stats = {
        "total": len(all_rows),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "by_generator": {
            gen: sum(1 for row in all_rows if row["generator"] == gen)
            for gen in ("gemini", "chatgpt", "grok")
        },
    }
    (output_dir / "split_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
