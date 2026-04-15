#!/usr/bin/env python3
"""
Benchmark Qwen3 base or fine-tuned model on new datasets.

Input: JSON list of claim objects with ground_truth_label and statements.
Output: Per-claim predictions + summary metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen3.common import (
    DEFAULT_BASE_MODEL_ID,
    SYSTEM_PROMPT,
    collapse_claim_label,
    extract_json_object,
    serialize_claim_for_student,
    validate_target_json,
)
from qwen3.model_io import generate_with_mlx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 on new datasets.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_prompt(input_text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Claim:\n{input_text}\n\n"
        "Respond with a JSON object containing:\n"
        "- probability_true: a number between 0.0 and 1.0\n"
        '- verdict: "true" or "not_true"\n'
        "- reasoning: a brief explanation\n"
        "- incongruences: a list of suspicious elements (empty if genuine)\n\n"
        "JSON response:"
    )


def load_claims(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of claims.")

    normalized: List[Dict[str, Any]] = []
    for row in data:
        if isinstance(row, dict) and "messages" in row and "input_text" not in row:
            for msg in row.get("messages", []):
                if msg.get("role") == "user" and msg.get("content"):
                    row = dict(row)
                    row["input_text"] = msg["content"]
                    break
        normalized.append(row)
    return normalized


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed_ids": [], "results": []}


def save_checkpoint(path: Path, checkpoint: Dict[str, Any]) -> None:
    checkpoint["last_updated"] = datetime.now().isoformat()
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if r.get("valid")]
    tp = sum(1 for r in valid if r["gt"] == "true" and r["pred"] == "true")
    tn = sum(1 for r in valid if r["gt"] == "not_true" and r["pred"] == "not_true")
    fp = sum(1 for r in valid if r["gt"] == "not_true" and r["pred"] == "true")
    fn = sum(1 for r in valid if r["gt"] == "true" and r["pred"] == "not_true")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(valid) if valid else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "total": len(results),
        "valid": len(valid),
        "json_success": (len(valid) / len(results)) if results else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    claims = load_claims(args.input_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.run_name}_checkpoint.json"
    results_path = args.output_dir / f"{args.run_name}_results.json"

    checkpoint = (
        load_checkpoint(checkpoint_path)
        if args.resume
        else {"completed_ids": [], "results": []}
    )
    completed_ids = set(checkpoint.get("completed_ids", []))
    results = checkpoint.get("results", [])

    print("=" * 60)
    print(f"RUN: {args.run_name}")
    print(f"Model: {args.model}")
    print(f"Adapter: {args.adapter_path if args.adapter_path else 'None'}")
    print(f"Samples: {len(claims)}")
    print("=" * 60)
    print(f"Completed: {len(completed_ids)}/{len(claims)}")

    for idx, claim in enumerate(claims, start=1):
        claim_id = str(claim.get("claim_id", f"sample_{idx}"))
        if claim_id in completed_ids:
            continue

        gt_raw = claim.get("binary_label") or claim.get("ground_truth_label")
        try:
            if gt_raw in {"true", "not_true"}:
                gt = str(gt_raw)
            elif gt_raw:
                gt = collapse_claim_label(str(gt_raw))
            else:
                gt = "unknown"
        except Exception:
            gt = "unknown"

        input_text = claim.get("input_text") or serialize_claim_for_student(claim)
        prompt = build_prompt(input_text)

        try:
            raw_output = generate_with_mlx(
                model_name=args.model,
                prompt=prompt,
                adapter_path=str(args.adapter_path) if args.adapter_path else None,
                max_tokens=args.max_tokens,
                temp=args.temperature,
            )
            parsed, _ = extract_json_object(raw_output)
            if parsed is None:
                result = {
                    "claim_id": claim_id,
                    "gt": gt,
                    "pred": None,
                    "prob": None,
                    "valid": False,
                }
            else:
                ok, _, cleaned = validate_target_json(parsed)
                verdict = cleaned.get("verdict") if ok and cleaned else None
                prob = cleaned.get("probability_true") if ok and cleaned else None
                result = {
                    "claim_id": claim_id,
                    "gt": gt,
                    "pred": verdict,
                    "prob": prob,
                    "valid": verdict is not None,
                }
        except Exception as exc:
            result = {
                "claim_id": claim_id,
                "gt": gt,
                "pred": None,
                "prob": None,
                "valid": False,
                "error": str(exc)[:200],
            }

        results.append(result)
        completed_ids.add(claim_id)
        checkpoint["completed_ids"] = list(completed_ids)
        checkpoint["results"] = results
        save_checkpoint(checkpoint_path, checkpoint)

        status = "OK" if result["pred"] == gt else "WRONG" if result["pred"] else "FAIL"
        print(
            f"[{len(completed_ids):4d}/{len(claims)}] {claim_id[:24]:24s} GT={gt:<8} Pred={str(result['pred']):<8} {status}"
        )

    metrics = compute_metrics(results)
    payload = {
        "run": {
            "name": args.run_name,
            "timestamp": datetime.now().isoformat(),
            "input_json": str(args.input_json),
            "model": args.model,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "metrics": metrics,
        "results": results,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 60)
    print("COMPLETE")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"Specificity: {metrics['specificity']:.2%}")
    print(f"F1: {metrics['f1']:.2%}")
    print(f"JSON success: {metrics['json_success']:.2%}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
