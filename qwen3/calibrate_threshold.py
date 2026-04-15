#!/usr/bin/env python3
"""Calibrate verdict threshold using validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate verdict threshold.")
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_results(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(results, list):
        raise ValueError("Invalid results format.")
    return results


def compute_metrics(results: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    valid = [r for r in results if r.get("prob") is not None]
    if not valid:
        return {
            "threshold": threshold,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    def pred_from_prob(prob: float) -> str:
        return "true" if prob >= threshold else "not_true"

    tp = tn = fp = fn = 0
    for r in valid:
        gt = r.get("gt")
        pred = pred_from_prob(float(r["prob"]))
        if gt == "true" and pred == "true":
            tp += 1
        elif gt == "not_true" and pred == "not_true":
            tn += 1
        elif gt == "not_true" and pred == "true":
            fp += 1
        elif gt == "true" and pred == "not_true":
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(valid) if valid else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "valid": len(valid),
    }


def main() -> None:
    args = parse_args()
    results = load_results(args.results_json)

    best: Dict[str, Any] | None = None
    for i in range(1, 100):
        threshold = i / 100.0
        metrics = compute_metrics(results, threshold)
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
