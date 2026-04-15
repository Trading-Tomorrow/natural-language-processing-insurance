#!/usr/bin/env python3
"""Apply a fixed threshold to benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply threshold to results.")
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    return parser.parse_args()


def load_results(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(results, list):
        raise ValueError("Invalid results format.")
    return results


def main() -> None:
    args = parse_args()
    results = load_results(args.results_json)
    valid = [r for r in results if r.get("prob") is not None]

    tp = tn = fp = fn = 0
    for r in valid:
        gt = r.get("gt")
        pred = "true" if float(r["prob"]) >= args.threshold else "not_true"
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
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    output = {
        "threshold": args.threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "valid": len(valid),
        "total": len(results),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
