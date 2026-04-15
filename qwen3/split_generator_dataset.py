#!/usr/bin/env python3
"""Split a generator dataset into val/test subsets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into val/test.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")

    rng = random.Random(args.seed)
    rng.shuffle(data)

    n_val = int(len(data) * args.val_ratio)
    val = data[:n_val]
    test = data[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "test.json").write_text(
        json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Total: {len(data)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")


if __name__ == "__main__":
    main()
