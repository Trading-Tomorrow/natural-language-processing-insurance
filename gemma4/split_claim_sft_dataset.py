from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gemma4.common import (
    DEFAULT_FULL_DATASET_PATH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPLIT_STATS_PATH,
    DEFAULT_TEST_PATH,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    DEFAULT_VAL_RATIO,
    label_distribution,
    load_jsonl,
    save_json,
    save_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split the claim-level SFT dataset into train/val/test.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_FULL_DATASET_PATH)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_SPLIT_STATS_PATH)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def build_stats(records: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    return {
        "binary_label_distribution": label_distribution(records, "binary_label"),
        "original_label_distribution": label_distribution(records, "original_label"),
    }


def main() -> None:
    args = parse_args()
    if args.val_ratio <= 0 or args.test_ratio <= 0 or args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1.0.")

    records = load_jsonl(args.input_path)
    stratify_labels = [record["original_label"] for record in records]
    temp_ratio = args.val_ratio + args.test_ratio

    train_records, temp_records = train_test_split(
        records,
        test_size=temp_ratio,
        random_state=args.seed,
        stratify=stratify_labels,
    )

    temp_stratify = [record["original_label"] for record in temp_records]
    relative_test_ratio = args.test_ratio / temp_ratio
    val_records, test_records = train_test_split(
        temp_records,
        test_size=relative_test_ratio,
        random_state=args.seed,
        stratify=temp_stratify,
    )

    save_jsonl(train_records, args.train_path)
    save_jsonl(val_records, args.val_path)
    save_jsonl(test_records, args.test_path)

    stats = {
        "input_path": str(args.input_path),
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "test_path": str(args.test_path),
        "seed": args.seed,
        "ratios": {
            "train": round(1.0 - args.val_ratio - args.test_ratio, 4),
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "num_records": {
            "full": len(records),
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "splits": {
            "train": build_stats(train_records),
            "val": build_stats(val_records),
            "test": build_stats(test_records),
        },
    }
    save_json(stats, args.stats_path)

    print(f"Saved train split to: {args.train_path}")
    print(f"Saved validation split to: {args.val_path}")
    print(f"Saved test split to: {args.test_path}")
    print(f"Saved split stats to: {args.stats_path}")
    print(f"Train/Val/Test sizes: {len(train_records)} / {len(val_records)} / {len(test_records)}")


if __name__ == "__main__":
    main()
