import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_gold.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "gold_splits"
DEFAULT_SEED = 42
LABELS = ("supports", "neutral", "contradicts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified train/validation/test splits for the pairwise gold dataset.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at line {line_number} in {path}.")
            records.append(payload)
    return records


def save_jsonl(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def label_distribution(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(record.get("label", "")) for record in records)
    return {label: counts.get(label, 0) for label in LABELS}


def build_split_counts(total: int, train_ratio: float, validation_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    train_count = int(round(total * train_ratio))
    validation_count = int(round(total * validation_ratio))
    test_count = total - train_count - validation_count

    while train_count + validation_count + test_count < total:
        train_count += 1
    while train_count + validation_count + test_count > total:
        if train_count >= validation_count and train_count >= test_count and train_count > 1:
            train_count -= 1
        elif validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        else:
            test_count -= 1

    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("Each split must have at least one example per class.")
    return train_count, validation_count, test_count


def build_stratified_splits(
    records: Sequence[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    rng = random.Random(seed)
    label_buckets: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        label = str(record.get("label", ""))
        label_buckets[label].append(record)

    train_records: List[Dict[str, Any]] = []
    validation_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for label in LABELS:
        bucket = list(label_buckets[label])
        if len(bucket) < 3:
            raise ValueError(f"Need at least three examples for label '{label}' to build 3-way splits.")

        rng.shuffle(bucket)
        train_count, validation_count, test_count = build_split_counts(
            len(bucket),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        train_records.extend(bucket[:train_count])
        validation_records.extend(bucket[train_count : train_count + validation_count])
        test_records.extend(bucket[train_count + validation_count : train_count + validation_count + test_count])

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    rng.shuffle(test_records)
    return train_records, validation_records, test_records


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_path)
    train_records, validation_records, test_records = build_stratified_splits(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "pairwise_dataset_gold_train.jsonl"
    validation_path = args.output_dir / "pairwise_dataset_gold_validation.jsonl"
    test_path = args.output_dir / "pairwise_dataset_gold_test.jsonl"
    stats_path = args.output_dir / "pairwise_dataset_gold_split_stats.json"

    save_jsonl(train_records, train_path)
    save_jsonl(validation_records, validation_path)
    save_jsonl(test_records, test_path)

    stats = {
        "input_path": str(args.input_path),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "counts": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "label_distribution": {
            "train": label_distribution(train_records),
            "validation": label_distribution(validation_records),
            "test": label_distribution(test_records),
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input gold dataset: {args.input_path}")
    print(f"Train split: {train_path} | {len(train_records)} examples")
    print(f"Validation split: {validation_path} | {len(validation_records)} examples")
    print(f"Test split: {test_path} | {len(test_records)} examples")
    print(f"Saved split stats to: {stats_path}")


if __name__ == "__main__":
    main()
