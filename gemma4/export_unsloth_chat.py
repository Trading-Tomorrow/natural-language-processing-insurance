from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gemma4.common import (
    DEFAULT_CHAT_EXPORT_DIR,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    build_chat_messages,
    load_jsonl,
    save_json,
    save_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export claim SFT splits to chat-format JSONL for Unsloth.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CHAT_EXPORT_DIR)
    return parser.parse_args()


def convert_split(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    exported = []
    for record in records:
        exported.append(
            {
                "claim_id": record["claim_id"],
                "messages": build_chat_messages(record["input_text"], record["target_json"]),
                "binary_label": record["binary_label"],
                "original_label": record["original_label"],
            }
        )
    return exported


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": load_jsonl(args.train_path),
        "val": load_jsonl(args.val_path),
        "test": load_jsonl(args.test_path),
    }

    stats = {}
    for split_name, records in split_map.items():
        exported = convert_split(records)
        output_path = args.output_dir / f"claim_sft_{split_name}_chat.jsonl"
        save_jsonl(exported, output_path)
        stats[split_name] = {
            "num_records": len(exported),
            "output_path": str(output_path),
        }
        print(f"Saved {split_name} chat export to: {output_path}")

    save_json(stats, args.output_dir / "claim_sft_chat_exports_stats.json")


if __name__ == "__main__":
    main()
