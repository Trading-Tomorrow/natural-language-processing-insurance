"""
Export claim-level SFT data to chat format for MLX-LM training.

This script converts the claim-level SFT dataset to the 'chat' format
expected by MLX-LM, which uses 'messages' arrays.

Format: {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen3.common import (
    DEFAULT_CHAT_EXPORT_DIR,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    DEFAULT_TEST_PATH,
    build_chat_messages,
    load_jsonl,
    save_json,
    save_jsonl,
)


def convert_record_to_chat(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a source record to MLX-LM chat format.
    
    Input record has: claim_id, input_text, binary_label, original_label, etc.
    Output record has: messages array suitable for MLX-LM.
    """
    # Build the assistant response from the target fields
    assistant_payload = {
        "probability_true": record.get("probability_true", 0.5),
        "verdict": record.get("verdict", record.get("binary_label", "true")),
        "reasoning": record.get("reasoning", ""),
        "incongruences": record.get("incongruences", []),
    }
    
    # Use the build_chat_messages function from common.py
    messages = build_chat_messages(record["input_text"], assistant_payload)
    
    return {
        "messages": messages,
        "metadata": {
            "claim_id": record.get("claim_id"),
            "binary_label": record.get("binary_label"),
            "original_label": record.get("original_label"),
        }
    }


def convert_dataset(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Convert an entire dataset file to chat format.
    
    Returns stats about the conversion.
    """
    records = load_jsonl(input_path)
    chat_records = []
    
    errors = []
    for i, record in enumerate(records):
        try:
            chat_record = convert_record_to_chat(record)
            chat_records.append(chat_record)
        except Exception as e:
            errors.append({"index": i, "claim_id": record.get("claim_id"), "error": str(e)})
    
    save_jsonl(chat_records, output_path)
    
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_records": len(records),
        "converted": len(chat_records),
        "errors": len(errors),
        "error_details": errors[:10] if errors else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export claim SFT data to MLX-LM chat format.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CHAT_EXPORT_DIR)
    parser.add_argument("--train-output", type=Path, default=None, help="Override train output path")
    parser.add_argument("--val-output", type=Path, default=None, help="Override val output path")
    parser.add_argument("--test-output", type=Path, default=None, help="Override test output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set output paths
    train_output = args.train_output or args.output_dir / "claim_sft_train_chat.jsonl"
    val_output = args.val_output or args.output_dir / "claim_sft_val_chat.jsonl"
    test_output = args.test_output or args.output_dir / "claim_sft_test_chat.jsonl"
    
    # Convert each split
    train_stats = convert_dataset(args.train_path, train_output)
    val_stats = convert_dataset(args.val_path, val_output)
    test_stats = convert_dataset(args.test_path, test_output)
    
    # Save combined stats
    stats = {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
    }
    stats_path = args.output_dir / "claim_sft_chat_exports_stats.json"
    save_json(stats, stats_path)
    
    print(f"Converted train: {train_stats['converted']}/{train_stats['total_records']} records")
    print(f"Converted val: {val_stats['converted']}/{val_stats['total_records']} records")
    print(f"Converted test: {test_stats['converted']}/{test_stats['total_records']} records")
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
