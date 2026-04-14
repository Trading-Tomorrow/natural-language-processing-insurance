from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gemma4.common import (
    DEFAULT_SOURCE_PATH,
    DEFAULT_SOURCE_STATS_PATH,
    build_source_record,
    label_distribution,
    save_json,
    save_jsonl,
)
from transformer.dataset_cleaning import DEFAULT_DATASET_PATHS, load_and_clean_claims, print_cleaning_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the claim-level source dataset for Gemma 4 SFT.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_SOURCE_STATS_PATH)
    parser.add_argument(
        "--input-paths",
        type=Path,
        nargs="*",
        default=list(DEFAULT_DATASET_PATHS),
        help="Optional override for the source claim datasets.",
    )
    parser.add_argument(
        "--pairwise-diagnostics-path",
        type=Path,
        default=None,
        help="Optional JSON or JSONL file keyed by claim_id with offline pairwise diagnostics.",
    )
    return parser.parse_args()


def load_pairwise_diagnostics(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            return {str(key): value for key, value in rows.items()}

    diagnostics: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        claim_id = str(row.get("claim_id", "")).strip()
        if claim_id:
            diagnostics[claim_id] = row
    return diagnostics


def main() -> None:
    args = parse_args()
    claims, cleaning_stats = load_and_clean_claims(args.input_paths)
    pairwise_diagnostics = load_pairwise_diagnostics(args.pairwise_diagnostics_path)

    records: List[Dict[str, Any]] = []
    for claim in claims:
        claim_id = str(claim.get("claim_id", "")).strip()
        record = build_source_record(claim, pairwise_diagnostics=pairwise_diagnostics.get(claim_id))
        records.append(record)

    stats = {
        "cleaning": cleaning_stats,
        "num_records": len(records),
        "binary_label_distribution": label_distribution(records, "binary_label"),
        "original_label_distribution": label_distribution(records, "original_label"),
        "pairwise_diagnostics_attached": sum(1 for row in records if row.get("pairwise_diagnostics") is not None),
        "output_path": str(args.output_path),
    }

    save_jsonl(records, args.output_path)
    save_json(stats, args.stats_path)
    print_cleaning_report(cleaning_stats)
    print(f"\nSaved claim source dataset to: {args.output_path}")
    print(f"Saved source stats to: {args.stats_path}")
    print(f"Source records: {len(records)}")
    print(f"Binary labels: {stats['binary_label_distribution']}")
    print(f"Original labels: {stats['original_label_distribution']}")


if __name__ == "__main__":
    main()
