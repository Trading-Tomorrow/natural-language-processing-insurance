import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_PATH = BASE_DIR / "data" / "pairwise_dataset_gold_annotation_template.jsonl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_gold.jsonl"
ALLOWED_RELATION_LABELS = {"supports", "neutral", "contradicts"}
ALLOWED_INCONSISTENCY_TYPES = {"none", "damage_mismatch", "dynamics_mismatch", "phantom_vehicle"}
REQUIRED_KEYS = (
    "gold_pair_id",
    "source_pair_id",
    "claim_id_a",
    "claim_id_b",
    "role_a",
    "role_b",
    "incident_type_a",
    "incident_type_b",
    "detected_damages_a",
    "detected_damages_b",
    "text_a",
    "text_b",
    "gold_label",
    "gold_inconsistency_type",
    "rationale_short",
    "annotation_status",
    "annotator_id",
    "review_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final pairwise gold dataset from the annotated template.")
    parser.add_argument("--template-path", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="Require review_status to be 'approved' before including examples.",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Skip pending examples instead of failing the build.",
    )
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


def save_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def validate_completed_record(record: Dict[str, Any], index: int, require_reviewed: bool) -> None:
    missing_keys = [key for key in REQUIRED_KEYS if key not in record]
    if missing_keys:
        raise ValueError(f"Record {index} is missing keys {missing_keys}.")

    annotation_status = normalize_space(record.get("annotation_status", "")).lower()
    if annotation_status != "complete":
        raise ValueError(f"Record {index} is not complete.")

    review_status = normalize_space(record.get("review_status", "")).lower()
    if require_reviewed and review_status != "approved":
        raise ValueError(f"Record {index} is not approved.")

    gold_label = normalize_space(record.get("gold_label", "")).lower()
    inconsistency_type = normalize_space(record.get("gold_inconsistency_type", "")).lower()
    rationale_short = normalize_space(record.get("rationale_short", ""))
    annotator_id = normalize_space(record.get("annotator_id", ""))

    if gold_label not in ALLOWED_RELATION_LABELS:
        raise ValueError(f"Record {index} has invalid gold_label '{gold_label}'.")
    if inconsistency_type not in ALLOWED_INCONSISTENCY_TYPES:
        raise ValueError(f"Record {index} has invalid gold_inconsistency_type '{inconsistency_type}'.")
    if not rationale_short:
        raise ValueError(f"Record {index} has empty rationale_short.")
    if not annotator_id:
        raise ValueError(f"Record {index} has empty annotator_id.")

    if gold_label != "contradicts" and inconsistency_type != "none":
        raise ValueError(
            f"Record {index} must have gold_inconsistency_type='none' when gold_label='{gold_label}'."
        )
    if gold_label == "contradicts" and inconsistency_type == "none":
        raise ValueError(f"Record {index} cannot use gold_inconsistency_type='none' for contradicts.")


def build_gold_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_id": record["gold_pair_id"],
        "source_pair_id": record["source_pair_id"],
        "claim_id_a": record["claim_id_a"],
        "claim_id_b": record["claim_id_b"],
        "role_a": record["role_a"],
        "role_b": record["role_b"],
        "incident_type_a": record["incident_type_a"],
        "incident_type_b": record["incident_type_b"],
        "detected_damages_a": record["detected_damages_a"],
        "detected_damages_b": record["detected_damages_b"],
        "text_a": record["text_a"],
        "text_b": record["text_b"],
        "label": normalize_space(record["gold_label"]).lower(),
        "inconsistency_type": normalize_space(record["gold_inconsistency_type"]).lower(),
        "rationale_short": normalize_space(record["rationale_short"]),
        "annotator_id": normalize_space(record["annotator_id"]),
        "review_status": normalize_space(record["review_status"]).lower(),
        "gold_source": "manual_annotation",
    }


def main() -> None:
    args = parse_args()
    template_records = load_jsonl(args.template_path)
    gold_records: List[Dict[str, Any]] = []
    skipped_pending = 0

    for index, record in enumerate(template_records, start=1):
        annotation_status = normalize_space(record.get("annotation_status", "")).lower()
        if args.allow_pending and annotation_status != "complete":
            skipped_pending += 1
            continue
        validate_completed_record(record, index=index, require_reviewed=args.require_reviewed)
        gold_records.append(build_gold_record(record))

    save_jsonl(gold_records, args.output_path)

    print(f"Template path: {args.template_path}")
    print(f"Saved gold dataset to: {args.output_path}")
    print(f"Examples: {len(gold_records)}")
    if args.allow_pending:
        print(f"Skipped pending examples: {skipped_pending}")


if __name__ == "__main__":
    main()
