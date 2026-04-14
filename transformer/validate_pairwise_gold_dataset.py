import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_gold_annotation_template.jsonl"
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
    parser = argparse.ArgumentParser(description="Validate a manually annotated pairwise gold dataset.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow pending examples with blank annotation fields.",
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


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def validate_record(record: Dict[str, Any], index: int, allow_pending: bool) -> List[str]:
    errors: List[str] = []
    missing_keys = [key for key in REQUIRED_KEYS if key not in record]
    if missing_keys:
        errors.append(f"record {index}: missing keys {missing_keys}")
        return errors

    for text_key in ("text_a", "text_b"):
        if not normalize_space(record.get(text_key, "")):
            errors.append(f"record {index}: {text_key} is empty")

    for damages_key in ("detected_damages_a", "detected_damages_b"):
        damages = record.get(damages_key)
        if not isinstance(damages, list):
            errors.append(f"record {index}: {damages_key} must be a list")

    annotation_status = normalize_space(record.get("annotation_status", "")).lower()
    gold_label = normalize_space(record.get("gold_label", "")).lower()
    inconsistency_type = normalize_space(record.get("gold_inconsistency_type", "")).lower()
    rationale_short = normalize_space(record.get("rationale_short", ""))
    annotator_id = normalize_space(record.get("annotator_id", ""))

    if allow_pending and annotation_status == "pending":
        return errors

    if gold_label not in ALLOWED_RELATION_LABELS:
        errors.append(f"record {index}: invalid gold_label '{gold_label}'")
    if inconsistency_type not in ALLOWED_INCONSISTENCY_TYPES:
        errors.append(f"record {index}: invalid gold_inconsistency_type '{inconsistency_type}'")
    if not rationale_short:
        errors.append(f"record {index}: rationale_short is empty")
    if not annotator_id:
        errors.append(f"record {index}: annotator_id is empty")

    if gold_label in {"supports", "neutral"} and inconsistency_type != "none":
        errors.append(
            f"record {index}: gold_inconsistency_type must be 'none' when gold_label is '{gold_label}'"
        )
    if gold_label == "contradicts" and inconsistency_type == "none":
        errors.append(
            f"record {index}: gold_inconsistency_type cannot be 'none' when gold_label is 'contradicts'"
        )

    return errors


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_path)
    all_errors: List[str] = []
    seen_ids = set()

    for index, record in enumerate(records, start=1):
        gold_pair_id = normalize_space(record.get("gold_pair_id", ""))
        if not gold_pair_id:
            all_errors.append(f"record {index}: gold_pair_id is empty")
        elif gold_pair_id in seen_ids:
            all_errors.append(f"record {index}: duplicate gold_pair_id '{gold_pair_id}'")
        else:
            seen_ids.add(gold_pair_id)

        all_errors.extend(validate_record(record, index, allow_pending=args.allow_pending))

    if all_errors:
        print(f"Validation failed for {args.input_path}")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Validation passed for {args.input_path}")
    print(f"Records: {len(records)}")


if __name__ == "__main__":
    main()
