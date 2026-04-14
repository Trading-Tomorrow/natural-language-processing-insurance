import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_full.jsonl"
DEFAULT_TEMPLATE_OUTPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_gold_annotation_template.jsonl"
DEFAULT_METADATA_OUTPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_gold_candidate_metadata.jsonl"
DEFAULT_SCHEMA_OUTPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_gold_schema.json"
DEFAULT_GUIDE_OUTPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_gold_guidelines.md"
DEFAULT_STATS_OUTPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_gold_sampling_stats.json"
RANDOM_SEED = 42

TARGET_RELATION_COUNTS = {
    "supports": 100,
    "neutral": 100,
    "contradicts": 100,
}
TARGET_CONTRADICTION_TYPE_COUNTS = {
    "damage_mismatch": 35,
    "dynamics_mismatch": 45,
    "phantom_vehicle": 20,
}
ALLOWED_RELATION_LABELS = ("supports", "neutral", "contradicts")
ALLOWED_INCONSISTENCY_TYPES = ("none", "damage_mismatch", "dynamics_mismatch", "phantom_vehicle")


GUIDELINES_TEXT = """# Pairwise Gold Annotation Guidelines

## Goal

Annotate pairwise accident-statement examples for:

-> `gold_label`
-> `gold_inconsistency_type`
-> `rationale_short`

## Relation Labels

Use exactly one:

-> `supports`: the two texts describe compatible versions of the same event
-> `neutral`: the two texts do not clearly support each other and do not clearly contradict each other
-> `contradicts`: the two texts contain a meaningful factual conflict

## Inconsistency Types

Use exactly one:

-> `none`
-> `damage_mismatch`
-> `dynamics_mismatch`
-> `phantom_vehicle`

Rules:

-> if `gold_label != contradicts`, then `gold_inconsistency_type` must be `none`
-> if `gold_label == contradicts`, choose the single strongest inconsistency type

Definitions:

-> `damage_mismatch`: the statements conflict about what damage exists or what visually happened to the vehicle
-> `dynamics_mismatch`: the statements conflict about motion, impact direction, traffic light state, who hit whom, stopping vs moving, lane behavior, or general accident mechanics
-> `phantom_vehicle`: one story depends on a missing or unsupported extra vehicle that the other story or context does not ground

## Rationale

Write one short sentence describing the main reason for the chosen label.

Good examples:

-> `One text says the insured was stopped and hit from behind, while the other says the insured rolled backward into the other vehicle.`
-> `Both texts describe the same rear-end collision and the same visible rear/front damage.`
-> `The witness only reports the aftermath and does not confirm the insured's version of the collision dynamics.`

## Annotation Policy

-> annotate from the texts and metadata shown in the template
-> do not use the hidden weak labels from the metadata file as annotation truth
-> prefer `neutral` over `contradicts` when the conflict is not explicit enough
-> prefer the most concrete contradiction type when `gold_label = contradicts`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pairwise gold-annotation starter set from the weakly supervised corpus.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--template-output-path", type=Path, default=DEFAULT_TEMPLATE_OUTPUT_PATH)
    parser.add_argument("--metadata-output-path", type=Path, default=DEFAULT_METADATA_OUTPUT_PATH)
    parser.add_argument("--schema-output-path", type=Path, default=DEFAULT_SCHEMA_OUTPUT_PATH)
    parser.add_argument("--guide-output-path", type=Path, default=DEFAULT_GUIDE_OUTPUT_PATH)
    parser.add_argument("--stats-output-path", type=Path, default=DEFAULT_STATS_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
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


def save_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_schema_payload() -> Dict[str, Any]:
    return {
        "description": "Schema for the manually annotated pairwise gold dataset.",
        "allowed_relation_labels": list(ALLOWED_RELATION_LABELS),
        "allowed_inconsistency_types": list(ALLOWED_INCONSISTENCY_TYPES),
        "rules": [
            "If gold_label is not contradicts, gold_inconsistency_type must be none.",
            "If gold_label is contradicts, gold_inconsistency_type must be one of damage_mismatch, dynamics_mismatch, phantom_vehicle.",
            "rationale_short must be a short human explanation of the annotation decision.",
        ],
        "example_record": {
            "gold_pair_id": "GOLD-000001",
            "source_pair_id": "PAIR-000123",
            "claim_id_a": "PT-ABC-2026-001",
            "claim_id_b": "PT-ABC-2026-001",
            "role_a": "insured_driver",
            "role_b": "third_party_driver",
            "incident_type_a": "Rear-end collision",
            "incident_type_b": "Rear-end collision",
            "detected_damages_a": ["dent", "scratch"],
            "detected_damages_b": ["dent", "scratch"],
            "text_a": "I was stopped at the red light when the other vehicle hit me from behind.",
            "text_b": "The insured rolled backward into the front of my vehicle while traffic was still stopped.",
            "gold_label": "contradicts",
            "gold_inconsistency_type": "dynamics_mismatch",
            "rationale_short": "One statement says the insured was stationary and hit from behind, while the other says the insured rolled backward into the other car.",
            "annotation_status": "complete",
            "annotator_id": "annotator_01",
            "review_status": "pending",
        },
    }


def role_pair_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return tuple(sorted((str(record.get("role_a", "")), str(record.get("role_b", "")))))


def diversity_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    claim_id_a = str(record.get("claim_id_a", ""))
    claim_id_b = str(record.get("claim_id_b", ""))
    label = str(record.get("label", ""))
    if claim_id_a == claim_id_b:
        return (claim_id_a, claim_id_b, label)
    ordered_claims = tuple(sorted((claim_id_a, claim_id_b)))
    return ordered_claims + (label,)


def distribute_quota(total: int, buckets: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[Tuple[str, str], int]:
    if not buckets:
        return {}
    available_counts = {bucket: len(records) for bucket, records in buckets.items() if records}
    quotas = {bucket: 0 for bucket in available_counts}
    remaining = total

    while remaining > 0 and available_counts:
        progressed = False
        for bucket in sorted(available_counts.keys()):
            if quotas[bucket] >= available_counts[bucket]:
                continue
            quotas[bucket] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    return quotas


def sample_diverse_records(
    records: Sequence[Dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    buckets: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[role_pair_key(record)].append(record)

    for bucket_records in buckets.values():
        rng.shuffle(bucket_records)

    quotas = distribute_quota(target_count, buckets)
    selected: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[str, str, str]] = set()

    for bucket_key in sorted(buckets.keys()):
        bucket_records = buckets[bucket_key]
        bucket_target = quotas.get(bucket_key, 0)
        bucket_selected = 0
        for record in bucket_records:
            if bucket_selected >= bucket_target:
                break
            key = diversity_key(record)
            if key in seen_keys:
                continue
            selected.append(record)
            seen_keys.add(key)
            bucket_selected += 1

    if len(selected) < target_count:
        remaining_records = list(records)
        rng.shuffle(remaining_records)
        for record in remaining_records:
            if len(selected) >= target_count:
                break
            key = diversity_key(record)
            if key in seen_keys:
                continue
            selected.append(record)
            seen_keys.add(key)

    if len(selected) < target_count:
        raise RuntimeError(f"Unable to sample {target_count} diverse records; only found {len(selected)}.")

    return selected[:target_count]


def build_annotation_template(record: Dict[str, Any], gold_pair_id: str) -> Dict[str, Any]:
    return {
        "gold_pair_id": gold_pair_id,
        "source_pair_id": record.get("pair_id", ""),
        "claim_id_a": record.get("claim_id_a", ""),
        "claim_id_b": record.get("claim_id_b", ""),
        "role_a": record.get("role_a", ""),
        "role_b": record.get("role_b", ""),
        "incident_type_a": record.get("incident_type_a", ""),
        "incident_type_b": record.get("incident_type_b", ""),
        "detected_damages_a": record.get("detected_damages_a", []),
        "detected_damages_b": record.get("detected_damages_b", []),
        "text_a": record.get("raw_text_a", record.get("text_a", "")),
        "text_b": record.get("raw_text_b", record.get("text_b", "")),
        "gold_label": "",
        "gold_inconsistency_type": "",
        "rationale_short": "",
        "annotation_status": "pending",
        "annotator_id": "",
        "review_status": "pending",
    }


def build_metadata_record(record: Dict[str, Any], gold_pair_id: str) -> Dict[str, Any]:
    return {
        "gold_pair_id": gold_pair_id,
        "source_pair_id": record.get("pair_id", ""),
        "weak_label": record.get("label", ""),
        "weak_inconsistency_type": record.get("inconsistency_type", "none"),
        "pair_origin": record.get("pair_origin", ""),
        "heuristic": record.get("heuristic", ""),
        "source_claim_label": record.get("source_claim_label", ""),
        "source_claim_label_a": record.get("source_claim_label_a", ""),
        "source_claim_label_b": record.get("source_claim_label_b", ""),
        "source_fraud_indicators": record.get("source_fraud_indicators", []),
    }


def sample_gold_candidates(
    records: Sequence[Dict[str, Any]],
    rng: random.Random,
    input_path: Path,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    supports_pool = [record for record in records if record.get("label") == "supports"]
    neutral_pool = [record for record in records if record.get("label") == "neutral"]
    contradict_pool = [record for record in records if record.get("label") == "contradicts"]

    sampled_supports = sample_diverse_records(supports_pool, TARGET_RELATION_COUNTS["supports"], rng)
    sampled_neutrals = sample_diverse_records(neutral_pool, TARGET_RELATION_COUNTS["neutral"], rng)

    sampled_contradicts: List[Dict[str, Any]] = []
    contradiction_sampling_stats: Dict[str, int] = {}
    contradiction_seen_keys: Set[Tuple[str, str, str]] = set()

    for inconsistency_type, target_count in TARGET_CONTRADICTION_TYPE_COUNTS.items():
        candidates = [
            record
            for record in contradict_pool
            if record.get("inconsistency_type") == inconsistency_type
        ]
        sampled = sample_diverse_records(candidates, target_count, rng)
        for record in sampled:
            contradiction_seen_keys.add(diversity_key(record))
        sampled_contradicts.extend(sampled)
        contradiction_sampling_stats[inconsistency_type] = len(sampled)

    combined_records = sampled_supports + sampled_neutrals + sampled_contradicts
    rng.shuffle(combined_records)

    gold_templates: List[Dict[str, Any]] = []
    metadata_records: List[Dict[str, Any]] = []
    for index, record in enumerate(combined_records, start=1):
        gold_pair_id = f"GOLD-{index:06d}"
        gold_templates.append(build_annotation_template(record, gold_pair_id))
        metadata_records.append(build_metadata_record(record, gold_pair_id))

    template_relation_counts = Counter(record.get("weak_label", "unknown") for record in metadata_records)
    template_inconsistency_counts = Counter(
        record.get("weak_inconsistency_type", "unknown") for record in metadata_records
    )
    stats = {
        "input_path": str(input_path),
        "seed": seed,
        "target_relation_counts": TARGET_RELATION_COUNTS,
        "target_contradiction_type_counts": TARGET_CONTRADICTION_TYPE_COUNTS,
        "sampled_relation_counts": dict(template_relation_counts),
        "sampled_inconsistency_counts": dict(template_inconsistency_counts),
        "supports_pool_size": len(supports_pool),
        "neutral_pool_size": len(neutral_pool),
        "contradicts_pool_size": len(contradict_pool),
        "sampled_total": len(gold_templates),
        "contradiction_sampling_counts": contradiction_sampling_stats,
    }
    return gold_templates, metadata_records, stats


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    records = load_jsonl(args.input_path)

    gold_templates, metadata_records, stats = sample_gold_candidates(
        records,
        rng,
        input_path=args.input_path,
        seed=args.seed,
    )

    save_jsonl(gold_templates, args.template_output_path)
    save_jsonl(metadata_records, args.metadata_output_path)
    save_json(build_schema_payload(), args.schema_output_path)
    args.guide_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.guide_output_path.write_text(GUIDELINES_TEXT, encoding="utf-8")
    save_json(stats, args.stats_output_path)

    print(f"Input dataset: {args.input_path}")
    print(f"Saved annotation template to: {args.template_output_path}")
    print(f"Saved hidden metadata to: {args.metadata_output_path}")
    print(f"Saved schema to: {args.schema_output_path}")
    print(f"Saved guidelines to: {args.guide_output_path}")
    print(f"Saved sampling stats to: {args.stats_output_path}")
    print(f"Total sampled pairs: {stats['sampled_total']}")
    print(f"Sampled relation counts: {stats['sampled_relation_counts']}")
    print(f"Sampled inconsistency counts: {stats['sampled_inconsistency_counts']}")


if __name__ == "__main__":
    main()
