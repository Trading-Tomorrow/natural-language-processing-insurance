import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

try:
    from .dataset_cleaning import load_and_clean_default_claims
except ImportError:
    from dataset_cleaning import load_and_clean_default_claims


Claim = Dict[str, Any]
Statement = Dict[str, Any]
PairRecord = Dict[str, Any]
ContextMode = Literal["plain", "contextual"]
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DATASET_PATH = BASE_DIR / "data" / "pairwise_dataset.jsonl"
OUTPUT_FULL_DATASET_PATH = BASE_DIR / "data" / "pairwise_dataset_full.jsonl"
OUTPUT_STATS_PATH = BASE_DIR / "data" / "pairwise_dataset_stats.json"
CONTEXTUAL_OUTPUT_DATASET_PATH = BASE_DIR / "data" / "pairwise_dataset_contextual.jsonl"
CONTEXTUAL_OUTPUT_FULL_DATASET_PATH = BASE_DIR / "data" / "pairwise_dataset_full_contextual.jsonl"
CONTEXTUAL_OUTPUT_STATS_PATH = BASE_DIR / "data" / "pairwise_dataset_contextual_stats.json"
RANDOM_SEED = 42

ROLE_TOKEN_MAP = {
    "insured_driver": "<insured_driver>",
    "third_party_driver": "<third_party_driver>",
    "impartial_witness": "<witness>",
}
SUPPORT_MARKERS = (
    "supporting insured driver",
    "supports the insured",
    "supports the insured driver",
    "supports the insured's narrative",
    "supporting insured driver’s narrative",
    "suspiciously well",
    "overly consistent",
    "coordinated",
    "scripted",
    "confirms sudden cut-in, supporting insured",
)
PHANTOM_MARKERS = (
    "phantom",
    "no evidence of second vehicle",
    "missing vehicle evidence",
    "no_third_party_evidence",
    "no supporting evidence of second vehicle",
    "phantom vehicle evidence gap",
    "missing phantom vehicle",
)
DAMAGE_MARKERS = (
    "missing from detected_damages",
    "damage mismatch",
    "damage_mismatch",
    "damage inflation",
    "damage_inflation",
    "class mismatch",
    "class_mismatch",
    "detected_damages mismatch",
    "inconsistent damage",
    "narrative claims more damage than detected",
)
DYNAMICS_MARKERS = (
    "physics mismatch",
    "physics_mismatch",
    "staged dynamic",
    "staged dynamics",
    "staged mechanics",
    "staged collision",
    "minor side scratch does not shatter glass",
    "right-of-way",
)
SCRIPTED_MARKERS = (
    "coordinated",
    "scripted",
    "suspiciously",
    "defensive scriptwriter",
    "impatient aggressor",
    "evasive opportunist",
    "overly controlled",
    "perfect account",
    "rush the claim",
    "threat",
)


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def format_detected_damages(detected_damages: Sequence[Any]) -> str:
    normalized = sorted(
        {
            normalize_space(damage).lower()
            for damage in detected_damages
            if normalize_space(damage)
        }
    )
    return ", ".join(normalized)


def build_statement_text(claim: Claim, statement: Statement, context_mode: ContextMode) -> str:
    role = str(statement.get("role", "")).strip()
    prefix = ROLE_TOKEN_MAP.get(role, f"<{role}>") if role else "<none>"
    text = normalize_space(statement.get("text", ""))

    if context_mode == "plain":
        return f"{prefix} {text}".strip()

    vehicle = normalize_space(statement.get("vehicle", ""))
    incident_type = normalize_space(claim.get("incident_type", ""))
    damages = format_detected_damages(claim.get("detected_damages", []))

    context_parts: List[str] = []
    if incident_type:
        context_parts.append(f"incident_type: {incident_type}")
    if damages:
        context_parts.append(f"detected_damages: {damages}")
    if vehicle and vehicle.lower() != "none":
        context_parts.append(f"vehicle: {vehicle}")
    context_parts.append(prefix)
    context_parts.append(text)
    return " ".join(part for part in context_parts if part).strip()


def build_statement_view(claim: Claim, statement: Statement, context_mode: ContextMode) -> Dict[str, Any]:
    return {
        "claim_id": claim.get("claim_id"),
        "ground_truth_label": claim.get("ground_truth_label"),
        "incident_type": claim.get("incident_type"),
        "detected_damages": list(claim.get("detected_damages", [])),
        "role": statement.get("role"),
        "vehicle": statement.get("vehicle"),
        "raw_text": normalize_space(statement.get("text", "")),
        "text": build_statement_text(claim, statement, context_mode),
    }


def get_claim_relation_label(claim: Claim) -> Optional[str]:
    claim_label = str(claim.get("ground_truth_label", "")).strip()
    indicators_text = " | ".join(claim.get("fraud_indicators", [])).lower()

    if claim_label == "genuine_accident":
        return "supports"

    if any(marker in indicators_text for marker in SUPPORT_MARKERS):
        return "supports"

    if claim_label in {
        "soft_fraud_exaggeration",
        "hard_fraud_staged",
        "hard_fraud_phantom_vehicle",
    }:
        return "contradicts"

    return None


def get_claim_inconsistency_type(claim: Claim) -> str:
    claim_label = normalize_space(claim.get("ground_truth_label", "")).lower()
    indicators = [
        normalize_space(indicator).lower()
        for indicator in claim.get("fraud_indicators", [])
        if normalize_space(indicator)
    ]
    indicators_text = " | ".join(indicators)

    if claim_label == "genuine_accident":
        return "none"

    if any(marker in indicators_text for marker in PHANTOM_MARKERS):
        return "phantom_vehicle"
    if any(marker in indicators_text for marker in DAMAGE_MARKERS):
        return "damage_mismatch"
    if any(marker in indicators_text for marker in DYNAMICS_MARKERS):
        return "dynamics_mismatch"
    if any(marker in indicators_text for marker in SCRIPTED_MARKERS):
        return "scripted_narrative"

    if claim_label == "hard_fraud_phantom_vehicle":
        return "phantom_vehicle"
    if claim_label == "soft_fraud_exaggeration":
        return "damage_mismatch"
    if claim_label == "hard_fraud_staged":
        return "dynamics_mismatch"

    return "none"


def build_candidate_key(text_a: str, text_b: str, label: str) -> Tuple[str, str, str]:
    normalized_a = " ".join(text_a.split()).strip()
    normalized_b = " ".join(text_b.split()).strip()
    return normalized_a, normalized_b, label


def build_within_claim_pairs(
    claims: Sequence[Claim],
    context_mode: ContextMode,
) -> Tuple[List[PairRecord], List[PairRecord]]:
    supports: List[PairRecord] = []
    contradicts: List[PairRecord] = []
    pair_index = 0

    for claim in claims:
        relation_label = get_claim_relation_label(claim)
        if relation_label is None:
            continue
        inconsistency_type = get_claim_inconsistency_type(claim)

        statements = [statement for statement in claim.get("statements", []) if isinstance(statement, dict)]
        insured_statements = [statement for statement in statements if statement.get("role") == "insured_driver"]
        other_statements = [statement for statement in statements if statement.get("role") != "insured_driver"]

        if not insured_statements or not other_statements:
            continue

        for insured_statement in insured_statements:
            for other_statement in other_statements:
                pair_index += 1
                pair = {
                    "pair_id": f"PAIR-{pair_index:06d}",
                    "claim_id_a": claim.get("claim_id"),
                    "claim_id_b": claim.get("claim_id"),
                    "role_a": insured_statement.get("role"),
                    "role_b": other_statement.get("role"),
                    "incident_type_a": claim.get("incident_type"),
                    "incident_type_b": claim.get("incident_type"),
                    "detected_damages_a": list(claim.get("detected_damages", [])),
                    "detected_damages_b": list(claim.get("detected_damages", [])),
                    "raw_text_a": normalize_space(insured_statement.get("text", "")),
                    "raw_text_b": normalize_space(other_statement.get("text", "")),
                    "text_a": build_statement_text(claim, insured_statement, context_mode),
                    "text_b": build_statement_text(claim, other_statement, context_mode),
                    "label": relation_label,
                    "inconsistency_type": inconsistency_type,
                    "pair_origin": "within_claim",
                    "source_claim_label": claim.get("ground_truth_label"),
                    "source_fraud_indicators": list(claim.get("fraud_indicators", [])),
                    "heuristic": (
                        "genuine_claim_support"
                        if claim.get("ground_truth_label") == "genuine_accident"
                        else "fraud_claim_relation_heuristic"
                    ),
                }

                if relation_label == "supports":
                    supports.append(pair)
                else:
                    contradicts.append(pair)

    return supports, contradicts


def sample_neutral_pairs(
    claims: Sequence[Claim],
    target_count: int,
    rng: random.Random,
    context_mode: ContextMode,
) -> List[PairRecord]:
    statements_pool: List[Dict[str, Any]] = []
    for claim in claims:
        for statement in claim.get("statements", []):
            if not isinstance(statement, dict):
                continue
            text = str(statement.get("text", "")).strip()
            role = str(statement.get("role", "")).strip()
            if not text or not role:
                continue
            statements_pool.append(build_statement_view(claim, statement, context_mode))

    neutrals: List[PairRecord] = []
    seen_keys: Set[Tuple[str, str, str]] = set()
    pair_index = 0
    max_attempts = max(target_count * 50, 1000)
    attempts = 0

    while len(neutrals) < target_count and attempts < max_attempts:
        attempts += 1
        statement_a, statement_b = rng.sample(statements_pool, 2)

        if statement_a["claim_id"] == statement_b["claim_id"]:
            continue

        key = build_candidate_key(statement_a["text"], statement_b["text"], "neutral")
        reverse_key = build_candidate_key(statement_b["text"], statement_a["text"], "neutral")
        if key in seen_keys or reverse_key in seen_keys:
            continue

        pair_index += 1
        pair = {
            "pair_id": f"NEUTRAL-{pair_index:06d}",
            "claim_id_a": statement_a["claim_id"],
            "claim_id_b": statement_b["claim_id"],
            "incident_type_a": statement_a["incident_type"],
            "incident_type_b": statement_b["incident_type"],
            "detected_damages_a": statement_a["detected_damages"],
            "detected_damages_b": statement_b["detected_damages"],
            "role_a": statement_a["role"],
            "role_b": statement_b["role"],
            "raw_text_a": statement_a["raw_text"],
            "raw_text_b": statement_b["raw_text"],
            "text_a": statement_a["text"],
            "text_b": statement_b["text"],
            "label": "neutral",
            "inconsistency_type": "none",
            "pair_origin": "cross_claim",
            "source_claim_label_a": statement_a["ground_truth_label"],
            "source_claim_label_b": statement_b["ground_truth_label"],
            "heuristic": "different_claim_pair",
        }
        seen_keys.add(key)
        neutrals.append(pair)

    if len(neutrals) < target_count:
        raise RuntimeError(
            f"Unable to sample the requested number of neutral pairs: {len(neutrals)}/{target_count}."
        )

    return neutrals


def sample_balanced_pairs(
    supports: Sequence[PairRecord],
    contradicts: Sequence[PairRecord],
    neutrals: Sequence[PairRecord],
    rng: random.Random,
) -> List[PairRecord]:
    class_target = min(len(supports), len(contradicts), len(neutrals))
    sampled_supports = rng.sample(list(supports), class_target)
    sampled_contradicts = rng.sample(list(contradicts), class_target)
    sampled_neutrals = rng.sample(list(neutrals), class_target)

    balanced_pairs = sampled_supports + sampled_contradicts + sampled_neutrals
    rng.shuffle(balanced_pairs)
    return balanced_pairs


def save_jsonl(records: Sequence[PairRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_pairwise_dataset(context_mode: ContextMode = "plain") -> Tuple[List[PairRecord], List[PairRecord], Dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    claims, cleaning_stats = load_and_clean_default_claims()
    supports, contradicts = build_within_claim_pairs(claims, context_mode=context_mode)
    full_neutral_target = len(supports) + len(contradicts)
    full_neutrals = sample_neutral_pairs(claims, full_neutral_target, rng, context_mode=context_mode)
    balanced_neutrals = rng.sample(full_neutrals, min(len(supports), len(contradicts)))
    balanced_pairs = sample_balanced_pairs(supports, contradicts, balanced_neutrals, rng)
    full_pairs = list(supports) + list(contradicts) + list(full_neutrals)
    rng.shuffle(full_pairs)

    label_counts = Counter(pair["label"] for pair in balanced_pairs)
    full_label_counts = Counter(pair["label"] for pair in full_pairs)
    inconsistency_counts = Counter(pair["inconsistency_type"] for pair in balanced_pairs)
    full_inconsistency_counts = Counter(pair["inconsistency_type"] for pair in full_pairs)
    role_pair_counts = Counter(
        tuple(sorted((pair["role_a"], pair["role_b"])))
        for pair in balanced_pairs
    )

    stats = {
        "cleaning": cleaning_stats,
        "candidate_counts": {
            "supports": len(supports),
            "contradicts": len(contradicts),
        },
        "full_counts": dict(full_label_counts),
        "balanced_counts": dict(label_counts),
        "full_inconsistency_counts": dict(full_inconsistency_counts),
        "balanced_inconsistency_counts": dict(inconsistency_counts),
        "full_examples": len(full_pairs),
        "final_examples": len(balanced_pairs),
        "neutral_examples_generated": len(full_neutrals),
        "role_pair_distribution": {
            " | ".join(role_pair): count for role_pair, count in role_pair_counts.items()
        },
        "context_mode": context_mode,
        "random_seed": RANDOM_SEED,
    }
    return balanced_pairs, full_pairs, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the weakly supervised pairwise dataset.")
    parser.add_argument(
        "--context-mode",
        choices=("plain", "contextual"),
        default="plain",
        help="Choose whether statement text should include only role prefix or explicit claim context.",
    )
    parser.add_argument("--output-dataset-path", type=Path, default=None)
    parser.add_argument("--output-full-dataset-path", type=Path, default=None)
    parser.add_argument("--output-stats-path", type=Path, default=None)
    return parser.parse_args()


def resolve_output_paths(
    context_mode: ContextMode,
    output_dataset_path: Optional[Path],
    output_full_dataset_path: Optional[Path],
    output_stats_path: Optional[Path],
) -> Tuple[Path, Path, Path]:
    if output_dataset_path is not None and output_full_dataset_path is not None and output_stats_path is not None:
        return output_dataset_path, output_full_dataset_path, output_stats_path

    if context_mode == "contextual":
        return (
            output_dataset_path or CONTEXTUAL_OUTPUT_DATASET_PATH,
            output_full_dataset_path or CONTEXTUAL_OUTPUT_FULL_DATASET_PATH,
            output_stats_path or CONTEXTUAL_OUTPUT_STATS_PATH,
        )

    return (
        output_dataset_path or OUTPUT_DATASET_PATH,
        output_full_dataset_path or OUTPUT_FULL_DATASET_PATH,
        output_stats_path or OUTPUT_STATS_PATH,
    )


def main() -> None:
    args = parse_args()
    context_mode: ContextMode = args.context_mode
    output_dataset_path, output_full_dataset_path, output_stats_path = resolve_output_paths(
        context_mode=context_mode,
        output_dataset_path=args.output_dataset_path,
        output_full_dataset_path=args.output_full_dataset_path,
        output_stats_path=args.output_stats_path,
    )

    balanced_pairs, full_pairs, stats = build_pairwise_dataset(context_mode=context_mode)
    stats["output_path"] = str(output_dataset_path)
    stats["full_output_path"] = str(output_full_dataset_path)
    save_jsonl(balanced_pairs, output_dataset_path)
    save_jsonl(full_pairs, output_full_dataset_path)
    output_stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Context mode: {context_mode}")
    print(f"Saved pairwise dataset to: {output_dataset_path}")
    print(f"Saved full pairwise dataset to: {output_full_dataset_path}")
    print(f"Saved stats to: {output_stats_path}")
    print(f"Final examples: {stats['final_examples']}")
    print(f"Balanced counts: {stats['balanced_counts']}")


if __name__ == "__main__":
    main()
