import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union


Claim = Dict[str, Any]
PathLike = Union[str, Path]
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATHS = (
    BASE_DIR / "data" / "dataset_sintetico_gemini.json",
    BASE_DIR / "data" / "dataset_sintetico_gemini_mixed_diverse.json",
    BASE_DIR / "data" / "dataset_sintetico_gemini_good_only.json",
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def load_claims_from_path(path: PathLike) -> List[Claim]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON array in {dataset_path}.")

    for index, claim in enumerate(payload):
        if not isinstance(claim, dict):
            raise ValueError(f"Expected claim object at index {index} in {dataset_path}.")

    return payload


def claim_fingerprint(claim: Claim) -> str:
    statements = claim.get("statements", [])
    statement_parts = []

    for statement in statements:
        if not isinstance(statement, dict):
            continue

        statement_parts.append(
            "::".join(
                [
                    normalize_text(statement.get("role")),
                    normalize_text(statement.get("vehicle")),
                    normalize_text(statement.get("text")),
                ]
            )
        )

    damage_key = "|".join(
        sorted(normalize_text(damage) for damage in claim.get("detected_damages", []))
    )

    content_key = "||".join(
        [
            normalize_text(claim.get("location")),
            normalize_text(claim.get("incident_type")),
            damage_key,
            "||".join(statement_parts),
        ]
    )
    return hashlib.sha256(content_key.encode("utf-8")).hexdigest()


def load_and_clean_claims(paths: Sequence[PathLike]) -> Tuple[List[Claim], Dict[str, Any]]:
    cleaned_claims: List[Claim] = []
    seen_ids = set()
    seen_fingerprints = set()
    duplicate_ids = 0
    duplicate_fingerprints = 0
    source_stats = []
    total_loaded = 0

    for raw_path in paths:
        path = Path(raw_path)
        claims = load_claims_from_path(path)
        source_stats.append(
            {
                "path": _display_path(path),
                "loaded_claims": len(claims),
            }
        )
        total_loaded += len(claims)

        for claim in claims:
            claim_id = normalize_text(claim.get("claim_id"))
            if claim_id and claim_id in seen_ids:
                duplicate_ids += 1
                continue

            fingerprint = claim_fingerprint(claim)
            if fingerprint in seen_fingerprints:
                duplicate_fingerprints += 1
                continue

            if claim_id:
                seen_ids.add(claim_id)
            seen_fingerprints.add(fingerprint)
            cleaned_claims.append(claim)

    stats = {
        "sources": source_stats,
        "input_claims": total_loaded,
        "duplicate_ids": duplicate_ids,
        "duplicate_fingerprints": duplicate_fingerprints,
        "removed_claims": duplicate_ids + duplicate_fingerprints,
        "final_claims": len(cleaned_claims),
    }
    return cleaned_claims, stats


def load_and_clean_default_claims() -> Tuple[List[Claim], Dict[str, Any]]:
    return load_and_clean_claims(DEFAULT_DATASET_PATHS)


def print_cleaning_report(stats: Dict[str, Any]) -> None:
    print("\n=== DATASET CLEANING REPORT ===")
    for source in stats.get("sources", []):
        print(f"Source: {source['path']} ({source['loaded_claims']} claims)")
    print(f"Input claims: {stats.get('input_claims', 0)}")
    print(f"Duplicate claim_id removals: {stats.get('duplicate_ids', 0)}")
    print(f"Duplicate content removals: {stats.get('duplicate_fingerprints', 0)}")
    print(f"Removed claims total: {stats.get('removed_claims', 0)}")
    print(f"Final unique claims: {stats.get('final_claims', 0)}")
