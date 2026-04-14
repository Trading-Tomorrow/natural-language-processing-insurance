from typing import Dict, List


LABEL_TO_ID: Dict[str, int] = {
    "supports": 0,
    "neutral": 1,
    "contradicts": 2,
}
ID_TO_LABEL: Dict[int, str] = {value: key for key, value in LABEL_TO_ID.items()}

INCONSISTENCY_TO_ID: Dict[str, int] = {
    "none": 0,
    "damage_mismatch": 1,
    "dynamics_mismatch": 2,
    "phantom_vehicle": 3,
    "scripted_narrative": 4,
}
ID_TO_INCONSISTENCY: Dict[int, str] = {
    value: key for key, value in INCONSISTENCY_TO_ID.items()
}
INCONSISTENCY_DESCRIPTIONS: Dict[str, str] = {
    "none": "No explicit inconsistency signal detected between the two stories.",
    "damage_mismatch": "The reported damage classes do not line up with the narrative or the visual evidence.",
    "dynamics_mismatch": "The accident dynamics, movement, or impact logic do not align between the stories.",
    "phantom_vehicle": "One story depends on a missing, unsupported, or weakly grounded third vehicle.",
    "scripted_narrative": "The stories look suspiciously coordinated, overly controlled, or narratively artificial.",
}


def label_to_id(label: str) -> int:
    normalized_label = str(label).strip().lower()
    if normalized_label not in LABEL_TO_ID:
        raise ValueError(f"Unsupported pairwise label: {label}")
    return LABEL_TO_ID[normalized_label]


def id_to_label(label_id: int) -> str:
    if label_id not in ID_TO_LABEL:
        raise ValueError(f"Unsupported pairwise label id: {label_id}")
    return ID_TO_LABEL[label_id]


def inconsistency_to_id(label: str) -> int:
    normalized_label = str(label).strip().lower()
    if normalized_label not in INCONSISTENCY_TO_ID:
        raise ValueError(f"Unsupported inconsistency label: {label}")
    return INCONSISTENCY_TO_ID[normalized_label]


def id_to_inconsistency(label_id: int) -> str:
    if label_id not in ID_TO_INCONSISTENCY:
        raise ValueError(f"Unsupported inconsistency label id: {label_id}")
    return ID_TO_INCONSISTENCY[label_id]


def describe_inconsistency(label: str) -> str:
    normalized_label = str(label).strip().lower()
    if normalized_label not in INCONSISTENCY_DESCRIPTIONS:
        raise ValueError(f"Unsupported inconsistency label: {label}")
    return INCONSISTENCY_DESCRIPTIONS[normalized_label]


def build_token_type_ids(input_ids: List[int], sep_token_id: int, pad_token_id: int) -> List[int]:
    token_type_ids: List[int] = []
    current_segment = 0
    sep_count = 0

    for token_id in input_ids:
        if token_id == pad_token_id:
            token_type_ids.append(0)
            continue

        token_type_ids.append(current_segment)

        if token_id == sep_token_id:
            sep_count += 1
            if sep_count == 1:
                current_segment = 1

    return token_type_ids
