from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = DATA_DIR / "exports"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_BASE_MODEL_ID = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
DEFAULT_SOURCE_PATH = DATA_DIR / "claim_teacher_source.jsonl"
DEFAULT_SOURCE_STATS_PATH = DATA_DIR / "claim_teacher_source_stats.json"
DEFAULT_FULL_DATASET_PATH = DATA_DIR / "claim_sft_full.jsonl"
DEFAULT_TEACHER_STATS_PATH = DATA_DIR / "claim_sft_teacher_stats.json"
DEFAULT_SPLIT_STATS_PATH = DATA_DIR / "claim_sft_split_stats.json"
DEFAULT_TRAIN_PATH = DATA_DIR / "claim_sft_train.jsonl"
DEFAULT_VAL_PATH = DATA_DIR / "claim_sft_val.jsonl"
DEFAULT_TEST_PATH = DATA_DIR / "claim_sft_test.jsonl"
DEFAULT_CHAT_EXPORT_DIR = EXPORT_DIR
DEFAULT_BENCHMARK_OUTPUT_DIR = BASE_DIR / "outputs"

DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_RATIO = 0.1
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_NEW_TOKENS = 256

TEACHER_PROMPT_VERSION = "claim_sft_teacher_v1"
CHAT_PROMPT_VERSION = "claim_truthfulness_json_v1"

VERDICT_TRUE = "true"
VERDICT_NOT_TRUE = "not_true"
ALLOWED_VERDICTS = {VERDICT_TRUE, VERDICT_NOT_TRUE}

ORIGINAL_TO_BINARY = {
    "genuine_accident": VERDICT_TRUE,
    "soft_fraud_exaggeration": VERDICT_NOT_TRUE,
    "hard_fraud_staged": VERDICT_NOT_TRUE,
    "hard_fraud_phantom_vehicle": VERDICT_NOT_TRUE,
}

BANNED_REASONING_TERMS = (
    "ground_truth_label",
    "fraud_indicators",
    "genuine_accident",
    "soft_fraud_exaggeration",
    "hard_fraud_staged",
    "hard_fraud_phantom_vehicle",
    "teacher",
    "synthetic dataset",
    "binary_label",
)

SYSTEM_PROMPT = (
    "You are an insurance claim consistency analyst. "
    "You receive one structured accident claim with detected damages and party statements. "
    "Return valid JSON only. "
    "Estimate the probability that the claim is true, choose verdict=true or verdict=not_true, "
    "explain the decision briefly, and list the main incongruences if they exist."
)

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

SUBTYPE_THEME_KEYWORDS = {
    "soft_fraud_exaggeration": (
        "minor impact",
        "low-speed",
        "low speed",
        "disproportionate",
        "exaggerated",
        "severity",
        "injury",
        "compensation",
    ),
    "hard_fraud_staged": (
        "staged",
        "coordinated",
        "scripted",
        "arranged",
        "pre-arranged",
        "sudden braking",
        "unusual narrative",
    ),
    "hard_fraud_phantom_vehicle": (
        "phantom",
        "unknown vehicle",
        "unidentified vehicle",
        "vehicle disappeared",
        "no witness",
        "vanished",
        "missing vehicle",
    ),
}


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


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


def save_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def collapse_claim_label(original_label: str) -> str:
    normalized = normalize_space(original_label)
    if normalized not in ORIGINAL_TO_BINARY:
        raise ValueError(f"Unsupported claim label: {original_label}")
    return ORIGINAL_TO_BINARY[normalized]


def serialize_claim_for_student(claim: Dict[str, Any]) -> str:
    damages = claim.get("detected_damages", [])
    damage_text = ", ".join(str(damage) for damage in damages) if damages else "none"
    lines = [
        f"Claim ID: {normalize_space(claim.get('claim_id'))}",
        f"Location: {normalize_space(claim.get('location'))}",
        f"Incident type: {normalize_space(claim.get('incident_type'))}",
        f"Detected damages: {damage_text}",
        "",
        "Statements:",
    ]

    statements = claim.get("statements", [])
    for index, statement in enumerate(statements, start=1):
        role = normalize_space(statement.get("role"))
        vehicle = normalize_space(statement.get("vehicle"))
        text = normalize_space(statement.get("text"))
        lines.extend(
            [
                f"{index}. Role: {role}",
                f"   Vehicle: {vehicle or 'unknown'}",
                f"   Text: {text}",
                "",
            ]
        )

    return "\n".join(lines).strip()


def build_chat_messages(input_text: str, assistant_payload: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]
    if assistant_payload is not None:
        messages.append({"role": "assistant", "content": target_json_to_text(assistant_payload)})
    return messages


def target_json_to_text(payload: Dict[str, Any]) -> str:
    ordered = {
        "probability_true": payload["probability_true"],
        "verdict": payload["verdict"],
        "reasoning": payload["reasoning"],
        "incongruences": payload["incongruences"],
    }
    return json.dumps(ordered, ensure_ascii=False)


def _normalize_incongruences(value: Any) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    if value is None:
        return [], errors
    if not isinstance(value, list):
        return [], ["incongruences must be a list"]

    cleaned: List[str] = []
    for item in value:
        text = normalize_space(item)
        if not text:
            continue
        cleaned.append(text)
    return cleaned, errors


def validate_target_json(
    payload: Any,
    *,
    enforce_reasoning_guardrails: bool = False,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["target payload must be a JSON object"], None

    try:
        probability_true = float(payload.get("probability_true"))
    except (TypeError, ValueError):
        errors.append("probability_true must be numeric")
        probability_true = -1.0

    verdict = normalize_space(payload.get("verdict")).lower()
    reasoning = normalize_space(payload.get("reasoning"))
    incongruences, incongruence_errors = _normalize_incongruences(payload.get("incongruences"))
    errors.extend(incongruence_errors)

    if not 0.0 <= probability_true <= 1.0:
        errors.append("probability_true must be between 0 and 1")
    if verdict not in ALLOWED_VERDICTS:
        errors.append("verdict must be true or not_true")
    if not reasoning:
        errors.append("reasoning must be non-empty")
    if verdict == VERDICT_TRUE and probability_true < 0.5:
        errors.append("verdict=true must have probability_true >= 0.5")
    if verdict == VERDICT_NOT_TRUE and probability_true >= 0.5:
        errors.append("verdict=not_true must have probability_true < 0.5")

    if enforce_reasoning_guardrails and reasoning:
        reasoning_lower = reasoning.lower()
        joined_incongruences = " ".join(incongruences).lower()
        for token in BANNED_REASONING_TERMS:
            if token in reasoning_lower or token in joined_incongruences:
                errors.append(f"response leaked hidden supervision term: {token}")
                break

    if errors:
        return False, errors, None

    cleaned = {
        "probability_true": round(probability_true, 4),
        "verdict": verdict,
        "reasoning": reasoning,
        "incongruences": incongruences,
    }
    return True, [], cleaned


def strip_json_fences(text: str) -> str:
    return JSON_FENCE_RE.sub("", text.strip()).strip()


def extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cleaned = strip_json_fences(text)
    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, None
    return None, "could not parse a JSON object from model output"


def render_chat_prompt(tokenizer: Any, input_text: str) -> str:
    messages = build_chat_messages(input_text)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                return tokenizer.apply_chat_template(messages, tokenize=False)

    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"User:\n{input_text}\n\n"
        "Assistant:\n"
    )


def label_distribution(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        label = normalize_space(record.get(key))
        counts[label] = counts.get(label, 0) + 1
    return counts


def subtype_theme_hit(text: str, original_label: str) -> bool:
    lowered = normalize_space(text).lower()
    for keyword in SUBTYPE_THEME_KEYWORDS.get(original_label, ()):
        if keyword in lowered:
            return True
    return False


def build_source_record(claim: Dict[str, Any], pairwise_diagnostics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    original_label = normalize_space(claim.get("ground_truth_label"))
    return {
        "claim_id": normalize_space(claim.get("claim_id")),
        "input_text": serialize_claim_for_student(claim),
        "binary_label": collapse_claim_label(original_label),
        "original_label": original_label,
        "location": normalize_space(claim.get("location")),
        "incident_type": normalize_space(claim.get("incident_type")),
        "detected_damages": [normalize_space(item) for item in claim.get("detected_damages", [])],
        "statements": claim.get("statements", []),
        "fraud_indicators": [normalize_space(item) for item in claim.get("fraud_indicators", [])],
        "pairwise_diagnostics": pairwise_diagnostics,
    }


def default_teacher_model_name() -> str:
    return os.getenv("GEMINI_MODEL_NAME", "models/gemini-3.1-flash-lite-preview")
