from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from qwen3.common import (
    SYSTEM_PROMPT,
    extract_json_object,
    serialize_claim_for_student,
    validate_target_json,
)
from qwen3.model_io import generate_with_mlx


def build_claim_payload(
    claim_id: str,
    location: str,
    incident_type: str,
    detected_damages: List[str],
    statements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "claim_id": claim_id,
        "location": location,
        "incident_type": incident_type,
        "detected_damages": detected_damages,
        "statements": statements,
    }


def build_prompt(payload: Dict[str, Any]) -> str:
    input_text = serialize_claim_for_student(payload)
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{input_text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_inference(
    *,
    model_id: str,
    adapter_path: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    claim_id: str,
    location: str,
    incident_type: str,
    detected_damages: List[str],
    statements: List[Dict[str, Any]],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], bool, List[str]]:
    payload = build_claim_payload(
        claim_id=claim_id,
        location=location,
        incident_type=incident_type,
        detected_damages=detected_damages,
        statements=statements,
    )
    prompt = build_prompt(payload)
    raw_output = generate_with_mlx(
        model_name=model_id,
        prompt=prompt,
        adapter_path=adapter_path,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
    )
    if "==========" in raw_output:
        raw_output = raw_output.split("==========", 1)[0].strip()
    parsed, parse_error = extract_json_object(raw_output)
    schema_valid = False
    validation_errors: List[str] = []
    cleaned: Optional[Dict[str, Any]] = None
    if parsed is not None:
        ok, errors, cleaned = validate_target_json(parsed)
        schema_valid = ok
        validation_errors = errors
    if schema_valid and cleaned is not None:
        raw_output = json.dumps(cleaned, ensure_ascii=False)
    return (
        raw_output,
        cleaned if cleaned is not None else parsed,
        parse_error,
        schema_valid,
        validation_errors,
    )
