"""
Run single-claim inference with base or fine-tuned Qwen3 using MLX.

This script provides inference capabilities for the Qwen3 model on Apple Silicon,
using MLX-LM for generation with optional LoRA adapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen3.common import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    SYSTEM_PROMPT,
    extract_json_object,
    serialize_claim_for_student,
    validate_target_json,
)
from qwen3.model_io import generate_with_mlx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-claim inference with Qwen3 on MLX.")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--claim-json-path", type=Path, default=None)
    parser.add_argument("--input-text-path", type=Path, default=None)
    parser.add_argument("--input-text", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-json", action="store_true", help="Output as JSON with metadata")
    return parser.parse_args()


def load_input_text(args: argparse.Namespace) -> str:
    """Load input text from various sources."""
    if args.input_text:
        return args.input_text
    
    if args.claim_json_path is not None:
        claim = json.loads(args.claim_json_path.read_text(encoding="utf-8"))
        if not isinstance(claim, dict):
            raise ValueError("claim-json-path must contain one JSON object.")
        return serialize_claim_for_student(claim)
    
    if args.input_text_path is not None:
        return args.input_text_path.read_text(encoding="utf-8").strip()
    
    raise ValueError("Provide --claim-json-path, --input-text-path, or --input-text.")


def build_prompt(input_text: str) -> str:
    """Build the full prompt for the model."""
    return f"""{SYSTEM_PROMPT}

Claim:
{input_text}

Respond with a JSON object containing:
- probability_true: a number between 0.0 and 1.0
- verdict: "true" or "not_true"
- reasoning: a brief explanation
- incongruences: a list of suspicious elements (empty if genuine)

JSON response:"""


def main() -> None:
    args = parse_args()
    input_text = load_input_text(args)
    prompt = build_prompt(input_text)
    
    try:
        raw_output = generate_with_mlx(
            model_name=args.model,
            prompt=prompt,
            adapter_path=str(args.adapter_path) if args.adapter_path else None,
            max_tokens=args.max_new_tokens,
            temp=args.temperature,
            seed=args.seed,
        )
    except Exception as exc:
        print(json.dumps({
            "error": str(exc),
            "raw_output": None,
            "parsed_output": None,
            "schema_valid": False,
            "validation_errors": [str(exc)],
        }, indent=2))
        return
    
    parsed, parse_error = extract_json_object(raw_output)
    
    result: Dict[str, Any] = {
        "raw_output": raw_output,
        "parsed_output": parsed,
        "parse_error": parse_error,
        "schema_valid": False,
        "validation_errors": [],
    }
    
    if parsed is not None:
        ok, errors, cleaned = validate_target_json(parsed)
        result["schema_valid"] = ok
        result["validation_errors"] = errors
        result["parsed_output"] = cleaned if ok else parsed
    
    if args.output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Raw output:")
        print(raw_output)
        if parsed:
            print(f"\nParsed:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        if result["validation_errors"]:
            print(f"\nValidation errors:")
            for err in result["validation_errors"]:
                print(f"  - {err}")


if __name__ == "__main__":
    main()
