from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gemma4.common import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    extract_json_object,
    render_chat_prompt,
    serialize_claim_for_student,
    validate_target_json,
)
from gemma4.model_io import load_model_and_tokenizer
from gemma4.model_io import decode_tokens, encode_text, get_eos_token_id, get_pad_token_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-claim inference with base or fine-tuned Gemma 4.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--claim-json-path", type=Path, default=None)
    parser.add_argument("--input-text-path", type=Path, default=None)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_input_text(args: argparse.Namespace) -> str:
    if args.claim_json_path is not None:
        claim = json.loads(args.claim_json_path.read_text(encoding="utf-8"))
        if not isinstance(claim, dict):
            raise ValueError("claim-json-path must contain one JSON object.")
        return serialize_claim_for_student(claim)
    if args.input_text_path is not None:
        return args.input_text_path.read_text(encoding="utf-8").strip()
    raise ValueError("Provide --claim-json-path or --input-text-path.")


def main() -> None:
    args = parse_args()
    input_text = load_input_text(args)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        adapter_path=args.adapter_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    prompt = render_chat_prompt(tokenizer, input_text)
    encoded = encode_text(tokenizer, prompt)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": get_pad_token_id(tokenizer),
        "eos_token_id": get_eos_token_id(tokenizer),
    }
    with torch.no_grad():
        output = model.generate(**encoded, **generation_kwargs)

    new_tokens = output[0][encoded["input_ids"].shape[1] :]
    raw_text = decode_tokens(tokenizer, new_tokens)
    parsed, parse_error = extract_json_object(raw_text)

    result: Dict[str, Any] = {
        "raw_output": raw_text,
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

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
