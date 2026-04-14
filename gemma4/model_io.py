from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def resolve_torch_dtype(name: str) -> Optional[torch.dtype]:
    normalized = (name or "auto").lower()
    if normalized == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[normalized]


def load_model_and_tokenizer(
    model_path: str,
    *,
    adapter_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    load_in_4bit: bool = False,
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
) -> Tuple[Any, Any]:
    try:
        tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    backend_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if getattr(backend_tokenizer, "pad_token", None) is None and getattr(backend_tokenizer, "eos_token", None) is not None:
        backend_tokenizer.pad_token = backend_tokenizer.eos_token
    if hasattr(tokenizer, "tokenizer"):
        tokenizer.tokenizer = backend_tokenizer

    dtype = resolve_torch_dtype(torch_dtype)
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("bitsandbytes / 4-bit loading requested, but BitsAndBytesConfig is unavailable.") from exc

        compute_dtype = dtype or torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("adapter loading requested, but peft is not installed.") from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def encode_text(processor_or_tokenizer: Any, prompt: str) -> Dict[str, torch.Tensor]:
    try:
        encoded = processor_or_tokenizer(text=prompt, return_tensors="pt")
    except TypeError:
        encoded = processor_or_tokenizer(prompt, return_tensors="pt")
    return encoded


def decode_tokens(processor_or_tokenizer: Any, token_ids: torch.Tensor) -> str:
    if hasattr(processor_or_tokenizer, "decode"):
        return processor_or_tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    backend_tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    return backend_tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def get_pad_token_id(processor_or_tokenizer: Any) -> Optional[int]:
    if hasattr(processor_or_tokenizer, "pad_token_id"):
        return processor_or_tokenizer.pad_token_id
    backend_tokenizer = getattr(processor_or_tokenizer, "tokenizer", None)
    return getattr(backend_tokenizer, "pad_token_id", None)


def get_eos_token_id(processor_or_tokenizer: Any) -> Optional[int]:
    if hasattr(processor_or_tokenizer, "eos_token_id"):
        return processor_or_tokenizer.eos_token_id
    backend_tokenizer = getattr(processor_or_tokenizer, "tokenizer", None)
    return getattr(backend_tokenizer, "eos_token_id", None)
