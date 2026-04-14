"""
MLX-specific model I/O utilities for Qwen3 on Apple Silicon.

This module provides utilities for working with MLX-LM models,
including generation, training configuration, and adapter management.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_mlx_command(cmd: list, *, capture: bool = False) -> tuple:
    """
    Run an MLX-LM command via subprocess.
    Returns (return_code, stdout, stderr).
    """
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True
    )
    return result.returncode, result.stdout if capture else "", result.stderr if capture else ""


def generate_with_mlx(
    model_name: str,
    prompt: str,
    *,
    adapter_path: Optional[str] = None,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> str:
    """
    Generate text using mlx_lm.generate command-line tool.
    Returns the generated text.
    
    Args:
        model_name: HuggingFace model name or local path
        prompt: The input prompt
        adapter_path: Optional path to LoRA adapters
        max_tokens: Maximum tokens to generate
        temp: Sampling temperature (0 = greedy)
        top_p: Top-p sampling parameter
        seed: Random seed for reproducibility
    
    Returns:
        Generated text (excluding the prompt)
    """
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", model_name,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
        "--top-p", str(top_p),
    ]
    
    if adapter_path:
        cmd.extend(["--adapter-path", str(adapter_path)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mlx_lm generate failed: {result.stderr}")
    
    # Extract generated text from output
    output = result.stdout
    # mlx_lm outputs format: "Prompt: ...\n\nGenerated: ...</s>"
    # We want to extract just the generation
    lines = output.strip().split("\n")
    
    # Find where generation starts (after the prompt section)
    in_generation = False
    generated_lines = []
    for line in lines:
        if ">>" in line or line.startswith("==") or line.startswith("--"):
            continue
        if in_generation:
            generated_lines.append(line)
        # Generation typically starts after the prompt is echoed
        if prompt.strip() in line or "Assistant:" in line:
            in_generation = True
    
    # Fallback: return everything after the last blank line
    if not generated_lines:
        parts = output.split("\n\n")
        if len(parts) > 1:
            return parts[-1].strip()
        return output.strip()
    
    return "\n".join(generated_lines).strip()


def create_lora_config(
    model: str,
    data_dir: str,
    adapter_path: str,
    *,
    num_layers: int = 8,
    batch_size: int = 1,
    grad_accumulation_steps: int = 4,
    iters: int = 600,
    learning_rate: float = 1e-5,
    max_seq_length: int = 2048,
    val_batches: int = 25,
    test_batches: int = 100,
    save_every: int = 100,
    steps_per_report: int = 10,
    steps_per_eval: int = 100,
    grad_checkpoint: bool = True,
    mask_prompt: bool = True,
    lora_rank: int = 8,
    lora_scale: float = 20.0,
    lora_dropout: float = 0.0,
    lora_keys: Optional[list] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create a MLX-LM LoRA training configuration dictionary.
    
    This can be saved as YAML for use with `mlx_lm.lora --config`.
    """
    if lora_keys is None:
        lora_keys = ["self_attn.q_proj", "self_attn.v_proj"]
    
    config = {
        "model": model,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "data": str(data_dir),
        "seed": seed,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "iters": iters,
        "val_batches": val_batches,
        "learning_rate": learning_rate,
        "steps_per_report": steps_per_report,
        "steps_per_eval": steps_per_eval,
        "grad_accumulation_steps": grad_accumulation_steps,
        "adapter_path": str(adapter_path),
        "save_every": save_every,
        "test": False,
        "test_batches": test_batches,
        "max_seq_length": max_seq_length,
        "grad_checkpoint": grad_checkpoint,
        "mask_prompt": mask_prompt,
        "lora_parameters": {
            "keys": lora_keys,
            "rank": lora_rank,
            "scale": lora_scale,
            "dropout": lora_dropout,
        },
    }
    return config


def save_lora_config(config: Dict[str, Any], path: Path) -> None:
    """Save LoRA config to YAML file."""
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def load_lora_config(path: Path) -> Dict[str, Any]:
    """Load LoRA config from YAML file."""
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_lora_training(config_path: Path) -> int:
    """
    Run MLX-LM LoRA training with a YAML config file.
    Returns the exit code.
    """
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path),
    ]
    result = subprocess.run(cmd)
    return result.returncode


def run_lora_evaluation(
    model: str,
    adapter_path: Path,
    data_dir: Path,
    *,
    test_batches: int = 100,
    batch_size: int = 1,
    max_seq_length: int = 2048,
) -> int:
    """
    Run MLX-LM evaluation on test set.
    Returns the exit code.
    """
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", str(model),
        "--adapter-path", str(adapter_path),
        "--data", str(data_dir),
        "--test",
        "--test-batches", str(test_batches),
        "--batch-size", str(batch_size),
        "--max-seq-length", str(max_seq_length),
    ]
    result = subprocess.run(cmd)
    return result.returncode


def fuse_adapters(
    model: str,
    adapter_path: Path,
    save_path: Path,
) -> int:
    """
    Fuse LoRA adapters into the base model.
    Returns the exit code.
    """
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", str(model),
        "--adapter-path", str(adapter_path),
        "--save-path", str(save_path),
    ]
    result = subprocess.run(cmd)
    return result.returncode
