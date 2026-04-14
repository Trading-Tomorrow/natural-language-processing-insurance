"""
Kaggle script for:
1. Benchmarking base Gemma 4 E4B on the frozen test split
2. Fine-tuning with QLoRA on the claim-level JSONL dataset
3. Benchmarking the fine-tuned model on the same frozen test split

Expected Kaggle inputs:
- Gemma 4 model added from Kaggle Models
- Project dataset uploaded as a Kaggle Dataset containing:
  claim_sft_train.jsonl
  claim_sft_val.jsonl
  claim_sft_test.jsonl

This file is self-contained so it can be pasted into a Kaggle notebook cell
or executed as a script after installing dependencies.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Kaggle path discovery
# ---------------------------------------------------------------------------

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
WORKDIR = Path("/kaggle/working/gemma4_claim_sft")
WORKDIR.mkdir(parents=True, exist_ok=True)

ADAPTER_OUTPUT_DIR = WORKDIR / "gemma4_e4b_claim_lora"
BASE_PREDICTIONS_PATH = WORKDIR / "benchmark_base_gemma4_test.jsonl"
FINETUNED_PREDICTIONS_PATH = WORKDIR / "benchmark_finetuned_gemma4_test.jsonl"
COMPARISON_JSON_PATH = WORKDIR / "benchmark_comparison.json"
COMPARISON_MD_PATH = WORKDIR / "benchmark_comparison.md"

MODEL_PATH_OVERRIDE = os.environ.get("MODEL_PATH_OVERRIDE")
MODEL_HINT = os.environ.get("MODEL_HINT", "gemma-4-e4b").strip().lower()
GPU_RESERVE_GIB = float(os.environ.get("GPU_RESERVE_GIB", "1.0"))
RUN_BASE_BENCHMARK = os.environ.get("RUN_BASE_BENCHMARK", "1").strip() != "0"
RUN_TRAIN = os.environ.get("RUN_TRAIN", "1").strip() != "0"
RUN_FINETUNED_BENCHMARK = os.environ.get("RUN_FINETUNED_BENCHMARK", "1").strip() != "0"


# ---------------------------------------------------------------------------
# Generation + training defaults
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 256
BENCHMARK_TEMPERATURE = 0.0
BENCHMARK_TOP_P = 1.0
SEED = 42

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
VERDICT_TRUE = "true"
VERDICT_NOT_TRUE = "not_true"

SYSTEM_PROMPT = (
    "You are an insurance claim consistency analyst. "
    "You receive one structured accident claim with detected damages and party statements. "
    "Return valid JSON only. "
    "Estimate the probability that the claim is true, choose verdict=true or verdict=not_true, "
    "explain the decision briefly, and list the main incongruences if they exist."
)

SUBTYPE_THEME_KEYWORDS = {
    "soft_fraud_exaggeration": (
        "minor impact",
        "low-speed",
        "low speed",
        "disproportionate",
        "exaggerated",
        "injury",
        "compensation",
        "severity",
    ),
    "hard_fraud_staged": (
        "staged",
        "coordinated",
        "scripted",
        "arranged",
        "pre-arranged",
        "unusual narrative",
        "sudden braking",
    ),
    "hard_fraud_phantom_vehicle": (
        "phantom",
        "unknown vehicle",
        "unidentified vehicle",
        "vanished",
        "missing vehicle",
        "no witness",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def find_dataset_root() -> Path:
    required = {"claim_sft_train.jsonl", "claim_sft_val.jsonl", "claim_sft_test.jsonl"}
    for dirpath, _, filenames in os.walk(KAGGLE_INPUT_ROOT):
        if required.issubset(set(filenames)):
            return Path(dirpath)
    raise FileNotFoundError(
        "Could not locate a dataset directory under /kaggle/input containing "
        "claim_sft_train.jsonl, claim_sft_val.jsonl, and claim_sft_test.jsonl."
    )


def find_model_root() -> Path:
    if MODEL_PATH_OVERRIDE:
        override_path = Path(MODEL_PATH_OVERRIDE)
        if not override_path.exists():
            raise FileNotFoundError(f"MODEL_PATH_OVERRIDE does not exist: {override_path}")
        return override_path

    required = {"config.json", "tokenizer.json", "processor_config.json"}
    preferred: List[Path] = []
    fallback: List[Path] = []
    for dirpath, _, filenames in os.walk(KAGGLE_INPUT_ROOT):
        filename_set = set(filenames)
        if required.issubset(filename_set) and any(name.endswith(".safetensors") for name in filename_set):
            path = Path(dirpath)
            path_lower = str(path).lower()
            if MODEL_HINT and MODEL_HINT in path_lower:
                preferred.append(path)
            else:
                fallback.append(path)
    if preferred:
        return preferred[0]
    if fallback:
        candidates = "\n".join(f"- {path}" for path in fallback[:10])
        raise FileNotFoundError(
            f"Could not find a mounted model matching MODEL_HINT={MODEL_HINT!r}. "
            "Mounted model candidates were:\n"
            f"{candidates}"
        )
    raise FileNotFoundError(
        "Could not locate a Gemma model directory under /kaggle/input with config/tokenizer/processor files."
    )


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at line {line_number} in {path}")
            records.append(payload)
    return records


def save_jsonl(records: Sequence[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    return None, "could not parse JSON object from output"


def validate_target_json(payload: Any) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["payload must be a JSON object"], None

    try:
        probability_true = float(payload.get("probability_true"))
    except (TypeError, ValueError):
        errors.append("probability_true must be numeric")
        probability_true = -1.0

    verdict = normalize_space(payload.get("verdict")).lower()
    reasoning = normalize_space(payload.get("reasoning"))
    incongruences = payload.get("incongruences")
    if not isinstance(incongruences, list):
        errors.append("incongruences must be a list")
        incongruences = []
    incongruences = [normalize_space(item) for item in incongruences if normalize_space(item)]

    if not 0.0 <= probability_true <= 1.0:
        errors.append("probability_true must be in [0, 1]")
    if verdict not in {VERDICT_TRUE, VERDICT_NOT_TRUE}:
        errors.append("verdict must be true or not_true")
    if not reasoning:
        errors.append("reasoning must be non-empty")
    if verdict == VERDICT_TRUE and probability_true < 0.5:
        errors.append("verdict=true requires probability_true >= 0.5")
    if verdict == VERDICT_NOT_TRUE and probability_true >= 0.5:
        errors.append("verdict=not_true requires probability_true < 0.5")

    if errors:
        return False, errors, None

    cleaned = {
        "probability_true": round(probability_true, 4),
        "verdict": verdict,
        "reasoning": reasoning,
        "incongruences": incongruences,
    }
    return True, [], cleaned


def subtype_theme_hit(text: str, original_label: str) -> bool:
    lowered = normalize_space(text).lower()
    for keyword in SUBTYPE_THEME_KEYWORDS.get(original_label, ()):
        if keyword in lowered:
            return True
    return False


def build_max_memory_map() -> Optional[Dict[int, str]]:
    if not torch.cuda.is_available():
        return None

    max_memory: Dict[int, str] = {}
    for device_index in range(torch.cuda.device_count()):
        total_bytes = torch.cuda.get_device_properties(device_index).total_memory
        total_gib = total_bytes / (1024 ** 3)
        usable_gib = max(1, int(total_gib - GPU_RESERVE_GIB))
        max_memory[device_index] = f"{usable_gib}GiB"
    return max_memory


def load_model_config_summary(model_path: str | Path) -> Dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = payload.get("text_config") or {}
    return {
        "model_type": payload.get("model_type"),
        "architectures": payload.get("architectures"),
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_attention_heads": text_config.get("num_attention_heads"),
    }


def load_processor_and_model(model_path: str | Path, load_for_training: bool) -> Tuple[Any, Any]:
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    backend_tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(backend_tokenizer, "pad_token", None) is None and getattr(backend_tokenizer, "eos_token", None) is not None:
        backend_tokenizer.pad_token = backend_tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = backend_tokenizer

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    max_memory = build_max_memory_map()
    device_map = "balanced" if max_memory and len(max_memory) > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        max_memory=max_memory,
        dtype=torch.float16,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    if load_for_training:
        model = prepare_model_for_kbit_training(model)

    return processor, model


def encode_prompt(processor: Any, prompt: str) -> Dict[str, torch.Tensor]:
    try:
        encoded = processor(text=prompt, return_tensors="pt")
    except TypeError:
        encoded = processor(prompt, return_tensors="pt")
    return encoded


def decode_tokens(processor: Any, token_ids: torch.Tensor) -> str:
    if hasattr(processor, "decode"):
        return processor.decode(token_ids, skip_special_tokens=True).strip()
    backend_tokenizer = getattr(processor, "tokenizer", processor)
    return backend_tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def get_pad_token_id(processor: Any) -> Optional[int]:
    if hasattr(processor, "pad_token_id"):
        return processor.pad_token_id
    return getattr(getattr(processor, "tokenizer", processor), "pad_token_id", None)


def get_eos_token_id(processor: Any) -> Optional[int]:
    if hasattr(processor, "eos_token_id"):
        return processor.eos_token_id
    return getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)


def build_chat_messages(input_text: str, assistant_payload: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]
    if assistant_payload is not None:
        messages.append({"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)})
    return messages


def render_chat_prompt(processor: Any, input_text: str, add_generation_prompt: bool) -> str:
    prompt = (
        "System:\n"
        f"{SYSTEM_PROMPT}\n\n"
        "User:\n"
        f"{input_text}\n\n"
    )
    if add_generation_prompt:
        prompt += "Assistant:\n"
    return prompt


def render_full_chat(processor: Any, input_text: str, target_json: Dict[str, Any]) -> str:
    return (
        "System:\n"
        f"{SYSTEM_PROMPT}\n\n"
        "User:\n"
        f"{input_text}\n\n"
        "Assistant:\n"
        f"{json.dumps(target_json, ensure_ascii=False)}"
    )


def tokenize_sft_example(example: Dict[str, Any], processor: Any, tokenizer: Any) -> Dict[str, Any]:
    prompt_text = render_chat_prompt(processor, example["input_text"], add_generation_prompt=True)
    full_text = render_full_chat(processor, example["input_text"], example["target_json"])

    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        add_special_tokens=False,
    )
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = list(input_ids)
    prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
    for index in range(prompt_len):
        labels[index] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def attach_lora(model: Any) -> Any:
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        # Gemma 4 wraps quantized projections in Gemma4ClippableLinear, so LoRA
        # must be attached to the nested Linear4bit child modules.
        target_modules=[
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "gate_proj.linear",
            "up_proj.linear",
            "down_proj.linear",
        ],
    )
    return get_peft_model(model, lora_config)


def generate_predictions(
    model: Any,
    processor: Any,
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prompt = render_chat_prompt(processor, record["input_text"], add_generation_prompt=True)
        encoded = encode_prompt(processor, prompt)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

        with torch.no_grad():
            generation = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=BENCHMARK_TEMPERATURE,
                top_p=BENCHMARK_TOP_P,
                pad_token_id=get_pad_token_id(processor),
                eos_token_id=get_eos_token_id(processor),
            )

        new_tokens = generation[0][encoded["input_ids"].shape[1] :]
        raw_text = decode_tokens(processor, new_tokens)
        parsed, parse_error = extract_json_object(raw_text)
        ok = False
        validation_errors: List[str] = []
        cleaned = None
        if parsed is not None:
            ok, validation_errors, cleaned = validate_target_json(parsed)

        if ok and cleaned is not None:
            predicted_verdict = cleaned["verdict"]
            probability_true = cleaned["probability_true"]
            explanation_text = " ".join([cleaned["reasoning"]] + cleaned["incongruences"]).strip()
            incongruence_count = len(cleaned["incongruences"])
            prediction_source = "verdict"
        else:
            if parsed and isinstance(parsed.get("probability_true"), (int, float)):
                probability_true = float(parsed["probability_true"])
                predicted_verdict = VERDICT_TRUE if probability_true >= 0.5 else VERDICT_NOT_TRUE
                prediction_source = "probability_fallback"
            else:
                probability_true = 0.5
                predicted_verdict = VERDICT_NOT_TRUE
                prediction_source = "invalid_default"
            explanation_text = raw_text
            incongruence_count = 0

        outputs.append(
            {
                "claim_id": record["claim_id"],
                "gold_binary_label": record["binary_label"],
                "gold_original_label": record["original_label"],
                "raw_output": raw_text,
                "parsed_output": cleaned if ok and cleaned is not None else parsed,
                "json_valid": parse_error is None,
                "schema_valid": ok,
                "validation_errors": validation_errors,
                "predicted_verdict": predicted_verdict,
                "probability_true": probability_true,
                "prediction_source": prediction_source,
                "incongruence_count": incongruence_count,
                "explanation_text": explanation_text,
            }
        )

        if index % 25 == 0 or index == len(records):
            print(f"Predicted {index}/{len(records)} claims")

    return outputs


def build_confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Dict[str, int]]:
    labels = [VERDICT_TRUE, VERDICT_NOT_TRUE]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        gold: {pred: int(matrix[i, j]) for j, pred in enumerate(labels)}
        for i, gold in enumerate(labels)
    }


def compute_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [record["gold_binary_label"] for record in records]
    y_pred = [record["predicted_verdict"] for record in records]
    prob_true = [float(record["probability_true"]) for record in records]
    gold_true = [1 if label == VERDICT_TRUE else 0 for label in y_true]
    gold_not_true = [1 if label == VERDICT_NOT_TRUE else 0 for label in y_true]
    prob_not_true = [1.0 - value for value in prob_true]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[VERDICT_TRUE, VERDICT_NOT_TRUE],
        zero_division=0,
    )

    def safe_auc(y_binary: Sequence[int], scores: Sequence[float]) -> Optional[float]:
        if len(set(y_binary)) < 2:
            return None
        return float(roc_auc_score(y_binary, scores))

    def safe_pr_auc(y_binary: Sequence[int], scores: Sequence[float]) -> Optional[float]:
        if len(set(y_binary)) < 2:
            return None
        return float(average_precision_score(y_binary, scores))

    suspicious_records = [record for record in records if record["gold_binary_label"] == VERDICT_NOT_TRUE]
    truthful_records = [record for record in records if record["gold_binary_label"] == VERDICT_TRUE]

    subtype_hits = 0
    subtype_total = 0
    for record in suspicious_records:
        label = record["gold_original_label"]
        if label == "genuine_accident":
            continue
        subtype_total += 1
        if subtype_theme_hit(record.get("explanation_text", ""), label):
            subtype_hits += 1

    per_class = {}
    for index, label in enumerate([VERDICT_TRUE, VERDICT_NOT_TRUE]):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
        }

    return {
        "num_examples": len(records),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(sum(precision) / len(precision)),
        "macro_recall": float(sum(recall) / len(recall)),
        "macro_f1": float(sum(f1) / len(f1)),
        "per_class": per_class,
        "confusion_matrix": build_confusion(y_true, y_pred),
        "roc_auc_true": safe_auc(gold_true, prob_true),
        "pr_auc_true": safe_pr_auc(gold_true, prob_true),
        "roc_auc_not_true": safe_auc(gold_not_true, prob_not_true),
        "pr_auc_not_true": safe_pr_auc(gold_not_true, prob_not_true),
        "json_validity_rate": float(sum(1 for row in records if row["json_valid"]) / len(records)),
        "schema_validity_rate": float(sum(1 for row in records if row["schema_valid"]) / len(records)),
        "valid_verdict_rate": float(sum(1 for row in records if row["prediction_source"] != "invalid_default") / len(records)),
        "valid_probability_rate": float(sum(1 for row in records if 0.0 <= float(row["probability_true"]) <= 1.0) / len(records)),
        "incongruence_presence_rate_on_not_true": float(
            sum(1 for row in suspicious_records if row["incongruence_count"] > 0) / max(1, len(suspicious_records))
        ),
        "empty_incongruence_rate_on_true": float(
            sum(1 for row in truthful_records if row["incongruence_count"] == 0) / max(1, len(truthful_records))
        ),
        "subtype_theme_hit_rate": float(subtype_hits / max(1, subtype_total)),
    }


def write_markdown_table(base_metrics: Dict[str, Any], finetuned_metrics: Dict[str, Any], output_path: Path) -> None:
    rows = [
        ("Accuracy", base_metrics["accuracy"], finetuned_metrics["accuracy"]),
        ("Macro F1", base_metrics["macro_f1"], finetuned_metrics["macro_f1"]),
        ("JSON validity", base_metrics["json_validity_rate"], finetuned_metrics["json_validity_rate"]),
        ("Schema validity", base_metrics["schema_validity_rate"], finetuned_metrics["schema_validity_rate"]),
        ("ROC-AUC true", base_metrics["roc_auc_true"], finetuned_metrics["roc_auc_true"]),
        ("PR-AUC not_true", base_metrics["pr_auc_not_true"], finetuned_metrics["pr_auc_not_true"]),
        (
            "Suspicious incongruence presence",
            base_metrics["incongruence_presence_rate_on_not_true"],
            finetuned_metrics["incongruence_presence_rate_on_not_true"],
        ),
        (
            "Truthful empty incongruences",
            base_metrics["empty_incongruence_rate_on_true"],
            finetuned_metrics["empty_incongruence_rate_on_true"],
        ),
    ]
    lines = [
        "| Metric | Base | Fine-tuned | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, base_value, ft_value in rows:
        if base_value is None or ft_value is None:
            base_text = "n/a" if base_value is None else f"{base_value:.4f}"
            ft_text = "n/a" if ft_value is None else f"{ft_value:.4f}"
            delta_text = "n/a"
        else:
            base_text = f"{base_value:.4f}"
            ft_text = f"{ft_value:.4f}"
            delta_text = f"{ft_value - base_value:+.4f}"
        lines.append(f"| {label} | {base_text} | {ft_text} | {delta_text} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class JsonValidityCallback(TrainerCallback):
    """Lightweight callback placeholder so the notebook can later be extended for JSON-validity tie-breaks."""

    pass


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

DATASET_ROOT = find_dataset_root()
MODEL_PATH = find_model_root()
MODEL_CONFIG_SUMMARY = load_model_config_summary(MODEL_PATH)
TRAIN_PATH = DATASET_ROOT / "claim_sft_train.jsonl"
VAL_PATH = DATASET_ROOT / "claim_sft_val.jsonl"
TEST_PATH = DATASET_ROOT / "claim_sft_test.jsonl"

print("=== Gemma 4 E4B claim-level benchmark + SFT ===")
print("Model path:", MODEL_PATH)
print("Model hint:", MODEL_HINT)
print("Model config summary:", MODEL_CONFIG_SUMMARY)
print("Dataset root:", DATASET_ROOT)
print("Train path:", TRAIN_PATH)
print("Validation path:", VAL_PATH)
print("Test path:", TEST_PATH)
print("Run base benchmark:", RUN_BASE_BENCHMARK)
print("Run training:", RUN_TRAIN)
print("Run finetuned benchmark:", RUN_FINETUNED_BENCHMARK)

test_records = load_jsonl(TEST_PATH)
print("Frozen test examples:", len(test_records))

base_metrics: Optional[Dict[str, Any]] = None
finetuned_metrics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Step 1: Base benchmark
# ---------------------------------------------------------------------------

if RUN_BASE_BENCHMARK:
    print("\n=== Step 1: Base benchmark ===")
    processor, base_model = load_processor_and_model(MODEL_PATH, load_for_training=False)
    base_predictions = generate_predictions(base_model, processor, test_records)
    save_jsonl(base_predictions, BASE_PREDICTIONS_PATH)
    base_metrics = compute_metrics(base_predictions)
    print("Base accuracy:", round(base_metrics["accuracy"], 4))
    print("Base macro F1:", round(base_metrics["macro_f1"], 4))
    print("Base JSON validity:", round(base_metrics["json_validity_rate"], 4))

    del base_model
    del processor
    torch.cuda.empty_cache()
elif BASE_PREDICTIONS_PATH.exists():
    print("\n=== Step 1: Base benchmark skipped; loading existing predictions ===")
    base_metrics = compute_metrics(load_jsonl(BASE_PREDICTIONS_PATH))
    print("Loaded base metrics from:", BASE_PREDICTIONS_PATH)


# ---------------------------------------------------------------------------
# Step 2: Fine-tuning
# ---------------------------------------------------------------------------

if RUN_TRAIN:
    print("\n=== Step 2: Fine-tuning ===")
    processor, train_model = load_processor_and_model(MODEL_PATH, load_for_training=True)
    train_model = attach_lora(train_model)
    train_model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=str(TRAIN_PATH), split="train")
    val_dataset = load_dataset("json", data_files=str(VAL_PATH), split="train")

    backend_tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(backend_tokenizer, "pad_token", None) is None and getattr(backend_tokenizer, "eos_token", None) is not None:
        backend_tokenizer.pad_token = backend_tokenizer.eos_token

    train_dataset = train_dataset.map(
        lambda row: tokenize_sft_example(row, processor, backend_tokenizer),
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda row: tokenize_sft_example(row, processor, backend_tokenizer),
        remove_columns=val_dataset.column_names,
    )

    train_model.config.use_cache = False
    if hasattr(train_model, "gradient_checkpointing_enable"):
        train_model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=str(ADAPTER_OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        bf16=False,
        fp16=True,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=train_model,
        tokenizer=backend_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=backend_tokenizer,
            model=train_model,
            padding=True,
            label_pad_token_id=-100,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1), JsonValidityCallback()],
    )

    trainer.train()
    trainer.model.save_pretrained(ADAPTER_OUTPUT_DIR)
    backend_tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
    print("Saved LoRA adapter to:", ADAPTER_OUTPUT_DIR)

    del trainer
    del train_model
    del processor
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 3: Fine-tuned benchmark
# ---------------------------------------------------------------------------

if RUN_FINETUNED_BENCHMARK:
    print("\n=== Step 3: Fine-tuned benchmark ===")
    processor, finetuned_base_model = load_processor_and_model(MODEL_PATH, load_for_training=False)
    finetuned_model = PeftModel.from_pretrained(finetuned_base_model, str(ADAPTER_OUTPUT_DIR))
    finetuned_model.eval()

    finetuned_predictions = generate_predictions(finetuned_model, processor, test_records)
    save_jsonl(finetuned_predictions, FINETUNED_PREDICTIONS_PATH)
    finetuned_metrics = compute_metrics(finetuned_predictions)
    print("Fine-tuned accuracy:", round(finetuned_metrics["accuracy"], 4))
    print("Fine-tuned macro F1:", round(finetuned_metrics["macro_f1"], 4))
    print("Fine-tuned JSON validity:", round(finetuned_metrics["json_validity_rate"], 4))

    del finetuned_model
    del finetuned_base_model
    del processor
    torch.cuda.empty_cache()
elif FINETUNED_PREDICTIONS_PATH.exists():
    print("\n=== Step 3: Fine-tuned benchmark skipped; loading existing predictions ===")
    finetuned_metrics = compute_metrics(load_jsonl(FINETUNED_PREDICTIONS_PATH))
    print("Loaded fine-tuned metrics from:", FINETUNED_PREDICTIONS_PATH)


# ---------------------------------------------------------------------------
# Step 4: Comparison artifacts
# ---------------------------------------------------------------------------

if base_metrics is not None and finetuned_metrics is not None:
    comparison = {
        "config": {
            "model_path": str(MODEL_PATH),
            "model_hint": MODEL_HINT,
            "model_config_summary": MODEL_CONFIG_SUMMARY,
            "train_path": str(TRAIN_PATH),
            "val_path": str(VAL_PATH),
            "test_path": str(TEST_PATH),
            "num_test_examples": len(test_records),
            "benchmark_temperature": BENCHMARK_TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "dtype": "float16",
        },
        "base": {
            "prediction_path": str(BASE_PREDICTIONS_PATH),
            "metrics": base_metrics,
        },
        "finetuned": {
            "prediction_path": str(FINETUNED_PREDICTIONS_PATH),
            "adapter_path": str(ADAPTER_OUTPUT_DIR),
            "metrics": finetuned_metrics,
        },
        "delta": {
            "accuracy": finetuned_metrics["accuracy"] - base_metrics["accuracy"],
            "macro_f1": finetuned_metrics["macro_f1"] - base_metrics["macro_f1"],
            "json_validity_rate": finetuned_metrics["json_validity_rate"] - base_metrics["json_validity_rate"],
            "schema_validity_rate": finetuned_metrics["schema_validity_rate"] - base_metrics["schema_validity_rate"],
        },
    }

    save_json(comparison, COMPARISON_JSON_PATH)
    write_markdown_table(base_metrics, finetuned_metrics, COMPARISON_MD_PATH)

    print("\n=== Finished ===")
    print("Base predictions:", BASE_PREDICTIONS_PATH)
    print("Fine-tuned predictions:", FINETUNED_PREDICTIONS_PATH)
    print("Comparison JSON:", COMPARISON_JSON_PATH)
    print("Comparison table:", COMPARISON_MD_PATH)
else:
    print("\n=== Finished partial stage run ===")
    if BASE_PREDICTIONS_PATH.exists():
        print("Base predictions:", BASE_PREDICTIONS_PATH)
    if FINETUNED_PREDICTIONS_PATH.exists():
        print("Fine-tuned predictions:", FINETUNED_PREDICTIONS_PATH)
    if ADAPTER_OUTPUT_DIR.exists():
        print("Adapter dir:", ADAPTER_OUTPUT_DIR)
