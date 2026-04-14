from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gemma4.common import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_BENCHMARK_OUTPUT_DIR,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEST_PATH,
    VERDICT_NOT_TRUE,
    VERDICT_TRUE,
    extract_json_object,
    render_chat_prompt,
    save_json,
    save_jsonl,
    subtype_theme_hit,
    validate_target_json,
)
from gemma4.model_io import load_model_and_tokenizer
from gemma4.model_io import decode_tokens, encode_text, get_eos_token_id, get_pad_token_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark base vs fine-tuned Gemma 4 on the frozen claim-level test split.")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--finetuned-model-path", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--finetuned-adapter-path", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def generate_predictions(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prompt = render_chat_prompt(tokenizer, record["input_text"])
        encoded = encode_text(tokenizer, prompt)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": get_pad_token_id(tokenizer),
            "eos_token_id": get_eos_token_id(tokenizer),
        }
        with torch.no_grad():
            output = model.generate(**encoded, **generation_kwargs)
        new_tokens = output[0][encoded["input_ids"].shape[1] :]
        raw_text = decode_tokens(tokenizer, new_tokens)
        parsed, parse_error = extract_json_object(raw_text)
        ok = False
        validation_errors: List[str] = []
        cleaned = None
        if parsed is not None:
            ok, validation_errors, cleaned = validate_target_json(parsed)

        if ok and cleaned is not None:
            verdict = cleaned["verdict"]
            probability_true = cleaned["probability_true"]
            explanation = " ".join([cleaned["reasoning"]] + cleaned["incongruences"]).strip()
            prediction_source = "verdict"
        else:
            if parsed and isinstance(parsed.get("probability_true"), (int, float)):
                probability_true = float(parsed["probability_true"])
                verdict = VERDICT_TRUE if probability_true >= 0.5 else VERDICT_NOT_TRUE
                prediction_source = "probability_fallback"
            else:
                probability_true = 0.5
                verdict = VERDICT_NOT_TRUE
                prediction_source = "invalid_default"
            explanation = raw_text

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
                "predicted_verdict": verdict,
                "probability_true": probability_true,
                "prediction_source": prediction_source,
                "incongruence_count": len(cleaned["incongruences"]) if ok and cleaned else 0,
                "explanation_text": explanation,
            }
        )
        if index % 25 == 0 or index == len(records):
            print(f"Predicted {index}/{len(records)} claims")
    return outputs


def build_confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Dict[str, int]]:
    labels = [VERDICT_TRUE, VERDICT_NOT_TRUE]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        gold: {
            pred: int(matrix[i, j])
            for j, pred in enumerate(labels)
        }
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

    explanation_texts = [" ".join(filter(None, [record.get("explanation_text", "")])) for record in records]
    suspicious_records = [record for record in records if record["gold_binary_label"] == VERDICT_NOT_TRUE]
    truthful_records = [record for record in records if record["gold_binary_label"] == VERDICT_TRUE]

    subtype_hits = 0
    subtype_total = 0
    for record in suspicious_records:
        original_label = record["gold_original_label"]
        if original_label == "genuine_accident":
            continue
        subtype_total += 1
        if subtype_theme_hit(record.get("explanation_text", ""), original_label):
            subtype_hits += 1

    per_class = {}
    for index, label in enumerate([VERDICT_TRUE, VERDICT_NOT_TRUE]):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
        }

    metrics = {
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
        "valid_probability_rate": float(
            sum(1 for row in records if 0.0 <= float(row["probability_true"]) <= 1.0) / len(records)
        ),
        "incongruence_presence_rate_on_not_true": float(
            sum(1 for row in suspicious_records if row["incongruence_count"] > 0) / max(1, len(suspicious_records))
        ),
        "empty_incongruence_rate_on_true": float(
            sum(1 for row in truthful_records if row["incongruence_count"] == 0) / max(1, len(truthful_records))
        ),
        "subtype_theme_hit_rate": float(subtype_hits / max(1, subtype_total)),
    }
    return metrics


def write_markdown_table(comparison: Dict[str, Any], output_path: Path) -> None:
    base_metrics = comparison["base"]["metrics"]
    finetuned_metrics = comparison["finetuned"]["metrics"]
    rows = [
        ("Accuracy", base_metrics["accuracy"], finetuned_metrics["accuracy"]),
        ("Macro F1", base_metrics["macro_f1"], finetuned_metrics["macro_f1"]),
        ("JSON validity", base_metrics["json_validity_rate"], finetuned_metrics["json_validity_rate"]),
        ("Schema validity", base_metrics["schema_validity_rate"], finetuned_metrics["schema_validity_rate"]),
        ("ROC-AUC (true)", base_metrics["roc_auc_true"], finetuned_metrics["roc_auc_true"]),
        ("PR-AUC (not_true)", base_metrics["pr_auc_not_true"], finetuned_metrics["pr_auc_not_true"]),
        (
            "Incongruence presence on suspicious",
            base_metrics["incongruence_presence_rate_on_not_true"],
            finetuned_metrics["incongruence_presence_rate_on_not_true"],
        ),
        (
            "Empty incongruences on truthful",
            base_metrics["empty_incongruence_rate_on_true"],
            finetuned_metrics["empty_incongruence_rate_on_true"],
        ),
    ]
    lines = [
        "| Metric | Base Gemma 4 | Fine-tuned Gemma 4 | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, base_value, finetuned_value in rows:
        if base_value is None or finetuned_value is None:
            delta = "n/a"
            base_text = "n/a" if base_value is None else f"{base_value:.4f}"
            finetuned_text = "n/a" if finetuned_value is None else f"{finetuned_value:.4f}"
        else:
            delta = f"{finetuned_value - base_value:+.4f}"
            base_text = f"{base_value:.4f}"
            finetuned_text = f"{finetuned_value:.4f}"
        lines.append(f"| {label} | {base_text} | {finetuned_text} | {delta} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qualitative_review(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    selected: Dict[str, List[Dict[str, Any]]] = {
        "genuine_accident": [],
        "soft_fraud_exaggeration": [],
        "hard_fraud_staged": [],
        "hard_fraud_phantom_vehicle": [],
    }
    for record in records:
        label = record["gold_original_label"]
        bucket = selected.get(label)
        if bucket is None or len(bucket) >= 3:
            continue
        bucket.append(record)

    lines = ["# Benchmark Qualitative Review", ""]
    for label, items in selected.items():
        lines.append(f"## {label}")
        lines.append("")
        if not items:
            lines.append("No examples selected.")
            lines.append("")
            continue
        for item in items:
            lines.extend(
                [
                    f"- `claim_id`: {item['claim_id']}",
                    f"- gold: {item['gold_binary_label']}",
                    f"- predicted: {item['predicted_verdict']}",
                    f"- probability_true: {item['probability_true']:.4f}",
                    f"- reasoning snippet: {item['raw_output'][:400]}",
                    "",
                ]
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def benchmark_one_model(
    *,
    name: str,
    model_path: str,
    adapter_path: Optional[str],
    records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    prediction_path: Path,
) -> Dict[str, Any]:
    print(f"\n=== Benchmarking {name} ===")
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        adapter_path=adapter_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    predictions = generate_predictions(
        model,
        tokenizer,
        records,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    save_jsonl(predictions, prediction_path)
    metrics = compute_metrics(predictions)
    return {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "prediction_path": str(prediction_path),
        "metrics": metrics,
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_records = load_jsonl(args.test_path)
    if args.limit is not None:
        test_records = test_records[: args.limit]

    base_output_path = args.output_dir / "benchmark_base_gemma4_test.jsonl"
    finetuned_output_path = args.output_dir / "benchmark_finetuned_gemma4_test.jsonl"

    base_result = benchmark_one_model(
        name="base",
        model_path=args.base_model_path,
        adapter_path=None,
        records=test_records,
        args=args,
        prediction_path=base_output_path,
    )

    finetuned_result = benchmark_one_model(
        name="finetuned",
        model_path=args.finetuned_model_path,
        adapter_path=args.finetuned_adapter_path,
        records=test_records,
        args=args,
        prediction_path=finetuned_output_path,
    )

    comparison = {
        "config": {
            "test_path": str(args.test_path),
            "num_examples": len(test_records),
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "load_in_4bit": args.load_in_4bit,
        },
        "base": {
            "model_path": base_result["model_path"],
            "adapter_path": base_result["adapter_path"],
            "prediction_path": base_result["prediction_path"],
            "metrics": base_result["metrics"],
        },
        "finetuned": {
            "model_path": finetuned_result["model_path"],
            "adapter_path": finetuned_result["adapter_path"],
            "prediction_path": finetuned_result["prediction_path"],
            "metrics": finetuned_result["metrics"],
        },
        "delta": {
            "accuracy": finetuned_result["metrics"]["accuracy"] - base_result["metrics"]["accuracy"],
            "macro_f1": finetuned_result["metrics"]["macro_f1"] - base_result["metrics"]["macro_f1"],
            "json_validity_rate": finetuned_result["metrics"]["json_validity_rate"] - base_result["metrics"]["json_validity_rate"],
            "schema_validity_rate": finetuned_result["metrics"]["schema_validity_rate"] - base_result["metrics"]["schema_validity_rate"],
        },
    }

    comparison_path = args.output_dir / "benchmark_comparison.json"
    save_json(comparison, comparison_path)
    write_markdown_table(comparison, args.output_dir / "benchmark_comparison.md")
    write_qualitative_review(finetuned_result["predictions"], args.output_dir / "benchmark_qualitative_samples.md")

    print(f"\nSaved base predictions to: {base_output_path}")
    print(f"Saved fine-tuned predictions to: {finetuned_output_path}")
    print(f"Saved benchmark comparison to: {comparison_path}")
    print(f"Base macro F1: {base_result['metrics']['macro_f1']:.4f}")
    print(f"Fine-tuned macro F1: {finetuned_result['metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
