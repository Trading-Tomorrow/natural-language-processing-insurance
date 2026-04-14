import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from .dataset import PairwiseClaimsDataset
    from .model import PairwiseTransformerClassifier, PairwiseTransformerConfig
    from .pairwise_utils import ID_TO_INCONSISTENCY, ID_TO_LABEL, inconsistency_to_id, label_to_id
except ImportError:
    from dataset import PairwiseClaimsDataset
    from model import PairwiseTransformerClassifier, PairwiseTransformerConfig
    from pairwise_utils import ID_TO_INCONSISTENCY, ID_TO_LABEL, inconsistency_to_id, label_to_id


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "pairwise_baseline" / "best_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained pairwise contradiction classifier.")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--split", choices=("validation", "train", "full"), default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def choose_device(device_override: Optional[str]) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_stratified_splits(
    records: Sequence[Dict[str, object]],
    validation_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    label_buckets: Dict[int, List[int]] = {label_id: [] for label_id in ID_TO_LABEL.keys()}
    for index, record in enumerate(records):
        label_buckets[label_to_id(str(record["label"]))].append(index)

    train_indices: List[int] = []
    validation_indices: List[int] = []
    for label_id, indices in label_buckets.items():
        if len(indices) < 2:
            raise ValueError(
                f"Need at least two examples for label '{ID_TO_LABEL[label_id]}' to recreate the split."
            )

        shuffled = list(indices)
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(len(shuffled) * validation_ratio)))
        validation_count = min(validation_count, len(shuffled) - 1)
        validation_indices.extend(shuffled[:validation_count])
        train_indices.extend(shuffled[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def label_distribution(indices: Iterable[int], records: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts = {label_name: 0 for label_name in ID_TO_LABEL.values()}
    for index in indices:
        label_name = str(records[index]["label"])
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def inconsistency_distribution(indices: Iterable[int], records: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts = {label_name: 0 for label_name in ID_TO_INCONSISTENCY.values()}
    for index in indices:
        label_name = str(records[index].get("inconsistency_type", "none"))
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_per_class_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    id_to_label_map: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for label_id in sorted(id_to_label_map.keys()):
        label_name = id_to_label_map[label_id]
        true_positive = ((predictions == label_id) & (labels == label_id)).sum().item()
        false_positive = ((predictions == label_id) & (labels != label_id)).sum().item()
        false_negative = ((predictions != label_id) & (labels == label_id)).sum().item()
        support = (labels == label_id).sum().item()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        metrics[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }

    return metrics


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    id_to_label_map: Dict[int, str],
) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for true_label_id in sorted(id_to_label_map.keys()):
        true_label_name = id_to_label_map[true_label_id]
        row: Dict[str, int] = {}
        for predicted_label_id in sorted(id_to_label_map.keys()):
            predicted_label_name = id_to_label_map[predicted_label_id]
            count = ((labels == true_label_id) & (predictions == predicted_label_id)).sum().item()
            row[predicted_label_name] = int(count)
        matrix[true_label_name] = row
    return matrix


def evaluate_model(
    model: PairwiseTransformerClassifier,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    inconsistency_class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_relation_loss = 0.0
    total_inconsistency_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_inconsistency_logits: List[torch.Tensor] = []
    all_inconsistency_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                **batch,
                class_weights=class_weights,
                inconsistency_class_weights=inconsistency_class_weights,
            )
            logits = outputs["logits"]
            loss = outputs["loss"]
            relation_loss = outputs["relation_loss"]
            inconsistency_logits = outputs["inconsistency_logits"]
            inconsistency_loss = outputs["inconsistency_loss"]

            if logits is None or loss is None:
                raise RuntimeError("Expected logits and loss during evaluation.")

            total_loss += loss.item() * batch["labels"].size(0)
            if relation_loss is not None:
                total_relation_loss += relation_loss.item() * batch["labels"].size(0)
            if inconsistency_loss is not None:
                total_inconsistency_loss += inconsistency_loss.item() * batch["labels"].size(0)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())
            if inconsistency_logits is not None:
                all_inconsistency_logits.append(inconsistency_logits.cpu())
                all_inconsistency_labels.append(batch["inconsistency_labels"].cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    per_class_metrics = compute_per_class_metrics(predictions, labels, ID_TO_LABEL)
    macro_precision = sum(metric["precision"] for metric in per_class_metrics.values()) / len(per_class_metrics)
    macro_recall = sum(metric["recall"] for metric in per_class_metrics.values()) / len(per_class_metrics)
    macro_f1 = sum(metric["f1"] for metric in per_class_metrics.values()) / len(per_class_metrics)
    metrics: Dict[str, object] = {
        "loss": total_loss / len(dataloader.dataset),
        "relation_loss": total_relation_loss / len(dataloader.dataset),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": compute_confusion_matrix(predictions, labels, ID_TO_LABEL),
        "num_examples": len(dataloader.dataset),
    }
    if all_inconsistency_logits:
        inconsistency_logits = torch.cat(all_inconsistency_logits, dim=0)
        inconsistency_labels = torch.cat(all_inconsistency_labels, dim=0)
        inconsistency_predictions = inconsistency_logits.argmax(dim=-1)
        inconsistency_accuracy = (inconsistency_predictions == inconsistency_labels).float().mean().item()
        inconsistency_per_class = compute_per_class_metrics(
            inconsistency_predictions,
            inconsistency_labels,
            ID_TO_INCONSISTENCY,
        )
        inconsistency_macro_precision = sum(
            metric["precision"] for metric in inconsistency_per_class.values()
        ) / len(inconsistency_per_class)
        inconsistency_macro_recall = sum(
            metric["recall"] for metric in inconsistency_per_class.values()
        ) / len(inconsistency_per_class)
        inconsistency_macro_f1 = sum(
            metric["f1"] for metric in inconsistency_per_class.values()
        ) / len(inconsistency_per_class)
        metrics["inconsistency_loss"] = total_inconsistency_loss / len(dataloader.dataset)
        metrics["inconsistency_accuracy"] = inconsistency_accuracy
        metrics["inconsistency_macro_precision"] = inconsistency_macro_precision
        metrics["inconsistency_macro_recall"] = inconsistency_macro_recall
        metrics["inconsistency_macro_f1"] = inconsistency_macro_f1
        metrics["inconsistency_per_class"] = inconsistency_per_class
        metrics["inconsistency_confusion_matrix"] = compute_confusion_matrix(
            inconsistency_predictions,
            inconsistency_labels,
            ID_TO_INCONSISTENCY,
        )
    return metrics


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def save_json(payload: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    checkpoint_train_config = checkpoint.get("train_config", {})
    model_config_dict = checkpoint["model_config"]
    model_config = PairwiseTransformerConfig(**model_config_dict)
    checkpoint_class_weights = checkpoint.get("class_weights")
    checkpoint_inconsistency_class_weights = checkpoint.get("inconsistency_class_weights")

    dataset_path = Path(args.dataset_path or checkpoint_train_config.get("dataset_path", BASE_DIR / "data" / "pairwise_dataset.jsonl"))
    tokenizer_path = Path(args.tokenizer_path or checkpoint_train_config.get("tokenizer_path", BASE_DIR / "tokenizers" / "claims_bpe"))
    validation_ratio = float(checkpoint_train_config.get("validation_ratio", 0.2))
    seed = int(checkpoint_train_config.get("seed", 42))

    dataset = PairwiseClaimsDataset(
        data_path=dataset_path,
        tokenizer_path=tokenizer_path,
        max_length=model_config.max_position_embeddings,
    )

    split_name = args.split
    if split_name == "full":
        indices = list(range(len(dataset)))
    else:
        train_indices, validation_indices = build_stratified_splits(dataset.records, validation_ratio=validation_ratio, seed=seed)
        indices = validation_indices if split_name == "validation" else train_indices

    evaluation_dataset = Subset(dataset, indices)
    dataloader = build_dataloader(evaluation_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = PairwiseTransformerClassifier(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    class_weights = None
    if checkpoint_class_weights is not None:
        class_weights = torch.tensor(checkpoint_class_weights, dtype=torch.float32, device=device)
    inconsistency_class_weights = None
    if checkpoint_inconsistency_class_weights is not None:
        inconsistency_class_weights = torch.tensor(
            checkpoint_inconsistency_class_weights,
            dtype=torch.float32,
            device=device,
        )

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        class_weights=class_weights,
        inconsistency_class_weights=inconsistency_class_weights,
    )
    distribution = label_distribution(indices, dataset.records)
    inconsistency_label_dist = inconsistency_distribution(indices, dataset.records)

    output_json = args.output_json
    if output_json is None:
        output_json = args.checkpoint_path.parent / f"evaluation_{split_name}.json"

    payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "dataset_path": str(dataset_path),
        "tokenizer_path": str(tokenizer_path),
        "device": str(device),
        "split": split_name,
        "label_distribution": distribution,
        "inconsistency_distribution": inconsistency_label_dist,
        "class_weights": None if class_weights is None else [float(weight) for weight in class_weights.cpu().tolist()],
        "inconsistency_class_weights": (
            None
            if inconsistency_class_weights is None
            else [float(weight) for weight in inconsistency_class_weights.cpu().tolist()]
        ),
        "metrics": metrics,
    }
    save_json(payload, output_json)

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Split: {split_name}")
    print(f"Examples: {metrics['num_examples']}")
    print(f"Label distribution: {distribution}")
    print(f"Inconsistency distribution: {inconsistency_label_dist}")
    if class_weights is not None:
        readable_weights = {
            ID_TO_LABEL[label_id]: round(float(class_weights[label_id].item()), 4)
            for label_id in sorted(ID_TO_LABEL.keys())
        }
        print(f"Class weights: {readable_weights}")
    if inconsistency_class_weights is not None:
        readable_inconsistency_weights = {
            ID_TO_INCONSISTENCY[label_id]: round(float(inconsistency_class_weights[label_id].item()), 4)
            for label_id in sorted(ID_TO_INCONSISTENCY.keys())
        }
        print(f"Inconsistency class weights: {readable_inconsistency_weights}")
    print(
        f"Loss={metrics['loss']:.4f} | "
        f"Accuracy={metrics['accuracy']:.4f} | "
        f"Macro Precision={metrics['macro_precision']:.4f} | "
        f"Macro Recall={metrics['macro_recall']:.4f} | "
        f"Macro F1={metrics['macro_f1']:.4f}"
    )
    if "inconsistency_macro_f1" in metrics:
        print(
            f"Inconsistency Loss={metrics['inconsistency_loss']:.4f} | "
            f"Inconsistency Accuracy={metrics['inconsistency_accuracy']:.4f} | "
            f"Inconsistency Macro F1={metrics['inconsistency_macro_f1']:.4f}"
        )
    print(f"Saved evaluation report to: {output_json}")


if __name__ == "__main__":
    main()
