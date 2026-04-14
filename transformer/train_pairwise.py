import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from .dataset import DEFAULT_TOKENIZER_PATH, PairwiseClaimsDataset
    from .model import PairwiseTransformerClassifier, PairwiseTransformerConfig
    from .pairwise_utils import ID_TO_INCONSISTENCY, ID_TO_LABEL, inconsistency_to_id, label_to_id
except ImportError:
    from dataset import DEFAULT_TOKENIZER_PATH, PairwiseClaimsDataset
    from model import PairwiseTransformerClassifier, PairwiseTransformerConfig
    from pairwise_utils import ID_TO_INCONSISTENCY, ID_TO_LABEL, inconsistency_to_id, label_to_id


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "pairwise_dataset.jsonl"
DEFAULT_FULL_DATASET_PATH = BASE_DIR / "data" / "pairwise_dataset_full.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "checkpoints" / "pairwise_baseline"


@dataclass
class TrainConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH
    max_length: int = 256
    batch_size: int = 16
    num_epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    validation_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0
    pooling: str = "cls"
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_segment_embeddings: bool = True
    use_pairwise_comparison_head: bool = True
    use_inconsistency_head: bool = True
    inconsistency_loss_weight: float = 0.5
    device: Optional[str] = None
    class_weighting: str = "none"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the pairwise contradiction classifier.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--disable-segment-embeddings", action="store_true")
    parser.add_argument("--disable-pairwise-comparison-head", action="store_true")
    parser.add_argument("--disable-inconsistency-head", action="store_true")
    parser.add_argument("--inconsistency-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced"),
        default="none",
        help="Apply class weights to cross entropy. 'balanced' uses N / (C * n_c) on the train split.",
    )
    args = parser.parse_args()

    return TrainConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pooling=args.pooling,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        use_segment_embeddings=not args.disable_segment_embeddings,
        use_pairwise_comparison_head=not args.disable_pairwise_comparison_head,
        use_inconsistency_head=not args.disable_inconsistency_head,
        inconsistency_loss_weight=args.inconsistency_loss_weight,
        device=args.device,
        class_weighting=args.class_weighting,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_override: Optional[str]) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def label_distribution(indices: Iterable[int], dataset: PairwiseClaimsDataset) -> Dict[str, int]:
    counts = {label_name: 0 for label_name in ID_TO_LABEL.values()}
    for index in indices:
        label_name = dataset.records[index]["label"]
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def inconsistency_distribution(indices: Iterable[int], dataset: PairwiseClaimsDataset) -> Dict[str, int]:
    counts = {label_name: 0 for label_name in ID_TO_INCONSISTENCY.values()}
    for index in indices:
        label_name = dataset.records[index].get("inconsistency_type", "none")
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def build_stratified_splits(
    dataset: PairwiseClaimsDataset,
    validation_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    label_buckets: Dict[int, List[int]] = {label_id: [] for label_id in ID_TO_LABEL.keys()}
    for index, record in enumerate(dataset.records):
        label_buckets[label_to_id(record["label"])].append(index)

    train_indices: List[int] = []
    validation_indices: List[int] = []

    for label_id, indices in label_buckets.items():
        if len(indices) < 2:
            raise ValueError(
                f"Need at least two examples for label '{ID_TO_LABEL[label_id]}' to create a stratified split."
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


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    id_to_label_map: Dict[int, str],
) -> Dict[str, float]:
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label_id in sorted(id_to_label_map.keys()):
        true_positive = ((predictions == label_id) & (labels == label_id)).sum().item()
        false_positive = ((predictions == label_id) & (labels != label_id)).sum().item()
        false_negative = ((predictions != label_id) & (labels == label_id)).sum().item()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    class_count = len(id_to_label_map)
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision / class_count,
        "macro_recall": macro_recall / class_count,
        "macro_f1": macro_f1 / class_count,
    }


def compute_class_weights(
    indices: Sequence[int],
    dataset: PairwiseClaimsDataset,
    weighting_mode: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    return compute_weights_for_field(
        indices=indices,
        dataset=dataset,
        field_name="label",
        weighting_mode=weighting_mode,
        device=device,
        encoder=label_to_id,
        id_to_label_map=ID_TO_LABEL,
    )


def compute_weights_for_field(
    indices: Sequence[int],
    dataset: PairwiseClaimsDataset,
    field_name: str,
    weighting_mode: str,
    device: torch.device,
    encoder,
    id_to_label_map: Dict[int, str],
) -> Optional[torch.Tensor]:
    if weighting_mode == "none":
        return None

    label_counts = {label_id: 0 for label_id in id_to_label_map.keys()}
    for index in indices:
        label_id = encoder(dataset.records[index].get(field_name, "none"))
        label_counts[label_id] += 1

    total_examples = sum(label_counts.values())
    num_classes = len(label_counts)
    weights = []
    for label_id in sorted(id_to_label_map.keys()):
        class_count = label_counts[label_id]
        if class_count == 0:
            raise ValueError(f"Cannot compute class weight for empty class '{id_to_label_map[label_id]}'.")
        weights.append(total_examples / (num_classes * class_count))

    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: PairwiseTransformerClassifier,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
    inconsistency_class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_relation_loss = 0.0
    total_inconsistency_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_inconsistency_logits: List[torch.Tensor] = []
    all_inconsistency_labels: List[torch.Tensor] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(
                **batch,
                class_weights=class_weights,
                inconsistency_class_weights=inconsistency_class_weights,
            )
            loss = outputs["loss"]
            logits = outputs["logits"]
            relation_loss = outputs["relation_loss"]
            inconsistency_logits = outputs["inconsistency_logits"]
            inconsistency_loss = outputs["inconsistency_loss"]

            if loss is None or logits is None:
                raise RuntimeError("Model outputs must contain loss and logits during training/evaluation.")

            if is_training:
                loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

        total_loss += loss.item() * batch["labels"].size(0)
        if relation_loss is not None:
            total_relation_loss += relation_loss.item() * batch["labels"].size(0)
        if inconsistency_loss is not None:
            total_inconsistency_loss += inconsistency_loss.item() * batch["labels"].size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
        if inconsistency_logits is not None:
            all_inconsistency_logits.append(inconsistency_logits.detach().cpu())
            all_inconsistency_labels.append(batch["inconsistency_labels"].detach().cpu())

    concatenated_logits = torch.cat(all_logits, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(concatenated_logits, concatenated_labels, ID_TO_LABEL)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    if total_relation_loss > 0:
        metrics["relation_loss"] = total_relation_loss / len(dataloader.dataset)
    if all_inconsistency_logits:
        inconsistency_logits = torch.cat(all_inconsistency_logits, dim=0)
        inconsistency_labels = torch.cat(all_inconsistency_labels, dim=0)
        inconsistency_metrics = compute_metrics(
            inconsistency_logits,
            inconsistency_labels,
            ID_TO_INCONSISTENCY,
        )
        metrics["inconsistency_accuracy"] = inconsistency_metrics["accuracy"]
        metrics["inconsistency_macro_precision"] = inconsistency_metrics["macro_precision"]
        metrics["inconsistency_macro_recall"] = inconsistency_metrics["macro_recall"]
        metrics["inconsistency_macro_f1"] = inconsistency_metrics["macro_f1"]
        metrics["inconsistency_loss"] = total_inconsistency_loss / len(dataloader.dataset)
    return metrics


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(
    model: PairwiseTransformerClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: Dict[str, float],
    validation_metrics: Dict[str, float],
    train_config: TrainConfig,
    model_config: PairwiseTransformerConfig,
    output_dir: Path,
    class_weights: Optional[torch.Tensor],
    inconsistency_class_weights: Optional[torch.Tensor],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "train_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(train_config).items()
            },
            "model_config": asdict(model_config),
            "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
            "inconsistency_class_weights": (
                None
                if inconsistency_class_weights is None
                else inconsistency_class_weights.detach().cpu().tolist()
            ),
            "label_mapping": {label_name: label_id for label_name, label_id in sorted(
                ((label_name, label_id) for label_id, label_name in ID_TO_LABEL.items()),
                key=lambda item: item[1],
            )},
            "inconsistency_mapping": {
                label_name: label_id for label_name, label_id in sorted(
                    ((label_name, label_id) for label_id, label_name in ID_TO_INCONSISTENCY.items()),
                    key=lambda item: item[1],
                )
            },
        },
        checkpoint_path,
    )


def build_model_config(
    train_config: TrainConfig,
    vocab_size: int,
    pad_token_id: int,
    cls_token_id: int,
    sep_token_id: int,
) -> PairwiseTransformerConfig:
    return PairwiseTransformerConfig(
        vocab_size=vocab_size,
        max_position_embeddings=train_config.max_length,
        hidden_size=train_config.hidden_size,
        num_hidden_layers=train_config.num_hidden_layers,
        num_attention_heads=train_config.num_attention_heads,
        intermediate_size=train_config.intermediate_size,
        dropout=train_config.dropout,
        attention_dropout=train_config.attention_dropout,
        pad_token_id=pad_token_id,
        cls_token_id=cls_token_id,
        sep_token_id=sep_token_id,
        pooling=train_config.pooling,
        use_segment_embeddings=train_config.use_segment_embeddings,
        use_pairwise_comparison_head=train_config.use_pairwise_comparison_head,
        use_inconsistency_head=train_config.use_inconsistency_head,
        inconsistency_loss_weight=train_config.inconsistency_loss_weight,
    )


def print_epoch_summary(epoch: int, num_epochs: int, train_metrics: Dict[str, float], validation_metrics: Dict[str, float]) -> None:
    train_inconsistency = ""
    validation_inconsistency = ""
    if "inconsistency_macro_f1" in train_metrics:
        train_inconsistency = f" train_inconsistency_f1={train_metrics['inconsistency_macro_f1']:.4f}"
    if "inconsistency_macro_f1" in validation_metrics:
        validation_inconsistency = f" val_inconsistency_f1={validation_metrics['inconsistency_macro_f1']:.4f}"
    print(
        f"Epoch {epoch:02d}/{num_epochs:02d} | "
        f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} train_macro_f1={train_metrics['macro_f1']:.4f}{train_inconsistency} | "
        f"val_loss={validation_metrics['loss']:.4f} val_acc={validation_metrics['accuracy']:.4f} val_macro_f1={validation_metrics['macro_f1']:.4f}{validation_inconsistency}"
    )


def main() -> None:
    train_config = parse_args()
    seed_everything(train_config.seed)
    device = choose_device(train_config.device)

    dataset = PairwiseClaimsDataset(
        data_path=train_config.dataset_path,
        tokenizer_path=train_config.tokenizer_path,
        max_length=train_config.max_length,
    )
    train_indices, validation_indices = build_stratified_splits(
        dataset,
        validation_ratio=train_config.validation_ratio,
        seed=train_config.seed,
    )

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)
    train_loader = build_dataloader(train_dataset, train_config.batch_size, shuffle=True, num_workers=train_config.num_workers)
    validation_loader = build_dataloader(
        validation_dataset,
        train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    model_config = build_model_config(
        train_config=train_config,
        vocab_size=len(dataset.tokenizer),
        pad_token_id=dataset.pad_token_id,
        cls_token_id=dataset.cls_token_id,
        sep_token_id=dataset.sep_token_id,
    )
    model = PairwiseTransformerClassifier(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    class_weights = compute_class_weights(
        indices=train_indices,
        dataset=dataset,
        weighting_mode=train_config.class_weighting,
        device=device,
    )
    inconsistency_class_weights = compute_weights_for_field(
        indices=train_indices,
        dataset=dataset,
        field_name="inconsistency_type",
        weighting_mode=train_config.class_weighting,
        device=device,
        encoder=inconsistency_to_id,
        id_to_label_map=ID_TO_INCONSISTENCY,
    )

    train_split_distribution = label_distribution(train_indices, dataset)
    validation_split_distribution = label_distribution(validation_indices, dataset)
    train_inconsistency_distribution = inconsistency_distribution(train_indices, dataset)
    validation_inconsistency_distribution = inconsistency_distribution(validation_indices, dataset)
    print(f"Dataset path: {train_config.dataset_path}")
    print(f"Output dir: {train_config.output_dir}")
    print(f"Device: {device}")
    print(f"Tokenizer path: {train_config.tokenizer_path}")
    print(f"Total examples: {len(dataset)}")
    print(f"Train examples: {len(train_dataset)} | label distribution: {train_split_distribution}")
    print(f"Validation examples: {len(validation_dataset)} | label distribution: {validation_split_distribution}")
    print(f"Train inconsistency distribution: {train_inconsistency_distribution}")
    print(f"Validation inconsistency distribution: {validation_inconsistency_distribution}")
    print(f"Class weighting: {train_config.class_weighting}")
    print(f"Pairwise comparison head: {train_config.use_pairwise_comparison_head}")
    print(f"Inconsistency head: {train_config.use_inconsistency_head}")
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
    print(f"Trainable parameters: {count_parameters(model):,}")

    history: List[Dict[str, object]] = []
    best_validation_macro_f1 = -math.inf
    best_epoch = -1

    for epoch in range(1, train_config.num_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=train_config.grad_clip_norm,
            class_weights=class_weights,
            inconsistency_class_weights=inconsistency_class_weights,
        )
        validation_metrics = run_epoch(
            model=model,
            dataloader=validation_loader,
            device=device,
            optimizer=None,
            class_weights=class_weights,
            inconsistency_class_weights=inconsistency_class_weights,
        )

        print_epoch_summary(epoch, train_config.num_epochs, train_metrics, validation_metrics)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

        if validation_metrics["macro_f1"] > best_validation_macro_f1:
            best_validation_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                train_config=train_config,
                model_config=model_config,
                output_dir=train_config.output_dir,
                class_weights=class_weights,
                inconsistency_class_weights=inconsistency_class_weights,
            )

    save_json(
        {
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_validation_macro_f1,
            "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
            "inconsistency_class_weights": (
                None
                if inconsistency_class_weights is None
                else inconsistency_class_weights.detach().cpu().tolist()
            ),
            "history": history,
            "train_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(train_config).items()
            },
            "model_config": asdict(model_config),
        },
        train_config.output_dir / "training_history.json",
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_validation_macro_f1:.4f}")
    print(f"Best checkpoint: {train_config.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
