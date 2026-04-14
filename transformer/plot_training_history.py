import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from train_pairwise.py history.")
    parser.add_argument("history_path", type=Path, help="Path to training_history.json")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <history_dir>/training_curves.png",
    )
    return parser.parse_args()


def load_history(history_path: Path) -> Dict[str, object]:
    return json.loads(history_path.read_text(encoding="utf-8"))


def extract_series(history: List[Dict[str, object]], split: str, metric: str) -> List[float]:
    return [float(epoch_record[split][metric]) for epoch_record in history]


def main() -> None:
    args = parse_args()
    payload = load_history(args.history_path)
    history = payload["history"]

    epochs = [int(epoch_record["epoch"]) for epoch_record in history]
    train_loss = extract_series(history, "train", "loss")
    validation_loss = extract_series(history, "validation", "loss")
    train_accuracy = extract_series(history, "train", "accuracy")
    validation_accuracy = extract_series(history, "validation", "accuracy")
    train_macro_f1 = extract_series(history, "train", "macro_f1")
    validation_macro_f1 = extract_series(history, "validation", "macro_f1")

    best_epoch = int(payload["best_epoch"])
    best_macro_f1 = float(payload["best_validation_macro_f1"])
    output_path = args.output_path or args.history_path.parent / "training_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, validation_loss, label="Validation Loss", color="#d62728", linewidth=2)
    axes[0].axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Best Epoch = {best_epoch}")
    axes[0].set_title("Pairwise Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, label="Train Accuracy", color="#9467bd", linewidth=2)
    axes[1].plot(epochs, validation_accuracy, label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    axes[1].plot(epochs, train_macro_f1, label="Train Macro F1", color="#17becf", linewidth=2)
    axes[1].plot(epochs, validation_macro_f1, label="Validation Macro F1", color="#2ca02c", linewidth=2)
    axes[1].axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5)
    axes[1].annotate(
        f"Best val Macro F1 = {best_macro_f1:.4f}\nEpoch = {best_epoch}",
        xy=(best_epoch, best_macro_f1),
        xytext=(best_epoch + 1, min(0.98, best_macro_f1 + 0.08)),
        arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
        fontsize=9,
    )
    axes[1].set_title("Pairwise Training Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    figure.suptitle("Pairwise Transformer Baseline Training Curves", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
