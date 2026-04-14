import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABEL_ORDER = ["supports", "neutral", "contradicts"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation charts from evaluate_pairwise.py JSON output.")
    parser.add_argument("report_path", type=Path, help="Path to evaluation JSON report.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <report_dir>/evaluation_dashboard.png",
    )
    return parser.parse_args()


def load_report(report_path: Path) -> Dict[str, object]:
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_confusion_matrix(report: Dict[str, object]) -> np.ndarray:
    confusion = report["metrics"]["confusion_matrix"]
    matrix = np.array(
        [
            [int(confusion[true_label][predicted_label]) for predicted_label in LABEL_ORDER]
            for true_label in LABEL_ORDER
        ],
        dtype=int,
    )
    return matrix


def compute_outcomes(confusion_matrix: np.ndarray) -> Dict[str, object]:
    correct_per_class = np.diag(confusion_matrix)
    support_per_class = confusion_matrix.sum(axis=1)
    incorrect_per_class = support_per_class - correct_per_class
    total_correct = int(correct_per_class.sum())
    total_examples = int(support_per_class.sum())
    total_incorrect = total_examples - total_correct
    return {
        "correct_per_class": correct_per_class,
        "incorrect_per_class": incorrect_per_class,
        "total_correct": total_correct,
        "total_incorrect": total_incorrect,
        "total_examples": total_examples,
    }


def main() -> None:
    args = parse_args()
    report = load_report(args.report_path)
    metrics = report["metrics"]
    per_class = metrics["per_class"]
    confusion_matrix = build_confusion_matrix(report)
    outcomes = compute_outcomes(confusion_matrix)

    output_path = args.output_path or args.report_path.parent / "evaluation_dashboard.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct_counts = outcomes["correct_per_class"]
    incorrect_counts = outcomes["incorrect_per_class"]
    total_correct = outcomes["total_correct"]
    total_incorrect = outcomes["total_incorrect"]
    total_examples = outcomes["total_examples"]

    precision_values = [float(per_class[label]["precision"]) for label in LABEL_ORDER]
    recall_values = [float(per_class[label]["recall"]) for label in LABEL_ORDER]
    f1_values = [float(per_class[label]["f1"]) for label in LABEL_ORDER]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    label_positions = np.arange(len(LABEL_ORDER))

    axes[0, 0].bar(
        ["Correct", "Incorrect"],
        [total_correct, total_incorrect],
        color=["#2ca02c", "#d62728"],
        width=0.6,
    )
    axes[0, 0].set_title("Overall Correct vs Incorrect")
    axes[0, 0].set_ylabel("Examples")
    axes[0, 0].set_ylim(0, max(total_examples, total_correct) * 1.1)
    axes[0, 0].text(0, total_correct + total_examples * 0.02, f"{total_correct} ({total_correct / total_examples:.1%})", ha="center")
    axes[0, 0].text(1, total_incorrect + total_examples * 0.02, f"{total_incorrect} ({total_incorrect / total_examples:.1%})", ha="center")

    axes[0, 1].bar(label_positions, correct_counts, color="#2ca02c", label="Correct")
    axes[0, 1].bar(label_positions, incorrect_counts, bottom=correct_counts, color="#d62728", label="Incorrect")
    axes[0, 1].set_xticks(label_positions, LABEL_ORDER)
    axes[0, 1].set_title("Per-Class Outcomes")
    axes[0, 1].set_ylabel("Examples")
    axes[0, 1].legend()
    for index, (correct_value, incorrect_value) in enumerate(zip(correct_counts, incorrect_counts)):
        axes[0, 1].text(index, correct_value / 2, str(int(correct_value)), ha="center", va="center", color="white", fontsize=9)
        if incorrect_value > 0:
            axes[0, 1].text(index, correct_value + incorrect_value / 2, str(int(incorrect_value)), ha="center", va="center", color="white", fontsize=9)

    width = 0.24
    axes[1, 0].bar(label_positions - width, precision_values, width=width, color="#1f77b4", label="Precision")
    axes[1, 0].bar(label_positions, recall_values, width=width, color="#ff7f0e", label="Recall")
    axes[1, 0].bar(label_positions + width, f1_values, width=width, color="#2ca02c", label="F1")
    axes[1, 0].set_xticks(label_positions, LABEL_ORDER)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Per-Class Metrics")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()

    image = axes[1, 1].imshow(confusion_matrix, cmap="Blues")
    axes[1, 1].set_xticks(label_positions, LABEL_ORDER, rotation=15)
    axes[1, 1].set_yticks(label_positions, LABEL_ORDER)
    axes[1, 1].set_title("Confusion Matrix")
    axes[1, 1].set_xlabel("Predicted label")
    axes[1, 1].set_ylabel("True label")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    threshold = confusion_matrix.max() / 2 if confusion_matrix.size else 0
    for row_index in range(confusion_matrix.shape[0]):
        for column_index in range(confusion_matrix.shape[1]):
            value = confusion_matrix[row_index, column_index]
            color = "white" if value > threshold else "black"
            axes[1, 1].text(column_index, row_index, str(value), ha="center", va="center", color=color, fontsize=10)

    figure.suptitle(
        f"Pairwise Validation Dashboard | Accuracy={metrics['accuracy']:.4f} | Macro F1={metrics['macro_f1']:.4f}",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved dashboard to: {output_path}")


if __name__ == "__main__":
    main()
