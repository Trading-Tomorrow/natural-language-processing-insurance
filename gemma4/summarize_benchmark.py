from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gemma4.common import DEFAULT_BENCHMARK_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a concise report-ready benchmark summary from benchmark_comparison.json.")
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=DEFAULT_BENCHMARK_OUTPUT_DIR / "benchmark_comparison.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_BENCHMARK_OUTPUT_DIR / "benchmark_report_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = json.loads(args.comparison_path.read_text(encoding="utf-8"))
    base = comparison["base"]["metrics"]
    finetuned = comparison["finetuned"]["metrics"]

    lines = [
        "# Gemma 4 Before/After Benchmark",
        "",
        f"- Test examples: `{comparison['config']['num_examples']}`",
        f"- Base macro F1: `{base['macro_f1']:.4f}`",
        f"- Fine-tuned macro F1: `{finetuned['macro_f1']:.4f}`",
        f"- Delta macro F1: `{comparison['delta']['macro_f1']:+.4f}`",
        f"- Base JSON validity: `{base['json_validity_rate']:.4f}`",
        f"- Fine-tuned JSON validity: `{finetuned['json_validity_rate']:.4f}`",
        f"- Base accuracy: `{base['accuracy']:.4f}`",
        f"- Fine-tuned accuracy: `{finetuned['accuracy']:.4f}`",
        "",
        "## Structured Output",
        "",
        f"- Base schema validity: `{base['schema_validity_rate']:.4f}`",
        f"- Fine-tuned schema validity: `{finetuned['schema_validity_rate']:.4f}`",
        f"- Base valid probability rate: `{base['valid_probability_rate']:.4f}`",
        f"- Fine-tuned valid probability rate: `{finetuned['valid_probability_rate']:.4f}`",
        "",
        "## Explanation Behavior",
        "",
        f"- Base suspicious incongruence presence: `{base['incongruence_presence_rate_on_not_true']:.4f}`",
        f"- Fine-tuned suspicious incongruence presence: `{finetuned['incongruence_presence_rate_on_not_true']:.4f}`",
        f"- Base truthful empty-incongruence rate: `{base['empty_incongruence_rate_on_true']:.4f}`",
        f"- Fine-tuned truthful empty-incongruence rate: `{finetuned['empty_incongruence_rate_on_true']:.4f}`",
        "",
        "## Conclusion",
        "",
        "The fine-tuned model should be considered better only if it improves binary predictive quality without harming JSON reliability.",
        "",
    ]
    args.output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary to: {args.output_path}")


if __name__ == "__main__":
    main()
