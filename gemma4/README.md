# Gemma 4 Claim-Level SFT Workflow

This directory contains the first claim-level LLM workflow for the project.

The objective is to fine-tune **Gemma 4 31B Instruct** so that it receives one structured insurance claim and returns strict JSON with:

- `probability_true`
- `verdict`
- `reasoning`
- `incongruences`

## Workflow

1. Build the claim-level source dataset:

```bash
python3 /Users/fzuin/nlp-dataset/gemma4/build_claim_sft_source.py
```

2. Generate teacher targets with Gemini:

```bash
python3 /Users/fzuin/nlp-dataset/gemma4/generate_claim_sft_teacher.py
```

Dry run:

```bash
python3 /Users/fzuin/nlp-dataset/gemma4/generate_claim_sft_teacher.py --dry-run --max-claims 4
```

3. Freeze the train/validation/test split:

```bash
python3 /Users/fzuin/nlp-dataset/gemma4/split_claim_sft_dataset.py
```

4. Export chat-format JSONL for Unsloth:

```bash
python3 /Users/fzuin/nlp-dataset/gemma4/export_unsloth_chat.py
```

5. Fine-tune on Kaggle using:

- [kaggle_gemma4_sft.ipynb](/Users/fzuin/nlp-dataset/gemma4/kaggle_gemma4_sft.ipynb)

6. Run base vs fine-tuned benchmark on the frozen test split:

```bash
python3 /Users/fzuin/nlp-dataset/gemma4/benchmark_gemma4.py \
  --base-model-path unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --finetuned-model-path unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --finetuned-adapter-path /path/to/lora_adapter \
  --load-in-4bit
```

7. Produce a compact report summary:

```bash
python3 /Users/fzuin/nlp-dataset/gemma4/summarize_benchmark.py
```

## Output Artifacts

Main data artifacts:

- `gemma4/data/claim_teacher_source.jsonl`
- `gemma4/data/claim_sft_full.jsonl`
- `gemma4/data/claim_sft_train.jsonl`
- `gemma4/data/claim_sft_val.jsonl`
- `gemma4/data/claim_sft_test.jsonl`

Main benchmark artifacts:

- `gemma4/outputs/benchmark_base_gemma4_test.jsonl`
- `gemma4/outputs/benchmark_finetuned_gemma4_test.jsonl`
- `gemma4/outputs/benchmark_comparison.json`
- `gemma4/outputs/benchmark_comparison.md`
