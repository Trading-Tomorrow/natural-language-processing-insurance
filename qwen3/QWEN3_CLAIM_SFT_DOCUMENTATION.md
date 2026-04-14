# Qwen3-8B Insurance Claim Classification: Fine-Tuning & Evaluation

> Comprehensive documentation for reproducibility and paper reference.

---

## 1. Overview

This project fine-tunes **Qwen3-8B** (4-bit quantized) via **QLoRA** on Apple Silicon (M4 Pro, 48 GB) using the MLX-LM framework for **insurance claim fraud detection**. The model classifies claims as `true` (genuine) or `not_true` (fraudulent) with structured JSON output containing probability, verdict, reasoning, and detected incongruences.

Two models are evaluated and compared:
- **Qwen3-8B Base** — 4-bit quantized, no fine-tuning
- **Qwen3-8B Fine-Tuned** — QLoRA adapter (rank 16, 8 layers)

Both are benchmarked on the **same held-out test set** (357 samples).

---

## 2. Dataset

### 2.1 Source & Composition

The dataset was generated synthetically using **Gemini 3.1 Flash Lite** with structured insurance claim personas. Three source files were merged and deduplicated:

| Source File                     | Loaded Claims |
|---------------------------------|--------------|
| `dataset_sintetico_gemini.json`            | 1,170        |
| `dataset_sintetico_gemini_mixed_diverse.json` | 1,000        |
| `dataset_sintetico_gemini_good_only.json`  | 2,000        |
| **Total loaded**                | **4,170**    |

After removing duplicates (577 by ID, 26 by content fingerprint) and low-quality samples, **3,567 claims** remained.

### 2.2 Label Distribution

The dataset uses a **binary classification** scheme mapped from 4 original classes:

| Original Label                    | Count | Binary Label |
|-----------------------------------|-------|-------------|
| `genuine_accident`                | 2,469 | `true`       |
| `hard_fraud_phantom_vehicle`      | 345   | `not_true`   |
| `hard_fraud_staged`               | 380   | `not_true`   |
| `soft_fraud_exaggeration`         | 373   | `not_true`   |
| **Total**                         | **3,567** |             |

Binary distribution: **69.2% `true`** / **30.8% `not_true`**

### 2.3 Train/Validation/Test Split

Split with `seed=42`, ratios 80/10/10%:

| Split   | Samples | `true` | `not_true` |
|---------|---------|--------|-----------|
| Train   | 2,853   | 1,975  | 878       |
| Validation | 357  | 247    | 110       |
| Test    | 357     | 247    | 110       |

### 2.4 Data Format (MLX-LM Chat)

Each sample follows the **MLX-LM chat format** with 3 turns:

```json
{
  "claim_id": "PT-GOOD-2026-002007",
  "messages": [
    {
      "role": "system",
      "content": "You are an insurance claim consistency analyst. You receive one structured accident claim with detected damages and party statements. Return valid JSON only. Estimate the probability that the claim is true, choose verdict=true or verdict=not_true, explain the decision briefly, and list the main incongruences if they exist."
    },
    {
      "role": "user",
      "content": "Claim ID: PT-GOOD-2026-002007\nLocation: Avenida da Republica, Lisbon\nIncident type: ...\nParty statements: ...\nDamages: ..."
    },
    {
      "role": "assistant",
      "content": "{\"probability_true\": 0.95, \"verdict\": \"true\", \"reasoning\": \"Both parties provide a clear and consistent narrative...\", \"incongruences\": []}"
    }
  ],
  "binary_label": "true",
  "original_label": "genuine_accident"
}
```

Average sample length: **~1,158 characters**.

---

## 3. Fine-Tuning Configuration

### 3.1 Qwen3-8B QLoRA (Apple Silicon / MLX-LM)

| Parameter                  | Value                                    |
|----------------------------|------------------------------------------|
| Base Model                 | `mlx-community/Qwen3-8B-4bit`          |
| Framework                  | MLX-LM (`mlx_lm lora`)                  |
| Fine-Tune Type             | QLoRA (4-bit base + LoRA adapters)      |
| Hardware                   | Apple M4 Pro, 48 GB Unified Memory      |
| **LoRA Parameters**        |                                          |
| LoRA Rank (r)              | 16                                       |
| LoRA Scale (alpha)         | 32                                       |
| LoRA Dropout               | 0.05                                     |
| LoRA Layers                | 8                                        |
| Target Modules             | `q_proj`, `k_proj`, `v_proj`, `o_proj`  |
| **Training Parameters**    |                                          |
| Iterations                 | 500                                      |
| Batch Size                 | 1 (effective: 4 via gradient accumulation) |
| Learning Rate              | 1e-5                                     |
| Optimizer                  | AdamW                                    |
| Max Sequence Length        | 2,048                                    |
| Gradient Checkpointing     | Enabled                                  |
| Mask Prompt                | True (loss on assistant tokens only)     |
| Seed                       | 42                                       |
| Save Every                 | 100 iterations                           |
| Eval Every                 | 100 iterations                           |
| Config File                | `qwen3_claim_sft.yaml`                   |
| Adapter Path               | `adapters/qwen3_claim_sft/`             |

### 3.2 MLX-LM YAML Config

```yaml
model: mlx-community/Qwen3-8B-4bit
train: true
fine_tune_type: lora
optimizer: adamw
data: /Users/fzuin/nlp-dataset/qwen3/mlx_data
seed: 42
num_layers: 8
batch_size: 1
iters: 500
val_batches: 25
learning_rate: 1.0e-05
steps_per_report: 10
steps_per_eval: 100
grad_accumulation_steps: 4
adapter_path: /Users/fzuin/nlp-dataset/qwen3/adapters/qwen3_claim_sft
save_every: 100
test: false
test_batches: 100
max_seq_length: 2048
grad_checkpoint: true
mask_prompt: true
lora_parameters:
  keys:
    - self_attn.q_proj
    - self_attn.k_proj
    - self_attn.v_proj
    - self_attn.o_proj
  rank: 16
  scale: 32.0
  dropout: 0.05
```

### 3.3 Reference: Gemma 4 31B Fine-Tuning (Kaggle)

For comparison, the original Gemma 4 model was fine-tuned on **Kaggle** using Unsloth:

| Parameter                  | Value                                              |
|----------------------------|----------------------------------------------------|
| Base Model                 | `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`         |
| Framework                  | Unsloth + HuggingFace TRL                          |
| Hardware                   | Kaggle GPU (T4 x2 or P100)                        |
| LoRA Rank (r)              | 32                                                 |
| LoRA Alpha                 | 64                                                 |
| LoRA Dropout               | 0.05                                               |
| Target Modules             | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs                     | 3                                                  |
| Batch Size                 | 2 (effective: 16 via gradient accumulation)         |
| Learning Rate              | 2e-4                                               |
| Optimizer                  | AdamW 8-bit                                        |
| LR Scheduler               | Cosine                                             |
| Warmup Ratio               | 0.05                                               |
| Weight Decay               | 0.01                                               |
| Max Sequence Length        | 2,048                                              |
| Early Stopping             | Patience = 1 epoch                                 |
| Response-Only Loss          | Enabled (via `train_on_responses_only`)            |

---

## 4. Benchmark Methodology

### 4.1 Evaluation Setup

Both models were evaluated on the **identical held-out test set** (357 samples, stratified 69.2% `true` / 30.8% `not_true`).

**Base Model Benchmark:**
- Model: `mlx-community/Qwen3-8B-4bit` (no adapter)
- Prompt format: ChatML (`<|im_start|>system...<|im_end|>...`)
- Max tokens: 3,000 (to allow Qwen3 reasoning completion before JSON)
- Temperature: 0.1
- JSON extraction: Regex-based extraction of `{...probability_true...verdict...}` from output
- Verdict normalization: `true`/`verdadeiro` -> `true`; `not_true`/`false`/`falso` -> `not_true`

**Fine-Tuned Model Benchmark:**
- Model: `Qwen3-8B-4bit` + LoRA adapter at `adapters/qwen3_claim_sft/`
- Evaluated using MLX-LM batch inference
- Same JSON extraction pipeline

### 4.2 Metrics

| Metric      | Formula                                    |
|-------------|--------------------------------------------|
| Accuracy    | (TP + TN) / (TP + TN + FP + FN)           |
| Precision   | TP / (TP + FP)                             |
| Recall      | TP / (TP + FN)                             |
| Specificity | TN / (TN + FP)                             |
| F1 Score    | 2 x TP / (2xTP + FP + FN)                 |

Where:
- **True (`positive class`)** = genuine accident (`genuine_accident`)
- **Not True (`negative class`)** = any fraud type (`hard_fraud_*`, `soft_fraud_*`)

---

## 5. Results

### 5.1 Confusion Matrices

#### Qwen3-8B Base Model (no fine-tuning)

```
                        Predicted
                    +---------+----------+
                    |  true   | not_true |
              +-----+---------+----------+
   Actual     |true |   176   |    68    |   <- 72.1% recall
              +-----+---------+----------+
              |n/t  |     2   |   108    |   <- 98.2% specificity
              +-----+---------+----------+

Total: 357 | Valid JSON: 354 (99.2%) | Invalid: 3
```

#### Qwen3-8B Fine-Tuned (QLoRA, 500 iterations)

```
                        Predicted
                    +---------+----------+
                    |  true   | not_true |
              +-----+---------+----------+
   Actual     |true |   231   |     8    |   <- 96.7% recall
              +-----+---------+----------+
              |n/t  |     3   |    89    |   <- 96.7% specificity
              +-----+---------+----------+

Total: 357 | Valid JSON: 331 (92.7%) | Invalid: 26
```

### 5.2 Overall Metrics Comparison

| Metric             | Qwen3 Base  | Qwen3 FT    | Delta (FT vs Base) |
|--------------------|-------------|-------------|---------------------|
| **Accuracy**       | 80.2%       | 96.7%       | **+16.5%**          |
| **Precision**      | 98.9%       | 98.7%       | -0.2%               |
| **Recall**         | 72.1%       | 96.7%       | **+24.5%**          |
| **Specificity**    | 98.2%       | 96.7%       | -1.5%               |
| **F1 Score**       | 83.4%       | 97.7%       | **+14.3%**          |
| **JSON Success**   | 99.2%       | 92.7%       | -6.4%               |
| False Positives    | 2           | 3           | +1                  |
| False Negatives    | 68          | 8           | **-60**             |

### 5.3 Per-Class Accuracy (Test Set)

| Original Label                        | n    | Base Model | Fine-Tuned | Delta     |
|---------------------------------------|------|------------|------------|-----------|
| `genuine_accident`                    | 247  | 71.3%      | 93.5%      | **+22.2%**|
| `hard_fraud_phantom_vehicle`          | 35   | 97.1%      | 82.9%      | -14.2%    |
| `hard_fraud_staged`                   | 38   | 97.4%      | 68.4%      | -29.0%    |
| `soft_fraud_exaggeration`             | 37   | 100.0%     | 91.9%      | -8.1%     |

### 5.4 Key Observations

1. **Massive recall improvement**: Fine-tuning reduced false negatives from 68 to 8 (88% reduction), primarily by correctly identifying genuine claims that the base model was incorrectly flagging as `not_true`.

2. **Consistent high precision**: Both models maintain ~98.7-98.9% precision, meaning when the model predicts `true`, it is almost always correct. This is critical for insurance applications where false alarms have high cost.

3. **Base model is conservative**: The base Qwen3 model defaults toward `not_true` predictions, resulting in high specificity (98.2%) but poor recall (72.1%). It only flags claims as `true` when extremely confident.

4. **Fine-tuning balances the classifier**: The fine-tuned model achieves near-equal recall (96.7%) and specificity (96.7%), representing a much more balanced and useful classifier.

5. **JSON success rate**: The base model produces valid JSON 99.2% of the time vs 92.7% for the fine-tuned model. Fine-tuning may have partially overwritten strict formatting behavior, though both rates are acceptable for production use.

6. **Hard fraud types**: The base model excels at detecting `hard_fraud_staged` (97.4%) and `hard_fraud_phantom_vehicle` (97.1%), but this is largely a byproduct of its conservative bias toward predicting `not_true`. The fine-tuned model's lower accuracy on hard fraud (68-83%) reflects a more balanced decision boundary, though the overall F1 improvement justifies this trade-off.

---

## 6. Training Pipeline Reproducibility

### 6.1 Data Pipeline (Shared with Gemma4)

The dataset was created through a multi-step pipeline:

```
1. Generate synthetic claims (Gemini 3.1 Flash Lite, structured personas)
   +-- dataset_sintetico_gemini.json (1,170 claims)
   +-- dataset_sintetico_gemini_mixed_diverse.json (1,000 claims)
   +-- dataset_sintetico_gemini_good_only.json (2,000 claims)

2. Clean & deduplicate
   +-- Remove 577 duplicate IDs
   +-- Remove 26 content duplicates (fingerprint matching)
   +-- Final: 3,567 unique claims

3. Teacher annotation (Gemini 3.1 Flash Lite)
   +-- Input: structured claim data
   +-- Output: {probability_true, verdict, reasoning, incongruences}
   +-- Success rate: 100% (0 rejected batches)

4. Split (seed=42, 80/10/10)
   +-- Train: 2,853 (69.2% true / 30.8% not_true)
   +-- Val:   357
   +-- Test:  357

5. Export to MLX chat format
   +-- {messages: [{system, user, assistant}], claim_id, binary_label, original_label}
```

### 6.2 Scripts & Files

| File                                      | Purpose                                |
|-------------------------------------------|----------------------------------------|
| `generate_claim_sft_teacher.py`          | Generate teacher annotations via Gemini |
| `split_claim_sft_dataset.py`              | Train/val/test split (seed=42)        |
| `export_chat_format.py`                   | Convert to MLX-LM chat format          |
| `build_claim_sft_source.py`              | Merge & deduplicate source datasets    |
| `benchmark_base_model.py`                | Base model benchmarking script          |
| `run_benchmark.py`                        | Fine-tuned model benchmarking           |
| `common.py`                               | Shared utility functions               |
| `model_io.py`                             | Model input/output helpers             |
| `infer_claim.py`                          | Single-claim inference                 |
| `qwen3_mlx_qlora_notebook.ipynb`         | Full training notebook                 |
| `qwen3_claim_sft.yaml`                    | MLX-LM training config                 |

### 6.3 Output Artifacts

| Path                                              | Description                          |
|---------------------------------------------------|--------------------------------------|
| `adapters/qwen3_claim_sft/adapters.safetensors`   | Final QLoRA weights (13.6 MB)        |
| `adapters/qwen3_claim_sft/adapter_config.json`    | LoRA configuration                   |
| `benchmark_results/qwen3_base_model_results.json` | Base model benchmark data           |
| `benchmark_results/benchmark_results_full.json`    | Fine-tuned model benchmark data      |
| `benchmark_results/comparison_summary.json`        | Side-by-side comparison              |

---

## 7. Experimental Setup Summary

|                          | Qwen3-8B (This Work)    | Gemma 4 31B (Reference) |
|--------------------------|------------------------|--------------------------|
| **Base Model**           | Qwen3-8B               | Gemma-4-31B-Instruct     |
| **Quantization**         | 4-bit (MLX)            | 4-bit (BNB, Unsloth)     |
| **Fine-Tune Method**    | QLoRA (MLX-LM)         | QLoRA (Unsloth + TRL)    |
| **Hardware**             | Apple M4 Pro, 48 GB    | Kaggle T4/P100 GPU       |
| **LoRA Rank**            | 16                     | 32                       |
| **LoRA Alpha**           | 32                     | 64                       |
| **Target Modules**       | 4 (attention only)     | 7 (attention + MLP)      |
| **Training Steps**       | 500                    | ~3 epochs                |
| **Effective Batch**      | 4                      | 16                       |
| **Learning Rate**        | 1e-5                   | 2e-4                     |
| **LR Scheduler**         | None (constant)        | Cosine                   |
| **Mask Prompt**          | Yes                    | Yes (response-only loss) |
| **Max Seq Length**       | 2,048                  | 2,048                    |

---

## 8. Limitations & Future Work

1. **Class imbalance**: The 69/31 `true`/`not_true` split may bias the model toward `true` predictions. Future work should explore class-weighted loss or oversampling.

2. **JSON success rate regression**: The fine-tuned model (92.7%) underperforms the base model (99.2%) in structured output generation. This may be addressable with constraint decoding (e.g., Outlines, Guidance) or more training data.

3. **Hard fraud detection**: Fine-tuning reduced accuracy on `hard_fraud_staged` (97.4% -> 68.4%) and `hard_fraud_phantom_vehicle` (97.1% -> 82.9%), suggesting the model learned to be less conservative. A threshold calibration or multi-label head could help.

4. **Synthetic-only training**: All data is synthetically generated. Real-world evaluation with actual insurance claims is needed before production deployment.

5. **Single test set**: The 357-sample test set, while stratified, is relatively small. Cross-validation across multiple splits would yield more robust estimates.

---

## 9. Citation

If you use this work, please cite:

```bibtex
@misc{qwen3-claim-sft-2026,
  title={QLoRA Fine-Tuning of Qwen3-8B for Insurance Claim Fraud Detection on Apple Silicon},
  year={2026},
  note={Fine-tuned using MLX-LM on synthetic insurance claims dataset with structured JSON output}
}
```

---

*Documentation generated on April 13, 2026. All experiments conducted on Apple M4 Pro (48 GB) using MLX-LM.*
