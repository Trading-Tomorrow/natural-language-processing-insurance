# MLX Fine-Tuning Skill (LLM-Agnostic)

This skill helps you make good decisions when fine-tuning with MLX-LM. It is LLM-agnostic and uses concrete, reusable examples.

## 0. First Questions (Answer Before You Tune)

Provide short answers to these:

- What is the task? (classification, extraction, style transfer, summarization, chat assistant)
- What output format is required? (free text, strict JSON, schema)
- What is the domain? (finance, legal, health, support, etc.)
- What is the minimum acceptable quality? (accuracy, recall, format-valid rate)
- How much data do you have? (number of examples, average length)
- What hardware and time budget do you have? (Mac RAM, training hours)

These answers determine whether you should tune, how much data you need, and how to evaluate.

## 1. When to Fine-Tune (And When Not To)

- Fine-tune if you need domain-specific behavior, strict output formats, or improved calibration on your task.
- Do not fine-tune if prompt engineering already meets your quality goals or you lack high-quality labeled data.

## 2. What Fine-Tuning Changes in a Transformer

- **Attention heads** learn what tokens to focus on (e.g., claim details, contradictions, and constraints).
- **MLP layers** transform those signals into task-specific reasoning and output behavior.
- **LoRA adapters** shift these behaviors without overwriting base knowledge, so you can keep general language capability.

## 3. Data First: The Real Performance Lever

Use this checklist before touching hyperparameters:

- **Role structure:** Use MLX-LM chat format with `system`, `user`, `assistant` roles.
- **Instruction clarity:** The assistant output must match exactly what you want (JSON schema, labels, tone).
- **Diversity:** Cover edge cases and varied phrasing. Otherwise the model becomes brittle.
- **Length balance:** Include long and short responses if you want both at inference.
- **Label noise:** Spot-check samples. One bad pattern can poison behavior.

Example from this repo (insurance claims):

- Consistent system message specifies JSON-only output with fields.
- User message includes structured claim details.
- Assistant message is valid JSON with probability, verdict, reasoning, and incongruences.

## 4. MLX-LM Chat Format (Minimal Example)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an insurance claim consistency analyst. Return valid JSON only..."
    },
    {
      "role": "user",
      "content": "Claim ID: ...\nLocation: ...\nIncident type: ..."
    },
    {
      "role": "assistant",
      "content": "{\"probability_true\": 0.95, \"verdict\": \"true\", ...}"
    }
  ]
}
```

## 5. MLX-LM LoRA Defaults (Start Here)

Good defaults for Qwen3-8B with MLX on Apple Silicon:

- **Base model:** any MLX-compatible instruct model
- **LoRA rank (r):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **Target layers:** attention only (`q_proj`, `k_proj`, `v_proj`, `o_proj`) to start
- **Max sequence length:** 2048
- **Gradient checkpointing:** true
- **Mask prompt:** true

These defaults work as a starting point for most 7B to 8B instruct models.

## 6. When to Expand Target Layers

Start with attention layers. Expand to MLP layers (`gate_proj`, `up_proj`, `down_proj`) when:

- The model follows format but lacks domain-specific reasoning.
- Accuracy plateaus even with more data.
- You need the model to learn new facts or structured logic.

## 7. Learning Rate and Steps (MLX Guidance)

- MLX QLoRA is stable with lower LRs compared to GPU stacks.
- Start around `1e-5` to `2e-5` for 7B to 8B models; scale down for larger models.
- Use short runs first (e.g., 200 to 500 iterations) and evaluate.

## 8. Evaluation: Look Beyond Loss

- Track **accuracy, recall, and precision** on a stable validation set.
- For JSON tasks, track **valid JSON rate** (format regressions are common).
- Monitor **class balance**: a good accuracy can hide bias toward one label.

In this project:

- Fine-tuning improved recall dramatically, but reduced valid JSON rate.
- This trade-off is visible only if you measure both quality and format.

## 9. Common Failure Patterns (Quick Diagnosis)

- **Model ignores system message:** inconsistent system prompts in training data.
- **Short or generic answers:** training data too short or repetitive.
- **Bad JSON:** missing constraints or too much variability in assistant outputs.
- **Overfitting:** training loss drops but validation loss rises.

## 10. Example MLX-LM Config (Generic Template)

```yaml
model: <mlx-compatible-instruct-model>
train: true
fine_tune_type: lora
optimizer: adamw
data: <path-to-mlx-chat-data>
seed: 42
num_layers: 8
batch_size: 1
iters: 500
val_batches: 25
learning_rate: 1.0e-05
steps_per_report: 10
steps_per_eval: 100
grad_accumulation_steps: 4
adapter_path: <path-to-save-adapter>
save_every: 100
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

## 11. Decision Guide (Use This Before You Train)

Ask yourself:

- Is my dataset clean, consistent, and aligned with the output I want?
- Do I have a small evaluation set to validate gains?
- Am I optimizing for accuracy, format fidelity, or calibration?

If the answer is "no" to any of these, fix data first.

## 12. Examples (Use This Structure)

**Example A: Domain classification with JSON output**

- Task: classify support tickets into `billing`, `tech`, or `account`.
- System: “Return JSON only: {"label": "", "confidence": 0.0, "rationale": ""}.”
- User: raw ticket text.
- Assistant: valid JSON with one of the labels.

**Example B: Style transfer**

- Task: rewrite a paragraph into legal tone.
- System: “Rewrite in formal legal style, preserve meaning.”
- User: informal paragraph.
- Assistant: formal rewrite.

**Example C: Extraction**

- Task: extract key fields from a document.
- System: “Return JSON with fields: name, date, amount.”
- User: document text.
- Assistant: JSON with extracted fields (empty string if missing).
