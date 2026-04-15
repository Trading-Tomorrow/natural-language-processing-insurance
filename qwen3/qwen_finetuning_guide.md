# Qwen Fine-Tuning Guide for ML Students: A Transformer-Focused Overview

This guide explains, at a high level, why fine-tuning works for Transformers and what you are changing when you adapt Qwen models. The emphasis is on intuition about attention heads and representation learning, with minimal math.

## 1. What a Transformer Is (In One Page)
Transformers process text as a sequence of tokens. Each layer builds a better representation of those tokens by mixing information across the sequence. Two main parts make this work:

- **Attention:** Lets each token look at other tokens that matter for the current prediction.
- **MLP (feed-forward) blocks:** Transform the mixed information into richer features.

Stacking many layers creates a hierarchy: early layers focus on local syntax, middle layers combine context, and later layers synthesize high-level meaning and task behavior.

## 2. Why Attention Heads Exist (And What They Do)
Think of attention heads as specialized “channels” of focus. Each head can learn a different pattern of dependency:

- One head may track subject-verb agreement.
- Another may focus on the last user question.
- Another might follow formatting patterns (lists, JSON, or code blocks).

By having multiple heads, the model can attend to multiple relationships simultaneously. This is why attention is so powerful for language tasks with long-range dependencies.

## 3. What Fine-Tuning Changes in a Transformer
During fine-tuning, you are not rewriting the whole model. You are nudging its internal representations so that the same architecture behaves differently for your domain.

- **Attention layers:** You adjust how the model decides *what to attend to*. This changes which context signals it considers important.
- **MLP layers:** You adjust how the model transforms those signals into meaning and output behavior.

If you use LoRA, you are adding small adapter matrices that slightly shift these internal computations without destroying the base model’s knowledge. Think of it as adding “steering handles” to a large ship rather than rebuilding the ship.

## 4. ChatML Data Formatting: What the Model Actually Sees
Qwen uses **ChatML** to mark conversation structure with explicit boundary tokens.

- **Under the Hood Tokens:** Qwen relies on `<|im_start|>` and `<|im_end|>` tokens.
  - A user message internally looks like: `<|im_start|>user\nWhat is machine learning?<|im_end|>\n`
- **Why roles matter:** The `system` message shapes behavior; the `user` message sets the task; the `assistant` content is what you want the model to learn to produce. If your data collapses these roles, the model learns to blur them and becomes less controllable.
- **One example (HuggingFace style):**
  ```json
  [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
  ```

**Packing vs. one-sample-per-row:** To improve throughput, trainers often pack multiple short dialogues into one long sequence. This improves GPU utilization but increases the importance of correct label masking (see Section 4).

## 5. Model Architecture Context (Why Targets Matter)
Qwen is a decoder-only Transformer. You will encounter:

- **Attention blocks:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **MLP blocks:** `gate_proj`, `up_proj`, `down_proj`
- **LayerNorms / RMSNorms** (depending on the variant)

When you choose LoRA target modules, you are deciding *where* the model is allowed to adapt. Limiting to attention often helps style and instruction-following, but adding MLP layers allows learning new facts and skills more efficiently.

## 6. Parameter-Efficient Fine-Tuning (PEFT): LoRA and QLoRA
Full fine-tuning of a 7B model often needs 120GB+ of VRAM. **LoRA** and **QLoRA** shift the work to small adapters and low-bit weights, keeping the base frozen.

- **LoRA mechanism:** Instead of updating a full matrix $W \in \mathbb{R}^{d \times k}$, LoRA injects low-rank matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$, creating $W' = W + B A$.
- **Rank ($r$):** Capacity of the adapter. Typical values are 8, 16, or 64. Smaller values reduce overfitting and VRAM but limit new knowledge.
- **Alpha ($\alpha$):** Scaling factor. The common rule is $\alpha = r$ or $\alpha = 2r$. Effective update magnitude is proportional to $\alpha / r$.
- **QLoRA:** Base model is 4-bit (NF4). Adapters remain in 16-bit. This lets consumer GPUs fine-tune at useful sequence lengths.
- **Target modules:** For Qwen, many experiments benefit from targeting *all linear layers* (attention + MLP). If you are VRAM-constrained, start with `q_proj`, `k_proj`, `v_proj`, `o_proj` and add MLPs later.

## 7. Loss Masking, Padding, and Why Training Sometimes "Looks Fine" but Fails
Causal LMs predict the next token. You only want loss on the assistant’s response tokens.

- **Mask prompts:** Labels for `system` and `user` tokens should be `-100` (ignored by CrossEntropyLoss).
- **The `-100` rule:** In PyTorch and HuggingFace `SFTTrainer`, any label ID set to `-100` is ignored in loss computation.
  - *Input IDs Example:* `[<|im_start|>, user, ..., <|im_end|>, <|im_start|>, assistant, ...]`
  - *Label IDs Example:* `[ -100, -100, ..., -100, <|im_start|>, assistant, ...]`
- **Padding:** Batch tensors must be rectangular. Pad tokens (often `<|im_end|>`) must also be masked to `-100` or your model will learn to output padding artifacts.

## 8. Dataset Quality: The Biggest Lever
Even perfect hyperparameters cannot save poor data.

- **Diversity:** Avoid training on a single phrasing style. Paraphrase or mix sources.
- **Instruction clarity:** If the user query is ambiguous, the model learns ambiguity.
- **Length distribution:** If most samples are short, the model will optimize short responses. Include long-form answers if you need them.
- **Contamination:** If your data includes incorrect or contradictory answers, the model learns them. Validate with spot checks and small eval sets.

## 9. Hyperparameters: Practical Defaults and Why They Work

- **Learning rate:** LoRA usually needs higher LR than full fine-tuning. `1e-4` to `2e-4` is a good starting band for Qwen LoRA.
- **Scheduler + warmup:** Cosine schedule with `0.05` to `0.10` warmup ratio prevents unstable early updates.
- **Batch size and grad accumulation:** Use accumulation to reach effective batch sizes of 16 to 128 without increasing VRAM.
- **Weight decay:** `0.01` to `0.1` keeps adapters from exploding and reduces overfitting.
- **Sequence length:** Longer context improves reasoning and instruction-following but is more expensive. Increase gradually and track loss stability.

## 10. Memory and Throughput Optimizations

- **Gradient checkpointing:** Saves VRAM by recomputing activations during backward. Expect ~20% slower training but large memory savings.
- **Flash Attention 2:** Faster and more memory efficient attention, especially for long sequences.
- **Paged optimizers:** Some stacks offer CPU offload or paged optimizers to handle large activations without OOM.

## 11. Evaluation: Avoiding Overfitting and Forgetting

- **Train/val split:** Keep 5% to 10% for validation. If training loss drops but val loss rises, you are memorizing.
- **Catastrophic forgetting:** Too many epochs can harm general capabilities. For SFT, 1 to 3 epochs is often enough.
- **Task-based evals:** Build a small, stable eval set that reflects your target tasks and measure outputs, not just loss.

## 12. Checkpointing and Deployment

- **Adapter saving:** LoRA saves only adapter weights (tens to hundreds of MB). The base model is unchanged.
- **Merging adapters:** You can merge LoRA weights into the base for single-file deployment, but you lose the ability to swap adapters.
- **Versioning:** Save hyperparameters and dataset version alongside adapters to ensure reproducibility.

## 13. Tooling Stack (What to Use and When)

- **Unsloth:** Fast, memory-efficient Qwen fine-tuning with optimized kernels. Great for student hardware.
- **Axolotl / LLaMA-Factory:** YAML-based training pipelines that reduce boilerplate and handle ChatML packing.
- **HuggingFace TRL:** `SFTTrainer` manages masking and dataset collation while still letting you write custom logic.

## 14. Common Failure Modes (And How to Recognize Them)

- **Model answers in the user’s voice:** Usually due to missing role tokens or incorrect label masking.
- **Exploding loss early in training:** Often too high LR or no warmup.
- **Model ignores system prompt:** Data does not consistently include system messages.
- **Overly short answers:** Dataset dominated by short responses or aggressive max length truncation.

## 15. Minimal Experimental Workflow

1. Start with a small, clean dataset (1k to 10k examples).
2. Run LoRA with `r=8` or `r=16`, LR `1e-4`, cosine warmup 5%.
3. Evaluate on a hand-built set of 50 to 200 prompts.
4. Increase dataset size or adapter rank only if the eval set demands it.
