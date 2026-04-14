# Pairwise Transformer Encoder for Insurance Statement Consistency Detection

## Abstract

This report consolidates the current state of the pairwise contradiction-learning pipeline implemented in the project. The system is designed for local consistency analysis between two accident-related statements rather than direct claim-level fraud prediction. The core objective is to classify a pair of statements into one of three mutually exclusive relation labels: `supports`, `neutral`, or `contradicts`.

The final implementation combines a custom domain-specific BPE tokenizer with a transformer encoder built from scratch in PyTorch. The model is trained on a weakly supervised pairwise dataset derived from synthetic insurance claims generated earlier in the project. The strongest validated configuration is a plain pairwise input formulation with class-balanced training, an 8-layer encoder, hidden size 128, intermediate size 512, and an explicit comparison head. On the current validation split, this configuration reaches a macro F1 of `0.8705`.

## 1. Problem Formulation

The project treats inconsistency detection as a pairwise natural language inference style problem. This design was chosen because many insurance-related contradictions do not emerge from a single statement in isolation. Instead, they appear when two descriptions of the same event are directly compared.

This formulation has four advantages.

-> It matches the structure of the problem: contradiction is relational, not purely local.
-> It is more interpretable than immediate claim-level classification because each decision is attached to a concrete pair of statements.
-> One claim can generate multiple training instances, which increases supervision density.
-> It creates a suitable first stage for later claim-level aggregation.

The current system therefore focuses on pairwise statement classification and does not yet implement a claim-level aggregation model.

## 2. Data Foundation

### 2.1 Source Corpora

The current pairwise corpus is derived from three cleaned synthetic claim datasets:

-> `data/dataset_sintetico_gemini.json`
-> `data/dataset_sintetico_gemini_mixed_diverse.json`
-> `data/dataset_sintetico_gemini_good_only.json`

These sources are merged through `dataset_cleaning.py`, which performs structured loading, normalization, and deduplication by both `claim_id` and normalized textual fingerprint.

### 2.2 Cleaning Outcome

The final cleaning statistics are:

-> input claims: `4170`
-> duplicate IDs removed: `577`
-> duplicate content fingerprints removed: `26`
-> final unique claims: `3567`

These values are recorded in [pairwise_dataset_stats.json](/Users/fzuin/nlp-dataset/transformer/data/pairwise_dataset_stats.json).

## 3. Tokenization Stage

The tokenizer was already implemented before the pairwise classifier and remains unchanged in the final baseline. It is a custom BPE tokenizer trained on the project corpus.

Its main characteristics are:

-> domain-specific subword vocabulary
-> support for special insurance tokens such as `<insured_driver>`, `<third_party_driver>`, `<insurance_adjuster>`, and `<speed>`
-> upstream normalization of structured expressions, such as:
`10 km/h -> <speed> 10 kmh`

This stage produces token sequences and token IDs, but it does not itself encode semantic relations between statements. Semantic comparison is handled by the transformer encoder.

## 4. Pairwise Dataset Construction

### 4.1 Pairwise Input Format

Each example is encoded exactly as:

`[CLS] statement_a [SEP] statement_b [SEP]`

The role of the special tokens is the following:

-> `[CLS]` marks the beginning of the entire sequence and supports global pooling
-> `[SEP]` separates the two statements and closes the sequence
-> `[PAD]` is used to reach the fixed maximum length of `256`

The dataset implementation tokenizes `statement_a` and `statement_b` without automatic special tokens, reserves three positions for `[CLS]` and the two `[SEP]` markers, and applies pair-aware truncation when the sequence exceeds the maximum length.

### 4.2 Labels

The relation labels are:

-> `supports = 0`
-> `neutral = 1`
-> `contradicts = 2`

In the final version, the pairwise builder also supports an auxiliary weak label called `inconsistency_type`, but this auxiliary task is not part of the final accepted baseline.

### 4.3 Pairwise Generation Strategy

The pairwise dataset is created by `build_pairwise_dataset.py`.

The builder currently performs:

-> within-claim pair generation between the `insured_driver` and another statement from the same claim
-> cross-claim sampling for `neutral` examples
-> weak supervision of `supports` and `contradicts` using the synthetic claim labels and available fraud indicators

This is therefore a weakly supervised pairwise corpus rather than a manually annotated gold dataset.

### 4.4 Final Pairwise Corpus Size

The final plain pairwise corpus contains:

-> full dataset: `4752` examples
-> balanced dataset: `2610` examples

The full label distribution is:

-> `supports = 1506`
-> `neutral = 2376`
-> `contradicts = 870`

The accepted training setup uses the full dataset with balanced class weighting rather than downsampling to the balanced subset.

## 5. Model Architecture

### 5.1 Core Encoder

The pairwise classifier is implemented in [model.py](/Users/fzuin/nlp-dataset/transformer/model.py) and is built entirely in PyTorch without using pretrained transformer model classes.

The encoder includes:

-> token embeddings
-> learnable positional embeddings
-> optional segment embeddings based on `token_type_ids`
-> a stack of transformer encoder blocks
-> residual connections
-> layer normalization
-> feed-forward sublayers with GELU activation

Each encoder block uses `nn.MultiheadAttention` with `batch_first=True`. Padding is ignored through the standard key padding mask derived from the attention mask.

### 5.2 Pooling

Two pooling strategies are implemented:

-> `cls`
-> masked `mean`

The accepted baseline uses `cls` pooling as the global representation.

### 5.3 Comparison Head

The strongest model variant extends the encoder with an explicit pairwise comparison head. Instead of classifying from the global pooled representation alone, the model constructs a richer comparison representation using:

-> global pooled representation
-> pooled representation of statement A
-> pooled representation of statement B
-> element-wise absolute difference `|A - B|`
-> element-wise product `A * B`

This design was motivated by the relational nature of the task. In the final ablation study, the comparison head provided the best overall validation result.

## 6. Implemented Software Components

The following files constitute the current pairwise stack.

### 6.1 Core Modeling Files

-> [model.py](/Users/fzuin/nlp-dataset/transformer/model.py): transformer encoder, pooling, optional comparison head, optional multitask inconsistency head
-> [pairwise_utils.py](/Users/fzuin/nlp-dataset/transformer/pairwise_utils.py): label mappings, token type helpers, inconsistency label mappings
-> [dataset.py](/Users/fzuin/nlp-dataset/transformer/dataset.py): JSON/JSONL pairwise dataset loading and tensorization
-> [test_model.py](/Users/fzuin/nlp-dataset/transformer/test_model.py): sanity checks for tensor shapes, loss computation, and pooling

### 6.2 Data and Training Utilities

-> [dataset_cleaning.py](/Users/fzuin/nlp-dataset/transformer/dataset_cleaning.py): loading, deduplication, and merging of claim datasets
-> [build_pairwise_dataset.py](/Users/fzuin/nlp-dataset/transformer/build_pairwise_dataset.py): weakly supervised pair construction
-> [train_pairwise.py](/Users/fzuin/nlp-dataset/transformer/train_pairwise.py): training loop, checkpointing, balanced class weighting
-> [evaluate_pairwise.py](/Users/fzuin/nlp-dataset/transformer/evaluate_pairwise.py): evaluation, confusion matrices, per-class metrics
-> [predict_pairwise.py](/Users/fzuin/nlp-dataset/transformer/predict_pairwise.py): inference for relation prediction and experimental inconsistency prediction

## 7. Training Protocol

The accepted training configuration is:

-> dataset: full plain pairwise dataset
-> validation split: `0.2`
-> random seed: `42`
-> optimizer: `AdamW`
-> learning rate: `3e-4`
-> weight decay: `0.01`
-> batch size: `32`
-> epochs: `32`
-> max length: `256`
-> pooling: `cls`
-> device: `mps`
-> class weighting: `balanced`

Balanced class weighting is computed as:

`weight_c = N / (C * n_c)`

where `N` is the number of training examples, `C` is the number of classes, and `n_c` is the number of examples in class `c`.

This allows the model to use the full pairwise corpus while compensating for the class imbalance in the relation labels.

## 8. Experimental Development Step by Step

### 8.1 Initial Pairwise Baseline

The first baseline used the pairwise transformer on the earlier dataset configuration. This stage established that the architecture was operational and capable of learning the three-way relation task.

### 8.2 Transition to the Full Pairwise Dataset

The next major step was to move from the balanced subset to the full weakly supervised pairwise dataset and to enable balanced class weighting. This increased the amount of supervision without discarding large portions of the available data.

### 8.3 Corpus Expansion

An additional diverse synthetic dataset was added to the original corpus, after which:

-> the tokenizer was retrained
-> the claim dataset was re-cleaned
-> the pairwise dataset was rebuilt

This led to the current corpus size of `3567` cleaned claims and `4752` pairwise examples.

### 8.4 Capacity Search

Several architecture variants were tested. The main outcome of this stage was that increasing the feed-forward capacity was more effective than increasing depth alone.

The most relevant comparison was:

-> `L8 H128 I256`: weaker
-> `L8 H128 I512`: stronger

Increasing only the number of layers did not consistently improve the validation macro F1.

### 8.5 Context Injection vs Pairwise Comparison

An ablation study was conducted to isolate two changes:

-> explicit contextual prefixes in the text input
-> an explicit comparison head in the classifier

The results were:

-> plain input + no comparison head: `macro_f1 = 0.8694`
-> plain input + comparison head: `macro_f1 = 0.8705`
-> contextual input + no comparison head: `macro_f1 = 0.8658`
-> contextual input + comparison head: `macro_f1 = 0.8580`

The conclusion was that the comparison head was beneficial, while explicit contextualization of the text input did not improve the model in its current form.

### 8.6 Multitask Extension

An experimental multitask variant was implemented to predict both:

-> the pairwise relation
-> a weakly supervised inconsistency category

The auxiliary inconsistency taxonomy currently includes:

-> `none`
-> `damage_mismatch`
-> `dynamics_mismatch`
-> `phantom_vehicle`
-> `scripted_narrative`

This model was promising for interpretability but did not surpass the best single-task baseline on the main relation task.

## 9. Final Accepted Baseline

The accepted baseline is the plain-input pairwise transformer with balanced class weighting and the explicit comparison head.

Its configuration is:

-> encoder depth: `8` layers
-> hidden size: `128`
-> attention heads: `8`
-> intermediate size: `512`
-> pooling: `cls`
-> segment embeddings: enabled
-> comparison head: enabled
-> class weighting: balanced

The checkpoint is:

-> [best_model.pt](/Users/fzuin/nlp-dataset/transformer/checkpoints/pairwise_full_weighted_expanded_ffn512_plain_comparison_head/best_model.pt)

The training artefacts are:

-> [training_history.json](/Users/fzuin/nlp-dataset/transformer/checkpoints/pairwise_full_weighted_expanded_ffn512_plain_comparison_head/training_history.json)
-> [training_curves.png](/Users/fzuin/nlp-dataset/transformer/figures/pairwise_full_weighted_expanded_ffn512_plain_comparison_head_training_curves.png)
-> [validation_dashboard.png](/Users/fzuin/nlp-dataset/transformer/figures/pairwise_full_weighted_expanded_ffn512_plain_comparison_head_validation_dashboard.png)

Its validation performance is:

-> loss: `0.4493`
-> accuracy: `0.8737`
-> macro precision: `0.8570`
-> macro recall: `0.8959`
-> macro F1: `0.8705`
-> best epoch: `26`

Per-class F1 scores are:

-> `supports`: `0.8612`
-> `neutral`: `0.8879`
-> `contradicts`: `0.8623`

## 10. Error Pattern Analysis

The best baseline is strong overall, but the confusion matrix reveals a clear residual error pattern. The dominant mistake is the prediction of `supports` for examples that are actually `neutral`.

The main confusion counts in the final baseline are:

-> `neutral -> supports = 58`
-> `neutral -> contradicts = 29`
-> `supports -> contradicts = 16`
-> `supports -> neutral = 9`
-> `contradicts -> supports = 6`
-> `contradicts -> neutral = 2`

This suggests that the model captures strong pairwise coherence reliably, but it still tends to over-predict `supports` for semantically close but not fully entailing pairs.

## 11. Experimental Multitask Result

The multitask model was trained with the same main encoder configuration but with an additional inconsistency head.

Checkpoint:

-> [best_model.pt](/Users/fzuin/nlp-dataset/transformer/checkpoints/pairwise_multitask_plain_comparison/best_model.pt)

Validation results:

-> relation macro F1: `0.8659`
-> relation accuracy: `0.8705`
-> inconsistency macro F1: `0.6109`
-> inconsistency accuracy: `0.8989`
-> best epoch: `22`

This result shows that the auxiliary task is learnable to a non-trivial degree, especially for `none`, `damage_mismatch`, and `phantom_vehicle`. However, the inconsistency head remains experimental and does not yet provide sufficiently reliable case-level explanations.

## 12. What Is Implemented and What Is Missing

### 12.1 Implemented

-> custom domain-specific tokenizer
-> cleaned combined claim corpus
-> weakly supervised pairwise dataset generation
-> transformer encoder from scratch in PyTorch
-> explicit comparison head
-> training and evaluation scripts
-> validation dashboards and training curves
-> experimental multitask inconsistency head

### 12.2 Missing

-> manually validated gold pairwise benchmark
-> claim-disjoint train, validation, and test protocol
-> claim-level aggregation model
-> final explanation mechanism that can reliably identify what exactly does not match between two statements

## 13. Limitations

The current system has several important limitations that should be stated explicitly.

First, the pairwise dataset is weakly supervised and derived from synthetic claims. This means that relation labels are operationally useful, but they are not equivalent to manual gold annotations.

Second, the current split is a pairwise train/validation split rather than a claim-disjoint evaluation protocol. This is acceptable for internal experimentation, but it is weaker than a stricter research-grade protocol.

Third, the auxiliary inconsistency labels are also weakly supervised. As a consequence, the multitask model can provide coarse category predictions, but these predictions should not yet be treated as trustworthy explanations.

Fourth, the project currently solves local pairwise consistency detection only. It does not yet aggregate evidence at claim level.

## 14. Step-by-Step Summary of the Final Pipeline

The final workflow can be summarized as follows.

1. Synthetic claim datasets are generated and stored in JSON format.
2. `dataset_cleaning.py` loads and deduplicates the claim corpora.
3. The custom BPE tokenizer is trained on the cleaned combined corpus.
4. `build_pairwise_dataset.py` converts cleaned claims into weakly supervised pairwise examples.
5. `dataset.py` converts each pair into the exact model input:
`[CLS] statement_a [SEP] statement_b [SEP]`
6. `train_pairwise.py` trains the transformer encoder with balanced class weighting.
7. `evaluate_pairwise.py` computes validation metrics, confusion matrices, and per-class scores.
8. The best checkpoint is selected using validation macro F1.
9. `predict_pairwise.py` supports direct inference on new statement pairs.

## 15. Conclusion

The current implementation constitutes a complete first pairwise transformer baseline for insurance statement consistency detection. The system is academically coherent, technically functional, and sufficiently documented to serve as a solid NLP course project baseline.

The strongest validated model is a transformer encoder trained from scratch on weakly supervised pairwise data with a custom tokenizer and an explicit comparison head. Its validation macro F1 of `0.8705` indicates that the architecture is effective at capturing local support, neutrality, and contradiction relations between accident statements.

The most appropriate next research step is not further depth scaling, but stronger supervision and stricter evaluation. In particular, future work should prioritize a cleaner benchmark for inconsistency categories and a claim-disjoint evaluation setup.
