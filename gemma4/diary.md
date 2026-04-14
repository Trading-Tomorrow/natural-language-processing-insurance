## Gemma 4 Claim-Level LLM

Nesta fase foi criada uma nova workflow em `gemma4/` para fine-tuning e benchmark de uma LLM claim-level.

Objetivo:

-> receber um claim estruturado em texto
-> estimar `probability_true`
-> devolver `verdict`
-> explicar a decisão
-> listar `incongruences`

Formato de output escolhido:

```json
{
  "probability_true": 0.82,
  "verdict": "true",
  "reasoning": "...",
  "incongruences": []
}
```

## O Que Foi Implementado

### `build_claim_sft_source.py`

Cria o dataset base claim-level a partir dos claims limpos já usados no stack transformer.

Cada record guarda:

-> `claim_id`
-> `input_text`
-> `binary_label`
-> `original_label`
-> `detected_damages`
-> `statements`
-> `fraud_indicators`

O `input_text` não expõe labels ocultas.

### `generate_claim_sft_teacher.py`

Gera os targets silver para SFT com Gemini.

O teacher vê:

-> claim visível
-> `binary_label`
-> `original_label`
-> `fraud_indicators`

Mas o output final continua limitado a:

-> `probability_true`
-> `verdict`
-> `reasoning`
-> `incongruences`

### `split_claim_sft_dataset.py`

Congela o split:

-> `train`
-> `validation`
-> `test`

O split é claim-level e estratificado por `original_label`, para preservar a mistura dos subtipos fraudulentos.

### `export_unsloth_chat.py`

Converte os splits para formato chat JSONL compatível com Unsloth:

-> `system`
-> `user`
-> `assistant`

### `kaggle_gemma4_sft.ipynb`

Notebook Kaggle para:

-> carregar Gemma 4 31B Instruct em 4-bit
-> aplicar LoRA
-> treinar com os ficheiros exportados
-> guardar o adapter final

### `infer_claim.py`

Script de inferência para um único claim.

Serve para testar:

-> Gemma base
-> Gemma fine-tuned

### `benchmark_gemma4.py`

Script oficial de benchmark before/after:

-> base Gemma 4
-> fine-tuned Gemma 4

No mesmo:

-> test split
-> prompt
-> schema
-> parser

Métricas implementadas:

-> accuracy
-> precision / recall / F1
-> ROC-AUC
-> PR-AUC
-> confusion matrix
-> JSON validity
-> schema validity
-> explanation behavior

### `summarize_benchmark.py`

Gera um resumo curto e citável do benchmark final.

## Decisões Importantes

Nesta versão:

-> o problema foi reformulado para claim-level
-> não há imagem
-> não há pairwise inference em runtime
-> o alvo principal é binário:
   -> `true`
   -> `not_true`

O pairwise transformer atual pode ajudar offline como teacher context, mas não faz parte do produto final desta etapa.

## Artefactos Esperados

Ficheiros principais:

-> `gemma4/data/claim_teacher_source.jsonl`
-> `gemma4/data/claim_sft_full.jsonl`
-> `gemma4/data/claim_sft_train.jsonl`
-> `gemma4/data/claim_sft_val.jsonl`
-> `gemma4/data/claim_sft_test.jsonl`

Outputs de benchmark:

-> `gemma4/outputs/benchmark_base_gemma4_test.jsonl`
-> `gemma4/outputs/benchmark_finetuned_gemma4_test.jsonl`
-> `gemma4/outputs/benchmark_comparison.json`

## Limitação Atual

Nesta fase a infraestrutura está implementada, mas o full pipeline ainda depende de:

-> gerar os targets silver com Gemini
-> correr o fine-tuning no Kaggle
-> correr o benchmark final base vs fine-tuned
