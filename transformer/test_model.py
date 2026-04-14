import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerFast

try:
    from .dataset import PairwiseClaimsDataset
    from .model import PairwiseTransformerClassifier, PairwiseTransformerConfig
except ImportError:
    from dataset import PairwiseClaimsDataset
    from model import PairwiseTransformerClassifier, PairwiseTransformerConfig


BASE_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = BASE_DIR / "tokenizers" / "claims_bpe"


def _write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_batch(dataset: PairwiseClaimsDataset) -> Dict[str, torch.Tensor]:
    samples = [dataset[index] for index in range(len(dataset))]
    keys = ("input_ids", "attention_mask", "token_type_ids", "labels", "inconsistency_labels")
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in keys
    }


def _assert_token_type_ids(batch: Dict[str, torch.Tensor], sep_token_id: int, pad_token_id: int) -> None:
    for input_ids, token_type_ids in zip(batch["input_ids"], batch["token_type_ids"]):
        ids = input_ids.tolist()
        types = token_type_ids.tolist()

        first_sep = ids.index(sep_token_id)
        second_sep = ids.index(sep_token_id, first_sep + 1)

        assert all(token_type == 0 for token_type in types[: first_sep + 1])
        assert all(token_type == 1 for token_type in types[first_sep + 1 : second_sep + 1])

        for token_id, token_type in zip(ids, types):
            if token_id == pad_token_id:
                assert token_type == 0


def _assert_mean_pooling_ignores_pad_tokens() -> None:
    config = PairwiseTransformerConfig(vocab_size=32, hidden_size=4, num_attention_heads=2, pooling="mean")
    model = PairwiseTransformerClassifier(config)

    hidden_states = torch.tensor(
        [[[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0], [100.0, 100.0, 0.0, 0.0]]]
    )
    attention_mask = torch.tensor([[1, 1, 0]])
    pooled = model.pool_sequence(hidden_states, attention_mask=attention_mask)
    expected = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
    assert torch.allclose(pooled, expected, atol=1e-6)


def _assert_segment_pooling_ignores_special_tokens() -> None:
    config = PairwiseTransformerConfig(
        vocab_size=32,
        hidden_size=2,
        num_attention_heads=1,
        cls_token_id=101,
        sep_token_id=102,
    )
    model = PairwiseTransformerClassifier(config)

    hidden_states = torch.tensor(
        [
            [
                [100.0, 100.0],
                [2.0, 0.0],
                [4.0, 0.0],
                [200.0, 200.0],
                [0.0, 3.0],
                [0.0, 5.0],
                [300.0, 300.0],
                [999.0, 999.0],
            ]
        ]
    )
    input_ids = torch.tensor([[101, 11, 12, 102, 21, 22, 102, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0]])

    pooled = model.pool_segments(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        input_ids=input_ids,
    )

    expected_a = torch.tensor([[3.0, 0.0]])
    expected_b = torch.tensor([[0.0, 4.0]])
    assert torch.allclose(pooled["segment_a"], expected_a, atol=1e-6)
    assert torch.allclose(pooled["segment_b"], expected_b, atol=1e-6)


def main() -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_PATH))

    examples = [
        {
            "pair_id": "pair-001",
            "text_a": "<insured_driver> I slowed to <speed> 10 kmh before the roundabout and lightly touched the rear bumper when traffic stopped suddenly in front of me.",
            "text_b": "<third_party_driver> Traffic stopped suddenly before the roundabout and the car behind tapped my rear bumper at very low speed.",
            "label": "supports",
            "inconsistency_type": "none",
        },
        {
            "pair_id": "pair-002",
            "text_a": "<insured_driver> I was reversing slowly out of a parking space near the supermarket when I heard a small scrape on the right side.",
            "text_b": "<witness> I was walking nearby and only noticed both cars already stopped after the noise, so I cannot confirm how the contact started.",
            "label": "neutral",
            "inconsistency_type": "none",
        },
        {
            "pair_id": "pair-003",
            "text_a": "<insured_driver> I was fully stopped at the red light when the other vehicle hit the back of my car.",
            "text_b": "<third_party_driver> The insured rolled backward from the incline and hit the front of my vehicle while the light was still red.",
            "label": "contradicts",
            "inconsistency_type": "dynamics_mismatch",
        },
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        jsonl_path = temp_dir_path / "pairwise_examples.jsonl"
        json_path = temp_dir_path / "pairwise_examples.json"

        _write_jsonl(jsonl_path, examples)
        json_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

        dataset_jsonl = PairwiseClaimsDataset(jsonl_path, tokenizer=tokenizer, max_length=256)
        dataset_json = PairwiseClaimsDataset(json_path, tokenizer=tokenizer, max_length=256)

        assert len(dataset_jsonl) == len(examples)
        assert len(dataset_json) == len(examples)

        batch = _build_batch(dataset_jsonl)

    batch_size = len(examples)
    assert batch["input_ids"].shape == (batch_size, 256)
    assert batch["attention_mask"].shape == (batch_size, 256)
    assert batch["token_type_ids"].shape == (batch_size, 256)
    assert batch["labels"].shape == (batch_size,)
    assert batch["inconsistency_labels"].shape == (batch_size,)

    _assert_token_type_ids(batch, tokenizer.sep_token_id, tokenizer.pad_token_id)
    _assert_mean_pooling_ignores_pad_tokens()
    _assert_segment_pooling_ignores_special_tokens()

    config = PairwiseTransformerConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id or 0,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        use_pairwise_comparison_head=True,
        use_inconsistency_head=True,
    )
    model = PairwiseTransformerClassifier(config)
    outputs = model(**batch)

    assert outputs["logits"] is not None
    assert outputs["probs"] is not None
    assert outputs["loss"] is not None
    assert outputs["inconsistency_logits"] is not None
    assert outputs["inconsistency_probs"] is not None
    assert outputs["inconsistency_loss"] is not None
    assert outputs["logits"].shape == (batch_size, 3)
    assert outputs["probs"].shape == (batch_size, 3)
    assert outputs["inconsistency_logits"].shape == (batch_size, 5)
    assert outputs["inconsistency_probs"].shape == (batch_size, 5)
    assert outputs["loss"].ndim == 0
    assert torch.allclose(outputs["probs"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    assert torch.allclose(outputs["inconsistency_probs"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    mean_model = PairwiseTransformerClassifier(replace(config, pooling="mean"))
    mean_outputs = mean_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
    )
    assert mean_outputs["logits"] is not None
    assert mean_outputs["logits"].shape == (batch_size, 3)

    print("Pairwise sanity check passed.")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {batch['input_ids'].shape[1]}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"CLS pooling logits shape: {tuple(outputs['logits'].shape)}")
    print(f"Inconsistency logits shape: {tuple(outputs['inconsistency_logits'].shape)}")
    print(f"Mean pooling logits shape: {tuple(mean_outputs['logits'].shape)}")
    print(f"Loss: {outputs['loss'].item():.6f}")


if __name__ == "__main__":
    main()
