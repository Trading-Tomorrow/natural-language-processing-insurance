import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

try:
    from .pairwise_utils import build_token_type_ids, inconsistency_to_id, label_to_id
except ImportError:
    from pairwise_utils import build_token_type_ids, inconsistency_to_id, label_to_id


PathLike = Union[str, Path]
PairwiseRecord = Dict[str, Any]
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TOKENIZER_PATH = BASE_DIR / "tokenizers" / "claims_bpe"


def truncate_pair(token_ids_a: List[int], token_ids_b: List[int], max_length: int) -> Sequence[List[int]]:
    max_pair_tokens = max_length - 3
    truncated_a = list(token_ids_a)
    truncated_b = list(token_ids_b)

    while len(truncated_a) + len(truncated_b) > max_pair_tokens:
        if len(truncated_a) >= len(truncated_b):
            truncated_a.pop()
        else:
            truncated_b.pop()

    return truncated_a, truncated_b


def encode_pair_texts(
    tokenizer: PreTrainedTokenizerFast,
    text_a: str,
    text_b: str,
    max_length: int,
    cls_token_id: int,
    sep_token_id: int,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    token_ids_a = tokenizer.encode(text_a, add_special_tokens=False)
    token_ids_b = tokenizer.encode(text_b, add_special_tokens=False)
    token_ids_a, token_ids_b = truncate_pair(token_ids_a, token_ids_b, max_length=max_length)

    input_ids = [cls_token_id, *token_ids_a, sep_token_id, *token_ids_b, sep_token_id]
    attention_mask = [1] * len(input_ids)
    token_type_ids = build_token_type_ids(input_ids, sep_token_id, pad_token_id)

    padding_length = max_length - len(input_ids)
    if padding_length < 0:
        raise ValueError("Encoded pair exceeded max_length after truncation.")

    if padding_length > 0:
        input_ids.extend([pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        token_type_ids.extend([0] * padding_length)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
    }


def load_pairwise_records(path: PathLike) -> List[PairwiseRecord]:
    data_path = Path(path)

    if data_path.suffix == ".jsonl":
        records: List[PairwiseRecord] = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                payload = json.loads(stripped_line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected object at line {line_number} in {data_path}.")
                records.append(payload)
        return records

    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON list in {data_path}.")

    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object at index {index} in {data_path}.")

    return payload


class PairwiseClaimsDataset(Dataset):
    def __init__(
        self,
        data_path: PathLike,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        tokenizer_path: PathLike = DEFAULT_TOKENIZER_PATH,
        max_length: int = 256,
    ) -> None:
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.records = load_pairwise_records(self.data_path)
        self.tokenizer = tokenizer or PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

        self.cls_token_id = self._require_token_id(self.tokenizer.cls_token_id, "[CLS]")
        self.sep_token_id = self._require_token_id(self.tokenizer.sep_token_id, "[SEP]")
        self.pad_token_id = self._require_token_id(self.tokenizer.pad_token_id, "[PAD]")

        for index, record in enumerate(self.records):
            self._validate_record(record, index)

    @staticmethod
    def _require_token_id(token_id: Optional[int], token_name: str) -> int:
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required token id for {token_name}.")
        return token_id

    @staticmethod
    def _validate_record(record: PairwiseRecord, index: int) -> None:
        required_keys = ("pair_id", "text_a", "text_b", "label")
        missing_keys = [key for key in required_keys if key not in record]
        if missing_keys:
            raise ValueError(f"Missing keys {missing_keys} at record index {index}.")

        for text_key in ("text_a", "text_b"):
            if not isinstance(record[text_key], str):
                raise ValueError(f"{text_key} must be a string at record index {index}.")
        label_to_id(record["label"])
        if "inconsistency_type" in record:
            inconsistency_to_id(record["inconsistency_type"])

    def __len__(self) -> int:
        return len(self.records)

    def _truncate_pair(self, token_ids_a: List[int], token_ids_b: List[int]) -> Sequence[List[int]]:
        return truncate_pair(token_ids_a, token_ids_b, max_length=self.max_length)

    def _encode_pair(self, text_a: str, text_b: str) -> Dict[str, torch.Tensor]:
        return encode_pair_texts(
            tokenizer=self.tokenizer,
            text_a=text_a,
            text_b=text_b,
            max_length=self.max_length,
            cls_token_id=self.cls_token_id,
            sep_token_id=self.sep_token_id,
            pad_token_id=self.pad_token_id,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self._encode_pair(record["text_a"], record["text_b"])
        encoded["labels"] = torch.tensor(label_to_id(record["label"]), dtype=torch.long)
        encoded["inconsistency_labels"] = torch.tensor(
            inconsistency_to_id(record.get("inconsistency_type", "none")),
            dtype=torch.long,
        )
        return encoded
