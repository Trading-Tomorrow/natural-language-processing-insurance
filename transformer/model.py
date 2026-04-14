from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PairwiseTransformerConfig:
    vocab_size: int
    max_position_embeddings: int = 256
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    type_vocab_size: int = 2
    num_labels: int = 3
    num_inconsistency_labels: int = 5
    pad_token_id: int = 0
    cls_token_id: Optional[int] = None
    sep_token_id: Optional[int] = None
    pooling: str = "cls"
    use_segment_embeddings: bool = True
    use_pairwise_comparison_head: bool = False
    use_inconsistency_head: bool = False
    inconsistency_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'.")


class PairwiseEmbeddings(nn.Module):
    def __init__(self, config: PairwiseTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = (
            nn.Embedding(config.type_vocab_size, config.hidden_size)
            if config.use_segment_embeddings
            else None
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: Tensor, token_type_ids: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_length = input_ids.shape
        if seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_length} exceeds max_position_embeddings={self.config.max_position_embeddings}."
            )

        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)

        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            hidden_states = hidden_states + self.token_type_embeddings(token_type_ids)

        hidden_states = self.layer_norm(hidden_states)
        return self.dropout(hidden_states)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: PairwiseTransformerConfig) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_ffn = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        normalized_states = self.norm_attention(hidden_states)
        attention_output, _ = self.self_attention(
            normalized_states,
            normalized_states,
            normalized_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.attention_dropout(attention_output)

        normalized_states = self.norm_ffn(hidden_states)
        ffn_output = self.ffn(normalized_states)
        hidden_states = hidden_states + self.ffn_dropout(ffn_output)
        return hidden_states


class PairwiseTransformerClassifier(nn.Module):
    def __init__(self, config: PairwiseTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = PairwiseEmbeddings(config)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier_dropout = nn.Dropout(config.dropout)
        classifier_input_size = config.hidden_size * 5 if config.use_pairwise_comparison_head else config.hidden_size
        self.classifier = nn.Linear(classifier_input_size, config.num_labels)
        self.inconsistency_classifier = (
            nn.Linear(classifier_input_size, config.num_inconsistency_labels)
            if config.use_inconsistency_head
            else None
        )

    @staticmethod
    def masked_mean(hidden_states: Tensor, mask: Tensor) -> Tensor:
        expanded_mask = mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden_states = hidden_states * expanded_mask
        return masked_hidden_states.sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1.0)

    def build_special_token_mask(self, input_ids: Optional[Tensor]) -> Optional[Tensor]:
        if input_ids is None:
            return None

        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.config.cls_token_id is not None:
            special_token_mask |= input_ids == self.config.cls_token_id
        if self.config.sep_token_id is not None:
            special_token_mask |= input_ids == self.config.sep_token_id
        return special_token_mask

    def pool_sequence(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if self.config.pooling == "cls":
            return hidden_states[:, 0]

        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.size()[:2], device=hidden_states.device, dtype=torch.long)

        return self.masked_mean(hidden_states, attention_mask.bool())

    def pool_segments(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        input_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        valid_token_mask = attention_mask.bool()
        special_token_mask = self.build_special_token_mask(input_ids)
        if special_token_mask is not None:
            valid_token_mask = valid_token_mask & ~special_token_mask

        segment_a_mask = valid_token_mask & (token_type_ids == 0)
        segment_b_mask = valid_token_mask & (token_type_ids == 1)

        return {
            "segment_a": self.masked_mean(hidden_states, segment_a_mask),
            "segment_b": self.masked_mean(hidden_states, segment_b_mask),
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inconsistency_labels: Optional[Tensor] = None,
        class_weights: Optional[Tensor] = None,
        inconsistency_class_weights: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        hidden_states = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.final_layer_norm(hidden_states)
        pooled_output = self.pool_sequence(hidden_states, attention_mask=attention_mask)
        classifier_input = pooled_output

        if self.config.use_pairwise_comparison_head:
            pooled_segments = self.pool_segments(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                input_ids=input_ids,
            )
            pooled_a = pooled_segments["segment_a"]
            pooled_b = pooled_segments["segment_b"]
            classifier_input = torch.cat(
                [
                    pooled_output,
                    pooled_a,
                    pooled_b,
                    torch.abs(pooled_a - pooled_b),
                    pooled_a * pooled_b,
                ],
                dim=-1,
            )

        logits = self.classifier(self.classifier_dropout(classifier_input))
        probs = torch.softmax(logits, dim=-1)
        inconsistency_logits = None
        inconsistency_probs = None
        inconsistency_loss = None

        if self.inconsistency_classifier is not None:
            inconsistency_logits = self.inconsistency_classifier(self.classifier_dropout(classifier_input))
            inconsistency_probs = torch.softmax(inconsistency_logits, dim=-1)

        relation_loss = None
        if labels is not None:
            if class_weights is not None:
                class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
            relation_loss = F.cross_entropy(logits, labels, weight=class_weights)

        if inconsistency_labels is not None and inconsistency_logits is not None:
            if inconsistency_class_weights is not None:
                inconsistency_class_weights = inconsistency_class_weights.to(
                    device=inconsistency_logits.device,
                    dtype=inconsistency_logits.dtype,
                )
            inconsistency_loss = F.cross_entropy(
                inconsistency_logits,
                inconsistency_labels,
                weight=inconsistency_class_weights,
            )

        loss = relation_loss
        if relation_loss is not None and inconsistency_loss is not None:
            loss = relation_loss + self.config.inconsistency_loss_weight * inconsistency_loss
        elif relation_loss is None and inconsistency_loss is not None:
            loss = inconsistency_loss

        return {
            "logits": logits,
            "probs": probs,
            "relation_loss": relation_loss,
            "inconsistency_logits": inconsistency_logits,
            "inconsistency_probs": inconsistency_probs,
            "inconsistency_loss": inconsistency_loss,
            "loss": loss,
        }
