import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerFast

try:
    from .dataset import DEFAULT_TOKENIZER_PATH, encode_pair_texts
    from .model import PairwiseTransformerClassifier, PairwiseTransformerConfig
    from .pairwise_utils import (
        ID_TO_INCONSISTENCY,
        ID_TO_LABEL,
        describe_inconsistency,
    )
except ImportError:
    from dataset import DEFAULT_TOKENIZER_PATH, encode_pair_texts
    from model import PairwiseTransformerClassifier, PairwiseTransformerConfig
    from pairwise_utils import ID_TO_INCONSISTENCY, ID_TO_LABEL, describe_inconsistency


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "pairwise_baseline" / "best_model.pt"
ROLE_TOKEN_MAP = {
    "insured_driver": "<insured_driver>",
    "third_party_driver": "<third_party_driver>",
    "impartial_witness": "<witness>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pairwise relation and inconsistency type.")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--context-mode", choices=("auto", "plain", "contextual"), default="auto")
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    parser.add_argument("--role-a", default="insured_driver")
    parser.add_argument("--role-b", default="third_party_driver")
    parser.add_argument("--vehicle-a", default="")
    parser.add_argument("--vehicle-b", default="")
    parser.add_argument("--incident-type-a", default="")
    parser.add_argument("--incident-type-b", default="")
    parser.add_argument("--detected-damages-a", default="")
    parser.add_argument("--detected-damages-b", default="")
    return parser.parse_args()


def choose_device(device_override: Optional[str]) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_space(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def format_damages(raw_value: str) -> str:
    items = [normalize_space(item).lower() for item in raw_value.split(",") if normalize_space(item)]
    return ", ".join(items)


def resolve_context_mode(requested_mode: str, checkpoint_dataset_path: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    return "contextual" if "contextual" in checkpoint_dataset_path else "plain"


def build_input_text(
    text: str,
    role: str,
    context_mode: str,
    vehicle: str,
    incident_type: str,
    detected_damages: str,
) -> str:
    role_key = normalize_space(role).lower()
    role_token = ROLE_TOKEN_MAP.get(role_key, f"<{role_key}>")
    clean_text = normalize_space(text)

    if context_mode == "plain":
        return f"{role_token} {clean_text}".strip()

    parts: List[str] = []
    clean_incident_type = normalize_space(incident_type)
    clean_damages = format_damages(detected_damages)
    clean_vehicle = normalize_space(vehicle)
    if clean_incident_type:
        parts.append(f"incident_type: {clean_incident_type}")
    if clean_damages:
        parts.append(f"detected_damages: {clean_damages}")
    if clean_vehicle and clean_vehicle.lower() != "none":
        parts.append(f"vehicle: {clean_vehicle}")
    parts.append(role_token)
    parts.append(clean_text)
    return " ".join(part for part in parts if part).strip()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    train_config = checkpoint.get("train_config", {})
    model_config = PairwiseTransformerConfig(**checkpoint["model_config"])

    tokenizer_path = Path(args.tokenizer_path or train_config.get("tokenizer_path", DEFAULT_TOKENIZER_PATH))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    model = PairwiseTransformerClassifier(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    context_mode = resolve_context_mode(
        args.context_mode,
        str(train_config.get("dataset_path", "")),
    )
    text_a = build_input_text(
        text=args.text_a,
        role=args.role_a,
        context_mode=context_mode,
        vehicle=args.vehicle_a,
        incident_type=args.incident_type_a,
        detected_damages=args.detected_damages_a,
    )
    text_b = build_input_text(
        text=args.text_b,
        role=args.role_b,
        context_mode=context_mode,
        vehicle=args.vehicle_b,
        incident_type=args.incident_type_b,
        detected_damages=args.detected_damages_b,
    )

    encoded = encode_pair_texts(
        tokenizer=tokenizer,
        text_a=text_a,
        text_b=text_b,
        max_length=model_config.max_position_embeddings,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**batch)

    relation_probs = outputs["probs"]
    if relation_probs is None:
        raise RuntimeError("Model did not return relation probabilities.")

    relation_id = int(relation_probs.argmax(dim=-1).item())
    relation_label = ID_TO_LABEL[relation_id]
    relation_score = float(relation_probs[0, relation_id].item())

    print(f"Context mode: {context_mode}")
    print(f"Predicted relation: {relation_label} ({relation_score:.4f})")

    inconsistency_probs = outputs.get("inconsistency_probs")
    if inconsistency_probs is not None:
        inconsistency_id = int(inconsistency_probs.argmax(dim=-1).item())
        inconsistency_label = ID_TO_INCONSISTENCY[inconsistency_id]
        inconsistency_score = float(inconsistency_probs[0, inconsistency_id].item())
        print(f"Predicted inconsistency type: {inconsistency_label} ({inconsistency_score:.4f})")
        print(f"Interpretation: {describe_inconsistency(inconsistency_label)}")


if __name__ == "__main__":
    main()
