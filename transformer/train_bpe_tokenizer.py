# Here we use BPE (Byte-Pair Encoding).
# BPE was used in models like GPT-2 and RoBERTa.
# BERT uses WordPiece, not BPE.

import re
from pathlib import Path
from typing import Dict, Iterable, List, Any, Tuple

from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from dataset_cleaning import load_and_clean_default_claims, print_cleaning_report


SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "<claim>",
    "<location>",
    "<incident_type>",
    "<label>",
    "<fraud_indicators>",
    "<statement>",
    "<insured_driver>",
    "<third_party_driver>",
    "<insurance_adjuster>",
    "<witness>",
    "<police_report>",
    "<expert_report>",
    "<vehicle>",
    "<speed>",
    "<none>",
    "<sep_stmt>",
]


def normalize_speed(text: str) -> str:
    """
    Normalize only speed expressions.
    Example:
        10 km/h -> <speed> 10 kmh
    """
    return re.sub(
        r"\b(\d+(?:[.,]\d+)?)\s*km\s*/\s*h\b",
        r"<speed> \1 kmh",
        text,
        flags=re.IGNORECASE,
    )


def normalize_role(role: str) -> str:
    """
    Map dataset roles to special tokens.
    Unknown roles are converted into a token-like format if possible.
    """
    if not role:
        return "<none>"

    known = {
        "insured_driver": "<insured_driver>",
        "third_party_driver": "<third_party_driver>",
        "insurance_adjuster": "<insurance_adjuster>",
        "witness": "<witness>",
        "police_report": "<police_report>",
        "expert_report": "<expert_report>",
    }

    if role in known:
        return known[role]

    safe = role.strip().lower().replace(" ", "_")
    return f"<{safe}>"


def flatten_claim_for_tokenizer(claim: Dict[str, Any]) -> str:
    """
    Convert one claim into a structured text block.
    This is what the BPE tokenizer will see during training.
    """
    claim_id = claim.get("claim_id", "")
    location = claim.get("location", "")
    incident_type = claim.get("incident_type", "")
    label = claim.get("ground_truth_label", "")
    fraud_indicators = claim.get("fraud_indicators", [])
    statements = claim.get("statements", [])

    parts = [
        "<claim>", str(claim_id),
        "<location>", str(location),
        "<incident_type>", str(incident_type),
    ]

    # Keep label/fraud indicators only if you want the tokenizer exposed
    # to those domain words. For pure inference realism, you can remove them.
    if label:
        parts.extend(["<label>", str(label)])

    if fraud_indicators:
        parts.extend(["<fraud_indicators>", " ; ".join(map(str, fraud_indicators))])

    for st in statements:
        role_token = normalize_role(st.get("role", ""))
        vehicle = st.get("vehicle", "none") or "none"
        text = normalize_speed(st.get("text", ""))

        parts.extend([
            "<statement>",
            role_token,
            "<vehicle>",
            str(vehicle),
            str(text),
            "<sep_stmt>",
        ])

    return " ".join(parts).strip()


def training_corpus(claims: List[Dict[str, Any]]) -> Iterable[str]:
    """
    Iterator used by train_from_iterator().
    Yields one structured text sequence per claim.
    """
    for claim in claims:
        yield flatten_claim_for_tokenizer(claim)


def build_tokenizer(vocab_size: int = 20000, min_frequency: int = 1) -> Tuple[Tokenizer, trainers.BpeTrainer]:
    """
    Create a BPE tokenizer with light normalization and pre-tokenization.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    return tokenizer, trainer


def save_fast_tokenizer(tokenizer: Tokenizer, output_dir: str) -> PreTrainedTokenizerFast:
    """
    Wrap the raw tokenizers.Tokenizer into a transformers-compatible fast tokenizer.
    """
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        additional_special_tokens=[
            tok for tok in SPECIAL_TOKENS
            if tok not in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
        ],
    )

    fast_tokenizer.save_pretrained(output_dir)
    return fast_tokenizer


def inspect_tokenizer(
    fast_tokenizer: PreTrainedTokenizerFast,
    samples: List[str],
    max_examples: int = 5,
) -> None:
    print("\n=== TOKENIZER INSPECTION ===\n")
    for i, text in enumerate(samples[:max_examples], start=1):
        encoded = fast_tokenizer(text, add_special_tokens=True)
        tokens = fast_tokenizer.convert_ids_to_tokens(encoded["input_ids"])

        print(f"--- Example {i} ---")
        print("TEXT:")
        print(text)
        print("\nTOKENS:")
        print(tokens)
        print("\nTOKEN IDS:")
        print(encoded["input_ids"])
        print("\n")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "tokenizers" / "claims_bpe"
    vocab_size = 20000
    min_frequency = 1  # Lower threshold to keep more rare domain terms

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    claims, cleaning_stats = load_and_clean_default_claims()
    print_cleaning_report(cleaning_stats)

    tokenizer, trainer = build_tokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    tokenizer.train_from_iterator(training_corpus(claims), trainer=trainer)

    # Save raw tokenizer JSON too
    raw_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(raw_json_path))

    fast_tokenizer = save_fast_tokenizer(tokenizer, str(output_dir))

    # Build some samples for inspection
    sample_structured = [flatten_claim_for_tokenizer(c) for c in claims[:3]]

    sample_pairwise = []
    for claim in claims[:3]:
        statements = claim.get("statements", [])
        if len(statements) >= 2:
            a = statements[0]
            b = statements[1]
            pair_text = (
                f"{normalize_role(a.get('role', ''))} {normalize_speed(a.get('text', ''))} "
                f"[SEP] "
                f"{normalize_role(b.get('role', ''))} {normalize_speed(b.get('text', ''))}"
            )
            sample_pairwise.append(pair_text)

    inspect_tokenizer(fast_tokenizer, sample_structured + sample_pairwise)

    print(f"Saved tokenizer to: {output_dir}")
    print(f"Raw tokenizer JSON: {raw_json_path}")
    print(f"Vocab size (actual): {fast_tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
