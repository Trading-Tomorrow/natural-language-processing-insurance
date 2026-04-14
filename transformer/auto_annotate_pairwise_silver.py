import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import google.generativeai as genai
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_full.jsonl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "pairwise_dataset_silver_teacher.jsonl"
DEFAULT_STATS_PATH = BASE_DIR / "data" / "pairwise_dataset_silver_teacher_stats.json"
DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-3.1-flash-lite-preview")
DEFAULT_PAIRS_PER_CALL = 8
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
ALLOWED_RELATION_LABELS = {"supports", "neutral", "contradicts"}
ALLOWED_INCONSISTENCY_TYPES = {"none", "damage_mismatch", "dynamics_mismatch", "phantom_vehicle"}
PROMPT_VERSION = "pairwise_silver_teacher_v1"


load_dotenv(dotenv_path=ENV_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-annotate pairwise statement pairs into a silver dataset using Gemini.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pairs-per-call", type=int, default=DEFAULT_PAIRS_PER_CALL)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on how many input pairs to annotate.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional start offset inside the input dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare batches and exit without calling Gemini.")
    return parser.parse_args()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at line {line_number} in {path}.")
            records.append(payload)
    return records


def save_jsonl(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_existing_output(output_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not output_path.exists():
        return [], {}
    records = load_jsonl(output_path)
    by_pair_id = {
        normalize_space(record.get("pair_id", "")): record
        for record in records
        if normalize_space(record.get("pair_id", ""))
    }
    return records, by_pair_id


def create_model(model_name: str, temperature: float, top_p: float) -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY nao encontrada. Defina GEMINI_API_KEY no ficheiro .env na raiz do projeto.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": temperature,
            "top_p": top_p,
        },
    )


def build_prompt(batch: Sequence[Dict[str, Any]]) -> str:
    prompt_pairs = []
    for record in batch:
        prompt_pairs.append(
            {
                "pair_id": record["pair_id"],
                "claim_id_a": record.get("claim_id_a", ""),
                "claim_id_b": record.get("claim_id_b", ""),
                "role_a": record.get("role_a", ""),
                "role_b": record.get("role_b", ""),
                "incident_type_a": record.get("incident_type_a", ""),
                "incident_type_b": record.get("incident_type_b", ""),
                "detected_damages_a": record.get("detected_damages_a", []),
                "detected_damages_b": record.get("detected_damages_b", []),
                "text_a": record.get("raw_text_a", record.get("text_a", "")),
                "text_b": record.get("raw_text_b", record.get("text_b", "")),
            }
        )

    payload = json.dumps(prompt_pairs, ensure_ascii=False)
    return f"""
You are a strict pairwise annotation model for contradiction learning in insurance accident narratives.

Task:
For each pair below, decide:
1. relation label: supports, neutral, or contradicts
2. inconsistency_type: none, damage_mismatch, dynamics_mismatch, or phantom_vehicle
3. rationale_short: one short sentence explaining the decision
4. confidence: float between 0.0 and 1.0

Definitions:
- supports: the two statements describe compatible versions of the same event
- neutral: the statements do not clearly support or clearly contradict each other
- contradicts: the statements contain a meaningful factual conflict

Inconsistency rules:
- If relation label is supports or neutral, inconsistency_type must be none.
- If relation label is contradicts, choose exactly one strongest inconsistency type:
  - damage_mismatch: damage claims or visible damage categories do not line up
  - dynamics_mismatch: movement, impact direction, traffic light state, who hit whom, stopped vs moving, lane behavior, or accident mechanics conflict
  - phantom_vehicle: one story depends on a missing or unsupported extra vehicle

Important:
- Prefer neutral over contradicts when the conflict is weak or not explicit.
- Do not use style or tone alone as contradiction.
- Use only the information present in the pair.
- Return ONLY a JSON array.

Input pairs:
{payload}

Output schema:
[
  {{
    "pair_id": "PAIR-000001",
    "label": "contradicts",
    "inconsistency_type": "dynamics_mismatch",
    "rationale_short": "One statement says the insured was stopped and hit from behind, while the other says the insured rolled backward into the other vehicle.",
    "confidence": 0.92
  }}
]
""".strip()


def parse_response_text(response_text: str) -> List[Dict[str, Any]]:
    cleaned = JSON_FENCE_RE.sub("", normalize_space(response_text)).strip()
    if not cleaned:
        raise ValueError("Resposta vazia do Gemini.")
    payload = json.loads(cleaned)
    if isinstance(payload, dict):
        if isinstance(payload.get("annotations"), list):
            payload = payload["annotations"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("A resposta do Gemini nao e uma lista JSON.")
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Todos os itens da resposta devem ser objetos.")
    return payload


def validate_annotation(
    raw_annotation: Dict[str, Any],
    expected_pair_ids: Sequence[str],
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    pair_id = normalize_space(raw_annotation.get("pair_id", ""))
    if pair_id not in expected_pair_ids:
        return False, f"unexpected pair_id: {pair_id}", None

    label = normalize_space(raw_annotation.get("label", "")).lower()
    inconsistency_type = normalize_space(raw_annotation.get("inconsistency_type", "")).lower()
    rationale_short = normalize_space(raw_annotation.get("rationale_short", ""))
    confidence_raw = raw_annotation.get("confidence", None)

    if label not in ALLOWED_RELATION_LABELS:
        return False, f"invalid label: {label}", None
    if inconsistency_type not in ALLOWED_INCONSISTENCY_TYPES:
        return False, f"invalid inconsistency_type: {inconsistency_type}", None
    if label != "contradicts" and inconsistency_type != "none":
        return False, "non-contradictory labels must use inconsistency_type=none", None
    if label == "contradicts" and inconsistency_type == "none":
        return False, "contradicts must not use inconsistency_type=none", None
    if not rationale_short:
        return False, "rationale_short is empty", None

    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        return False, f"invalid confidence: {confidence_raw}", None
    if not 0.0 <= confidence <= 1.0:
        return False, f"confidence out of range: {confidence}", None

    cleaned = {
        "pair_id": pair_id,
        "label": label,
        "inconsistency_type": inconsistency_type,
        "rationale_short": rationale_short,
        "confidence": confidence,
    }
    return True, "ok", cleaned


def merge_teacher_annotation(source_record: Dict[str, Any], annotation: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    merged = dict(source_record)
    merged["weak_label"] = source_record.get("label", "")
    merged["weak_inconsistency_type"] = source_record.get("inconsistency_type", "none")
    merged["label"] = annotation["label"]
    merged["inconsistency_type"] = annotation["inconsistency_type"]
    merged["teacher_rationale_short"] = annotation["rationale_short"]
    merged["teacher_confidence"] = annotation["confidence"]
    merged["label_source"] = "gemini_teacher"
    merged["teacher_model"] = model_name
    merged["teacher_prompt_version"] = PROMPT_VERSION
    return merged


def build_batches(records: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [list(records[index : index + batch_size]) for index in range(0, len(records), batch_size)]


def main() -> None:
    args = parse_args()
    input_records = load_jsonl(args.input_path)
    existing_output_records, existing_by_pair_id = load_existing_output(args.output_path)

    selected_records = input_records[args.start_index :]
    if args.max_pairs is not None:
        selected_records = selected_records[: args.max_pairs]

    pending_records = [
        record for record in selected_records
        if normalize_space(record.get("pair_id", "")) not in existing_by_pair_id
    ]
    pending_batches = build_batches(pending_records, args.pairs_per_call)

    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Stats path: {args.stats_path}")
    print(f"Selected records: {len(selected_records)}")
    print(f"Already annotated: {len(selected_records) - len(pending_records)}")
    print(f"Pending records: {len(pending_records)}")
    print(f"Batches: {len(pending_batches)}")

    if args.dry_run:
        print("Dry run enabled; no Gemini calls executed.")
        return

    model = create_model(args.model_name, args.temperature, args.top_p)
    merged_records = list(existing_output_records)
    accepted = 0
    rejected = 0
    changed_label = 0
    changed_inconsistency = 0

    for batch_index, batch in enumerate(pending_batches, start=1):
        expected_pair_ids = [normalize_space(record.get("pair_id", "")) for record in batch]
        prompt = build_prompt(batch)
        response = model.generate_content(prompt)
        parsed_annotations = parse_response_text(response.text)

        annotations_by_pair_id: Dict[str, Dict[str, Any]] = {}
        for raw_annotation in parsed_annotations:
            is_valid, reason, cleaned_annotation = validate_annotation(raw_annotation, expected_pair_ids)
            if not is_valid or cleaned_annotation is None:
                rejected += 1
                print(f"Rejected annotation in batch {batch_index}: {reason}")
                continue
            annotations_by_pair_id[cleaned_annotation["pair_id"]] = cleaned_annotation

        for source_record in batch:
            pair_id = normalize_space(source_record.get("pair_id", ""))
            cleaned_annotation = annotations_by_pair_id.get(pair_id)
            if cleaned_annotation is None:
                rejected += 1
                print(f"Missing teacher annotation for pair_id {pair_id} in batch {batch_index}")
                continue

            if cleaned_annotation["label"] != source_record.get("label", ""):
                changed_label += 1
            if cleaned_annotation["inconsistency_type"] != source_record.get("inconsistency_type", "none"):
                changed_inconsistency += 1

            merged_records.append(merge_teacher_annotation(source_record, cleaned_annotation, args.model_name))
            accepted += 1

        save_jsonl(merged_records, args.output_path)
        print(
            f"Batch {batch_index}/{len(pending_batches)} | accepted={accepted} rejected={rejected} "
            f"changed_label={changed_label} changed_inconsistency={changed_inconsistency}"
        )
        if batch_index < len(pending_batches):
            time.sleep(args.sleep_seconds)

    label_counts = Counter(record.get("label", "") for record in merged_records)
    inconsistency_counts = Counter(record.get("inconsistency_type", "none") for record in merged_records)
    weak_label_counts = Counter(record.get("weak_label", record.get("label", "")) for record in merged_records)
    weak_inconsistency_counts = Counter(
        record.get("weak_inconsistency_type", record.get("inconsistency_type", "none"))
        for record in merged_records
    )

    stats = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "model_name": args.model_name,
        "prompt_version": PROMPT_VERSION,
        "selected_records": len(selected_records),
        "annotated_records": len(merged_records),
        "accepted": accepted,
        "rejected": rejected,
        "changed_label": changed_label,
        "changed_inconsistency_type": changed_inconsistency,
        "label_counts": dict(label_counts),
        "inconsistency_counts": dict(inconsistency_counts),
        "weak_label_counts": dict(weak_label_counts),
        "weak_inconsistency_counts": dict(weak_inconsistency_counts),
        "pairs_per_call": args.pairs_per_call,
        "sleep_seconds": args.sleep_seconds,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    save_json(stats, args.stats_path)
    print(f"Saved stats to: {args.stats_path}")


if __name__ == "__main__":
    main()
