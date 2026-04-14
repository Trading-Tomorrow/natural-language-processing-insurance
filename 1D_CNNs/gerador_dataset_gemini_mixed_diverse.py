import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import google.generativeai as genai
from dotenv import load_dotenv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-3.1-flash-lite-preview")
TARGET_CASES = 1000
CLAIMS_PER_CALL = 10
SLEEP_SECONDS = 2.0
TEMPERATURE = 1.2
TOP_P = 0.95
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "dataset_sintetico_gemini_mixed_diverse.json")

ALLOWED_LABELS = (
    "genuine_accident",
    "soft_fraud_exaggeration",
    "hard_fraud_staged",
    "hard_fraud_phantom_vehicle",
)
ALLOWED_DAMAGE_CLASSES = (
    "crack",
    "dent",
    "glass shatter",
    "lamp broken",
    "scratch",
)
ALLOWED_STATEMENT_ROLES = (
    "insured_driver",
    "third_party_driver",
    "impartial_witness",
)
REQUIRED_CLAIM_KEYS = (
    "claim_id",
    "location",
    "incident_type",
    "ground_truth_label",
    "detected_damages",
    "fraud_indicators",
    "statements",
)
MIN_INSURED_DRIVER_WORDS = 35
MIN_OTHER_ROLE_WORDS = 20
MIN_TOTAL_WORDS_PER_CLAIM = 80
MIN_STATEMENTS_PER_CLAIM = 2
MAX_STATEMENTS_PER_CLAIM = 3
SIMILARITY_THRESHOLD = 0.82
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
CLAIM_ID_SUFFIX_RE = re.compile(r"(\d+)$")
LABEL_CYCLE = (
    "genuine_accident",
    "soft_fraud_exaggeration",
    "hard_fraud_staged",
    "hard_fraud_phantom_vehicle",
)


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def tokenize_for_similarity(text: str) -> List[str]:
    return re.findall(r"[a-z0-9<>]+", normalize_space(text).lower())


def build_shingles(text: str, size: int = 3) -> Set[str]:
    tokens = tokenize_for_similarity(text)
    if not tokens:
        return set()
    if len(tokens) < size:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return intersection / union


def build_claim_id_batch(start_number: int, batch_size: int) -> List[str]:
    return [
        f"PT-MIXD-2026-{claim_number:06d}"
        for claim_number in range(start_number, start_number + batch_size)
    ]


def build_batch_specs(start_number: int, batch_size: int) -> List[Dict[str, str]]:
    claim_ids = build_claim_id_batch(start_number, batch_size)
    return [
        {
            "claim_id": claim_id,
            "ground_truth_label": LABEL_CYCLE[(start_number - 1 + offset) % len(LABEL_CYCLE)],
        }
        for offset, claim_id in enumerate(claim_ids)
    ]


def build_prompt(batch_specs: Sequence[Dict[str, str]]) -> str:
    batch_size = len(batch_specs)
    batch_specs_json = json.dumps(list(batch_specs), ensure_ascii=False)
    damages_json = json.dumps(list(ALLOWED_DAMAGE_CLASSES), ensure_ascii=False)
    roles_json = json.dumps(list(ALLOWED_STATEMENT_ROLES), ensure_ascii=False)

    return f"""
You are an expert synthetic data generator for NLP tasks in the insurance sector.
Generate exactly {batch_size} synthetic car accident insurance claims set in Portugal.

CRITICAL OUTPUT DISCIPLINE:
1. Return ONLY a valid JSON array.
2. Do not add markdown, explanations, or commentary.
3. Use these exact claim_id and label pairs exactly once each. Do not invent extra IDs or change labels:
{batch_specs_json}

GLOBAL RULES:
1. Language: Every statement must be in English. The setting must be Portugal, with realistic Portuguese locations, roundabouts, roads, avenues, neighborhoods, weather, and driving context.
2. Use metric units only. Never use mph.
3. Do not generate theft, vandalism, fire, or internal mechanical failure cases.
4. Use only these damage classes in detected_damages: {damages_json}
5. Use only these statement roles: {roles_json}
6. Every claim must have 2 or 3 statements.
7. Every claim must include at least one insured_driver statement and at least one non-insured statement.
8. If the role is impartial_witness, vehicle must be "none".

VISUAL DAMAGE LOGIC:
Our Computer Vision model detects only the PRESENCE of these 5 classes and cannot detect severity.
- For genuine_accident claims, the damage classes described in the statements must align with detected_damages.
- For fraudulent claims, fraud must come from class mismatch or physics mismatch, not from "big damage" language alone.
- A class mismatch means the narrative claims a class that is missing from detected_damages.
- A physics mismatch means the narrative describes an accident dynamic that does not make sense for the detected_damages or the incident type.

LABEL-SPECIFIC RULES:
- genuine_accident: coherent, honest, consistent narratives.
- soft_fraud_exaggeration: opportunistic inflation of what happened or what was damaged.
- hard_fraud_staged: coordinated or artificial narrative, staged dynamics, suspiciously over-controlled account.
- hard_fraud_phantom_vehicle: claims involving a supposed vehicle that cannot be independently grounded by the evidence.

WRITING QUALITY AND DIVERSITY:
1. The dataset must be lexically diverse. Avoid template-like repetition.
2. Do NOT keep repeating the same opening sentence pattern such as "I was driving", "I was stopped", "The other car", or "I was going" across the batch.
3. Vary narrative structure aggressively:
- time-first
- location-first
- action-first
- damage-first
- witness-aftermath perspective
- emotional reaction first
- report-like chronology
- story-like chronology
4. Vary syntax, sentence length, punctuation, rhythm, and discourse markers.
5. Use different cities, streets, weather conditions, maneuvers, vehicle models, and claim dynamics.
6. Do not recycle nearly identical phrasings from one claim to another.
7. Use richer wording and more paraphrase diversity than a standard synthetic dataset.
8. Insured_driver statements should usually be 45 to 110 words.
9. Other roles should usually be 25 to 80 words.
10. Statements must sound like different people, not the same author with different labels.

PERSONAS:
Genuine personas:
- Anxious Youth: informal, worried, filler words, confused about process or cost.
- Pragmatic Professional: dry, direct, report-like, factual.
- Verbose Senior: storytelling, side details, messy punctuation, long-winded.

Fraud personas:
- Defensive Scriptwriter: too tidy, too prepared, self-justifying before being asked.
- Evasive Opportunist: vague about time or camera coverage, focused on damage and payout.
- Impatient Aggressor: pushy, rushing the process, sometimes capitalizes or threatens escalation.

OUTPUT SCHEMA:
[
  {{
    "claim_id": "PT-MIXD-2026-000001",
    "location": "Avenida da Liberdade, Lisbon",
    "incident_type": "Rear-end collision",
    "ground_truth_label": "genuine_accident",
    "detected_damages": ["dent", "scratch"],
    "fraud_indicators": [],
    "statements": [
      {{
        "role": "insured_driver",
        "vehicle": "Renault Clio",
        "text": "English statement."
      }},
      {{
        "role": "third_party_driver",
        "vehicle": "Peugeot 208",
        "text": "English statement."
      }}
    ]
  }}
]
""".strip()


def create_model() -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY nao encontrada. Defina GEMINI_API_KEY no ficheiro .env na raiz do projeto."
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
    )


def parse_response_text(response_text: str) -> List[Any]:
    raw_text = normalize_space(response_text)
    if not raw_text:
        raise ValueError("Resposta vazia do Gemini.")

    cleaned_text = JSON_FENCE_RE.sub("", raw_text).strip()
    parsed = json.loads(cleaned_text)

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        claims = parsed.get("claims")
        if isinstance(claims, list):
            return claims
        return [parsed]

    raise ValueError("A resposta JSON nao tem um array nem um objeto valido.")


def build_claim_fingerprint(claim: Dict[str, Any]) -> str:
    damage_key = "|".join(sorted(normalize_space(item).lower() for item in claim["detected_damages"]))
    statement_key_parts = []
    for statement in claim["statements"]:
        statement_key_parts.append(
            "::".join(
                [
                    normalize_space(statement["role"]).lower(),
                    normalize_space(statement["vehicle"]).lower(),
                    normalize_space(statement["text"]).lower(),
                ]
            )
        )

    key = "||".join(
        [
            normalize_space(claim["location"]).lower(),
            normalize_space(claim["incident_type"]).lower(),
            normalize_space(claim["ground_truth_label"]).lower(),
            damage_key,
            "||".join(statement_key_parts),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def next_claim_number(existing_claims: Sequence[Dict[str, Any]]) -> int:
    max_number = 0
    for claim in existing_claims:
        claim_id = normalize_space(claim.get("claim_id", ""))
        match = CLAIM_ID_SUFFIX_RE.search(claim_id)
        if not match:
            continue
        max_number = max(max_number, int(match.group(1)))
    return max_number + 1


def validate_claim(
    raw_claim: Any,
    expected_labels: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    if not isinstance(raw_claim, dict):
        return False, "claim is not an object", None

    missing_keys = [key for key in REQUIRED_CLAIM_KEYS if key not in raw_claim]
    if missing_keys:
        return False, f"missing keys: {', '.join(missing_keys)}", None

    claim_id = normalize_space(raw_claim.get("claim_id"))
    if not claim_id:
        return False, "claim_id is empty", None

    label = normalize_space(raw_claim.get("ground_truth_label"))
    if label not in ALLOWED_LABELS:
        return False, f"invalid label: {label}", None
    if expected_labels is not None and expected_labels.get(claim_id) != label:
        return False, f"unexpected label for {claim_id}: {label}", None

    location = normalize_space(raw_claim.get("location"))
    incident_type = normalize_space(raw_claim.get("incident_type"))
    if not location:
        return False, "location is empty", None
    if not incident_type:
        return False, "incident_type is empty", None

    detected_damages = raw_claim.get("detected_damages")
    if not isinstance(detected_damages, list) or not detected_damages:
        return False, "detected_damages must be a non-empty list", None

    normalized_damages: List[str] = []
    seen_damages: Set[str] = set()
    for damage in detected_damages:
        normalized_damage = normalize_space(damage).lower()
        if normalized_damage not in ALLOWED_DAMAGE_CLASSES:
            return False, f"unsupported damage: {damage}", None
        if normalized_damage not in seen_damages:
            normalized_damages.append(normalized_damage)
            seen_damages.add(normalized_damage)

    fraud_indicators = raw_claim.get("fraud_indicators")
    if not isinstance(fraud_indicators, list):
        return False, "fraud_indicators must be a list", None
    cleaned_indicators = [normalize_space(item) for item in fraud_indicators if normalize_space(item)]
    if label == "genuine_accident" and cleaned_indicators:
        return False, "genuine_accident must have empty fraud_indicators", None
    if label != "genuine_accident" and not cleaned_indicators:
        return False, "fraud labels must include fraud_indicators", None

    statements = raw_claim.get("statements")
    if not isinstance(statements, list):
        return False, "statements must be a list", None
    if len(statements) < MIN_STATEMENTS_PER_CLAIM or len(statements) > MAX_STATEMENTS_PER_CLAIM:
        return False, "claims must contain 2 or 3 statements", None

    cleaned_statements: List[Dict[str, str]] = []
    total_words = 0
    has_insured_driver = False
    has_other_role = False

    for statement in statements:
        if not isinstance(statement, dict):
            continue

        role = normalize_space(statement.get("role")).lower()
        vehicle = normalize_space(statement.get("vehicle"))
        text = normalize_space(statement.get("text"))

        if not role or not vehicle or not text:
            continue
        if role not in ALLOWED_STATEMENT_ROLES:
            return False, f"unsupported role: {role}", None

        words = count_words(text)
        if role == "insured_driver":
            has_insured_driver = True
            if words < MIN_INSURED_DRIVER_WORDS:
                return False, f"insured_driver statement too short: {words} words", None
        else:
            has_other_role = True
            if words < MIN_OTHER_ROLE_WORDS:
                return False, f"{role} statement too short: {words} words", None

        cleaned_statements.append(
            {
                "role": role,
                "vehicle": vehicle,
                "text": text,
            }
        )
        total_words += words

    if len(cleaned_statements) < MIN_STATEMENTS_PER_CLAIM:
        return False, "not enough valid statements", None
    if not has_insured_driver:
        return False, "missing insured_driver statement", None
    if not has_other_role:
        return False, "missing non-insured statement", None
    if total_words < MIN_TOTAL_WORDS_PER_CLAIM:
        return False, f"claim total text too short: {total_words} words", None

    cleaned_claim = {
        "claim_id": claim_id,
        "location": location,
        "incident_type": incident_type,
        "ground_truth_label": label,
        "detected_damages": normalized_damages,
        "fraud_indicators": cleaned_indicators,
        "statements": cleaned_statements,
    }
    return True, "ok", cleaned_claim


def load_existing_claims(output_file: str) -> Tuple[List[Dict[str, Any]], Set[str], Set[str], Dict[str, List[Set[str]]]]:
    if not os.path.exists(output_file):
        return [], set(), set(), {role: [] for role in ALLOWED_STATEMENT_ROLES}

    with open(output_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"O ficheiro existente em {output_file} nao contem uma lista JSON.")

    valid_claims: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    seen_fingerprints: Set[str] = set()
    statement_shingles_by_role: Dict[str, List[Set[str]]] = {role: [] for role in ALLOWED_STATEMENT_ROLES}

    for raw_claim in payload:
        is_valid, _, cleaned_claim = validate_claim(raw_claim)
        if not is_valid or cleaned_claim is None:
            continue

        claim_id = cleaned_claim["claim_id"]
        fingerprint = build_claim_fingerprint(cleaned_claim)
        if claim_id in seen_ids or fingerprint in seen_fingerprints:
            continue

        seen_ids.add(claim_id)
        seen_fingerprints.add(fingerprint)
        valid_claims.append(cleaned_claim)
        for statement in cleaned_claim["statements"]:
            statement_shingles_by_role[statement["role"]].append(build_shingles(statement["text"]))

    return valid_claims, seen_ids, seen_fingerprints, statement_shingles_by_role


def save_claims(claims: Sequence[Dict[str, Any]], output_file: str) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(list(claims), handle, indent=4, ensure_ascii=False)


def find_similarity_issue(
    claim: Dict[str, Any],
    statement_shingles_by_role: Dict[str, List[Set[str]]],
) -> Optional[str]:
    for statement in claim["statements"]:
        role = statement["role"]
        shingles = build_shingles(statement["text"])
        if not shingles:
            continue
        for existing_shingles in statement_shingles_by_role.get(role, []):
            similarity = jaccard_similarity(shingles, existing_shingles)
            if similarity >= SIMILARITY_THRESHOLD:
                return f"{role} statement too similar to previous data ({similarity:.2f})"
    return None


def register_claim_statements(
    claim: Dict[str, Any],
    statement_shingles_by_role: Dict[str, List[Set[str]]],
) -> None:
    for statement in claim["statements"]:
        statement_shingles_by_role[statement["role"]].append(build_shingles(statement["text"]))


def generate_dataset(
    target_cases: int = TARGET_CASES,
    claims_per_call: int = CLAIMS_PER_CALL,
    sleep_seconds: float = SLEEP_SECONDS,
    output_file: str = OUTPUT_FILE,
    max_calls: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if target_cases <= 0:
        raise ValueError("target_cases deve ser maior que zero.")
    if claims_per_call <= 0:
        raise ValueError("claims_per_call deve ser maior que zero.")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds nao pode ser negativo.")

    claims, seen_ids, seen_fingerprints, statement_shingles_by_role = load_existing_claims(output_file)
    model = create_model()

    accepted_total = len(claims)
    rejected_total = 0
    duplicate_total = 0
    missing_total = 0
    similar_total = 0
    next_number = next_claim_number(claims)

    print(f"Modelo: {MODEL_NAME}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Top-p: {TOP_P}")
    print(f"Throttle: {sleep_seconds:.1f}s")
    print(f"Ficheiro de saida: {output_file}")
    print(f"Claims validos ja existentes: {accepted_total}")
    print(f"Objetivo final: {target_cases} claims mistos")

    if accepted_total >= target_cases:
        print("O ficheiro ja contem casos suficientes. Nao ha nada para gerar.")
        return claims[:target_cases]

    call_number = 0

    while accepted_total < target_cases:
        if max_calls is not None and call_number >= max_calls:
            print("Interrompido porque max_calls foi atingido.")
            break

        call_number += 1
        remaining = target_cases - accepted_total
        batch_size = min(claims_per_call, remaining)
        batch_specs = build_batch_specs(next_number, batch_size)
        next_number += batch_size
        expected_labels = {spec["claim_id"]: spec["ground_truth_label"] for spec in batch_specs}

        print(f"\nChamada {call_number}: pedir {batch_size} claims (progresso {accepted_total}/{target_cases})")
        print(f" -> Labels esperadas: {dict(Counter(spec['ground_truth_label'] for spec in batch_specs))}")

        try:
            response = model.generate_content(build_prompt(batch_specs))
            generated_claims = parse_response_text(getattr(response, "text", ""))
        except json.JSONDecodeError as error:
            missing_total += batch_size
            print(f" -> Resposta sem JSON valido: {error}")
        except Exception as error:
            missing_total += batch_size
            print(f" -> Erro durante a geracao: {error}")
        else:
            batch_accepted: List[Dict[str, Any]] = []
            batch_rejected = 0
            batch_duplicates = 0
            batch_missing = max(0, batch_size - len(generated_claims))
            batch_similar = 0

            for raw_claim in generated_claims:
                is_valid, reason, cleaned_claim = validate_claim(raw_claim, expected_labels=expected_labels)
                if not is_valid or cleaned_claim is None:
                    batch_rejected += 1
                    print(f" -> Claim rejeitado: {reason}")
                    continue

                claim_id = cleaned_claim["claim_id"]
                fingerprint = build_claim_fingerprint(cleaned_claim)
                if claim_id in seen_ids or fingerprint in seen_fingerprints:
                    batch_duplicates += 1
                    continue

                similarity_issue = find_similarity_issue(cleaned_claim, statement_shingles_by_role)
                if similarity_issue is not None:
                    batch_similar += 1
                    print(f" -> Claim rejeitado por repeticao: {similarity_issue}")
                    continue

                seen_ids.add(claim_id)
                seen_fingerprints.add(fingerprint)
                register_claim_statements(cleaned_claim, statement_shingles_by_role)
                batch_accepted.append(cleaned_claim)

                if accepted_total + len(batch_accepted) >= target_cases:
                    break

            if batch_accepted:
                claims.extend(batch_accepted)
                accepted_total = len(claims)
                save_claims(claims, output_file)

            rejected_total += batch_rejected
            duplicate_total += batch_duplicates
            missing_total += batch_missing
            similar_total += batch_similar

            print(
                " -> Batch concluido: "
                f"aceites={len(batch_accepted)} "
                f"rejeitados={batch_rejected} "
                f"duplicados={batch_duplicates} "
                f"repeticao={batch_similar} "
                f"faltantes={batch_missing} "
                f"total={accepted_total}/{target_cases}"
            )

        if accepted_total >= target_cases:
            break

        if sleep_seconds > 0:
            print(f" -> A dormir {sleep_seconds:.1f}s antes da proxima chamada...")
            time.sleep(sleep_seconds)

    print("\nResumo final:")
    print(f" - Aceites: {accepted_total}")
    print(f" - Rejeitados: {rejected_total}")
    print(f" - Duplicados: {duplicate_total}")
    print(f" - Repeticao: {similar_total}")
    print(f" - Faltantes: {missing_total}")
    print(f" - Ficheiro: {output_file}")

    return claims[:target_cases]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera um dataset Gemini misto mais diverso, com menos repeticao lexical."
    )
    parser.add_argument("--target-cases", type=int, default=TARGET_CASES)
    parser.add_argument("--claims-per-call", type=int, default=CLAIMS_PER_CALL)
    parser.add_argument("--sleep-seconds", type=float, default=SLEEP_SECONDS)
    parser.add_argument("--output-file", default=OUTPUT_FILE)
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Limita o numero de chamadas ao modelo. Util para smoke tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate_dataset(
        target_cases=arguments.target_cases,
        claims_per_call=arguments.claims_per_call,
        sleep_seconds=arguments.sleep_seconds,
        output_file=arguments.output_file,
        max_calls=arguments.max_calls,
    )
