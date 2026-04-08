import argparse
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import google.generativeai as genai
from dotenv import load_dotenv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-3.1-flash-lite-preview")
TARGET_CASES = 2000
CLAIMS_PER_CALL = 10
SLEEP_SECONDS = 8
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "dataset_sintetico_gemini_good_only.json")

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
MIN_INSURED_DRIVER_WORDS = 45
MIN_OTHER_ROLE_WORDS = 25
MIN_SINGLE_STATEMENT_CLAIM_WORDS = 70
MIN_TOTAL_WORDS_PER_CLAIM = 85
REQUIRED_CLAIM_KEYS = (
    "claim_id",
    "location",
    "incident_type",
    "ground_truth_label",
    "detected_damages",
    "fraud_indicators",
    "statements",
)
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
CLAIM_ID_SUFFIX_RE = re.compile(r"(\d+)$")


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def build_claim_id_batch(start_number: int, batch_size: int) -> List[str]:
    return [
        f"PT-GOOD-2026-{claim_number:06d}"
        for claim_number in range(start_number, start_number + batch_size)
    ]


def build_prompt(claim_ids: Sequence[str]) -> str:
    claims_requested = len(claim_ids)
    claim_ids_json = json.dumps(list(claim_ids), ensure_ascii=False)
    damages_json = json.dumps(list(ALLOWED_DAMAGE_CLASSES), ensure_ascii=False)
    roles_json = json.dumps(list(ALLOWED_STATEMENT_ROLES), ensure_ascii=False)

    return f"""
You are an expert synthetic data generator for NLP tasks, specializing in the insurance sector and fraud detection.
Your task is to generate a dataset of exactly {claims_requested} synthetic car accident insurance claims set in Portugal.
This specific batch must contain ONLY genuine claims. Generate no fraud cases.

CONTEXT & RULES:
1. Language: All output, including the statements, MUST be strictly in English. However, the locations, culture, and context must reflect Portugal using realistic Portuguese cities, roads, roundabouts, avenues, landmarks, and local driving situations.
2. Metrics: Use the metric system strictly (km/h, meters). DO NOT use mph.
3. Exclusions: Do not generate cases involving vehicle theft, fire, vandalism, internal mechanical failures, or unrelated historical damage.
4. Label restriction: Every single object must have "ground_truth_label": "genuine_accident". Do not generate soft fraud, staged accidents, phantom vehicles, exaggeration, or any suspicious/anomalous claim style.
5. Coherence: The "statements" must perfectly align with the "ground_truth_label". These are honest claims, so the narrative can contain stress, confusion, or imperfect memory about timing, but it must not contain deception, contradiction, invented impact dynamics, or damage inflation.
6. VISUAL DAMAGE CONSTRAINTS (CRITICAL LOGIC):
Our Computer Vision model only detects the PRESENCE of these 5 classes and nothing else: {damages_json}
- It DOES NOT detect severity or size of the damage.
- Therefore, do not introduce extra hidden damage categories or infer severity from the class.
- For genuine_accident claims, the damage classes described in the statements must match the classes listed in "detected_damages".
- If a claim has "dent" and "scratch", the statements may describe a dent and a scratch, but must not also claim broken glass or a broken lamp unless those classes are present in "detected_damages".
- Do not mention engine issues, suspension damage, alignment problems, airbags, whiplash, total loss, or any unsupported non-visual damage.
7. Statement roles: Use only these roles: {roles_json}
- If the role is "impartial_witness", set "vehicle" to "none".
- Use 1 to 3 statements per claim.
 - Every claim must include at least one "insured_driver" statement.
8. Personas for statements: Use only genuine baseline personas and vary them naturally:
- The Anxious Youth (20-25 y/o): informal, anxious, worried about costs and the process, may use fillers like "like", "you know", "dude", or "man".
- The Pragmatic Professional (40-50 y/o): formal, direct, dry, factual, sometimes report-like.
- The Verbose Senior (65+ y/o): long-winded, storytelling style, slightly messy punctuation, mentions side details like weather, errands, or family context.
9. LENGTH & WRITING QUALITY (IMPORTANT):
- Do not write short or minimal statements.
- Write natural multi-sentence narratives with useful context, not one-line summaries.
- The insured_driver statement should normally be around 55 to 110 words.
- third_party_driver and impartial_witness statements should normally be around 30 to 80 words.
- If a claim has only one statement, that single statement should normally be around 80 to 140 words.
- Most claims should feel similar in richness and detail to a real written insurance account, not a caption.
10. Diversity: Vary city, road type, maneuver, time of day, weather, traffic density, vehicle models, statement combinations, and incident type. Use realistic genuine scenarios such as rear-end collisions, side-swipes, parking scrapes, intersection collisions, roundabout contact, lane-change contact, or low-speed urban impacts.
11. Exact IDs: Use these exact claim_id values exactly once each and do not invent other IDs: {claim_ids_json}
12. Output discipline: Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json or any introductory text.
13. Schema discipline: Do not omit any field. Every object must explicitly include claim_id, location, incident_type, ground_truth_label, detected_damages, fraud_indicators, and statements.
14. fraud_indicators must always be [] for every object.

OUTPUT SCHEMA:
[
  {{
    "claim_id": "PT-GOOD-2026-000001",
    "location": "Avenida da Liberdade, Lisbon",
    "incident_type": "Rear-end collision",
    "ground_truth_label": "genuine_accident",
    "detected_damages": ["dent", "scratch"],
    "fraud_indicators": [],
    "statements": [
      {{
        "role": "insured_driver",
        "vehicle": "Renault Clio",
        "text": "English statement matching the accident and the detected_damages."
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
    expected_ids: Optional[Set[str]] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    if not isinstance(raw_claim, dict):
        return False, "claim is not an object", None

    missing_keys = [key for key in REQUIRED_CLAIM_KEYS if key not in raw_claim]
    if missing_keys:
        return False, f"missing keys: {', '.join(missing_keys)}", None

    claim_id = normalize_space(raw_claim.get("claim_id"))
    if not claim_id:
        return False, "claim_id is empty", None
    if expected_ids is not None and claim_id not in expected_ids:
        return False, f"unexpected claim_id: {claim_id}", None

    location = normalize_space(raw_claim.get("location"))
    incident_type = normalize_space(raw_claim.get("incident_type"))
    if not location:
        return False, "location is empty", None
    if not incident_type:
        return False, "incident_type is empty", None

    label = normalize_space(raw_claim.get("ground_truth_label"))
    if label != "genuine_accident":
        return False, f"invalid label: {label}", None

    fraud_indicators = raw_claim.get("fraud_indicators")
    if not isinstance(fraud_indicators, list) or fraud_indicators:
        return False, "fraud_indicators must be []", None

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

    statements = raw_claim.get("statements")
    if not isinstance(statements, list) or not statements:
        return False, "statements must be a non-empty list", None

    cleaned_statements: List[Dict[str, str]] = []
    has_insured_driver = False
    total_words = 0
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
        elif words < MIN_OTHER_ROLE_WORDS:
            return False, f"{role} statement too short: {words} words", None

        cleaned_statements.append(
            {
                "role": role,
                "vehicle": vehicle,
                "text": text,
            }
        )
        total_words += words

    if not cleaned_statements:
        return False, "no valid statements found", None
    if not has_insured_driver:
        return False, "missing insured_driver statement", None
    if len(cleaned_statements) == 1 and total_words < MIN_SINGLE_STATEMENT_CLAIM_WORDS:
        return False, f"single-statement claim too short: {total_words} words", None
    if total_words < MIN_TOTAL_WORDS_PER_CLAIM:
        return False, f"claim total text too short: {total_words} words", None

    cleaned_claim = {
        "claim_id": claim_id,
        "location": location,
        "incident_type": incident_type,
        "ground_truth_label": "genuine_accident",
        "detected_damages": normalized_damages,
        "fraud_indicators": [],
        "statements": cleaned_statements,
    }
    return True, "ok", cleaned_claim


def load_existing_claims(output_file: str) -> Tuple[List[Dict[str, Any]], Set[str], Set[str]]:
    if not os.path.exists(output_file):
        return [], set(), set()

    with open(output_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"O ficheiro existente em {output_file} nao contem uma lista JSON.")

    valid_claims: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    seen_fingerprints: Set[str] = set()

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

    return valid_claims, seen_ids, seen_fingerprints


def save_claims(claims: Sequence[Dict[str, Any]], output_file: str) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(list(claims), handle, indent=4, ensure_ascii=False)


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

    claims, seen_ids, seen_fingerprints = load_existing_claims(output_file)
    model = create_model()

    accepted_total = len(claims)
    rejected_total = 0
    duplicate_total = 0
    missing_total = 0
    next_number = next_claim_number(claims)

    print(f"Modelo: {MODEL_NAME}")
    print(f"Ficheiro de saida: {output_file}")
    print(f"Claims validos ja existentes: {accepted_total}")
    print(f"Objetivo final: {target_cases} claims genuinos")

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
        expected_batch_ids = build_claim_id_batch(next_number, batch_size)
        next_number += batch_size

        print(
            f"\nChamada {call_number}: pedir {batch_size} claims "
            f"(progresso {accepted_total}/{target_cases})"
        )

        try:
            response = model.generate_content(build_prompt(expected_batch_ids))
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
            expected_ids = set(expected_batch_ids)

            for raw_claim in generated_claims:
                is_valid, reason, cleaned_claim = validate_claim(raw_claim, expected_ids=expected_ids)
                if not is_valid or cleaned_claim is None:
                    batch_rejected += 1
                    print(f" -> Claim rejeitado: {reason}")
                    continue

                claim_id = cleaned_claim["claim_id"]
                fingerprint = build_claim_fingerprint(cleaned_claim)
                if claim_id in seen_ids or fingerprint in seen_fingerprints:
                    batch_duplicates += 1
                    continue

                seen_ids.add(claim_id)
                seen_fingerprints.add(fingerprint)
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

            print(
                " -> Batch concluido: "
                f"aceites={len(batch_accepted)} "
                f"rejeitados={batch_rejected} "
                f"duplicados={batch_duplicates} "
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
    print(f" - Faltantes: {missing_total}")
    print(f" - Ficheiro: {output_file}")

    return claims[:target_cases]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera um dataset Gemini apenas com casos genuine_accident."
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
