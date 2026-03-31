import google.generativeai as genai
import json
import os
import time

genai.configure(api_key="AIzaSyBeBQhA8JTyaV224AfJax6sQNLKBj16EF0")

# 2. Configurar o Modelo
# Recomendado o uso do gemini-2.5-flash para tarefas rápidas de geração de dados
# ou gemini-2.5-pro se precisares de raciocínio ainda mais complexo
model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

# 3. Preparar o Prompt
# Usamos few-shot prompting com a estrutura do teu dataset para garantir o formato correto.
PROMPT = """
You are an expert synthetic data generator for NLP tasks, specializing in the insurance sector and fraud detection. 
Your task is to generate a dataset of 10 synthetic car accident insurance claims set in Portugal.

CONTEXT & RULES:
1. Language: All output, including the statements, MUST be strictly in English. However, the locations, culture, and context must reflect Portugal (use Portuguese cities, streets, and realistic local names).
2. Metrics: Use the metric system strictly (km/h, meters). DO NOT use mph.
3. Exclusions: Do not generate cases involving vehicle theft.
4. Fraud Types: Incorporate staged accidents (e.g., swoop and squat, wave-in, phantom vehicles) and exaggerated claims (soft fraud).
5. Coherence: The "statements" must perfectly align with the "ground_truth_label".
6. VISUAL DAMAGE CONSTRAINTS (CRITICAL LOGIC): 
Our Computer Vision model only detects the PRESENCE of these 5 classes: 'crack', 'dent', 'glass shatter', 'lamp broken', and 'scratch'. It DOES NOT detect the severity or size of the damage (e.g., it cannot tell a small scratch from a huge scratch).
- Therefore, DO NOT base fraud on the size of the damage. 
- FRAUD DYNAMICS: For 'soft_fraud_exaggeration' or 'hard_fraud_staged', the fraud must be based on CLASS MISMATCH or PHYSICS MISMATCH between the statements and the 'detected_damages' array. 
  * Class Mismatch: The driver claims a 'lamp broken' or 'glass shatter' in their statement, but the 'detected_damages' array shows those damages do not exist (e.g., containing only a 'dent' or 'scratch').
  * Physics Mismatch: The driver claims a minor parking 'scratch', but also tries to claim a massive 'glass shatter' that makes no physical sense for that type of accident.
- For 'genuine_accident', the damage classes claimed by the driver must perfectly match the classes listed in the 'detected_damages' array. No internal mechanical failures should be mentioned.

PERSONAS FOR STATEMENTS (Apply these psychological profiles to the English text):
Group 1: Genuine Claims (Baseline)
- The Anxious Youth (20-25 y/o): Informal, anxious about the cost and the confusing process. Uses filler words (e.g., "like", "you know", "dude/man" as an equivalent to the Portuguese "pá").
- The Pragmatic Professional (40-50 y/o): Formal, direct, dry. Uses technical terminology and structures their statement almost like a police report.
- The Verbose Senior (65+ y/o): Low digital literacy. Writes like they are telling a story, including irrelevant details about their family, where they were going, or the weather. Bad punctuation.

Group 2: Fraudulent Claims (Anomalies)
- The Defensive Scriptwriter (Staged/Alibi): Text is too linear, technical, and artificial. They justify their innocence before being asked (e.g., explicitly stating they hadn't been drinking, or detailing their exact speed defensively).
- The Evasive Opportunist (Pre-existing damage): Short, vague texts. Avoids giving exact times or locations that might have cameras. Focuses heavily on the vehicle damage and getting paid, rather than the accident dynamics.
- The Impatient Aggressor (Intimidation): Uses ALL CAPS frequently. Threatens to call lawyers or use the "Complaints Book" (Livro de Reclamações). Tries to rush the claims process to avoid detailed analysis.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json or any introductory text.
Each object must strictly follow this schema:
{
  "claim_id": "PT-XXX-2026-[UNIQUE_NUM]",
  "location": "[Realistic location in Portugal]",
  "incident_type": "[e.g., Rear-end collision, Side-swipe, Intersection collision, Pedestrian collision]",
  "ground_truth_label": "[Choose one, keep the dataset balanced: genuine_accident, soft_fraud_exaggeration, hard_fraud_staged, hard_fraud_phantom_vehicle]",
  "detected_damages": ["[Array containing ONLY a selection of these exact strings representing the ground truth physical damage found: 'crack', 'dent', 'glass shatter', 'lamp broken', 'scratch']"],
  "fraud_indicators": ["[List of logical red flags based on the discrepancy between statements and detected damages, e.g., 'Claimed glass shatter but it is missing from detected_damages'. Leave empty [] if genuine_accident]"],
  "statements": [
    {
      "role": "[insured_driver, third_party_driver, or impartial_witness]",
      "vehicle": "[Vehicle model/brand, or 'none' if witness]",
      "text": "[The statement in English, reflecting the assigned Persona]"
    }
  ]
}
"""

def gerar_dados(num_pedidos=1):
    novos_casos = []
    print("A iniciar as chamadas ao Google AI Studio (Gemini)...")
    
    for i in range(num_pedidos):
        print(f"Pedido {i+1} de {num_pedidos}...")
        try:
            # Fazer a chamada à API
            response = model.generate_content(PROMPT)
            
            # Interpretar o JSON (o response_mime_type garante que ele tenta devolver JSON, mas validamos na mesma)
            dados = json.loads(response.text)
            
            if isinstance(dados, list):
                novos_casos.extend(dados)
                print(f" -> Sucesso: {len(dados)} novos claims gerados.")
            elif isinstance(dados, dict):
                # Caso ele devolva um dicionário único por algum motivo
                novos_casos.append(dados)
                print(" -> Sucesso: 1 novo claim gerado.")
                
            # É boa prática adicionar uma pausa para evitar bater nos rate limits da versão gratuita
            time.sleep(20)
            
        except json.JSONDecodeError as e:
            print("Erro: A resposta não é um JSON válido.")
            print("Resposta bruta:", response.text)
        except Exception as e:
            print(f"Erro durante a geração: {e}")
            
    return novos_casos

def guardar_json(dados_novos, nome_ficheiro="dataset_sintetico_gemini.json"):
    if not dados_novos:
        print("Não há dados novos para guardar.")
        return
        
    dados_existentes = []
    
    if os.path.exists(nome_ficheiro):
        try:
            with open(nome_ficheiro, 'r', encoding='utf-8') as f:
                dados_existentes = json.load(f)
                print(f" -> Ficheiro encontrado com {len(dados_existentes)} registos antigos.")
        except json.JSONDecodeError:
            print(" -> Aviso: O ficheiro existente estava corrompido ou vazio. Vai ser reescrito.")
            dados_existentes = []

    dados_existentes.extend(dados_novos)
        
    with open(nome_ficheiro, 'w', encoding='utf-8') as f:
        json.dump(dados_existentes, f, indent=4, ensure_ascii=False)
        
    print(f"\nGravação concluída: O ficheiro tem agora um TOTAL de {len(dados_existentes)} registos.")

if __name__ == "__main__":
    casos = gerar_dados(num_pedidos=10)
    guardar_json(casos)
