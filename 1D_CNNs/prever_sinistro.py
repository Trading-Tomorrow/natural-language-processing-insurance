import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parâmetros
MODEL_PATH = 'modelo_fraude_final.keras'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_LEN = 400

# Cache em memória
_modelo_carregado = None
_tokenizer_carregado = None

def inicilizar_sistema():
    """Carrega o modelo Keras treinado e o Tokenizer para inferência."""
    global _modelo_carregado, _tokenizer_carregado
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError("Modelo ou Tokenizer não encontrados.")
        
    with open(TOKENIZER_PATH, 'rb') as f:
        _tokenizer_carregado = pickle.load(f)
        
    _modelo_carregado = load_model(MODEL_PATH)

def prever_fraude(texto_sinistro: str):
    """Retorna o veredito (Fraude vs Genuíno) e a confiança do modelo."""
    if _modelo_carregado is None or _tokenizer_carregado is None:
        inicilizar_sistema()
        
    # Passagem do texto pela mesma pipeline de tokens e padding usada no treino
    sequencia = _tokenizer_carregado.texts_to_sequences([texto_sinistro])
    sequencia_padded = pad_sequences(sequencia, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Previsão pela Sigmoid (Output de 0 a 1)
    probabilidade = _modelo_carregado.predict(sequencia_padded, verbose=0)[0][0]
    
    if probabilidade > 0.5:
        return "⚠️ ALERTA DE FRAUDE", probabilidade * 100, probabilidade
    else:
        return "✅ ACIDENTE GENUÍNO", (1.0 - probabilidade) * 100, probabilidade

if __name__ == "__main__":
    caso_verdadeiro = (
        "detected_damages: shattered right headlight, dented front right fender, scraped bumper | "
        "insured_driver: I was proceeding through the intersection on a green light at around 30 mph. "
        "The other driver attempted to make a left turn across my lane without yielding, and I struck the passenger side of his vehicle. | "
        "third_party_driver: I thought I had enough time to make the turn before the light turned red, but I misjudged the distance. I didn't see him coming. | "
        "insurance_adjuster: Scene photos, point of impact, and debris patterns are completely consistent with the statements. Third-party admits fault."
    )
    
    caso_falso = (
        "detected_damages: minor paint transfer on rear bumper, no structural damage | "
        "insured_driver: I was stopped at a red light when I was violently rear-ended by a massive truck. The impact was horrific and threw my car forward. "
        "I immediately felt severe, debilitating pain in my lower back and neck, radiating down both legs. I require a full MRI, weeks of therapy, and maximum compensation for emotional distress. | "
        "third_party_driver: We were in stop-and-go traffic. My foot slipped off the brake and I tapped his bumper at maybe 2 or 3 mph. It was barely a nudge. | "
        "insurance_adjuster: Insured's vehicle shows only negligible superficial scratches. Claimed injuries and narrative of a 'horrific impact' are completely disproportionate to the physical evidence."
    )
    caso_fraude_contradicao = "detected_damages: minor paint scratch on the rear bumper | insured_driver: The impact was completely devastating! My neck violently snapped back, my entire transmission is ruined, and I couldn't even walk out of the car. I need a brand new car and maximum medical compensation! | impartial_witness: It was just a tiny bump at a red light. Both drivers got out walking, checked the cars, and seemed completely fine."

    caso_genuino_grave = "detected_damages: completely crushed front bumper, deployed airbags, shattered windshield | insured_driver: The road was heavily flooded from the storm. I tried to brake when the traffic stopped, but my car hydroplaned and I rear-ended the SUV in front of me. My chest hurts from the airbag. | third_party_driver: I was stopped in traffic and suddenly felt a massive impact from behind. | insurance_adjuster: Damages are completely consistent with a high-speed rear-end collision on a wet surface."

    caso_fraude_fantasma = "detected_damages: deep dent on the right side panel | insured_driver: I was driving alone at 3 AM on a dark isolated road. Suddenly, a huge black truck with no license plates and no lights crossed into my lane. I swerved into a tree to save my life! The truck vanished into the night. I have no witnesses, but it totally wasn't my fault and you have to pay for the tree and my car!"

    # Atualiza o teu ciclo for para testar todos:
    lista_testes = [
        ("Genuíno Simples", caso_verdadeiro), 
        ("Fantasma Básico", caso_falso),
        ("Fraude por Contradição", caso_fraude_contradicao),
        ("Genuíno Grave", caso_genuino_grave),
        ("Fraude Fantasma Extrema", caso_fraude_fantasma)
    ]
    
    print("\n" + "="*50)
    print("INÍCIO DOS TESTES DA 1D CNN")
    print("="*50)
    
    for nome, texto in lista_testes:
        v, p, prob_crua = prever_fraude(texto)
        print(f"Cenário: {nome}")
        print(f"-> VEREDITO: {v} ({p:.2f}%)")
        print(f"-> Probabilidade Sigmoid: {prob_crua:.4f}")
        print("-" * 50)
