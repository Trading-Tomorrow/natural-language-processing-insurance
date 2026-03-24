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
    caso_verdadeiro = "Ia devagar na rotunda quando o outro senhor não parou no STOP e bateu-me."
    caso_falso = "Do nada apareceu um carro fantasma sem luzes, bati-lhe mas ele fugiu logo. Exijo indemnização."
    
    for texto in [caso_verdadeiro, caso_falso]:
        v, p, prob_crua = prever_fraude(texto)
        print(f"Texto: '{texto}'")
        print(f"-> VEREDITO: {v} ({p:.2f}%)")
        print("-" * 50)
