import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def process_row(row):
    damages = row.get('detected_damages', [])
    damages_text = "detected_damages: " + (", ".join(damages) if isinstance(damages, list) and damages else "none")

    statements = row.get('statements', [])
    formatted_statements = []
    valid_roles = ['insured_driver', 'third_party_driver', 'impartial_witness']
    
    if isinstance(statements, list):
        for stmt in statements:
            role = stmt.get('role', 'unknown_role')
            if role not in valid_roles:
                continue 
            text = stmt.get('text', '').strip()
            if text: 
                formatted_statements.append(f"{role}: {text}")
                
    statements_text = " | ".join(formatted_statements)
    return f"{damages_text} | {statements_text}"

def main():
    input_filepath = 'dataset_sintetico_chatgpt.json'
    
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar {input_filepath}: {e}")
        return
        
    df = pd.DataFrame(data)
    
    required_cols = ['claim_id', 'ground_truth_label', 'statements']
    if not all(col in df.columns for col in required_cols):
        print("Erro: Colunas em falta no JSON.")
        return

    df['X_text'] = df.apply(process_row, axis=1)
    df['Y_label'] = df['ground_truth_label'].apply(
        lambda x: 0 if x == 'genuine_accident' else 1
    )
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    X_seq = tokenizer.texts_to_sequences(df['X_text'].astype(str))
    X_pad = pad_sequences(X_seq, maxlen=400, padding='post', truncating='post')
    Y_label = df['Y_label'].values
    
    model = load_model('modelo_fraude_final.keras')
    
    y_pred_prob = model.predict(X_pad)
    limiar = 0.5
    y_pred = (y_pred_prob > limiar).astype(int)
    
    print("\n--- AVALIAÇÃO: dataset_sintetico_chatgpt.json ---")
    
    cm = confusion_matrix(Y_label, y_pred)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Previsto: Genuíno', 'Previsto: Fraude'],
                yticklabels=['Real: Genuíno', 'Real: Fraude'])
    plt.title('Matriz de Confusão (ChatGPT)')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    plt.savefig('matriz_confusao_chatgpt.png')
    plt.close()
    
    print("Matriz de Confusão salva como 'matriz_confusao_chatgpt.png'")
    print(classification_report(Y_label, y_pred, target_names=['Genuíno', 'Fraude']))

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
