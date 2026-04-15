import pandas as pd
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

def get_x_y(json_file, tokenize_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    def process_row(row):
        damages = row.get('detected_damages', [])
        damages_text = "detected_damages: " + (", ".join(damages) if isinstance(damages, list) and damages else "none")
        statements = row.get('statements', [])
        valid_roles = ['insured_driver', 'third_party_driver', 'impartial_witness']
        formatted_statements = [f"{s.get('role', '')}: {s.get('text', '').strip()}" 
                                for s in statements if isinstance(s, dict) and s.get('role') in valid_roles and s.get('text', '').strip()]
        statements_text = " | ".join(formatted_statements)
        return f"{damages_text} | {statements_text}"

    df['X_text'] = df.apply(process_row, axis=1)
    df['Y_label'] = df['ground_truth_label'].apply(lambda x: 0 if x == 'genuine_accident' else 1)
    
    with open(tokenize_file, 'rb') as f:
        tokenizer = pickle.load(f)
        
    X_seq = tokenizer.texts_to_sequences(df['X_text'].astype(str))
    X_pad = pad_sequences(X_seq, maxlen=400, padding='post', truncating='post')
    return X_pad, df['Y_label'].values

def evaluate(model_path, tokenizer_path, dataset_path, description):
    print(f"\n--- {description} ---")
    try:
        X_pad, y_true = get_x_y(dataset_path, tokenizer_path)
        model = load_model(model_path)
        y_pred_prob = model.predict(X_pad, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.2f}")
        print(classification_report(y_true, y_pred, target_names=['Genuíno (0)', 'Fraude (1)']))
    except Exception as e:
        print(f"Failed to evaluate: {e}")

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    evaluate('modelo_fraude_old.keras', 'tokenizer_old.pkl', 'dataset_test_20.json', 'Velho Modelo vs 20 Novos Exemplos')
    evaluate('modelo_fraude_final.keras', 'tokenizer.pkl', 'dataset_test_20.json', 'Novo Modelo (Combinado) vs 20 Novos Exemplos')
    evaluate('modelo_fraude_old.keras', 'tokenizer_old.pkl', 'dataset_sintetico_grok.json', 'Velho Modelo vs Grok Dataset (Baseline de Antes)')
    evaluate('modelo_fraude_final.keras', 'tokenizer.pkl', 'dataset_sintetico_grok.json', 'Novo Modelo (Combinado) vs Grok Dataset')
