import os
import pickle
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

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

def evaluate():
    X_pad, Y_label = get_x_y('dataset_sintetico_variado.json', 'tokenizer_old.pkl')
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_pad, Y_label, test_size=0.20, 
        random_state=42, stratify=Y_label
    )
    
    model = load_model('modelo_fraude_old.keras')
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, target_names=['Genuíno (0)', 'Fraude (1)']))

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    evaluate()
