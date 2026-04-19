import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def get_metrics_for_final_model():
    df = pd.read_csv('dataset_preparado.csv', engine='python', encoding='utf-8')
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    X_seq = tokenizer.texts_to_sequences(df['X_text'].astype(str))
    X_pad = pad_sequences(X_seq, maxlen=400, padding='post', truncating='post')
    Y_label = df['Y_label'].values


    X_train, X_val, y_train, y_val = train_test_split(
        X_pad, Y_label, test_size=0.20, 
        random_state=42, stratify=Y_label
    )
    
    model = load_model('modelo_fraude_final.keras')
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, target_names=['Genuíno (0)', 'Fraude (1)']))

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    get_metrics_for_final_model()
