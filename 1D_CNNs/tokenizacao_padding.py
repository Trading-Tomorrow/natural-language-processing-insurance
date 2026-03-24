import pandas as pd
import numpy as np
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

input_filepath = 'dataset_preparado.csv'
tokenizer_filepath = 'tokenizer.pkl'

# Parâmetros
num_words = 5000 
maxlen = 400 

def prepare_data_for_cnn():
    # Carregar dados preparados
    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Erro: '{input_filepath}' não encontrado.")
        return

    df['X_text'] = df['X_text'].fillna('').astype(str)
    text_data = df['X_text'].tolist()
    y_data = np.array(df['Y_label'].tolist())

    # Inicializar a Tokenização e construir o vocabulário
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(text_data)

    # Converter os textos em sequências de números e aplicar Padding
    sequences = tokenizer.texts_to_sequences(text_data)
    X_data = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

    # Guardar Tokenizer (Pickle)
    with open(tokenizer_filepath, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"Tokenização concluída. Features: {X_data.shape}, Labels: {y_data.shape}")
    return X_data, y_data, tokenizer

if __name__ == '__main__':
    prepare_data_for_cnn()
