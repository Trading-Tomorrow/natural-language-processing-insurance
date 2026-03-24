import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modelo_cnn import criar_modelo

# Parâmetros
TEST_SIZE = 0.20       
RANDOM_STATE = 42      
BATCH_SIZE = 16        
EPOCHS = 50            
PATIENCE = 7           

def main():
    # 1. Pipeline de Carregamento e Preparação com dados reais
    df = pd.read_csv('dataset_preparado.csv')
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    X_seq = tokenizer.texts_to_sequences(df['X_text'].astype(str))
    X_pad = pad_sequences(X_seq, maxlen=400, padding='post', truncating='post')
    Y_label = df['Y_label'].values

    # 2. Train-test Split Estratificado
    X_train, X_val, y_train, y_val = train_test_split(
        X_pad, Y_label, test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, stratify=Y_label
    )
    
    # 3. Treino da Rede com técnica de EarlyStopping
    model = criar_modelo()
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop]
    )

    # 4. Gravar os Pesos (.keras)
    model.save('modelo_fraude_final.keras')
    print("Modelo guardado como 'modelo_fraude_final.keras'")

    # 5. Visualização Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h = history.history
    
    axes[0].plot(h['loss'], label='Train Loss', marker='o')
    axes[0].plot(h['val_loss'], label='Val Loss', marker='o')
    axes[0].set_title('Loss')
    axes[0].legend()
    
    axes[1].plot(h['accuracy'], label='Train Acc', marker='o')
    axes[1].plot(h['val_accuracy'], label='Val Acc', marker='o')
    axes[1].set_title('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
