import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modelo_cnn import criar_modelo

def main():
    df = pd.read_csv('dataset_preparado.csv')
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    X_seq = tokenizer.texts_to_sequences(df['X_text'].astype(str))
    X_pad = pad_sequences(X_seq, maxlen=400, padding='post', truncating='post')
    Y_label = df['Y_label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_pad, Y_label, test_size=0.20, 
        random_state=42, stratify=Y_label
    )

    classes_unicas = np.unique(y_train)
    pesos = compute_class_weight(class_weight='balanced', classes=classes_unicas, y=y_train)
    class_weights_dict = dict(zip(classes_unicas, pesos))
    print(f"Pesos das Classes calculados: Genuíno (0): {class_weights_dict[0]:.2f} | Fraude (1): {class_weights_dict[1]:.2f}")
    
    model = criar_modelo()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=50, 
        batch_size=16, 
        class_weight=class_weights_dict, 
        callbacks=[early_stop]
    )

    model.save('modelo_fraude_final.keras')
    print("Modelo guardado como 'modelo_fraude_final.keras'")


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


    print("\n--- AVALIAÇÃO FINAL: MATRIZ DE CONFUSÃO ---")
    

    y_pred_prob = model.predict(X_val)


    limiar = 0.5
    y_pred = (y_pred_prob > limiar).astype(int)


    cm = confusion_matrix(y_val, y_pred)


    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Previsto: Genuíno', 'Previsto: Fraude'],
                yticklabels=['Real: Genuíno', 'Real: Fraude'])
    plt.title('Matriz de Confusão (Validação)')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png')
    plt.close()


    print(classification_report(y_val, y_pred, target_names=['Genuíno', 'Fraude']))

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()