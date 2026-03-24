import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Parâmetros
vocab_size = 5000 
maxlen = 400      
embedding_dim = 50 

def criar_modelo():
    """Constrói e compila a arquitetura 1D CNN."""
    model = Sequential()

    # Input Layer e Embedding (Representação vetorial das palavras)
    model.add(Input(shape=(maxlen,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

    # CNN e Max Pooling (Extração de contexto local / n-grams)
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())

    # Classificador Intermédio e Dropout (Prevenção de overfitting severo)
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.6))
    
    # Layer de Output (Classificação binária Sigmoid)
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    modelo_fraude = criar_modelo()
    modelo_fraude.summary()
