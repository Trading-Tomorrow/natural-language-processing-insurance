import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

from tensorflow.keras.metrics import Precision, Recall, AUC


vocab_size = 5000 
maxlen = 400      
embedding_dim = 50 

def criar_modelo():
    model = Sequential()


    model.add(Input(shape=(maxlen,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))


    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())


    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.6))
    

    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    )

    return model

if __name__ == "__main__":
    modelo_fraude = criar_modelo()
    modelo_fraude.summary()