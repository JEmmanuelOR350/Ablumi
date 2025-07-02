import os
import numpy as np
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from helpers import get_word_ids, get_sequences_and_labels
from constants import *
import sys

def training_model(model_path, epochs=500):
    word_ids = get_word_ids(WORDS_JSON_PATH)  # ['word1', 'word2', 'word3']
    
    sequences, labels = get_sequences_and_labels(word_ids)

    # Ver distribución de clases
    from collections import Counter
    import matplotlib.pyplot as plt

    counts = Counter(labels)
    print("Cantidad de muestras por clase (índice):", counts)

    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Clase (índice en word_ids)")
    plt.ylabel("Cantidad de ejemplos")
    plt.title("Distribución de clases")
    plt.show()

    # Preprocesamiento
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float16')
    X = np.array(sequences)   # shape esperada: (N, 84)
    y = to_categorical(labels)  # shape: (N, num_clases)
    print(X.shape, y.shape)

    with np.printoptions(threshold=sys.maxsize):
        print(f"X: {X[0]} Type: {type(X[0])} Tamaño:{len(X)}")
    print(f"y: {y} Type: {type(y)} Tamaño:{len(y)}")

    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

    if os.path.exists(model_path):
        respuesta = input(f"Modelo existente encontrado en '{model_path}'. ¿Quieres continuar entrenando este modelo? (s/n): ").strip().lower()
        if respuesta == 's':
            print("Cargando modelo existente...")
            from keras.models import load_model
            model = load_model(model_path)
        else:
            print("Creando un nuevo modelo desde cero...")
            model = get_model(num_classes=y.shape[1])
    else:
        print("No se encontró modelo guardado. Creando un nuevo modelo desde cero...")
        model = get_model(num_classes=y.shape[1])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=8)

    model.summary()
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    training_model(MODEL_PATH)
