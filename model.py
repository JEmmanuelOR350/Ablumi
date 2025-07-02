from keras.models import Sequential
from keras.layers import Dense, Dropout,Input
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS

def get_model(num_classes):
    model = Sequential()
    model.add(Input(shape=(84,)))
    model.add(Dense(128, activation='relu'))  # Asume entrada de 84 features
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Salida con 'num_classes' neuronas
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
    return model