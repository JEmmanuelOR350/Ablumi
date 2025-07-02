import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from constants import MODEL_PATH, LENGTH_KEYPOINTS
from helpers import get_word_ids

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Carga modelo y clases
model = load_model(MODEL_PATH)
clases = get_word_ids("hand_data.json")  # Usa el mismo JSON de entrenamiento

# Procesa landmarks
def extract_keypoints(results):
    # Inicializa keypoints para ambas manos desde la perspectiva del usuario
    hand_kps = {'Left': [0]*(LENGTH_KEYPOINTS//2), 'Right':[0]*(LENGTH_KEYPOINTS//2)}

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        model_label = handedness.classification[0].label  # 'Left' o 'Right' desde el modelo
        user_label = 'Right' if model_label == 'Left' else 'Left'  # Invertir para vista usuario

        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
        hand_kps[user_label] = keypoints

    # Orden garantizado: primero izquierda del usuario, luego derecha
    full_keypoints = hand_kps['Left'] + hand_kps['Right']
    return np.array(full_keypoints)

# Captura de cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) <= 2:
            keypoints = extract_keypoints(results)

            if len(keypoints) == LENGTH_KEYPOINTS:
                prediction = model.predict(np.expand_dims(keypoints, axis=0))[0]
                max_idx = np.argmax(prediction)
                max_val = prediction[max_idx]

                # Dibuja si la confianza supera 0.85
                if max_val > 0.85:
                    text = f"{clases[max_idx]} ({max_val:.2f})"
                    cv2.putText(frame, text, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # Mostrar top 10 predicciones
                sorted_indices = np.argsort(prediction)[::-1]
                line_height = 25
                box_height = min(len(clases), 10) * line_height + 10
                cv2.rectangle(frame, (10, 60), (310, 60 + box_height), (255, 255, 255), -1)

                for i, idx in enumerate(sorted_indices[:10]):
                    clase = clases[idx]
                    score = prediction[idx]
                    text_line = f"{clase}: {score:.2f}"
                    y_pos = 80 + i * line_height

                    color = (0, 200, 0) if i == 0 else (120, 120, 120)
                    thickness = 2 if i == 0 else 1

                    cv2.putText(frame, text_line, (15, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

        # Dibuja landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Ablumi Sign Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
