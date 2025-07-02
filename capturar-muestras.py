import os
import json
import cv2
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp

# Paths
JSON_PATH = './hand_data.json'
dataset_size = 100

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Load existing data
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'r') as f:
        all_data = json.load(f)
else:
    all_data = []

def extract_hand_keypoints(results):
    points = []
    for hand_landmarks in results.multi_hand_landmarks or []:
        for lm in hand_landmarks.landmark:
            points.append([lm.x, lm.y])
    return points

def save_data_to_json(class_name, label, data):
    global all_data
    # Remove existing class if user confirmed overwrite
    all_data = [entry for entry in all_data if entry["class"] != class_name]
    all_data.append({
        "class": class_name,
        "label": label,
        "xy_points": data
    })
    with open(JSON_PATH, 'w') as f:
        json.dump(all_data, f, indent=2)

def capture_hand_data(class_name, label):
    cap = cv2.VideoCapture(0)
    collected_data = []

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Visual feedback
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract points
        hand_points = extract_hand_keypoints(results)
        if hand_points:
            collected_data.extend(hand_points)
            counter += 1

        cv2.putText(frame, f"Capturando: {counter}/{dataset_size}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Detección de Manos", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    save_data_to_json(class_name, label, collected_data)
    messagebox.showinfo("Completado", f"Captura completada para la clase: {label}")

def on_start():
    class_name = entry_class.get().strip()
    label = entry_label.get().strip()

    if not class_name or not label:
        messagebox.showerror("Error", "Debes ingresar la clase y su etiqueta.")
        return

    exists = any(entry['class'] == class_name for entry in all_data)
    if exists:
        confirm = messagebox.askyesno("Confirmar", f"La clase '{class_name}' ya existe. ¿Deseas reemplazarla?")
        if not confirm:
            return

    capture_hand_data(class_name, label)

def on_cancel():
    root.destroy()

# GUI
root = tk.Tk()
root.title("Capturador de Datos de Manos")

tk.Label(root, text="Nombre de clase (ej: hola):").pack(pady=5)
entry_class = tk.Entry(root, width=30)
entry_class.pack(pady=5)

tk.Label(root, text="Etiqueta a mostrar (ej: HOLA):").pack(pady=5)
entry_label = tk.Entry(root, width=30)
entry_label.pack(pady=5)

btn_start = tk.Button(root, text="Iniciar grabación", command=on_start)
btn_start.pack(pady=10)

btn_cancel = tk.Button(root, text="Cancelar", command=on_cancel)
btn_cancel.pack(pady=5)

root.mainloop()
