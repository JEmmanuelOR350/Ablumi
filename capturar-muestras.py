import os
import cv2
import json
import threading
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATASET_SIZE = 100
JSON_FILE = "hand_data.json"

capturing = False
frame_count = 0
current_class = ""
current_label = ""

# Variables para compartir frames y resultados
current_frame = None
current_results = None

def video_loop():
    global current_frame, current_results, capturing, frame_count, current_class, current_label

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        current_results = results
        current_frame = img.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if capturing and frame_count < DATASET_SIZE:
            # Extraer puntos
            left_points = []
            right_points = []

            if results.multi_handedness:
                for idx, hand_handedness in enumerate(results.multi_handedness):
                    hand_label = hand_handedness.classification[0].label
                    hand_landmarks = results.multi_hand_landmarks[idx]
                    points = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

                    if hand_label == 'Left':
                        left_points = points
                    elif hand_label == 'Right':
                        right_points = points

            if not left_points:
                left_points = [[0, 0]] * 21
            if not right_points:
                right_points = [[0, 0]] * 21

            # Cargar JSON seguro
            if os.path.exists(JSON_FILE):
                try:
                    with open(JSON_FILE, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    data = []
            else:
                data = []

            # Buscar o crear entrada
            existing_entry = None
            for entry in data:
                if entry["class"] == current_class and entry["label"] == current_label:
                    existing_entry = entry
                    break

            if not existing_entry:
                new_entry = {
                    "class": current_class,
                    "label": current_label,
                    "left_points": {},
                    "right_points": {}
                }
                data.append(new_entry)
                existing_entry = new_entry

            frame_key = f"frame{frame_count}"
            existing_entry["left_points"][frame_key] = left_points
            existing_entry["right_points"][frame_key] = right_points

            with open(JSON_FILE, 'w') as f:
                json.dump(data, f, indent=4)

            frame_count += 1

            cv2.putText(img, f"Capturando {frame_count}/{DATASET_SIZE}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if frame_count >= DATASET_SIZE:
                capturing = False
                messagebox.showinfo("Completado", f"Captura completada para clase '{current_class}' y label '{current_label}'")

        cv2.imshow("Ventana Cámara - Presiona ESC para salir", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            capturing = False
            break

    cap.release()
    cv2.destroyAllWindows()
    # Si cierras la cámara, también cierra la app:
    root.quit()

def start_capture():
    global capturing, current_class, current_label, frame_count
    cls = entry_class.get().strip()
    lbl = entry_label.get().strip()

    if not cls or not lbl:
        messagebox.showerror("Error", "Debes ingresar clase y label.")
        return

    # Cargar JSON y buscar si la clase+label existe
    data = []
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

    for entry in data:
        if entry["class"] == cls and entry["label"] == lbl:
            # Preguntar si reemplazar
            if not messagebox.askyesno("Confirmar", f"La clase '{cls}' con label '{lbl}' ya existe. ¿Deseas reemplazar los datos?"):
                return
            else:
                data.remove(entry)
                with open(JSON_FILE, 'w') as f:
                    json.dump(data, f, indent=4)
                break

    current_class = cls
    current_label = lbl
    frame_count = 0
    capturing = True

def on_close():
    global capturing
    capturing = False
    root.destroy()

# --- GUI ---

root = tk.Tk()
root.title("Capturador de Lenguaje de Señas")

tk.Label(root, text="Nombre de la clase:").pack(pady=5)
entry_class = tk.Entry(root, width=30)
entry_class.pack(pady=5)

tk.Label(root, text="Label:").pack(pady=5)
entry_label = tk.Entry(root, width=30)
entry_label.pack(pady=5)

btn_start = tk.Button(root, text="Iniciar Captura", command=start_capture)
btn_start.pack(pady=10)

btn_exit = tk.Button(root, text="Salir", command=on_close)
btn_exit.pack(pady=5)

root.protocol("WM_DELETE_WINDOW", on_close)

# Lanzar hilo de cámara desde el inicio
threading.Thread(target=video_loop, daemon=True).start()

root.mainloop()
