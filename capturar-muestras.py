import os
import cv2
import tkinter as tk
from tkinter import messagebox

DATA_DIR = './data'
dataset_size = 100

def capture_images(label):
    class_path = os.path.join(DATA_DIR, label)

    # Verificar si existe y preguntar
    if os.path.exists(class_path):
        confirm = messagebox.askyesno("Confirmar", f"La clase '{label}' ya existe. ¿Deseas reemplazar las imágenes anteriores?")
        if confirm:
            for file in os.listdir(class_path):
                os.remove(os.path.join(class_path, file))
        else:
            return
    else:
        os.makedirs(class_path)

    # Abrir cámara
    cap = cv2.VideoCapture(0)

    # Contador regresivo
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, f"Iniciando en {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow('Cuenta regresiva', frame)
        cv2.waitKey(1000)

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f"Capturando {counter+1}/{dataset_size}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Captura', frame)
        cv2.imwrite(os.path.join(class_path, '{}.jpg'.format(counter)), frame)
        counter += 1
        cv2.waitKey(25)

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Completado", f"Captura completada para: {label}")

def on_start():
    label = entry.get().strip()
    if not label:
        messagebox.showerror("Error", "Debes ingresar una letra o palabra.")
        return
    root.withdraw()  # Oculta la ventana durante la grabación
    capture_images(label)
    entry.delete(0, tk.END)
    root.deiconify()  # Muestra de nuevo la ventana

def on_cancel():
    root.destroy()

# Interfaz gráfica
root = tk.Tk()
root.title("Capturador de Lenguaje de Señas")

tk.Label(root, text="Letra o palabra:").pack(pady=10)
entry = tk.Entry(root, width=30)
entry.pack(pady=5)

btn_start = tk.Button(root, text="Iniciar grabación", command=on_start)
btn_start.pack(pady=10)

btn_cancel = tk.Button(root, text="Cancelar", command=on_cancel)
btn_cancel.pack(pady=5)

root.mainloop()