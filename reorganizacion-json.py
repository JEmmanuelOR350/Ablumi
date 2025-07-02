import json

# Cargar el archivo original
with open("hand_data.json", "r") as f:
    data = json.load(f)

# Lista para guardar todas las estructuras procesadas
procesados = []

for i in data:
    l = []
    r = []

    # Se asume que las claves son "frame0", "frame1", ..., consecutivamente
    total_frames = len(i["left_points"])
    for j in range(total_frames):
        l.append(i["left_points"][f"frame{j}"])
        r.append(i["right_points"][f"frame{j}"])

    struct = {
        "class": i["class"],
        "label": i["label"],
        "left_points": l,
        "right_points": r
    }

    procesados.append(struct)

# Guardar en un nuevo archivo
with open("hand_data_reorganizado.json", "w") as f:
    json.dump(procesados, f, indent=4)

print("Archivo reorganizado guardado como 'hand_data_reorganizado.json'")