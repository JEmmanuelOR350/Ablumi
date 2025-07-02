import json
import pandas as pd

with open("hand_data.json", "r") as f:
    data = json.load(f)

print(type(data))
print(list(data.keys())[:5])  # primeras 5 claves, si es diccionario
print(data["frame92"])  # ejemplo de un frame

for frame_key in data["right_points"]:
    frame_data = {"frame": frame_key}
    
    left_points = data["left_points"].get(frame_key, [])
    for i, point in enumerate(left_points):
        frame_data[f"left_{i}_x"] = point[0]
        frame_data[f"left_{i}_y"] = point[1]

    right_points = data["right_points"].get(frame_key, [])
    for i, point in enumerate(right_points):
        frame_data[f"right_{i}_x"] = point[0]
        frame_data[f"right_{i}_y"] = point[1]
    
    frames.append(frame_data)

df = pd.DataFrame(frames)
print(df)