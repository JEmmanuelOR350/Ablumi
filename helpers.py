import json
import numpy as np
from constants import LENGTH_KEYPOINTS, MODEL_FRAMES, WORDS_JSON_PATH

def get_word_ids(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    clases = list(dict.fromkeys(entry["class"] for entry in data))
    #input(f"lista: {clases} tamaño: {len(clases)}")
    return clases

def get_sequences_and_labels(word_ids):
    import json
    from constants import WORDS_JSON_PATH
    with open(WORDS_JSON_PATH, 'r') as f:
        data = json.load(f)

    X, y = [], []

    for entry in data:
        cls = entry["class"]
        if cls not in word_ids:
            continue

        left_frames = entry["left_points"]  # Lista de 100 frames
        right_frames = entry["right_points"]  # Lista de 100 frames

        assert len(left_frames) == len(right_frames), "Inconsistencia en cantidad de frames"

        for l_frame, r_frame in zip(left_frames, right_frames):
            l_flat = np.array(l_frame).flatten()  # (42,)
            r_flat = np.array(r_frame).flatten()  # (42,)
            combined = np.concatenate([l_flat, r_flat])  # (84,)
            X.append(combined)
            y.append(word_ids.index(cls))

    return X, y