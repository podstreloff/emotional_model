from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "faces"
OUT_CSV = PROJECT_DIR / "dataset.csv"

rows = []

emotion_map_crema = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

emotion_map_ravdess = {
    "01": "neutral",
    "02": "neutral",   # calm -> neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

for img in DATASET_DIR.rglob("*.jpg"):
    parts = img.parts
    path_str = str(img)

    emotion = None

    if "CREMA-D" in parts:
        # ...\faces\CREMA-D\1001_DFA_ANG_XX\frame_000001.jpg
        clip_name = img.parent.name
        tokens = clip_name.split("_")
        if len(tokens) >= 3:
            emotion_code = tokens[2]
            emotion = emotion_map_crema.get(emotion_code)

    elif "RAVDESS" in parts:
        # ...\faces\RAVDESS\...\01-02-05-01-...\frame_000001.jpg
        clip_name = img.parent.name
        tokens = clip_name.split("-")
        if len(tokens) >= 3:
            emotion_code = tokens[2]
            emotion = emotion_map_ravdess.get(emotion_code)

    if emotion is None:
        continue

    rows.append({
        "path": path_str,
        "emotion": emotion,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")

print("Saved:", OUT_CSV)
print("Rows:", len(df))
print(df["emotion"].value_counts())