from pathlib import Path
import cv2
import numpy as np
from collections import deque
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import mediapipe as mp

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "emotion_efficientnet_b0_best.pt"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load model
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(class_names)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# MediaPipe face detector
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

# smoothing
emotion_buffer = deque(maxlen=10)

def majority_vote(buffer):
    if not buffer:
        return None
    values, counts = np.unique(np.array(buffer), return_counts=True)
    return values[np.argmax(counts)]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

prev_tick = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    emotion_text = "no face"
    conf_text = ""

    if results.detections:
        det = results.detections[0]
        h, w, _ = frame.shape
        box = det.location_data.relative_bounding_box

        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        pad_x = int(0.15 * bw)
        pad_y = int(0.15 * bh)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        face = frame[y1:y2, x1:x2]

        if face.size > 0:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            inp = transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_conf = float(np.max(probs))
            emotion_buffer.append(pred_idx)

            smoothed_idx = majority_vote(emotion_buffer)
            emotion_text = class_names[smoothed_idx]
            conf_text = f"{pred_conf:.2f}"

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # top-3 probs
            top3_idx = np.argsort(probs)[-3:][::-1]
            y_text = y1 - 10 if y1 - 10 > 20 else y2 + 25
            for i, idx in enumerate(top3_idx):
                txt = f"{class_names[idx]}: {probs[idx]:.2f}"
                cv2.putText(display, txt, (x1, y_text + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # FPS
    current_tick = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (current_tick - prev_tick)
    prev_tick = current_tick

    cv2.putText(display, f"Emotion: {emotion_text}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    if conf_text:
        cv2.putText(display, f"Confidence: {conf_text}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(display, f"FPS: {fps:.1f}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Realtime Emotion Demo", display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()