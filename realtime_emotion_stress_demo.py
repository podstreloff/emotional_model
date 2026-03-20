from pathlib import Path
from collections import deque
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import mediapipe as mp

# =========================
# Paths / config
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "emotion_efficientnet_b0_best.pt"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTION_WINDOW = 20
EMOTIONALIZE_WINDOW = 24
BASELINE_FRAMES = 60
CALIBRATION_HISTORY = 120
EMA_ALPHA = 0.12

# усиливаем контраст шкалы
EMOTIONALIZE_GAIN = 1.15
BASELINE_SCALE = 2.2

# =========================
# Recording / event logging
# =========================
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_TS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
VIDEO_OUT_PATH = OUTPUT_DIR / f"session_{SESSION_TS}.mp4"
EVENTS_OUT_PATH = OUTPUT_DIR / f"session_{SESSION_TS}_events.txt"

EVENT_START_THRESHOLD = 90.0
EVENT_END_THRESHOLD = 85.0
EVENT_MIN_DURATION_SEC = 0.7

# =========================
# Model transform
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =========================
# Load emotion model
# =========================
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

emotion_to_idx = {name: i for i, name in enumerate(class_names)}

# =========================
# MediaPipe
# =========================
mp_face_det = mp.solutions.face_detection
face_detector = mp_face_det.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Buffers
# =========================
emotion_buffer = deque(maxlen=EMOTION_WINDOW)
emotionalize_buffer = deque(maxlen=EMOTIONALIZE_WINDOW)
raw_emotionalize_history = deque(maxlen=CALIBRATION_HISTORY)
baseline_buffer = deque(maxlen=BASELINE_FRAMES)

baseline_value = None
ema_emotionalize_value = None

# event tracking
event_active = False
event_start_time = None
event_peak_score = 0.0
event_peak_emotion = "n/a"
event_below_counter = 0
event_index = 0

# =========================
# Helpers
# =========================
def majority_vote(buffer):
    if not buffer:
        return None
    vals, counts = np.unique(np.array(buffer), return_counts=True)
    return int(vals[np.argmax(counts)])

def smooth_mean(buffer, default=0.0):
    if not buffer:
        return default
    return float(np.mean(buffer))

def ema_update(current_value, new_value, alpha=EMA_ALPHA):
    if current_value is None:
        return float(new_value)
    return float(alpha * new_value + (1.0 - alpha) * current_value)

def dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def norm01(x, a, b):
    if b - a == 0:
        return 0.0
    return clamp((x - a) / (b - a), 0.0, 1.0)

def extract_face_roi(frame, detection):
    h, w, _ = frame.shape
    box = detection.location_data.relative_bounding_box

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
    return face, (x1, y1, x2, y2)

def emotion_postprocess(probs, class_names):
    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    pred_name = class_names[pred_idx]

    if pred_name == "disgust" and "sad" in class_names:
        pred_idx = class_names.index("sad")
        pred_conf = float(probs[pred_idx])

    return pred_idx, pred_conf

def compute_landmark_emotionalize(mesh_landmarks, probs):
    lm = mesh_landmarks.landmark
    pts = [(p.x, p.y) for p in lm]

    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

    LEFT_BROW_INNER = 70
    RIGHT_BROW_INNER = 300

    iod = dist(pts[LEFT_EYE_OUTER], pts[RIGHT_EYE_OUTER]) + 1e-6

    left_eye_open = dist(pts[LEFT_EYE_TOP], pts[LEFT_EYE_BOTTOM]) / iod
    right_eye_open = dist(pts[RIGHT_EYE_TOP], pts[RIGHT_EYE_BOTTOM]) / iod
    eye_open = 0.5 * (left_eye_open + right_eye_open)

    mouth_open = dist(pts[MOUTH_TOP], pts[MOUTH_BOTTOM]) / iod
    brow_inner_dist = dist(pts[LEFT_BROW_INNER], pts[RIGHT_BROW_INNER]) / iod

    eye_tension = abs(eye_open - 0.11)
    eye_tension = norm01(eye_tension, 0.00, 0.06)

    mouth_tension = norm01(mouth_open, 0.01, 0.12)
    brow_tension = 1.0 - norm01(brow_inner_dist, 0.30, 0.55)

    angry_p = probs[emotion_to_idx.get("angry", 0)]
    fear_p = probs[emotion_to_idx.get("fear", 0)]
    sad_p = probs[emotion_to_idx.get("sad", 0)]
    surprise_p = probs[emotion_to_idx.get("surprise", 0)]
    happy_p = probs[emotion_to_idx.get("happy", 0)]
    neutral_p = probs[emotion_to_idx.get("neutral", 0)]

    emotion_energy = (
        0.28 * angry_p +
        0.24 * fear_p +
        0.14 * sad_p +
        0.16 * surprise_p +
        0.10 * happy_p -
        0.10 * neutral_p
    )
    emotion_energy = clamp(emotion_energy, 0.0, 1.0)

    emotionalize01 = (
        0.45 * emotion_energy +
        0.20 * brow_tension +
        0.15 * eye_tension +
        0.10 * mouth_tension
    )
    # dead zone — игнорируем слабые движения
    if emotionalize01 < 0.15:
        emotionalize01 = 0.0
    emotionalize01 = clamp(emotionalize01 * EMOTIONALIZE_GAIN, 0.0, 1.0)
    return float(emotionalize01 * 100.0)
def draw_transparent_rect(img, x1, y1, x2, y2, color=(30,30,30), alpha=0.45):
    """
    Рисует полупрозрачный прямоугольник поверх img
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
def baseline_corrected_emotionalize(raw_score, baseline_val):
    if baseline_val is None:
        return raw_score
    corrected = 50.0 + (raw_score - baseline_val) * BASELINE_SCALE
    return clamp(corrected, 0.0, 100.0)

def calibrate_emotionalize(score, history):
    history.append(score)

    if len(history) < 30:
        return score

    smin = float(np.percentile(history, 3))
    smax = float(np.percentile(history, 97))

    if smax - smin < 1e-6:
        return score

    calibrated = 100.0 * (score - smin) / (smax - smin)
    return clamp(calibrated, 0.0, 100.0)

def emotionalize_label(score):
    if score < 25:
        return "low"
    if score < 60:
        return "medium"
    return "high"

def format_timecode(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"

def append_event_to_log(log_path, event_idx, start_sec, end_sec, peak_score, peak_emotion):
    duration = end_sec - start_sec
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Event {event_idx}\n")
        f.write(f"start_time_sec: {start_sec:.3f}\n")
        f.write(f"end_time_sec: {end_sec:.3f}\n")
        f.write(f"start_timecode: {format_timecode(start_sec)}\n")
        f.write(f"end_timecode: {format_timecode(end_sec)}\n")
        f.write(f"duration_sec: {duration:.3f}\n")
        f.write(f"peak_emotionalize: {peak_score:.1f}\n")
        f.write(f"peak_emotion: {peak_emotion}\n")
        f.write("\n")

# =========================
# Webcam loop
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_out = cap.get(cv2.CAP_PROP_FPS)
if fps_out is None or fps_out <= 1:
    fps_out = 25.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(str(VIDEO_OUT_PATH), fourcc, fps_out, (frame_w, frame_h))

with open(EVENTS_OUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Session start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Video file: {VIDEO_OUT_PATH.name}\n\n")

session_start_tick = cv2.getTickCount()
prev_tick = cv2.getTickCount()

cv2.namedWindow("Realtime Emotion + Emotionalize Demo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Realtime Emotion + Emotionalize Demo", 1280, 900)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    det_res = face_detector.process(rgb)

    emotion_text = "no face"
    confidence_text = ""
    emotionalize_text = "n/a"
    emotionalize_score = 0.0
    baseline_status = "calibrating"

    current_tick = cv2.getTickCount()
    current_time_sec = (current_tick - session_start_tick) / cv2.getTickFrequency()

    if det_res.detections:
        det = det_res.detections[0]
        face, (x1, y1, x2, y2) = extract_face_roi(frame, det)

        if face.size > 0:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            inp = transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

            # Проверяем, есть ли neutral в классе
            if "neutral" in class_names:
                neutral_idx = class_names.index("neutral")
                increment = 0.2  # увеличиваем на 10%

                # прибавляем к текущему значению, но не больше 1.0
                probs[neutral_idx] = min(probs[neutral_idx] + increment, 1.0)

                # нормализуем остальные вероятности, чтобы сумма = 1
                total = probs.sum()
                probs = probs / total

            pred_idx, pred_conf = emotion_postprocess(probs, class_names)
            emotion_buffer.append(pred_idx)
            smoothed_idx = majority_vote(emotion_buffer)

            emotion_text = class_names[smoothed_idx]
            confidence_text = f"{pred_conf:.2f}"

            mesh_res = face_mesh.process(rgb)
            if mesh_res.multi_face_landmarks:
                emotionalize_raw = compute_landmark_emotionalize(
                    mesh_res.multi_face_landmarks[0], probs
                )

                if len(baseline_buffer) < BASELINE_FRAMES:
                    baseline_buffer.append(emotionalize_raw)
                    baseline_value = float(np.mean(baseline_buffer))
                    baseline_status = f"calibrating {len(baseline_buffer)}/{BASELINE_FRAMES}"
                else:
                    baseline_status = "calibrated"

                emotionalize_corr = baseline_corrected_emotionalize(
                    emotionalize_raw, baseline_value
                )
                emotionalize_now = calibrate_emotionalize(
                    emotionalize_corr, raw_emotionalize_history
                )
                emotionalize_buffer.append(emotionalize_now)

                emotionalize_score_raw = smooth_mean(emotionalize_buffer, default=0.0)
                ema_emotionalize_value = ema_update(
                    ema_emotionalize_value,
                    emotionalize_score_raw,
                    alpha=EMA_ALPHA
                )
                emotionalize_score = ema_emotionalize_value
                emotionalize_text = emotionalize_label(emotionalize_score)

                # ===== Event detection =====
                if emotionalize_score >= EVENT_START_THRESHOLD:
                    if not event_active:
                        event_active = True
                        event_start_time = current_time_sec
                        event_peak_score = emotionalize_score
                        event_peak_emotion = emotion_text
                        event_below_counter = 0
                    else:
                        if emotionalize_score > event_peak_score:
                            event_peak_score = emotionalize_score
                            event_peak_emotion = emotion_text
                        event_below_counter = 0

                elif event_active:
                    if emotionalize_score < EVENT_END_THRESHOLD:
                        event_below_counter += 1
                    else:
                        event_below_counter = 0
                        if emotionalize_score > event_peak_score:
                            event_peak_score = emotionalize_score
                            event_peak_emotion = emotion_text

                    if event_below_counter >= 5:
                        event_end_time = current_time_sec
                        duration = event_end_time - event_start_time

                        if duration >= EVENT_MIN_DURATION_SEC:
                            event_index += 1
                            append_event_to_log(
                                EVENTS_OUT_PATH,
                                event_index,
                                event_start_time,
                                event_end_time,
                                event_peak_score,
                                event_peak_emotion
                            )

                        event_active = False
                        event_start_time = None
                        event_peak_score = 0.0
                        event_peak_emotion = "n/a"
                        event_below_counter = 0

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            top3_idx = np.argsort(probs)[-3:][::-1]
            yy = y1 - 10 if y1 - 10 > 50 else y2 + 25
            for i, idx in enumerate(top3_idx):
                txt = f"{class_names[idx]}: {probs[idx]:.2f}"
                draw_transparent_rect(display, x1 - 2, yy + i * 22 - 18, x1 + 90, yy + i * 22 + 4, alpha=0.45)
                cv2.putText(display, txt, (x1, yy + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            face_preview_size = 80
            face_preview = cv2.resize(face, (face_preview_size, face_preview_size))
            display[12:12 + face_preview_size, display.shape[1] - 12 - face_preview_size:display.shape[
                                                                                             1] - 12] = face_preview
            cv2.rectangle(display, (display.shape[1] - 12 - face_preview_size, 12),
                          (display.shape[1] - 12, 12 + face_preview_size), (255, 255, 255), 1)
    fps = cv2.getTickFrequency() / (current_tick - prev_tick)
    prev_tick = current_tick

    draw_transparent_rect(display, 12, 12, 240, 118, alpha=0.5)

    cv2.putText(display, "Emotion", (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
    cv2.putText(display, f"{emotion_text}", (22, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if confidence_text:
        cv2.putText(display, f"Conf {confidence_text}", (22, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 220, 255), 1)
    cv2.putText(display, f"Baseline {baseline_status}", (22, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
    cv2.putText(display, f"FPS {fps:.1f}", (22, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 220, 255), 1)

    # ===== Emotionalize bar: LEFT-BOTTOM (simple blue bar) =====
    # Синяя полоска Emotionalize с полупрозрачным фоном
    bar_w, bar_h = 320, 14
    margin = 20
    bar_x1 = margin
    bar_y1 = display.shape[0] - margin - bar_h
    display_score = 100.0 * ((emotionalize_score / 100.0) ** 2.0)  # non-linear рост
    fill_w = int((display_score / 100.0) * bar_w)

    draw_transparent_rect(display, bar_x1 - 12, bar_y1 - 28, bar_x1 + bar_w + 48, bar_y1 + bar_h + 12, alpha=0.5)
    cv2.putText(display, "EMOTIONALIZE", (bar_x1, bar_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
    cv2.rectangle(display, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (70, 70, 70), -1)
    cv2.rectangle(display, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), (255, 80, 0), -1)
    cv2.rectangle(display, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (230, 230, 230), 1)
    cv2.putText(display, f"{int(display_score)}", (bar_x1 + bar_w + 10, bar_y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)



    # REC indicator + session time
    cv2.circle(display, (display.shape[1] - 28, display.shape[0] - 28), 8, (0, 0, 255), -1)
    cv2.putText(display, f"REC {format_timecode(current_time_sec)}", (display.shape[1] - 150, display.shape[0] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    video_writer.write(display)

    cv2.imshow("Realtime Emotion + Emotionalize Demo", display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# close active event on exit
final_tick = cv2.getTickCount()
final_time_sec = (final_tick - session_start_tick) / cv2.getTickFrequency()

if event_active and event_start_time is not None:
    duration = final_time_sec - event_start_time
    if duration >= EVENT_MIN_DURATION_SEC:
        event_index += 1
        append_event_to_log(
            EVENTS_OUT_PATH,
            event_index,
            event_start_time,
            final_time_sec,
            event_peak_score,
            event_peak_emotion
        )

video_writer.release()
cap.release()
cv2.destroyAllWindows()

print(f"Saved video to: {VIDEO_OUT_PATH}")
print(f"Saved events log to: {EVENTS_OUT_PATH}")