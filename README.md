# emotional_model
Emotion recognition model based in CREMA-D and RAVDESS datasets. Supposed to be used in medtech.

# 🎭 Realtime Emotion Recognition & Emotionalize System

Real-time emotion recognition system with custom emotional activity metric **Emotionalize**, built using deep learning and facial landmark analysis.

---

## 🚀 Features

- 🎥 Real-time emotion recognition (webcam)
- 🧠 EfficientNet-B0 model (PyTorch)
- 😐 7 emotion classes:
  - angry, disgust, fear, happy, neutral, sad, surprise
- 📊 Custom **Emotionalize metric**
- 📉 EMA smoothing + baseline calibration
- 🎯 High-emotion event detection
- 💾 Video recording + event logging
- 📺 Live HUD interface:
  - Emotion + confidence
  - Top-3 predictions
  - Emotionalize bar
  - Face preview
  - REC indicator

---

## 🧠 Model

- Architecture: `EfficientNet-B0`
- Framework: `PyTorch + timm`
- Input: `224x224` face crop
- Output: softmax probabilities (7 classes)

### 📈 Performance

| Metric | Value |
|------|------|
| Accuracy | **93%** |

#### Classification Report

| Emotion | Precision | Recall | F1 |
|--------|----------|--------|----|
| angry | 0.88 | 0.93 | 0.90 |
| disgust | 0.95 | 0.92 | 0.94 |
| fear | 0.92 | 0.91 | 0.92 |
| happy | 0.98 | 0.95 | 0.97 |
| neutral | 0.95 | 0.93 | 0.94 |
| sad | 0.90 | 0.91 | 0.91 |
| surprise | 0.96 | 0.99 | 0.98 |

---

## 📦 Dataset

- ~200,000 labeled images
- 7 emotion classes

### Distribution

- neutral: 38k  
- sad: 33k  
- angry: 32k  
- fear: 31k  
- happy: 31k  
- disgust: 26k  
- surprise: 6.8k  

⚠️ Class imbalance present  
⚠️ Similar classes: sad / disgust, fear / surprise  

---

## 🧮 Emotionalize Metric

Custom metric combining:
- CNN emotion probabilities
- Facial landmarks (MediaPipe)

### Formula
Emotionalize = 0.45 * emotion_energy + 0.20 * brow_tension + 0.15 * eye_tension + 0.10 * mouth_tension

---

### ⚙️ Processing Pipeline

1. Face detection (MediaPipe)
2. Emotion classification (CNN)
3. Landmark extraction (468 points)
4. Feature engineering
5. Emotionalize computation
6. Smoothing (EMA)
7. Baseline calibration
8. Normalization (percentiles)

---

### 🧠 Key Techniques

- EMA smoothing (reduce jitter)
- Dead-zone filtering (<15%)
- Baseline personalization (first 60 frames)
- Non-linear scaling for UI
- Post-processing (disgust → sad)

---

## 🎯 Event Detection

System detects high emotional activity:

| Parameter | Value |
|----------|------|
| Start threshold | 90 |
| End threshold | 85 |
| Min duration | 0.7 sec |

### Logged data:
- start time
- end time
- duration
- peak emotionalize
- peak emotion

---

## 🖥️ UI (HUD)

- Emotion label + confidence
- Top-3 predictions
- Emotionalize bar (dynamic)
- Face preview
- Recording indicator

---

## ⚡ Performance

- FPS: ~25–35
- Latency: ~30–50 ms
- Real-time processing

---

## 🛠️ Tech Stack

- Python
- PyTorch
- timm
- OpenCV
- MediaPipe
- NumPy

---

## 📂 Project Structure
emotion_model/
│
├── models/
│ └── emotion_efficientnet_b0_best.pt
│
├── scripts/
│ ├── build_dataset.py
│ ├── train_emotion_model.py
│ └── realtime_emotion_stress_demo.py
│
├── output/
│ ├── videos/
│ └── logs/
│
└── dataset.csv

---

## ▶️ Run

```bash
pip install -r requirements.txt
python realtime_emotion_stress_demo.py
