from pathlib import Path
import cv2
import mediapipe as mp

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_DIR / "frames"
OUTPUT_DIR = PROJECT_DIR / "faces"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

images = list(INPUT_DIR.rglob("*.jpg"))

print("PROJECT_DIR:", PROJECT_DIR)
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("images found:", len(images))

saved_count = 0
no_face_count = 0
read_fail_count = 0

for i, img_path in enumerate(images, 1):
    img = cv2.imread(str(img_path))

    if img is None:
        read_fail_count += 1
        if read_fail_count <= 5:
            print("[read fail]", img_path)
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if not results.detections:
        no_face_count += 1
        if no_face_count <= 5:
            print("[no face]", img_path)
        continue

    h, w, _ = img.shape
    det = results.detections[0]
    box = det.location_data.relative_bounding_box

    x = int(box.xmin * w)
    y = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)

    # Небольшой padding вокруг лица
    pad_x = int(0.15 * bw)
    pad_y = int(0.15 * bh)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    face = img[y1:y2, x1:x2]

    if face.size == 0:
        continue

    face = cv2.resize(face, (224, 224))

    rel_path = img_path.relative_to(INPUT_DIR)
    save_path = OUTPUT_DIR / rel_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(save_path), face)
    if ok:
        saved_count += 1

    if i % 5000 == 0:
        print(f"[{i}/{len(images)}] saved={saved_count}, no_face={no_face_count}, read_fail={read_fail_count}")

print("DONE")
print("saved:", saved_count)
print("no_face:", no_face_count)
print("read_fail:", read_fail_count)