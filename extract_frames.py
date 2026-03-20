from pathlib import Path
import cv2
import os

# ===== config =====
FPS_TARGET = 5
VIDEO_EXTS = (".flv", ".mp4", ".avi", ".mkv", ".mov")
# ==================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # emotion_model/
INPUT_DIR = PROJECT_DIR / "datasets"
OUTPUT_DIR = PROJECT_DIR / "frames"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] cannot open: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps is None or video_fps <= 0:
        video_fps = 25.0

    frame_interval = max(1, int(round(video_fps / FPS_TARGET)))

    # subfolder per dataset + file stem to avoid collisions
    rel = video_path.relative_to(INPUT_DIR)
    out_dir = OUTPUT_DIR / rel.parent / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            out_path = out_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_id += 1

    cap.release()
    return saved

def main():
    if not INPUT_DIR.exists():
        print(f"[error] INPUT_DIR not found: {INPUT_DIR}")
        print("Создай папку emotion_model/datasets и положи туда датасеты.")
        return

    videos = [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    print("INPUT_DIR:", INPUT_DIR)
    print("videos found:", len(videos))

    # print a few examples
    for p in videos[:5]:
        print("  ", p)

    total = 0
    for i, v in enumerate(videos, 1):
        saved = extract_frames(v)
        total += saved
        if i % 50 == 0:
            print(f"[{i}/{len(videos)}] frames saved so far: {total}")

    print("DONE. Total frames saved:", total)

if __name__ == "__main__":
    main()