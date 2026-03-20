from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

import timm
from tqdm import tqdm

# -------------------------
# Paths
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CSV_PATH = PROJECT_DIR / "dataset.csv"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "emotion_efficientnet_b0_best.pt"
LABELS_PATH = MODELS_DIR / "emotion_labels.txt"

# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LR = 3e-4
NUM_WORKERS = 0  # для Windows safest
RANDOM_STATE = 42

# -------------------------
# Dataset
# -------------------------
class EmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label

# -------------------------
# Main
# -------------------------
def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["emotion"])

    class_names = list(encoder.classes_)
    print("Classes:", class_names)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = EmotionDataset(train_df, transform=train_transform)
    val_ds = EmotionDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=len(class_names)
    ).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for images, labels in train_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]")
            for images, labels in val_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(f"\nEpoch {epoch}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": IMG_SIZE,
            }, BEST_MODEL_PATH)
            print(f"Best model saved to: {BEST_MODEL_PATH}")

        print("\nValidation report:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4
        ))
        print("-" * 60)

    print("Training finished.")
    print("Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()