import csv
import torch
import torchvision.transforms as T
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EarDataset(Dataset):
    def __init__(self, csv_file, transform=None, label_to_idx=None):
        self.samples = []
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.samples.append((r["path"], r["label"]))

        if label_to_idx is None:
            labels = sorted({label for _, label in self.samples})
            self.label_to_idx = {l: i for i, l in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label_to_idx[label]


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

train_ds = EarDataset("data/splits/train.csv", transform)
num_classes = len(train_ds.label_to_idx)

print(f"Device: {DEVICE}")
print(f"Num classes: {num_classes}")
print(f"Train samples: {len(train_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS} - avg loss: {avg_loss:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "label_to_idx": train_ds.label_to_idx,
}, "ear_cnn.pth")

print("Model saved to ear_cnn.pth")
