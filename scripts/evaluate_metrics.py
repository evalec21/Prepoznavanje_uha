import csv
import random
from collections import defaultdict

import torch
import torchvision.transforms as T
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
SEED = 42

NUM_GENUINE_PAIRS = 500
NUM_IMPOSTOR_PAIRS = 500


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

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(num_classes: int):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return float((a * b).sum().item())


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    ckpt = torch.load("ear_cnn.pth", map_location="cpu")
    label_to_idx = ckpt["label_to_idx"]
    num_classes = len(label_to_idx)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    test_ds = EarDataset("data/splits/test.csv", transform=transform, label_to_idx=label_to_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    embed_model = torch.nn.Sequential(*list(model.children())[:-1])
    embed_model.to(DEVICE)
    embed_model.eval()

    embeddings = []
    print(f"Evaluating on device: {DEVICE}")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Test batches"):
            imgs = imgs.to(DEVICE)

            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().tolist()

            true_idx = [label_to_idx[l] for l in labels]
            for p, t in zip(preds, true_idx):
                correct += int(p == t)
                total += 1

            emb = embed_model(imgs)
            emb = emb.squeeze(-1).squeeze(-1)
            emb = emb.detach().cpu()
            for l, e in zip(labels, emb):
                embeddings.append((l, e))

    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy (test): {accuracy:.4f} ({correct}/{total})")

    by_label = defaultdict(list)
    for i, (label, emb) in enumerate(embeddings):
        by_label[label].append((i, emb))

    labels_list = list(by_label.keys())
    if len(labels_list) < 2:
        raise RuntimeError("Premalo različitih identiteta u test setu za FAR/FRR.")

    genuine_pairs = []
    for _ in range(NUM_GENUINE_PAIRS):
        valid = [l for l in labels_list if len(by_label[l]) >= 2]
        if not valid:
            break
        l = random.choice(valid)
        (i1, e1), (i2, e2) = random.sample(by_label[l], 2)
        genuine_pairs.append((e1, e2))

    impostor_pairs = []
    for _ in range(NUM_IMPOSTOR_PAIRS):
        l1, l2 = random.sample(labels_list, 2)
        i1, e1 = random.choice(by_label[l1])
        i2, e2 = random.choice(by_label[l2])
        impostor_pairs.append((e1, e2))

    if not genuine_pairs:
        print("\nNe mogu izračunati FRR jer u test setu nema labela s ≥2 uzorka.")
        return

    genuine_scores = [cosine_similarity(a, b) for a, b in genuine_pairs]
    impostor_scores = [cosine_similarity(a, b) for a, b in impostor_pairs]

    all_scores = sorted(set(genuine_scores + impostor_scores))
    best_thr = None
    best_diff = 1e9
    best_far = None
    best_frr = None

    for thr in all_scores:
        far = sum(s >= thr for s in impostor_scores) / len(impostor_scores)
        frr = sum(s < thr for s in genuine_scores) / len(genuine_scores)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_thr = thr
            best_far = far
            best_frr = frr

    print("\n--- FAR / FRR (verifikacija, cosine similarity) ---")
    print(f"Pairs genuine:  {len(genuine_scores)}")
    print(f"Pairs impostor: {len(impostor_scores)}")
    print(f"Chosen threshold (approx EER): {best_thr:.4f}")
    print(f"FAR: {best_far:.4f}")
    print(f"FRR: {best_frr:.4f}")
    print(f"|FAR-FRR|: {best_diff:.4f}")


if __name__ == "__main__":
    main()
