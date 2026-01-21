from __future__ import annotations

from pathlib import Path
import random
import csv
from collections import defaultdict

DATASET_DIR = Path("Ear_dataset")
OUT_DIR = Path("data/splits")
TRAIN_RATIO = 0.8
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def collect_samples(dataset_dir: Path) -> list[tuple[str, str]]:
    """
    Vraća listu (rel_path, label).
    Label je ime foldera (identitet), rel_path je putanja relativno od root projekta.
    Pretpostavka: data/awe/<label>/*.jpg
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Ne postoji dataset direktorij: {dataset_dir.resolve()}")

    samples: list[tuple[str, str]] = []

    for person_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        label = person_dir.name
        imgs = [img for img in person_dir.rglob("*") if is_image_file(img)]
        for img in imgs:
            rel_path = img.as_posix()
            samples.append((rel_path, label))

    return samples


def train_test_split_by_label(samples: list[tuple[str, str]], train_ratio: float, seed: int):
    """
    Radi split unutar svake labele:
    - svaka osoba ide i u train i u test, ako ima barem 2 slike
    - ako osoba ima 1 sliku, ide u train (da ne padne test; to kasnije možeš filtrirati)
    """
    rng = random.Random(seed)

    by_label: dict[str, list[str]] = defaultdict(list)
    for rel_path, label in samples:
        by_label[label].append(rel_path)

    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []

    for label, paths in by_label.items():
        rng.shuffle(paths)

        if len(paths) == 1:
            train_rows.append((paths[0], label))
            continue

        n_train = max(1, int(round(len(paths) * train_ratio)))
        n_train = min(n_train, len(paths) - 1)  # ostavi barem 1 za test

        train_paths = paths[:n_train]
        test_paths = paths[n_train:]

        train_rows.extend((p, label) for p in train_paths)
        test_rows.extend((p, label) for p in test_paths)

    return train_rows, test_rows, by_label


def write_csv(rows: list[tuple[str, str]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)


def main():
    print(f"Dataset dir: {DATASET_DIR.resolve()}")
    samples = collect_samples(DATASET_DIR)
    if not samples:
        raise RuntimeError("Nisam pronašao nijednu sliku. Provjeri strukturu data/awe/...")

    train_rows, test_rows, by_label = train_test_split_by_label(samples, TRAIN_RATIO, SEED)

    num_labels = len(by_label)
    total_imgs = len(samples)
    imgs_per_label = sorted((label, len(paths)) for label, paths in by_label.items())
    min_imgs = min(n for _, n in imgs_per_label)
    max_imgs = max(n for _, n in imgs_per_label)

    print("\n--- STATISTIKA ---")
    print(f"Broj identiteta (labela): {num_labels}")
    print(f"Ukupno slika: {total_imgs}")
    print(f"Min slika po identitetu: {min_imgs}")
    print(f"Max slika po identitetu: {max_imgs}")
    print(f"Train: {len(train_rows)}  | Test: {len(test_rows)}")

    # Zapiši CSV
    write_csv(train_rows, OUT_DIR / "train.csv")
    write_csv(test_rows, OUT_DIR / "test.csv")

    print("\nSnimljeno:")
    print(f"- {(OUT_DIR / 'train.csv').resolve()}")
    print(f"- {(OUT_DIR / 'test.csv').resolve()}")


if __name__ == "__main__":
    main()
