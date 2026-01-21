import csv
import cv2
import matplotlib.pyplot as plt

with open("data/splits/train.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

sample = rows[0]["path"]

img = cv2.imread(sample)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

plt.imshow(img)
plt.title("Preprocessed ear image (224x224)")
plt.axis("off")
plt.show()
