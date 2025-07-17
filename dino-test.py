import torch
from torchvision import transforms
from PIL import Image
import os
import csv
from tqdm import tqdm
from math import ceil

IMAGE_DIR = "/home/xeing0de/gallery/memes"
OUTPUT_CSV = "image_batches.csv"
BATCH_SIZE = 16

if not torch.cuda.is_available():
    print("[INFO] CUDA не найден → работаем на CPU, xFormers отключен")
    os.environ["DINO_DISABLE_XFORMERS"] = "1"
    os.environ["XFORMERS_DISABLED"] = "1"

dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dino_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
dino_model = dino_model.to(device)

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
image_files.sort()

total_images = len(image_files)
total_batches = ceil(total_images / BATCH_SIZE)

print(f"[INFO] Найдено {total_images} изображений, будет {total_batches} батчей")

with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "batch_index"])

    for batch_idx in tqdm(range(total_batches)):
        batch_files = image_files[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        batch_tensors = []
        for fname in batch_files:
            img_path = os.path.join(IMAGE_DIR, fname)
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image)
            batch_tensors.append(tensor)

        batch_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            batch_embeddings = dino_model(batch_tensor)

        batch_embeddings=batch_embeddings.cpu().numpy()

        for fname, emb in zip(batch_files, batch_embeddings):
            emb_str = ";".join(map(str, emb))
            writer.writerow([fname, emb_str])
