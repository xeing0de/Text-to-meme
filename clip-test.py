import torch
import clip
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# --- БАЗА КАРТИНОК ---
dirbase = "/home/xeing0de/gallery/memes"
N = 1000

# --- Сканируем папку ---
all_images = sorted([
    f for f in os.listdir(dirbase)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not all_images:
    raise FileNotFoundError(f"В папке {dirbase} не найдено картинок")

print(f"Найдено {len(all_images)} картинок. Берём первые {N}.")
selected_images = all_images[:N]

image_features = []

for fname in tqdm(selected_images, desc = "Обрабатываю"):
    img_path = os.path.join(dirbase, fname)

    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)  # нормализация
        image_features.append(feat.cpu().numpy())

# Превращаем в numpy-массив (N, 512)
image_features = np.vstack(image_features)

# --- ЗАПРОС ---
query = "Гена Букин король мира"
text_tokens = clip.tokenize([query]).to(device)

with torch.no_grad():
    text_feat = model.encode_text(text_tokens)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu().numpy()

# --- СХОДСТВО (косинус) ---
similarities = image_features @ text_feat.T  # (N,1)
best_idx = similarities.argmax()
print("Лучшее совпадение:", selected_images[best_idx])

