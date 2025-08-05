import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.hub
from config import SRC_DIR
import easyocr
from sentence_transformers import SentenceTransformer

OUTPUT = "image_embeddings.npy"
NUM_WORKERS = 8
BATCH_SIZE = 16 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not torch.cuda.is_available():
    print("[INFO] CUDA не доступен, используется CPU")
    os.environ["DINO_DISABLE_XFORMERS"] = "1"
    os.environ["XFORMERS_DISABLED"] = "1"

dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dino_model = dino_model.eval().to(DEVICE)

reader = easyocr.Reader(['en', 'ru'], gpu = torch.cuda.is_available())

sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sbert = sbert.eval().to(DEVICE)

class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.files = [f for f in os.listdir(img_dir) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        self.base_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = self.base_transform(img)
            return fname, tensor, img_path
        except Exception as e:
            print(f"Ошибка обработки {fname}: {str(e)}")
            return fname, None, img_path

dataset = ImageDataset(SRC_DIR)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    prefetch_factor=3,
    pin_memory=True,
    collate_fn=lambda batch: [(f, t, p) for f, t, p in batch if t is not None]
)

print(f"[INFO] Найдено {len(dataset)} изображений для обработки")

all_files = []
with open(OUTPUT, "wb") as file:
    for batch in tqdm(dataloader, desc="Обработка изображений"):
        filenames, tensors, path = zip(*batch)
        all_files.extend(filenames)

        batch_size = len(tensors)
        batch_tensor = torch.empty(
            (batch_size, 3, 224, 224),
            dtype=torch.float32,
            device=DEVICE
        )
        
        for i, tensor in enumerate(tensors):
            batch_tensor[i].copy_(tensor.to(DEVICE, non_blocking=True))
        
        with torch.no_grad():
            image_embs = dino_model(batch_tensor)
        
        final_emb = []
        for i, (fname, path) in enumerate(zip(filenames, path)):
            results = reader.readtext(path)
            result = ""
            for res in result:
                if res[2] > 0.6:
                    result += res[1] + " "
            if len(result.split()) > 3:
                text_emb = sbert.encode(
                    result, 
                    convert_to_tensor=True,
                    device=DEVICE
                ).cpu().numpy()
                
                combined_emb = 0.7 * image_embs[i] + 0.3 * text_emb
                final_emb.append(combined_emb)
            else:
                final_emb.append(image_embs[i])

        np.save(file, np.vstack(final_emb), allow_pickle=False)

np.save("filenames.npy", np.array(all_files))

print("[INFO] Обработка завершена. Результаты сохранены в", OUTPUT)
