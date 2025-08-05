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

OUTPUT = "image_embeddings.npy"
NUM_WORKERS = 8
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not torch.cuda.is_available():
    print("[INFO] CUDA не доступен, используется CPU")
    os.environ["DINO_DISABLE_XFORMERS"] = "1"
    os.environ["XFORMERS_DISABLED"] = "1"

dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dino_model = dino_model.eval().to(DEVICE)

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
            return fname, tensor
        except Exception as e:
            print(f"Ошибка обработки {fname}: {str(e)}")
            return fname, None

dataset = ImageDataset(SRC_DIR)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    prefetch_factor=3,
    pin_memory=True,
    collate_fn=lambda batch: [(f, t) for f, t in batch if t is not None]
)

print(f"[INFO] Найдено {len(dataset)} изображений для обработки")

all_files = []
with open(OUTPUT, "wb") as file:
    for batch in tqdm(dataloader, desc="Обработка изображений"):
        filenames, tensors = zip(*batch)
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
            embeddings = dino_model(batch_tensor)
        
        embeddings = embeddings.cpu().numpy()
        np.save(file, embeddings, allow_pickle=False)

np.save("filenames.npy", np.array(all_files))

print("[INFO] Обработка завершена. Результаты сохранены в", OUTPUT)
