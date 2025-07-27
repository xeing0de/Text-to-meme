import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

from config import SRC_DIR

script_dir = os.path.dirname(os.path.abspath(__file__))
new_folder = "tensors"
DST_DIR = os.path.join(script_dir, new_folder)
os.makedirs(DST_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

image_files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for fname in tqdm(image_files, desc="Transform to tensors"):
    src_path = os.path.join(SRC_DIR, fname)

    base_name = os.path.splitext(fname)[0]
    dst_path = os.path.join(DST_DIR, base_name + ".pt")

    img = Image.open(src_path).convert("RGB")
    tensor = transform(img)

    torch.save(tensor, dst_path)
