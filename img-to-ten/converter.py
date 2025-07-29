import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

from config import SRC_DIR

script_dir = os.path.dirname(os.path.abspath(__file__))
new_folder = "tensors"
DST_DIR = os.path.join(script_dir, new_folder)
os.makedirs(DST_DIR, exist_ok=True)


def new_dino_size(size: tuple[int, int]) -> tuple[int, int]:
    x, y = size
    return 14 * (x // 14), 14 * (y // 14)


def convert(fname):
    transform = transforms.Compose([
            # transforms.Resize(new_dino_size(img.size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
    
    src_path = os.path.join(SRC_DIR, fname)

    base_name = os.path.splitext(fname)[0]
    dst_path = os.path.join(DST_DIR, base_name + ".pt")

    img = Image.open(src_path).convert("RGB")
    
    crop_size = new_dino_size(img.size)
    img_cropped = transforms.CenterCrop(crop_size)(img)
    tensor = transform(img_cropped)

    torch.save(tensor, dst_path)

if __name__ == "__main__":
    image_files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    with ProcessPoolExecutor() as executor:
        conv = {executor.submit(convert, fname): fname
                for fname in image_files[:1000]}
        for unit in tqdm(as_completed(conv), desc="Transform to tensors"):
            unit.result()
    
