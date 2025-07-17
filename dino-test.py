import torch
from torchvision import transforms
from PIL import Image
from time import time
import os

if not torch.cuda.is_available():
    print("[INFO] CUDA не найден → отключаю xFormers (работа на CPU)")
    os.environ["DINO_DISABLE_XFORMERS"] = "1"
    os.environ["XFORMERS_DISABLED"] = "1"

dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dino_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
dino_model = dino_model.to(device)

start = time()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

img_path = "/home/xeing0de/gallery/memes/0.jpg"
image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    embedding = dino_model(image_tensor)

embedding = embedding[0].numpy()

end = time()

print("Размер эмбеддинга:", embedding.shape)
print(embedding)

print(end - start)

