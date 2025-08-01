import easyocr
import os
import torch

from config import SRC_DIR

image_files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

reader = easyocr.Reader(['en', 'ru'], gpu=torch.cuda.is_available())

for fname in image_files[:10]:
    src_path = os.path.join(SRC_DIR, fname)
    base_name = os.path.splitext(fname)[0]

    text = reader.readtext(src_path)

    print("Изображение:", base_name)
    for res in text:
        print("Результат:", res[1], "\nВероятность:", res[2])
    print('\n\n')


