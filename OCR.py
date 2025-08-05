import easyocr
import os
import torch

from config import SRC_DIR

image_files = sorted([f for f in os.listdir(SRC_DIR) 
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                     key=lambda x: int(os.path.splitext(x)[0]))

reader = easyocr.Reader(['en', 'ru'], gpu=torch.cuda.is_available())

for fname in image_files[:10]:
    src_path = os.path.join(SRC_DIR, fname)
    base_name = os.path.splitext(fname)[0]

    text = reader.readtext(src_path)
    result = ""
    print("Изображение:", base_name)
#    result = " ".join(x[1] for x in text)
    for res in text:
        if res[2] > 0.6:
            result += res[1] + " "
    if len(result) > 3:
        print(result + "\n\n")


