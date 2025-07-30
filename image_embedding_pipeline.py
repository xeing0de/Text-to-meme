import os
import time
from multiprocessing import Process, Queue

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms
from tqdm import tqdm

from config import SRC_DIR
from database import WDBObject

COUNT_DINO_WORKERS = 6
MAX_QUEUE_TORCH_SIZE = COUNT_DINO_WORKERS * 5
IMAGE_FILES = image_files = tuple(f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
IMAGE_FILES_SIZE = len(IMAGE_FILES)
OUTPUT_BIN = "image_batches.bin"

pbar = tqdm(total=IMAGE_FILES_SIZE, unit="img", smoothing=0, colour="green", desc="IMG to embedding")


# Запись в файл
class Writer(Process):
    def __init__(self, q_emb: Queue):
        super().__init__()
        self.bin_file = None
        self.q_emb = q_emb

        self.count_dino_workers_end = 0

    def writer(self, item: tuple[str, Tensor]):
        wdb_object = WDBObject.from_base(item)
        byte_obj = wdb_object.to_bytes()
        self.bin_file.write(byte_obj)

    def run(self):
        self.bin_file = open(OUTPUT_BIN, "wb")

        while True:
            item = self.q_emb.get()
            if item is None:
                self.count_dino_workers_end += 1
                if self.count_dino_workers_end >= COUNT_DINO_WORKERS:
                    break
                continue
            self.writer(item)
            pbar.update(1)

        pbar.close()
        self.bin_file.close()


# Превращение в эмбеддинги
class DinoWorker(Process):
    # def __init__(self, q_torch: Queue[tuple[str, Tensor]], q_out: Queue[tuple[str, Tensor]]):
    def __init__(self, q_torch: Queue, q_emb: Queue):
        super().__init__()

        self.device = None
        self.dino_model = None
        self.q_torch = q_torch
        self.q_emb = q_emb

    def process_item(self, img_name: str, tensor: Tensor) -> tuple[str, Tensor]:
        with torch.no_grad():
            tensor = tensor.to(self.device)
            embedding = self.dino_model(tensor.unsqueeze(0))
            embedding = embedding.squeeze(0).cpu()
        del tensor
        torch.cuda.empty_cache()
        return img_name, embedding

    def run(self):
        if not torch.cuda.is_available():
            print("[INFO][DinoWorker] CUDA не найден → CPU")
            torch.backends.xformers_enabled = False
            self.device = "cpu"
        else:
            self.device = "cuda"

        self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.dino_model.eval()
        self.dino_model.to(self.device)

        while True:
            item = self.q_torch.get(timeout=5)
            if item is None:
                break

            img_name, tensor = item
            result = self.process_item(img_name, tensor)
            self.q_emb.put(result)

        # Конец очереди
        self.q_emb.put(None)

        while self.q_emb.qsize() > 0:
            time.sleep(1)


# Конвертация в torch
class Converter(Process):
    # def __init__(self, q_torch: Queue[tuple[str, Tensor]]):
    def __init__(self, q_torch: Queue):
        super().__init__()

        self.q_torch = q_torch

        self.it = iter(IMAGE_FILES)

    @staticmethod
    def new_dino_size(size: tuple[int, int]) -> tuple[int, int]:
        x, y = size
        return 14 * (x // 14), 14 * (y // 14)

    def to_torch(self, img_name: str) -> Tensor:
        src_path = os.path.join(SRC_DIR, img_name)

        img = Image.open(src_path).convert("RGB")

        transform = transforms.Compose([
            # transforms.Resize(new_dino_size(img.size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.new_dino_size(img.size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        tensor: Tensor = transform(img)
        return tensor

    def run(self):
        while True:
            if self.q_torch.qsize() >= MAX_QUEUE_TORCH_SIZE:
                time.sleep(0.5)
                continue

            item = next(self.it, None)
            if item is None:
                break

            q_item = (item, self.to_torch(item))
            self.q_torch.put(q_item)

        for i in range(COUNT_DINO_WORKERS):
            self.q_torch.put(None)

        while self.q_torch.qsize() > 0:
            time.sleep(1)


def main():
    q_torch = Queue()
    q_emb = Queue()

    converter = Converter(q_torch)
    dino_workers = [DinoWorker(q_torch, q_emb) for _ in range(COUNT_DINO_WORKERS)]
    writer_csv = Writer(q_emb)

    converter.start()
    for w in dino_workers:
        w.start()
    writer_csv.start()

    converter.join()
    for w in dino_workers:
        w.join()
    writer_csv.join()


if __name__ == '__main__':
    main()
