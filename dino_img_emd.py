import torch
import os
import csv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import TENSOR_DIR

OUTPUT_CSV = "image_batches.csv"
NUM_WORKERS = 4
BATCH_SIZE = 64

if not torch.cuda.is_available():
    print("[INFO] CUDA не найден → работаем на CPU, xFormers отключен")
    os.environ["DINO_DISABLE_XFORMERS"] = "1"
    os.environ["XFORMERS_DISABLED"] = "1"

dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dino_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
dino_model = dino_model.to(device)

class TensorDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.files = [f for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        fpath = os.path.join(self.tensor_dir, fname)
        tensor = torch.load(fpath)
        return fname, tensor

dataset = TensorDataset(TENSOR_DIR)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"[INFO] Найдено {len(dataset)} готовых тензоров")

with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "embedding"])

    for batch_filenames, batch_tensors in tqdm(dataloader, desc="Инференс"):
        batch_tensors = batch_tensors.to(device)

        with torch.no_grad():
            batch_embeddings = dino_model(batch_tensors)

        batch_embeddings = batch_embeddings.cpu().numpy()

        for fname, emb in zip(batch_filenames, batch_embeddings):
            emb_str = ";".join(map(str, emb))
            writer.writerow([fname, emb_str])
