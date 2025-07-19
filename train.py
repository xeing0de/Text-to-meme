import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CSV_PATH        = "train_data.csv"
BATCH_SIZE      = 64
EPOCHS          = 5
LEARNING_RATE   = 1e-5
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextImageDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.sentences = df.iloc[:, 0].tolist()
        self.img_embs  = [
            torch.tensor([float(x) for x in row.split(';')], dtype=torch.float32)
            for row in df.iloc[:, 1]
        ]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.img_embs[idx]


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.sbert.eval()
        for p in self.sbert.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(384, 384, bias=False)
    
    def forward(self, sentences: list[str]) -> torch.Tensor:
        with torch.no_grad():
            emb = self.sbert.encode(sentences, convert_to_tensor=True)
        return self.proj(emb)


def collate_fn(batch):
    sents, imgs = zip(*batch)
    return list(sents), torch.stack(imgs)


def main() -> None:
    dataset = TextImageDataset(CSV_PATH)
    loader  = DataLoader(dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         collate_fn=collate_fn)

    model = TextEncoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for sentences, img_emb in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            img_emb = img_emb.to(DEVICE)
            pred = model(sentences)

            loss = criterion(pred, img_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(sentences)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}: MSE = {avg_loss:.6f}")

    torch.save(model.state_dict(), "text_to_meme_model.pt")
    print("Модель сохранена в text_to_meme_model.pt")


if __name__ == "__main__":
    main()

