import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import TranslationDataset
from model import Seq2Seq
import argparse

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])  # exclude <eos> for input
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TranslationDataset("data/train.en", "data/train.de")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Seq2Seq().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), "results/model.pth")
    print("Model saved at results/model.pth")
