import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

from dataset import NewsDataset

# Load data
train_df = pd.read_csv("../data/train.csv")
val_df   = pd.read_csv("../data/val.csv")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset
train_dataset = NewsDataset(train_df["content"].tolist(), train_df["label"].tolist(), tokenizer)
val_dataset   = NewsDataset(val_df["content"].tolist(), val_df["label"].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train function
def train_epoch():
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

# Validation
def evaluate(loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# Training loop
for epoch in range(3):
    loss = train_epoch()
    val_acc = evaluate(val_loader)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), "bert_model.pt")