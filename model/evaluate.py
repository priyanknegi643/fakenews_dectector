import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from dataset import NewsDataset

# Load data
test_df = pd.read_csv("../data/test.csv")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset
test_dataset = NewsDataset(test_df["content"].tolist(), test_df["label"].tolist(), tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("bert_model.pt"))
model.to(device)

# Evaluation
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total:.4f}")