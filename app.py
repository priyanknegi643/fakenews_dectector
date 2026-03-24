from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("model/bert_model.pt", map_location=device))
model.to(device)
model.eval()

# Prediction function
def predict(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).item()

    return "Real News ✅" if preds == 1 else "Fake News ❌"


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form["news"]
        result = predict(text)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)