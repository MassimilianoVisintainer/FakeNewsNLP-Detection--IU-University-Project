# src/bert_quick_train.py
"""
Quick fine-tuning of DistilBERT for fast evaluation.
Uses a subset of the dataset to speed up CPU training.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Paths
DATA_PATH = "data/processed/cleaned.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Training params
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 8          # small batch for CPU
EPOCHS = 1              # just 1 epoch for fast run
MAX_LEN = 128           # max token length
SUBSET_SIZE = 5000      # use a subset for speed

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("")
    df = df.sample(n=SUBSET_SIZE, random_state=42)  # subset for fast training

    texts = df["clean_text"].tolist()
    labels = LabelEncoder().fit_transform(df["label"].tolist())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Tokenizer & model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # Datasets & loaders
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, leave=False)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            loop.set_description(f"Epoch {epoch+1} Loss {loss.item():.4f}")

    # Evaluation
    model.eval()
    y_pred = []
    y_proba = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
            y_proba.extend(probs[:, 1].cpu().numpy())

    y_test_array = np.array(y_test)
    y_pred_array = np.array(y_pred)
    y_proba_array = np.array(y_proba)

    # Save predictions
    np.save(os.path.join(MODELS_DIR, "bert_y_test.npy"), y_test_array)
    np.save(os.path.join(MODELS_DIR, "bert_y_pred.npy"), y_pred_array)
    np.save(os.path.join(MODELS_DIR, "bert_y_proba.npy"), y_proba_array)

    print("✅ Quick DistilBERT training complete. Predictions saved to /models")

if __name__ == "__main__":
    main()
