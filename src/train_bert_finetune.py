"""
train_bert_finetune.py
----------------------
Fine-tunes DistilBERT on the Fake News dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    # FIX: Ensure TrainingArguments is imported
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
# REMOVED: Removed IntervalStrategy import to use string aliases for compatibility
import evaluate

# Paths
DATA_PATH = "data/processed/cleaned.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
    test_dataset = Dataset.from_dict({"text": X_test, "label": y_test})
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_datasets = dataset.map(tokenize, batched=True, batch_size=512)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # Added average="weighted" for robust metrics calculation
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
            "precision": precision.compute(predictions=preds, references=labels, average="weighted")["precision"],
            "recall": recall.compute(predictions=preds, references=labels, average="weighted")["recall"],
        }

    # -----------------------
    # 5. TrainingArguments (Optimized & Compatible Block)
    # -----------------------
    training_args = TrainingArguments(
        output_dir="models/bert_finetuned",
        
        # OPTIMIZATION 1: Evaluate and save less frequently (once per epoch)
        do_eval=True,
        # FIX: Reverting to step-based configuration for compatibility.
        # Calculated steps per epoch: 978. This evaluates/saves once per epoch.
        # Removed problematic 'evaluation_strategy' and 'save_strategy' arguments.
        save_steps=978,      
        eval_steps=978,

        # OPTIMIZATION 2: Increase batch size to reduce total number of steps
        per_device_train_batch_size=32, # Increased from 16
        per_device_eval_batch_size=64,  # Increased from 32
        
        # OPTIMIZATION 3: UNCOMMENT if you have an NVIDIA GPU (CRITICAL for speed!)
        # fp16=True, 
        
        # Standard settings
        learning_rate=2e-5,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=500, # Log training progress every 500 steps
    )

    # -----------------------
    # 6. Trainer
    # -----------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -----------------------
    # 7. Train & Save
    # -----------------------
    trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")
    preds_output = trainer.predict(tokenized_datasets["test"])
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_proba = preds_output.predictions[:, 1]  # probability for "fake"
    y_test = preds_output.label_ids

    # Save predictions + labels
    np.save(os.path.join(MODELS_DIR, "bert_y_test.npy"), y_test)
    np.save(os.path.join(MODELS_DIR, "bert_y_pred.npy"), y_pred)
    np.save(os.path.join(MODELS_DIR, "bert_y_proba.npy"), y_proba)

    # Save fine-tuned model and tokenizer
    trainer.save_model(os.path.join(MODELS_DIR, "bert_finetuned"))
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "bert_finetuned"))

    print("✅ BERT model and predictions saved to /models")

if __name__ == "__main__":
    main()
