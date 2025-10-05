import torch
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ---- Configuration ----
model_name = 'bert-base-uncased'  # or 'distilbert-base-uncased' for lighter model
max_text_len = 500                 # truncate input text to first 500 characters
num_lime_samples = 100             # reduce LIME samples to save memory

# ---- Load Model ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)
model.eval()

# ---- Sample text (replace with your own text) ----
text_to_explain = """
Joe Scarborough Defends President Obama’s Emotional Announcement On Gun Control...
"""[:max_text_len]  # truncate for memory safety

# ---- Prediction function for LIME ----
def predict_proba(texts):
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    return probs

# ---- LIME Explanation ----
explainer = LimeTextExplainer(class_names=['REAL', 'FAKE'])
exp = explainer.explain_instance(
    text_to_explain,
    predict_proba,
    num_features=10,
    num_samples=num_lime_samples
)

# ---- Save explanation ----
html_path = 'results/bert_lime_explanation.html'
exp.save_to_file(html_path)
print(f"LIME explanation saved to {html_path}")
