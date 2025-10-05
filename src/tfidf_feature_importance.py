import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ------------------------------
# Load cleaned dataset
# ------------------------------
data_path = 'data/processed/cleaned.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found!")

data = pd.read_csv(data_path, encoding='utf-8')

# Check columns
required_columns = ['text', 'label']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in CSV. Available columns: {data.columns.tolist()}")

# Check class distribution
label_counts = data['label'].value_counts()
print("Label distribution:\n", label_counts)

if len(label_counts) < 2:
    raise ValueError("Dataset must contain at least 2 classes (FAKE and REAL) to train the model.")

# ------------------------------
# Prepare data
# ------------------------------
texts = data['text'].tolist()
labels = data['label'].tolist()  # Use numeric labels directly

print(f"Number of REAL: {labels.count(0)}, Number of FAKE: {labels.count(1)}")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ------------------------------
# TF-IDF vectorization
# ------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# ------------------------------
# Train Logistic Regression
# ------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ------------------------------
# Feature importance
# ------------------------------
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

importance_df = pd.DataFrame({
    'word': feature_names,
    'coefficient': coefficients
})
importance_df['abs_coeff'] = importance_df['coefficient'].abs()
importance_df = importance_df.sort_values(by='abs_coeff', ascending=False)

# Top 20 words predicting FAKE
top_fake = importance_df.sort_values(by='coefficient', ascending=False).head(20)

# Top 20 words predicting REAL
top_real = importance_df.sort_values(by='coefficient').head(20)

# ------------------------------
# Plot results
# ------------------------------
os.makedirs('results', exist_ok=True)

plt.figure(figsize=(12, 6))
plt.barh(top_fake['word'], top_fake['coefficient'], color='red')
plt.title('Top Words Predicting FAKE')
plt.xlabel('Coefficient')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/top_words_fake.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.barh(top_real['word'], top_real['coefficient'], color='green')
plt.title('Top Words Predicting REAL')
plt.xlabel('Coefficient')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/top_words_real.png')
plt.show()

# ------------------------------
# Save top words to CSV
# ------------------------------
top_fake.to_csv('results/top_words_fake.csv', index=False)
top_real.to_csv('results/top_words_real.csv', index=False)

print("Top words for FAKE saved to results/top_words_fake.csv")
print("Top words for REAL saved to results/top_words_real.csv")
