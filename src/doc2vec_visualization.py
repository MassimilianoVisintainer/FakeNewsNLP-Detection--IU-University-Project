import os
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

texts = data['text'].tolist()
labels = data['label'].tolist()  # 0=REAL, 1=FAKE

# ------------------------------
# Prepare TaggedDocuments for Doc2Vec
# ------------------------------
tagged_docs = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]

# ------------------------------
# Train Doc2Vec model
# ------------------------------
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
doc2vec_model.build_vocab(tagged_docs)
doc2vec_model.train(tagged_docs, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# ------------------------------
# Get document vectors
# ------------------------------
doc_vectors = [doc2vec_model.dv[str(i)] for i in range(len(tagged_docs))]

# ------------------------------
# Dimensionality reduction for visualization
# ------------------------------
# Option 1: PCA
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(doc_vectors)

# Option 2: t-SNE (optional, slower but often clearer)
# tsne = TSNE(n_components=2, random_state=42)
# vectors_2d = tsne.fit_transform(doc_vectors)

vectors_df = pd.DataFrame(vectors_2d, columns=['x', 'y'])
vectors_df['label'] = labels

# ------------------------------
# Plot
# ------------------------------
os.makedirs('results', exist_ok=True)

plt.figure(figsize=(10, 6))
for label, color, name in zip([0, 1], ['green', 'red'], ['REAL', 'FAKE']):
    subset = vectors_df[vectors_df['label'] == label]
    plt.scatter(subset['x'], subset['y'], c=color, label=name, alpha=0.5, s=10)

plt.title('Doc2Vec Document Vectors (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('results/doc2vec_2d.png')
plt.show()
