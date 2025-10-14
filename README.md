# Fake News Detection — IU University Project

### Detecting Fake News using Natural Language Processing (NLP) and Machine Learning

This project implements an NLP-based fake news detection system that uses various machine learning models to classify news articles as **real** or **fake**.  
It was developed as part of the **IU International University of Applied Sciences** coursework.

---

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Features](#features)  
4. [Setup & Installation](#setup--installation)  
5. [Usage](#usage)  
6. [Model Training](#model-training)  
7. [Evaluation](#evaluation)  
8. [Flask Web Prototype](#flask-web-prototype)  
9. [Results Summary](#results-summary)  
10. [Future Improvements](#future-improvements)  
11. [License & References](#license--references)

---

## Overview

The **FakeNewsNLP-Detection** project explores multiple **text classification algorithms** to detect misinformation using **linguistic features**.  
The pipeline includes:

- Text preprocessing and cleaning using **spaCy**
- Feature extraction using **TF–IDF**, **word embeddings**, and **vectorization**
- Model training using **supervised ML algorithms**
- Model evaluation with **accuracy, precision, recall, and F1-score**
- Deployment of a **Flask-based prototype web app**

This approach allows for the comparison of traditional machine learning models and demonstrates the effectiveness of feature engineering in NLP-based fake news detection.

---

---

## ⚙️ Features

- ✅ **Complete ML pipeline**: preprocessing → feature extraction → model training → evaluation  
- 🔤 **Advanced text preprocessing** using configurable `PreprocessConfig` class  
- 📊 **Multiple ML models**: Logistic Regression, SVM, Random Forest, Naive Bayes, XGBoost  
- 🌐 **Flask web prototype** for interactive classification  
- 📈 **Evaluation metrics and visualizations** (ROC curves, confusion matrices, etc.)  
- 🧠 **Explainable, modular code structure** for easy experimentation  

---

## 💻 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/MassimilianoVisintainer/FakeNewsNLP-Detection--IU-University-Project.git
cd FakeNewsNLP-Detection--IU-University-Project
```

### 2. Create and activate a virtual environment
```bash
python -m venv env
```
Activate the enve
```bash
env\Scripts\activate     # On Windows
source env/bin/activate  # On macOS/Linux
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy English model
```bash
python -m spacy download en_core_web_sm
```

### Usage
Run the Flask Web Prototype
```bash
cd app
python app.py
```

Then open your browser at:
👉 http://127.0.0.1:5000

You can train models individually using the scripts in the src/ directory.
```bash
python src/train_tfidf_baseline.py
python src/train_svm_fast.py
python src/train_random_forest_fast.py
python src/train_naive_bayes.py
python src/train_xgboost_fast.py
```

Each script saves the trained model and its corresponding vectorizer to the /models directory.

📊 Evaluation

To evaluate models on the test set:
```bash
python src/evaluate_models.py
```

This script computes and saves key metrics:

- Accuracy

- Precision

- Recall

- F1-score

- ROC-AUC

Plots (e.g., confusion matrices and ROC curves) are generated automatically.

🌍 Flask Web Prototype

The Flask app (app/app.py) integrates all trained models and the preprocessing pipeline into a simple, intuitive user interface.

Features:

- Dropdown model selection

- Real-time classification

- Clean, responsive UI (Tailwind CSS)

- Dynamic color-coded results (green = real, red = fake)
│
├── requirements.txt
└── README.
