import pandas as pd
from data_loader import load_and_merge
from preprocessing import TextPreprocessor, PreprocessConfig

# Step 1: Load and merge dataset
df = load_and_merge()

# Step 2: Take a few samples
samples = df['text'].head(5).tolist()

# Step 3: Initialize preprocessor
pp = TextPreprocessor(PreprocessConfig())

# Step 4: Apply preprocessing
for i, text in enumerate(samples, 1):
    print("---- SAMPLE", i, "----")
    print("RAW:", text[:150])
    clean = pp.transform(text)
    print("CLEAN:", clean[:150])
