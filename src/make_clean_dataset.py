import pandas as pd
from data_loader import load_and_merge
from preprocessing import TextPreprocessor, PreprocessConfig

def main():
    df = load_and_merge()
    pp = TextPreprocessor(PreprocessConfig())

    print("Preprocessing with nlp.pipe... (this will be much faster)")
    texts = df["text"].fillna("").tolist()

    # Use the fast version
    df["clean_text"] = pp.transform_corpus_fast(texts, batch_size=200, n_process=1)

    df.to_csv("data/processed/cleaned.csv", index=False)
    print("Saved cleaned dataset with", len(df), "rows.")

if __name__ == "__main__":
    main()
