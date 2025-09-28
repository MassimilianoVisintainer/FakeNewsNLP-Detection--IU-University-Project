import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    # Step 1: Load cleaned dataset
    df = pd.read_csv("data/processed/cleaned.csv")

    # Step 2: Split into train/test (stratify = keep class balance)
    train, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    # Step 3: Save to processed folder
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Saved {train_path} and {test_path}")

if __name__ == "__main__":
    main()
