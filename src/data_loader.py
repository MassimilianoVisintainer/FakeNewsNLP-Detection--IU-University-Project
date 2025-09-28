from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/raw_combined.csv")

def load_and_merge(raw_dir=RAW_DIR, out_path=OUT_PATH, random_state=42):
    # try to find the typical files
    fake_path = raw_dir / "Fake.csv"
    true_path = raw_dir / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(f"Expected Fake.csv and True.csv in {raw_dir}")

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    # label: 1 = fake, 0 = real
    fake = fake.assign(label=1)
    true = true.assign(label=0)

    # unify columns: safe approach is to keep title and text if present
    for df in (fake, true):
        if "text" not in df.columns:
            df["text"] = df.get("content", "")  # fallback
        if "title" not in df.columns:
            df["title"] = ""

    combined = pd.concat([fake, true], ignore_index=True, sort=False)

    # drop exact duplicates on title+text
    combined = combined.drop_duplicates(subset=["title", "text"])

    # shuffle
    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # save
    combined.to_csv(out_path, index=False)
    print(f"Saved merged dataset to {out_path}")
    print("Counts:\n", combined["label"].value_counts())

    return combined

if __name__ == "__main__":
    load_and_merge()
