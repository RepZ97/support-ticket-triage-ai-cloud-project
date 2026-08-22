"""Fetch the CFPB complaint dataset and build train/test splits."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PARQUET_URL = (
    "https://huggingface.co/api/datasets/milesbutler/consumer_complaints"
    "/parquet/default/train/0.parquet"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_DIR / "raw.parquet"

TEXT_COL = "Consumer Complaint"
LABEL_COL = "Product"

# CFPB renamed its product categories several times, so the raw data carries
# overlapping labels for the same thing. Collapse them into one canonical set.
LABEL_MAP = {
    "Credit reporting": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
    "Credit card": "Credit or prepaid card",
    "Credit card or prepaid card": "Credit or prepaid card",
    "Prepaid card": "Credit or prepaid card",
    "Checking or savings account": "Bank account",
    "Bank account or service": "Bank account",
    "Student loan": "Student loan",
    "Consumer Loan": "Consumer loan",
    "Vehicle loan or lease": "Consumer loan",
    "Payday loan": "Consumer loan",
    "Payday loan, title loan, or personal loan": "Consumer loan",
    "Money transfer, virtual currency, or money service": "Money transfer / virtual currency",
    "Money transfers": "Money transfer / virtual currency",
    "Virtual currency": "Money transfer / virtual currency",
    # "Other financial service" (221 rows) is dropped: a catch-all bucket with
    # too few examples to learn anything useful from.
}

MIN_CHARS = 40


def download():
    if RAW.exists():
        print(f"using cached {RAW.name} ({RAW.stat().st_size / 1e6:.0f} MB)")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("downloading dataset (~170 MB)...")
    pd.read_parquet(PARQUET_URL).to_parquet(RAW, index=False)
    print(f"saved {RAW.name}")


def main():
    download()

    df = pd.read_parquet(RAW, columns=[TEXT_COL, LABEL_COL])
    print(f"raw rows: {len(df):,}")

    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df["text"] = df[TEXT_COL].str.strip()
    df = df[df["text"].str.len() >= MIN_CHARS]
    print(f"with usable narrative: {len(df):,}")

    df["label"] = df[LABEL_COL].map(LABEL_MAP)
    dropped = df["label"].isna().sum()
    df = df.dropna(subset=["label"])
    print(f"after label mapping: {len(df):,} ({dropped:,} dropped as unmappable)")

    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"after dedupe: {len(df):,} ({before - len(df):,} duplicates removed)")

    df = df[["text", "label"]].reset_index(drop=True)

    print("\nclass distribution:")
    counts = df["label"].value_counts()
    for name, n in counts.items():
        print(f"  {n:>7,}  {name}")

    if counts.min() < 2:
        sys.exit("a class has fewer than 2 examples; cannot stratify")

    train, test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    train.to_parquet(DATA_DIR / "train.parquet", index=False)
    test.to_parquet(DATA_DIR / "test.parquet", index=False)
    print(f"\ntrain: {len(train):,}   test: {len(test):,}")


if __name__ == "__main__":
    main()
