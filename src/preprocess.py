import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

cwd = os.getcwd()
if cwd.endswith("notebooks"):
    os.chdir("..")

RAW_PATH = Path("data/raw/bank-full.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RAW_PATH, sep=";")

    df = df[df["education"] != "unknown"].copy()
    df["education"] = df["education"].map({"primary": 0, "secondary": 1, "tertiary": 2})

    nominal_features = [
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
        "y",
    ]
    df = pd.get_dummies(df, columns=nominal_features, drop_first=True)

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    numerical = ["age", "balance", "day", "campaign", "pdays", "previous"]
    scaler = StandardScaler()
    df[numerical] = scaler.fit_transform(df[numerical])

    out_unbalanced = PROCESSED_DIR / "bank_processed_unbalanced.csv"
    out_default = PROCESSED_DIR / "bank_processed.csv"

    df.to_csv(out_unbalanced, index=False)
    df.to_csv(out_default, index=False)

    print(f"Saved processed files to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
