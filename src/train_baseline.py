import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

cwd = os.getcwd()
if cwd.endswith("notebooks"):
    os.chdir("..")

DATA_PATH = Path("data/processed/bank_processed_unbalanced.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

USE_DOWNSAMPLED = False


def load_data():
    df = pd.read_csv(DATA_PATH)

    if USE_DOWNSAMPLED:
        minority = df[df["y_yes"] == 1]
        majority = df[df["y_yes"] == 0]

        majority_down = resample(
            majority,
            replace=False,
            n_samples=len(minority),
            random_state=42
        )

        df = (
            pd.concat([majority_down, minority])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    X = df.drop(columns=["y_yes"])
    y = df["y_yes"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / "logistic_regression.pkl")


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / "random_forest.pkl")


def main():
    X_train, X_test, y_train, y_test = load_data()
    train_logistic(X_train, y_train)
    train_random_forest(X_train, y_train)
    print(f"Saved models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
