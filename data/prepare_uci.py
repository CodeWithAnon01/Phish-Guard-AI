import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pickle
import os

def prepare_uci_dataset():
    print("Fetching UCI Phishing Websites dataset (id=327)...")
    try:
        phishing_websites = fetch_ucirepo(id=327)
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return

    X = phishing_websites.data.features
    y = phishing_websites.data.targets

    print("\n--- Inspecting Features and Target ---")
    print("Feature columns:", X.columns.tolist())
    target_col = y.columns[0]
    print(f"Target column: {target_col}")
    print("Unique labels in original target:", y[target_col].unique())

    df = pd.concat([X, y], axis=1)

    print("\n--- Cleaning Data ---")
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {initial_len - len(df)} rows with missing values.")

    print(f"Encoding labels: mapping Legitimate (1) to 0 (Safe) and Phishing (-1) to 1 (Phishing)")
    df[target_col] = df[target_col].map({1: 0, -1: 1, 0: 1})

    if df[target_col].isna().any():
        print("Warning: some labels were not mapped properly. Filling NaNs with 1 (phishing) as a fallback.")
        df[target_col] = df[target_col].fillna(1).astype(int)

    X_clean = df.drop(columns=[target_col])
    y_clean = df[target_col]

    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.20, random_state=42)
    print(f"Train size: {len(X_train)} rows")
    print(f"Test size: {len(X_test)} rows")

    print("\n--- Saving Data ---")
    data_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(data_dir, 'uci_train.pkl')
    test_path = os.path.join(data_dir, 'uci_test.pkl')

    with open(train_path, 'wb') as f:
        pickle.dump({'X_train': X_train, 'y_train': y_train}, f)

    with open(test_path, 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)

    print(f"Saved train set to: {train_path}")
    print(f"Saved test set to: {test_path}")

    print("\n--- Summary ---")
    print(f"Total Feature Count: {X_clean.shape[1]}")
    print("\nClass Distribution (Train):")
    print(y_train.value_counts().rename(index={0: "Safe (0)", 1: "Phishing (1)"}))
    print("\nClass Distribution (Test):")
    print(y_test.value_counts().rename(index={0: "Safe (0)", 1: "Phishing (1)"}))
    print("\nSample Rows (Features):")
    print(X_clean.head())

if __name__ == "__main__":
    prepare_uci_dataset()
