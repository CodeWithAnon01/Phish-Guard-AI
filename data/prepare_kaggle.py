import pandas as pd
import numpy as np
import pickle
import os
import argparse
from sklearn.model_selection import train_test_split

MAX_LEN = 200

def prepare(sample_size=None):
    print("Loading Kaggle dataset...")
    data_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(data_dir), 'datasets', 'phishing_site_urls.csv')

    df = pd.read_csv(csv_path)

    if 'Label' in df.columns:
        df['Label'] = df['Label'].map({'good': 0, 'bad': 1})

    print("Balancing and preparing data...")
    if sample_size and sample_size > 0:
        bad_df = df[df['Label'] == 1]
        good_df = df[df['Label'] == 0]
        bad_df = bad_df.sample(n=min(sample_size, len(bad_df)), random_state=42)
        good_df = good_df.sample(n=min(sample_size, len(good_df)), random_state=42)
        df = pd.concat([bad_df, good_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total rows: {len(df)}")

    print("Building character vocabulary...")
    chars = set()
    for url in df['URL']:
        chars.update(list(url))

    char_to_ix = {c: i+1 for i, c in enumerate(sorted(chars))}
    char_to_ix['<PAD>'] = 0

    print("Saving character vocabulary...")
    vocab_path = os.path.join(data_dir, 'char_vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(char_to_ix, f)

    print("Tokenizing URLs...")
    def tokenize(url):
        vec = [char_to_ix[c] for c in url[:MAX_LEN]]
        if len(vec) < MAX_LEN:
            vec.extend([0] * (MAX_LEN - len(vec)))
        return vec

    X = np.array([tokenize(url) for url in df['URL']], dtype=np.int32)
    y = df['Label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Saving tokenized arrays...")
    with open(os.path.join(data_dir, 'kaggle_train.pkl'), 'wb') as f:
        pickle.dump({'X': X_train, 'y': y_train}, f)

    with open(os.path.join(data_dir, 'kaggle_test.pkl'), 'wb') as f:
        pickle.dump({'X': X_test, 'y': y_test}, f)

    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare the Kaggle Dataset")
    parser.add_argument('--full', action='store_true', help="Use the entire dataset (removes the 40k cap).")
    args = parser.parse_args()
    size = None if args.full else 40000
    prepare(sample_size=size)
