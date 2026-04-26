"""
train_lstm.py
=============
Trains a Bidirectional LSTM on character-level URL tokenizations
to classify URLs as phishing (1) or safe (0).

Architecture:
  Embedding(vocab_size, 32) → BiLSTM(64 hidden) → Dropout(0.5) → Linear(1) → Sigmoid

Usage:
    python models/train_lstm.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import pickle
import numpy as np
import os

MAX_LEN = 200


class URL_LSTM(nn.Module):
    """Bidirectional LSTM for URL phishing detection.

    Args:
        vocab_size (int): Number of unique characters + 1 (for padding idx 0).
        embedding_dim (int): Character embedding dimension. Default 32.
        hidden_dim (int): LSTM hidden state size (per direction). Default 64.
        drop_prob (float): Dropout rate applied to the final LSTM hidden state.
        bidirectional (bool): Use bidirectional LSTM. Default True.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        drop_prob: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim    = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(drop_prob)

        # When bidirectional: final hidden = [forward || backward] → 2 * hidden_dim
        fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc      = nn.Linear(fc_input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)                       # (B, T, E)
        _, (hidden, _) = self.lstm(embedded)               # hidden: (2, B, H) if bidir

        if self.bidirectional:
            # Concatenate forward (hidden[0]) and backward (hidden[1]) final states
            out = torch.cat([hidden[0], hidden[1]], dim=-1)   # (B, 2H)
        else:
            out = hidden[-1]                                   # (B, H)

        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out).squeeze(-1)               # (B,)


def train():
    print("Loading prepared Kaggle dataset...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    with open(os.path.join(data_dir, 'kaggle_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, 'kaggle_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    with open(os.path.join(data_dir, 'char_vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    config     = {'vocab': vocab, 'max_len': MAX_LEN}
    models_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(models_dir, 'lstm_config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    X_all = torch.tensor(train_data['X'], dtype=torch.long)
    y_all = torch.tensor(train_data['y'], dtype=torch.float32)

    val_size  = int(0.1 * len(X_all))
    indices   = torch.randperm(len(X_all))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    X_train_t, y_train_t = X_all[train_idx], y_all[train_idx]
    X_val_t,   y_val_t   = X_all[val_idx],   y_all[val_idx]

    X_test_t = torch.tensor(test_data['X'], dtype=torch.long)
    y_test_t = torch.tensor(test_data['y'], dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model     = URL_LSTM(vocab_size=vocab_size, bidirectional=True).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Bidirectional LSTM — trainable parameters: {num_params:,}")

    epochs          = 15
    patience        = 4
    best_val_loss   = np.inf
    epochs_no_improve = 0
    model_path      = os.path.join(models_dir, 'lstm_model.pt')

    print("Starting training with Validation-Based Early Stopping...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss  = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(device)).cpu()
            val_loss  = criterion(val_preds, y_val_t).item()

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  → New best val_loss={val_loss:.4f} — model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered on validation loss!")
                break

    # ── Final test evaluation ────────────────────────────────────────────────
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t.to(device)).cpu()
        y_pred     = (test_preds > 0.5).float().numpy()
        y_true     = y_test_t.numpy()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    print(f"\nTest Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Test F1 Score : {f1:.4f}")
    print(f"Best model saved to {model_path}")

    if acc >= 0.85:
        print("\n✅  Target accuracy ≥ 85% REACHED!")
    else:
        print(f"\n⚠️   Below target — {acc*100:.2f}%  (target: 85.00%)")


if __name__ == '__main__':
    train()
