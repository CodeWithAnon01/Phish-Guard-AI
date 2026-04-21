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
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, drop_prob=0.5):
        super(URL_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

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

    config = {'vocab': vocab, 'max_len': MAX_LEN}
    models_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(models_dir, 'lstm_config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    X_all = torch.tensor(train_data['X'], dtype=torch.long)
    y_all = torch.tensor(train_data['y'], dtype=torch.float32)

    val_size = int(0.1 * len(X_all))
    indices = torch.randperm(len(X_all))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    X_train_t, y_train_t = X_all[train_idx], y_all[train_idx]
    X_val_t, y_val_t     = X_all[val_idx],   y_all[val_idx]

    X_test_t = torch.tensor(test_data['X'], dtype=torch.long)
    y_test_t = torch.tensor(test_data['y'], dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = URL_LSTM(vocab_size=vocab_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 15
    patience = 3
    best_val_loss = np.inf
    epochs_no_improve = 0
    model_path = os.path.join(models_dir, 'lstm_model.pt')

    print("Starting training with Validation-Based Early Stopping...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(device)).cpu()
            val_loss = criterion(val_preds, y_val_t).item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> New best val_loss={val_loss:.4f} — model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered on validation loss!")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t.to(device)).cpu()
        y_pred = (test_preds > 0.5).float().numpy()
        y_true = y_test_t.numpy()

    print("\nEvaluating on Test Set...")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Best model saved to {model_path}")

if __name__ == '__main__':
    train()
