import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib


# -----------------------------
# Model Definition
# -----------------------------

class RankerMLP(nn.Module):
    def __init__(self, input_dim):
        super(RankerMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# Training Function
# -----------------------------

def train_model(feature_path, model_path, scaler_path, epochs=30, batch_size=256):

    data = np.load(feature_path)

    X = data[:, :-1]
    y = data[:, -1]

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    joblib.dump(scaler, scaler_path)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    model = RankerMLP(input_dim=X.shape[1])

    # Handle class imbalance
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_auc = 0

    for epoch in range(epochs):
        model.train()

        permutation = torch.randperm(X_train.size()[0])

        epoch_loss = 0

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_probs = torch.sigmoid(val_outputs).numpy()
            val_auc = roc_auc_score(y_val.numpy(), val_probs)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_path)

    print(f"\nBest Validation AUC: {best_val_auc:.4f}")
    print("Model saved to:", model_path)


if __name__ == "__main__":
    train_model(
        feature_path="../data/processed/features.npy",
        model_path="../models/ranker_mlp.pt",
        scaler_path="../models/feature_scaler.pkl"
    )