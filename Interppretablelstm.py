import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pickle

# ===============================
# Paramètres
# ===============================
INPUT_DIM = 8  # ['Close', 'Volume', 'Return', 'RSI', 'MACD', 'BB_width', 'Stoch_K', 'OBV']
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
OUTPUT_DIM = 1
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# Modèle LSTM avec Attention
# ===============================
class InterpretableLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)
        output = self.fc(context)  # (batch, output_dim)
        return output, attn_weights

# ===============================
# Fonctions Entraînement/Évaluation
# ===============================
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output, _ = model(x_batch)
        loss = criterion(output.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    preds, reals, attns = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            output, attn = model(x_batch)
            preds.append(output.cpu().numpy())
            reals.append(y_batch.cpu().numpy())
            attns.append(attn.cpu().numpy())
    return np.concatenate(preds), np.concatenate(reals), np.concatenate(attns, axis=0)

# ===============================
# Visualisation et Métriques
# ===============================
def plot_prediction_vs_truth(preds, reals, title="", save_path=None):
    plt.figure(figsize=(10,6))
    plt.plot(reals, label="Réel")
    plt.plot(preds, label="Prédiction")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_attention_map(attn_weights, sample_index=0, save_path=None):
    weights = attn_weights[sample_index].squeeze()
    plt.figure(figsize=(10,4))
    plt.plot(weights, marker='o')
    plt.title(f"Poids d'attention - échantillon {sample_index}")
    plt.xlabel("Timestep")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_metrics(preds, reals):
    rmse = mean_squared_error(reals, preds, squared=False)
    mask = np.abs(reals) > 1e-6  # évite les divisions par zéro
    safe_reals = reals[mask]
    safe_preds = preds[mask]
    mape = mean_absolute_percentage_error(safe_reals, safe_preds) if len(safe_reals) > 0 else float('inf')
    print(f"\n[Metrics] RMSE: {rmse:.4f} | MAPE: {mape*100:.2f}%")
    return rmse, mape

# ===============================
# Split temporel
# ===============================
def temporal_split(X, y):
    total = len(X)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    with open("data/AAPL_sequences.pkl", "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"].squeeze()

    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print(f"y min: {y.min():.4f}, max: {y.max():.4f}, mean: {y.mean():.4f}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = temporal_split(X, y)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float()), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float()), batch_size=BATCH_SIZE, shuffle=False)

    model = InterpretableLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM).to(DEVICE)
    torch.save(model.state_dict(), "model_checkpoint.pt")
    print("[INFO] Modèle sauvegardé dans model_checkpoint.pt")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS+1):
        loss = train(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {loss:.4f}")

    preds, reals, attns = evaluate(model, test_loader, criterion)
    compute_metrics(preds, reals)
    plot_prediction_vs_truth(preds, reals, title="Prédiction vs Réel", save_path=f"{SAVE_DIR}/prediction.png")
    plot_attention_map(attns, sample_index=0, save_path=f"{SAVE_DIR}/attention_map.png")