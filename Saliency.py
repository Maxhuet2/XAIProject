import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# ===============================
# Paramètres
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 50
FEATURE_NAMES = ['Close', 'Volume', 'Return', 'RSI', 'MACD', 'BB_width', 'Stoch_K', 'OBV']
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# Chargement du modèle et des données
# ===============================
from Interppretablelstm import InterpretableLSTM, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM

with open("data/AAPL_sequences.pkl", "rb") as f:
    data = pickle.load(f)

X = torch.tensor(data["X"], dtype=torch.float32).to(DEVICE)
y = torch.tensor(data["y"], dtype=torch.float32).squeeze().to(DEVICE)

X_train, y_train = X[:int(0.7 * len(X))], y[:int(0.7 * len(y))]
X_val, y_val = X[int(0.7 * len(X)):int(0.85 * len(X))], y[int(0.7 * len(y)):int(0.85 * len(y))]
X_test, y_test = X[int(0.85 * len(X)):], y[int(0.85 * len(y)):]


model = InterpretableLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM).to(DEVICE)
model.load_state_dict(torch.load("model_checkpoint.pt", map_location=DEVICE))
model.eval()

# ===============================
# Saliency par gradients
# ===============================
def compute_saliency(model, input_tensor):
    input_tensor.requires_grad = True
    output, _ = model(input_tensor)
    output[0].backward()  # backward sur la première prédiction
    saliency = input_tensor.grad.abs().squeeze().detach().cpu().numpy()
    return saliency

# ===============================
# Visualisation d’une carte de saliency
# ===============================
def plot_saliency_map(saliency, feature_names, save_path=None):
    plt.figure(figsize=(12,6))
    plt.imshow(saliency.T, aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(label="Importance (gradient)")
    plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
    plt.xlabel("Timestep")
    plt.title("Carte de saliency (feature x temps)")
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ===============================
# Exécution sur un échantillon test
# ===============================
sample_index = 0
sample_input = X_test[sample_index:sample_index+1].clone()
saliency_map = compute_saliency(model, sample_input)

plot_saliency_map(saliency_map, FEATURE_NAMES, save_path=f"{SAVE_DIR}/saliency_sample_{sample_index}.png")