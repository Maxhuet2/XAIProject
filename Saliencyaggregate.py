import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from Interppretablelstm import InterpretableLSTM, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM

# ===============================
# Paramètres
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_NAMES = ['Close', 'Volume', 'Return', 'RSI', 'MACD', 'BB_width', 'Stoch_K', 'OBV']
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# Chargement du modèle et des données
# ===============================
with open("data/AAPL_sequences.pkl", "rb") as f:
    data = pickle.load(f)

X = torch.tensor(data["X"], dtype=torch.float32).to(DEVICE)
y = torch.tensor(data["y"], dtype=torch.float32).squeeze().to(DEVICE)


total = len(X)
X_test = X[int(0.85 * total):]
y_test = y[int(0.85 * total):]

model = InterpretableLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM).to(DEVICE)
model.load_state_dict(torch.load("model_checkpoint.pt", map_location=DEVICE))
model.eval()

# ===============================
# Saliency sur plusieurs échantillons
# ===============================
def compute_saliency(model, input_tensor):
    input_tensor.requires_grad = True
    output, _ = model(input_tensor)
    output[0].backward()
    saliency = input_tensor.grad.abs().squeeze().detach().cpu().numpy()
    return saliency

saliency_list = []
sample_indices = range(min(30, len(X_test)))  # max 30 samples

for idx in sample_indices:
    sample_input = X_test[idx:idx+1].clone()
    saliency = compute_saliency(model, sample_input)
    saliency_list.append(saliency)

# ===============================
# Agrégation et Visualisation
# ===============================
mean_saliency = np.mean(np.stack(saliency_list), axis=0)

plt.figure(figsize=(12,6))
plt.imshow(mean_saliency.T, aspect='auto', cmap='hot', origin='lower')
plt.colorbar(label="Importance moyenne")
plt.yticks(ticks=np.arange(len(FEATURE_NAMES)), labels=FEATURE_NAMES)
plt.xlabel("Timestep")
plt.title("Carte de saliency moyenne (30 échantillons)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/saliency_avg_heatmap.png")
plt.show()


feature_importance = mean_saliency.mean(axis=1)

plt.figure(figsize=(8,5))
plt.barh(FEATURE_NAMES, mean_saliency.mean(axis=1))
plt.xlabel("Importance moyenne (agrégée sur 30 samples)")
plt.title("Importance globale des variables")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/saliency_avg_features.png")
plt.show()
