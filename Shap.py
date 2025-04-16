import shap
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
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

X_test = X[int(0.85 * len(X)):]  # test set uniquement

model = InterpretableLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM).to(DEVICE)
model.load_state_dict(torch.load("model_checkpoint.pt", map_location=DEVICE))
model.eval()

class WrappedLSTM(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output, _ = self.model(x)
        return output



# ===============================
# SHAP DeepExplainer
# ===============================
background_tensor = X_test[:50].detach()
test_tensor = X_test[::100][:10].detach()  
# ===============================
# Visualisation brute d'une séquence d'entrée
# ===============================
sample_input = test_tensor[0].cpu().numpy().T  # shape: (features, seq_len)

plt.figure(figsize=(12, 6))
plt.imshow(sample_input, aspect='auto', cmap='viridis', origin='lower')
plt.yticks(np.arange(len(FEATURE_NAMES)), FEATURE_NAMES)
plt.colorbar(label="Valeur normalisée")
plt.title("Séquence d'entrée X_test[50] - features x timesteps")
plt.xlabel("Timestep")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/raw_input_sample0.png")
plt.show()


with torch.no_grad():
    outputs = model(test_tensor.to(DEVICE))[0].squeeze().cpu().numpy()

print("[DEBUG] Prédictions du modèle sur les 10 samples SHAP :")
for i, out in enumerate(outputs):
    print(f"Sample {i}: prédiction = {out:.6f}")

explainer = shap.DeepExplainer(WrappedLSTM(model), background_tensor)
shap_values = explainer.shap_values(test_tensor)

# ===============================
# Visualisation SHAP : 1 sample
# ===============================
sample_idx = 0
shap_sample = shap_values[0][sample_idx]  # shape (seq_len, n_features)

plt.figure(figsize=(12,6))
plt.imshow(np.abs(shap_sample).T, aspect='auto', cmap='coolwarm', origin='lower')
plt.yticks(np.arange(len(FEATURE_NAMES)), FEATURE_NAMES)
plt.colorbar(label="SHAP value (abs)")
plt.title(f"Carte SHAP - sample {sample_idx}")
plt.xlabel("Timestep")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/shap_map_sample_{sample_idx}.png")
plt.show()

# ===============================
# Agrégation : importance globale
# ===============================
shap_values_np = np.abs(np.array(shap_values)[0])  # shape (N, seq_len, n_features)
mean_importance = shap_values_np.mean(axis=(0, 1))

plt.figure(figsize=(8,5))
plt.barh(FEATURE_NAMES, mean_importance)
plt.title("Importance globale SHAP par feature")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/shap_feature_importance.png")
plt.show()