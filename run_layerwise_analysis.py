import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel

# ================= Configuration =================
SUBJECT = "M01"
PARADIGM = "sentences"  # Best paradigm for layer comparison
DATA_ROOT = "./data"
STIMULI_DIR = "./data/stimuli"

# Select layers to test (Indices for GPT-2 XL which has 48 layers)
# We test: Embedding (0), Early (6, 12), Middle (24), Late (36), Final (48)
LAYERS_TO_TEST = [0, 6, 12, 24, 36, 48]
MODEL_NAME = "gpt2-xl"


# =============================================

def get_brain_data():
    brain_file = os.path.join(DATA_ROOT, "fMRI", f"{SUBJECT}_{PARADIGM}_betas.npy")
    if not os.path.exists(brain_file):
        raise FileNotFoundError(f"Run preprocess_brain_data.py for {PARADIGM} first.")
    return np.load(brain_file)


def get_stimuli(n_samples):
    # Try loading real text, or fallback to dummy if csv missing
    stim_file = os.path.join(STIMULI_DIR, f"stimuli_order_{SUBJECT}_{PARADIGM}.csv")
    if os.path.exists(stim_file):
        df = pd.read_csv(stim_file)
        return df['stimulus'].tolist()
    else:
        print("⚠️ Stimuli file not found. Using dummy text for code verification.")
        return [f"sample text {i}" for i in range(n_samples)]


def extract_layer_features(texts, model, tokenizer, layer_idx, device):
    """Extracts features specifically from the requested layer index."""
    embeddings = []
    batch_size = 8

    # Handle layer index logic (0 = embeddings, >0 = hidden states)
    # Note: HuggingFace output_hidden_states=True returns a tuple where:
    # index 0 = Initial Embeddings
    # index 1 = Layer 1 Output ... index 48 = Layer 48 Output

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Access specific hidden state
        # Tuple contains (Embeddings, Layer 1, ..., Layer 48)
        # So layer_idx 0 maps to tuple index 0
        hidden_state = outputs.hidden_states[layer_idx]

        # Pooling strategy (Last Token)
        if tokenizer.padding_side == "left":
            batch_emb = hidden_state[:, -1, :]
        else:
            last_token_idxs = inputs.attention_mask.sum(dim=1) - 1
            batch_emb = hidden_state[torch.arange(hidden_state.size(0)), last_token_idxs]

        embeddings.append(batch_emb.cpu().numpy())

    return np.vstack(embeddings)


def run_layerwise():
    print(f"--- 🧠 Layer-wise Analysis: {SUBJECT} - {PARADIGM} ---")

    # 1. Setup Data
    brain_data = get_brain_data()
    texts = get_stimuli(brain_data.shape[0])

    # 2. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
    model.eval()

    layer_scores = []

    # 3. Loop Through Layers
    for layer in tqdm(LAYERS_TO_TEST, desc="Analyzing Layers"):
        # Extract
        features = extract_layer_features(texts, model, tokenizer, layer, device)

        # Train Ridge Regression (Fast 5-fold)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        predictions = np.zeros_like(brain_data)
        ridge = RidgeCV(alphas=[1, 10, 100])  # Simplified alphas for speed

        for train_idx, test_idx in kf.split(features):
            ridge.fit(features[train_idx], brain_data[train_idx])
            predictions[test_idx] = ridge.predict(features[test_idx])

        # Compute Correlation
        Y_true = brain_data - brain_data.mean(axis=0)
        Y_pred = predictions - predictions.mean(axis=0)
        numer = np.sum(Y_true * Y_pred, axis=0)
        denom = np.sqrt(np.sum(Y_true ** 2, axis=0)) * np.sqrt(np.sum(Y_pred ** 2, axis=0)) + 1e-8
        corrs = numer / denom

        # We store the MEAN correlation across all voxels (or top 10% best voxels)
        # Taking the mean of positive correlations is a common metric
        mean_score = np.nanmean(corrs[corrs > 0])
        layer_scores.append(mean_score)

    # 4. Plot Results
    plt.figure(figsize=(10, 6))
    plt.plot(LAYERS_TO_TEST, layer_scores, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.title(f"Layer-wise Brain Alignment ({MODEL_NAME})")
    plt.xlabel("Model Layer Depth")
    plt.ylabel("Brain Encoding Performance (Mean Pearson r)")
    plt.grid(True, alpha=0.3)

    out_path = f"./results/{SUBJECT}_{PARADIGM}_layerwise_plot.png"
    plt.savefig(out_path)
    print(f"✅ Plot saved to {out_path}")
    print(f"Scores: {layer_scores}")


if __name__ == "__main__":
    run_layerwise()