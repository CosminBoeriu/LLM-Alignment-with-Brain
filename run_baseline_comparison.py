import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ================= Configuration =================
SUBJECT = "M01"
PARADIGM = "sentences"
DATA_ROOT = "data"
MODEL_NAME = "gpt2-xl"


# =============================================

def get_brain_data():
    brain_file = os.path.join(DATA_ROOT, "fMRI", f"{SUBJECT}_{PARADIGM}_betas.npy")
    if not os.path.exists(brain_file):
        raise FileNotFoundError(f"Run preprocess_brain_data.py for {PARADIGM} first.")
    return np.load(brain_file)


def get_stimuli_dummy(n_samples):
    # We use dummy text or load real if available;
    # for baseline comparison, keeping text consistent is key.
    # If you have the real CSV, load it here.
    return [f"sample text {i}" for i in range(n_samples)]


def extract_features(texts, model, tokenizer, device):
    model.eval()
    embeddings = []
    batch_size = 8

    print("Extracting features from RANDOM model...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Use last hidden state (Standard output)
        hidden_state = outputs.last_hidden_state

        if tokenizer.padding_side == "left":
            batch_emb = hidden_state[:, -1, :]
        else:
            last_token_idxs = inputs.attention_mask.sum(dim=1) - 1
            batch_emb = hidden_state[torch.arange(hidden_state.size(0)), last_token_idxs]

        embeddings.append(batch_emb.cpu().numpy())

    return np.vstack(embeddings)


def run_baseline():
    print(f"--- 🧪 Running Baseline (Untrained) Check: {SUBJECT} ---")

    # 1. Load Data
    brain_data = get_brain_data()
    texts = get_stimuli_dummy(brain_data.shape[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Initialize Models
    # A. TRAINED Model (Reference)
    # We assume you ran this before, but we can re-load results or re-run briefly.
    # To save time, let's load the SAVED results from previous steps if available.
    trained_file = f"./results/{SUBJECT}_{PARADIGM}_{MODEL_NAME}_correlations.npy"

    if os.path.exists(trained_file):
        print("✅ Found pre-computed TRAINED results.")
        corr_trained = np.load(trained_file)
    else:
        print("⚠️ Trained results not found. You should run reproduce_all_paradigms.py first.")
        return

    # B. UNTRAINED Model (The Baseline)
    print("Initializing UNTRAINED (Random) Network...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    # This creates the architecture but with random initialization
    model_random = AutoModel.from_config(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3. Extract & Train on Random Features
    feats_random = extract_features(texts, model_random, tokenizer, device)

    print("Training Ridge Regression on Random Features...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds_random = np.zeros_like(brain_data)
    model = RidgeCV(alphas=[100])  # Fast alpha

    for train_idx, test_idx in tqdm(kf.split(feats_random)):
        model.fit(feats_random[train_idx], brain_data[train_idx])
        preds_random[test_idx] = model.predict(feats_random[test_idx])

    # 4. Compute Correlations
    Y_true = brain_data - brain_data.mean(axis=0)
    Y_pred = preds_random - preds_random.mean(axis=0)
    numer = np.sum(Y_true * Y_pred, axis=0)
    denom = np.sqrt(np.sum(Y_true ** 2, axis=0)) * np.sqrt(np.sum(Y_pred ** 2, axis=0)) + 1e-8
    corr_random = numer / denom

    # 5. Visualization: Trained vs Untrained
    print("Generating Comparison Plot...")

    # Filter for valid voxels (where at least one model found signal)
    mask = (corr_trained > -1.0) | (corr_random > -1.0)
    valid_trained = corr_trained[mask]
    valid_random = corr_random[mask]

    plt.figure(figsize=(10, 6))

    # Scatter Plot
    plt.scatter(valid_random, valid_trained, alpha=0.3, s=10, color='purple')

    # Reference Line (x=y)
    max_val = max(np.max(valid_trained), np.max(valid_random))
    plt.plot([0, max_val], [0, max_val], 'k--', label="Equal Performance")

    plt.xlabel("Untrained (Random) Correlation")
    plt.ylabel("Trained (Pre-trained) Correlation")
    plt.title(f"Impact of Training: {MODEL_NAME} vs Random Baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = f"./results/{SUBJECT}_{PARADIGM}_baseline_comparison.png"
    plt.savefig(out_path)
    print(f"✅ Saved comparison plot to {out_path}")

    print(f"Average Trained: {np.mean(valid_trained):.4f}")
    print(f"Average Random:  {np.mean(valid_random):.4f}")


if __name__ == "__main__":
    run_baseline()