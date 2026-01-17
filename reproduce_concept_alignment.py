import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel, ViTModel, ViTImageProcessor


# ==========================================
# 1. Feature Extractors (The "Encoders")
# ==========================================

class Llama3FeatureExtractor:
    def __init__(self, model_name, layer=-1, device="cuda"):
        """
        Extracts text embeddings using Llama 3.
        Note: You need access approval from Meta on HuggingFace and a valid token.
        """
        print(f"Loading Llama 3 model: {model_name}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()
        self.layer = layer

        # Llama 3 needs a pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract(self, texts):
        """
        Extracts embeddings for a list of text strings.
        Returns: numpy array of shape (n_samples, hidden_dim)
        """
        embeddings = []
        batch_size = 16  # Adjust based on GPU memory

        print("Extracting text features...")
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get hidden states from the specified layer
            # shape: (batch, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states[self.layer]

            # Strategy: Use the representation of the last token (common for causal LLMs)
            # Alternatively, you can use mean pooling: hidden_states.mean(dim=1)
            last_token_idxs = inputs.attention_mask.sum(dim=1) - 1
            batch_embeddings = hidden_states[torch.arange(hidden_states.size(0)), last_token_idxs]

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)


class VisionFeatureExtractor:
    def __init__(self, model_type="clip", model_name="openai/clip-vit-base-patch32", device="cuda"):
        """
        Extracts image embeddings using CLIP or ViT.
        model_type: 'clip' or 'vit'
        """
        print(f"Loading Vision model: {model_name}...")
        self.device = device
        self.model_type = model_type.lower()

        if self.model_type == "clip":
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).vision_model  # Use only the vision tower
        elif self.model_type == "vit":
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def extract(self, image_paths):
        """
        Extracts embeddings for a list of image file paths.
        """
        embeddings = []
        batch_size = 32

        print("Extracting vision features...")
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use pooled output (CLS token representation)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                # Fallback for some ViT implementations: take first token (CLS)
                emb = outputs.last_hidden_state[:, 0, :]

            embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings)


# ==========================================
# 2. Brain Mapping (Ridge Regression)
# ==========================================

class BrainEncodingModel:
    def __init__(self, alphas=[1e-1, 1, 10, 100, 1000, 10000]):
        """
        Ridge Regression with internal Cross-Validation (RidgeCV).
        """
        self.model = RidgeCV(alphas=alphas)

    def evaluate(self, X, Y, n_splits=5):
        """
        Performs cross-validation to evaluate encoding performance.
        X: Stimulus features (n_samples, feature_dim)
        Y: Brain data (n_samples, n_voxels)

        Returns: Pearson correlation per voxel (n_voxels,)
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        n_voxels = Y.shape[1]
        correlations = np.zeros(n_voxels)

        print(f"Running {n_splits}-fold Cross-Validation Ridge Regression...")

        # We collect predictions for all samples to compute correlation once
        # (Or compute per fold and average. Here we predict test sets and concat).
        all_preds = np.zeros_like(Y)
        all_actual = np.zeros_like(Y)

        # Note: In standard encoding, we often just train on (n-1) folds and predict on 1,
        # then correlate the full concatenated prediction with full actual data.

        for train_idx, test_idx in tqdm(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Fit model
            self.model.fit(X_train, Y_train)

            # Predict
            preds = self.model.predict(X_test)

            all_preds[test_idx] = preds
            all_actual[test_idx] = Y_test

        # Calculate Pearson R for each voxel
        print("Calculating correlations...")
        for v in range(n_voxels):
            # Handle constant/zero outputs to avoid NaNs
            if np.std(all_preds[:, v]) == 0 or np.std(all_actual[:, v]) == 0:
                correlations[v] = 0
            else:
                corr, _ = pearsonr(all_preds[:, v], all_actual[:, v])
                correlations[v] = corr

        return correlations


# ==========================================
# 3. Main Reproduction Flow
# ==========================================

def run_experiment(subject_id="M01", paradigm="sentences", data_root="./data"):
    """
    Simulates the main pipeline.
    """

    # --- Step A: Load Stimuli ---
    # You must create/download a CSV with columns ['stimulus', 'type']
    # If paradigm is 'pictures', 'stimulus' should be the filename.
    stimuli_file = os.path.join(data_root, "stimuli", f"stimuli_order_{subject_id}_{paradigm}.csv")

    if not os.path.exists(stimuli_file):
        print(f"Error: Stimuli file not found at {stimuli_file}")
        return

    df = pd.read_csv(stimuli_file)
    print(f"Loaded {len(df)} stimuli for {paradigm}.")

    # --- Step B: Extract Features ---
    if paradigm in ["sentences", "word_clouds"]:
        # Use Llama 3
        # Note: 'stimulus' col should contain the text
        extractor = Llama3FeatureExtractor(model_name="gpt2-xl")
        features = extractor.extract(df['stimulus'].tolist())

    elif paradigm == "pictures":
        # Use CLIP or ViT
        # Note: 'stimulus' col should contain image filenames
        image_dir = os.path.join(data_root, "stimuli", "images")
        image_paths = [os.path.join(image_dir, f) for f in df['stimulus'].tolist()]

        # Switch to 'vit' if preferred
        extractor = VisionFeatureExtractor(model_type="clip", model_name="openai/clip-vit-base-patch32")
        features = extractor.extract(image_paths)

    print(f"Feature shape: {features.shape}")

    # --- Step C: Load Brain Data ---
    # Expecting numpy array of shape (n_samples, n_voxels)
    brain_file = os.path.join(data_root, "fMRI", f"{subject_id}_{paradigm}_betas.npy")

    if not os.path.exists(brain_file):
        print(f"Error: Brain data not found at {brain_file}")
        # creating dummy data for demonstration
        print("WARNING: Creating dummy brain data for demonstration.")
        brain_data = np.random.randn(len(df), 1000)  # 1000 fake voxels
    else:
        brain_data = np.load(brain_file)
        # Ensure shape matches
        if brain_data.shape[0] != features.shape[0]:
            # Sometimes brain data is (Voxels, Trials), transpose if needed
            if brain_data.shape[1] == features.shape[0]:
                brain_data = brain_data.T

    # --- Step D: Train & Evaluate ---
    encoder = BrainEncodingModel()
    correlations = encoder.evaluate(features, brain_data)

    print(f"Mean Encoding Performance (Pearson R): {np.mean(correlations):.4f}")
    print(f"Max Encoding Performance: {np.max(correlations):.4f}")

    # Save results
    os.makedirs("./results", exist_ok=True)
    np.save(f"./results/{subject_id}_{paradigm}_llama3_correlations.npy", correlations)
    print("Results saved.")


if __name__ == "__main__":
    # Example usage
    # Ensure you have logged in to huggingface-cli for Llama 3
    # huggingface-cli login

    run_experiment(subject_id="M01", paradigm="word_clouds")