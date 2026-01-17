import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel, CLIPVisionModel

# ================= Configuration =================
SUBJECT = "M01"
# Choose one: "sentences", "word_clouds", "pictures"
PARADIGM = "word_clouds"

DATA_ROOT = "./data"
IMAGE_DIR = "./Pereira_Materials/IARPA_expt1_stim_images"
STIMULI_DIR = "./data/stimuli"  # Where your CSVs are


# =============================================

class TextFeatureExtractor:
    def __init__(self, model_name="gpt2-xl", layer=-1, device="cuda"):
        print(f"Loading Text Model: {model_name}...")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.model.eval()
        self.layer = layer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract(self, texts):
        embeddings = []
        batch_size = 8
        print(f"Extracting features for {len(texts)} texts...")
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs.hidden_states[self.layer]
            # Extract last token (EOS) for representation
            if self.tokenizer.padding_side == "left":
                batch_embeddings = hidden_states[:, -1, :]
            else:
                last_token_idxs = inputs.attention_mask.sum(dim=1) - 1
                batch_embeddings = hidden_states[torch.arange(hidden_states.size(0)), last_token_idxs]

            embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings)


class VisionFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        print(f"Loading Vision Model: {model_name}...")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def extract(self, image_paths):
        embeddings = []
        batch_size = 32
        print(f"Extracting features for {len(image_paths)} images...")
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except:
                    print(f"Skipping {p}")
            if not images: continue

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.pooler_output.cpu().numpy())
        return np.vstack(embeddings)


def run_experiment():
    print(f"\n--- 🧠 Running Experiment: {SUBJECT} - {PARADIGM} ---")

    # 1. Load Brain Data
    brain_file = os.path.join(DATA_ROOT, "fMRI", f"{SUBJECT}_{PARADIGM}_betas.npy")
    if not os.path.exists(brain_file):
        print(f"❌ Brain data missing: {brain_file}\nRun preprocess_brain_data.py first.")
        return
    brain_data = np.load(brain_file)
    print(f"Loaded Brain Data: {brain_data.shape}")

    # 2. Extract Features
    if PARADIGM in ["sentences", "word_clouds"]:
        # --- TEXT FLOW ---
        # Look for stimuli csv
        stim_file = os.path.join(STIMULI_DIR, f"stimuli_order_{SUBJECT}_{PARADIGM}.csv")

        # Fallback: If no CSV, look for raw text files in Pereira Materials if available
        # But usually you need to construct the CSV first.
        if not os.path.exists(stim_file):
            print(f"❌ Stimuli CSV missing: {stim_file}")
            # DUMMY FALLBACK for demonstration if you don't have the CSV yet
            print("⚠️ Generating dummy text data for testing flow...")
            texts = [f"sample text {i}" for i in range(brain_data.shape[0])]
        else:
            df = pd.read_csv(stim_file)
            texts = df['stimulus'].tolist()

        extractor = TextFeatureExtractor(model_name="gpt2-xl")  # Using GPT2-XL as robust text baseline
        features = extractor.extract(texts)
        model_name = "gpt2-xl"

    elif PARADIGM == "pictures":
        # --- VISION FLOW ---
        image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "**", "*.jpg"), recursive=True))
        # Filter hidden files
        image_paths = [p for p in image_paths if "._" not in p]

        # Truncate to match brain data
        n_samples = min(len(image_paths), brain_data.shape[0])
        image_paths = image_paths[:n_samples]
        brain_data = brain_data[:n_samples]

        extractor = VisionFeatureExtractor()
        features = extractor.extract(image_paths)
        model_name = "clip"

    # 3. Alignment (Ridge Regression)
    print(f"Training Ridge Regression ({features.shape[0]} samples)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.zeros_like(brain_data)
    model = RidgeCV(alphas=[1e-1, 1, 10, 100, 1000])

    for train_idx, test_idx in tqdm(kf.split(features)):
        model.fit(features[train_idx], brain_data[train_idx])
        predictions[test_idx] = model.predict(features[test_idx])

    # 4. Correlation
    print("Computing correlations...")
    # Fast vectorized correlation
    Y_true = brain_data - brain_data.mean(axis=0)
    Y_pred = predictions - predictions.mean(axis=0)
    numer = np.sum(Y_true * Y_pred, axis=0)
    denom = np.sqrt(np.sum(Y_true ** 2, axis=0)) * np.sqrt(np.sum(Y_pred ** 2, axis=0)) + 1e-8
    correlations = numer / denom

    # 5. Save
    os.makedirs("./results", exist_ok=True)
    out_file = f"./results/{SUBJECT}_{PARADIGM}_{model_name}_correlations.npy"
    np.save(out_file, correlations)
    print(f"✅ Saved results to {out_file}")
    print(f"Mean r: {np.mean(correlations):.4f} | Max r: {np.max(correlations):.4f}")


if __name__ == "__main__":
    run_experiment()