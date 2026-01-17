import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel

# ================= Configuration =================
SUBJECT = "M01"
PARADIGM = "pictures"
DATA_ROOT = "./data"
# UPDATE THIS PATH to where your images folder is located
IMAGE_DIR = "./data/Pereira_Materials/Pereira_Materials/IARPA_expt1_stim_images/"


# =============================================

class VisionFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        print(f"Loading Vision Model: {model_name}...")
        self.device = device if torch.cuda.is_available() else "cpu"

        # We use the CLIP Vision Model specifically
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
            valid_batch_indices = []

            # Load images (skipping any corrupt ones)
            for idx, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    valid_batch_indices.append(idx)
                except Exception as e:
                    print(f"Warning: Could not load {p}")

            if not images:
                continue

            # Preprocess for CLIP
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use the "pooled" output (CLS token), which represents the whole image
            emb = outputs.pooler_output.cpu().numpy()
            embeddings.append(emb)

        return np.vstack(embeddings)


def run_vision_experiment():
    # 1. Load Brain Data
    brain_file = os.path.join(DATA_ROOT, "fMRI", f"{SUBJECT}_{PARADIGM}_betas.npy")
    if not os.path.exists(brain_file):
        print(f"❌ Error: Brain data not found at {brain_file}")
        print("Did you run preprocess_brain_data.py for 'pictures'?")
        return

    brain_data = np.load(brain_file)
    print(f"Loaded Brain Data: {brain_data.shape} (Samples, Voxels)")

    # 2. Locate Images
    # The Pereira dataset images are usually organized in subfolders or flat.
    # We need to find all .jpg files.
    # CRITICAL: The order of images must match the brain data trials.
    # For this reproduction, we will sort them alphanumerically as a best guess
    # if you don't have the specific 'stimuli_order.csv'.
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "**", "*.jpg"), recursive=True))

    # Filter out hidden files or thumbnails if any
    image_paths = [p for p in image_paths if "._" not in p]

    print(f"Found {len(image_paths)} images.")

    # Safety Check: Length Mismatch
    # If the number of images doesn't match brain trials, we trim to the smaller one
    n_samples = min(len(image_paths), brain_data.shape[0])
    if len(image_paths) != brain_data.shape[0]:
        print(f"⚠️ Mismatch! Images: {len(image_paths)}, Brain Trials: {brain_data.shape[0]}")
        print(f"   Using the first {n_samples} samples for alignment.")

    image_paths = image_paths[:n_samples]
    brain_data = brain_data[:n_samples]

    # 3. Extract Features (CLIP)
    extractor = VisionFeatureExtractor()
    features = extractor.extract(image_paths)
    print(f"Feature Matrix Shape: {features.shape}")

    # 4. Train Ridge Regression (Brain Encoding)
    print("Training Ridge Regression (5-fold CV)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    correlations = np.zeros(brain_data.shape[1])

    # Scikit-Learn RidgeCV is efficient enough to handle this without a manual loop
    # but we loop folds to get the true cross-validated correlation

    model = RidgeCV(alphas=[1e-1, 1, 10, 100, 1000])

    predictions = np.zeros_like(brain_data)

    for train_idx, test_idx in tqdm(kf.split(features)):
        X_train, X_test = features[train_idx], features[test_idx]
        Y_train, Y_test = brain_data[train_idx], brain_data[test_idx]

        model.fit(X_train, Y_train)
        predictions[test_idx] = model.predict(X_test)

    # 5. Compute Correlation per Voxel
    print("Computing Correlations...")
    # Vectorized correlation calculation for speed
    # (Pearson R is the cosine similarity of centered vectors)
    Y_true_centered = brain_data - brain_data.mean(axis=0)
    Y_pred_centered = predictions - predictions.mean(axis=0)

    # Normalize
    Y_true_norm = np.sqrt(np.sum(Y_true_centered ** 2, axis=0))
    Y_pred_norm = np.sqrt(np.sum(Y_pred_centered ** 2, axis=0))

    # Avoid division by zero
    epsilon = 1e-8
    correlations = np.sum(Y_true_centered * Y_pred_centered, axis=0) / (Y_true_norm * Y_pred_norm + epsilon)

    # Save Results
    os.makedirs("./results", exist_ok=True)
    out_file = f"./results/{SUBJECT}_{PARADIGM}_clip_correlations.npy"
    np.save(out_file, correlations)

    print(f"✅ Done! Results saved to {out_file}")
    print(f"   Mean Correlation: {np.mean(correlations):.4f}")
    print(f"   Max Correlation:  {np.max(correlations):.4f}")


if __name__ == "__main__":
    run_vision_experiment()