import os
import glob
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPVisionModel
from nilearn import plotting, datasets, surface

# ================= Configuration =================
SUBJECT = "M01"
# Change this to "sentences" or "pictures"
PARADIGM = "pictures"

DATA_ROOT = "./data"
STIMULI_DIR = "./data/stimuli"
IMAGE_DIR = "./data/Pereira_Materials/Pereira_Materials/IARPA_expt1_stim_images/"
OUTPUT_DIR = "./results/layer_depth_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Model Selection Logic
if PARADIGM == "pictures":
    # CLIP ViT-Base has 12 layers
    MODEL_NAME = "openai/clip-vit-base-patch32"
    LAYERS_TO_TEST = [0, 2, 4, 6, 8, 10, 11]
else:
    # GPT-2 XL has 48 layers
    MODEL_NAME = "gpt2-xl"
    LAYERS_TO_TEST = [0, 6, 12, 18, 24, 30, 36, 42, 48]


# =============================================

def get_brain_data():
    brain_file = os.path.join(DATA_ROOT, "fMRI", f"{SUBJECT}_{PARADIGM}_betas.npy")
    if not os.path.exists(brain_file):
        raise FileNotFoundError(f"Run preprocess_brain_data.py for {PARADIGM} first.")
    return np.load(brain_file)


def get_stimuli_data(n_samples):
    """
    Returns either a list of texts (for sentences) or list of image paths (for pictures).
    """
    if PARADIGM == "pictures":
        # Find images
        image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "**", "*.jpg"), recursive=True))
        image_paths = [p for p in image_paths if "._" not in p]

        # Sync with brain data size
        if len(image_paths) > n_samples:
            image_paths = image_paths[:n_samples]
        elif len(image_paths) < n_samples:
            print(f"⚠️ Warning: Found {len(image_paths)} images but {n_samples} brain samples.")
            # Pad with last image to avoid crash (visualization might be slightly off at end)
            image_paths += [image_paths[-1]] * (n_samples - len(image_paths))

        return image_paths

    else:
        # Load Text
        stim_file = os.path.join(STIMULI_DIR, f"stimuli_order_{SUBJECT}_{PARADIGM}.csv")
        if os.path.exists(stim_file):
            df = pd.read_csv(stim_file)
            return df['stimulus'].tolist()
        else:
            print("⚠️ Stimuli file not found. Using dummy text.")
            return [f"sample text {i}" for i in range(n_samples)]


def extract_features(inputs, model, processor_or_tokenizer, layer_idx, device):
    """
    Universal extractor for both Text (GPT) and Vision (CLIP).
    """
    embeddings = []
    batch_size = 16

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]

        with torch.no_grad():
            if PARADIGM == "pictures":
                # --- VISION PATH ---
                # Load images
                loaded_images = []
                for p in batch:
                    try:
                        loaded_images.append(Image.open(p).convert("RGB"))
                    except:
                        # Fallback for corrupt image
                        loaded_images.append(Image.new('RGB', (224, 224)))

                # Process with CLIPProcessor
                pixel_inputs = processor_or_tokenizer(images=loaded_images, return_tensors="pt").to(device)
                outputs = model(**pixel_inputs)

                # CLIP Vision hidden states: (Batch, Sequence, Hidden)
                # Index 0 is embeddings, Index 1 is Layer 1...
                hidden_state = outputs.hidden_states[layer_idx]

                # For ViT, the first token [CLS] represents the whole image
                batch_emb = hidden_state[:, 0, :]

            else:
                # --- TEXT PATH ---
                token_inputs = processor_or_tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                                      max_length=128).to(device)
                outputs = model(**token_inputs)

                hidden_state = outputs.hidden_states[layer_idx]

                if processor_or_tokenizer.padding_side == "left":
                    batch_emb = hidden_state[:, -1, :]
                else:
                    last_token_idxs = token_inputs.attention_mask.sum(dim=1) - 1
                    batch_emb = hidden_state[torch.arange(hidden_state.size(0)), last_token_idxs]

        embeddings.append(batch_emb.cpu().numpy())

    return np.vstack(embeddings)


def create_layer_depth_map():
    print(f"--- 🧠 Generating Layer Depth Map: {SUBJECT} ({PARADIGM}) ---")

    # 1. Load Data
    brain_data = get_brain_data()
    stimuli = get_stimuli_data(brain_data.shape[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Initialize correct model type
    if PARADIGM == "pictures":
        print(f"Loading Vision Model: {MODEL_NAME}")
        # Use CLIPVisionModel to isolate the vision tower
        model = CLIPVisionModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        tokenizer = processor  # Alias for the function call
    else:
        print(f"Loading Text Model: {MODEL_NAME}")
        model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    n_voxels = brain_data.shape[1]
    all_layer_scores = np.zeros((n_voxels, len(LAYERS_TO_TEST)))

    # 3. Sweep Through Layers
    for i, layer in enumerate(tqdm(LAYERS_TO_TEST, desc="Testing Layers")):
        features = extract_features(stimuli, model, tokenizer, layer, device)

        # Fit Ridge (2-Fold for speed/validity)
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        preds = np.zeros_like(brain_data)
        ridge = RidgeCV(alphas=[100])

        for train_idx, test_idx in kf.split(features):
            ridge.fit(features[train_idx], brain_data[train_idx])
            preds[test_idx] = ridge.predict(features[test_idx])

        # Compute Correlation
        Y_true = brain_data - brain_data.mean(axis=0)
        Y_pred = preds - preds.mean(axis=0)
        numer = np.sum(Y_true * Y_pred, axis=0)
        denom = np.sqrt(np.sum(Y_true ** 2, axis=0)) * np.sqrt(np.sum(Y_pred ** 2, axis=0)) + 1e-8
        corrs = numer / denom

        all_layer_scores[:, i] = corrs

    # 4. Create "Best Layer" Map
    best_layer_indices = np.argmax(all_layer_scores, axis=1)
    best_layer_map = np.array([LAYERS_TO_TEST[i] for i in best_layer_indices])

    # Mask noise
    max_corrs = np.max(all_layer_scores, axis=1)
    # Lower threshold for layer mapping to show more gradients
    best_layer_map[max_corrs < 0.02] = 0

    # 5. Save NIfTI
    mask_path = "./data/mask.volume.brainmask.nii"  # Ensure this path is correct
    if not os.path.exists(mask_path):
        # Fallback logic if mask is missing (unlikely if previous steps worked)
        print("⚠️ Mask missing, skipping NIfTI save.")
        return

    mask_img = nib.load(mask_path)
    mask_bool = mask_img.get_fdata() > 0
    final_vol = np.zeros(mask_bool.shape)

    # Fill mask
    flat_indices = np.where(mask_bool.flatten())[0]
    if len(best_layer_map) == len(flat_indices):
        np.put(final_vol, flat_indices, best_layer_map)
    else:
        # Mismatch handling: Trim or pad
        print(f"⚠️ Size mismatch: Data {len(best_layer_map)} vs Mask {len(flat_indices)}. Resizing...")
        limit = min(len(best_layer_map), len(flat_indices))
        np.put(final_vol, flat_indices[:limit], best_layer_map[:limit])

    nii = nib.Nifti1Image(final_vol, mask_img.affine)
    nib.save(nii, f"{OUTPUT_DIR}/{SUBJECT}_{PARADIGM}_best_layer_map.nii.gz")

    # 6. Visualize Surface
    print("Generating Surface Plot...")
    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(nii, fsaverage.pial_left)

    view = plotting.view_surf(
        fsaverage.pial_left,
        texture,
        cmap='jet',  # Jet is good for layer depth (Blue=Early, Red=Deep)
        colorbar=True,
        threshold=0.1,
        title=f"Best Layer Map: {PARADIGM} (Blue=Early, Red=Deep)"
    )

    out_html = f"{OUTPUT_DIR}/{SUBJECT}_{PARADIGM}_layer_depth.html"
    view.save_as_html(out_html)
    print(f"✅ Open {out_html} to see results!")


if __name__ == "__main__":
    create_layer_depth_map()