import os
import glob
import torch
import numpy as np
import pandas as pd
import argparse
import nibabel as nib
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPVisionModel
from nilearn import plotting, datasets, surface


def get_stimuli_data(subject, paradigm, stimuli_dir, image_dir, n_samples):
    if paradigm == "pictures":
        image_paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
        image_paths = [p for p in image_paths if "._" not in p]
        if len(image_paths) > n_samples:
            image_paths = image_paths[:n_samples]
        elif len(image_paths) < n_samples:
            image_paths += [image_paths[-1]] * (n_samples - len(image_paths))
        return image_paths
    else:
        stim_file = os.path.join(stimuli_dir, f"stimuli_order_{subject}_{paradigm}.csv")
        if os.path.exists(stim_file):
            return pd.read_csv(stim_file)['stimulus'].tolist()
        return [f"sample text {i}" for i in range(n_samples)]


def extract_features(inputs, model, processor, layer_idx, device, paradigm):
    embeddings = []
    batch_size = 16
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        with torch.no_grad():
            if paradigm == "pictures":
                loaded_images = []
                for p in batch:
                    try:
                        loaded_images.append(Image.open(p).convert("RGB"))
                    except:
                        loaded_images.append(Image.new('RGB', (224, 224)))
                pixel_inputs = processor(images=loaded_images, return_tensors="pt").to(device)
                outputs = model(**pixel_inputs)
                batch_emb = outputs.hidden_states[layer_idx][:, 0, :]
            else:
                token_inputs = processor(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                    device)
                outputs = model(**token_inputs)
                hidden_state = outputs.hidden_states[layer_idx]
                if processor.padding_side == "left":
                    batch_emb = hidden_state[:, -1, :]
                else:
                    last_token_idxs = token_inputs.attention_mask.sum(dim=1) - 1
                    batch_emb = hidden_state[torch.arange(hidden_state.size(0)), last_token_idxs]
        embeddings.append(batch_emb.cpu().numpy())
    return np.vstack(embeddings)


def create_layer_depth_map(subject, paradigm, data_root, stimuli_dir, image_dir, results_dir, mask_path):
    print(f"--- 🧠 Layer Depth Map: {subject} ({paradigm}) ---")

    # Paths
    subject_res_dir = os.path.join(results_dir, subject, "layer_depth_maps")
    os.makedirs(subject_res_dir, exist_ok=True)

    # Load Brain Data
    brain_file = os.path.join(data_root, "fMRI", subject, f"{subject}_{paradigm}_betas.npy")
    if not os.path.exists(brain_file):
        brain_file = os.path.join(data_root, "fMRI", f"{subject}_{paradigm}_betas.npy")
    brain_data = np.load(brain_file)
    stimuli = get_stimuli_data(subject, paradigm, stimuli_dir, image_dir, brain_data.shape[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if paradigm == "pictures":
        model_name = "openai/clip-vit-base-patch32"
        layers = [0, 2, 4, 6, 8, 10, 11]
        model = CLIPVisionModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    else:
        model_name = "gpt2-xl"
        layers = [0, 6, 12, 18, 24, 30, 36, 42, 48]
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        processor = AutoTokenizer.from_pretrained(model_name)
        if processor.pad_token is None: processor.pad_token = processor.eos_token
    model.eval()

    all_layer_scores = np.zeros((brain_data.shape[1], len(layers)))

    for i, layer in enumerate(tqdm(layers, desc="Layers")):
        feats = extract_features(stimuli, model, processor, layer, device, paradigm)
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        preds = np.zeros_like(brain_data)
        model_reg = RidgeCV(alphas=[100])
        for train, test in kf.split(feats):
            model_reg.fit(feats[train], brain_data[train])
            preds[test] = model_reg.predict(feats[test])

        # Correlation
        Y_true = brain_data - brain_data.mean(axis=0)
        Y_pred = preds - preds.mean(axis=0)
        num = np.sum(Y_true * Y_pred, axis=0)
        den = np.sqrt(np.sum(Y_true ** 2, axis=0)) * np.sqrt(np.sum(Y_pred ** 2, axis=0)) + 1e-8
        all_layer_scores[:, i] = num / den

    # Create Map
    best_layer_idx = np.argmax(all_layer_scores, axis=1)
    best_layer_map = np.array([layers[i] for i in best_layer_idx])
    best_layer_map[np.max(all_layer_scores, axis=1) < 0.02] = 0

    # Save NIfTI
    mask_img = nib.load(mask_path)
    mask_bool = mask_img.get_fdata() > 0
    final_vol = np.zeros(mask_bool.shape)
    flat_indices = np.where(mask_bool.flatten())[0]
    limit = min(len(best_layer_map), len(flat_indices))
    np.put(final_vol, flat_indices[:limit], best_layer_map[:limit])

    nii = nib.Nifti1Image(final_vol, mask_img.affine)
    nib.save(nii, os.path.join(subject_res_dir, f"{subject}_{paradigm}_best_layer_map.nii.gz"))
    print(f"✅ Saved NIfTI map.")

    # Surface View
    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(nii, fsaverage.pial_left)
    view = plotting.view_surf(fsaverage.pial_left, texture, cmap='jet', colorbar=True, threshold=0.1,
                              title=f"Layers: {paradigm}")
    view.save_as_html(os.path.join(subject_res_dir, f"{subject}_{paradigm}_layer_depth.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="M01")
    parser.add_argument("--paradigm", type=str, default="pictures")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--stimuli_dir", type=str, default="./data/stimuli")
    parser.add_argument("--image_dir", type=str, default="./Pereira_Materials/IARPA_expt1_stim_images")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--mask_path", type=str, default="./data/mask.volume.brainmask.nii")
    args = parser.parse_args()

    create_layer_depth_map(args.subject, args.paradigm, args.data_root, args.stimuli_dir, args.image_dir,
                           args.results_dir, args.mask_path)