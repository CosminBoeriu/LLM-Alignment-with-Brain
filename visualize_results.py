import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib  # pip install nibabel

# ==========================================
# Configuration
# ==========================================
SUBJECT = "M01"
PARADIGM = "pictures"  # <--- CHANGED TO PICTURES
# Automatically select the correct results file based on paradigm
if PARADIGM == "pictures":
    RESULTS_FILE = f"./results/{SUBJECT}_{PARADIGM}_clip_correlations.npy"
    MODEL_NAME = "CLIP (Vision)"
elif PARADIGM == "sentences":
    RESULTS_FILE = f"./results/{SUBJECT}_{PARADIGM}_llama3_correlations.npy"
    MODEL_NAME = "Llama 3 (Text)"
else:
    RESULTS_FILE = f"./results/{SUBJECT}_{PARADIGM}_llama3_correlations.npy"
    MODEL_NAME = "Llama 3 (WORDCLOUDS)"


ORIGINAL_DATA_DIR = "./data/GLMsingle_outputs_M01-002/GLMsingle_outputs"
MASK_PATH = "./data/mask.volume.brainmask.nii"

def visualize():
    # 1. Load the correlations
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: Could not find {RESULTS_FILE}")
        return
    correlations = np.load(RESULTS_FILE)
    print(f"Loaded {len(correlations)} voxel correlations for {PARADIGM}.")

    # 2. Plot Histogram (Sanity Check)
    plt.figure(figsize=(10, 5))
    plt.hist(correlations, bins=50, color='orange', edgecolor='black', alpha=0.7) # Changed color to distinguish
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"Histogram of Brain Encoding Performance ({SUBJECT} - {PARADIGM})")
    plt.xlabel("Pearson Correlation (r)")
    plt.ylabel("Number of Voxels")
    plt.savefig(f"./results/{SUBJECT}_{PARADIGM}_histogram.png")
    print(f"Saved histogram to ./results/{SUBJECT}_{PARADIGM}_histogram.png")

    # 3. Reconstruct 3D Brain Map
    mask = None
    affine = np.eye(4)

    if os.path.exists(MASK_PATH):
        print(f"Using mask from {MASK_PATH}")
        img = nib.load(MASK_PATH)
        mask = img.get_fdata() > 0
        affine = img.affine
    else:
        print("Mask file not found. Regenerating mask from raw data...")
        # Note: Ensure you have the raw picture data if mask is missing
        raw_file = f"{ORIGINAL_DATA_DIR}/{SUBJECT}_{PARADIGM}_TYPED_FITHRF_GLMDENOISE_RR.npy"
        if not os.path.exists(raw_file):
            print(f"Error: Need raw data at {raw_file} to reconstruct 3D map.")
            return

        raw_data = np.load(raw_file, allow_pickle=True).item()['betasmd']
        variance_map = np.var(raw_data, axis=-1)
        mask = variance_map > 1e-6
        affine = np.eye(4)

    # Sanity Check for Mismatches
    if np.sum(mask) != len(correlations):
        print(f"WARNING: Mask size ({np.sum(mask)}) != Result size ({len(correlations)}).")
        print("Trimming or padding to fit...")
        temp_corr = np.zeros(int(np.sum(mask)))
        min_len = min(len(correlations), len(temp_corr))
        temp_corr[:min_len] = correlations[:min_len]
        correlations = temp_corr

    # 4. Project back to 3D volume
    brain_vol = np.zeros(mask.shape)
    brain_vol[mask] = correlations

    # 5. Save as NIfTI
    save_path = f"./results/{SUBJECT}_{PARADIGM}_map.nii.gz"
    nifti_img = nib.Nifti1Image(brain_vol, affine)
    nib.save(nifti_img, save_path)
    print(f"Saved 3D Brain Map to {save_path}")

    # 6. Plot Orthographic View (All 3 Axes)
    # Find the location of the absolute maximum value
    max_loc = np.unravel_index(np.argmax(brain_vol), brain_vol.shape)
    x, y, z = max_loc

    print(f"Max correlation found at voxel coordinates: X={x}, Y={y}, Z={z}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Settings
    cmap = 'coolwarm'
    vmin, vmax = -0.2, 0.2

    # -- 1. Sagittal View (Side) --
    sagittal = np.rot90(brain_vol[x, :, :])
    axes[0].imshow(sagittal, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Sagittal (Side)\nSlice X={x}")
    axes[0].axis('off')

    # -- 2. Coronal View (Front) --
    coronal = np.rot90(brain_vol[:, y, :])
    axes[1].imshow(coronal, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Coronal (Front)\nSlice Y={y}")
    axes[1].axis('off')

    # -- 3. Axial View (Top) --
    axial = np.rot90(brain_vol[:, :, z])
    im = axes[2].imshow(axial, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Axial (Top)\nSlice Z={z}")
    axes[2].axis('off')

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Pearson Correlation (r)")

    plt.suptitle(f"{MODEL_NAME} Brain Alignment - {SUBJECT} ({PARADIGM})", fontsize=16)

    # Save image
    save_path = f"./results/{SUBJECT}_{PARADIGM}_ortho_view.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved orthographic view to {save_path}")

if __name__ == "__main__":
    visualize()