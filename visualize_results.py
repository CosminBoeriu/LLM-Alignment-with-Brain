import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def visualize_results(subject, paradigm, results_dir, data_root, mask_path, input_dir):
    """
    Generates histograms and orthographic projections of the results.
    """
    # 1. Setup Paths
    subject_dir = os.path.join(results_dir, subject)
    os.makedirs(subject_dir, exist_ok=True)

    # Logic to find the correct correlation file
    if paradigm == "pictures":
        raw_name = f"{subject}_{paradigm}_clip_correlations.npy"
        model_name = "CLIP (Vision)"
    else:
        # Check if GPT-2 exists, otherwise Llama
        gpt_path = os.path.join(subject_dir, f"{subject}_{paradigm}_gpt2-xl_correlations.npy")
        if os.path.exists(gpt_path):
            raw_name = f"{subject}_{paradigm}_gpt2-xl_correlations.npy"
            model_name = "GPT-2 XL"
        else:
            raw_name = f"{subject}_{paradigm}_llama3_correlations.npy"
            model_name = "Llama 3"

    results_file = os.path.join(subject_dir, raw_name)

    # 2. Load Data
    if not os.path.exists(results_file):
        print(f"❌ Error: Could not find results file: {results_file}")
        return

    correlations = np.load(results_file)
    print(f"Loaded {len(correlations)} voxel correlations for {paradigm}.")

    # 3. Plot Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(correlations, bins=50, color='orange', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"Histogram of Brain Encoding Performance ({subject} - {paradigm})")
    plt.xlabel("Pearson Correlation (r)")
    plt.ylabel("Number of Voxels")

    hist_path = os.path.join(subject_dir, f"{subject}_{paradigm}_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"✅ Saved histogram to {hist_path}")

    # 4. Reconstruct 3D Brain Map (Masking Logic)
    mask = None
    affine = np.eye(4)

    if mask_path and os.path.exists(mask_path):
        # print(f"Using mask from {mask_path}")
        img = nib.load(mask_path)
        mask = img.get_fdata() > 0
        affine = img.affine
    else:
        print("⚠️ Mask file not found. Trying to regenerate from raw GLM outputs...")
        raw_file = os.path.join(input_dir, f"{subject}_{paradigm}_TYPED_FITHRF_GLMDENOISE_RR.npy")

        if not os.path.exists(raw_file):
            print(f"❌ Error: Need raw data at {raw_file} to reconstruct 3D map.")
            return

        raw_data = np.load(raw_file, allow_pickle=True).item()['betasmd']
        variance_map = np.var(raw_data, axis=-1)
        mask = variance_map > 1e-6
        affine = np.eye(4)

    # Sanity Check for Mismatches
    if np.sum(mask) != len(correlations):
        print(f"⚠️ Warning: Mask size ({np.sum(mask)}) != Result size ({len(correlations)}). Resizing...")
        temp_corr = np.zeros(int(np.sum(mask)))
        min_len = min(len(correlations), len(temp_corr))
        temp_corr[:min_len] = correlations[:min_len]
        correlations = temp_corr

    # 5. Project back to 3D volume & Save NIfTI
    brain_vol = np.zeros(mask.shape)
    brain_vol[mask] = correlations

    nifti_path = os.path.join(subject_dir, f"{subject}_{paradigm}_map.nii.gz")
    nifti_img = nib.Nifti1Image(brain_vol, affine)
    nib.save(nifti_img, nifti_path)
    print(f"✅ Saved 3D Brain Map to {nifti_path}")

    # 6. Plot Orthographic View
    max_loc = np.unravel_index(np.argmax(brain_vol), brain_vol.shape)
    x, y, z = max_loc

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = 'coolwarm'
    vmin, vmax = -0.2, 0.2

    # Sagittal
    axes[0].imshow(np.rot90(brain_vol[x, :, :]), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Sagittal (X={x})")
    axes[0].axis('off')

    # Coronal
    axes[1].imshow(np.rot90(brain_vol[:, y, :]), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Coronal (Y={y})")
    axes[1].axis('off')

    # Axial
    im = axes[2].imshow(np.rot90(brain_vol[:, :, z]), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Axial (Z={z})")
    axes[2].axis('off')

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Pearson Correlation (r)")

    plt.suptitle(f"{model_name} Alignment - {subject} ({paradigm})", fontsize=16)

    ortho_path = os.path.join(subject_dir, f"{subject}_{paradigm}_ortho_view.png")
    plt.savefig(ortho_path, dpi=150)
    plt.close()
    print(f"✅ Saved orthographic view to {ortho_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="M01")
    parser.add_argument("--paradigm", type=str, default="pictures")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--mask_path", type=str, default="./data/mask.volume.brainmask.nii")
    parser.add_argument("--input_dir", type=str, default="./data/GLMsingle_outputs_M01-002/GLMsingle_outputs")

    args = parser.parse_args()

    visualize_results(args.subject, args.paradigm, args.results_dir, args.data_root, args.mask_path, args.input_dir)