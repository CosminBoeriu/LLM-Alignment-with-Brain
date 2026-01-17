import numpy as np
import os
import nibabel as nib  # You may need: pip install nibabel


def process_glmsingle_output(subject_id, paradigm, input_dir, output_dir, mask_path=None):
    """
    Converts GLMsingle 4D output to a 2D (samples x voxels) matrix.
    """
    # 1. Construct file path based on your screenshot structure
    filename = f"{subject_id}_{paradigm}_TYPED_FITHRF_GLMDENOISE_RR.npy"
    file_path = os.path.join(input_dir, filename)

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"Loading {filename}...")
    # Allow pickle=True because these are often dictionaries
    data = np.load(file_path, allow_pickle=True).item()

    # The key 'betasmd' usually contains the (X, Y, Z, Trials) data
    if 'betasmd' not in data:
        print(f"❌ 'betasmd' key not found in {filename}. Available keys: {list(data.keys())}")
        return

    betas_4d = data['betasmd']  # Shape expected: (X, Y, Z, Trials)
    print(f"  Raw shape: {betas_4d.shape}")

    # 2. Masking
    # We need to flatten 3D space (X,Y,Z) into 1D (Voxels)
    if mask_path and os.path.exists(mask_path):
        print(f"  Using mask: {mask_path}")
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata() > 0
    else:
        print("  ⚠️ Mask not found. Generating mask from non-zero variance voxels...")
        # Fallback: create mask where signal varies (i.e., inside the brain)
        variance_map = np.var(betas_4d, axis=-1)
        mask = variance_map > 1e-6  # Threshold for "signal exists"

    # Ensure mask shape matches betas shape (X, Y, Z)
    if betas_4d.shape[:3] != mask.shape:
        print(f"❌ Shape mismatch! Data: {betas_4d.shape[:3]}, Mask: {mask.shape}")
        return

    # 3. Apply mask
    # This flattens the first 3 dims -> (Voxels, Trials)
    betas_2d = betas_4d[mask]

    # 4. Transpose to (Trials, Voxels) for Scikit-Learn
    betas_final = betas_2d.T

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"{subject_id}_{paradigm}_betas.npy"
    out_path = os.path.join(output_dir, out_name)
    np.save(out_path, betas_final)
    print(f"✅ Saved processed data to {out_path} with shape {betas_final.shape}")


if __name__ == "__main__":
    # CONFIGURATION BASED ON YOUR SCREENSHOT
    # Path to the folder containing the .npy files
    INPUT_DIR = "./data/GLMsingle_outputs_M01-002/GLMsingle_outputs"

    # Where you want the clean files to go (matches the main script's expectation)
    OUTPUT_DIR = "./data/fMRI"

    # Path to the official mask if you have it (recommended)
    MASK_PATH = "./data/mask.volume.brainmask.nii"

    # 1. Run for SENTENCES
    process_glmsingle_output("M01", "sentences", INPUT_DIR, OUTPUT_DIR, MASK_PATH)

    # 2. Run for PICTURES
    process_glmsingle_output("M01", "pictures", INPUT_DIR, OUTPUT_DIR, MASK_PATH)

    # 3. Run for WORD CLOUDS (New Addition)
    process_glmsingle_output("M01", "word_clouds", INPUT_DIR, OUTPUT_DIR, MASK_PATH)