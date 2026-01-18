import numpy as np
import os
import argparse
import nibabel as nib


def process_glmsingle_output(subject_id, paradigm, input_dir, output_dir, mask_path=None):
    """
    Converts GLMsingle 4D output to a 2D (samples x voxels) matrix.
    """
    filename = f"{subject_id}_{paradigm}_TYPED_FITHRF_GLMDENOISE_RR.npy"
    file_path = os.path.join(input_dir, filename)

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"Loading {filename}...")
    data = np.load(file_path, allow_pickle=True).item()

    if 'betasmd' not in data:
        print(f"❌ 'betasmd' key not found in {filename}. Available keys: {list(data.keys())}")
        return

    betas_4d = data['betasmd']
    print(f"  Raw shape: {betas_4d.shape}")

    # Masking
    if mask_path and os.path.exists(mask_path):
        print(f"  Using mask: {mask_path}")
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata() > 0
    else:
        print("  ⚠️ Mask not found. Generating mask from non-zero variance voxels...")
        variance_map = np.var(betas_4d, axis=-1)
        mask = variance_map > 1e-6

    if betas_4d.shape[:3] != mask.shape:
        print(f"❌ Shape mismatch! Data: {betas_4d.shape[:3]}, Mask: {mask.shape}")
        return

    betas_2d = betas_4d[mask]
    betas_final = betas_2d.T

    # Save to subject specific folder
    subject_output_dir = os.path.join(output_dir, subject_id)  # Organized by subject
    os.makedirs(subject_output_dir, exist_ok=True)

    out_name = f"{subject_id}_{paradigm}_betas.npy"
    out_path = os.path.join(subject_output_dir, out_name)
    np.save(out_path, betas_final)
    print(f"✅ Saved processed data to {out_path} with shape {betas_final.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GLMsingle brain data.")
    parser.add_argument("--subject", type=str, default="M01", help="Subject ID (e.g., M01)")
    parser.add_argument("--paradigm", type=str, default="all",
                        help="Paradigm: 'sentences', 'pictures', 'word_clouds', or 'all'")
    parser.add_argument("--input_dir", type=str, default="./data/GLMsingle_outputs_M01-002/GLMsingle_outputs")
    parser.add_argument("--output_dir", type=str, default="./data/fMRI")
    parser.add_argument("--mask_path", type=str, default="./data/mask.volume.brainmask.nii")

    args = parser.parse_args()

    paradigms_to_run = ["sentences", "pictures", "word_clouds"] if args.paradigm == "all" else [args.paradigm]

    for p in paradigms_to_run:
        process_glmsingle_output(args.subject, p, args.input_dir, args.output_dir, args.mask_path)