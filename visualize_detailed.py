import os
import argparse
import numpy as np
import nibabel as nib
from nilearn import plotting, datasets, surface


def load_and_prep_data(subject, paradigm, results_dir, mask_path):
    # Construct paths
    subject_dir = os.path.join(results_dir, subject)

    # Files
    img_path_nii = os.path.join(subject_dir, f"{subject}_{paradigm}_map.nii.gz")

    # Determine raw file name based on paradigm
    if paradigm == "pictures":
        raw_name = f"{subject}_{paradigm}_clip_correlations.npy"
    elif paradigm in ["sentences", "word_clouds"]:
        raw_name = f"{subject}_{paradigm}_gpt2-xl_correlations.npy"
        if not os.path.exists(os.path.join(subject_dir, raw_name)):
            # Fallback to llama3 if gpt2 not found
            raw_name = f"{subject}_{paradigm}_llama3_correlations.npy"

    img_path_npy = os.path.join(subject_dir, raw_name)

    # 1. Try Loading NIfTI
    if os.path.exists(img_path_nii):
        print(f"✅ Found NIfTI map: {img_path_nii}")
        return nib.load(img_path_nii)

    # 2. Try Loading Raw .npy and converting
    elif os.path.exists(img_path_npy):
        print(f"⚠️ NIfTI map not found. Converting from: {img_path_npy}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"❌ Mask missing: {mask_path}")

        correlations = np.load(img_path_npy)
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata() > 0

        # Safety Check
        if len(correlations) != np.sum(mask_data):
            print(f"⚠️ Warning: Size mismatch. Data {len(correlations)} vs Mask {np.sum(mask_data)}")
            if len(correlations) < np.sum(mask_data):
                temp = np.zeros(int(np.sum(mask_data)))
                temp[:len(correlations)] = correlations
                correlations = temp

        vol_data = np.zeros(mask_data.shape)
        vol_data[mask_data] = correlations

        new_img = nib.Nifti1Image(vol_data, mask_img.affine)
        nib.save(new_img, img_path_nii)
        print(f"✅ Saved NIfTI to: {img_path_nii}")
        return new_img
    else:
        raise FileNotFoundError(f"❌ No data found in {subject_dir}")


def create_visualizations(subject, paradigm, results_dir, mask_path):
    img = load_and_prep_data(subject, paradigm, results_dir, mask_path)

    # Output Folder
    output_dir = os.path.join(results_dir, subject, "detailed_plots", paradigm)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating visualizations in {output_dir}...")
    mni = datasets.load_mni152_template()

    # 1. Volume Viewer
    html_view = plotting.view_img(img, threshold=0.05, vmax=0.2, cmap='coolwarm', bg_img=mni,
                                  title=f"{subject} {paradigm}")
    html_view.save_as_html(os.path.join(output_dir, "interactive_volume.html"))

    # 2. Surface Views
    fsaverage = datasets.fetch_surf_fsaverage()

    # Combined HTML container
    combined_html_path = os.path.join(output_dir, "surface_BOTH.html")

    for hemi in ['left', 'right']:
        mesh = fsaverage.pial_left if hemi == 'left' else fsaverage.pial_right
        bg_data = surface.load_surf_data(fsaverage.sulc_left if hemi == 'left' else fsaverage.sulc_right)
        bright_bg = (bg_data * 0.15) + 0.85

        texture = surface.vol_to_surf(img, mesh)
        surf_view = plotting.view_surf(mesh, texture, threshold=0.05, vmax=0.2, cmap='coolwarm', bg_map=bright_bg,
                                       bg_on_data=False, title=hemi)
        surf_view.save_as_html(os.path.join(output_dir, f"surface_{hemi}.html"))

    print(f"✅ Visualizations saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="M01")
    parser.add_argument("--paradigm", type=str, default="word_clouds")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--mask_path", type=str, default="./data/mask.volume.brainmask.nii")
    args = parser.parse_args()

    create_visualizations(args.subject, args.paradigm, args.results_dir, args.mask_path)