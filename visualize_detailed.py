import os
import numpy as np
import nibabel as nib
from nilearn import plotting, datasets, surface

# ================= Configuration =================
# SELECT PARADIGM HERE: "sentences" (Text) OR "pictures" (Image)
PARADIGM = "word_clouds"
# PARADIGM = "sentences"

SUBJECT = "M01"
OUTPUT_DIR = f"./results/{PARADIGM}_detailed_plots"
MASK_PATH = "./data/mask.volume.brainmask.nii"

# Automatic Path Logic
# 1. Look for the finished map first
img_path_nii = f"./results/{SUBJECT}_{PARADIGM}_map.nii.gz"
# 2. If not found, look for the raw correlations (common for new runs)
img_path_npy = f"./results/{SUBJECT}_{PARADIGM}_clip_correlations.npy" if PARADIGM == "pictures" else f"./results/{SUBJECT}_{PARADIGM}_llama3_correlations.npy"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================================================

def load_and_prep_data():
    """
    Intelligently loads data:
    - If NIfTI (.nii.gz) exists, load it directly.
    - If only .npy exists, load it, apply mask, save as NIfTI, then load.
    """
    # Case 1: NIfTI Map already exists
    if os.path.exists(img_path_nii):
        print(f"✅ Found NIfTI map: {img_path_nii}")
        return nib.load(img_path_nii)

    # Case 2: Raw .npy exists -> needs conversion
    elif os.path.exists(img_path_npy):
        print(f"⚠️ NIfTI map not found. Converting raw correlations from: {img_path_npy}")

        if not os.path.exists(MASK_PATH):
            raise FileNotFoundError(f"❌ Cannot convert .npy without mask file at: {MASK_PATH}")

        # Load Raw Data
        correlations = np.load(img_path_npy)

        # Load Mask
        mask_img = nib.load(MASK_PATH)
        mask_data = mask_img.get_fdata() > 0

        # Safety Check: Size Mismatch
        if len(correlations) != np.sum(mask_data):
            print(f"⚠️ Warning: Data size ({len(correlations)}) != Mask size ({np.sum(mask_data)}).")
            # If mismatch, we try to create a new mask from raw data or trim
            # (Simplest fix for now: Assume user provided correct files or resize)
            if len(correlations) < np.sum(mask_data):
                # Pad with zeros if data is smaller
                temp = np.zeros(int(np.sum(mask_data)))
                temp[:len(correlations)] = correlations
                correlations = temp

        # Reconstruct 3D Volume
        vol_data = np.zeros(mask_data.shape)
        vol_data[mask_data] = correlations

        # Save as NIfTI for future use
        new_img = nib.Nifti1Image(vol_data, mask_img.affine)
        nib.save(new_img, img_path_nii)
        print(f"✅ Converted & Saved new map to: {img_path_nii}")
        return new_img

    else:
        raise FileNotFoundError(f"❌ No data found! Checked:\n   1. {img_path_nii}\n   2. {img_path_npy}")


def create_combined_surface_html(left_path, right_path, output_path):
    """
    Combines Left and Right hemispheres into one side-by-side view.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bilateral Surface View - {PARADIGM.capitalize()}</title>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background: #ffffff; }}
            .container {{ display: flex; width: 100vw; height: 100vh; }}
            .pane {{ flex: 1; border: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <iframe src="{os.path.basename(left_path)}" class="pane" title="Left Hemisphere"></iframe>
            <iframe src="{os.path.basename(right_path)}" class="pane" title="Right Hemisphere"></iframe>
        </div>
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Saved Combined Surface View to: {output_path}")


def create_clean_visualizations():
    # 1. Load Data (Handles .npy -> .nii automatically)
    img = load_and_prep_data()

    # ---------------------------------------------------------
    # 2. Standard Interactive Volume Viewer
    # ---------------------------------------------------------
    print("Generating Interactive Volume Viewer...")
    mni_template = datasets.load_mni152_template()

    html_view = plotting.view_img(
        img,
        threshold=0.05,
        vmax=0.2,  # Adjust this if CLIP correlations are higher (e.g., 0.4)
        cmap='coolwarm',
        bg_img=mni_template,
        title=f"Interactive View: {SUBJECT} {PARADIGM}"
    )

    vol_path = os.path.join(OUTPUT_DIR, f"{SUBJECT}_{PARADIGM}_interactive_volume.html")
    html_view.save_as_html(vol_path)
    print(f"✅ Saved Volume Viewer: {vol_path}")

    # ---------------------------------------------------------
    # 3. Surface Views (Low Shadows / Clean Look)
    # ---------------------------------------------------------
    print("Generating Surface Views...")
    fsaverage = datasets.fetch_surf_fsaverage()

    left_html_name = f"{SUBJECT}_{PARADIGM}_surface_L.html"
    right_html_name = f"{SUBJECT}_{PARADIGM}_surface_R.html"

    for hemi, filename in [('left', left_html_name), ('right', right_html_name)]:
        print(f"   Processing {hemi} hemisphere...")

        if hemi == 'left':
            mesh = fsaverage.pial_left
            bg_data = surface.load_surf_data(fsaverage.sulc_left)
        else:
            mesh = fsaverage.pial_right
            bg_data = surface.load_surf_data(fsaverage.sulc_right)

        # Low Shadow Logic (Clean Look)
        brightened_bg = (bg_data * 0.15) + 0.85

        texture = surface.vol_to_surf(img, mesh)

        surf_view = plotting.view_surf(
            mesh,
            texture,
            threshold=0.05,
            vmax=0.2,
            cmap='coolwarm',
            bg_map=brightened_bg,
            bg_on_data=False,
            title=f"{hemi.capitalize()} ({PARADIGM})"
        )

        surf_view.save_as_html(os.path.join(OUTPUT_DIR, filename))

    # Combine them
    combined_path = os.path.join(OUTPUT_DIR, f"{SUBJECT}_{PARADIGM}_surface_BOTH.html")
    create_combined_surface_html(
        os.path.join(OUTPUT_DIR, left_html_name),
        os.path.join(OUTPUT_DIR, right_html_name),
        combined_path
    )
    print(f"✅ Saved Surface View: {combined_path}")


if __name__ == "__main__":
    create_clean_visualizations()