import os
import time

# --- Import functions from your other scripts ---

from preprocess_brain_data import process_glmsingle_output
from reproduce_all_paradigms import run_experiment
from visualize_detailed import create_visualizations
from visualize_layer_depth import create_layer_depth_map
# NEW: Import the refactored standard visualizer
from visualize_results import visualize_results

# ================= Configuration =================
SUBJECTS = ["M02", "M03", "M04", "M05"]
INPUT_DIRS = {
  "M01": "GLMsingle_outputs_M01-002",
  "M02": "GLMsingle_outputs_M02-008",
  "M03": "GLMsingle_outputs_M03-005",
  "M04":"GLMsingle_outputs_M04-016",
  "M05":"GLMsingle_outputs_M05-017"
}
PARADIGMS = ["sentences", "pictures", "word_clouds"]

# Path Configuration (Centralized)
CONFIG = {
    "input_dir": "./data/",
    "data_root": "./data",
    "results_dir": "./results",
    "stimuli_dir": "./data/stimuli",
    "image_dir": "./data/Pereira_Materials/Pereira_Materials/IARPA_expt1_stim_images",
    "mask_path": "./data/mask.volume.brainmask.nii"
}


# =============================================

def main():
    start_time = time.time()

    for subject in SUBJECTS:
        print(f"\n{'=' * 40}")
        print(f"🚀 STARTING PIPELINE FOR SUBJECT: {subject}")
        print(f"{'=' * 40}")

        # -------------------------------------------------
        # 1. Preprocessing
        # -------------------------------------------------
        print(f"\n--- 1. Preprocessing Data ---")
        for paradigm in PARADIGMS:
            try:
                process_glmsingle_output(
                    subject_id=subject,
                    paradigm=paradigm,
                    input_dir='./data/' + INPUT_DIRS[subject] + '/GLMsingle_outputs/',
                    output_dir=os.path.join(CONFIG["data_root"], "fMRI"),
                    mask_path=CONFIG["mask_path"]
                )
            except Exception as e:
                print(f"❌ Preprocessing failed for {subject}/{paradigm}: {e}")

        # -------------------------------------------------
        # 2. Main Loop (Alignment & Visualization)
        # -------------------------------------------------
        for paradigm in PARADIGMS:
            print(f"\n--- Processing Paradigm: {paradigm} ---")

            # A. Alignment (Train Models)
            try:
                run_experiment(
                    subject=subject,
                    paradigm=paradigm,
                    data_root=CONFIG["data_root"],
                    stimuli_dir=CONFIG["stimuli_dir"],
                    image_dir=CONFIG["image_dir"],
                    results_dir=CONFIG["results_dir"]
                )
            except Exception as e:
                print(f"⚠️ Alignment failed for {paradigm}: {e}")
                continue

                # B. Standard Visualization (Histograms & Ortho View) <-- NEW STEP
            try:
                visualize_results(
                    subject=subject,
                    paradigm=paradigm,
                    results_dir=CONFIG["results_dir"],
                    data_root=CONFIG["data_root"],
                    mask_path=CONFIG["mask_path"],
                    input_dir=CONFIG["input_dir"]
                )
            except Exception as e:
                print(f"⚠️ Standard Visualization failed for {paradigm}: {e}")

            # C. Interactive Visualization (HTML Plots)
            try:
                create_visualizations(
                    subject=subject,
                    paradigm=paradigm,
                    results_dir=CONFIG["results_dir"],
                    mask_path=CONFIG["mask_path"]
                )
            except Exception as e:
                print(f"⚠️ Interactive Visualization failed for {paradigm}: {e}")

            # D. Layer Depth Maps (Advanced Vis)
            try:
                create_layer_depth_map(
                    subject=subject,
                    paradigm=paradigm,
                    data_root=CONFIG["data_root"],
                    stimuli_dir=CONFIG["stimuli_dir"],
                    image_dir=CONFIG["image_dir"],
                    results_dir=CONFIG["results_dir"],
                    mask_path=CONFIG["mask_path"]
                )
            except Exception as e:
                print(f"⚠️ Layer map generation failed for {paradigm}: {e}")

    elapsed = time.time() - start_time
    print(f"\n✅ PIPELINE COMPLETE in {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()