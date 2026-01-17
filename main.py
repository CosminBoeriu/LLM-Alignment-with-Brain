import os
import time
import subprocess

# ================= Configuration =================
# List of subjects to process
SUBJECTS = ["M01", "M02", "M03", "M04", "M05"]

# List of paradigms to run for each subject
# Options: "sentences", "pictures", "word_clouds"
PARADIGMS = ["sentences", "pictures", "word_clouds"]

RUN_PREPROCESSING = True  # Converts GLMsingle .npy -> fMRI .npy
RUN_ALIGNMENT = True  # Trains Ridge Regression (GPT-2 / CLIP)
RUN_VISUALIZATION = True  # Generates HTML plots and Ortho views
RUN_LAYERS = True  # Generates Layer-wise plots and Depth maps
RUN_BASELINE = True  # Generates Random vs Trained comparison

PYTHON_EXEC = "python"


# =============================================

def run_command(cmd, step_name):
    """Helper to run a shell command and handle errors."""
    print(f"\n[pipeline] Starting: {step_name}...")
    try:
        # Run the command and wait for it to finish
        subprocess.check_call(cmd, shell=True)
        print(f"[pipeline] ✅ Finished: {step_name}")
    except subprocess.CalledProcessError as e:
        print(f"[pipeline] ❌ FAILED: {step_name}")
        print(f"Error command: {e.cmd}")
        exit(1)


def update_config_in_file(filepath, variable, new_value):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip().startswith(f"{variable} ="):
                # Preserve indentation if any, though usually these are top-level
                f.write(f'{variable} = "{new_value}"\n')
            else:
                f.write(line)


def main():
    start_time = time.time()

    for subject in SUBJECTS:
        print(f"\n{'=' * 40}")
        print(f"🚀 PROCESSING SUBJECT: {subject}")
        print(f"{'=' * 40}")

        # Create subject directory if it doesn't exist
        subject_dir = f"./results/{subject}"
        os.makedirs(subject_dir, exist_ok=True)

        # -------------------------------------------------
        # 1. Preprocessing (One-time setup per subject)
        # -------------------------------------------------
        if RUN_PREPROCESSING:
            # We assume preprocess_brain_data.py handles the loop internally 
            # or we need to update it. Let's update it to be safe.
            # However, your preprocess script currently runs ALL paradigms at once.
            # So we only need to run it once per subject.

            # NOTE: You might need to modify preprocess_brain_data.py to accept a subject
            # strictly via variable, or just run it as is if it loops internally.
            # For safety, we will assume it needs the variable update.
            # Check preprocess_brain_data.py structure first.
            pass
            # (Skipping dynamic update for preprocess since it usually hardcodes the loop
            # as seen in your previous snippets. If needed, modify it to read a variable).

        for paradigm in PARADIGMS:
            print(f"\n--- Paradigm: {paradigm} ---")

            # -------------------------------------------------
            # 2. Alignment (Training the Model)
            # -------------------------------------------------
            if RUN_ALIGNMENT:
                target_script = "reproduce_all_paradigms.py"
                update_config_in_file(target_script, "SUBJECT", subject)
                update_config_in_file(target_script, "PARADIGM", paradigm)
                run_command(f"{PYTHON_EXEC} {target_script}", f"Alignment ({subject}/{paradigm})")

            # -------------------------------------------------
            # 3. Visualization (Plots & HTML)
            # -------------------------------------------------
            if RUN_VISUALIZATION:
                # Run the Universal Debug Visualizer
                target_script = "visualize_universal_debug.py"
                update_config_in_file(target_script, "SUBJECT", subject)
                update_config_in_file(target_script, "PARADIGM", paradigm)
                run_command(f"{PYTHON_EXEC} {target_script}", f"Visualization ({subject}/{paradigm})")

                # Run the ROI Analysis (Bar Charts)
                if paradigm != "pictures":  # Glasser ROI mostly for language, but works for pics too
                    target_script = "reproduce_roi_analysis.py"
                    update_config_in_file(target_script, "SUBJECT", subject)
                    update_config_in_file(target_script, "PARADIGM", paradigm)
                    run_command(f"{PYTHON_EXEC} {target_script}", f"ROI Analysis ({subject}/{paradigm})")

            # -------------------------------------------------
            # 4. Layer-wise Analysis (Depth Maps)
            # -------------------------------------------------
            if RUN_LAYERS:
                # Layer-wise Line Plot
                target_script = "run_layerwise_analysis.py"
                update_config_in_file(target_script, "SUBJECT", subject)
                update_config_in_file(target_script, "PARADIGM", paradigm)
                run_command(f"{PYTHON_EXEC} {target_script}", f"Layer Curve ({subject}/{paradigm})")

                # Best Layer Map (Brain Surface)
                target_script = "visualize_layer_depth_universal.py"
                update_config_in_file(target_script, "SUBJECT", subject)
                update_config_in_file(target_script, "PARADIGM", paradigm)
                run_command(f"{PYTHON_EXEC} {target_script}", f"Layer Depth Map ({subject}/{paradigm})")

            # -------------------------------------------------
            # 5. Baseline Comparison (Trained vs Random)
            # -------------------------------------------------
            if RUN_BASELINE and paradigm != "pictures":
                # (Assuming baseline script is currently optimized for text/GPT-2)
                target_script = "run_baseline_comparison.py"
                update_config_in_file(target_script, "SUBJECT", subject)
                update_config_in_file(target_script, "PARADIGM", paradigm)
                run_command(f"{PYTHON_EXEC} {target_script}", f"Baseline Check ({subject}/{paradigm})")

    elapsed = time.time() - start_time
    print(f"\n✅ All tasks completed in {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()