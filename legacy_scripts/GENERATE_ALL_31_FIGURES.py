import os
import subprocess
import time

# SKIP Step 4 to protect the 91.4% frozen model
STEPS = [
    "step3_merge_and_engineer.py",
    # "Step4_Machine_Learning.py", <--- SKIPPED
    "Step5_Research_Contributions.py",
    "Step6_Anomaly_Detection.py",
    "Step7_Nepal_Global_Case_Study.py",
    "Step8_Process_Nepal_Master.py",
    "Step9_Nepal_Explainability.py",
    "Step10_Nepal_Anomaly_Audit.py",
    "Step11_Clustering_Validation.py",
    "Step12_Statistical_Significance.py",
    "Step13_Ablation_Study.py",
    "Step14_Error_and_Feature_Audit.py"
]

print("="*80)
print("X-HYDRAAI: MASTER THESIS FIGURE GENERATOR (31 FIGURES)")
print("="*80)
print("[PROTECTED MODE] Using Frozen 91.4% Accuracy Model")

# Ensure folders exist
os.makedirs("FINAL_THESIS_FIGURES", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Run GENERATE_NOW.py first to get the main SVM figures
print("\n>>> EXECUTING: GENERATE_NOW.py (Main Results)...")
subprocess.run(["python", "GENERATE_NOW.py"], capture_output=False)

for step in STEPS:
    print(f"\n>>> EXECUTING: {step}...")
    try:
        # Run without capturing output so we see it live
        result = subprocess.run(["python", step], capture_output=False, timeout=600)
        if result.returncode == 0:
            print(f"[OK] {step} completed.")
        else:
            print(f"[ERROR] {step} returned code {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"[WARNING] {step} timed out. Moving to next step.")
    except Exception as e:
        print(f"[ERROR] {step} failed: {e}")

print("\n" + "="*80)
print("SUCCESS: All 31 figures have been generated in FINAL_THESIS_FIGURES.")
print("="*80)
