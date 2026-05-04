"""
X-HydraAI Nepal Pipeline: Full Explainability Audit (Academic Standard)
Using 100 samples per zone (400 total) for statistical significance.
Note: Calculation time increased for SVM fidelity.
Target: figures/nepal_shap_drivers.png
"""
import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from config import MODELS_DIR, ML_FEATURES, FIGURES_DIR, MERGED_DATA_DIR_STR

# 1. LOAD ASSETS
print("Loading assets for high-fidelity audit...")
pipeline = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
nepal_master = pd.read_csv(os.path.join(MERGED_DATA_DIR_STR, 'master_nepal_treated_2023.csv'))

# 2. ACADEMIC SAMPLE (100 per zone)
print("Generating academic standard sample (400 rows)...")
analysis_df = nepal_master.groupby('region').sample(n=100, random_state=42)
X = analysis_df[ML_FEATURES]

# 3. RUN PERMUTATION EXPLAINER
# Using a background summary to speed up SVM kernel math slightly
print("Calculating high-fidelity SHAP values. This will take ~15-20 minutes.")
print("Calculations are running in the background...")
explainer = shap.Explainer(pipeline.predict, X)
shap_values = explainer(X)

# 4. GENERATE SUMMARY PLOT
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_values, max_display=12, show=False)

plt.title("X-HydraAI: Global Sustainability Drivers (Nepal Case Study - 400 Samples)", fontsize=14, pad=20)
save_path = os.path.join(FIGURES_DIR, "nepal_shap_drivers.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\nSUCCESS: High-fidelity SHAP Audit completed. Chart saved: {save_path}")
