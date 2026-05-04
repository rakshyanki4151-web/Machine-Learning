"""
Step 15: Premium Research-Grade ROC Curve Generator
===================================================
X-HydraAI 2023 

This script generates a publication-grade, high-DPI ROC curve for the 
winning SVM model, ensuring academic clarity for the final thesis.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Path Configuration
MODELS_DIR = 'models'
FIGURES_DIR = 'figures'
DATA_DIR = 'data/us'
MERGED_DATA_PATH = os.path.join(DATA_DIR, 'us_master_2023.csv')

# Style Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
FIGURE_DPI = 300

def generate_premium_roc():
    print("="*80)
    print("STEP 15: GENERATING PREMIUM RESEARCH ROC CURVE")
    print("="*80)

    try:
        # 1. LOAD MODEL & DATA
        print("[1/4] Loading trained artifacts...")
        model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        
        # Load test data (Oct-Dec 2023 per thesis methodology)
        master = pd.read_csv(MERGED_DATA_PATH)
        master['datetime'] = pd.to_datetime(master['period'])
        test = master[master['datetime'] >= '2023-10-01'].copy()
        
        # 2. PREPROCESS TEST SET
        print("[2/4] Preprocessing test set...")
        from config import ML_FEATURES, CARBON_FACTORS
        from utils import compute_carbon_intensity, wet_bulb_stull
        
        # Re-derive labels for the test set to ensure ground truth parity
        mms_c = joblib.load(os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
        mms_w = joblib.load(os.path.join(MODELS_DIR, 'water_scaler.pkl'))
        
        test['carbon_intensity'] = compute_carbon_intensity(test, CARBON_FACTORS)
        test['wbtmp'] = wet_bulb_stull(test['tmpf'], test['relh'])
        test['carbon_norm'] = mms_c.transform(test[['carbon_intensity']])
        test['wbtmp_norm'] = mms_w.transform(test[['wbtmp']])
        test['CW_raw'] = 0.60 * test['carbon_norm'] + 0.40 * test['wbtmp_norm']
        
        # Use exact thesis thresholds
        test['CW_Stress'] = np.where(test['CW_raw'] < 0.4388, 0,
                            np.where(test['CW_raw'] < 0.5241, 1, 2))
        
        test['region_encoded'] = le.transform(test['region'])
        test['Hour'] = test['datetime'].dt.hour
        test['hour_sin'] = np.sin(2 * np.pi * test['Hour']/24.0)
        test['hour_cos'] = np.cos(2 * np.pi * test['Hour']/24.0)
        
        X_test = test[ML_FEATURES].fillna(0)
        y_test = test['CW_Stress']

        # 3. CALCULATE ROC METRICS
        print("[3/4] Calculating multi-class ROC metrics...")
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        y_score = model.predict_proba(X_test)

        # 4. PLOTTING
        print("[4/4] Rendering publication-grade figure...")
        plt.figure(figsize=(10, 8))
        
        # Custom premium palette
        colors = ['#27ae60', '#f39c12', '#e74c3c'] # Efficient, Moderate, Stressed
        class_names = ['Tier 0: Efficient', 'Tier 1: Moderate', 'Tier 2: Stressed']
        
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=3,
                     label=f'{class_names[i]} (AUC = {roc_auc:0.3f})')
        
        # Add Macro Average
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            mean_tpr += np.interp(fpr_grid, fpr, tpr)
        mean_tpr /= n_classes
        macro_auc = auc(fpr_grid, mean_tpr)
        plt.plot(fpr_grid, mean_tpr, color='navy', linestyle=':', lw=4,
                 label=f'Macro-average ROC (AUC = {macro_auc:0.3f})')

        # Baseline
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
        
        # Aesthetics
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, labelpad=10)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, labelpad=10)
        plt.title('Receiver Operating Characteristic (ROC) Audit\nX-HydraAI SVM-RBF Diagnostic Classifier', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", frameon=True, shadow=True, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save results
        output_path = os.path.join(FIGURES_DIR, 'roc_curve_premium.png')
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"\n[SUCCESS] Premium ROC curve saved to: {output_path}")
        print(f"Final Macro-AUC: {macro_auc:.4f}")
        print("="*80)

    except Exception as e:
        print(f"\n[ERROR] ROC Generation failed: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    generate_premium_roc()
