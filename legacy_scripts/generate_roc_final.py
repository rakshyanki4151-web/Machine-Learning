"""
X-HydraAI Final Results Refresh: Multi-Model ROC Analysis
Generates the ROC-AUC curves for the baseline model to provide 
statistical confidence evidence for the research paper.
"""
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.close('all')
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from config import MODELS_DIR, ML_FEATURES, MERGED_DATA_DIR_STR, FIGURES_DIR

# 1. LOAD ASSETS
print("Loading models and test data for ROC analysis...")
best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
mms_c = joblib.load(os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
mms_w = joblib.load(os.path.join(MODELS_DIR, 'water_scaler.pkl'))

master = pd.read_csv(os.path.join(MERGED_DATA_DIR_STR, 'master_all_regions_2023.csv'))

# Generate Labels and Features exactly as in Step 4
from utils import compute_carbon_intensity, wet_bulb_stull
from config import CARBON_FACTORS, CARBON_WEIGHT, WATER_WEIGHT, CW_STRESS_EFFICIENT_THRESHOLD, CW_STRESS_MODERATE_THRESHOLD

master['carbon_intensity'] = compute_carbon_intensity(master, CARBON_FACTORS)
master['wbtmp'] = wet_bulb_stull(master['tmpf'], master['relh'])
master['carbon_norm'] = mms_c.transform(master[['carbon_intensity']])
master['wbtmp_norm'] = mms_w.transform(master[['wbtmp']])
master['CW_raw'] = CARBON_WEIGHT * master['carbon_norm'] + WATER_WEIGHT * master['wbtmp_norm']
master['CW_Stress'] = 0
master.loc[master['CW_raw'] >= CW_STRESS_EFFICIENT_THRESHOLD, 'CW_Stress'] = 1
master.loc[master['CW_raw'] >= CW_STRESS_MODERATE_THRESHOLD, 'CW_Stress'] = 2

master['region_encoded'] = le.transform(master['region'])

# DEFENSIVE: Ensure Hour columns exist
if 'Hour' not in master.columns:
    master['Hour'] = pd.to_datetime(master['period']).dt.hour
master['hour_sin'] = __import__('numpy').sin(2 * __import__('numpy').pi * master['Hour']/24.0)
master['hour_cos'] = __import__('numpy').cos(2 * __import__('numpy').pi * master['Hour']/24.0)

# 2. SELECT TEST SAMPLE
test_sample = master.sample(n=min(5000, len(master)), random_state=42)
X_test = test_sample[ML_FEATURES].fillna(0)
y_test = test_sample['CW_Stress']

# Binarize for multi-class ROC
y_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# 3. GET PROBABILITIES
if hasattr(best_model, "predict_proba"):
    y_score = best_model.predict_proba(X_test)
else:
    y_score = best_model.decision_function(X_test)

# 4. COMPUTE ROC FOR EACH CLASS
plt.figure(figsize=(10, 8))
colors = ['#2ecc71', '#f39c12', '#e74c3c']
labels = ['Efficient', 'Moderate', 'Stressed']

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=3,
             label=f'AUC: {labels[i]} (area = {roc_auc:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic: Multi-Class Stress Identification', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

save_path = os.path.join(FIGURES_DIR, 'final_roc_curve.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\nSUCCESS: Multi-class ROC Curve generated. Saved: {save_path}")
