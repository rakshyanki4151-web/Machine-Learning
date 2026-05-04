import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier

# Paths
ROOT = r"c:\Users\shid0\OneDrive\Desktop\hydra\X-HydraAI_2023_Project"
MODELS_DIR = os.path.join(ROOT, "models")
MERGED_DATA_DIR = os.path.join(ROOT, "data", "merged")
FIGURES_DIR = os.path.join(ROOT, "FINAL_THESIS_FIGURES")

from config import ML_FEATURES, RANDOM_STATE

print(">>> [FAST SHAP] Loading Model and Data...")
master = pd.read_csv(os.path.join(MERGED_DATA_DIR, 'master_all_regions_2023.csv'))
pipeline = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))

# Prepare Features
print(">>> [FAST SHAP] Preparing features...")
master['region_encoded'] = le.transform(master['region'])
if 'Hour' not in master.columns:
    master['Hour'] = pd.to_datetime(master['period']).dt.hour
master['hour_sin'] = np.sin(2 * np.pi * master['Hour']/24.0)
master['hour_cos'] = np.cos(2 * np.pi * master['Hour']/24.0)

# NOW select features
X = master[ML_FEATURES].fillna(0)

# Extract components
clf = pipeline.named_steps['model']
scaler = pipeline.named_steps['scaler']

# Sample for speed (Small sample but representative)
X_sample_raw = X.sample(50, random_state=42)
X_sample_scaled = scaler.transform(X_sample_raw)

print(">>> [FAST SHAP] Calculating SHAP values...")
# Use KernelExplainer for broad compatibility (sampled background)
explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_sample_scaled, 10))
shap_values = explainer.shap_values(X_sample_scaled)

# Class 2 is Stressed
sv_s = shap_values[2] if isinstance(shap_values, list) else shap_values[:, :, 2]

# 1. Global Importance
print(">>> [FAST SHAP] Saving shap_importance.png...")
plt.figure(figsize=(10, 6))
shap.summary_plot(sv_s, X_sample_raw, feature_names=ML_FEATURES, plot_type='dot', show=False)
plt.title('SHAP: Feature Importance for Stressed Class (91.4% Model)')
plt.savefig(os.path.join(FIGURES_DIR, 'shap_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Per-Region
print(">>> [FAST SHAP] Saving shap_per_region.png...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
for ax, region in zip(axes.flatten(), ['CAL', 'ERCO', 'NW', 'PJM']):
    code = le.transform([region])[0]
    reg_mask = X['region_encoded'] == code
    X_reg = X[reg_mask].reset_index(drop=True)
    if len(X_reg) == 0: continue
    X_reg_sample = X_reg.sample(min(30, len(X_reg)), random_state=42)
    X_reg_scaled = scaler.transform(X_reg_sample)
    
    sv = explainer.shap_values(X_reg_scaled)
    sv_reg_s = sv[2] if isinstance(sv, list) else sv[:, :, 2]
    
    mean_shap = pd.Series(np.abs(sv_reg_s).mean(axis=0), index=ML_FEATURES).sort_values(ascending=True)
    mean_shap.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(f'{region} — What Drives Stress?', fontsize=12)

plt.suptitle('Per-Region Feature Importance (SHAP)', fontsize=14, y=1.01)
plt.savefig(os.path.join(FIGURES_DIR, 'shap_per_region.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Waterfall
print(">>> [FAST SHAP] Saving shap_waterfall_stressed.png...")
expected_val = explainer.expected_value[2] if hasattr(explainer.expected_value, '__getitem__') else explainer.expected_value
exp = shap.Explanation(
    values=sv_s[0],
    base_values=expected_val,
    data=X_sample_raw.iloc[0],
    feature_names=ML_FEATURES)
plt.figure()
shap.waterfall_plot(exp, show=False)
plt.title('SHAP Waterfall — Hourly Stress Driver')
plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall_stressed.png'), dpi=300, bbox_inches='tight')
plt.close()

print(">>> [FAST SHAP] COMPLETE. All 31 figures reached.")
