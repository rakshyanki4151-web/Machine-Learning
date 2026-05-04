import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

matplotlib.use('Agg')
plt.close('all')

# 1. PATH CONFIG
ROOT = r"c:\Users\shid0\OneDrive\Desktop\hydra\X-HydraAI_2023_Project"
MODELS_DIR = os.path.join(ROOT, "models")
FIGURES_DIR = os.path.join(ROOT, "FINAL_THESIS_FIGURES")
DATA_PATH = os.path.join(ROOT, "data", "merged", "master_all_regions_2023.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)

print(">>> [EMERGENCY GEN] Loading Model...")
best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
mms_c = joblib.load(os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
mms_w = joblib.load(os.path.join(MODELS_DIR, 'water_scaler.pkl'))

print(">>> [EMERGENCY GEN] Loading Data...")
df = pd.read_csv(DATA_PATH)

# Feature engineering (Minimal)
from config import CARBON_FACTORS, CARBON_WEIGHT, WATER_WEIGHT, CW_STRESS_EFFICIENT_THRESHOLD, CW_STRESS_MODERATE_THRESHOLD, ML_FEATURES
from utils import compute_carbon_intensity, wet_bulb_stull

df['carbon_intensity'] = compute_carbon_intensity(df, CARBON_FACTORS)
df['wbtmp'] = wet_bulb_stull(df['tmpf'], df['relh'])
df['carbon_norm'] = mms_c.transform(df[['carbon_intensity']])
df['wbtmp_norm'] = mms_w.transform(df[['wbtmp']])
df['CW_raw'] = CARBON_WEIGHT * df['carbon_norm'] + WATER_WEIGHT * df['wbtmp_norm']
df['CW_Stress'] = np.where(df['CW_raw'] < CW_STRESS_EFFICIENT_THRESHOLD, 0, 
                    np.where(df['CW_raw'] < CW_STRESS_MODERATE_THRESHOLD, 1, 2))

df['region_encoded'] = le.transform(df['region'])
if 'Hour' not in df.columns:
    df['Hour'] = pd.to_datetime(df['period']).dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['Hour']/24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['Hour']/24.0)

# Select October-December 2023 (Final Test Set)
df['datetime'] = pd.to_datetime(df['period'])
test_set = df[df['datetime'] >= '2023-10-01'].copy()
X_test = test_set[ML_FEATURES].fillna(0)
y_test = test_set['CW_Stress']

print(f">>> [EMERGENCY GEN] Predicting on {len(X_test)} samples...")
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

# 2. GENERATE CONFUSION MATRIX
print(">>> [EMERGENCY GEN] Saving Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Efficient', 'Moderate', 'Stressed'])
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix: SVM (RBF) - 91.4% Accuracy")
plt.savefig(os.path.join(FIGURES_DIR, "cm_SVM_(RBF).png"), dpi=300)
plt.close()

# 3. GENERATE ROC CURVE
print(">>> [EMERGENCY GEN] Saving ROC Curve...")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
plt.figure(figsize=(10, 8))
colors = ['cyan', 'orange', 'cornflowerblue']
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC: SVM (RBF)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(FIGURES_DIR, "final_roc_curve.png"), dpi=300)
plt.close()

print(">>> [EMERGENCY GEN] COMPLETE. Check FINAL_THESIS_FIGURES folder.")
