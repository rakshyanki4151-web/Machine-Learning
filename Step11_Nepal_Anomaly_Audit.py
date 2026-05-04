"""
X-HydraAI Nepal Pipeline: Anomaly & Integrity Audit (v2 - Self Healing)
1. Learns US baseline patterns.
2. Identifies Nepal outliers.
Target: figures/nepal_anomaly_audit.png
"""
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from config import MODELS_DIR, ANOMALY_FEATURES, FIGURES_DIR, MERGED_DATA_DIR_STR, ANOMALY_CONTAMINATION

# 1. ESTABLISH BASELINE
US_MASTER_PATH = os.path.join(MERGED_DATA_DIR_STR, 'master_all_regions_2023.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'anomaly_model.pkl')

if os.path.exists(MODEL_PATH):
    print("Loading existing US Anomaly Model...")
    iso_forest = joblib.load(MODEL_PATH)
else:
    print("Building US Anomaly Baseline (Self-Healing Mode)...")
    us_data = pd.read_csv(US_MASTER_PATH)
    
    # --- ON-THE-FLY FEATURE GENERATION ---
    from utils import compute_carbon_intensity
    from config import CARBON_FACTORS, CARBON_WEIGHT, WATER_WEIGHT, CW_STRESS_EFFICIENT_THRESHOLD, CW_STRESS_MODERATE_THRESHOLD
    us_data['carbon_intensity'] = compute_carbon_intensity(us_data, CARBON_FACTORS)
    try:
        mms_c = joblib.load(os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
        mms_w = joblib.load(os.path.join(MODELS_DIR, 'water_scaler.pkl'))
        us_data['carbon_norm'] = mms_c.transform(us_data[['carbon_intensity']])
        us_data['wbtmp_norm'] = mms_w.transform(us_data[['wbtmp']])
        us_data['CW_raw'] = CARBON_WEIGHT * us_data['carbon_norm'] + WATER_WEIGHT * us_data['wbtmp_norm']
        us_data['CW_Stress'] = np.where(us_data['CW_raw'] < CW_STRESS_EFFICIENT_THRESHOLD, 0, 
                                np.where(us_data['CW_raw'] < CW_STRESS_MODERATE_THRESHOLD, 1, 2))
    except Exception as e:
        print(f"Warning: Could not load scalers: {e}")
    # -------------------------------------

    iso_forest = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=42)
    iso_forest.fit(us_data[ANOMALY_FEATURES])
    joblib.dump(iso_forest, MODEL_PATH)

# 2. RUN NEPAL AUDIT
print("Auditing Nepal simulation against US standards...")
nepal_master = pd.read_csv(os.path.join(MERGED_DATA_DIR_STR, 'master_nepal_treated_2023.csv'))
nepal_master['anomaly_score'] = iso_forest.predict(nepal_master[ANOMALY_FEATURES])

# 3. ANALYZE RESULTS
# Isolation Forest: -1 = Anomaly, 1 = Normal
audit_results = nepal_master.groupby('region')['anomaly_score'].value_counts(normalize=True).unstack().fillna(0) * 100
audit_results.rename(columns={1: 'Normal (%)', -1: 'Anomalous (%)'}, inplace=True)

print("\nNEPAL WEATHER INTEGRITY REPORT:")
print(audit_results)

# 4. GENERATE CHART
plt.figure(figsize=(10, 6))
colors = ['#2ecc71' if x < 15 else '#e74c3c' for x in audit_results['Anomalous (%)']]
audit_results['Anomalous (%)'].plot(kind='bar', color=colors)

plt.title("X-HydraAI Model Compatibility Audit", fontsize=14)
plt.ylabel("Potential Outlier Hours (%)")
plt.xlabel("Nepal Research Zone")
plt.grid(axis='y', alpha=0.3)
plt.axhline(y=10, color='gray', linestyle='--', label='Warning Threshold')

save_path = os.path.join(FIGURES_DIR, "nepal_anomaly_audit.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\nSUCCESS: Anomaly Audit complete. Saved to: {save_path}")
