"""
Step 6: Anomaly Detection
=========================
Uses IsolationForest to detect operational anomalies and generates
visualizations comparing anomalies vs normal operations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

from config import MERGED_DATA_DIR, FIGURES_DIR, ANOMALY_FEATURES, ANOMALY_CONTAMINATION, ANOMALY_RANDOM_STATE

os.makedirs(FIGURES_DIR, exist_ok=True)

print("======================================================================")
print("STEP 6: ANOMALY DETECTION")
print("======================================================================")

master_path = os.path.join(MERGED_DATA_DIR, 'master_all_regions_2023.csv')
if not os.path.exists(master_path):
    raise FileNotFoundError(f"Master file not found: {master_path}")

master = pd.read_csv(master_path)
master['datetime'] = pd.to_datetime(master['period'])

# Recompute carbon intensity for normalization
from utils import compute_carbon_intensity
from config import (
    CARBON_FACTORS, CARBON_WEIGHT, WATER_WEIGHT, 
    CW_STRESS_EFFICIENT_THRESHOLD, CW_STRESS_MODERATE_THRESHOLD,
    FIGURE_DPI, PARADOX_CARBON_THRESHOLD, MODELS_DIR
)
master['carbon_intensity'] = compute_carbon_intensity(master, CARBON_FACTORS)

import joblib
try:
    mms_c = joblib.load(os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
    mms_w = joblib.load(os.path.join(MODELS_DIR, 'water_scaler.pkl'))
    master['carbon_norm'] = mms_c.transform(master[['carbon_intensity']])
    master['wbtmp_norm'] = mms_w.transform(master[['wbtmp']])
    master['CW_raw'] = CARBON_WEIGHT * master['carbon_norm'] + WATER_WEIGHT * master['wbtmp_norm']
    master['CW_Stress'] = np.where(master['CW_raw'] < CW_STRESS_EFFICIENT_THRESHOLD, 0, np.where(master['CW_raw'] < CW_STRESS_MODERATE_THRESHOLD, 1, 2))
except Exception as e:
    print(f"Warning: Could not load scalers for paradox detection: {e}")

# Prepare features
features = [f for f in ANOMALY_FEATURES if f in master.columns]
X = master[features].ffill().bfill().fillna(0)

print(f"Using features: {features}")

iso = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=ANOMALY_RANDOM_STATE)
iso.fit(X)
anomaly_score = iso.decision_function(X)
anomaly_flag = iso.predict(X)  # -1 for anomaly, 1 for normal
master['anomaly'] = (anomaly_flag == -1).astype(int)
master['anomaly_score'] = -anomaly_score  # higher = more anomalous

# 1. Scatter: temperature vs water stress
plt.figure(figsize=(8,6))
sns.scatterplot(data=master.sample(min(len(master), 5000), random_state=42), x='tmpf', y='CW_Stress', hue='anomaly', palette=['#2ecc71','#e74c3c'], alpha=0.6)
plt.title('Anomaly Scatter: Temperature vs CW_Stress')
plt.savefig(os.path.join(FIGURES_DIR, 'anomaly_scatter_tmpf_cw.png'), dpi=FIGURE_DPI)
plt.close()

# 2. Feature distributions (anomaly vs normal) for top features
top_feats = features[:4]
for feat in top_feats:
    plt.figure(figsize=(8,4))
    sns.kdeplot(master[master['anomaly']==0][feat].dropna(), label='Normal', fill=True)
    sns.kdeplot(master[master['anomaly']==1][feat].dropna(), label='Anomaly', fill=True)
    plt.title(f'Distribution: {feat} (Anomaly vs Normal)')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, f'anomaly_dist_{feat}.png'), dpi=FIGURE_DPI)
    plt.close()

# 3. Regional anomaly rates
region_rates = master.groupby('region')['anomaly'].mean().sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=region_rates.index, y=region_rates.values, palette='viridis')
plt.ylabel('Anomaly Rate')
plt.title('Regional Anomaly Rates')
plt.savefig(os.path.join(FIGURES_DIR, 'regional_anomaly_rates.png'), dpi=FIGURE_DPI)
plt.close()

# 4. Temporal anomaly patterns (monthly)
master['month'] = master['datetime'].dt.month
monthly = master.groupby('month')['anomaly'].mean()
plt.figure(figsize=(10,4))
sns.lineplot(x=monthly.index, y=monthly.values, marker='o')
plt.title('Monthly Anomaly Rate')
plt.xlabel('Month')
plt.ylabel('Anomaly Rate')
plt.savefig(os.path.join(FIGURES_DIR, 'monthly_anomaly_trend.png'), dpi=FIGURE_DPI)
plt.close()

# 5. SUMMARY STATS (Matches Technical Improvements Report)
n_anomalies = master['anomaly'].sum()
mean_stress_anom = master[master['anomaly']==1]['CW_raw'].mean()
mean_stress_norm = master[master['anomaly']==0]['CW_raw'].mean()
difference = mean_stress_anom - mean_stress_norm

print(f"\nAnomalies detected: {n_anomalies} hours ({n_anomalies/len(master)*100:.1f}%)")
print(f"Mean CW_raw (Anomaly): {mean_stress_anom:.3f}")
print(f"Mean CW_raw (Normal):  {mean_stress_norm:.3f}")
print(f"Difference: {difference:.3f}")

# 6. CARBON-WATER PARADOX HOURS
if 'carbon_norm' in master.columns and 'CW_Stress' in master.columns:
    paradox = master[
        (master['anomaly'] == 1) &          # flagged as anomaly
        (master['CW_Stress'] == 2) &         # stressed class
        (master['carbon_norm'] < PARADOX_CARBON_THRESHOLD) # low carbon (from config)
    ].copy()

    print(f"\nCarbon-Water Paradox Hours: {len(paradox)}")
    print("(Low carbon but high water stress — missed by carbon-only optimisation)")
    print("\nBy region:")
    print(paradox['region'].value_counts())

    fig, ax = plt.subplots(figsize=(10, 7))
    sample = master.sample(min(5000, len(master)), random_state=42)
    sc = ax.scatter(
        sample['carbon_norm'], sample['wbtmp_norm'],
        c=sample['CW_Stress'], cmap='RdYlGn_r',
        alpha=0.3, s=5, label='All hours')
    ax.scatter(
        paradox['carbon_norm'], paradox['wbtmp_norm'],
        c='blue', s=50, marker='x', linewidths=1.5,
        zorder=5, label=f'Paradox hours (n={len(paradox)})')
    plt.colorbar(sc, ax=ax, label='CW-Stress Level')
    ax.set_xlabel('Carbon Intensity (normalised)')
    ax.set_ylabel('Wet-Bulb Temperature (normalised)')
    ax.set_title('Carbon-Water Paradox Hours\nBlue X = Low carbon but high water stress')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'paradox_scatter.png'), dpi=FIGURE_DPI)
    plt.close()
    print("  [OK] Paradox scatter saved")

print("\n  [OK] Anomaly detection complete. Figures saved to figures/ folder.")

print("STEP 6 COMPLETE.")
