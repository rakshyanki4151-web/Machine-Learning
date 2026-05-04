import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from config import MERGED_DATA_DIR, FIGURES_DIR, ML_FEATURES, RANDOM_STATE, CW_STRESS_EFFICIENT_THRESHOLD, CW_STRESS_MODERATE_THRESHOLD

# 1. Load Data
master_path = os.path.join(MERGED_DATA_DIR, 'master_all_regions_2023.csv')
df = pd.read_csv(master_path)

# Label Generation (using frozen thresholds)
from utils import compute_carbon_intensity, wet_bulb_stull
from config import CARBON_FACTORS, CARBON_WEIGHT, WATER_WEIGHT

df['carbon_intensity'] = compute_carbon_intensity(df, CARBON_FACTORS)
df['wbtmp'] = wet_bulb_stull(df['tmpf'], df['relh'])

# Normalise for labels (simple fit for counts)
c_min, c_max = df['carbon_intensity'].min(), df['carbon_intensity'].max()
w_min, w_max = df['wbtmp'].min(), df['wbtmp'].max()
df['c_norm'] = (df['carbon_intensity'] - c_min) / (c_max - c_min)
df['w_norm'] = (df['wbtmp'] - w_min) / (w_max - w_min)
df['CW_raw'] = CARBON_WEIGHT * df['c_norm'] + WATER_WEIGHT * df['w_norm']
df['CW_Stress'] = np.where(df['CW_raw'] < CW_STRESS_EFFICIENT_THRESHOLD, 0,
                   np.where(df['CW_raw'] < CW_STRESS_MODERATE_THRESHOLD, 1, 2))

# 2. Get Counts Before
before_counts = df['CW_Stress'].value_counts().sort_index()

# 3. Apply SMOTE
# Dummy X for SMOTE
X_dummy = np.random.rand(len(df), 2)
sm = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = sm.fit_resample(X_dummy, df['CW_Stress'])
after_counts = pd.Series(y_res).value_counts().sort_index()

# 4. Plot
labels = ['Efficient', 'Moderate', 'Stressed']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, before_counts, width, label='Before SMOTE (Imbalanced)', color='lightgrey', edgecolor='black')
ax.bar(x + width/2, after_counts, width, label='After SMOTE (Balanced)', color='skyblue', edgecolor='black')

ax.set_ylabel('Number of Samples (Hours)')
ax.set_title('Class Balancing: Synthetic Minority Over-sampling Technique (SMOTE)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add values on top
for i, v in enumerate(before_counts):
    ax.text(i - width/2, v + 100, str(v), ha='center', fontweight='bold')
for i, v in enumerate(after_counts):
    ax.text(i + width/2, v + 100, str(v), ha='center', fontweight='bold', color='blue')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'smote_balancing_audit.png'), dpi=300)
plt.close()

print(f">>> [SUCCESS] SMOTE Audit Chart saved to {FIGURES_DIR}")
