import matplotlib
matplotlib.use('Agg')
print(">>> SCRIPT STARTING: Step 5")
import os
print(">>> OS IMPORTED")
import joblib
print(">>> JOBLIB IMPORTED")
import pandas as pd
print(">>> PANDAS IMPORTED")
import numpy as np
print(">>> NUMPY IMPORTED")
import matplotlib.pyplot as plt
print(">>> MATPLOTLIB.PYPLOT IMPORTED")
import seaborn as sns
print(">>> SEABORN IMPORTED")
from scipy.stats import spearmanr
print(">>> SCIPY IMPORTED")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
print(">>> SKLEARN IMPORTED")

try:
    import shap
    HAS_SHAP = False # Temporarily disabled for speed
except ImportError:
    HAS_SHAP = False
    print("  [WARN] SHAP not installed; skipping SHAP analysis")

from config import MODELS_DIR, MERGED_DATA_DIR, FIGURES_DIR, ML_FEATURES, CARBON_FACTORS, FIGURE_DPI, CARBON_WEIGHT, WATER_WEIGHT, RANDOM_STATE, CW_STRESS_EFFICIENT_THRESHOLD, CW_STRESS_MODERATE_THRESHOLD

print(">>> IMPORTS COMPLETE: Step 5")


os.makedirs(FIGURES_DIR, exist_ok=True)

print("======================================================================")
print("STEP 5: RESEARCH CONTRIBUTIONS & EXPLAINABILITY")
print("======================================================================")

# Load merged data
master_path = os.path.join(MERGED_DATA_DIR, 'master_all_regions_2023.csv')
if not os.path.exists(master_path):
    raise FileNotFoundError(f"Master file not found: {master_path}")

master = pd.read_csv(master_path)
master['datetime'] = pd.to_datetime(master['period'])

# RE-ENGINEER FEATURES FOR STATS
try:
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    master['region_encoded'] = le.transform(master['region'])
except Exception as e:
    print(f"  [WARN] Could not load encoder: {e}. Refitting as fallback.")
    le = LabelEncoder()
    master['region_encoded'] = le.fit_transform(master['region'])

# DEFENSIVE: Ensure 'Hour' column exists
if 'Hour' not in master.columns:
    if 'period' in master.columns:
        master['Hour'] = pd.to_datetime(master['period']).dt.hour
    elif 'datetime' in master.columns:
        master['Hour'] = pd.to_datetime(master['datetime']).dt.hour
    else:
        master['Hour'] = master.index.hour

master['hour_sin'] = np.sin(2 * np.pi * master['Hour']/24.0)
master['hour_cos'] = np.cos(2 * np.pi * master['Hour']/24.0)

# Recompute carbon intensity for consistency using utils
from utils import compute_carbon_intensity
master['carbon_intensity'] = compute_carbon_intensity(master, CARBON_FACTORS)

# Load scalers and regenerate normalized values
try:
    mms_c = joblib.load(os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
    mms_w = joblib.load(os.path.join(MODELS_DIR, 'water_scaler.pkl'))
except Exception as e:
    print(f"  [WARN] Could not load scalers: {e}")
    from sklearn.preprocessing import MinMaxScaler
    mms_c = MinMaxScaler()
    mms_w = MinMaxScaler()
    mms_c.fit(master[['carbon_intensity']])
    mms_w.fit(master[['wbtmp']])

master['carbon_norm'] = mms_c.transform(master[['carbon_intensity']])
master['wbtmp_norm'] = mms_w.transform(master[['wbtmp']])
master['CW_raw'] = CARBON_WEIGHT * master['carbon_norm'] + WATER_WEIGHT * master['wbtmp_norm']
master['CW_Stress'] = np.where(master['CW_raw'] < CW_STRESS_EFFICIENT_THRESHOLD, 0, np.where(master['CW_raw'] < CW_STRESS_MODERATE_THRESHOLD, 1, 2))

print("\n[0/6] Generating Base Statistical Suite (Heatmap & Pairplot)...")
# A. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = master[ML_FEATURES].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Research Audit: Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'feature_correlation_matrix.png'), dpi=FIGURE_DPI)
plt.close()

# B. Pairplot (Sampled for speed)
pair_df = master[ML_FEATURES + ['CW_Stress']].sample(min(800, len(master)), random_state=42)
sns.pairplot(pair_df, hue='CW_Stress', palette='viridis', diag_kind='kde')
plt.savefig(os.path.join(FIGURES_DIR, 'feature_pairplot.png'), dpi=150)
plt.close()
print("  [OK] Heatmap and Pairplot generated.")

print("\n[1/4] Generating Tension Index (Carbon-Water Goal Conflict)...")
tension = {}
for r in master['region'].unique():
    sub = master[master['region'] == r]
    if len(sub) > 1:
        corr, pval = spearmanr(sub['carbon_norm'].fillna(0), sub['wbtmp_norm'].fillna(0))
        tension[r] = corr

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in tension.values()] # Professional Red and Green
ax.bar(tension.keys(), tension.values(), color=colors, edgecolor='black')
ax.axhline(0, color='black', lw=1, ls='--')
ax.set_title('Carbon-Water Tension Index (Paradox Detection)')
ax.set_ylabel('Spearman Correlation')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'tension_index.png'), dpi=FIGURE_DPI)
plt.close()
print("  [OK] Tension index generated")

print("\n[2/4] Generating Seasonal Heatmaps...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for idx, r in enumerate(master['region'].unique()):
    ax = axes.flatten()[idx]
    sub = master[master['region'] == r].copy()
    sub['month'] = sub['datetime'].dt.month
    sub['hour'] = sub['datetime'].dt.hour
    pivot = sub.pivot_table(values='CW_Stress', index='month', columns='hour', aggfunc='mean')
    sns.heatmap(pivot, cmap='RdYlGn_r', vmin=0, vmax=2, ax=ax, cbar_kws={'label': 'Stress Level'})
    ax.set_title(f'{r} - Hourly Sustainability Stress by Month')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Month')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'seasonal_heatmap.png'), dpi=FIGURE_DPI)
plt.close()
print("  [OK] Seasonal heatmaps generated")

print("\n[3/4] Generating Correlation Matrix...")
corr_cols = ['tmpf', 'dwpf', 'relh', 'carbon_intensity', 'CW_Stress']
corr_cols = [c for c in corr_cols if c in master.columns]
corr_matrix = master[corr_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
plt.title('Feature-Target Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'correlation_matrix.png'), dpi=FIGURE_DPI)
plt.close()
print("  [OK] Correlation matrix generated")

print("\n[3.5/5] Generating PCA 2D Visualization...")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fixed index alignment logic
sample = master.sample(min(3000, len(master)), random_state=42)
X_pca_input = sample[['tmpf','dwpf','relh','COL','NG','NUC','SUN','WAT','WND']].fillna(0)
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X_pca_input)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
y_sample = sample['CW_Stress'].values

fig, ax = plt.subplots(figsize=(10, 7))
for cls, color, label in [(0,'#2ecc71','Efficient'),
                           (1,'#f39c12','Moderate'),
                           (2,'#e74c3c','Stressed')]:
    mask = y_sample == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=color, label=label, alpha=0.4, s=8)


ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('PCA: Class Separability of CW-Stress Labels\n'
             f'Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%')
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'pca_class_separation.png'), dpi=300)
plt.close()
print(f"  [OK] PCA generated (PC1: {pca.explained_variance_ratio_[0]*100:.1f}%)")

print("\n[4/5] Generating SHAP Explanations (From True Model)...")
if HAS_SHAP:
    try:
        # 1. Provide exact features to match Step 4 training
        # Must exactly match ML_FEATURES from config
        X = master[ML_FEATURES].fillna(0)
        
        # 2. LOAD the actual winner, do NOT retrain
        best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError("Run Step 4 first. best_model.pkl missing.")
            
        final_pipeline = joblib.load(best_model_path)
        
        # Extract the underlying classifier from the Pipeline
        clf = final_pipeline.named_steps['model']
        scaler = final_pipeline.named_steps['scaler']
        
        X_sample_raw = X.sample(min(100, len(X)), random_state=RANDOM_STATE)
        X_sample_scaled = scaler.transform(X_sample_raw)
        
        # 3. Create explainer based on model type
        plt.figure(figsize=(10, 6))
        
        if isinstance(clf, RandomForestClassifier):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample_scaled)
            # Handle both old and new SHAP multi-class formats
            if isinstance(shap_values, list): sv_s = shap_values[2]
            elif len(shap_values.shape) == 3: sv_s = shap_values[:, :, 2]
            else: sv_s = shap_values
        else:
            # KernelExplainer for SVM
            explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_sample_scaled, 20))
            shap_values = explainer.shap_values(X_sample_scaled)
            sv_s = shap_values[2] if isinstance(shap_values, list) else shap_values[:, :, 2]
            
        shap.summary_plot(sv_s, X_sample_raw, feature_names=ML_FEATURES, plot_type='dot', show=False)

        plt.title('SHAP: Feature Importance for Stressed Class (True Model)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_importance.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("  [OK] SHAP global analysis complete")

        # Addition 2 — Per-region SHAP comparison
        print("  > Generating Per-Region SHAP comparisons...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        for ax, region in zip(axes.flatten(), ['CAL', 'ERCO', 'NW', 'PJM']):
            code = le.transform([region])[0]
            reg_mask = X['region_encoded'] == code
            X_reg = X[reg_mask].reset_index(drop=True)
            if len(X_reg) < 10: continue
            
            X_reg_sample = X_reg.sample(min(50, len(X_reg)), random_state=42)
            X_reg_scaled = scaler.transform(X_reg_sample)
            
            sv = explainer.shap_values(X_reg_scaled)
            if isinstance(sv, list):
                sv_s = sv[2]  # RF: list of arrays per class
            elif len(np.array(sv).shape) == 3:
                sv_s = sv[:, :, 2]  # 3D array: (samples, features, classes)
            else:
                sv_s = sv  # fallback
                
            mean_shap = pd.Series(np.abs(sv_s).mean(axis=0), index=ML_FEATURES).sort_values(ascending=True)
            mean_shap.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'{region} — What Drives Stress?', fontsize=12)
            ax.set_xlabel('Mean |SHAP Value|')

        plt.suptitle('Per-Region Feature Importance — Why Each Grid Stresses Differently', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_per_region.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("  [OK] Regional SHAP generated")

        # Addition 3 — SHAP local waterfall
        print("  > Generating SHAP local waterfall for a stressed hour...")
        master_sample_indices = X_sample_raw.index
        sample_stress = master.loc[master_sample_indices, 'CW_Stress']
        stressed_sample_idx = sample_stress[sample_stress == 2].index
        if len(stressed_sample_idx) > 0:
            s_idx = stressed_sample_idx[0]
            pos_in_sample = list(X_sample_raw.index).index(s_idx)
            
            expected_val = explainer.expected_value[2] if hasattr(explainer.expected_value, '__getitem__') else explainer.expected_value
            
            exp = shap.Explanation(
                values=sv_s[pos_in_sample],
                base_values=expected_val,
                data=X_sample_raw.iloc[pos_in_sample],
                feature_names=ML_FEATURES)
            shap.waterfall_plot(exp, show=False)
            plt.title('SHAP Waterfall — Why This Hour Is Classified as Stressed')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall_stressed.png'), dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("  [OK] SHAP waterfall generated")

    except Exception as e:
        print(f"  [WARN] SHAP analysis failed (Kernel Explainers are slow): {e}")
        import traceback; traceback.print_exc()
else:
    print("  [SKIP] SHAP not available")

print("\n[5/5] Generating Scheduling Recommendations...")
sched_rows = []
for region in ['CAL', 'ERCO', 'NW', 'PJM']:
    sub = master[master['region'] == region].copy()
    sub['Month'] = sub['datetime'].dt.month
    best = sub.nsmallest(500, 'CW_raw')
    worst = sub.nlargest(500, 'CW_raw')
    sched_rows.append({
        'Region': region,
        'Best 500 avg CW': round(best['CW_raw'].mean(), 3),
        'Worst 500 avg CW': round(worst['CW_raw'].mean(), 3),
        'Potential saving': round(worst['CW_raw'].mean() - best['CW_raw'].mean(), 3),
        'Optimal hour': int(best['Hour'].mode().iloc[0] if len(best['Hour'].mode())>0 else 0),
        'Optimal month': int(best['Month'].mode().iloc[0] if len(best['Month'].mode())>0 else 0)
    })

sched_df = pd.DataFrame(sched_rows)
print("\nScheduling Recommendation Table:")
print(sched_df.to_string(index=False))
sched_df.to_csv(os.path.join(FIGURES_DIR, 'scheduling_recommendations.csv'), index=False)
print("  [OK] Scheduling table saved")

# Generation of the Scheduling Windows Visualisation
fig, ax = plt.subplots(figsize=(10, 6))
regions = sched_df['Region']
best_cw = sched_df['Best 500 avg CW']
worst_cw = sched_df['Worst 500 avg CW']
x = np.arange(len(regions))
width = 0.35

ax.bar(x - width/2, best_cw, width, label='Best 500 Hours (Clean/Cool)', color='#2ecc71', edgecolor='black')
ax.bar(x + width/2, worst_cw, width, label='Worst 500 Hours (Dirty/Hot)', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Average CW-Stress Score')
ax.set_title('Workload Scheduling Potential: Best vs. Worst Operational Windows')
ax.set_xticks(x)
ax.set_xticklabels(regions)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'scheduling_windows_chart.png'), dpi=FIGURE_DPI)
plt.close()
print("  [OK] Scheduling chart generated")

print("\nSTEP 5 COMPLETE.")
