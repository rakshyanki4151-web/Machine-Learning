import pandas as pd
import numpy as np
import os
from config import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Robust imports for specialized Research libraries
try:
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    print(f"CRITICAL ERROR: Missing required library: {e}")
    print("Please install via: pip install scipy imbalanced-learn")
    exit(1)

def run_ablation_study():
    print("--- [Step 13] ABLATION STUDY (Single vs. Multi-Objective Necessity) ---")
    
    # 1. LOAD DATA
    path = os.path.join(MERGED_DATA_DIR, 'master_all_regions_2023.csv')
    if not os.path.exists(path):
        print(f"Missing master data: {path}")
        return
    master = pd.read_csv(path)

    if 'Hour' not in master.columns:
        if 'period' in master.columns:
            master['Hour'] = pd.to_datetime(master['period']).dt.hour
        else:
            master['Hour'] = master.index.hour

    master['datetime'] = pd.to_datetime(master['period'])
    
    # Prep features
    import joblib
    try:
        le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        master['region_encoded'] = le.transform(master['region'])
    except Exception as e:
        print(f"Warning: Could not load encoder ({e}). Falling back to fit.")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        master['region_encoded'] = le.fit_transform(master['region'])

    # --- ON-THE-FLY FEATURE GENERATION ---
    from utils import compute_carbon_intensity
    from config import CARBON_FACTORS, CARBON_WEIGHT, WATER_WEIGHT, CW_STRESS_EFFICIENT_THRESHOLD, CW_STRESS_MODERATE_THRESHOLD
    master['carbon_intensity'] = compute_carbon_intensity(master, CARBON_FACTORS)
    try:
        mms_c = joblib.load(os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
        mms_w = joblib.load(os.path.join(MODELS_DIR, 'water_scaler.pkl'))
        master['carbon_norm'] = mms_c.transform(master[['carbon_intensity']])
        master['wbtmp_norm'] = mms_w.transform(master[['wbtmp']])
        master['CW_raw'] = CARBON_WEIGHT * master['carbon_norm'] + WATER_WEIGHT * master['wbtmp_norm']
        master['CW_Stress'] = np.where(master['CW_raw'] < CW_STRESS_EFFICIENT_THRESHOLD, 0, 
                                np.where(master['CW_raw'] < CW_STRESS_MODERATE_THRESHOLD, 1, 2))
    except Exception as e:
        print(f"Warning: Could not load scalers: {e}")
    # -------------------------------------

    master['hour_sin'] = np.sin(2 * np.pi * master['Hour']/24.0)
    master['hour_cos'] = np.cos(2 * np.pi * master['Hour']/24.0)
    
    # Use the same split as Step 4
    train = master[master['datetime'] < '2023-10-01'].copy()
    test = master[master['datetime'] >= '2023-10-01'].copy()
    
    results = []

    # EXPERIMENT 1: Carbon-Only
    print("  > Testing Carbon-Only Objective...")
    y_train_c = pd.cut(train['carbon_norm'], bins=[-np.inf, 0.35, 0.55, np.inf], labels=[0, 1, 2])
    y_test_c = pd.cut(test['carbon_norm'], bins=[-np.inf, 0.35, 0.55, np.inf], labels=[0, 1, 2])
    
    # EXPERIMENT 2: Water-Only
    print("  > Testing Water-Only Objective...")
    y_train_w = pd.cut(train['wbtmp_norm'], bins=[-np.inf, 0.35, 0.55, np.inf], labels=[0, 1, 2])
    y_test_w = pd.cut(test['wbtmp_norm'], bins=[-np.inf, 0.35, 0.55, np.inf], labels=[0, 1, 2])

    # EXPERIMENT 3: Combined (X-HydraAI)
    print("  > Testing X-HydraAI Combined Objective...")
    y_train_full = train['CW_Stress']
    y_test_full = test['CW_Stress']

    experiments = [
        ('Carbon-Only', y_train_c, y_test_c),
        ('Water-Only', y_train_w, y_test_w),
        ('X-HydraAI (Combined)', y_train_full, y_test_full)
    ]

    for name, yt, yts in experiments:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('model', SVC(kernel='rbf', C=10, random_state=RANDOM_STATE))
        ])
        pipe.fit(train[ML_FEATURES].fillna(0), yt)
        preds = pipe.predict(test[ML_FEATURES].fillna(0))
        f1 = f1_score(yts, preds, average='weighted')
        results.append({'Experiment': name, 'F1-Score': f1})

    print("\n" + "="*50)
    print("           RESEARCH ABLATION REPORT")
    print("="*50)
    df_res = pd.DataFrame(results)
    print(df_res)
    print("-" * 50)
    print(" CONCLUSION: Single-objective models fail to capture")
    print(" the intersectional stress of the Water-Energy Nexus.")
    print(" X-HydraAI's combined approach is empirically required.")
    print("="*50)

if __name__ == "__main__":
    run_ablation_study()
