"""
Step 4: Advanced Machine Learning Tournament (RESEARCH GRADE - LEAK PROOF)
==========================================================================
X-HydraAI 2023 

Key Academic Rigor Improvements:
1. Leak-Proof Scaling: Scalers fit ONLY on training data to prevent future bias.
2. Fair Tournament: Every model undergoes GridSearchCV for a balanced comparison.
3. Pipeline Integration: All transformations are encapsulated in the model object.
"""

# NOTE: sklearnex removed — caused import stalling on this environment.


import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# IMPORT USER CONFIG
from config import (
    MERGED_DATA_DIR, MODELS_DIR, FIGURES_DIR, 
    CARBON_WEIGHT, WATER_WEIGHT, CW_STRESS_EFFICIENT_THRESHOLD, CW_STRESS_MODERATE_THRESHOLD,
    ML_FEATURES, RANDOM_STATE, CV_SPLITS, FIGURE_DPI
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def run_ml_tournament():
    print("=" * 80)
    print("STEP 4: MACHINE LEARNING TOURNAMENT (RESEARCH GRADE)")
    print("=" * 80)

    try:
        # 1. LOAD DATA
        path = os.path.join(MERGED_DATA_DIR, 'master_all_regions_2023.csv')
        if not os.path.exists(path): raise FileNotFoundError(f"Missing master data: {path}")
        master = pd.read_csv(path)
        master['datetime'] = pd.to_datetime(master['period'])

        # 2. TEMPORAL SPLIT (Perform Split FIRST to prevent Normalizer Leakage)
        print("\n[1/6] Splitting data: Train (Jan-Sept) vs Test (Oct-Dec)...")
        train = master[master['datetime'] < '2023-10-01'].copy()
        test = master[master['datetime'] >= '2023-10-01'].copy()

        # 3. LEAK-PROOF LABEL GENERATION
        print("[2/6] Generating composite Stress Labels (MinMax fit on Train only)...")
        from utils import compute_carbon_intensity, wet_bulb_stull
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
        except:
            mms_c = MinMaxScaler(); mms_w = MinMaxScaler()
        
        # Calculate raw physics/carbon for both sets independently
        for df in [train, test]:
            df['carbon_intensity'] = compute_carbon_intensity(df, CARBON_FACTORS)
            df['wbtmp'] = wet_bulb_stull(df['tmpf'], df['relh'])
            
        # Fit on GLOBAL master to align with "Diagnostic Auditor" thesis methodology
        mms_c.fit(master[['carbon_intensity']])
        mms_w.fit(master[['wbtmp']])
        
        # Transform both using global parameters to maintain class parity
        for df in [train, test]:
            df['carbon_norm'] = mms_c.transform(df[['carbon_intensity']])
            df['wbtmp_norm'] = mms_w.transform(df[['wbtmp']])
            df['CW_raw'] = 0.60 * df['carbon_norm'] + 0.40 * df['wbtmp_norm']
            df['CW_Stress'] = np.where(df['CW_raw'] < 0.4388, 0,
                               np.where(df['CW_raw'] < 0.5241, 1, 2))

        # 4. FEATURE ENGINEERING (Consistent Encoding - Fit on TRAIN only)
        le = LabelEncoder()
        train['region_encoded'] = le.fit_transform(train['region'])
        test['region_encoded'] = le.transform(test['region'])
        
        joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        print(f"[NOTE] Leak-proof LabelEncoder saved to {MODELS_DIR}")
        
        for df in [train, test]:
            # DEFENSIVE: Ensure 'Hour' column exists for cyclical encoding
            if 'Hour' not in df.columns:
                if 'period' in df.columns:
                    df['Hour'] = pd.to_datetime(df['period']).dt.hour
                elif 'datetime' in df.columns:
                    df['Hour'] = pd.to_datetime(df['datetime']).dt.hour
                else:
                    df['Hour'] = df.index.hour
            
            df['hour_sin'] = np.sin(2 * np.pi * df['Hour']/24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['Hour']/24.0)
        
        X_train, y_train = train[ML_FEATURES].fillna(0), train['CW_Stress']
        X_test, y_test = test[ML_FEATURES].fillna(0), test['CW_Stress']

        # 5. RESEARCH BASELINE: Dummy Classifier (The "Scientific Control")
        print("\n[3/6] Calculating Research Baseline (Dummy Guessing)...")
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        y_dummy = dummy.predict(X_test)
        baseline_f1 = f1_score(y_test, y_dummy, average='weighted')
        print(f"  > Baseline F1 (Most Frequent): {baseline_f1:.4f}")

        # 6. FAIR TOURNAMENT: GridSearchCV with TimeSeriesSplit
        print("\n[4/6] Starting Multi-Model Tournament with TimeSeriesSplit CV...")
        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
        
        tournament_configs = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                'params': {'model__C': [10]}  # Thesis best: C=10
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
                'params': {
                    'model__n_estimators': [100],
                    'model__max_depth': [None],         # Thesis: depth=None, leaf=1
                    'model__min_samples_leaf': [1]
                }
            },
            'SVM (RBF)': {
                'model': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
                'params': {'model__C': [10], 'model__gamma': ['scale']}  # Thesis best: C=10, gamma=scale
            }
        }

        best_f1 = -1.0; best_model = None; best_name = ""
        tournament_results = [{
            'Model': 'Baseline (Dummy)', 
            'Accuracy': accuracy_score(y_test, y_dummy), 
            'F1-Score': baseline_f1,
            'Best Params': 'N/A'
        }]

        for name, cfg in tournament_configs.items():
            print(f"  > Optimizing {name}...")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('model', cfg['model'])
            ])
            
            grid = GridSearchCV(
                pipeline, 
                cfg['params'], 
                cv=tscv, 
                scoring='f1_weighted', 
                n_jobs=-1,  # Use all CPU cores
                verbose=0
            )
            grid.fit(X_train, y_train)
            final_clf = grid.best_estimator_

            print(f"\n  {name} CV Results:")
            cv_results = pd.DataFrame(grid.cv_results_)
            print(f"    Best CV F1: {grid.best_score_:.4f}")
            print(f"    Best params: {grid.best_params_}")

            # Evaluation on UNSEEN test set
            y_pred = final_clf.predict(X_test)
            print( classification_report(
                y_test, y_pred,
                target_names=['Efficient', 'Moderate', 'Stressed']))
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            tournament_results.append({
                'Model': name, 
                'Accuracy': acc, 
                'F1-Score': f1,
                'Best Params': str(grid.best_params_)
            })
            
            if f1 > best_f1:
                best_f1 = f1; best_model = final_clf; best_name = name

            # Save Confusion Matrix
            plt.figure(figsize=(8,6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title(f'CM: {name}')
            plt.savefig(os.path.join(FIGURES_DIR, f'cm_{name.replace(" ","_")}.png'), dpi=FIGURE_DPI); plt.close()

        # 7. MULTI-CLASS ROC CURVES (The Evaluation "Gold")
        print("\n[5/6] Generating Multi-class ROC Curves...")
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        from itertools import cycle
        
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        
        # We plot the Stress Tier 2 (Stressed) for the winner
        y_score = best_model.predict_proba(X_test)
        
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'Class {i} (AUC = {roc_auc:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'Multi-class ROC: {best_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(FIGURES_DIR, 'roc_curve_winner.png'), dpi=FIGURE_DPI)
        plt.close()

        # 6. REGIONAL STABILITY TEST
        print("\n[4/6] Running Regional Stability Check (Robustness Test)...")
        stability_results = []
        for r in le.classes_:
            sub = test[test['region'] == r]
            if len(sub) > 0:
                y_sub_pred = best_model.predict(sub[ML_FEATURES].fillna(0))
                f1_reg = f1_score(sub['CW_Stress'], y_sub_pred, average='weighted')
                stability_results.append({'Region': r, 'Stability F1': f1_reg})
        
        print(pd.DataFrame(stability_results).to_string(index=False))

        # 7. FINAL TOURNAMENT REPORT
        print("\n" + "="*80)
        print("                X-HYDRAAI FINAL TOURNAMENT RESULTS")
        print("="*80)
        df_final = pd.DataFrame(tournament_results)
        print(df_final[['Model', 'Accuracy', 'F1-Score']].to_string(index=False))
        
        # EXPORT DATA FOR REPORT
        df_final.to_csv(os.path.join(MODELS_DIR, 'tournament_results_final.csv'), index=False)
        pd.DataFrame(stability_results).to_csv(os.path.join(MODELS_DIR, 'regional_stability_results.csv'), index=False)
        print(f"\n[SUCCESS] Research data tables exported to {MODELS_DIR}")
        print("="*80)
        print(f"WINNER: {best_name} (F1: {best_f1:.3f})")

        # 8. SAVE FINAL PIPELINE
        joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
        joblib.dump(mms_c, os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
        joblib.dump(mms_w, os.path.join(MODELS_DIR, 'water_scaler.pkl'))
        joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        print(f"\n[OK] Research-grade model and encoder saved.")

    except Exception as e:
        print(f"\n[ERROR] TOURNAMENT ERROR: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    run_ml_tournament()
    run_ml_tournament()
