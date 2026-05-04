"""
X-HydraAI Nepal Pipeline: Full Preprocessing (v2 - Calibrated)
Mirroring the scientific treatment of the US master dataset.
Target: data/merged/master_nepal_treated_2023.csv (35,040 rows)
"""
import pandas as pd
import numpy as np
import os
import joblib
from config import (MODELS_DIR, DATA_DIR, MERGED_DATA_DIR_STR, 
                    CARBON_WEIGHT, WATER_WEIGHT,
                    CW_STRESS_EFFICIENT_THRESHOLD, CW_STRESS_MODERATE_THRESHOLD)
from utils import wet_bulb_stull

# 1. SETUP PATHS
NEPAL_DIR = os.path.join(DATA_DIR, 'nepal')
grid_path = os.path.join(NEPAL_DIR, 'National_Nepal_Grid_2023.csv')
out_path = os.path.join(NEPAL_DIR, 'master_nepal_treated_2023.csv')

# Load scalers for consistency with US training
print("Loading X-HydraAI scalers...")
mms_c = joblib.load(os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
mms_w = joblib.load(os.path.join(MODELS_DIR, 'water_scaler.pkl'))

# 2. LOAD GRID
grid_df = pd.read_csv(grid_path, index_col='datetime', parse_dates=True)
full_year = pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='h')
grid_df = grid_df.reindex(full_year).ffill().bfill()

def treat_region(name, weather_filename, region_id):
    print(f"  -> Treating {name}...")
    w_path = os.path.join(NEPAL_DIR, weather_filename)
    
    # Load and clean weather
    w_raw = pd.read_csv(w_path, na_values=['M', ' '])
    # Handle both 'valid' and 'datetime' column names
    dt_col = 'valid' if 'valid' in w_raw.columns else 'datetime'
    w_raw['datetime'] = pd.to_datetime(w_raw[dt_col])
    
    # DEFENSIVE: Only select columns that exist in the file
    cols = [c for c in ['tmpf', 'dwpf', 'relh', 'mslp'] if c in w_raw.columns]
    w_clean = w_raw.set_index('datetime')[cols].resample('h').mean().interpolate().ffill().bfill()
    w_clean = w_clean.reindex(full_year).ffill().bfill()
    
    # Merge with grid
    df = w_clean.join(grid_df)
    
    # FEATURE ENGINEERING
    df['wbtmp'] = wet_bulb_stull(df['tmpf'], df['relh'])
    df['Month'] = df.index.month
    df['Hour'] = df.index.hour
    
    # Energy Ratios (Synchronized with US Pipeline naming)
    df['Renewable_MWh'] = df['WAT'] + df['SUN'] + df['WND']
    df['Fossil_MWh'] = df['COL'] + df['NG'] + df['OIL']
    df['renewable_percent'] = (df['Renewable_MWh'] / df['Total_Energy_MWh'] * 100).fillna(0)
    df['fossil_percent'] = (df['Fossil_MWh'] / df['Total_Energy_MWh'] * 100).fillna(0)
    
    # Normalization (Crucial for ML consistency)
    df['carbon_norm'] = mms_c.transform(df[['carbon_intensity']])
    df['wbtmp_norm'] = mms_w.transform(df[['wbtmp']])
    
    # Class Selection (Official X-HydraAI Metric)
    # Using official weights from config.py
    df['CW_raw'] = (df['carbon_norm'] * CARBON_WEIGHT) + (df['wbtmp_norm'] * WATER_WEIGHT)
    
    # Assign Classes based on official thresholds
    df['CW_Stress'] = 1 # Moderate
    df.loc[df['CW_raw'] <= CW_STRESS_EFFICIENT_THRESHOLD, 'CW_Stress'] = 0 # Efficient
    df.loc[df['CW_raw'] > CW_STRESS_MODERATE_THRESHOLD, 'CW_Stress'] = 2 # Stressed
    
    df['Water_Stress'] = 0 # Baseline
    df['region'] = name
    df['region_encoded'] = region_id
    
    # Cyclic hour
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour']/24.0)
    
    return df

# 3. RUN PIPELINE FOR ALL 4 ZONES
nepal_treated = [
    treat_region('Kathmandu', 'Kathmandu_VNKT_2023_Raw.csv', 4),
    treat_region('Biratnagar', 'Biratnagar_Terai_Synthetic_2023.csv', 5),
    treat_region('Pokhara', 'Pokhara_Hill_Synthetic_2023.csv', 6),
    treat_region('Lukla', 'Lukla_Alpine_Synthetic_2023.csv', 7)
]

# 4. CONSOLIDATE
master_treated = pd.concat(nepal_treated)
master_treated.index.name = 'datetime'
master_treated.to_csv(out_path)

print(f"\nFINISH: Created the Treated Master file ({len(master_treated)} rows)")
print(f"Path: {out_path}")
print(f"Summary Concentration:")
print(master_treated.groupby('region')['CW_Stress'].value_counts(normalize=True).unstack().fillna(0) * 100)
