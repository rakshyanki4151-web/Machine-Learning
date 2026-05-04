"""
Nepal Case Study: Master Aggregator
Combining all 4 climate zones with the National Grid into a single 35,040-row master file.
Mirroring the structure of the US 'master_all_regions_2023.csv'.
"""
import pandas as pd
import numpy as np
import os
import joblib
from config import MODELS_DIR, ML_FEATURES
from utils import wet_bulb_stull

# 1. SETUP PATHS
BASE_DIR = r"./data/kathmandu_case_study"
grid_path = os.path.join(BASE_DIR, 'National_Nepal_Grid_2023.csv')
weather_path = os.path.join(BASE_DIR, 'Kathmandu_VNKT_2023_Raw.csv')
out_path = os.path.join(BASE_DIR, 'Nepal_Master_All_Zones_2023.csv')

# 2. LOAD INFRASTRUCTURE
print("Loading model and grid...")
best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
mms_c = joblib.load(os.path.join(MODELS_DIR, 'carbon_scaler.pkl'))
mms_w = joblib.load(os.path.join(MODELS_DIR, 'water_scaler.pkl'))

grid_df = pd.read_csv(grid_path, index_col='datetime', parse_dates=True)
raw_weather = pd.read_csv(weather_path, na_values=['M', ' '])
raw_weather['datetime'] = pd.to_datetime(raw_weather['valid'] if 'valid' in raw_weather.columns else raw_weather['datetime'])
ktm_weather = raw_weather.set_index('datetime')[['tmpf', 'dwpf', 'relh']].resample('h').mean().interpolate().ffill().bfill()
full_year = pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='h')
ktm_weather = ktm_weather.reindex(full_year).ffill().bfill()

def process_zone(name, temp_offset, rh_offset):
    print(f"  -> Processing {name}...")
    df = ktm_weather.copy()
    
    # Merge with grid
    for col in grid_df.columns:
        df[col] = grid_df[col]
    
    # Apply Offsets
    df['tmpf'] = df['tmpf'] + (temp_offset * 1.8)
    df['relh'] = (df['relh'] + rh_offset).clip(0, 100)
    
    # Feature Engineering
    df['wbtmp'] = wet_bulb_stull(df['tmpf'], df['relh'])
    df['carbon_norm'] = mms_c.transform(df[['carbon_intensity']])
    df['wbtmp_norm'] = mms_w.transform(df[['wbtmp']])
    df['Hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour']/24.0)
    df['region'] = name
    df['region_encoded'] = 4 # All Nepal zones share code 4
    
    # Predict Classes using the trained AI
    df['CW_predicted_Stress'] = best_model.predict(df[ML_FEATURES])
    
    return df

# 3. GENERATE ALL 4 REGIONS
nepal_zones = [
    process_zone('Kathmandu', 0, 0),
    process_zone('Biratnagar', +6.0, +10.0),
    process_zone('Pokhara', +2.0, +20.0),
    process_zone('Lukla', -12.0, -20.0)
]

# 4. CONCAT AND SAVE
master_nepal = pd.concat(nepal_zones)
master_nepal.to_csv(out_path)

print(f"/nSUCCESS: Created {out_path}")
print(f"Total Rows: {len(master_nepal)} (4 regions * 8,760 hours)")
print(f"Total Columns: {len(master_nepal.columns)}")
