"""
Step 3: Full Merge and Feature Engineering (Documented - Audit Ready)
=====================================================================
X-HydraAI 2023

This script merges fuel-mix (EIA) and weather (NOAA) data.
Includes psychrometric wet-bulb temperature and temporal encoding.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from config import DATA_DIR, MERGED_DATA_DIR, REGIONS, RENEWABLE_FUELS, FOSSIL_FUELS

os.makedirs(MERGED_DATA_DIR, exist_ok=True)

def _find_fuel_file(region):
    candidates = [
        DATA_DIR / 'fuelmix_raw' / f"{region}_year_2023.csv",
        DATA_DIR / 'fuelmix' / f"{region}_year_2023.csv",
        DATA_DIR / 'raw' / f"{region}_fuel_2023.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _find_weather_file(station, region):
    import glob
    # Flexible search: matches "JYO_Virginia_PJM.csv" or "JYO_PJM_2023.csv"
    patterns = [
        DATA_DIR / 'weather_raw' / f"{station}*{region}*.csv",
        DATA_DIR / 'weather' / f"{station}*{region}*.csv",
        DATA_DIR / 'raw' / f"{station}*{region}*.csv",
    ]
    for pattern in patterns:
        files = glob.glob(str(pattern))
        if files:
            return Path(files[0])
    return None

def merge_and_engineer_region(region, station):
    """
    Merge fuel and weather data for a specific region and perform engineering.
    """
    print(f"\n[Processing {region}]", flush=True)
    
    try:
        # LOAD FUEL
        fuel_path = _find_fuel_file(region)
        if fuel_path is None: raise FileNotFoundError(f"Fuel file not found for {region}")
        fuel = pd.read_csv(fuel_path)
        
        # FIX: Ensure values are numeric before pivoting
        if 'value' in fuel.columns:
            fuel['value'] = pd.to_numeric(fuel['value'], errors='coerce').fillna(0)
        
        if 'period' in fuel.columns: fuel['period'] = pd.to_datetime(fuel['period'])
        elif 'datetime' in fuel.columns: fuel['period'] = pd.to_datetime(fuel['datetime'])
        
        # FIX: Check for fueltype or fuel-source
        fcol = 'fueltype' if 'fueltype' in fuel.columns else 'fuel-source' if 'fuel-source' in fuel.columns else None
        
        if fcol and 'value' in fuel.columns:
            fuel_pivot = fuel.pivot_table(index='period', columns=fcol, values='value', aggfunc='sum').fillna(0)
        else:
            fuel_pivot = fuel.set_index('period').select_dtypes(include=[np.number]).fillna(0)
        
        # Ensure only numeric columns are summed
        fuel_pivot = fuel_pivot.select_dtypes(include=[np.number])
        fuel_pivot['Total_Energy_MWh'] = fuel_pivot.sum(axis=1)

        # LOAD WEATHER
        weather_path = _find_weather_file(station, region)
        if weather_path is None: raise FileNotFoundError(f"Weather file not found for {station}")
        weather = pd.read_csv(weather_path, na_values=['M', ' '])
        if 'valid' in weather.columns: weather['datetime'] = pd.to_datetime(weather['valid'])
        else: weather['datetime'] = pd.to_datetime(weather['datetime'])
        
        # Ensure only numeric columns are resampled to avoid agg errors
        numeric_weather = ['tmpf', 'dwpf', 'relh', 'mslp']
        available_cols = [c for c in numeric_weather if c in weather.columns]
        for col in available_cols:
            weather[col] = pd.to_numeric(weather[col], errors='coerce')
        weather = weather.set_index('datetime')
        weather = weather[available_cols].resample('h').mean().interpolate().ffill().bfill()

        # ALIGN AND MERGE
        full_range = pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='h')
        fuel_pivot.index = fuel_pivot.index.tz_localize(None) if fuel_pivot.index.tz else fuel_pivot.index
        weather.index = weather.index.tz_localize(None) if weather.index.tz else weather.index
        
        # FIX: Ensure no duplicate labels before reindexing
        fuel_pivot = fuel_pivot.groupby(level=0).mean()
        weather = weather.groupby(level=0).mean()
        
        fuel_pivot = fuel_pivot.reindex(full_range).ffill().bfill()
        weather = weather.reindex(full_range).interpolate().ffill().bfill()
        
        master = fuel_pivot.join(weather, how='left').interpolate().ffill().bfill()

        # ENGINEERING
        master['Hour'] = master.index.hour
        master['Month'] = master.index.month
        
        master['hour_sin'] = np.sin(2 * np.pi * master['Hour']/24.0)
        master['hour_cos'] = np.cos(2 * np.pi * master['Hour']/24.0)
        
        from utils import wet_bulb_stull, compute_carbon_intensity
        from config import CARBON_FACTORS
        
        if 'tmpf' in master.columns and 'relh' in master.columns:
            master['wbtmp'] = wet_bulb_stull(master['tmpf'], master['relh'])
        elif 'tmpf' in master.columns and 'dwpf' in master.columns:
            # Estimate relh from dewpoint as fallback
            master['relh_est'] = 100 * np.exp(
                17.625 * (master['dwpf'] - 32) * 5/9 / 
                (243.04 + (master['dwpf'] - 32) * 5/9)) / \
                np.exp(17.625 * (master['tmpf'] - 32) * 5/9 / 
                (243.04 + (master['tmpf'] - 32) * 5/9))
            master['wbtmp'] = wet_bulb_stull(master['tmpf'], master.get('relh', master['relh_est']))
        
        renewable_cols = [c for c in RENEWABLE_FUELS if c in master.columns]
        if renewable_cols: 
            master['Renewable_MWh'] = master[renewable_cols].sum(axis=1)
            master['renewable_percent'] = (master['Renewable_MWh'] / master['Total_Energy_MWh'].replace(0, np.nan)) * 100.0
            
        fossil_cols = [c for c in FOSSIL_FUELS if c in master.columns]
        if fossil_cols:
            master['Fossil_MWh'] = master[fossil_cols].sum(axis=1)
            master['Fossil_Pct'] = (master['Fossil_MWh'] / master['Total_Energy_MWh'].replace(0, np.nan)) * 100.0
            
        master['carbon_intensity'] = compute_carbon_intensity(master, CARBON_FACTORS)
        
        out_path = MERGED_DATA_DIR / f"{region}_master_2023.csv"
        master.reset_index(names='period').to_csv(out_path, index=False)
        return True, {'region': region, 'rows': len(master)}
    
    except Exception as e:
        print(f"  [Error] {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}

def main():
    """Main execution block for Step 3."""
    for region, station in REGIONS.items():
        merge_and_engineer_region(region, station)
        
    parts = [pd.read_csv(os.path.join(MERGED_DATA_DIR, f"{r}_master_2023.csv")).assign(region=r) 
             for r in REGIONS.keys() if os.path.exists(os.path.join(MERGED_DATA_DIR, f"{r}_master_2023.csv"))]
    if parts:
        master_all = pd.concat(parts, ignore_index=True)
        master_all.to_csv(os.path.join(MERGED_DATA_DIR, 'master_all_regions_2023.csv'), index=False)
        print("\n[Step 3 Complete - All Regions Merged]")
        
        # SUMMARY TABLE (Matches Technical Improvements Report)
        e = len(master_all[master_all['wbtmp'] < 50])
        m = len(master_all[(master_all['wbtmp'] >= 50) & (master_all['wbtmp'] < 65)])
        s = len(master_all[master_all['wbtmp'] >= 65])
        tot = len(master_all)
        print("\nWater Stress Distribution (ASHRAE-based):")
        print(f"  Class 0 - EFFICIENT (wbtmp < 50.0°F):   {e:5d} hours ( {e/tot*100:4.1f}%)")
        print(f"  Class 1 - MODERATE  (50.0-65.0°F):      {m:5d} hours ( {m/tot*100:4.1f}%)")
        print(f"  Class 2 - STRESSED  (wbtmp >= 65.0°F):   {s:5d} hours ( {s/tot*100:4.1f}%)")

if __name__ == '__main__':
    main()
