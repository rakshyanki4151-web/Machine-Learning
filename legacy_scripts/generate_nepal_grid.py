"""
Nepal Case Study: High-Performance Grid Generator (Vectorized)
Implementing seasonal carbon logic using NumPy Vectorization.
Logic remains identical: 15 / 220 / 250 gCO2/kWh.
Efficiency: Broad-array selection (O[1] logic) vs. Row-iteration.
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = r"./data/kathmandu_case_study"

# 1. GENERATE TIMESTAMPS
full_year = pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='h')
df = pd.DataFrame(index=full_year)
df.index.name = 'datetime'

# 2. VECTORIZED LOGIC ENGINE (The "np.select" professional standard)
print("Executing vectorized carbon selection...")

# Extracts variables for whole array at once
months = df.index.month
hours = df.index.hour

# Define logical conditions
is_monsoon = np.isin(months, [6, 7, 8, 9])
is_evening_peak = (hours >= 18) & (hours <= 21)

conditions = [
    is_monsoon,                  # Monsoon dominance
    (~is_monsoon) & is_evening_peak # Dry season electrical stress
]

choices = [
    15.0,   # Carbon Intensity for Monsoon
    250.0   # Carbon Intensity for Dry/Peak
]

# Apply: defaulting to 220.0 for normal Dry season hours
df['carbon_intensity'] = np.select(conditions, choices, default=220.0)

# 3. FUEL COMPATIBILITY (Fixed ratios for Nepal context)
df['WAT'] = np.where(df['carbon_intensity'] == 15.0, 99.0, 70.0)
df['OTH'] = 100.0 - df['WAT'] - 2.0
df['SUN'] = 2.0
df['Total_Energy_MWh'] = 100.0
for col in ['COL', 'NG', 'OIL', 'NUC', 'WND', 'GEO', 'BIO', 'WST']:
    df[col] = 0.0

# 4. SAVE CORE FILE
out_path = os.path.join(BASE_DIR, "National_Nepal_Grid_2023.csv")
df.to_csv(out_path)

print(f"/nSUCCESS: National Nepal Grid generated using Vectorized Boolean Selection.")
print(f"Path: {out_path}")
print(f"Verification: {df['carbon_intensity'].unique()} gCO2 values found.")
