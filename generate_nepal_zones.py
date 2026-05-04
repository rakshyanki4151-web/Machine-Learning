"""
Nepal Case Study: Weather Generator (FIXED)
Renaming station IDs and rounding decimals for professional data representation.
"""
import pandas as pd
import os

BASE_DIR = r"./data/kathmandu_case_study"
KTM_PATH = os.path.join(BASE_DIR, "Kathmandu_VNKT_2023_Raw.csv")

df = pd.read_csv(KTM_PATH, na_values=['M', ' '])

ZONES = {
    "VNVT": {"name": "Biratnagar_Terai", "temp_f": +10.8, "rh": +10},
    "VNPK": {"name": "Pokhara_Hill",     "temp_f": +3.6,  "rh": +20},
    "VNLK": {"name": "Lukla_Alpine",     "temp_f": -21.6, "rh": -20}
}

print("Regenerating cleaned files...")

for sid, config in ZONES.items():
    synthetic = df.copy()
    synthetic['station'] = sid # Fix: Rename the station ID
    synthetic['tmpf'] = (synthetic['tmpf'] + config['temp_f']).round(1) # Round for realism
    synthetic['relh'] = (synthetic['relh'] + config['rh']).clip(0, 100).round(1)
    
    out_path = os.path.join(BASE_DIR, f"{config['name']}_Synthetic_2023.csv")
    synthetic.to_csv(out_path, index=False)
    print(f"  [FIXED] Saved {out_path} with station code {sid}")

print("/nSUCCESS: Files are now clean and correctly labeled.")
