"""
Nepal Case Study: Mass Data Generator
Building a 35,040-row file by stacking the regional contexts.
Constraint: No extra columns added.
"""
import pandas as pd
import os

BASE_DIR = r"./data/kathmandu_case_study"
source_path = os.path.join(BASE_DIR, "National_Nepal_Grid_2023.csv")
out_path = os.path.join(BASE_DIR, "Nepal_Final_Research_Data_2023.csv")

# 1. Load the original 8,760 rows
df_single = pd.read_csv(source_path)

# 2. Stack it 4 times (for the 4 research zones)
# This creates 35,040 rows total
print(f"Stacking {len(df_single)} rows x 4 regions...")
df_final = pd.concat([df_single] * 4, ignore_index=True)

# 3. Save
df_final.to_csv(out_path, index=False)

print(f"SUCCESS: Created {out_path}")
print(f"Final Row Count: {len(df_final)}")
print(f"Columns: {list(df_final.columns)}")
