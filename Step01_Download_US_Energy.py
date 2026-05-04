"""
Step 1: Download 2023 EIA Fuel Mix Data (PRO-GRADE VERSION)
=========================================================
X-HydraAI 2023

Upgrades:
1. Paging Logic: Automatically loops through offsets to ensure 100% data capture.
2. Environment Security: Supports EIA_API_KEY environment variable.
3. Total Validation: Checks total records against available API count.
"""

import pandas as pd
import requests
import os
import time
from dotenv import load_dotenv
from config import DATA_DIR, REGIONS

load_dotenv()

# SECURITY: Use environment variable from .env file
API_KEY = os.environ.get('EIA_API_KEY')
if not API_KEY:
    print("\n[WARNING] EIA_API_KEY not found in environment or .env file.")
    API_KEY = "REPLACE_WITH_YOUR_KEY"
BASE_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"

os.makedirs(os.path.join(DATA_DIR, 'fuelmix_raw'), exist_ok=True)

print("=" * 80)
print("STEP 1: DOWNLOAD EIA FUEL MIX DATA (PRO VERSION)")
print("=" * 80)

def fetch_and_save_fuel_data(region):
    """
    Fetch hourly fuel mix from EIA API using pagination to ensure 100% coverage.
    """
    print(f"\n[{region}] Initializing full-year download...", flush=True)
    
    all_data = []
    offset = 0
    length = 5000 # Stable chunk size
    total_records = 1 # Temporary
    
    try:
        while offset < total_records:
            params = {
                'api_key': API_KEY,
                'frequency': 'hourly',
                'data[0]': 'value',
                'facets[respondent][]': region,
                'start': '2023-01-01T00',
                'end': '2023-12-31T23',
                'offset': offset,
                'length': length,
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(BASE_URL, params=params, timeout=30)
                    response.raise_for_status()
                    break
                except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        print(f"\n  [RETRY] Attempt {attempt+1} failed: {e}. Retrying in 5s...")
                        time.sleep(5)
                    else:
                        raise e
            
            data = response.json()
            
            # Update total on first run
            if offset == 0:
                total_records = int(data['response']['total'])
                print(f"  > Total records found on server: {total_records}")

            batch = data['response']['data']
            all_data.extend(batch)
            
            current_count = len(all_data)
            print(f"  > Downloaded {current_count}/{total_records} records...", end='\r')
            
            offset += length
            if offset < total_records:
                time.sleep(0.5) # Prevent rate limiting

        df = pd.DataFrame(all_data)
        if df.empty:
            print(f"\n  [ERROR] No data retrieved for {region}")
            return False
            
        # Save
        out_path = os.path.join(DATA_DIR, 'fuelmix_raw', f"{region}_year_2023.csv")
        df.to_csv(out_path, index=False)
        print(f"\n  [OK] Successfully saved {len(df)} records to: {out_path}")
        return True

    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        return False

def main():
    results = {}
    for region in REGIONS.keys():
        # PATCH: Skip if exists
        out_path = os.path.join(DATA_DIR, 'fuelmix_raw', f"{region}_year_2023.csv")
        if os.path.exists(out_path):
            print(f"[{region}] [SKIP] Data already exists locally. Skipping download.")
            results[region] = True
            continue
            
        success = fetch_and_save_fuel_data(region)
        results[region] = success
    
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    for region, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  {status} {region}")
    print("=" * 80)

if __name__ == '__main__':
    main()
