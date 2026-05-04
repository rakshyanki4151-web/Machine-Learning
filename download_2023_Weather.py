"""
Step 2: Download 2023 Weather Data (PERFECT STATE)
=================================================
X-HydraAI 2023

Fetches hourly weather observations from NOAA ASOS archive
for 4 US stations aligned to grid regions. Data includes:
- Ambient temperature (tmpf)
- Dew point (dwpf)
- Relative humidity (relh)
- Sea-level pressure (mslp)

Source: Iowa State University Mesonet (mesonet.agron.iastate.edu)
"""

import pandas as pd
import requests
import os
from config import DATA_DIR, REGIONS

os.makedirs(os.path.join(DATA_DIR, 'weather_raw'), exist_ok=True)

print("=" * 70)
print("STEP 2: DOWNLOAD NOAA WEATHER DATA")
print("=" * 70)
print(f"\nTargeting stations: {list(REGIONS.values())}")
print(f"Time period: 2023-01-01 to 2023-12-31 (hourly)")


def fetch_and_save_weather(region, station):
    """
    Fetch hourly weather from NOAA ASOS archive.
    
    Args:
        region (str): Region code (NW, CAL, PJM, ERCO)
        station (str): ASOS station code (4S2, SJC, PHL, AUS)
    
    Returns:
        bool: Success status
    """
    print(f"\n[{region}] Station {station} - fetching weather data...", flush=True)
    
    # NOAA ASOS CSV download endpoint
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={station}&"
        f"data=tmpf&data=dwpf&data=relh&data=mslp&"
        f"year1=2023&month1=1&day1=1&"
        f"year2=2023&month2=12&day2=31&"
        f"tz=Etc/UTC&format=onlycomma"
    )
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Check if response has content
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            print(f"  ✗ No data returned")
            return False
        
        record_count = len(lines) - 1  # Exclude header
        print(f"  [OK] Fetched {record_count} weather records")
        
        # Save
        out_path = os.path.join(DATA_DIR, 'weather_raw', f"{station}_{region}_2023.csv")
        with open(out_path, 'w') as f:
            f.write(response.text)
        print(f"  [OK] Saved to: {out_path}")
        
        return True
    
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout - server slow/unreachable")
        return False
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Network error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Main download pipeline."""
    results = {}
    
    for region, station in REGIONS.items():
        # PATCH: Skip if exists
        out_path = os.path.join(DATA_DIR, 'weather_raw', f"{station}_{region}_2023.csv")
        if os.path.exists(out_path):
            print(f"[{region}] [SKIP] Weather data already exists locally. Skipping download.")
            results[region] = True
            continue
            
        success = fetch_and_save_weather(region, station)
        results[region] = success
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    successful = sum(1 for s in results.values() if s)
    print(f"[OK] Successful: {successful}/{len(REGIONS)}")
    for region, success in results.items():
        status = "[OK]" if success else "✗"
        print(f"  {status} {region}")
    
    if successful == len(REGIONS):
        print("\n[OK] All weather data downloaded successfully!")
    else:
        print(f"\n⚠ {len(REGIONS) - successful} regions failed - check network")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
