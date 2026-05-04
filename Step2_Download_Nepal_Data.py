import requests
import os

OUT_DIR = r"./data/kathmandu_case_study"
os.makedirs(OUT_DIR, exist_ok=True)

station = "VNKT"
name = "Kathmandu"

print(f"Downloading {name} ({station})...")
url = (
    f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    f"station={station}&data=tmpf&data=dwpf&data=relh&"
    f"year1=2023&month1=1&day1=1&year2=2023&month2=12&day2=31&"
    f"tz=Etc/UTC&format=onlycomma"
)

try:
    r = requests.get(url, timeout=60)
    if len(r.text) > 100:
        path = os.path.join(OUT_DIR, f"Kathmandu_VNKT_2023_Raw.csv")
        with open(path, 'w') as f:
            f.write(r.text)
        print(f"  [OK] Saved to {path}")
    else:
        print(f"  [Error] No data for {station}")
except Exception as e:
    print(f"  [Error] {e}")
