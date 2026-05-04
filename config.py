"""
X-HydraAI Configuration Module
================================
Single source of truth. All parameters, thresholds, paths.

References:
- IPCC AR6 WG3 (2022) Table 7.SM.7 — Carbon factors
- ASHRAE 90.4-2019 — Data center thermal guidelines  
- Stull (2011) JAMC — Wet-bulb temperature formula
"""

import os
from pathlib import Path

# ============================================================================
# PATHS — defined once using pathlib
# ============================================================================
# --- PATHS ------------------------------------------------------------------
PROJECT_ROOT    = Path('.').resolve()
DATA_DIR        = PROJECT_ROOT / 'data'
US_DATA_DIR     = DATA_DIR / 'us'
NEPAL_DATA_DIR  = DATA_DIR / 'nepal'
MODELS_DIR      = PROJECT_ROOT / 'models'
FIGURES_DIR     = PROJECT_ROOT / 'figures'

# For backward compatibility with older scripts
MERGED_DATA_DIR = US_DATA_DIR 

for d in [DATA_DIR, US_DATA_DIR, NEPAL_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# String versions for compatibility
DATA_DIR_STR = str(DATA_DIR)
MERGED_DATA_DIR_STR = str(MERGED_DATA_DIR)
FIGURES_DIR_STR = str(FIGURES_DIR)
MODELS_DIR_STR = str(MODELS_DIR)

# ============================================================================
# REGIONS — defined once as dict (station code needed for download)
# ============================================================================
REGIONS = {
    'NW':   '4S2',   # WECC Northwest — Oregon
    'CAL':  'KSFO',  # FIX: San Francisco (CAISO) - More representative of California Grid
    'PJM':  'JYO',   # PJM — Virginia (Leesburg/JYO)
    'ERCO': 'JWY',   # ERCOT — Texas (Midlothian/JWY)
}

# ============================================================================
# CARBON-WATER COMPOSITE LABEL (CW-Score)
# ============================================================================
CARBON_WEIGHT = 0.60   # 60% carbon intensity
WATER_WEIGHT  = 0.40   # 40% water/thermal stress
# Must sum to 1.0

# Fixed thresholds on normalised CW_raw score (0-1 range)
# NOT percentile-based — calibrated for realistic class distribution
CW_STRESS_EFFICIENT_THRESHOLD = 0.35   # below = Efficient (Class 0)
CW_STRESS_MODERATE_THRESHOLD  = 0.55   # below = Moderate  (Class 1)
                                        # above = Stressed  (Class 2)

# ============================================================================
# ASHRAE THRESHOLDS (for reference and paper context only)
# Source: ASHRAE 90.4-2019 Data Center Power Usage Effectiveness
# ============================================================================
WATER_STRESS_EFFICIENT_THRESHOLD = 50.0   # °F wet-bulb
WATER_STRESS_MODERATE_THRESHOLD  = 65.0   # °F wet-bulb
WATER_STRESS_CRITICAL_THRESHOLD  = 80.0   # °F wet-bulb

# ============================================================================
# IPCC AR6 CARBON EMISSION FACTORS (gCO2eq/kWh — lifecycle)
# Source: IPCC AR6 WG3 (2022), Annex III, Table 7.SM.7
# ============================================================================
CARBON_FACTORS = {
    'COL': 820,   # Coal
    'NG':  490,   # Natural gas
    'OIL': 650,   # Oil
    'NUC': 12,    # Nuclear
    'SUN': 41,    # Solar PV
    'WND': 11,    # Wind
    'WAT': 24,    # Hydroelectric
    'GEO': 45,    # Geothermal
    'BIO': 230,   # Biomass
    'WST': 200,   # Waste
    'OTH': 300,   # Other (conservative)
}

# ============================================================================
# ML FEATURES (Audit Proof: No direct formula inputs like COL or SUN)
# ============================================================================
ML_FEATURES = [
    'tmpf',               # dry bulb temperature °F
    'relh',               # relative humidity %
    'hour_sin',           # cyclical hour encoding
    'hour_cos',           # cyclical hour encoding
    'renewable_percent',  # Smart proxy for grid cleanliness
    'region_encoded',     # grid region
]

# ============================================================================
# ANOMALY DETECTION
# ============================================================================
ANOMALY_FEATURES = [
    'carbon_norm',
    'wbtmp_norm',
    'relh',
    'hour_sin',
    'hour_cos',
]
ANOMALY_CONTAMINATION = 0.05   # 5% expected anomaly rate
ANOMALY_RANDOM_STATE  = 42

# ============================================================================
# ML SETTINGS
# ============================================================================
RANDOM_STATE   = 42
CV_SPLITS      = 5
CV_SHUFFLE     = False   # preserve temporal order
PARADOX_CARBON_THRESHOLD = 0.30   # Threshold for "low carbon" in anomaly analysis
RF_N_ESTIMATORS = 200
RF_CLASS_WEIGHT = 'balanced'
LR_MAX_ITER     = 1000
SVM_KERNEL      = 'rbf'

# ============================================================================
# NEPAL GRID
# ============================================================================
NEPAL_FUEL_MIX = {
    'WAT': 0.95,
    'SUN': 0.02,
    'OTH': 0.03,
}
NEPAL_CARBON_INTENSITY_BASELINE = 24.0  # gCO2/kWh

# ============================================================================
# FUEL CATEGORIES
# ============================================================================
RENEWABLE_FUELS = ['SUN', 'WND', 'WAT', 'GEO', 'BIO']
FOSSIL_FUELS    = ['COL', 'NG', 'OIL']

# ============================================================================
# VISUALISATION
# ============================================================================
FIGURE_DPI    = 300
FIGURE_FORMAT = 'png'
CMAP_STRESS   = 'RdYlGn_r'

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    assert abs(CARBON_WEIGHT + WATER_WEIGHT - 1.0) < 0.001
    assert CW_STRESS_EFFICIENT_THRESHOLD < CW_STRESS_MODERATE_THRESHOLD
    assert all(v > 0 for v in CARBON_FACTORS.values())
    assert abs(sum(NEPAL_FUEL_MIX.values()) - 1.0) < 0.001
    assert len(ML_FEATURES) == len(set(ML_FEATURES))
    assert 'wbtmp' not in ML_FEATURES, \
        "wbtmp must not be in ML_FEATURES — it is used in label creation"
    return True

if __name__ == '__main__':
    validate_config()
    print("Config validated successfully")
    print(f"Regions: {list(REGIONS.keys())}")
    print(f"Features: {len(ML_FEATURES)}")
    print(f"CW thresholds: {CW_STRESS_EFFICIENT_THRESHOLD} / {CW_STRESS_MODERATE_THRESHOLD}")
