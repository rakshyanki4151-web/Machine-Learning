# utils.py
# X-HydraAI 2023 
# Shared scientific functions for thermodynamically accurate feature engineering.

import numpy as np
import pandas as pd

def wet_bulb_stull(tmpf, relh):
    """
    Calculate wet-bulb temperature using the Stull (2011) formula.
    
    Citation: Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity 
    and Air Temperature. Journal of Applied Meteorology and Climatology, 
    50(11), 2267-2269. https://doi.org/10.1175/JAMC-D-11-0143.1
    
    Valid range: -20C to 50C, 5% to 99% RH
    
    Args:
        tmpf: Temperature in Fahrenheit (scalar or array)
        relh: Relative humidity in percent (scalar or array)
    
    Returns:
        Wet-bulb temperature in Fahrenheit
    """
    # Convert Fahrenheit to Celsius
    T_c = (tmpf - 32) * 5 / 9
    RH  = relh.fillna(50).clip(5, 99)  # Defensive: fill missing humidity with moderate 50%
    
    # Stull (2011) equation — Eq. 1
    Tw_c = (T_c * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
            + np.arctan(T_c + RH)
            - np.arctan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
            - 4.686035)
    
    # Convert back to Fahrenheit
    return Tw_c * 9 / 5 + 32


def compute_carbon_intensity(df, carbon_factors, total_col='Total_Energy_MWh'):
    """
    Calculate grid carbon intensity using IPCC AR6 lifecycle emission factors.
    
    Citation: IPCC (2022). Mitigation of Climate Change. Contribution of 
    Working Group III to the Sixth Assessment Report. Annex III, Table 7.SM.7.
    
    Args:
        df: DataFrame with fuel generation columns in MWh
        carbon_factors: dict of {fuel_code: gCO2/kWh} from config.py
        total_col: name of total energy column
    
    Returns:
        Series of carbon intensity values in gCO2/kWh
    """
    weighted_sum = pd.Series(0.0, index=df.index)
    for fuel, factor in carbon_factors.items():
        if fuel in df.columns:
            weighted_sum += df[fuel].fillna(0) * factor
    
    total = df[total_col].replace(0, np.nan)
    ci = weighted_sum / total
    return ci.fillna(ci.median() if not ci.isna().all() else 0.0)
