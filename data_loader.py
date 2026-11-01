"""
Step 1: Load and preprocess all three datasets.
- UCI CCPP dataset (via ucimlrepo or local CSV/Excel)
- OWID energy data (carbon intensity per country)
- Cloud region mapping (74 regions)
"""

import pandas as pd
import numpy as np
import os
from config import *


def load_ccpp():
    """Load the UCI Combined Cycle Power Plant dataset."""

    # Method 1: Try ucimlrepo (cleanest approach)
    try:
        from ucimlrepo import fetch_ucirepo
        print("[OK] Fetching CCPP from UCI ML Repository...")
        dataset = fetch_ucirepo(id=294)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        print(f"  Loaded via ucimlrepo: {df.shape[0]} rows")
        # Save a local copy for future runs
        df.to_csv(CCPP_FILE, index=False)
        return _standardise_ccpp(df)
    except ImportError:
        print("[INFO] ucimlrepo not installed, trying local files...")
    except Exception as e:
        print(f"[WARN] ucimlrepo fetch failed: {e}. Trying local files...")

    # Method 2: Try local CSV files
    for name in [CCPP_FILE, 'ccpp_data.csv', 'Folds5x2_pp.csv', 'CCPP.csv', 'data.csv']:
        if os.path.exists(name):
            df = pd.read_csv(name)
            print(f"[OK] CCPP loaded from {name}: {df.shape[0]} rows")
            return _standardise_ccpp(df)

    # Method 3: Try Excel files
    for name in ['CCPP_data.xlsx', 'Folds5x2_pp.xlsx', 'CCPP.xlsx']:
        if os.path.exists(name):
            print(f"[OK] Found Excel file: {name}")
            df = pd.read_excel(name)
            df.to_csv(CCPP_FILE, index=False)
            print(f"     Converted to {CCPP_FILE}: {df.shape[0]} rows")
            return _standardise_ccpp(df)

    print("[ERROR] CCPP dataset not found!")
    print("  Option 1: pip install ucimlrepo  (auto-downloads)")
    print("  Option 2: Download manually from https://archive.ics.uci.edu/dataset/294")
    print(f"            Save as {CCPP_FILE} in this folder")
    exit(1)


def _standardise_ccpp(df):
    """Rename columns to standard names AT, V, AP, RH, PE."""
    col_map = {}
    for col in df.columns:
        c = col.strip().upper()
        if c in ['AT', 'TEMPERATURE', 'AMBIENT_TEMPERATURE']:
            col_map[col] = 'AT'
        elif c in ['V', 'VACUUM', 'EXHAUST_VACUUM']:
            col_map[col] = 'V'
        elif c in ['AP', 'PRESSURE', 'AMBIENT_PRESSURE']:
            col_map[col] = 'AP'
        elif c in ['RH', 'HUMIDITY', 'RELATIVE_HUMIDITY']:
            col_map[col] = 'RH'
        elif c in ['PE', 'POWER', 'ENERGY_OUTPUT', 'NET_HOURLY_ELECTRICAL_ENERGY_OUTPUT']:
            col_map[col] = 'PE'

    if col_map:
        df = df.rename(columns=col_map)

    # Validate
    needed = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[WARN] Columns {missing} not found. Using positional fallback.")
        df.columns = FEATURE_COLS + [TARGET_COL] + list(df.columns[5:])

    df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    print(f"  Features: {FEATURE_COLS}, Target: {TARGET_COL}")
    print(f"  Clean rows: {len(df)}")
    return df


def load_owid():
    """Load OWID energy data and extract carbon intensity per country."""
    for name in [OWID_FILE, 'owid_energy_data.csv']:
        if os.path.exists(name):
            df = pd.read_csv(name, low_memory=False)
            print(f"[OK] OWID loaded from {name}: {df.shape[0]} rows")
            return _process_owid(df)

    print("[ERROR] OWID energy dataset not found!")
    print("  Download: https://github.com/owid/energy-data")
    print(f"  Save as {OWID_FILE} in this folder")
    exit(1)


def _process_owid(df):
    """Extract latest carbon intensity and renewables share per country."""
    # Find carbon intensity column
    carbon_col = None
    for col in df.columns:
        if 'carbon_intensity' in col.lower():
            carbon_col = col
            break

    if carbon_col is None:
        print("[ERROR] No carbon_intensity column found in OWID data")
        exit(1)

    # Filter valid rows
    filtered = df[['iso_code', 'country', 'year', carbon_col]].dropna(subset=[carbon_col])
    filtered = filtered[filtered['iso_code'].notna()]
    filtered = filtered[~filtered['iso_code'].str.startswith('OWID')]

    # Latest year per country
    idx = filtered.groupby('iso_code')['year'].idxmax()
    result = filtered.loc[idx, ['iso_code', 'country', 'year', carbon_col]].copy()
    result = result.rename(columns={carbon_col: 'carbon_intensity_gCO2_per_kWh'})

    # Renewables share
    renew_col = None
    for col in df.columns:
        if 'renewables_share' in col.lower() and 'elec' in col.lower():
            renew_col = col
            break

    if renew_col:
        df_r = df[['iso_code', 'year', renew_col]].dropna(subset=[renew_col])
        df_r = df_r[df_r['iso_code'].notna()]
        idx_r = df_r.groupby('iso_code')['year'].idxmax()
        renew_latest = df_r.loc[idx_r, ['iso_code', renew_col]].rename(columns={renew_col: 'renewables_share'})
        result = result.merge(renew_latest, on='iso_code', how='left')
        result['renewables_share'] = result['renewables_share'].fillna(0)
    else:
        result['renewables_share'] = 0

    print(f"  Carbon intensity for {len(result)} countries")
    print(f"  Range: {result['carbon_intensity_gCO2_per_kWh'].min():.1f} - "
          f"{result['carbon_intensity_gCO2_per_kWh'].max():.1f} gCO2/kWh")
    return result


def load_regions(df_carbon):
    """Load cloud region mapping and merge with carbon intensity data."""
    if not os.path.exists(REGION_FILE):
        print(f"[ERROR] {REGION_FILE} not found! Place it in this folder.")
        exit(1)

    df = pd.read_csv(REGION_FILE)
    print(f"[OK] Cloud regions loaded: {len(df)} regions, {df['provider'].nunique()} providers")

    # Merge with carbon data
    merged = df.merge(
        df_carbon[['iso_code', 'carbon_intensity_gCO2_per_kWh', 'renewables_share']],
        on='iso_code', how='left'
    )

    # Fill missing with global average
    avg = df_carbon['carbon_intensity_gCO2_per_kWh'].mean()
    n_missing = merged['carbon_intensity_gCO2_per_kWh'].isna().sum()
    if n_missing > 0:
        print(f"  [WARN] {n_missing} regions missing carbon data, using global avg ({avg:.1f})")
    merged['carbon_intensity_gCO2_per_kWh'] = merged['carbon_intensity_gCO2_per_kWh'].fillna(avg)
    merged['renewables_share'] = merged['renewables_share'].fillna(0)

    # Save processed version
    merged.to_csv(os.path.join(OUTPUT_DIR, 'region_carbon_intensity.csv'), index=False)

    # Print extremes
    print(f"\n  Cleanest 3 regions:")
    for _, r in merged.nsmallest(3, 'carbon_intensity_gCO2_per_kWh').iterrows():
        print(f"    {r['provider']} {r['region_code']} ({r['country']}): {r['carbon_intensity_gCO2_per_kWh']:.1f} gCO2/kWh")
    print(f"  Dirtiest 3 regions:")
    for _, r in merged.nlargest(3, 'carbon_intensity_gCO2_per_kWh').iterrows():
        print(f"    {r['provider']} {r['region_code']} ({r['country']}): {r['carbon_intensity_gCO2_per_kWh']:.1f} gCO2/kWh")

    return merged


if __name__ == '__main__':
    print("Testing data loading...\n")
    df_ccpp = load_ccpp()
    print()
    df_carbon = load_owid()
    print()
    df_regions = load_regions(df_carbon)
    print("\nAll datasets loaded successfully!")