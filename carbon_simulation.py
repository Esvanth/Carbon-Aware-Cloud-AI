"""
Step 3: Simulate carbon emissions across 74 cloud regions.
Takes training results and multiplies energy by each region's carbon intensity.
Calculates CCEI and CSP novel metrics.
"""

import numpy as np
import pandas as pd
import os
from config import *
from data_loader import load_owid, load_regions

def run_simulation():
    """Simulate carbon for every model-seed-region combination."""
    # Load training results
    train_path = os.path.join(OUTPUT_DIR, 'training_results.csv')
    if not os.path.exists(train_path):
        print("[ERROR] training_results.csv not found. Run train_models.py first.")
        exit(1)
    df_train = pd.read_csv(train_path)
    print(f"[OK] Training results: {len(df_train)} rows")
    # Load region carbon data
    region_path = os.path.join(OUTPUT_DIR, 'region_carbon_intensity.csv')
    if os.path.exists(region_path):
        df_regions = pd.read_csv(region_path)
    else:
        df_carbon = load_owid()
        df_regions = load_regions(df_carbon)
    print(f"[OK] Regions: {len(df_regions)}")

    # Simulate carbon for every combination
    rows = []
    for _, model in df_train.iterrows():
        for _, region in df_regions.iterrows():
            ci = region['carbon_intensity_gCO2_per_kWh']
            carbon = model['energy_kwh'] * ci * 1000  # gCO2
            ccei = model['r2'] / (model['energy_kwh'] * ci) if (model['energy_kwh'] * ci) > 0 else 0

            rows.append({
                'model': model['model'],
                'seed': model['seed'],
                'provider': region['provider'],
                'region_code': region['region_code'],
                'country': region['country'],
                'carbon_intensity_gCO2_per_kWh': ci,
                'renewables_share': region['renewables_share'],
                'energy_kwh': model['energy_kwh'],
                'carbon_gCO2': round(carbon, 4),
                'ccei': round(ccei, 4),
                'r2': model['r2'],
                'train_time_s': model['train_time_s']
            })
    df_sim = pd.DataFrame(rows)
    sim_path = os.path.join(OUTPUT_DIR, 'simulation_results.csv')
    df_sim.to_csv(sim_path, index=False)
    print(f"\nSimulation: {len(df_sim)} rows saved to {sim_path}")

    # Calculate CSP per model
    print("\nCarbon Savings Potential (CSP):")
    csp_rows = []
    models = df_train['model'].unique()

    for m in models:
        sub = df_sim[df_sim['model'] == m]
        best = sub['carbon_gCO2'].min()
        worst = sub['carbon_gCO2'].max()
        csp = 1 - (best / worst) if worst > 0 else 0
        best_row = sub.loc[sub['carbon_gCO2'].idxmin()]
        worst_row = sub.loc[sub['carbon_gCO2'].idxmax()]
        csp_rows.append({
            'model': m,
            'best_carbon_gCO2': round(best, 4),
            'worst_carbon_gCO2': round(worst, 4),
            'best_region': f"{best_row['provider']} {best_row['region_code']}",
            'worst_region': f"{worst_row['provider']} {worst_row['region_code']}",
            'csp': round(csp, 4),
            'csp_percent': round(csp * 100, 1)})
        print(f"  {m}: {csp*100:.1f}% savings possible")
        print(f"    Best:  {best_row['provider']} {best_row['region_code']} ({best_row['country']})")
        print(f"    Worst: {worst_row['provider']} {worst_row['region_code']} ({worst_row['country']})")

    df_csp = pd.DataFrame(csp_rows)
    csp_path = os.path.join(OUTPUT_DIR, 'csp_results.csv')
    df_csp.to_csv(csp_path, index=False)
    print(f"\nSaved: {csp_path}")
    return df_sim, df_csp

if __name__ == '__main__':
    print("="*50)
    print("CARBON SIMULATION ACROSS 74 CLOUD REGIONS")
    print("="*50)
    run_simulation()