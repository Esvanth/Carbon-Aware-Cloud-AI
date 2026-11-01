"""
Step 5: Generate all 5 figures for the report.
Reads from CSVs produced by the previous steps.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import *

# Plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300
})


def load_data():
    """Load all result CSVs."""
    train = pd.read_csv(os.path.join(OUTPUT_DIR, 'training_results.csv'))
    sim = pd.read_csv(os.path.join(OUTPUT_DIR, 'simulation_results.csv'))
    csp = pd.read_csv(os.path.join(OUTPUT_DIR, 'csp_results.csv'))
    regions = pd.read_csv(os.path.join(OUTPUT_DIR, 'region_carbon_intensity.csv'))

    agg = train.groupby('model').agg({
        'train_time_s': 'mean', 'energy_kwh': 'mean',
        'mae': 'mean', 'rmse': 'mean', 'r2': 'mean'
    }).round(6).reset_index()

    return train, sim, csp, regions, agg


def fig1_energy(train, agg):
    """Bar chart: energy consumption by model."""
    print("  Figure 1: Energy by model...")
    models = agg['model'].values
    means = agg['energy_kwh'].values
    stds = train.groupby('model')['energy_kwh'].std().reindex(models).values

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(range(len(models)), means, yerr=stds,
                  color=COLOURS, edgecolor='black', linewidth=0.5, capsize=5, alpha=0.85,
                  width=0.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=0, ha='center', fontsize=11)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_title('Energy Consumption per Model (Mean Across 3 Seeds)', fontsize=14, pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds.max()*0.1,
                f'{val:.6f} kWh', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_energy_by_model.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig2_heatmap(sim, regions):
    """Heatmap: carbon emissions across models x top 15 regions."""
    print("  Figure 2: Carbon heatmap...")

    sim_avg = sim.groupby(['model', 'region_code', 'provider', 'country',
                            'carbon_intensity_gCO2_per_kWh']).agg({'carbon_gCO2': 'mean'}).reset_index()

    # Pick 15 regions spread across carbon intensity range
    sorted_r = regions.drop_duplicates('region_code').sort_values('carbon_intensity_gCO2_per_kWh')
    indices = np.linspace(0, len(sorted_r)-1, 15, dtype=int)
    top15 = sorted_r.iloc[indices]['region_code'].tolist()

    data = sim_avg[sim_avg['region_code'].isin(top15)]
    pivot = data.pivot_table(values='carbon_gCO2', index='model', columns='region_code', aggfunc='mean')

    region_order = sorted_r[sorted_r['region_code'].isin(top15)]['region_code'].tolist()
    pivot = pivot.reindex(columns=[r for r in region_order if r in pivot.columns])

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, linewidths=0.8,
                annot_kws={'size': 9}, cbar_kws={'label': 'Carbon Emissions (gCO2)', 'shrink': 0.8})
    ax.set_title('Simulated Carbon Emissions (gCO2) Across Models and Cloud Regions',
                 fontsize=14, pad=15)
    ax.set_xlabel('Cloud Region (sorted by carbon intensity: low to high)', fontsize=12)
    ax.set_ylabel('ML Model', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_carbon_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig3_country_carbon(regions):
    """Horizontal bar chart: carbon intensity by country."""
    print("  Figure 3: Carbon intensity by country...")

    country = regions.groupby('country').agg({
        'carbon_intensity_gCO2_per_kWh': 'mean',
        'region_code': 'count'
    }).reset_index().sort_values('carbon_intensity_gCO2_per_kWh')

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(country)))
    bars = ax.barh(range(len(country)), country['carbon_intensity_gCO2_per_kWh'].values,
                   color=bar_colors, edgecolor='black', linewidth=0.3, height=0.7)
    ax.set_yticks(range(len(country)))
    ax.set_yticklabels(country['country'].values, fontsize=10)
    ax.set_xlabel('Carbon Intensity (gCO2/kWh)', fontsize=12)
    ax.set_title('Grid Carbon Intensity by Country (Cloud Region Host Countries)',
                 fontsize=14, pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for bar, val in zip(bars, country['carbon_intensity_gCO2_per_kWh'].values):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val:.0f} gCO2/kWh',
                ha='left', va='center', fontsize=9)

    ax.set_xlim(0, country['carbon_intensity_gCO2_per_kWh'].max() * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_carbon_intensity_by_country.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig4_ccei(sim, regions, agg):
    """Grouped bar chart: CCEI across 5 representative regions."""
    print("  Figure 4: CCEI comparison...")

    sorted_r = regions.sort_values('carbon_intensity_gCO2_per_kWh')
    n = len(sorted_r)
    rep = sorted_r.iloc[[0, n//4, n//2, 3*n//4, n-1]]

    models = agg['model'].values
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(rep))
    width = 0.18

    for i, m in enumerate(models):
        ccei_vals = []
        r2_val = agg[agg['model'] == m]['r2'].values[0]
        energy_val = agg[agg['model'] == m]['energy_kwh'].values[0]

        for _, reg in rep.iterrows():
            ci = reg['carbon_intensity_gCO2_per_kWh']
            ccei = r2_val / (energy_val * ci) if (energy_val * ci) > 0 else 0
            ccei_vals.append(ccei)

        ax.bar(x + i * width, ccei_vals, width, label=m, color=COLOURS[i],
               edgecolor='black', linewidth=0.3, alpha=0.85)

    labels = [f"{r['provider']}\n{r['region_code']}\n({r['country']})\n{r['carbon_intensity_gCO2_per_kWh']:.0f} gCO2/kWh"
              for _, r in rep.iterrows()]
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=9, ha='center')
    ax.set_ylabel('CCEI (Cloud Carbon Efficiency Index)', fontsize=12)
    ax.set_title('CCEI Comparison Across Representative Cloud Regions', fontsize=14, pad=15)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_ccei_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig5_csp(csp):
    """Bar chart: Carbon Savings Potential per model."""
    print("  Figure 5: CSP bar chart...")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(range(len(csp)), csp['csp_percent'].values,
                  color=COLOURS[:len(csp)], edgecolor='black', linewidth=0.5, alpha=0.85,
                  width=0.6)
    ax.set_xticks(range(len(csp)))
    ax.set_xticklabels(csp['model'].values, rotation=0, ha='center', fontsize=11)
    ax.set_ylabel('Carbon Savings Potential (%)', fontsize=12)
    ax.set_title('Carbon Savings Potential (CSP) per Model\n(% Carbon Reduction: Best Region vs Worst Region)',
                 fontsize=14, pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)

    for bar, val in zip(bars, csp['csp_percent'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_csp_by_model.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("="*50)
    print("GENERATING ALL FIGURES")
    print("="*50)

    train, sim, csp, regions, agg = load_data()
    fig1_energy(train, agg)
    fig2_heatmap(sim, regions)
    fig3_country_carbon(regions)
    fig4_ccei(sim, regions, agg)
    fig5_csp(csp)

    print(f"\nAll 5 figures saved to {OUTPUT_DIR}/")