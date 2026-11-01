"""
Master runner -- executes the full experiment pipeline in order.
Just run: python run_all.py
"""

print("=" * 60)
print("=" * 60)

# Step 1: Load data
print("\n[STEP 1/5] Loading datasets...")
from data_loader import load_ccpp, load_owid, load_regions
df_ccpp = load_ccpp()
df_carbon = load_owid()
df_regions = load_regions(df_carbon)

# Step 2: Train models
print("\n[STEP 2/5] Training models with energy measurement...")
from train_models import run_training
run_training()

# Step 3: Carbon simulation
print("\n[STEP 3/5] Simulating carbon across 74 cloud regions...")
from carbon_simulation import run_simulation
run_simulation()

# Step 4: Scheduler meta-model
print("\n[STEP 4/5] Building Carbon-Aware Scheduler...")
from scheduler_model import build_scheduler
build_scheduler()

# Step 5: Generate figures
print("\n[STEP 5/5] Generating figures...")
from generate_figures import *
train, sim, csp, regions, agg = load_data()
fig1_energy(train, agg)
fig2_heatmap(sim, regions)
fig3_country_carbon(regions)
fig4_ccei(sim, regions, agg)
fig5_csp(csp)

# Done
print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print("\nCheck the results/ folder for:")
print("  CSVs:    training_results, simulation_results, csp_results, scheduler_results")
print("  Figures: fig1 through fig5 (all .png)")
