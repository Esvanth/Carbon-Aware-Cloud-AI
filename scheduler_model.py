"""
Step 4: Build the Carbon-Aware Scheduler meta-model.
Predicts carbon emissions from workload + region features.
Evaluates ranking accuracy (Spearman correlation, top-3 recall).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import os
from config import *

def build_scheduler():
    """Train and evaluate the scheduler meta-model."""
    sim_path = os.path.join(OUTPUT_DIR, 'simulation_results.csv')
    if not os.path.exists(sim_path):
        print("[ERROR] simulation_results.csv not found. Run carbon_simulation.py first.")
        exit(1)
    df = pd.read_csv(sim_path)
    print(f"[OK] Simulation data: {len(df)} rows")

    # Encode model names as numbers
    model_names = df['model'].unique()
    model_map = {name: i for i, name in enumerate(model_names)}
    df['model_encoded'] = df['model'].map(model_map)

    # Features and target
    features = ['model_encoded', 'energy_kwh', 'carbon_intensity_gCO2_per_kWh',
                'renewables_share', 'train_time_s']
    target = 'carbon_gCO2'
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train two scheduler variants
    schedulers = {
        'RF Scheduler': RandomForestRegressor(n_estimators=100, random_state=42),
        'GBT Scheduler': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    results = []
    for name, model in schedulers.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # Ranking evaluation: per model-seed group
        spearman_scores = []
        top3_recalls = []
        for m in model_names:
            for seed in SEEDS:
                group = df[(df['model'] == m) & (df['seed'] == seed)]
                if len(group) < 5:
                    continue
                X_g = group[features].values
                y_actual = group[target].values
                y_predicted = model.predict(X_g)
                if len(set(y_actual)) > 1:
                    corr, _ = spearmanr(y_actual, y_predicted)
                    spearman_scores.append(corr)
                actual_top3 = set(np.argsort(y_actual)[:3])
                pred_top3 = set(np.argsort(y_predicted)[:3])
                top3_recalls.append(len(actual_top3 & pred_top3) / 3)
        avg_spearman = np.mean(spearman_scores) if spearman_scores else 0
        avg_top3 = np.mean(top3_recalls) if top3_recalls else 0
        results.append({
            'scheduler': name,
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'r2': round(r2, 6),
            'spearman_rank_corr': round(avg_spearman, 4),
            'top3_recall': round(avg_top3, 4)})
        print(f"    MAE:  {mae:.4f} gCO2")
        print(f"    R2:   {r2:.4f}")
        print(f"    Spearman rank correlation: {avg_spearman:.4f}")
        print(f"    Top-3 recall: {avg_top3:.4f}")
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            print(f"    Feature importance:")
            for fname, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
                print(f"      {fname}: {imp:.4f}")
    df_out = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, 'scheduler_results.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df_out

if __name__ == '__main__':
    print("="*50)
    print("CARBON-AWARE SCHEDULER META-MODEL")
    print("="*50)
    build_scheduler()