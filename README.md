# Carbon-Aware Cloud AI

**Minimizing Machine Learning Carbon Emissions Through Grid-Intensity-Based Workload Scheduling**

A framework that measures real ML training energy consumption and simulates carbon emissions across 74 cloud regions (AWS, Azure, GCP) spanning 25 countries. Introduces two novel metrics (CCEI and CSP) and a Carbon-Aware Scheduler for automated low-carbon region selection.

## Overview

The same ML training job produces vastly different carbon emissions depending on the cloud region. A model trained in Norway (28 gCO2/kWh) generates 32x less carbon than the same model in Bahrain (902 gCO2/kWh). This project quantifies that gap and automates the solution.

## Key Features

- **Real Energy Measurement** -- CodeCarbon tracks actual kWh during training
- **74 Cloud Region Simulation** -- AWS (23), Azure (27), GCP (24) mapped to country-level carbon intensity
- **Novel Metrics:**
  - CCEI (Cloud Carbon Efficiency Index) = R-squared / (Energy x Carbon Intensity)
  - CSP (Carbon Savings Potential) = 1 - (Best Region Carbon / Worst Region Carbon)
- **Carbon-Aware Scheduler** -- RF meta-model that ranks regions by predicted carbon output
- **4 ML Models Compared** -- Linear Regression, Random Forest, Gradient Boosted Trees, MLP Neural Network

## Results

| Model | R-squared | Energy (kWh) | CSP (%) |
|---|---|---|---|
| Linear Regression | 0.9264 | 0.000026 | 100.0 |
| Random Forest | 0.9607 | 0.000008 | 97.9 |
| Gradient Boosted Trees | 0.9459 | 0.000010 | 97.5 |
| MLP Neural Network | 0.9346 | 0.000096 | 97.8 |

**Scheduler Performance:** Spearman rank correlation of 0.9995, Top-3 Recall of 1.0

**Best Region:** Azure norwayeast (Norway, 28.11 gCO2/kWh)
**Worst Region:** AWS me-south-1 (Bahrain, 902.24 gCO2/kWh)

## Project Structure

```
Carbon-Aware-Cloud-AI/
├── config.py                  # Central configuration (seeds, paths, constants)
├── data_loader.py             # Dataset loading and preprocessing
├── train_models.py            # Model training with CodeCarbon energy tracking
├── carbon_simulation.py       # Carbon emission simulation and metric calculation
├── scheduler_model.py         # Carbon-Aware Scheduler meta-model
├── generate_figures.py        # Report visualisations (5 figures)
├── run_all.py                 # Master runner (executes full pipeline)
├── requirements.txt           # Python dependencies
├── cloud_region_mapping.csv   # 74 cloud regions mapped to countries
├── owid-energy-data.csv       # OWID carbon intensity data
└── results/                   # Output CSVs and figures (generated)
```

## Installation

```bash
git clone https://github.com/esvanth/Carbon-Aware-Cloud-AI.git
cd Carbon-Aware-Cloud-AI
pip install -r requirements.txt
```

## Usage

```bash
python run_all.py
```

This runs the full pipeline:
1. Loads datasets (CCPP via ucimlrepo, OWID, cloud region mapping)
2. Trains 4 models with energy tracking (3 seeds each)
3. Simulates carbon across 74 regions
4. Builds the Carbon-Aware Scheduler
5. Generates all figures

Results are saved to the `results/` folder.

## Datasets

| Dataset | Source | Access |
|---|---|---|
| UCI CCPP (9,568 rows) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/294) | Auto-downloaded via `ucimlrepo` |
| OWID Energy (23,378 rows) | [Our World in Data](https://github.com/owid/energy-data) | CSV included in repo |
| Cloud Region Mapping (74 rows) | Custom | CSV included in repo |

## SDG Alignment

- **SDG 7** (Affordable and Clean Energy) -- Encourages renewable-powered cloud regions
- **SDG 9** (Industry, Innovation and Infrastructure) -- Novel metrics and scheduler tooling
- **SDG 13** (Climate Action) -- 97-100% carbon reduction achievable through region selection

## Tech Stack

- Python, scikit-learn, CodeCarbon, pandas, NumPy, matplotlib, seaborn, scipy

## Authors

- Adith Kattuparambil Biji
- Esvanth Mohankumar
- Karthick Subramanian Muthukkaruppan

MSc Computing (Artificial Intelligence), National College of Ireland
Module: Emerging AI Technologies and Sustainability

## License

This project is for the academic module project