"""
Configuration file for Carbon-Aware Cloud AI experiment.
All settings in one place.
"""

import os

# Output directory
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random seeds for reproducibility
SEEDS = [42, 123, 456]

# Dataset file names (place these in the project folder)
CCPP_FILE = "CCPP_data.csv"
OWID_FILE = "owid-energy-data.csv"
REGION_FILE = "cloud_region_mapping.csv"

# CCPP dataset columns
FEATURE_COLS = ['AT', 'V', 'AP', 'RH']
TARGET_COL = 'PE'

# Plot style
COLOURS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']