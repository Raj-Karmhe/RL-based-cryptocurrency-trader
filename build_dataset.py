"""
build_dataset.py

Creates one final processed dataset.

Run ONLY once.
"""

import pandas as pd

from import_data import get_data
from indicators import add_indicators
from preprocess import preprocess_data


# ---------------------------------
# Download Data
# ---------------------------------

print("Downloading data...")

df = get_data()


# ---------------------------------
# Add Indicators
# ---------------------------------

print("Adding indicators...")

df = add_indicators(df)


# ---------------------------------
# Clean Dataset
# ---------------------------------

print("Preprocessing...")

df = preprocess_data(df)


# ---------------------------------
# Save
# ---------------------------------

df.to_csv("processed_data.csv", index=False)

print()

print("="*50)

print("Processed dataset saved!")

print(f"Rows : {len(df)}")

print(f"Columns : {len(df.columns)}")

print("="*50)