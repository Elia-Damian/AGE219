"""
=============================================================================
AGE 219 Capstone Project – Script 01: Data Loading, Merging & Cleaning
-----------------------------------------------------------------------------
Problem : Effect of Temperature (Frost and Heat Stress) on Crop Growth
          and Yield Loss in Farm Production (10-Dataset Study)
Author  : [Student Name] | Registration No: [Reg No]
Course  : AGE 219 – Basics of Computer Programming, SUA
-----------------------------------------------------------------------------
Description:
  Reads 10 Excel files (YEAR1.xlsx – YEAR10.xlsx), each containing
  experimental records of temperature stress (frost or heat) applied to
  field crops at various growth stages, along with the resulting yield
  loss fraction. The files are merged into a single Pandas DataFrame,
  cleaned, and saved as 'merged_data.csv' for downstream analysis.

  Columns in each raw file:
    Crop      -> Crop species (Wheat, Barley, Oat, Canola, Chickpea)
    F_H       -> Stress type: 'Frost' (low temp) or 'Heat' (high temp)
    Temp      -> Temperature at which stress was applied (°C)
    Duration  -> Duration of temperature exposure (hours)
    organ     -> Plant organ/developmental structure affected
    Stage     -> Growth stage (Pre-Anthesis, Anthesis, Post-Anthesis, etc.)
    loss      -> Fractional yield loss (0.0 = no loss, 1.0 = total loss)
    significant -> 's' (statistically significant) or 'ns' (not significant)
    Method of damage measurement -> free-text description
    Measurement -> What was measured (Survival, Grain Number, Yield, etc.)
    Reference -> Source citation

  Agronomic Context:
    Both frost (sub-zero temperatures) and heat stress (>30°C) during
    sensitive reproductive growth stages (especially Anthesis/flowering)
    cause severe yield losses in cereal and oilseed crops. This dataset
    compiles experimental evidence across 5 crops to quantify these
    temperature-yield relationships.
=============================================================================
"""

import os
import glob
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = DATA_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1.  Load all 10 Excel files programmatically ─────────────────────────────
print("=" * 65)
print("STEP 1 — Loading raw Excel (.xlsx) files")
print("=" * 65)

excel_files = sorted(
    glob.glob(os.path.join(DATA_DIR, 'YEAR*.xlsx')),
    key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0)
)

if len(excel_files) == 0:
    raise FileNotFoundError(f"No Excel files found in {DATA_DIR}. "
                            "Please ensure YEAR1.xlsx … YEAR10.xlsx are present.")

frames = []
for i, filepath in enumerate(excel_files, start=1):
    df_temp = pd.read_excel(filepath)
    df_temp['dataset_id'] = i
    df_temp['source_file'] = os.path.basename(filepath)
    frames.append(df_temp)
    print(f"  Loaded  {os.path.basename(filepath):14s}  →  {df_temp.shape[0]:3d} records")

print(f"\n  Total files loaded : {len(frames)}")

# ── 2.  Concatenate into a single DataFrame ───────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Concatenating all files")
print("=" * 65)

df = pd.concat(frames, ignore_index=True)
print(f"  Combined shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns         : {list(df.columns)}")

# ── 3.  Data Cleaning ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — Data Cleaning")
print("=" * 65)

# 3a. Standardise column names (remove spaces, lowercase-friendly)
df.rename(columns={
    'Crop': 'crop',
    'F_H': 'stress_type',
    'Temp': 'temperature_C',
    'Duration': 'duration_hr',
    'organ': 'plant_organ',
    'Stage': 'growth_stage',
    'loss': 'yield_loss_fraction',
    'significant': 'significance',
    'Method of damage measurement': 'damage_method',
    'Measurement': 'measurement_type',
    'Reference': 'reference'
}, inplace=True)

# 3b. Report missing values
print("\n  Missing values per column (before cleaning):")
print(df.isnull().sum().to_string())

# 3c. Drop rows with missing temperature or yield loss (core variables)
before = len(df)
df.dropna(subset=['temperature_C', 'yield_loss_fraction'], inplace=True)
print(f"\n  Rows dropped (missing temp or loss) : {before - len(df)}")

# 3d. Drop rows with implausible yield loss values (must be 0-1, allow small
#     negative measurement noise down to -0.2 as seen in raw data, but cap)
before = len(df)
df = df[(df['yield_loss_fraction'] >= -0.2) & (df['yield_loss_fraction'] <= 1.0)]
print(f"  Rows dropped (implausible loss values) : {before - len(df)}")

# Clip small negative noise to zero (measurement error, not real negative loss)
df['yield_loss_fraction'] = df['yield_loss_fraction'].clip(lower=0.0)

# 3e. Fill missing categorical fields with 'Unknown' / 'Not specified'
df['plant_organ']   = df['plant_organ'].fillna('Not specified')
df['damage_method'] = df['damage_method'].fillna('Not specified')
df['reference']     = df['reference'].fillna('Not cited')
df['significance']  = df['significance'].fillna('ns')

# 3f. Correct data types
df['temperature_C'] = df['temperature_C'].astype(float)
df['duration_hr']   = df['duration_hr'].astype(float)
df['yield_loss_fraction'] = df['yield_loss_fraction'].astype(float)

# 3g. Convert yield loss fraction to percentage for readability
df['yield_loss_pct'] = df['yield_loss_fraction'] * 100

# 3h. Classify stress severity bands
def classify_stress(row):
    t = row['temperature_C']
    if row['stress_type'] == 'Frost':
        if t >= -3:
            return 'Mild Frost'
        elif t >= -7:
            return 'Moderate Frost'
        else:
            return 'Severe Frost'
    else:  # Heat
        if t <= 32:
            return 'Mild Heat'
        elif t <= 36:
            return 'Moderate Heat'
        else:
            return 'Severe Heat'

df['stress_severity'] = df.apply(classify_stress, axis=1)

# 3i. Flag high-risk records (yield loss > 50%)
YIELD_LOSS_THRESHOLD = 0.50
df['high_loss_risk'] = (df['yield_loss_fraction'] > YIELD_LOSS_THRESHOLD).map(
    {True: 'High Loss', False: 'Acceptable'}
)

print(f"\n  Final shape after cleaning : {df.shape[0]} rows × {df.shape[1]} columns")

print(f"\n  Crop distribution:")
print(df['crop'].value_counts().to_string())

print(f"\n  Stress type distribution:")
print(df['stress_type'].value_counts().to_string())

print(f"\n  Yield loss risk summary:")
print(df['high_loss_risk'].value_counts().to_string())

# ── 4.  Save cleaned data ─────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'merged_data.csv')
df.to_csv(out_path, index=False)
print(f"\n  ✓  Cleaned data saved  →  {out_path}")

# ── 5.  Summary report ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — Descriptive Statistics by Crop and Stress Type")
print("=" * 65)

summary = (df.groupby(['crop', 'stress_type'])
             .agg(mean_temp=('temperature_C', 'mean'),
                  mean_loss_pct=('yield_loss_pct', 'mean'),
                  max_loss_pct=('yield_loss_pct', 'max'),
                  n_records=('yield_loss_pct', 'count'))
             .round(2))
print(summary.to_string())

print("\n  ✓  Script 01 complete.\n")
