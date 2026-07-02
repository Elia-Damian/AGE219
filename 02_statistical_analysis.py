"""
=============================================================================
AGE 219 Capstone Project – Script 02: Statistical & Scientific Analysis
-----------------------------------------------------------------------------
Problem : Effect of Temperature (Frost and Heat Stress) on Crop Growth
          and Yield Loss in Farm Production (10-Dataset Study)
Author  : [Student Name] | Registration No: [Reg No]
Course  : AGE 219 – Basics of Computer Programming, SUA
-----------------------------------------------------------------------------
Description:
  Uses NumPy and SciPy to perform scientific analysis on the merged
  temperature-crop yield dataset produced by Script 01.

  (A) NumPy vectorised operations
      • Compute absolute temperature deviation from each crop's optimal
        growth temperature (20°C) using vectorised array maths
      • Compute a Temperature Stress Severity Index (TSSI)

  (B) SciPy statistical operations
      1. Linear regression of yield loss vs temperature deviation for
         FROST data — quantifies how much yield is lost per degree of
         cooling below freezing.
      2. Linear regression of yield loss vs temperature for HEAT data —
         quantifies yield loss per degree of warming above optimal.
      3. Pearson correlation between exposure Duration and Yield Loss —
         tests whether longer stress exposure worsens damage.
      4. Independent two-sample t-test comparing mean yield loss between
         Frost-stressed and Heat-stressed crops.
      5. One-way ANOVA (f_oneway) comparing yield loss across the 5 crop
         species to test whether crop type significantly affects
         temperature sensitivity.

  Results are printed to the console and saved to 'stats_results.csv'.
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = DATA_DIR
df = pd.read_csv(os.path.join(DATA_DIR, 'merged_data.csv'))

OPTIMAL_TEMP = 20.0   # °C — generalised optimal growth temperature for cereals/oilseeds

# ═══════════════════════════════════════════════════════════════════════════
# A.  NumPy Vectorised Operations
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("SECTION A — NumPy Vectorised Operations")
print("=" * 65)

# A1: Absolute temperature deviation from optimal growth temperature
temp_array = df['temperature_C'].values
df['temp_deviation_C'] = np.abs(temp_array - OPTIMAL_TEMP)

print("\n  A1: Temperature deviation from optimal (20°C) computed")
print(f"      Mean deviation : {df['temp_deviation_C'].mean():.2f} °C")
print(f"      Max  deviation : {df['temp_deviation_C'].max():.2f} °C")

# A2: Temperature Stress Severity Index (TSSI)
#     TSSI = (temperature deviation / max possible deviation) * duration weight
max_dev = df['temp_deviation_C'].max()
duration_array = df['duration_hr'].values
df['TSSI'] = np.multiply(
    np.divide(df['temp_deviation_C'].values, max_dev),
    np.sqrt(duration_array)   # vectorised sqrt for duration weighting
)
print(f"\n  A2: Temperature Stress Severity Index (TSSI) computed")
print(f"      Mean TSSI : {df['TSSI'].mean():.4f}")
print(f"      Max  TSSI : {df['TSSI'].max():.4f}")

# A3: Vectorised correlation matrix between numeric stress variables
numeric_cols = ['temperature_C', 'duration_hr', 'temp_deviation_C',
                'TSSI', 'yield_loss_pct']
corr_matrix = np.corrcoef(df[numeric_cols].values.T)
print(f"\n  A3: NumPy correlation matrix (yield_loss_pct row):")
yield_idx = numeric_cols.index('yield_loss_pct')
for i, col in enumerate(numeric_cols):
    print(f"      {col:20s} : r = {corr_matrix[yield_idx, i]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# B.  SciPy Statistical & Scientific Operations
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SECTION B — SciPy Statistical Analysis")
print("=" * 65)

# ── B1: Linear Regression — FROST: Yield Loss vs Temperature ─────────────
frost_df = df[df['stress_type'] == 'Frost'].copy()
slope_f, intercept_f, r_f, p_f, se_f = stats.linregress(
    frost_df['temperature_C'].values, frost_df['yield_loss_pct'].values)

print("\n  B1 — Linear Regression: FROST Yield Loss vs Temperature")
print(f"       Slope       : {slope_f:.4f}  (% yield loss per °C)")
print(f"       Intercept   : {intercept_f:.4f}")
print(f"       R²          : {r_f**2:.4f}")
print(f"       p-value     : {p_f:.6f}")
direction_f = "increases" if slope_f < 0 else "decreases"
print(f"       → As temperature drops (more negative), yield loss {direction_f}")

# ── B2: Linear Regression — HEAT: Yield Loss vs Temperature ──────────────
heat_df = df[df['stress_type'] == 'Heat'].copy()
slope_h, intercept_h, r_h, p_h, se_h = stats.linregress(
    heat_df['temperature_C'].values, heat_df['yield_loss_pct'].values)

print("\n  B2 — Linear Regression: HEAT Yield Loss vs Temperature")
print(f"       Slope       : {slope_h:.4f}  (% yield loss per °C)")
print(f"       Intercept   : {intercept_h:.4f}")
print(f"       R²          : {r_h**2:.4f}")
print(f"       p-value     : {p_h:.6f}")
direction_h = "increases" if slope_h > 0 else "decreases"
print(f"       → As temperature rises, yield loss {direction_h}")

# ── B3: Pearson Correlation — Duration vs Yield Loss ──────────────────────
r_dur, p_dur = stats.pearsonr(df['duration_hr'].values, df['yield_loss_pct'].values)
print(f"\n  B3 — Pearson Correlation: Exposure Duration vs Yield Loss")
print(f"       r = {r_dur:.4f},  p = {p_dur:.4f}")
if p_dur < 0.05:
    print("       → Statistically significant: longer exposure relates to higher loss")
else:
    print("       → No statistically significant relationship found")

# ── B4: Independent t-test — Frost vs Heat yield loss ─────────────────────
t_stat, p_ttest = stats.ttest_ind(
    frost_df['yield_loss_pct'].values,
    heat_df['yield_loss_pct'].values,
    equal_var=False  # Welch's t-test (unequal variance assumption)
)
print(f"\n  B4 — Independent t-test: Frost vs Heat Yield Loss")
print(f"       Frost mean loss : {frost_df['yield_loss_pct'].mean():.2f}%")
print(f"       Heat  mean loss : {heat_df['yield_loss_pct'].mean():.2f}%")
print(f"       t-statistic     : {t_stat:.4f}")
print(f"       p-value         : {p_ttest:.4f}")
if p_ttest < 0.05:
    worse = "Heat" if heat_df['yield_loss_pct'].mean() > frost_df['yield_loss_pct'].mean() else "Frost"
    print(f"       → SIGNIFICANT difference: {worse} stress causes significantly higher yield loss")
else:
    print("       → No significant difference between frost and heat damage")

# ── B5: One-Way ANOVA — Yield Loss across 5 Crop Species ──────────────────
crop_groups = [df[df['crop'] == c]['yield_loss_pct'].values
              for c in df['crop'].unique()]
f_stat, p_anova = stats.f_oneway(*crop_groups)
print(f"\n  B5 — One-Way ANOVA: Yield Loss Across Crop Species")
print(f"       F-statistic : {f_stat:.4f}")
print(f"       p-value     : {p_anova:.4f}")
if p_anova < 0.05:
    print("       → SIGNIFICANT: Crop species significantly affects temperature sensitivity")
else:
    print("       → No significant difference in sensitivity across crop species")

# Per-crop mean yield loss for context
crop_means = df.groupby('crop')['yield_loss_pct'].mean().sort_values(ascending=False)
print(f"\n       Mean yield loss by crop (most to least sensitive):")
for crop, val in crop_means.items():
    print(f"         {crop:10s} : {val:.2f}%")

# ── Save summary stats ────────────────────────────────────────────────────
regression_params = {
    'frost': {'slope': slope_f, 'intercept': intercept_f,
              'r_squared': r_f**2, 'p_value': p_f},
    'heat': {'slope': slope_h, 'intercept': intercept_h,
             'r_squared': r_h**2, 'p_value': p_h},
    'duration_correlation': {'r': r_dur, 'p': p_dur},
    'frost_vs_heat_ttest': {'t_stat': t_stat, 'p_value': p_ttest},
    'crop_anova': {'f_stat': f_stat, 'p_value': p_anova}
}
with open(os.path.join(OUTPUT_DIR, 'regression_params.json'), 'w') as fp:
    json.dump(regression_params, fp, indent=2)

# Save crop-level summary table
crop_summary = (df.groupby(['crop', 'stress_type'])
                   .agg(mean_temp=('temperature_C', 'mean'),
                        mean_loss_pct=('yield_loss_pct', 'mean'),
                        max_loss_pct=('yield_loss_pct', 'max'),
                        mean_TSSI=('TSSI', 'mean'),
                        n_records=('yield_loss_pct', 'count'))
                   .round(3)
                   .reset_index())
crop_summary.to_csv(os.path.join(OUTPUT_DIR, 'stats_results.csv'), index=False)

# Save the enriched dataset (with TSSI and deviation columns) for plotting
df.to_csv(os.path.join(OUTPUT_DIR, 'merged_data_enriched.csv'), index=False)

print(f"\n  ✓  Statistics saved to  →  {OUTPUT_DIR}/stats_results.csv")
print(f"  ✓  Regression parameters saved to  →  {OUTPUT_DIR}/regression_params.json")
print("  ✓  Script 02 complete.\n")
