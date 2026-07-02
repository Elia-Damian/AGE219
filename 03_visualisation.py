"""
=============================================================================
AGE 219 Capstone Project – Script 03: Scientific Visualization
-----------------------------------------------------------------------------
Problem : Effect of Temperature (Frost and Heat Stress) on Crop Growth
          and Yield Loss in Farm Production (10-Dataset Study)
Author  : [Student Name] | Registration No: [Reg No]
Course  : AGE 219 – Basics of Computer Programming, SUA
-----------------------------------------------------------------------------
Description:
  Generates four publication-quality Matplotlib plots:

  Plot 1 — Trend Analysis (Line Chart)
    Mean yield loss percentage per crop across temperature severity bands,
    showing how damage escalates from mild to severe stress.

  Plot 2 — Categorical Comparison (Bar Chart)
    Mean yield loss percentage by crop species, split by stress type
    (Frost vs Heat), with error bars.

  Plot 3 — Correlation Plot (Scatter + Trend Line)
    Temperature vs Yield Loss for Frost data, with SciPy regression line
    overlaid, demonstrating the temperature-damage relationship.

  Plot 4 — Bonus: Heat Stress Scatter with Trend Line
    Temperature vs Yield Loss for Heat data, colour-coded by crop.

  All plots comply with engineering plot standards: descriptive title,
  axis labels with units, legend, and grid.
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, 'merged_data_enriched.csv'))
stats_df = pd.read_csv(os.path.join(DATA_DIR, 'stats_results.csv'))
with open(os.path.join(DATA_DIR, 'regression_params.json')) as fp:
    reg = json.load(fp)

crop_colors = {
    'Wheat': '#d4a017', 'Barley': '#8b7355', 'Oat': '#c9a96e',
    'Canola': '#f4d03f', 'Chickpea': '#7d8c4f'
}
severity_order = ['Mild Frost', 'Moderate Frost', 'Severe Frost',
                  'Mild Heat', 'Moderate Heat', 'Severe Heat']

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.grid': True, 'grid.alpha': 0.35, 'grid.linestyle': '--'
})

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Trend Analysis Line Chart (Yield Loss by Severity Band)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Plot 1 — Trend Analysis …")

fig1, ax1 = plt.subplots(figsize=(11, 6))

for crop in df['crop'].unique():
    crop_df = df[df['crop'] == crop]
    severity_means = (crop_df.groupby('stress_severity')['yield_loss_pct']
                      .mean()
                      .reindex(severity_order)
                      .dropna())
    if len(severity_means) > 1:
        ax1.plot(range(len(severity_means)), severity_means.values,
                 marker='o', linewidth=2, markersize=7,
                 label=crop, color=crop_colors.get(crop, 'gray'))

present_severities = [s for s in severity_order
                      if s in df['stress_severity'].unique()]
ax1.set_xticks(range(len(present_severities)))
ax1.set_xticklabels(present_severities, rotation=30, ha='right')
ax1.set_title('Trend of Mean Yield Loss Across Temperature Stress\n'
              'Severity Bands by Crop Species')
ax1.set_xlabel('Temperature Stress Severity Category')
ax1.set_ylabel('Mean Yield Loss [%]')
ax1.legend(loc='upper left', framealpha=0.9)
plt.tight_layout()
out1 = os.path.join(PLOTS_DIR, 'plot1_trend_analysis.png')
fig1.savefig(out1, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"  ✓  Saved → {out1}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 — Categorical Comparison Bar Chart (Crop × Stress Type)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Plot 2 — Categorical Comparison …")

pivot_mean = df.pivot_table(values='yield_loss_pct', index='crop',
                            columns='stress_type', aggfunc='mean')
pivot_std  = df.pivot_table(values='yield_loss_pct', index='crop',
                            columns='stress_type', aggfunc='std')

crops_list = pivot_mean.index.tolist()
x = np.arange(len(crops_list))
width = 0.35

fig2, ax2 = plt.subplots(figsize=(11, 6))

if 'Frost' in pivot_mean.columns:
    frost_vals = pivot_mean['Frost'].fillna(0).values
    frost_err  = pivot_std['Frost'].fillna(0).values
    ax2.bar(x - width/2, frost_vals, width, yerr=frost_err,
           label='Frost Stress', color='#5dade2', capsize=4,
           edgecolor='white', linewidth=0.7)

if 'Heat' in pivot_mean.columns:
    heat_vals = pivot_mean['Heat'].fillna(0).values
    heat_err  = pivot_std['Heat'].fillna(0).values
    ax2.bar(x + width/2, heat_vals, width, yerr=heat_err,
           label='Heat Stress', color='#e74c3c', capsize=4,
           edgecolor='white', linewidth=0.7)

ax2.set_title('Mean Yield Loss by Crop Species and Temperature\n'
              'Stress Type (Frost vs Heat) with Standard Deviation')
ax2.set_xlabel('Crop Species')
ax2.set_ylabel('Mean Yield Loss [%]')
ax2.set_xticks(x)
ax2.set_xticklabels(crops_list)
ax2.legend(framealpha=0.9)
plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, 'plot2_categorical_comparison.png')
fig2.savefig(out2, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"  ✓  Saved → {out2}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Correlation Plot: Frost Temperature vs Yield Loss
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Plot 3 — Correlation Plot (Frost) …")

frost_df = df[df['stress_type'] == 'Frost'].copy()

fig3, ax3 = plt.subplots(figsize=(9, 7))

for crop in frost_df['crop'].unique():
    cdf = frost_df[frost_df['crop'] == crop]
    ax3.scatter(cdf['temperature_C'], cdf['yield_loss_pct'],
               label=crop, color=crop_colors.get(crop, 'gray'),
               s=60, alpha=0.75, edgecolors='black', linewidth=0.4)

x_line = np.linspace(frost_df['temperature_C'].min(),
                     frost_df['temperature_C'].max(), 100)
y_line = reg['frost']['slope'] * x_line + reg['frost']['intercept']
ax3.plot(x_line, y_line, 'k--', linewidth=2.2,
         label=f"Trend: y={reg['frost']['slope']:.2f}x+{reg['frost']['intercept']:.2f}\n"
               f"R²={reg['frost']['r_squared']:.3f}, p={reg['frost']['p_value']:.4f}")

ax3.set_title('Correlation: Frost Temperature vs Yield Loss\n'
              'Across 5 Crop Species (with SciPy Linear Regression)')
ax3.set_xlabel('Temperature [°C]  (Sub-Zero Frost Exposure)')
ax3.set_ylabel('Yield Loss [%]')
ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
plt.tight_layout()
out3 = os.path.join(PLOTS_DIR, 'plot3_correlation.png')
fig3.savefig(out3, dpi=300, bbox_inches='tight')
plt.close(fig3)
print(f"  ✓  Saved → {out3}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 — Bonus: Heat Stress Correlation Plot
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Plot 4 — Heat Stress Correlation …")

heat_df = df[df['stress_type'] == 'Heat'].copy()

fig4, ax4 = plt.subplots(figsize=(9, 7))

for crop in heat_df['crop'].unique():
    cdf = heat_df[heat_df['crop'] == crop]
    ax4.scatter(cdf['temperature_C'], cdf['yield_loss_pct'],
               label=crop, color=crop_colors.get(crop, 'gray'),
               s=60, alpha=0.75, edgecolors='black', linewidth=0.4)

x_line_h = np.linspace(heat_df['temperature_C'].min(),
                       heat_df['temperature_C'].max(), 100)
y_line_h = reg['heat']['slope'] * x_line_h + reg['heat']['intercept']
ax4.plot(x_line_h, y_line_h, 'k--', linewidth=2.2,
         label=f"Trend: y={reg['heat']['slope']:.2f}x+{reg['heat']['intercept']:.2f}\n"
               f"R²={reg['heat']['r_squared']:.3f}, p={reg['heat']['p_value']:.4f}")

ax4.set_title('Correlation: Heat Stress Temperature vs Yield Loss\n'
              'Across 5 Crop Species (with SciPy Linear Regression)')
ax4.set_xlabel('Temperature [°C]  (High-Temperature Exposure)')
ax4.set_ylabel('Yield Loss [%]')
ax4.legend(loc='upper left', fontsize=8, framealpha=0.9)
plt.tight_layout()
out4 = os.path.join(PLOTS_DIR, 'plot4_heat_correlation.png')
fig4.savefig(out4, dpi=300, bbox_inches='tight')
plt.close(fig4)
print(f"  ✓  Saved → {out4}")

print("\n  ✓  All plots generated.  Script 03 complete.\n")
