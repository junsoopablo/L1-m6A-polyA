#!/usr/bin/env python3
"""
Figure 3: m6A dose-dependently protects L1 poly(A) under stress.
Generates individual panel PDFs: fig3a.pdf ... fig3d.pdf
  (a) ECDF poly(A) by m6A quartile (8 curves: Q1-Q4 x HeLa/Ars)
  (b) Δpoly(A) bar chart by m6A quartile (dose-response)
  (c) Per-read scatter m6A/kb vs poly(A) — HeLa vs Ars (side-by-side)
  (d) Slope chart: Normal→Stress Pearson r by age subgroup
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import glob
from scipy import stats
from fig_style import *

setup_style()

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
RESULTS = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──
df_strat = pd.read_csv(f'{BASE}/topic_05_cellline/m6a_polya_stratified/m6a_polya_hela_stratified.tsv', sep='\t')

# ── Load per-read data for scatter ──
raw_frames = []
for f in sorted(glob.glob(f'{RESULTS}/*/g_summary/*_L1_summary.tsv')):
    group = os.path.basename(f).replace('_L1_summary.tsv', '')
    cl = group.rsplit('_', 1)[0]
    if cl in ['HeLa', 'HeLa-Ars']:
        tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'polya_length', 'qc_tag', 'gene_id', 'read_length'])
        tmp['cell_line'] = cl
        raw_frames.append(tmp)
df_l1 = pd.concat(raw_frames, ignore_index=True)
df_l1 = df_l1[df_l1['qc_tag'] == 'PASS'].copy()
df_l1['is_young'] = df_l1['gene_id'].str.match(r'^L1HS$|^L1PA[1-3]$')

cache_frames = []
for f in sorted(glob.glob(f'{BASE}/topic_05_cellline/part3_l1_per_read_cache/*_l1_per_read.tsv')):
    group = os.path.basename(f).split('_l1_per_read')[0]
    cl = group.rsplit('_', 1)[0]
    if cl in ['HeLa', 'HeLa-Ars']:
        tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
        cache_frames.append(tmp)
df_cache = pd.concat(cache_frames, ignore_index=True)
df_cache['m6a_per_kb'] = df_cache['m6a_sites_high'] / (df_cache['read_length'] / 1000)

df_merged = df_l1.merge(df_cache[['read_id', 'm6a_per_kb']], on='read_id', how='inner')

q_bounds = df_merged['m6a_per_kb'].quantile([0.25, 0.50, 0.75]).values
def assign_quartile(x):
    if x <= q_bounds[0]: return 'Q1'
    elif x <= q_bounds[1]: return 'Q2'
    elif x <= q_bounds[2]: return 'Q3'
    else: return 'Q4'
df_merged['m6a_quartile'] = df_merged['m6a_per_kb'].apply(assign_quartile)


# ═══════════════════════════════════════════
# Panel (c): Per-read scatter — m6A/kb vs poly(A), HeLa vs Ars
# ═══════════════════════════════════════════
fig_a, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.65, HALF_WIDTH * 0.85),
                            sharey=True)
ax_a1, ax_a2 = axes
panel_label(ax_a1, 'a', x=-0.22)

xlim_s = (0, 20)
ylim_s = (0, 300)

for ax, cond, color, title in [
    (ax_a1, 'HeLa', C_NORMAL, 'HeLa (normal)'),
    (ax_a2, 'HeLa-Ars', C_STRESS, 'HeLa-Ars (stress)')
]:
    sub = df_merged[df_merged['cell_line'] == cond].copy()
    x = sub['m6a_per_kb'].values
    y = sub['polya_length'].values

    ax.scatter(x, y, s=S_POINT_SMALL, alpha=0.15, color=color, edgecolors='none',
               rasterized=True, zorder=1)

    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x_f, y_f = x[mask], y[mask]
    if len(x_f) > 50:
        slope, intercept, r, p, se = stats.linregress(x_f, y_f)
        x_line = np.linspace(xlim_s[0], xlim_s[1], 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='black', lw=1.0, zorder=3)

        n = len(x_f)
        x_mean = x_f.mean()
        ss_x = np.sum((x_f - x_mean)**2)
        se_fit = np.sqrt(np.sum((y_f - (slope * x_f + intercept))**2) / (n - 2)) * \
                 np.sqrt(1/n + (x_line - x_mean)**2 / ss_x)
        from scipy.stats import t as t_dist
        t_crit = t_dist.ppf(0.975, n - 2)
        ax.fill_between(x_line, y_line - t_crit * se_fit, y_line + t_crit * se_fit,
                         alpha=0.15, color=color, lw=0, zorder=2)

        ax.text(0.97, 0.06, f'r = {r:.3f}\nP = {p:.1e}\nn = {len(x_f):,}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=FS_ANNOT_SMALL, color=C_TEXT,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlim(xlim_s)
    ax.set_ylim(ylim_s)
    ax.set_xlabel('m6A sites per kb')
    ax.set_title(title, pad=4)

ax_a1.set_ylabel('Poly(A) length (nt)')
ax_a2.set_ylabel('')
ax_a2.tick_params(labelleft=False)
fig_a.tight_layout()
save_figure(fig_a, f'{OUTDIR}/figS14a')
n_hela = len(df_merged[df_merged['cell_line'] == 'HeLa'])
n_ars = len(df_merged[df_merged['cell_line'] == 'HeLa-Ars'])
print(f"figS14a (scatter): {n_hela:,} HeLa + {n_ars:,} Ars reads")


# ═══════════════════════════════════════════
# Panel (b): Δpoly(A) bar chart by m6A quartile (dose-response)
# ═══════════════════════════════════════════
fig_b, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))
panel_label(ax, 'b')

quartile_order_b = ['Q1', 'Q2', 'Q3', 'Q4']
q_cmap_stress_b = ['#E8B5C5', '#CC7799', '#AA3366', '#771144']

deltas, ns_b, p_vals_b = [], [], []
for q in quartile_order_b:
    normal = df_merged[(df_merged['cell_line'] == 'HeLa') &
                       (df_merged['m6a_quartile'] == q)]['polya_length'].dropna()
    stress = df_merged[(df_merged['cell_line'] == 'HeLa-Ars') &
                       (df_merged['m6a_quartile'] == q)]['polya_length'].dropna()
    delta = stress.median() - normal.median()
    _, p = stats.mannwhitneyu(normal, stress, alternative='two-sided')
    deltas.append(delta)
    ns_b.append((len(normal), len(stress)))
    p_vals_b.append(p)

x_pos = np.arange(4)
# Lollipop chart (vertical, ref at 0)
lollipop_plot(ax, ['Q1\n(low m6A)', 'Q2', 'Q3', 'Q4\n(high m6A)'], deltas,
              colors=q_cmap_stress_b, horizontal=False, marker_size=50,
              stem_width=2.0, ref_value=0)
ax.axhline(0, color=C_GREY, lw=0.5, ls='--')

# Delta labels + significance stars above/below each bar
for i, (d, p) in enumerate(zip(deltas, p_vals_b)):
    sig = significance_text(p)
    y_txt = d - 2.5 if d < 0 else d + 1.5
    ax.text(i, y_txt, f'{d:.0f} nt\n{sig}', ha='center', va='top' if d < 0 else 'bottom',
            fontsize=FS_ANNOT_SMALL, color=C_TEXT)

# Sample size below bars
y_bottom = min(deltas) - 12
for i, (n_n, n_s) in enumerate(ns_b):
    ax.text(i, y_bottom, f'n={n_n+n_s:,}', ha='center', va='top',
            fontsize=FS_ANNOT_SMALL, color='#999999')

# Per-read Spearman trend: quartile rank (1-4) vs poly(A) within stress
df_stress_q = df_merged[df_merged['cell_line'] == 'HeLa-Ars'].copy()
q_rank_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
df_stress_q['q_rank'] = df_stress_q['m6a_quartile'].map(q_rank_map)
rho_trend, p_trend = stats.spearmanr(df_stress_q['q_rank'], df_stress_q['polya_length'])
if p_trend < 1e-4:
    exp_t = int(np.floor(np.log10(p_trend)))
    mant_t = p_trend / 10**exp_t
    p_trend_str = rf'{mant_t:.1f}$\times$10$^{{{exp_t}}}$'
else:
    p_trend_str = f'{p_trend:.2g}'
ax.text(0.97, 0.97,
        f'Trend (per-read): $\\rho$ = {rho_trend:.3f}\n$P$ = {p_trend_str}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=FS_ANNOT_SMALL, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

ax.set_ylabel('$\\Delta$ poly(A) length (nt)')
ax.set_xlabel('m6A quartile')
ax.set_ylim(y_bottom - 5, max(0, max(deltas)) + 15)

save_figure(fig_b, f'{OUTDIR}/figS14b')
print(f"figS14b (Δpoly(A) bar): {[f'{d:.1f}' for d in deltas]}")


# ═══════════════════════════════════════════
# Panel (d): Slope chart — Normal→Stress Pearson r by age subgroup
# ═══════════════════════════════════════════
fig_c, ax = plt.subplots(figsize=(HALF_WIDTH * 0.95, HALF_WIDTH * 0.65))
panel_label(ax, 'b')

# Extract r and p values from df_strat
slope_groups = {
    'Young':   ('normal_young',      'stress_young'),
    'Ancient': ('normal_ancient',    'stress_ancient'),
    'All':     ('condition=normal',  'condition=stress'),
}
group_labels = ['Young', 'Ancient', 'All']
r_normal_d, r_stress_d, p_normal_d, p_stress_d, n_normal_d, n_stress_d = [], [], [], [], [], []
for g in group_labels:
    k_n, k_s = slope_groups[g]
    row_n = df_strat[df_strat['label'] == k_n].iloc[0]
    row_s = df_strat[df_strat['label'] == k_s].iloc[0]
    r_normal_d.append(row_n['r'])
    r_stress_d.append(row_s['r'])
    p_normal_d.append(row_n['p'])
    p_stress_d.append(row_s['p'])
    n_normal_d.append(int(row_n['n']))
    n_stress_d.append(int(row_s['n']))

y_pos = np.arange(len(group_labels))

# Connecting lines (Normal → Stress)
for i in range(len(group_labels)):
    lw = 1.8 if group_labels[i] == 'Ancient' else 1.0
    ax.plot([r_normal_d[i], r_stress_d[i]], [i, i],
            color='#AAAAAA', lw=lw, zorder=1)

# Normal dots (open circles)
ax.scatter(r_normal_d, y_pos, s=S_POINT_LARGE, facecolors='white',
           edgecolors=C_NORMAL, linewidths=1.2, zorder=3, label='Normal')

# Stress dots (filled circles)
ax.scatter(r_stress_d, y_pos, s=S_POINT_LARGE, facecolors=C_STRESS,
           edgecolors=C_STRESS, linewidths=0.5, zorder=3, label='Stress')

# Annotations: r value + significance next to each dot
for i, g in enumerate(group_labels):
    sig_n = significance_text(p_normal_d[i])
    sig_s = significance_text(p_stress_d[i])
    # Normal annotation (left side)
    ax.text(r_normal_d[i] - 0.008, i + 0.22,
            f'r={r_normal_d[i]:.2f} {sig_n}',
            ha='right', va='bottom', fontsize=FS_ANNOT_SMALL, color=C_NORMAL)
    # Stress annotation (right side)
    ax.text(r_stress_d[i] + 0.008, i + 0.22,
            f'r={r_stress_d[i]:.2f} {sig_s}',
            ha='left', va='bottom', fontsize=FS_ANNOT_SMALL, color=C_STRESS)

# Bold the Ancient row label
ax.set_yticks(y_pos)
ytick_labels = ax.set_yticklabels(group_labels)
ytick_labels[1].set_fontweight('bold')

ax.axvline(0, color='#DDDDDD', lw=0.5, zorder=0)
ax.set_xlabel('Pearson r (m6A/kb vs poly(A))')
ax.set_xlim(-0.05, 0.28)
ax.set_ylim(-0.5, len(group_labels) - 0.5)
ax.invert_yaxis()

ax.legend(fontsize=FS_LEGEND, loc='upper right',
          frameon=False)

save_figure(fig_c, f'{OUTDIR}/fig3b')
print(f"fig3b (slope chart): Normal→Stress r for {len(group_labels)} subgroups")


# ═══════════════════════════════════════════
# Panel (a): ECDF poly(A) by m6A quartile — 8 curves
# ═══════════════════════════════════════════
fig_d, ax = plt.subplots(figsize=(HALF_WIDTH * 1.2, HALF_WIDTH * 0.85))
panel_label(ax, 'a')

from matplotlib.lines import Line2D

q_cmap_normal = ['#B8D4E8', '#7EB1D4', '#4488BA', '#2A5F8F']
q_cmap_stress = ['#E8B5C5', '#CC7799', '#AA3366', '#771144']

quartile_order = ['Q1', 'Q2', 'Q3', 'Q4']
xlim_ecdf = 300

for qi, q in enumerate(quartile_order):
    d_n = df_merged[(df_merged['cell_line'] == 'HeLa') &
                     (df_merged['m6a_quartile'] == q)]['polya_length'].dropna().values
    if len(d_n) > 10:
        ecdf_plot(ax, np.clip(d_n, 0, xlim_ecdf), q_cmap_normal[qi],
                  f'{q} normal (n={len(d_n):,})', lw=1.0, ls='--')

    d_s = df_merged[(df_merged['cell_line'] == 'HeLa-Ars') &
                     (df_merged['m6a_quartile'] == q)]['polya_length'].dropna().values
    if len(d_s) > 10:
        ecdf_plot(ax, np.clip(d_s, 0, xlim_ecdf), q_cmap_stress[qi],
                  f'{q} stress (n={len(d_s):,})', lw=1.2)

ax.axvline(30, color='#CCCCCC', lw=0.7, ls=':')
ax.text(32, 0.02, 'Decay zone\n(<30 nt)', fontsize=FS_ANNOT_SMALL, color='#999999', va='bottom')

ax.set_xlim(0, xlim_ecdf)
ax.set_ylim(0, 1.02)
ax.set_xlabel('Poly(A) length (nt)')
ax.set_ylabel('Cumulative fraction')

legend_lines = [
    Line2D([0], [0], color=q_cmap_stress[0], lw=1.2, label='Q1 (low m6A) arsenite'),
    Line2D([0], [0], color=q_cmap_stress[3], lw=1.2, label='Q4 (high m6A) arsenite'),
    Line2D([0], [0], color=q_cmap_normal[0], lw=1.0, ls='--', label='Q1 normal'),
    Line2D([0], [0], color=q_cmap_normal[3], lw=1.0, ls='--', label='Q4 normal'),
]
ax.legend(handles=legend_lines, fontsize=FS_ANNOT_SMALL, loc='lower right', frameon=False)

q1_stress = df_merged[(df_merged['cell_line'] == 'HeLa-Ars') &
                       (df_merged['m6a_quartile'] == 'Q1')]['polya_length'].dropna()
q4_stress = df_merged[(df_merged['cell_line'] == 'HeLa-Ars') &
                       (df_merged['m6a_quartile'] == 'Q4')]['polya_length'].dropna()
if len(q1_stress) > 10 and len(q4_stress) > 10:
    med_q1 = q1_stress.median()
    med_q4 = q4_stress.median()
    ax.text(0.03, 0.95, f'Q1 median {med_q1:.0f} nt  •  Q4 median {med_q4:.0f} nt\n'
                           f'$\\Delta$ = +{med_q4-med_q1:.0f} nt',
              transform=ax.transAxes, va='top', fontsize=FS_ANNOT, color=C_STRESS)

save_figure(fig_d, f'{OUTDIR}/fig3a')
print(f"fig3a: {len(df_merged):,} reads across 8 curves")

print("\nAll Fig 3 panels saved.")
