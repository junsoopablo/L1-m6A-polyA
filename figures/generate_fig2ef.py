#!/usr/bin/env python3
"""
Figure 2 panels e–f: Arsenite mechanism panels.
  (e) CHX rescue — poly(A) recovery under translation inhibition
  (f) XRN1 KD effect — pathway convergence (paired FC bars)

Old panel (e) PAS × stress moved to Supplementary.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scipy import stats
from fig_style import *

setup_style()

OUTDIR = os.path.dirname(os.path.abspath(__file__))
BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
XRN1 = '/vault/external-datasets/2026/PRJNA842344_HeLA_under_oxidative-stress_RNA002/xrn1_analysis/analysis'

# ═══════════════════════════════════════════
# Panel (e): CHX rescue — poly(A) recovery
# ═══════════════════════════════════════════
fig_e, ax = plt.subplots(figsize=(HALF_WIDTH * 0.85, HALF_WIDTH * 0.85))
panel_label(ax, 'e')

df_chx = pd.read_csv(f'{XRN1}/ars_chx_analysis.tsv', sep='\t')
df_chx = df_chx.dropna(subset=['polya_length'])

# Use PRJNA842344 conditions for internal consistency
cond_map = {
    'mock': ('Mock', C_NORMAL),
    'Ars_mock': ('Arsenite', C_STRESS),
    'Ars+CHX': ('Ars+CHX', C_HIGHLIGHT),
}

chx_data = []
chx_colors = []
chx_labels = []

for cond, (label, color) in cond_map.items():
    sub = df_chx[df_chx['condition'] == cond]
    vals = np.clip(sub['polya_length'].values, 0, 300)
    chx_data.append(vals)
    chx_colors.append(color)
    chx_labels.append(label)

chx_positions = [0, 1, 2]

for pos, data, color in zip(chx_positions, chx_data, chx_colors):
    if len(data) > 10:
        vp = ax.violinplot([data], positions=[pos], showmedians=False,
                           showextrema=False, widths=0.6)
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.3)
            body.set_linewidth(0.4)

add_strip(ax, chx_data, chx_positions, colors=chx_colors, size=1.5, alpha=0.15, jitter=0.12)

meds = []
for pos, data in zip(chx_positions, chx_data):
    m = median_line(ax, data, pos, width=0.15, lw=1.2)
    meds.append(m)

ax.set_xticks(chx_positions)
ax.set_xticklabels(chx_labels, fontsize=7)
ax.set_ylabel('Poly(A) length (nt)')
ax.set_ylim(0, 280)

# Delta: Ars vs Mock
delta_ars = meds[1] - meds[0]
_, p_ars = stats.mannwhitneyu(chx_data[0], chx_data[1], alternative='two-sided')
significance_bracket(ax, 0, 1, 240, 5,
                     f'$\\Delta$={delta_ars:.0f} {significance_text(p_ars)}',
                     fontsize=6.5)

# Delta: Ars+CHX vs Mock
delta_chx = meds[2] - meds[0]
_, p_chx = stats.mannwhitneyu(chx_data[0], chx_data[2], alternative='two-sided')
significance_bracket(ax, 0, 2, 258, 5,
                     f'$\\Delta$={delta_chx:+.0f} {significance_text(p_chx)}',
                     fontsize=6.5)

# n annotations
for i, d in enumerate(chx_data):
    ax.text(i, -15, f'n={len(d):,}', ha='center', fontsize=5.5, color='#888888',
            clip_on=False)

save_figure(fig_e, f'{OUTDIR}/fig2e')
print(f"fig2e: mock={len(chx_data[0])}, Ars={len(chx_data[1])}, "
      f"Ars+CHX={len(chx_data[2])}")
print(f"  Medians: mock={meds[0]:.1f}, Ars={meds[1]:.1f}, Ars+CHX={meds[2]:.1f}")
print(f"  Ars delta={delta_ars:.1f} (p={p_ars:.2e}), CHX delta={delta_chx:.1f} (p={p_chx:.2e})")

# ═══════════════════════════════════════════
# Panel (f): XRN1 KD effect — pathway convergence
# Show fold-change of XRN1 KD in Normal vs Arsenite context
# If both use the same pathway, XRN1 KD effect vanishes under Ars
# ═══════════════════════════════════════════
fig_f, ax = plt.subplots(figsize=(HALF_WIDTH * 0.85, HALF_WIDTH * 0.85))
panel_label(ax, 'f')

df_xrn = pd.read_csv(f'{XRN1}/xrn1_l1_expression.tsv', sep='\t')

# Extract counts for fold-change and chi-squared test
def get_row(cond):
    row = df_xrn[df_xrn['condition'] == cond].iloc[0]
    return int(row['l1_reads']), int(row['total_mapped'])

n_mock, t_mock = get_row('mock')
n_xrn1, t_xrn1 = get_row('XRN1')
n_ars, t_ars = get_row('Ars')
n_arsxrn, t_arsxrn = get_row('Ars_XRN1')

# Fold-change (RPM ratio) in each context
fc_normal = (n_xrn1 / t_xrn1) / (n_mock / t_mock)    # XRN1 KD / Mock
fc_ars = (n_arsxrn / t_arsxrn) / (n_ars / t_ars)      # Ars+XRN1 / Ars

# Chi-squared test for each context
_, p_normal, _, _ = stats.chi2_contingency([
    [n_mock, t_mock - n_mock],
    [n_xrn1, t_xrn1 - n_xrn1]
])
_, p_ars_ctx, _, _ = stats.chi2_contingency([
    [n_ars, t_ars - n_ars],
    [n_arsxrn, t_arsxrn - n_arsxrn]
])

# 95% CI for RPM fold-change via log-rate-ratio SE (Poisson assumption)
# Var(log(FC_RPM)) ≈ 1/n_ctrl + 1/n_treat (total mapped >> L1 counts)
def fc_ci(fc_rpm, n_ctrl, n_treat):
    se = np.sqrt(1/n_ctrl + 1/n_treat)
    log_fc = np.log(fc_rpm)
    return np.exp(log_fc - 1.96 * se), np.exp(log_fc + 1.96 * se)

ci_normal = fc_ci(fc_normal, n_mock, n_xrn1)
ci_ars_ctx = fc_ci(fc_ars, n_ars, n_arsxrn)

# Forest plot: horizontal point + CI whisker
bar_colors_f = [C_NORMAL, C_STRESS]
fcs = [fc_normal, fc_ars]
ci_bounds_lo = [ci_normal[0], ci_ars_ctx[0]]
ci_bounds_hi = [ci_normal[1], ci_ars_ctx[1]]

forest_plot(ax, ['Normal', 'Arsenite'], fcs, ci_bounds_lo, ci_bounds_hi,
            colors=bar_colors_f, ref_line=1.0, horizontal=True, marker_size=60)

ax.set_xlabel('XRN1 KD effect\n(fold-change in L1 proportion)')
ax.set_xlim(0.85, 1.90)
ax.invert_yaxis()

# FC value + significance labels
for i, (fc, p, c) in enumerate(zip(fcs, [p_normal, p_ars_ctx], bar_colors_f)):
    sig = significance_text(p)
    ax.text(ci_bounds_hi[i] + 0.03, i, f'{fc:.2f}x {sig}',
            ha='left', va='center', fontsize=7, fontweight='bold', color=c)

save_figure(fig_f, f'{OUTDIR}/fig2f')
print(f"\nfig2f (XRN1 KD paired FC):")
print(f"  Normal context: XRN1 KD / Mock = {fc_normal:.3f}x "
      f"(95% CI {ci_normal[0]:.2f}\u2013{ci_normal[1]:.2f}, P={p_normal:.2e})")
print(f"  Ars context: Ars+XRN1 / Ars = {fc_ars:.3f}x "
      f"(95% CI {ci_ars_ctx[0]:.2f}\u2013{ci_ars_ctx[1]:.2f}, P={p_ars_ctx:.2e})")
print(f"  Reads: mock={n_mock}, XRN1={n_xrn1}, Ars={n_ars}, Ars+XRN1={n_arsxrn}")

print("\nAll Fig 2 e-f panels saved.")
