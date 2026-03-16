#!/usr/bin/env python3
"""
Supplementary Figure S6: Regulatory L1 destabilization under stress.
4 panels: (a) ECDF regulatory vs non-regulatory, (b) decay zone by chromatin,
(c) m6A-matched regulatory vs non-reg, (d) within-regulatory m6A quartile ECDF.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scipy import stats
from fig_style import *

setup_style()

OUTDIR = os.path.dirname(os.path.abspath(__file__))
CHROMHMM = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv'

# ── Load data ──
df = pd.read_csv(CHROMHMM, sep='\t')
# Filter: ancient only, HeLa + HeLa-Ars
df = df[(df['l1_age'] == 'ancient') & (df['cellline'].isin(['HeLa', 'HeLa-Ars']))].copy()
df['is_stress'] = df['condition'] == 'stress'
df['is_regulatory'] = df['chromhmm_group'].isin(['Enhancer', 'Promoter'])

print(f"Total ancient HeLa/Ars reads: {len(df)}")
print(f"  Normal: {(~df['is_stress']).sum()}, Stress: {df['is_stress'].sum()}")
print(f"  Regulatory: {df['is_regulatory'].sum()}, Non-reg: {(~df['is_regulatory']).sum()}")

# ── Create figure ──
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.82))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.45,
                      left=0.10, right=0.97, top=0.96, bottom=0.08)

# ── Color definitions ──
C_REG_NORM = '#4477AA'   # steel blue — regulatory normal
C_REG_STRESS = '#CC6677' # rose — regulatory stress
C_NR_NORM = '#88CCEE'    # light cyan — non-reg normal
C_NR_STRESS = '#DDCC77'  # sand — non-reg stress

# State colors for panel (b)
STATE_COLORS = {
    'Enhancer': '#CC6677',
    'Promoter': '#882255',
    'Transcribed': '#DDCC77',
    'Quiescent': '#88CCEE',
}

# ────────────────────────────────────────────
# Panel (a): ECDF — Regulatory vs Non-regulatory, HeLa vs Ars
# ────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'a')

reg_norm = df[(df['is_regulatory']) & (~df['is_stress'])]['polya_length'].dropna().values
reg_str = df[(df['is_regulatory']) & (df['is_stress'])]['polya_length'].dropna().values
nr_norm = df[(~df['is_regulatory']) & (~df['is_stress'])]['polya_length'].dropna().values
nr_str = df[(~df['is_regulatory']) & (df['is_stress'])]['polya_length'].dropna().values

# Clip for plotting
xmax = 300
for data, color, label, lw, ls in [
    (nr_norm, C_NR_NORM, f'Non-regulatory, HeLa (n={len(nr_norm):,})', 1.0, '--'),
    (nr_str, C_NR_STRESS, f'Non-regulatory, Arsenite (n={len(nr_str):,})', 1.0, '-'),
    (reg_norm, C_REG_NORM, f'Regulatory, HeLa (n={len(reg_norm):,})', 1.2, '--'),
    (reg_str, C_REG_STRESS, f'Regulatory, Arsenite (n={len(reg_str):,})', 1.2, '-'),
]:
    clipped = np.clip(data, 0, xmax)
    ecdf_plot(ax_a, clipped, color, label, lw=lw, ls=ls)

# Median lines
for data, color, ls in [(reg_norm, C_REG_NORM, '--'), (reg_str, C_REG_STRESS, '-')]:
    med = np.median(data)
    ax_a.axvline(med, color=color, ls=':', lw=0.7, alpha=0.6)
    ax_a.text(med, 0.02, f'{med:.0f}', fontsize=FS_ANNOT_SMALL, color=color, ha='center')

# Delta annotation
med_rn = np.median(reg_norm)
med_rs = np.median(reg_str)
ax_a.annotate('', xy=(med_rs, 0.52), xytext=(med_rn, 0.52),
              arrowprops=dict(arrowstyle='->', color=C_REG_STRESS, lw=1.2))
ax_a.text((med_rn + med_rs) / 2, 0.55, f'Δ = {med_rs - med_rn:.0f} nt',
          ha='center', fontsize=FS_ANNOT, fontweight='bold', color=C_REG_STRESS)

ax_a.set_xlabel('Poly(A) tail length (nt)')
ax_a.set_ylabel('Cumulative fraction')
ax_a.set_xlim(0, xmax)
ax_a.set_ylim(0, 1.02)
ax_a.legend(fontsize=5.5, loc='lower right', handlelength=1.8)

# ────────────────────────────────────────────
# Panel (b): Decay zone fraction by chromatin state
# ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'b')

states_order = ['Enhancer', 'Promoter', 'Transcribed', 'Quiescent']
state_labels = ['Enhancer', 'Promoter', 'Transcribed', 'Quiescent']

decay_thresh = 30
x_pos = np.arange(len(states_order))
bar_w = 0.35

dz_hela = []
dz_ars = []
n_hela_list = []
n_ars_list = []

for st in states_order:
    sub_h = df[(df['chromhmm_group'] == st) & (~df['is_stress'])]['polya_length'].dropna()
    sub_a = df[(df['chromhmm_group'] == st) & (df['is_stress'])]['polya_length'].dropna()
    n_h = len(sub_h)
    n_a = len(sub_a)
    dz_h = (sub_h < decay_thresh).sum() / n_h * 100 if n_h > 0 else 0
    dz_a = (sub_a < decay_thresh).sum() / n_a * 100 if n_a > 0 else 0
    dz_hela.append(dz_h)
    dz_ars.append(dz_a)
    n_hela_list.append(n_h)
    n_ars_list.append(n_a)

# Slope chart: HeLa → Arsenite connected dots for each chromatin state
for i, st in enumerate(states_order):
    ax_b.plot([0, 1], [dz_hela[i], dz_ars[i]], color=STATE_COLORS[st],
              lw=1.5, zorder=2, alpha=0.8)
    ax_b.scatter(0, dz_hela[i], s=40, color=STATE_COLORS[st],
                 edgecolors='white', linewidths=0.5, zorder=3)
    ax_b.scatter(1, dz_ars[i], s=40, color=STATE_COLORS[st],
                 edgecolors='white', linewidths=0.5, zorder=3)
    # Label on right side
    ax_b.text(1.08, dz_ars[i], f'{st} {dz_ars[i]:.0f}%',
              fontsize=FS_ANNOT_SMALL, va='center', color=STATE_COLORS[st],
              fontweight='bold')

ax_b.set_xticks([0, 1])
ax_b.set_xticklabels(['HeLa', 'Arsenite'], fontsize=FS_ANNOT)
ax_b.set_ylabel('Reads in decay zone (<30 nt) (%)')
ax_b.set_ylim(0, 55)
ax_b.set_xlim(-0.2, 1.8)
ax_b.axhline(y=0, color='grey', lw=0.3)

# ────────────────────────────────────────────
# Panel (c): m6A-matched: regulatory vs non-reg poly(A) under stress
# ────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
panel_label(ax_c, 'c')

# Stress reads only
df_stress = df[df['is_stress']].copy()
df_stress = df_stress.dropna(subset=['m6a_per_kb', 'polya_length'])

# Compute m6A quartiles across all stress reads
df_stress['m6a_q'] = pd.qcut(df_stress['m6a_per_kb'], 4, labels=['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)'])

q_labels = ['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)']
x_q = np.arange(len(q_labels))

reg_medians = []
nr_medians = []
reg_ns = []
nr_ns = []
p_vals = []

for q in q_labels:
    reg = df_stress[(df_stress['is_regulatory']) & (df_stress['m6a_q'] == q)]['polya_length']
    nr = df_stress[(~df_stress['is_regulatory']) & (df_stress['m6a_q'] == q)]['polya_length']
    reg_medians.append(reg.median() if len(reg) > 0 else np.nan)
    nr_medians.append(nr.median() if len(nr) > 0 else np.nan)
    reg_ns.append(len(reg))
    nr_ns.append(len(nr))
    if len(reg) >= 5 and len(nr) >= 5:
        _, p = stats.mannwhitneyu(reg, nr, alternative='two-sided')
        p_vals.append(p)
    else:
        p_vals.append(np.nan)

# Connected paired dots
for i in range(len(q_labels)):
    ax_c.plot([x_q[i] - 0.12, x_q[i] + 0.12], [reg_medians[i], nr_medians[i]],
              color='#BDC3C7', lw=1.5, zorder=1)

ax_c.scatter(x_q - 0.12, reg_medians, s=S_POINT_LARGE, color=C_REG_STRESS, edgecolors='#2C3E50',
             linewidths=0.5, zorder=3, marker='s', label='Regulatory')
ax_c.scatter(x_q + 0.12, nr_medians, s=S_POINT_LARGE, color=C_NR_STRESS, edgecolors='#2C3E50',
             linewidths=0.5, zorder=3, marker='o', label='Non-regulatory')

# Significance stars
for i, p in enumerate(p_vals):
    if not np.isnan(p):
        star = significance_text(p)
        y_top = max(reg_medians[i], nr_medians[i]) + 8
        ax_c.text(x_q[i], y_top, star, ha='center', fontsize=FS_ANNOT, fontweight='bold',
                  color='#373737')

# Delta annotation
for i in range(len(q_labels)):
    delta = reg_medians[i] - nr_medians[i]
    if not np.isnan(delta):
        y_pos = min(reg_medians[i], nr_medians[i]) - 12
        ax_c.text(x_q[i], y_pos, f'Δ={delta:.0f} nt', ha='center', fontsize=FS_LEGEND_SMALL,
                  color=C_REG_STRESS)

ax_c.set_xticks(x_q)
ax_c.set_xticklabels(q_labels, fontsize=FS_ANNOT)
ax_c.set_xlabel('m6A/kb quartile (matched)')
ax_c.set_ylabel('Median poly(A) length (nt)')
ax_c.set_ylim(0, 165)
ax_c.legend(fontsize=FS_ANNOT, loc='upper left')
ax_c.set_title('Arsenite-stressed reads only', pad=8)

# ────────────────────────────────────────────
# Panel (d): Within regulatory L1: ECDF by m6A quartile (stress only)
# ────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
panel_label(ax_d, 'd')

df_reg_stress = df_stress[df_stress['is_regulatory']].copy()
print(f"\nRegulatory stress reads for panel (d): {len(df_reg_stress)}")

# Recompute quartiles within regulatory stress reads
if len(df_reg_stress) >= 20:
    try:
        df_reg_stress['m6a_q_inner'] = pd.qcut(df_reg_stress['m6a_per_kb'], 4,
                                                labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    except ValueError:
        # Duplicate bin edges (many zeros at high threshold) — use rank-based split
        df_reg_stress['m6a_q_inner'] = pd.qcut(df_reg_stress['m6a_per_kb'].rank(method='first'), 4,
                                                labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])

    # Gradient colors from cold (low m6A) to warm (high m6A)
    q_colors = ['#4477AA', '#88CCEE', '#DDCC77', '#CC6677']
    q_inner_labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']

    for qi, (ql, c) in enumerate(zip(q_inner_labels, q_colors)):
        data = df_reg_stress[df_reg_stress['m6a_q_inner'] == ql]['polya_length'].dropna().values
        clipped = np.clip(data, 0, xmax)
        n = len(data)
        med = np.median(data)
        lw = 1.0 if qi in [0, 3] else 0.8
        ecdf_plot(ax_d, clipped, c, f'{ql} (n={n}, med={med:.0f})', lw=lw)

    # Vertical lines at Q1 and Q4 medians
    q1_data = df_reg_stress[df_reg_stress['m6a_q_inner'] == 'Q1 (low)']['polya_length'].dropna()
    q4_data = df_reg_stress[df_reg_stress['m6a_q_inner'] == 'Q4 (high)']['polya_length'].dropna()
    q1_med = q1_data.median()
    q4_med = q4_data.median()

    ax_d.axvline(q1_med, color=q_colors[0], ls=':', lw=0.7, alpha=0.7)
    ax_d.axvline(q4_med, color=q_colors[3], ls=':', lw=0.7, alpha=0.7)

    # Delta annotation
    ax_d.annotate('', xy=(q4_med, 0.45), xytext=(q1_med, 0.45),
                  arrowprops=dict(arrowstyle='->', color='#373737', lw=1.0))
    ax_d.text((q1_med + q4_med) / 2, 0.48,
              f'Δ = +{q4_med - q1_med:.0f} nt', ha='center', fontsize=FS_ANNOT,
              fontweight='bold', color='#373737')

    # Decay zone shading
    ax_d.axvspan(0, 30, alpha=0.08, color='#CC6677', zorder=0)
    ax_d.text(15, 0.92, 'Decay\nzone', ha='center', fontsize=FS_ANNOT_SMALL,
              color='#CC6677', alpha=0.8, style='italic')

    # Decay zone percentages
    dz_q1 = (q1_data < 30).sum() / len(q1_data) * 100
    dz_q4 = (q4_data < 30).sum() / len(q4_data) * 100
    ax_d.text(0.97, 0.35, f'Q1 decay zone: {dz_q1:.0f}%\nQ4 decay zone: {dz_q4:.0f}%',
              transform=ax_d.transAxes, fontsize=FS_ANNOT_SMALL, ha='right', va='top',
              color='#373737', fontweight='bold')

ax_d.set_xlabel('Poly(A) tail length (nt)')
ax_d.set_ylabel('Cumulative fraction')
ax_d.set_xlim(0, xmax)
ax_d.set_ylim(0, 1.02)
ax_d.legend(fontsize=5.5, loc='lower right', handlelength=1.8)
ax_d.set_title('Regulatory L1 under arsenite', pad=8)

# ── Save ──
save_figure(fig, f'{OUTDIR}/figS6')
print(f"\nFig S6 saved to {OUTDIR}/figS6.pdf")

# ── Print discoveries ──
print(f"\n[DISCOVERY] Regulatory L1 ECDF: HeLa median={np.median(reg_norm):.0f}, Ars median={np.median(reg_str):.0f}, Δ={np.median(reg_str)-np.median(reg_norm):.0f} nt")
print(f"[DISCOVERY] Non-reg L1 ECDF: HeLa median={np.median(nr_norm):.0f}, Ars median={np.median(nr_str):.0f}, Δ={np.median(nr_str)-np.median(nr_norm):.0f} nt")
print(f"[DISCOVERY] Decay zone: Enhancer Ars={dz_ars[0]:.0f}%, Promoter Ars={dz_ars[1]:.0f}%, Txn Ars={dz_ars[2]:.0f}%, Quies Ars={dz_ars[3]:.0f}%")
for i, (q, rm, nrm, p) in enumerate(zip(q_labels, reg_medians, nr_medians, p_vals)):
    print(f"[DISCOVERY] m6A-matched {q}: reg={rm:.0f}, non-reg={nrm:.0f}, Δ={rm-nrm:.0f}, p={p:.2e}")
