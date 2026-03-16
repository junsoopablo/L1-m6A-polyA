#!/usr/bin/env python3
"""
Supplementary Figure S10: Exon overlap threshold validation.

(a) Histogram of exon overlaps for all stage 1 reads (PASS vs rejected)
(b) PASS rate by overlap bin — cliff at 25bp
(c) Poly(A) by overlap bin within PASS reads — no gradient
(d) Threshold sweep — reads passing at each cutoff
"""
import sys
from pathlib import Path
sys.path.insert(0, '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures')
from fig_style import *
import pandas as pd
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
VALDIR = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic/exon_overlap_validation'

setup_style()

# ── Load data ──
ov = pd.read_csv(VALDIR / 'exon_overlap_all_stage1.tsv', sep='\t')
print(f'Total stage 1 reads: {len(ov)}')
print(f'  PASS: {ov["is_pass"].sum()}, Rejected: {(~ov["is_pass"]).sum()}')

# Load poly(A) data for PASS reads
group_map = {
    'HeLa_1_1': 'HeLa_1', 'HeLa_2_1': 'HeLa_2', 'HeLa_3_1': 'HeLa_3',
    'HeLa-Ars_1_1': 'HeLa-Ars_1', 'HeLa-Ars_2_1': 'HeLa-Ars_2', 'HeLa-Ars_3_1': 'HeLa-Ars_3',
}

polya_dfs = []
for sample, group in group_map.items():
    path = PROJECT / f'results_group/{group}/g_summary/{group}_L1_summary.tsv'
    if path.exists():
        df = pd.read_csv(path, sep='\t')
        df['sample'] = sample
        df['condition'] = 'HeLa' if 'Ars' not in sample else 'HeLa-Ars'
        polya_dfs.append(df)

polya = pd.concat(polya_dfs, ignore_index=True)

ov_pass = ov[ov['is_pass']].copy()
merged = ov_pass.merge(polya[['read_id', 'polya_length']], on='read_id', how='inner')
print(f'Merged with poly(A): {len(merged)} reads')

# ── Figure: 2×2 layout ──
fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.70))
ax_a, ax_b, ax_c, ax_d = axes.flat

# =====================================================================
# Panel (a): Histogram of exon overlaps — PASS vs Rejected
# =====================================================================
bins_hist = np.arange(0, 310, 10)

pass_ov = ov[ov['is_pass']]['exon_overlap'].values
rej_ov = ov[~ov['is_pass']]['exon_overlap'].values

# Clip for display
pass_ov_clip = np.clip(pass_ov, 0, 299)
rej_ov_clip = np.clip(rej_ov, 0, 299)

ax_a.hist(rej_ov_clip, bins=bins_hist, color=C_GREY, alpha=0.6,
          label=f'Rejected (n={len(rej_ov):,})', edgecolor='none')
ax_a.hist(pass_ov_clip, bins=bins_hist, color=C_L1, alpha=0.8,
          label=f'Non-embedded (n={len(pass_ov):,})', edgecolor='none')

ax_a.axvline(100, color=C_TEXT, ls='--', lw=0.8, zorder=5)
ax_a.text(105, ax_a.get_ylim()[1] * 0.85, 'Threshold\n= 100 bp',
          fontsize=FS_LEGEND_SMALL, color=C_TEXT, va='top')

ax_a.set_xlabel('Max protein-coding exon overlap (bp)')
ax_a.set_ylabel('Number of reads')
ax_a.set_xlim(-5, 305)
ax_a.legend(loc='upper right', frameon=False, fontsize=FS_LEGEND_SMALL)
ax_a.set_yscale('log')
ax_a.set_ylim(1, None)
panel_label(ax_a, 'a')

# =====================================================================
# Panel (b): PASS rate by overlap bin
# =====================================================================
bin_edges = [0, 25, 50, 75, 100, 125, 150, 200]
bin_labels = ['0–24', '25–49', '50–74', '75–99', '100–124', '125–149', '150–199']
pass_rates = []
total_counts = []

for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (ov['exon_overlap'] >= lo) & (ov['exon_overlap'] < hi)
    n_total = mask.sum()
    n_pass = (mask & ov['is_pass']).sum()
    rate = 100 * n_pass / n_total if n_total > 0 else 0
    pass_rates.append(rate)
    total_counts.append(n_total)

x_pos = np.arange(len(bin_labels))
colors_bar = [C_L1 if i < 4 else C_GREY for i in range(len(bin_labels))]

# Lollipop chart (vertical)
lollipop_plot(ax_b, bin_labels, pass_rates, colors=colors_bar,
              horizontal=False, marker_size=40, stem_width=1.5, ref_value=0)

# Annotate rates
for i, (rate, n) in enumerate(zip(pass_rates, total_counts)):
    ax_b.text(i, rate + 1.5, f'{rate:.1f}%', ha='center', va='bottom',
              fontsize=5.5, color=C_TEXT)

ax_b.axvline(3.5, color=C_TEXT, ls='--', lw=0.7, alpha=0.5)
ax_b.text(3.7, 45, '100 bp\nthreshold', fontsize=FS_LEGEND_SMALL, color=C_TEXT, alpha=0.7)

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(bin_labels, fontsize=FS_LEGEND_SMALL, rotation=30, ha='right')
ax_b.set_xlabel('Exon overlap bin (bp)')
ax_b.set_ylabel('Non-embedded rate (%)')
ax_b.set_ylim(0, 55)
panel_label(ax_b, 'b')

# =====================================================================
# Panel (c): Poly(A) by overlap bin within PASS reads — HeLa vs Ars
# =====================================================================
# Bins: 0bp exact, 1-49bp, 50-99bp
ov_groups = [
    (0, 0, '0'),
    (1, 49, '1–49'),
    (50, 99, '50–99'),
]

positions_hela = []
positions_ars = []
data_hela = []
data_ars = []

for gi, (lo, hi, label) in enumerate(ov_groups):
    if lo == 0 and hi == 0:
        mask = merged['exon_overlap'] == 0
    else:
        mask = (merged['exon_overlap'] >= lo) & (merged['exon_overlap'] <= hi)

    hela_sub = merged[mask & (merged['condition'] == 'HeLa')]['polya_length'].dropna().values
    ars_sub = merged[mask & (merged['condition'] == 'HeLa-Ars')]['polya_length'].dropna().values

    data_hela.append(hela_sub)
    data_ars.append(ars_sub)
    positions_hela.append(gi * 3)
    positions_ars.append(gi * 3 + 1)

# Violins — HeLa
for i, (data, pos) in enumerate(zip(data_hela, positions_hela)):
    if len(data) > 10:
        vp = ax_c.violinplot([data], positions=[pos], showextrema=False, widths=0.7)
        for body in vp['bodies']:
            body.set_facecolor(C_NORMAL)
            body.set_alpha(0.3)
            body.set_edgecolor('none')
        med = median_line(ax_c, data, pos, color=C_NORMAL, width=0.2, lw=1.2)
        ax_c.text(pos, med + 8, f'{med:.0f}', ha='center', va='bottom',
                  fontsize=5.5, color=C_NORMAL)

# Violins — Ars
for i, (data, pos) in enumerate(zip(data_ars, positions_ars)):
    if len(data) > 10:
        vp = ax_c.violinplot([data], positions=[pos], showextrema=False, widths=0.7)
        for body in vp['bodies']:
            body.set_facecolor(C_STRESS)
            body.set_alpha(0.3)
            body.set_edgecolor('none')
        med = median_line(ax_c, data, pos, color=C_STRESS, width=0.2, lw=1.2)
        ax_c.text(pos, med - 12, f'{med:.0f}', ha='center', va='top',
                  fontsize=5.5, color=C_STRESS)

# Significance brackets
for i in range(3):
    hela_d = data_hela[i]
    ars_d = data_ars[i]
    if len(hela_d) >= 5 and len(ars_d) >= 5:
        _, p = stats.mannwhitneyu(hela_d, ars_d, alternative='two-sided')
        delta = np.median(ars_d) - np.median(hela_d)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        y_top = max(np.percentile(hela_d, 90), np.percentile(ars_d, 90))
        significance_bracket(ax_c, positions_hela[i], positions_ars[i],
                             y_top + 10, 5, sig, fontsize=FS_LEGEND_SMALL)

# Sample sizes
for i, (lo, hi, label) in enumerate(ov_groups):
    n_h = len(data_hela[i])
    n_a = len(data_ars[i])
    ax_c.text((positions_hela[i] + positions_ars[i]) / 2, -25,
              f'n={n_h+n_a}', ha='center', fontsize=5.5, color=C_TEXT)

ax_c.set_xticks([(positions_hela[i] + positions_ars[i]) / 2 for i in range(3)])
ax_c.set_xticklabels([g[2] for g in ov_groups])
ax_c.set_xlabel('Exon overlap (bp)')
ax_c.set_ylabel('Poly(A) length (nt)')
ax_c.set_ylim(-35, 350)

# Manual legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=C_NORMAL, alpha=0.5, label='HeLa'),
                   Patch(facecolor=C_STRESS, alpha=0.5, label='HeLa-Ars')]
ax_c.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=FS_LEGEND_SMALL)
panel_label(ax_c, 'c')

# =====================================================================
# Panel (d): Threshold sweep — line plot
# =====================================================================
thresholds = np.arange(10, 310, 5)
all_overlaps = ov['exon_overlap'].values

n_pass_arr = np.array([(all_overlaps < thr).sum() for thr in thresholds])
pct_pass = 100 * n_pass_arr / len(all_overlaps)

ax_d.plot(thresholds, pct_pass, color=C_L1, lw=1.5, zorder=3)
ax_d.fill_between(thresholds, pct_pass, alpha=0.1, color=C_L1)

# Mark current threshold
n100 = (all_overlaps < 100).sum()
pct100 = 100 * n100 / len(all_overlaps)
ax_d.plot(100, pct100, 'o', color=C_STRESS, ms=5, zorder=5)
ax_d.annotate(f'Current\n({pct100:.1f}%)',
              xy=(100, pct100), xytext=(140, pct100 - 12),
              fontsize=FS_LEGEND_SMALL, color=C_STRESS,
              arrowprops=dict(arrowstyle='->', color=C_STRESS, lw=0.7))

# Mark alternative thresholds
for thr, offset_y in [(50, -8), (150, 5)]:
    n_thr = (all_overlaps < thr).sum()
    pct_thr = 100 * n_thr / len(all_overlaps)
    ax_d.plot(thr, pct_thr, 'o', color=C_GREY, ms=3, zorder=4)
    ax_d.text(thr + 5, pct_thr + offset_y, f'{thr} bp\n({pct_thr:.1f}%)',
              fontsize=5.5, color=C_GREY)

# Dashed at PASS-only reads for context
n_pass_all_filters = ov['is_pass'].sum()
pct_pass_all = 100 * n_pass_all_filters / len(all_overlaps)
ax_d.axhline(pct_pass_all, color=C_CTRL, ls=':', lw=0.7, alpha=0.7)
ax_d.text(250, pct_pass_all + 1.5, f'All filters\n({pct_pass_all:.1f}%)',
          fontsize=5.5, color=C_CTRL, ha='center')

ax_d.set_xlabel('Exon overlap threshold (bp)')
ax_d.set_ylabel('Reads passing threshold (%)')
ax_d.set_xlim(0, 310)
ax_d.set_ylim(0, 105)
panel_label(ax_d, 'd')

# ── Save ──
plt.tight_layout()
outpath = PROJECT / 'manuscript/figures/figS10'
save_figure(fig, str(outpath))
print(f'Saved: {outpath}.pdf')
print('Done!')
