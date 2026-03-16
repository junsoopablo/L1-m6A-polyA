#!/usr/bin/env python3
"""
Supplementary Figure S6: L1 Overlap Fraction Classifier & Internal-A Validation.

(a) Overlap fraction threshold sweep — arsenite Δpoly(A) by overlap bin
(b) Baseline poly(A) by overlap group (rules out internal-A inflation at baseline)
(c) Sensitivity: exclude high 3' A-content elements — immunity persists
(d) Within high-overlap: A-content does not correlate with immunity

Simple narrative: high overlap = structurally intact L1 with own PAS = arsenite-resistant.
Not driven by internal-A artifact.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures')
from fig_style import *
import pandas as pd
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'

setup_style()

# ── Load data ──
df = pd.read_csv(TOPIC_07 / 'internal_a_validation/internal_a_annotated.tsv', sep='\t')
df['is_ars'] = df['cell_line'] == 'HeLa-Ars'

def ars_delta(sub):
    hela = sub[sub['cell_line'] == 'HeLa']['polya_length'].dropna()
    ars = sub[sub['cell_line'] == 'HeLa-Ars']['polya_length'].dropna()
    if len(hela) < 5 or len(ars) < 5:
        return np.nan, np.nan, len(hela) + len(ars)
    _, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
    return ars.median() - hela.median(), p, len(hela) + len(ars)

# ── Figure: 2×2 layout ──
fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.70))
ax_a, ax_b, ax_c, ax_d = axes.flat

# =====================================================================
# Panel (a): Overlap fraction threshold sweep
# =====================================================================
thresholds = np.arange(0.05, 0.96, 0.05)
deltas_lo, deltas_hi, ns_lo, ns_hi = [], [], [], []

for thr in thresholds:
    sub_lo = df[df['overlap_frac'] <= thr]
    sub_hi = df[df['overlap_frac'] > thr]
    d_lo, _, n_lo = ars_delta(sub_lo)
    d_hi, _, n_hi = ars_delta(sub_hi)
    deltas_lo.append(d_lo)
    deltas_hi.append(d_hi)
    ns_lo.append(n_lo)
    ns_hi.append(n_hi)

ax_a.plot(thresholds, deltas_hi, color=C_L1, lw=1.5, label='≤ threshold', zorder=3)
ax_a.plot(thresholds, deltas_lo, color=C_CTRL, lw=1.5, label='> threshold', zorder=3)
ax_a.axhline(0, color=C_GREY, lw=0.5, ls='--', zorder=1)
ax_a.axvline(0.7, color=C_TEXT, lw=0.5, ls=':', alpha=0.5, zorder=1)
ax_a.set_xlabel('Overlap fraction threshold')
ax_a.set_ylabel('Δ poly(A) (nt)')
ax_a.legend(loc='lower right', frameon=False)
ax_a.set_xlim(0.05, 0.95)
ax_a.set_ylim(-45, 15)
ax_a.text(0.72, 8, 'threshold\n= 0.7', fontsize=FS_LEGEND_SMALL, color=C_TEXT, alpha=0.7)
panel_label(ax_a, 'a')

# =====================================================================
# Panel (b): Baseline poly(A) by overlap group — normal condition only
# =====================================================================
hela_normal = df[df['cell_line'] == 'HeLa'].copy()
ov_bins = [0, 0.3, 0.5, 0.7, 1.01]
ov_labels = ['<0.3', '0.3–0.5', '0.5–0.7', '>0.7']
hela_normal['ov_grp'] = pd.cut(hela_normal['overlap_frac'], bins=ov_bins, labels=ov_labels)

positions = [1, 2, 3, 4]
colors_violin = [C_CTRL, C_CTRL, C_CTRL, C_L1]
data_by_grp = [hela_normal[hela_normal['ov_grp'] == g]['polya_length'].dropna().values for g in ov_labels]

vp = ax_b.violinplot(data_by_grp, positions=positions, showextrema=False, widths=0.65)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(colors_violin[i])
    body.set_alpha(0.3)
    body.set_edgecolor('none')

add_strip(ax_b, data_by_grp, positions, colors=colors_violin, size=0.8, alpha=0.15)
medians = []
for i, (data, pos) in enumerate(zip(data_by_grp, positions)):
    med = median_line(ax_b, data, pos, color=colors_violin[i], lw=1.2)
    medians.append(med)
    ax_b.text(pos, med + 8, f'{med:.0f}', ha='center', va='bottom', fontsize=FS_LEGEND_SMALL, color=C_TEXT)

ax_b.set_xticks(positions)
ax_b.set_xticklabels(ov_labels)
ax_b.set_xlabel('Overlap fraction')
ax_b.set_ylabel('Poly(A) length (nt)\n(HeLa, normal)')
ax_b.set_ylim(0, 400)
# Annotate: high-ov is NOT inflated
ax_b.annotate('Not inflated', xy=(4, medians[3]), xytext=(4.3, medians[3] + 60),
              fontsize=FS_LEGEND_SMALL, color=C_TEXT, alpha=0.7,
              arrowprops=dict(arrowstyle='->', color=C_TEXT, lw=0.5))
panel_label(ax_b, 'b')

# =====================================================================
# Panel (c): Sensitivity — exclude high A-content elements
# =====================================================================
a_thresholds = [1.0, 0.6, 0.5, 0.4, 0.3]
dd_values = []
hi_deltas = []
lo_deltas = []

for a_thr in a_thresholds:
    sub = df[df['a_frac_last30bp'] <= a_thr]
    hi = sub[sub['overlap_frac'] > 0.7]
    lo = sub[sub['overlap_frac'] <= 0.7]
    d_hi, _, _ = ars_delta(hi)
    d_lo, _, _ = ars_delta(lo)
    dd = abs(d_hi - d_lo) if not (np.isnan(d_hi) or np.isnan(d_lo)) else np.nan
    hi_deltas.append(d_hi)
    lo_deltas.append(d_lo)
    dd_values.append(dd)

x_pos = np.arange(len(a_thresholds))
bar_width = 0.35
bars_hi = ax_c.bar(x_pos - bar_width/2, hi_deltas, bar_width,
                    color=C_L1, alpha=0.7, label='Overlap > 0.7')
bars_lo = ax_c.bar(x_pos + bar_width/2, lo_deltas, bar_width,
                    color=C_CTRL, alpha=0.7, label='Overlap ≤ 0.7')
ax_c.axhline(0, color=C_GREY, lw=0.5, ls='--')

labels_c = ['None\n(all)', '> 0.6', '> 0.5', '> 0.4', '> 0.3']
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(labels_c, fontsize=FS_LEGEND_SMALL)
ax_c.set_xlabel('Excluded: 3ʹ A-fraction')
ax_c.set_ylabel('Δ poly(A) (nt)')
ax_c.legend(loc='lower left', frameon=False, fontsize=FS_LEGEND_SMALL)

# Annotate |ΔΔ| on top
for i, dd in enumerate(dd_values):
    if not np.isnan(dd):
        y_top = max(hi_deltas[i], lo_deltas[i]) + 3
        ax_c.text(i, y_top + 2, f'|ΔΔ|={dd:.0f}', ha='center', fontsize=5.5, color=C_TEXT)
ax_c.set_ylim(-40, 45)
panel_label(ax_c, 'c')

# =====================================================================
# Panel (d): Within high-overlap — A-content vs poly(A), scatter
# =====================================================================
hi_ov = df[df['overlap_frac'] > 0.7].dropna(subset=['a_frac_last30bp', 'polya_length']).copy()

for cond, color, marker, label in [('HeLa', C_NORMAL, 'o', 'Normal'),
                                     ('HeLa-Ars', C_STRESS, 's', 'Arsenite')]:
    sub = hi_ov[hi_ov['cell_line'] == cond]
    ax_d.scatter(sub['a_frac_last30bp'], sub['polya_length'],
                 s=2, alpha=0.2, color=color, edgecolors='none',
                 rasterized=True, label=label)

    # Regression line
    x = sub['a_frac_last30bp'].values
    y = sub['polya_length'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() > 10:
        slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
        x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
        ax_d.plot(x_line, slope * x_line + intercept, color=color, lw=1.0, ls='--')
        # Annotate r
        y_pos = 370 if cond == 'HeLa' else 340
        ax_d.text(0.05, y_pos, f'{label}: r = {r:.3f}', fontsize=FS_LEGEND_SMALL, color=color)

ax_d.set_xlabel('L1 3ʹ A-fraction (last 30 bp)')
ax_d.set_ylabel('Poly(A) length (nt)')
ax_d.set_xlim(0, 1)
ax_d.set_ylim(0, 400)
ax_d.legend(loc='upper right', frameon=False, fontsize=FS_LEGEND_SMALL, markerscale=3)
panel_label(ax_d, 'd')

# ── Save ──
plt.tight_layout()
outpath = PROJECT / 'manuscript/figures/figS9'
save_figure(fig, str(outpath))
print(f"Saved: {outpath}.pdf")
print("Done!")
