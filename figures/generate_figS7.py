#!/usr/bin/env python3
"""
Supplementary Figure S7: Bidirectional regulatory L1 stress response.

(a) Gene-level scatter of Δpoly(A) vs m6A/kb for regulatory L1 host genes.
    m6A predicts direction: high m6A → lengthened (protective genes),
    low m6A → shortened (proliferative/inflammatory genes).
(b) Per-read scatter at enhancer L1 under arsenite stress.
    Positive m6A-poly(A) correlation at enhancers (r=0.347, p=2.3e-4).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import *
from scipy import stats
import pandas as pd

setup_style()

BASEDIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
CHROMHMM_DIR = f'{BASEDIR}/analysis/01_exploration/topic_08_regulatory_chromatin'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──
# Gene-level delta
gene_df = pd.read_csv(f'{CHROMHMM_DIR}/regulatory_stress_response/gene_polya_delta.tsv', sep='\t')
print(f"Genes with both HeLa and HeLa-Ars: {len(gene_df)}")
print(f"  shortened: {(gene_df['response']=='shortened').sum()}")
print(f"  stable: {(gene_df['response']=='stable').sum()}")
print(f"  lengthened: {(gene_df['response']=='lengthened').sum()}")

# Per-read regulatory data
per_read = pd.read_csv(f'{CHROMHMM_DIR}/stress_gene_analysis/regulatory_l1_per_read.tsv', sep='\t')
per_read = per_read[per_read['l1_age'] == 'ancient'].copy()  # ancient only
print(f"Ancient regulatory per-read: {len(per_read)}")

# ── Colors ──
C_SHORT = C_STRESS   # wine — shortened
C_LONG  = C_YOUNG    # teal — lengthened
C_STAB  = C_GREY     # grey — stable

# ── Figure ──
fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.42))

# ════════════════════════════════════════
# Panel (a): Gene-level Δpoly(A) vs m6A/kb
# ════════════════════════════════════════
ax = axes[0]

# Color by response
colors = {'shortened': C_SHORT, 'stable': C_STAB, 'lengthened': C_LONG}
sizes = np.clip(gene_df['n_total'] * 3, 8, 120)

for resp, grp in gene_df.groupby('response'):
    idx = grp.index
    ax.scatter(grp['m6a_avg'], grp['delta'],
               c=colors[resp], s=sizes[idx], alpha=0.7,
               edgecolors='white', linewidths=0.3,
               label=f'{resp.capitalize()} ({len(grp)})',
               zorder=3 if resp != 'stable' else 2)

# Regression line (all genes)
valid = gene_df.dropna(subset=['m6a_avg', 'delta'])
slope, intercept, r, p, se = stats.linregress(valid['m6a_avg'], valid['delta'])
x_fit = np.linspace(0, valid['m6a_avg'].max() * 1.05, 100)
ax.plot(x_fit, slope * x_fit + intercept, color=C_TEXT, lw=0.8, ls='--', zorder=1)

# Spearman
rho, sp = stats.spearmanr(valid['m6a_avg'], valid['delta'])
ax.text(0.03, 0.97, f'Spearman ρ = {rho:.2f}\nP = {sp:.1e}',
        transform=ax.transAxes, fontsize=FS_ANNOT_SMALL, va='top', ha='left',
        color=C_TEXT)

# Label key genes
labels_to_show = {
    # Lengthened (protective)
    'HDAC5': (0.6, 11),    # histone deacetylase
    'PON2': (0.4, 8),      # antioxidant
    'BRCA1;NBR2': (-1.5, 6),  # DNA repair
    'USP10': (0.5, 6),     # p53 stabilizer
    # Shortened (proliferative/inflammatory)
    'CKS2': (0.5, -8),     # cell cycle
    'GSDMD': (0.5, -6),    # pyroptosis
    'TTLL4': (0.5, -8),    # tubulin modification
    'RP4-714D9.5': (-2, 6), # lncRNA
}

for _, row in gene_df.iterrows():
    gene = row['host_gene']
    if gene in labels_to_show:
        dx, dy = labels_to_show[gene]
        # Display name: clean up composite names
        display = gene.split(';')[0]
        if display.startswith('RP') and '-' in display:
            display = display  # keep lncRNA names as-is
        ax.annotate(display, (row['m6a_avg'], row['delta']),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=FS_LEGEND_SMALL, fontstyle='italic', color=C_TEXT,
                    arrowprops=dict(arrowstyle='-', color=C_GREY, lw=0.4),
                    zorder=5)

ax.axhline(0, color=C_GREY, lw=0.5, ls=':', zorder=0)
ax.axhline(25, color=C_GREY, lw=0.3, ls=':', alpha=0.5, zorder=0)
ax.axhline(-25, color=C_GREY, lw=0.3, ls=':', alpha=0.5, zorder=0)
ax.set_xlabel('Average m$^6$A/kb')
ax.set_ylabel('Δ poly(A) (Ars − HeLa, nt)')
ax.legend(fontsize=FS_LEGEND_SMALL, loc='lower right', framealpha=0.8, edgecolor='none',
          markerscale=0.8)
panel_label(ax, 'a')

# ════════════════════════════════════════
# Panel (b): Per-read enhancer scatter under stress
# ════════════════════════════════════════
ax = axes[1]

# Enhancer ancient L1, HeLa-Ars only
enh_stress = per_read[(per_read['chromhmm_group'] == 'Enhancer') &
                       (per_read['condition'] == 'stress')].copy()
enh_normal = per_read[(per_read['chromhmm_group'] == 'Enhancer') &
                       (per_read['condition'] == 'normal')].copy()

print(f"Enhancer reads — stress: {len(enh_stress)}, normal: {len(enh_normal)}")

# Plot normal as background (low alpha)
if len(enh_normal) > 0:
    ax.scatter(enh_normal['m6a_per_kb'], enh_normal['polya_length'],
               s=3, alpha=0.15, color=C_NORMAL, edgecolors='none',
               rasterized=True, label=f'HeLa (n={len(enh_normal)})', zorder=1)

# Plot stress
if len(enh_stress) > 0:
    ax.scatter(enh_stress['m6a_per_kb'], enh_stress['polya_length'],
               s=4, alpha=0.3, color=C_STRESS, edgecolors='none',
               rasterized=True, label=f'HeLa-Ars (n={len(enh_stress)})', zorder=2)

# Regression line for stress reads
if len(enh_stress) > 10:
    valid_s = enh_stress.dropna(subset=['m6a_per_kb', 'polya_length'])
    slope_s, int_s, r_s, p_s, _ = stats.linregress(valid_s['m6a_per_kb'], valid_s['polya_length'])
    rho_s, sp_s = stats.spearmanr(valid_s['m6a_per_kb'], valid_s['polya_length'])
    x_fit = np.linspace(0, valid_s['m6a_per_kb'].quantile(0.98), 100)
    ax.plot(x_fit, slope_s * x_fit + int_s, color=C_STRESS, lw=1.0, ls='--', zorder=3)

    # Also regression for normal
    valid_n = enh_normal.dropna(subset=['m6a_per_kb', 'polya_length'])
    if len(valid_n) > 10:
        slope_n, int_n, r_n, p_n, _ = stats.linregress(valid_n['m6a_per_kb'], valid_n['polya_length'])
        rho_n, sp_n = stats.spearmanr(valid_n['m6a_per_kb'], valid_n['polya_length'])
        ax.plot(x_fit, slope_n * x_fit + int_n, color=C_NORMAL, lw=0.8, ls='--', zorder=3)

        ax.text(0.03, 0.97,
                f'HeLa-Ars: r = {r_s:.3f}, P = {p_s:.1e}\n'
                f'HeLa:     r = {r_n:.3f}, P = {p_n:.2f}',
                transform=ax.transAxes, fontsize=FS_ANNOT_SMALL, va='top', ha='left',
                color=C_TEXT, family='monospace')

# Decay zone shading
ax.axhspan(0, 30, color=C_STRESS, alpha=0.06, zorder=0)
ax.text(0.97, 0.03, 'decay zone (<30 nt)', transform=ax.transAxes,
        fontsize=FS_LEGEND_SMALL, ha='right', va='bottom', color=C_STRESS, alpha=0.6)

ax.set_xlabel('m$^6$A/kb')
ax.set_ylabel('Poly(A) tail length (nt)')
ax.set_xlim(-0.5, 25)
ax.set_ylim(-5, 350)
ax.legend(fontsize=FS_LEGEND_SMALL, loc='upper right', framealpha=0.8, edgecolor='none',
          markerscale=1.5)
panel_label(ax, 'b')

# ── Save ──
fig.tight_layout(w_pad=2.5)
save_figure(fig, f'{OUTDIR}/figS7')
print(f"\nSaved figS7.pdf")
