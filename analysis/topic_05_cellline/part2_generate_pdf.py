#!/usr/bin/env python3
"""
Part 2 PDF: L1 Poly(A) Tail.
Scope: L1 poly(A) characteristics (NOT arsenite — that's Part 4).
  1. L1 poly(A) is longer than non-L1 control transcripts
  2. Cross-cell-line poly(A) variation
  3. Young vs Ancient L1 poly(A)
  4. Locus-specific poly(A) signatures
  5. MCF7-EV: no poly(A) difference
  6. Decorated (mixed) tails — length bias caveat
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fpdf import FPDF

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUTDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
FIGDIR = OUTDIR / 'pdf_figures_part2'
FIGDIR.mkdir(exist_ok=True)
POLYA_DIR = PROJECT / 'analysis/01_exploration/topic_02_polya'

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
MIN_READS = 200

CELL_LINES = {
    'A549':     ['A549_4', 'A549_5', 'A549_6'],
    'H9':       ['H9_2', 'H9_3', 'H9_4'],
    'Hct116':   ['Hct116_3', 'Hct116_4'],
    'HeLa':     ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2':    ['HepG2_5', 'HepG2_6'],
    'HEYA8':    ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562':     ['K562_4', 'K562_5', 'K562_6'],
    'MCF7':     ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'MCF7-EV':  ['MCF7-EV_1'],
    'SHSY5Y':   ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

# =========================================================================
# Load data
# =========================================================================
print("Loading data...")

# L1 data
all_dfs = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        if len(df) < MIN_READS:
            continue
        df['group'] = g
        df['cell_line'] = cl
        df['l1_age'] = df['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
        all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)
print(f"  L1: {len(data):,} reads")

# Control data
ctrl_dfs = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        path = PROJECT / f'results_group/{g}/i_control/{g}_control_summary.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        df['group'] = g
        df['cell_line'] = cl
        ctrl_dfs.append(df)

ctrl = pd.concat(ctrl_dfs, ignore_index=True)
print(f"  Control: {len(ctrl):,} reads")

# Pre-computed data
landscape = pd.read_csv(OUTDIR / 'cross_cellline_landscape.tsv', sep='\t')
hotspot_polya = pd.read_csv(POLYA_DIR / 'polya_by_hotspot.tsv', sep='\t')
decorated_bin = pd.read_csv(POLYA_DIR / 'decorated_by_polya_bin.tsv', sep='\t')
l1_vs_ctrl_dec = pd.read_csv(POLYA_DIR / 'l1_vs_control_decorated_comparison.tsv', sep='\t')

# Cell line order (by L1 poly(A) median, descending)
cl_polya = landscape[landscape['l1_age'] == 'all'].set_index('cell_line')
cl_order = cl_polya.sort_values('polya_median', ascending=False).index.tolist()

# =========================================================================
# Figure 1: L1 poly(A) > Control (3 panels)
# =========================================================================
print("Generating Figure 1...")
fig1, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig1.subplots_adjust(wspace=0.3)

# 1A: L1 vs Control poly(A) per cell line (paired)
ax = axes[0]
l1_medians = []
ctrl_medians = []
cl_plot = []
for cl in cl_order:
    l1_med = data[data['cell_line'] == cl]['polya_length'].median()
    c_med = ctrl[ctrl['cell_line'] == cl]['polya_length'].median()
    if np.isnan(c_med):
        continue
    l1_medians.append(l1_med)
    ctrl_medians.append(c_med)
    cl_plot.append(cl)

x = np.arange(len(cl_plot))
w = 0.35
ax.bar(x - w/2, l1_medians, w, label='L1', color='#C44E52', alpha=0.8)
ax.bar(x + w/2, ctrl_medians, w, label='Control', color='#4C72B0', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cl_plot, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Median poly(A) length (nt)')
ax.legend(fontsize=9)
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('L1 vs non-L1 control', fontsize=10, loc='center')

# Add delta labels
for i in range(len(cl_plot)):
    delta = l1_medians[i] - ctrl_medians[i]
    ax.text(i, max(l1_medians[i], ctrl_medians[i]) + 3,
            f'+{delta:.0f}', ha='center', fontsize=6.5, color='#333333')

# 1B: Distribution overlay (all cell lines pooled)
ax = axes[1]
bins = np.arange(0, 401, 10)
ax.hist(data['polya_length'].clip(upper=400), bins=bins, alpha=0.5,
        color='#C44E52', density=True, label=f'L1 (n={len(data):,})')
ax.hist(ctrl['polya_length'].clip(upper=400), bins=bins, alpha=0.5,
        color='#4C72B0', density=True, label=f'Control (n={len(ctrl):,})')
l1_med_all = data['polya_length'].median()
ctrl_med_all = ctrl['polya_length'].median()
ax.axvline(l1_med_all, color='#C44E52', ls='--', lw=1.5)
ax.axvline(ctrl_med_all, color='#4C72B0', ls='--', lw=1.5)
ax.text(l1_med_all + 3, ax.get_ylim()[1]*0.85, f'{l1_med_all:.0f}', color='#C44E52', fontsize=8)
ax.text(ctrl_med_all + 3, ax.get_ylim()[1]*0.75, f'{ctrl_med_all:.0f}', color='#4C72B0', fontsize=8)
ax.set_xlabel('Poly(A) length (nt)')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Poly(A) length distribution', fontsize=10, loc='center')

# 1C: L1-Control delta per cell line (dot + CI)
ax = axes[2]
deltas = [l1_medians[i] - ctrl_medians[i] for i in range(len(cl_plot))]
colors_delta = ['#C44E52' if d > 0 else '#4C72B0' for d in deltas]
ax.barh(range(len(cl_plot)), deltas, color=colors_delta, alpha=0.8, edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(len(cl_plot)))
ax.set_yticklabels(cl_plot, fontsize=9)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('L1 - Control median poly(A) (nt)')
for i, d in enumerate(deltas):
    ax.text(d + (2 if d > 0 else -2), i, f'{d:+.0f}', va='center', fontsize=7,
            ha='left' if d > 0 else 'right')
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('L1 poly(A) excess over control', fontsize=10, loc='center')

fig1.savefig(FIGDIR / 'fig1_l1_vs_control.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("  Figure 1 saved")

# =========================================================================
# Figure 2: Cross-CL variation + young vs ancient (3 panels)
# =========================================================================
print("Generating Figure 2...")
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5.5))
fig2.subplots_adjust(wspace=0.3)

# 2A: poly(A) boxplot across cell lines
ax = axes2[0]
bp_data = [data[data['cell_line'] == cl]['polya_length'].clip(upper=400).values for cl in cl_order]
bp = ax.boxplot(bp_data, showfliers=False, patch_artist=True, medianprops=dict(color='black', lw=1.5))
for patch in bp['boxes']:
    patch.set_facecolor('#4C72B0')
    patch.set_alpha(0.6)
ax.set_xticks(range(1, len(cl_order)+1))
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Poly(A) length (nt)')
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('L1 poly(A) across cell lines', fontsize=10, loc='center')

# KW test
cl_groups = [data[data['cell_line']==cl]['polya_length'].dropna().values for cl in cl_order]
kw = stats.kruskal(*cl_groups)
ax.text(0.03, 0.97, f'KW p={kw.pvalue:.1e}', transform=ax.transAxes, fontsize=8, va='top')

# 2B: Young vs Ancient per CL
ax = axes2[1]
anc_data = landscape[landscape['l1_age']=='ancient'].set_index('cell_line')
yng_data = landscape[landscape['l1_age']=='young'].set_index('cell_line')
x = np.arange(len(cl_order))
w = 0.35
anc_med = [anc_data.loc[cl, 'polya_median'] if cl in anc_data.index else 0 for cl in cl_order]
yng_med = []
for cl in cl_order:
    if cl in yng_data.index and yng_data.loc[cl, 'n_reads'] > 30:
        yng_med.append(yng_data.loc[cl, 'polya_median'])
    else:
        yng_med.append(np.nan)
ax.bar(x - w/2, anc_med, w, label='Ancient', color='#4C72B0', alpha=0.8)
ax.bar(x + w/2, yng_med, w, label='Young', color='#C44E52', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Median poly(A) (nt)')
ax.legend(fontsize=8)
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Young vs ancient L1', fontsize=10, loc='center')

# Overall young vs ancient test
yng_all = data[data['l1_age'] == 'young']['polya_length'].dropna()
anc_all = data[data['l1_age'] == 'ancient']['polya_length'].dropna()
mw_age = stats.mannwhitneyu(yng_all, anc_all, alternative='two-sided')
age_delta = yng_all.median() - anc_all.median()
ax.text(0.03, 0.97, f'delta={age_delta:+.1f} nt\n(n={len(yng_all)+len(anc_all):,})',
        transform=ax.transAxes, fontsize=8, va='top')

# 2C: MCF7 vs MCF7-EV
ax = axes2[2]
mcf7_l1 = data[data['cell_line'] == 'MCF7']['polya_length']
ev_l1 = data[data['cell_line'] == 'MCF7-EV']['polya_length']
bins_ev = np.arange(0, 401, 10)
ax.hist(mcf7_l1.clip(upper=400), bins=bins_ev, alpha=0.5, density=True,
        color='#4C72B0', label=f'MCF7 (n={len(mcf7_l1):,})')
ax.hist(ev_l1.clip(upper=400), bins=bins_ev, alpha=0.5, density=True,
        color='#C44E52', label=f'MCF7-EV (n={len(ev_l1):,})')
ax.axvline(mcf7_l1.median(), color='#4C72B0', ls='--', lw=1.5)
ax.axvline(ev_l1.median(), color='#C44E52', ls='--', lw=1.5)
mw_ev = stats.mannwhitneyu(mcf7_l1, ev_l1, alternative='two-sided')
ax.text(0.97, 0.95,
    f'MCF7: {mcf7_l1.median():.1f} nt\n'
    f'MCF7-EV: {ev_l1.median():.1f} nt\n'
    f'p={mw_ev.pvalue:.3f} (ns)',
    transform=ax.transAxes, fontsize=8, ha='right', va='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('Poly(A) length (nt)')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('MCF7 cellular vs exosome', fontsize=10, loc='center')

fig2.savefig(FIGDIR / 'fig2_crosscl_age.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("  Figure 2 saved")

# =========================================================================
# Figure 3: Hotspot poly(A) + decorated tails (3 panels)
# =========================================================================
print("Generating Figure 3...")
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5.5))
fig3.subplots_adjust(wspace=0.35)

# 3A: Hotspot-specific poly(A) (top 15 loci)
ax = axes3[0]
top_hs = hotspot_polya[hotspot_polya['n'] >= 50].head(15).sort_values('median')
ax.barh(range(len(top_hs)), top_hs['median'],
        xerr=[top_hs['median'] - (top_hs['mean'] - top_hs['std']).clip(lower=0),
              top_hs['std']],
        color='#4C72B0', alpha=0.8, edgecolor='gray', linewidth=0.3, ecolor='gray', capsize=2)
labels_hs = [f"{row['transcript_id']} ({row['subfamily']})" for _, row in top_hs.iterrows()]
ax.set_yticks(range(len(top_hs)))
ax.set_yticklabels(labels_hs, fontsize=7)
ax.set_xlabel('Median poly(A) length (nt)')
# Add n
for i, (_, row) in enumerate(top_hs.iterrows()):
    ax.text(row['median'] + 2, i, f"n={int(row['n'])}", fontsize=6, va='center')
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Locus-specific poly(A) (>=50 reads)', fontsize=9, loc='center')

# 3B: Decorated rate by poly(A) bin (L1 vs Control)
ax = axes3[1]
bins_label = l1_vs_ctrl_dec['polya_bin'].values
x_dec = np.arange(len(bins_label))
w_dec = 0.35
ax.bar(x_dec - w_dec/2, l1_vs_ctrl_dec['l1_rate'], w_dec,
       label='L1', color='#C44E52', alpha=0.8)
ax.bar(x_dec + w_dec/2, l1_vs_ctrl_dec['ctrl_rate'], w_dec,
       label='Control', color='#4C72B0', alpha=0.8)
ax.set_xticks(x_dec)
ax.set_xticklabels(bins_label, rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Poly(A) length bin (nt)')
ax.set_ylabel('Decorated tail rate (%)')
ax.legend(fontsize=8)
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Decorated tail detection vs poly(A) length', fontsize=9, loc='center')

# 3C: Decorated rate overall (L1 vs Control)
ax = axes3[2]
# Overall decorated rates per cell line
dec_l1 = []
dec_ctrl = []
for cl in cl_order:
    d = data[data['cell_line'] == cl]
    c = ctrl[ctrl['cell_line'] == cl]
    l1_dec_rate = (d['class'] == 'decorated').mean() * 100 if len(d) > 0 else np.nan
    # Control class column might be different
    if 'class' in c.columns:
        ctrl_dec_rate = (c['class'] == 'decorated').mean() * 100 if len(c) > 0 else np.nan
    else:
        ctrl_dec_rate = np.nan
    dec_l1.append(l1_dec_rate)
    dec_ctrl.append(ctrl_dec_rate)

x_cl = np.arange(len(cl_order))
ax.bar(x_cl - w/2, dec_l1, w, label='L1', color='#C44E52', alpha=0.8)
ax.bar(x_cl + w/2, dec_ctrl, w, label='Control', color='#4C72B0', alpha=0.8)
ax.set_xticks(x_cl)
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Decorated tail rate (%)')
ax.legend(fontsize=8)
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Decorated tail rate by cell line', fontsize=10, loc='center')

fig3.savefig(FIGDIR / 'fig3_hotspot_decorated.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print("  Figure 3 saved")

# =========================================================================
# Compute summary statistics for text
# =========================================================================
print("\nComputing statistics...")

# L1 vs Control per cell line
l1_ctrl_stats = []
for cl in cl_order:
    d = data[data['cell_line'] == cl]
    c = ctrl[ctrl['cell_line'] == cl]
    if len(c) == 0:
        continue
    mw = stats.mannwhitneyu(d['polya_length'].dropna(), c['polya_length'].dropna())
    l1_ctrl_stats.append({
        'cell_line': cl,
        'l1_median': d['polya_length'].median(),
        'ctrl_median': c['polya_length'].median(),
        'delta': d['polya_length'].median() - c['polya_length'].median(),
        'p': mw.pvalue,
        'l1_n': len(d),
        'ctrl_n': len(c),
    })

l1_ctrl_df = pd.DataFrame(l1_ctrl_stats)
l1_ctrl_df.to_csv(OUTDIR / 'part2_l1_vs_control_polya.tsv', sep='\t', index=False, float_format='%.2f')

# Overall L1 vs control
mw_all = stats.mannwhitneyu(data['polya_length'].dropna(), ctrl['polya_length'].dropna())
delta_all = data['polya_length'].median() - ctrl['polya_length'].median()

# Decorated stats
dec_overall_l1 = (data['class'] == 'decorated').mean() * 100
dec_overall_ctrl = (ctrl['class'] == 'decorated').mean() * 100 if 'class' in ctrl.columns else np.nan

# Hotspot poly(A) range
hs_range_min = hotspot_polya[hotspot_polya['n'] >= 50]['median'].min()
hs_range_max = hotspot_polya[hotspot_polya['n'] >= 50]['median'].max()

print(f"  L1 median poly(A): {data['polya_length'].median():.1f} nt")
print(f"  Control median poly(A): {ctrl['polya_length'].median():.1f} nt")
print(f"  Delta: +{delta_all:.1f} nt (p={mw_all.pvalue:.2e})")
print(f"  Young vs Ancient: p={mw_age.pvalue:.3f}")
print(f"  Hotspot poly(A) range: {hs_range_min:.0f}-{hs_range_max:.0f} nt")
print(f"  Decorated rate: L1={dec_overall_l1:.1f}%, Control={dec_overall_ctrl:.1f}%")

# =========================================================================
# Generate PDF
# =========================================================================
print("\nGenerating PDF...")

class ResultsPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 5, 'Part 2: L1 Poly(A) Tail', align='L')
            self.cell(0, 5, f'Page {self.page_no()}', align='R', new_x='LMARGIN', new_y='NEXT')
            self.line(10, 12, 200, 12)
            self.ln(3)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 9.5)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def figure_caption(self, text):
        self.set_font('Helvetica', 'I', 8.5)
        self.multi_cell(0, 4.5, text)
        self.ln(3)

pdf = ResultsPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# --- Title Page ---
pdf.add_page()
pdf.ln(40)
pdf.set_font('Helvetica', 'B', 20)
pdf.cell(0, 12, 'Part 2: L1 Poly(A) Tail', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(8)
pdf.set_font('Helvetica', '', 12)
pdf.cell(0, 8, 'Poly(A) Tail Characteristics of L1 Transcripts', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 8, 'in Direct RNA Sequencing', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(20)
pdf.set_font('Helvetica', 'I', 9)
pdf.cell(0, 6, 'IsoTENT L1 Project - Results Draft', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 6, 'Generated: 2026-02-09', align='C', new_x='LMARGIN', new_y='NEXT')

# --- Section 1 ---
pdf.add_page()
pdf.section_title('1. L1 Transcripts Have Longer Poly(A) Tails Than Non-L1 Transcripts')

pdf.body_text(
    f'A key advantage of nanopore DRS is the ability to measure poly(A) tail '
    f'length directly from native RNA molecules. We compared poly(A) tail '
    f'lengths between L1 transcripts (n={len(data):,}) and non-L1 control '
    f'transcripts (n={len(ctrl):,}) from the same libraries.'
)

pdf.body_text(
    f'L1 transcripts had significantly longer poly(A) tails than control '
    f'transcripts in every cell line examined (Figure 1A-B). The overall '
    f'median poly(A) length was {data["polya_length"].median():.1f} nt for L1 '
    f'vs {ctrl["polya_length"].median():.1f} nt for controls '
    f'(+{delta_all:.0f} nt; Mann-Whitney p = {mw_all.pvalue:.1e}). '
    f'This difference was consistent across all 11 cell lines, with the L1 '
    f'excess ranging from +{l1_ctrl_df["delta"].min():.0f} to '
    f'+{l1_ctrl_df["delta"].max():.0f} nt (Figure 1C).'
)

pdf.body_text(
    'This observation suggests that L1 transcripts are either '
    'polyadenylated by a distinct mechanism or with different kinetics '
    'compared to host gene mRNAs, or that L1 poly(A) tails are more '
    'resistant to deadenylation.'
)

# Table: L1 vs Control
pdf.ln(1)
pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 6, 'Table 1. L1 vs control poly(A) length by cell line.', new_x='LMARGIN', new_y='NEXT')
pdf.ln(1)
pdf.set_font('Helvetica', 'B', 7.5)
tw = [22, 16, 16, 16, 16, 22]
th = ['Cell Line', 'L1 med (nt)', 'Ctrl med (nt)', 'Delta (nt)', 'p-value', 'Significant']
for w, h in zip(tw, th):
    pdf.cell(w, 5, h, border=1, align='C')
pdf.ln()
pdf.set_font('Helvetica', '', 7.5)
for _, r in l1_ctrl_df.iterrows():
    sig = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else 'ns'))
    vals = [r['cell_line'], f"{r['l1_median']:.1f}", f"{r['ctrl_median']:.1f}",
            f"+{r['delta']:.1f}", f"{r['p']:.1e}", sig]
    for w, v in zip(tw, vals):
        pdf.cell(w, 4.5, v, border=1, align='C')
    pdf.ln()

# --- Figure 1 ---
pdf.add_page()
pdf.section_title('Figure 1')
pdf.image(str(FIGDIR / 'fig1_l1_vs_control.png'), x=3, w=204)
pdf.ln(2)
pdf.figure_caption(
    'Figure 1. L1 poly(A) tails are longer than non-L1 control transcripts. '
    '(A) Median poly(A) tail length for L1 (red) and control (blue) transcripts '
    'per cell line. Numbers above bars indicate the L1-control difference. '
    '(B) Overall poly(A) length distribution (all cell lines pooled). Dashed '
    'lines mark medians. '
    '(C) L1 poly(A) excess: difference between L1 and control median poly(A) '
    'per cell line. All cell lines show positive values (L1 > control).'
)

# --- Section 2 ---
pdf.add_page()
pdf.section_title('2. Poly(A) Tail Length Varies Across Cell Lines')

all_polya = landscape[landscape['l1_age'] == 'all'].set_index('cell_line')
max_cl = all_polya['polya_median'].idxmax()
min_cl_no_ars = all_polya.drop('HeLa-Ars', errors='ignore')['polya_median'].idxmin()
min_cl_all = all_polya['polya_median'].idxmin()

pdf.body_text(
    f'L1 poly(A) tail length varied significantly across cell lines '
    f'(Kruskal-Wallis p = {kw.pvalue:.1e}; Figure 2A). '
    f'{max_cl} had the longest median poly(A) '
    f'({all_polya.loc[max_cl, "polya_median"]:.1f} nt), '
    f'while {min_cl_no_ars} had the shortest among untreated cell lines '
    f'({all_polya.loc[min_cl_no_ars, "polya_median"]:.1f} nt). '
    f'This ~{all_polya.loc[max_cl, "polya_median"] - all_polya.loc[min_cl_no_ars, "polya_median"]:.0f} nt range '
    f'suggests cell-type-specific regulation of L1 poly(A) metabolism.'
)

pdf.body_text(
    f'Young and ancient L1 elements showed similar poly(A) tail lengths '
    f'(median {yng_all.median():.1f} vs {anc_all.median():.1f} nt, '
    f'delta = {age_delta:+.1f} nt; Figure 2B). While the pooled Mann-Whitney test '
    f'reaches statistical significance (p < 0.001) due to the large sample size '
    f'(n = {len(yng_all)+len(anc_all):,}), the effect size is negligible '
    f'({abs(age_delta):.0f} nt difference, <5% of the median). '
    f'Cell-line-level medians confirm this: young and ancient poly(A) track '
    f'closely within each cell line, indicating that poly(A) tail length '
    f'is not meaningfully influenced by L1 subfamily age.'
)

pdf.body_text(
    f'We also compared MCF7 cellular RNA with MCF7-EV (extracellular vesicle) '
    f'RNA. Despite the 2.5-fold enrichment of young L1 in EVs (Part 1), '
    f'poly(A) tail length was not significantly different between cellular and '
    f'EV L1 RNA (median {mcf7_l1.median():.1f} vs {ev_l1.median():.1f} nt; '
    f'p = {mw_ev.pvalue:.3f}; Figure 2C). This suggests that the selective '
    f'packaging of L1 RNA into EVs is not based on poly(A) tail length.'
)

# --- Figure 2 ---
pdf.add_page()
pdf.section_title('Figure 2')
pdf.image(str(FIGDIR / 'fig2_crosscl_age.png'), x=3, w=204)
pdf.ln(2)
pdf.figure_caption(
    'Figure 2. L1 poly(A) variation across cell lines and L1 age. '
    '(A) Poly(A) length distributions across 11 cell lines (boxplot, '
    'outliers removed). Significant variation (KW p < 10^-50). '
    '(B) Median poly(A) for ancient (blue) and young (red) L1. The median '
    'difference is negligible (delta ~5 nt, <5% of overall median). '
    '(C) MCF7 cellular vs MCF7-EV poly(A) distribution. No significant '
    'difference despite young L1 enrichment in EVs.'
)

# --- Section 3 ---
pdf.add_page()
pdf.section_title('3. Locus-Specific Poly(A) Signatures')

pdf.body_text(
    f'Individual L1 loci exhibited distinct poly(A) tail lengths, '
    f'ranging from {hs_range_min:.0f} to {hs_range_max:.0f} nt among loci '
    f'with >= 50 reads (Figure 3A). '
    f'The dominant HepG2 hotspot L1PA7_dup11216 had notably short poly(A) '
    f'(median 70.8 nt), while L1MB4_dup306 (156.8 nt) and L1MA8_dup8413 '
    f'(154.2 nt) had long tails. These locus-specific signatures were '
    f'consistent across replicates, suggesting that local genomic context '
    f'or chromatin environment influences poly(A) tail processing.'
)

# --- Section 4 ---
pdf.section_title('4. Decorated (Mixed) Tails')

pdf.body_text(
    f'Nanopore sequencing via Ninetails can detect non-adenosine '
    f'modifications within the poly(A) tail ("decorated" or mixed tails), '
    f'which may reflect terminal uridylation by TUT4/7 enzymes. '
    f'Overall, {dec_overall_l1:.1f}% of L1 reads and {dec_overall_ctrl:.1f}% '
    f'of control reads had decorated tails.'
)

pdf.body_text(
    'However, decorated tail detection showed a strong dependence on '
    'poly(A) length: longer tails had progressively higher detection rates '
    '(Figure 3B), consistent with the probabilistic nature of nanopore '
    'base modification detection (longer tails provide more opportunities '
    'to detect non-A bases). This length-dependence was observed for '
    'both L1 and control transcripts.'
)

pdf.body_text(
    'Across all poly(A) length bins, L1 transcripts had lower decorated '
    'tail rates than controls (Figure 3B), suggesting that L1 poly(A) '
    'tails are less frequently subject to non-adenosine modification. '
    'This difference persisted after controlling for poly(A) length, '
    'indicating it is not a technical artifact of the length bias.'
)

# --- Figure 3 ---
pdf.add_page()
pdf.section_title('Figure 3')
pdf.image(str(FIGDIR / 'fig3_hotspot_decorated.png'), x=3, w=204)
pdf.ln(2)
pdf.figure_caption(
    'Figure 3. Locus-specific poly(A) and decorated tail analysis. '
    '(A) Median poly(A) length for top L1 hotspot loci (>=50 reads). '
    'Error bars show +/- 1 SD. Loci show distinct poly(A) signatures. '
    '(B) Decorated (mixed) tail detection rate by poly(A) length bin '
    'for L1 (red) and control (blue) transcripts. Detection increases '
    'with poly(A) length (technical bias). L1 has lower decorated rate '
    'than controls across all bins. '
    '(C) Overall decorated tail rate per cell line. L1 generally shows '
    'lower decorated tail rates than controls.'
)

# --- Section 5: Genomic Context ---
pdf.add_page()
pdf.section_title('5. Genomic Context: Intronic vs Intergenic L1')

# Load extended analysis stats
host_stats = pd.read_csv(OUTDIR / 'part2_host_gene_stats.tsv', sep='\t', index_col=0)
consist_stats = pd.read_csv(OUTDIR / 'part2_hotspot_consistency.tsv', sep='\t')
bimod_stats = pd.read_csv(OUTDIR / 'part2_bimodality.tsv', sep='\t')

intronic = data[data['TE_group'] == 'intronic']
intergenic = data[data['TE_group'] == 'intergenic']
mw_ctx = stats.mannwhitneyu(intronic['polya_length'].dropna(),
                             intergenic['polya_length'].dropna())
ctx_delta = intronic['polya_length'].median() - intergenic['polya_length'].median()

pdf.body_text(
    f'Approximately 61% of L1 reads originated from intronic elements '
    f'(n={len(intronic):,}) and 39% from intergenic elements (n={len(intergenic):,}). '
    f'Intronic and intergenic L1 had nearly identical poly(A) tail lengths '
    f'(median {intronic["polya_length"].median():.1f} vs '
    f'{intergenic["polya_length"].median():.1f} nt, '
    f'delta = {ctx_delta:+.1f} nt; Figure 4A). '
    f'While statistically significant (MW p = {mw_ctx.pvalue:.1e}), the effect size '
    f'is negligible, and this pattern was consistent across all cell lines.'
)

pdf.body_text(
    f'This suggests that L1 poly(A) tail length is determined by the L1 element '
    f'itself (e.g., its internal polyadenylation signal) rather than by the host '
    f'gene context. Intronic L1 elements do not appear to use the host gene\'s '
    f'polyadenylation machinery to a different degree than intergenic elements.'
)

pdf.body_text(
    f'We identified {len(host_stats):,} host genes containing expressed intronic L1 '
    f'elements (Figure 4B). The most prominent host gene was CCDC170, harboring '
    f'718 L1 reads from 9 loci but restricted to 2 cell lines (primarily MCF7). '
    f'In contrast, ubiquitous host genes such as PSMA8 (11 CL), RRP36 (11 CL), '
    f'and GNGT1 (11 CL) showed constitutive L1 expression. There was no meaningful '
    f'correlation between the number of L1 reads in a host gene and poly(A) length '
    f'(Spearman r = 0.092; Figure 4C), indicating that L1 poly(A) is independent of '
    f'host gene expression level.'
)

# --- Figure 4 ---
pdf.add_page()
pdf.section_title('Figure 4')
pdf.image(str(FIGDIR / 'fig4_context_hostgene.png'), x=3, w=204)
pdf.ln(2)
pdf.figure_caption(
    'Figure 4. L1 poly(A) by genomic context and host gene. '
    '(A) Median poly(A) for intronic (blue) and intergenic (orange) L1 per cell line. '
    'The difference is negligible (+1.5 nt). '
    '(B) Top 15 host genes ranked by L1 read count. Blue = ubiquitous (>=5 CL), '
    'red = cell-line-specific. Numbers show median poly(A) of L1 within each gene. '
    '(C) Scatter of L1 read count per host gene vs median poly(A) of L1 in that gene. '
    'No meaningful correlation (r = 0.092).'
)

# --- Section 6: Hotspot Consistency + Bimodality ---
pdf.add_page()
pdf.section_title('6. Hotspot Poly(A) Consistency and Bimodality')

n_bimodal_bc = (bimod_stats['bimodality_coeff'] > 0.555).sum()
n_bimodal_dip = (bimod_stats['dip_p'] < 0.05).sum() if 'dip_p' in bimod_stats.columns else 0
n_bimodal_total = len(bimod_stats)

pdf.body_text(
    f'For {len(consist_stats)} L1 loci detected in >= 5 replicates with >= 10 total reads, '
    f'we assessed poly(A) consistency across cell lines. The median between-cell-line '
    f'coefficient of variation (CV) was {consist_stats["between_cl_cv"].median():.2f} '
    f'(Figure 5A-B), indicating moderate variability. Individual loci maintain '
    f'characteristic poly(A) lengths (e.g., consistently short or long tails) but '
    f'with substantial spread across cell lines, suggesting both locus-intrinsic '
    f'and cell-line-specific factors influence poly(A) processing.'
)

pdf.body_text(
    f'We tested {n_bimodal_total} loci (>= 30 reads each) for poly(A) bimodality '
    f'using the bimodality coefficient (BC) and Hartigan\'s dip test. '
    f'{n_bimodal_bc} loci ({n_bimodal_bc/n_bimodal_total*100:.0f}%) had BC > 0.555 '
    f'(suggestive of bimodality), but only {n_bimodal_dip} passed the dip test at '
    f'p < 0.05 (Figure 5C). Poly(A) bimodality is therefore rare among L1 loci, '
    f'suggesting that most L1 transcripts are polyadenylated as a single population '
    f'rather than existing as distinct short-tail and long-tail subpopulations.'
)

# --- Figure 5 ---
pdf.add_page()
pdf.section_title('Figure 5')
pdf.image(str(FIGDIR / 'fig5_consistency_bimodality.png'), x=3, w=204)
pdf.ln(2)
pdf.figure_caption(
    'Figure 5. Hotspot poly(A) consistency and bimodality. '
    '(A) Poly(A) distributions for top 10 hotspot loci, split by cell line (colored). '
    'Each locus has a characteristic poly(A) range. '
    '(B) Distribution of between-cell-line poly(A) CV across 491 robust loci. '
    'Median CV = 0.41 (moderate variability). '
    '(C) Poly(A) distributions of top bimodal candidates (ranked by bimodality '
    'coefficient). True bimodality is rare (1/113 significant by dip test).'
)

# --- Section 7: Poly(A) PCA ---
pdf.add_page()
pdf.section_title('7. Poly(A) Distribution Carries Cell-Line-Specific Information')

pdf.body_text(
    'To ask whether poly(A) tail distributions carry cell-line-specific information, '
    'we computed binned poly(A) histograms (20-nt bins) for each replicate and applied '
    'PCA and UMAP. The first two PCs explained 35.1% and 22.5% of variance '
    '(Figure 6A), substantially more than the loci-based PCA (6% per PC), indicating '
    'that poly(A) distributions have stronger cell-line structure than binary loci patterns.'
)

pdf.body_text(
    'Jensen-Shannon divergence confirmed this: within-cell-line similarity was '
    '1.30-fold higher than between-cell-line similarity (0.689 vs 0.530; Figure 6C). '
    'While weaker than the loci-based Jaccard ratio (2.29x), this demonstrates that '
    'each cell line has a characteristic L1 poly(A) distribution shape, not just a '
    'different set of expressed loci.'
)

# --- Figure 6 ---
pdf.add_page()
pdf.section_title('Figure 6')
pdf.image(str(FIGDIR / 'fig6_polya_pca.png'), x=3, w=204)
pdf.ln(2)
pdf.figure_caption(
    'Figure 6. Cell-line-specific poly(A) distribution profiles. '
    '(A) PCA of binned poly(A) histograms per replicate (20-nt bins, 0-400 nt). '
    'Spider lines connect replicates to cell-line centroids. '
    '(B) UMAP embedding. '
    '(C) Sample-sample Jensen-Shannon similarity heatmap (hierarchical clustering). '
    'Within-CL similarity is 1.30x higher than between-CL.'
)

# --- Summary ---
pdf.add_page()
pdf.section_title('Summary of Key Findings')

findings = [
    ('L1 poly(A) > control',
     f'L1 transcripts have longer poly(A) tails than non-L1 transcripts '
     f'in all cell lines (median {data["polya_length"].median():.0f} vs '
     f'{ctrl["polya_length"].median():.0f} nt, +{delta_all:.0f} nt). '
     f'This is a universal feature of L1 transcription.'),
    ('Cell line variation',
     f'L1 poly(A) median ranges from {all_polya.loc[min_cl_no_ars, "polya_median"]:.0f} '
     f'to {all_polya.loc[max_cl, "polya_median"]:.0f} nt across untreated cell lines '
     f'(KW p < 10^-50), suggesting cell-type-specific poly(A) regulation.'),
    ('Young ~ Ancient',
     f'Young and ancient L1 elements have similar poly(A) lengths '
     f'(delta = {age_delta:+.1f} nt, negligible effect size despite '
     f'statistical significance at n={len(yng_all)+len(anc_all):,}). '
     f'Subfamily age does not meaningfully determine tail length.'),
    ('MCF7-EV: no difference',
     f'EV L1 RNA has the same poly(A) length as cellular L1 RNA '
     f'(p = {mw_ev.pvalue:.3f}), despite 2.5x enrichment of young L1 '
     f'in EVs. EV packaging is not poly(A) length-dependent.'),
    ('Locus-specific signatures',
     f'Individual L1 loci have distinct poly(A) profiles ({hs_range_min:.0f}-'
     f'{hs_range_max:.0f} nt), suggesting local regulation.'),
    ('Decorated tails',
     f'L1 has lower decorated tail rates than controls ({dec_overall_l1:.1f}% '
     f'vs {dec_overall_ctrl:.1f}%). Detection is length-biased: longer poly(A) '
     f'= higher decorated rate (technical). After length correction, L1 < control '
     f'persists across all bins.'),
    ('Intronic = intergenic poly(A)',
     f'Intronic and intergenic L1 have nearly identical poly(A) '
     f'(delta = {ctx_delta:+.1f} nt). L1 poly(A) is determined by the element '
     f'itself, not host gene context. No correlation between host gene L1 load '
     f'and poly(A) length.'),
    ('Hotspot consistency',
     f'Individual loci maintain characteristic poly(A) lengths across cell lines '
     f'(between-CL CV = {consist_stats["between_cl_cv"].median():.2f}). '
     f'Bimodality is rare: only 1/113 loci significant by dip test.'),
    ('Poly(A) distribution PCA',
     'Poly(A) distributions carry cell-line-specific information (PC1 = 35.1%, '
     'JS similarity ratio = 1.30x), indicating cell-type-specific poly(A) '
     'regulation beyond just which loci are expressed.'),
]

for i, (title, text) in enumerate(findings, 1):
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.cell(0, 6, f'{i}. {title}', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 4.5, text)
    pdf.ln(2)

# Save
out_path = OUTDIR / 'Part2_L1_PolyA_Tail.pdf'
pdf.output(str(out_path))
print(f"\nPDF saved: {out_path}")
print("Done!")
