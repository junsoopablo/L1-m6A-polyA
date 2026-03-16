#!/usr/bin/env python3
"""
Part 1: L1 Expression Landscape across cell lines.

Analyses:
  1. Cell line summary table (read counts, loci, read length, age distribution)
  2. Read length distribution (DRS 3' bias characterization)
  3. Gene context (intronic vs intergenic) by cell line and L1 age
  4. Detection rate (reference L1 loci vs expressed loci)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUTDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
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
# 0. Load data
# =========================================================================
print("=" * 80)
print("Part 1: L1 Expression Landscape")
print("=" * 80)

print("\nLoading L1 summaries...")
all_dfs = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not path.exists():
            print(f"  WARNING: {path.name} not found")
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        if len(df) < MIN_READS:
            print(f"  Skipping {g}: {len(df)} < {MIN_READS} reads")
            continue
        df['group'] = g
        df['cell_line'] = cl
        df['l1_age'] = df['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
        all_dfs.append(df)
        print(f"  {g}: {len(df):,} reads")

data = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal: {len(data):,} reads across {data['cell_line'].nunique()} cell lines")

# =========================================================================
# 1. Summary table
# =========================================================================
print("\n" + "=" * 80)
print("1. Cell Line Summary Table")
print("=" * 80)

summary_rows = []
for cl in CELL_LINES:
    d = data[data['cell_line'] == cl]
    if len(d) == 0:
        continue
    n_reps = d['group'].nunique()
    n_reads = len(d)
    n_loci = d['transcript_id'].nunique()
    n_subfam = d['gene_id'].nunique()
    young_n = (d['l1_age'] == 'young').sum()
    young_pct = young_n / n_reads * 100
    rl_med = d['read_length'].median()
    rl_q25 = d['read_length'].quantile(0.25)
    rl_q75 = d['read_length'].quantile(0.75)
    # Per-replicate mean read count
    reads_per_rep = n_reads / n_reps

    summary_rows.append({
        'cell_line': cl,
        'n_reps': n_reps,
        'n_reads': n_reads,
        'reads_per_rep': reads_per_rep,
        'n_loci': n_loci,
        'n_subfamilies': n_subfam,
        'young_pct': young_pct,
        'rdlen_median': rl_med,
        'rdlen_Q25': rl_q25,
        'rdlen_Q75': rl_q75,
    })

summary = pd.DataFrame(summary_rows).sort_values('n_reads', ascending=False)
summary.to_csv(OUTDIR / 'part1_summary_table.tsv', sep='\t', index=False, float_format='%.1f')

print(f"\n{'Cell Line':<12} {'Reps':>4} {'Reads':>7} {'Reads/Rep':>9} "
      f"{'Loci':>6} {'Subfam':>6} {'Young%':>7} "
      f"{'rdLen med':>10} {'Q25-Q75':>14}")
print("-" * 95)
for _, r in summary.iterrows():
    print(f"{r['cell_line']:<12} {int(r['n_reps']):>4} {int(r['n_reads']):>7,} "
          f"{r['reads_per_rep']:>9,.0f} "
          f"{int(r['n_loci']):>6,} {int(r['n_subfamilies']):>6} "
          f"{r['young_pct']:>6.1f}% "
          f"{r['rdlen_median']:>10,.0f} "
          f"{r['rdlen_Q25']:>6,.0f}-{r['rdlen_Q75']:>5,.0f}")

# Total
print("-" * 95)
print(f"{'TOTAL':<12} {data['group'].nunique():>4} {len(data):>7,} "
      f"{'':>9} {data['transcript_id'].nunique():>6,} {data['gene_id'].nunique():>6}")

# =========================================================================
# 2. Read length distribution (DRS 3' bias)
# =========================================================================
print("\n" + "=" * 80)
print("2. Read Length Distribution")
print("=" * 80)

# 2a. Overall stats
for age in ['young', 'ancient']:
    d = data[data['l1_age'] == age]
    print(f"\n  {age.upper()} L1 (n={len(d):,}):")
    print(f"    Read length: median={d['read_length'].median():.0f}, "
          f"mean={d['read_length'].mean():.0f}, "
          f"Q25-Q75=[{d['read_length'].quantile(0.25):.0f}-{d['read_length'].quantile(0.75):.0f}]")

# 2b. Per-cell-line read length
print(f"\n  {'Cell Line':<12} {'Ancient med':>11} {'Young med':>10} {'Ancient n':>10} {'Young n':>8}")
print("  " + "-" * 55)
for cl in summary['cell_line']:
    d = data[data['cell_line'] == cl]
    anc = d[d['l1_age'] == 'ancient']
    yng = d[d['l1_age'] == 'young']
    anc_med = anc['read_length'].median() if len(anc) > 0 else np.nan
    yng_med = yng['read_length'].median() if len(yng) > 10 else np.nan
    print(f"  {cl:<12} {anc_med:>11,.0f} {yng_med:>10,.0f} "
          f"{len(anc):>10,} {len(yng):>8,}")

# KW test for read length across cell lines
cl_groups_rl = [g['read_length'].values for _, g in data.groupby('cell_line')]
kw_rl = stats.kruskal(*cl_groups_rl)
print(f"\n  KW test (read length across cell lines): H={kw_rl.statistic:.1f}, p={kw_rl.pvalue:.2e}")

# 2c. dist_to_3prime analysis (3' bias characterization)
print("\n  --- 3' Coverage Bias (dist_to_3prime) ---")
for age in ['young', 'ancient']:
    d = data[(data['l1_age'] == age) & (data['dist_to_3prime'].notna())]
    if len(d) == 0:
        continue
    print(f"\n  {age.upper()} L1 (n={len(d):,}):")
    print(f"    dist_to_3prime: median={d['dist_to_3prime'].median():.0f}, "
          f"mean={d['dist_to_3prime'].mean():.0f}, "
          f"max={d['dist_to_3prime'].max():.0f}")
    # Fraction within 1kb of 3' end
    frac_1kb = (d['dist_to_3prime'] <= 1000).mean() * 100
    frac_2kb = (d['dist_to_3prime'] <= 2000).mean() * 100
    print(f"    Within 1kb of 3' end: {frac_1kb:.1f}%")
    print(f"    Within 2kb of 3' end: {frac_2kb:.1f}%")

# =========================================================================
# 3. Gene Context (Intronic vs Intergenic)
# =========================================================================
print("\n" + "=" * 80)
print("3. Gene Context Analysis (TE_group)")
print("=" * 80)

# 3a. Overall distribution
te_group_dist = data['TE_group'].value_counts()
print("\n  Overall TE_group distribution:")
for grp, cnt in te_group_dist.items():
    print(f"    {grp}: {cnt:,} ({cnt/len(data)*100:.1f}%)")

# 3b. By cell line
print(f"\n  {'Cell Line':<12} ", end='')
te_groups = [g for g in ['intronic', 'intergenic'] if g in data['TE_group'].values]
# Also check other groups
other_groups = [g for g in data['TE_group'].unique() if g not in te_groups and pd.notna(g)]
all_te_groups = te_groups + sorted(other_groups)
for grp in all_te_groups[:4]:  # Top 4 groups
    print(f"{'%'+grp:>12}", end='')
print(f"  {'n':>7}")
print("  " + "-" * 65)

context_rows = []
for cl in summary['cell_line']:
    d = data[data['cell_line'] == cl]
    row = {'cell_line': cl, 'n': len(d)}
    print(f"  {cl:<12} ", end='')
    for grp in all_te_groups[:4]:
        pct = (d['TE_group'] == grp).sum() / len(d) * 100
        row[f'{grp}_pct'] = pct
        print(f"{pct:>11.1f}%", end='')
    print(f"  {len(d):>7,}")
    context_rows.append(row)

context_df = pd.DataFrame(context_rows)
context_df.to_csv(OUTDIR / 'part1_gene_context.tsv', sep='\t', index=False, float_format='%.1f')

# 3c. By L1 age
print("\n  Gene context by L1 age:")
for age in ['young', 'ancient']:
    d = data[data['l1_age'] == age]
    print(f"\n    {age.upper()} L1 (n={len(d):,}):")
    for grp in all_te_groups[:4]:
        pct = (d['TE_group'] == grp).sum() / len(d) * 100
        print(f"      {grp}: {pct:.1f}%")

# Chi-square test: intronic vs intergenic by young/ancient
if 'intronic' in data['TE_group'].values and 'intergenic' in data['TE_group'].values:
    ct = pd.crosstab(data['l1_age'], data['TE_group'].isin(['intronic']))
    chi2, p_chi, _, _ = stats.chi2_contingency(ct)
    print(f"\n  Chi-square (intronic rate: young vs ancient): chi2={chi2:.1f}, p={p_chi:.2e}")

# =========================================================================
# 4. Detection Rate (Reference L1 loci vs Expressed)
# =========================================================================
print("\n" + "=" * 80)
print("4. L1 Detection Rate")
print("=" * 80)

# Load reference L1 BED
ref_bed = PROJECT / 'reference/L1_TE_L1_family.bed'
ref_l1 = pd.read_csv(ref_bed, sep='\t', header=None,
                      names=['chr', 'start', 'end', 'subfamily', 'locus_id'])
ref_l1['length'] = ref_l1['end'] - ref_l1['start']
ref_l1['l1_age'] = ref_l1['subfamily'].apply(lambda x: 'young' if x in YOUNG else 'ancient')

n_ref_total = len(ref_l1)
n_ref_young = (ref_l1['l1_age'] == 'young').sum()
n_ref_ancient = (ref_l1['l1_age'] == 'ancient').sum()

print(f"\n  Reference L1 loci: {n_ref_total:,}")
print(f"    Young (L1HS/PA1-3): {n_ref_young:,} ({n_ref_young/n_ref_total*100:.1f}%)")
print(f"    Ancient: {n_ref_ancient:,} ({n_ref_ancient/n_ref_total*100:.1f}%)")

# Reference subfamily size distribution
ref_subfam = ref_l1.groupby('subfamily').agg(
    n_loci=('locus_id', 'count'),
    median_length=('length', 'median')
).sort_values('n_loci', ascending=False)
print(f"\n  Top 10 reference subfamilies:")
for sf, row in ref_subfam.head(10).iterrows():
    print(f"    {sf:<12} {int(row['n_loci']):>7,} loci  (median len={int(row['median_length']):,} bp)")

# Per-cell-line detection rate
print(f"\n  {'Cell Line':<12} {'Express':>8} {'of Ref':>7} {'Det%':>6}  "
      f"{'Young expr':>10} {'Young ref':>10} {'Y Det%':>7}  "
      f"{'Anc expr':>9} {'Anc ref':>9} {'A Det%':>7}")
print("  " + "-" * 105)

detect_rows = []
for cl in summary['cell_line']:
    d = data[data['cell_line'] == cl]
    expr_loci = set(d['transcript_id'].unique())
    detected = expr_loci & set(ref_l1['locus_id'])
    det_pct = len(detected) / n_ref_total * 100

    # Young detection
    expr_young = set(d[d['l1_age'] == 'young']['transcript_id'].unique())
    ref_young_set = set(ref_l1[ref_l1['l1_age'] == 'young']['locus_id'])
    det_young = expr_young & ref_young_set
    det_young_pct = len(det_young) / n_ref_young * 100 if n_ref_young > 0 else 0

    # Ancient detection
    expr_anc = set(d[d['l1_age'] == 'ancient']['transcript_id'].unique())
    ref_anc_set = set(ref_l1[ref_l1['l1_age'] == 'ancient']['locus_id'])
    det_anc = expr_anc & ref_anc_set
    det_anc_pct = len(det_anc) / n_ref_ancient * 100 if n_ref_ancient > 0 else 0

    print(f"  {cl:<12} {len(detected):>8,} {n_ref_total:>7,} {det_pct:>5.2f}%  "
          f"{len(det_young):>10,} {n_ref_young:>10,} {det_young_pct:>6.2f}%  "
          f"{len(det_anc):>9,} {n_ref_ancient:>9,} {det_anc_pct:>6.2f}%")

    detect_rows.append({
        'cell_line': cl,
        'expressed_loci': len(detected),
        'reference_loci': n_ref_total,
        'detection_pct': det_pct,
        'young_expressed': len(det_young),
        'young_reference': n_ref_young,
        'young_det_pct': det_young_pct,
        'ancient_expressed': len(det_anc),
        'ancient_reference': n_ref_ancient,
        'ancient_det_pct': det_anc_pct,
    })

detect_df = pd.DataFrame(detect_rows)
detect_df.to_csv(OUTDIR / 'part1_detection_rate.tsv', sep='\t', index=False, float_format='%.3f')

# Union across all cell lines
all_expr = set(data['transcript_id'].unique())
all_det = all_expr & set(ref_l1['locus_id'])
print(f"\n  Union (all cell lines): {len(all_det):,} / {n_ref_total:,} "
      f"({len(all_det)/n_ref_total*100:.2f}%) loci detected")

# Singleton analysis (loci with only 1 read)
loci_counts = data.groupby('transcript_id').size()
n_singleton = (loci_counts == 1).sum()
n_multi = (loci_counts > 1).sum()
print(f"\n  Loci read coverage:")
print(f"    Singleton (1 read): {n_singleton:,} ({n_singleton/len(loci_counts)*100:.1f}%)")
print(f"    Multi-read (>1):    {n_multi:,} ({n_multi/len(loci_counts)*100:.1f}%)")
print(f"    ≥5 reads:           {(loci_counts>=5).sum():,}")
print(f"    ≥10 reads:          {(loci_counts>=10).sum():,}")

# =========================================================================
# 5. Replicate consistency summary
# =========================================================================
print("\n" + "=" * 80)
print("5. Replicate Consistency (reads per replicate)")
print("=" * 80)

rep_rows = []
for cl in summary['cell_line']:
    d = data[data['cell_line'] == cl]
    for g in d['group'].unique():
        gd = d[d['group'] == g]
        rep_rows.append({
            'cell_line': cl, 'replicate': g,
            'n_reads': len(gd), 'n_loci': gd['transcript_id'].nunique(),
            'rdlen_median': gd['read_length'].median(),
            'young_pct': (gd['l1_age'] == 'young').mean() * 100,
        })

rep_df = pd.DataFrame(rep_rows)
print(f"\n  {'Replicate':<18} {'Reads':>7} {'Loci':>6} {'rdLen med':>10} {'Young%':>7}")
print("  " + "-" * 55)
for _, r in rep_df.iterrows():
    print(f"  {r['replicate']:<18} {int(r['n_reads']):>7,} {int(r['n_loci']):>6,} "
          f"{r['rdlen_median']:>10,.0f} {r['young_pct']:>6.1f}%")
rep_df.to_csv(OUTDIR / 'part1_replicate_stats.tsv', sep='\t', index=False, float_format='%.1f')

# =========================================================================
# 6. Figures
# =========================================================================
print("\n" + "=" * 80)
print("6. Generating figures...")
print("=" * 80)

# Color map for cell lines
cl_order = summary['cell_line'].tolist()
cmap = plt.cm.tab20
cl_colors = {cl: cmap(i / len(cl_order)) for i, cl in enumerate(cl_order)}

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# --- Panel A: Read counts per cell line (bar chart) ---
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(range(len(cl_order)),
               [summary[summary['cell_line']==cl]['n_reads'].values[0] for cl in cl_order],
               color=[cl_colors[cl] for cl in cl_order])
ax1.set_xticks(range(len(cl_order)))
ax1.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Total L1 reads')
ax1.set_title('A. L1 Read Counts')
ax1.ticklabel_format(axis='y', style='plain')

# --- Panel B: Read length distribution (boxplot) ---
ax2 = fig.add_subplot(gs[0, 1])
bp_data = [data[data['cell_line']==cl]['read_length'].values for cl in cl_order]
bp = ax2.boxplot(bp_data, showfliers=False, patch_artist=True)
for patch, cl in zip(bp['boxes'], cl_order):
    patch.set_facecolor(cl_colors[cl])
    patch.set_alpha(0.7)
ax2.set_xticks(range(1, len(cl_order)+1))
ax2.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Read length (bp)')
ax2.set_title('B. Read Length Distribution')

# --- Panel C: Young L1 fraction (bar chart) ---
ax3 = fig.add_subplot(gs[0, 2])
young_pcts = [summary[summary['cell_line']==cl]['young_pct'].values[0] for cl in cl_order]
bars3 = ax3.bar(range(len(cl_order)), young_pcts,
                color=[cl_colors[cl] for cl in cl_order])
ax3.set_xticks(range(len(cl_order)))
ax3.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Young L1 (%)')
ax3.set_title('C. Young L1 Fraction')
ax3.axhline(y=np.mean(young_pcts), color='red', ls='--', lw=0.8, alpha=0.5)

# --- Panel D: Gene context (stacked bar) ---
ax4 = fig.add_subplot(gs[1, 0])
te_cats = ['intronic', 'intergenic']
# Check which TE_group values actually exist
existing_cats = [c for c in te_cats if c in data['TE_group'].values]
other_cat_values = [g for g in data['TE_group'].unique() if g not in existing_cats and pd.notna(g)]
plot_cats = existing_cats + sorted(other_cat_values)[:2]  # Top categories

cat_colors = {'intronic': '#4C72B0', 'intergenic': '#DD8452', 'unclassified': '#55A868'}
bottom = np.zeros(len(cl_order))
for cat in plot_cats:
    vals = []
    for cl in cl_order:
        d = data[data['cell_line'] == cl]
        vals.append((d['TE_group'] == cat).sum() / len(d) * 100)
    color = cat_colors.get(cat, '#AAAAAA')
    ax4.bar(range(len(cl_order)), vals, bottom=bottom, label=cat, color=color, alpha=0.8)
    bottom += np.array(vals)
ax4.set_xticks(range(len(cl_order)))
ax4.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Fraction (%)')
ax4.set_title('D. Gene Context')
ax4.legend(fontsize=7, loc='upper right')

# --- Panel E: Detection rate (bar chart, young vs ancient) ---
ax5 = fig.add_subplot(gs[1, 1])
x = np.arange(len(cl_order))
w = 0.35
young_det = [detect_df[detect_df['cell_line']==cl]['young_det_pct'].values[0] for cl in cl_order]
anc_det = [detect_df[detect_df['cell_line']==cl]['ancient_det_pct'].values[0] for cl in cl_order]
ax5.bar(x - w/2, anc_det, w, label='Ancient', color='#4C72B0', alpha=0.8)
ax5.bar(x + w/2, young_det, w, label='Young', color='#C44E52', alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax5.set_ylabel('Detection rate (%)')
ax5.set_title('E. L1 Loci Detection Rate')
ax5.legend(fontsize=8)

# --- Panel F: dist_to_3prime histogram (young vs ancient) ---
ax6 = fig.add_subplot(gs[1, 2])
for age, color, label in [('ancient', '#4C72B0', 'Ancient'), ('young', '#C44E52', 'Young')]:
    d = data[(data['l1_age'] == age) & (data['dist_to_3prime'].notna())]
    if len(d) > 0:
        ax6.hist(d['dist_to_3prime'].clip(upper=5000), bins=50,
                 alpha=0.5, color=color, label=f'{label} (n={len(d):,})',
                 density=True)
ax6.set_xlabel("Distance to 3' end (bp)")
ax6.set_ylabel('Density')
ax6.set_title("F. 3' Coverage Bias")
ax6.legend(fontsize=8)

# --- Panel G: Loci read coverage distribution ---
ax7 = fig.add_subplot(gs[2, 0])
counts = loci_counts.clip(upper=20)
ax7.hist(counts, bins=range(1, 22), color='#4C72B0', alpha=0.8, edgecolor='white')
ax7.set_xlabel('Reads per locus')
ax7.set_ylabel('Number of loci')
ax7.set_title('G. Loci Coverage Distribution')
ax7.set_xticks(range(1, 21))
ax7.set_xticklabels([str(i) if i < 20 else '20+' for i in range(1, 21)], fontsize=7)

# --- Panel H: Read length young vs ancient by cell line ---
ax8 = fig.add_subplot(gs[2, 1])
anc_rl = [data[(data['cell_line']==cl) & (data['l1_age']=='ancient')]['read_length'].median()
          for cl in cl_order]
yng_rl = []
for cl in cl_order:
    d = data[(data['cell_line']==cl) & (data['l1_age']=='young')]
    yng_rl.append(d['read_length'].median() if len(d) > 10 else np.nan)
ax8.bar(x - w/2, anc_rl, w, label='Ancient', color='#4C72B0', alpha=0.8)
ax8.bar(x + w/2, yng_rl, w, label='Young', color='#C44E52', alpha=0.8)
ax8.set_xticks(x)
ax8.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax8.set_ylabel('Median read length (bp)')
ax8.set_title('H. Read Length by L1 Age')
ax8.legend(fontsize=8)

# --- Panel I: Replicate consistency (reads per replicate) ---
ax9 = fig.add_subplot(gs[2, 2])
for i, cl in enumerate(cl_order):
    reps = rep_df[rep_df['cell_line'] == cl]['n_reads'].values
    for r in reps:
        ax9.scatter(i, r, color=cl_colors[cl], s=40, alpha=0.7, zorder=3)
    if len(reps) > 1:
        ax9.plot([i, i], [reps.min(), reps.max()], color=cl_colors[cl], lw=1.5, alpha=0.5)
ax9.set_xticks(range(len(cl_order)))
ax9.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax9.set_ylabel('Reads per replicate')
ax9.set_title('I. Replicate Read Counts')

plt.suptitle('Part 1: L1 Expression Landscape', fontsize=14, fontweight='bold', y=0.98)
plt.savefig(OUTDIR / 'part1_expression_landscape.png', dpi=200, bbox_inches='tight')
print(f"\n  Figure saved: part1_expression_landscape.png")
plt.close()

# =========================================================================
# 7. Key findings summary
# =========================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)

total_reads = len(data)
total_loci = data['transcript_id'].nunique()
max_cl = summary.iloc[0]
min_cl = summary.iloc[-1]

print(f"""
1. SCALE: {total_reads:,} L1 reads across {data['cell_line'].nunique()} cell lines,
   mapping to {total_loci:,} unique loci from {data['gene_id'].nunique()} subfamilies.

2. RANGE: {max_cl['cell_line']} has most reads ({int(max_cl['n_reads']):,}),
   {min_cl['cell_line']} fewest ({int(min_cl['n_reads']):,}).
   Read counts vary >10x across cell lines.

3. ANCIENT DOMINANCE: Young L1 fraction ranges {summary['young_pct'].min():.1f}%-{summary['young_pct'].max():.1f}%,
   mean={summary['young_pct'].mean():.1f}%. Ancient L1 dominates in all cell lines.

4. READ LENGTH: Median {data['read_length'].median():.0f} bp
   (Ancient: {data[data['l1_age']=='ancient']['read_length'].median():.0f} bp,
    Young: {data[data['l1_age']=='young']['read_length'].median():.0f} bp).

5. DETECTION: {len(all_det):,} / {n_ref_total:,} ({len(all_det)/n_ref_total*100:.1f}%)
   reference L1 loci detected (union of all cell lines).

6. SPARSITY: {n_singleton:,}/{len(loci_counts):,} ({n_singleton/len(loci_counts)*100:.0f}%)
   of expressed loci have only 1 read (singleton).
""")

print("Done!")
