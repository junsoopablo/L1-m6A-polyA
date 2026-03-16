#!/usr/bin/env python3
"""
Tests 2-4: Regulatory L1 poly(A) decay ↔ host gene expression (RNA-seq).

Test 2: Do regulatory L1 host genes show differential expression under arsenite?
Test 3: m6A-stratified — do high-m6A regulatory L1 host genes respond differently?
Test 4: Bidirectional validation — shortened vs lengthened L1 → host gene RNA-seq FC?
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC = PROJECT / 'analysis/01_exploration/topic_10_rnaseq_validation'
CHROM = PROJECT / 'analysis/01_exploration/topic_08_regulatory_chromatin'
OUTPUT = TOPIC / 'regulatory_l1_rnaseq_tests'
OUTPUT.mkdir(exist_ok=True)

# ====================================================================
# 1. Build gene name → gene ID mapping
# ====================================================================
print("Building gene name → ID mapping...")
gtf_file = PROJECT / 'reference/Human.gtf'
name_to_ids = {}
id_to_name = {}
with open(gtf_file) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene':
            continue
        attrs = parts[8]
        if 'gene_id' not in attrs or 'gene_name' not in attrs:
            continue
        gid = attrs.split('gene_id "')[1].split('"')[0]
        gname = attrs.split('gene_name "')[1].split('"')[0]
        name_to_ids.setdefault(gname, []).append(gid)
        id_to_name[gid] = gname

# ====================================================================
# 2. Load DESeq2 results (gene-level)
# ====================================================================
print("Loading DESeq2 results...")
deseq = pd.read_csv(TOPIC / 'tetranscripts_output/HeLa_SA_vs_UN_gene_TE_analysis.txt',
                     sep='\t', index_col=0)
deseq_genes = deseq[~deseq.index.str.contains(':')].copy()
deseq_genes = deseq_genes.dropna(subset=['log2FoldChange', 'padj'])
deseq_genes['gene_name'] = deseq_genes.index.map(lambda x: id_to_name.get(x, ''))
print(f"  DESeq2 genes with names: {(deseq_genes['gene_name'] != '').sum():,}")

# ====================================================================
# 3. Load regulatory L1 host gene data
# ====================================================================
print("Loading regulatory L1 host gene data...")
reg_delta = pd.read_csv(CHROM / 'regulatory_stress_response/gene_polya_delta.tsv', sep='\t')
reg_annot = pd.read_csv(CHROM / 'regulatory_stress_response/gene_response_annotated.tsv', sep='\t')

# Some host_gene entries have multiple genes (e.g., "BRCA1;NBR2"), expand
reg_rows = []
for _, row in reg_delta.iterrows():
    for gene in row['host_gene'].split(';'):
        gene = gene.strip()
        r = row.copy()
        r['gene_symbol'] = gene
        reg_rows.append(r)
reg = pd.DataFrame(reg_rows)

# Match to DESeq2
def get_deseq_fc(gene_symbol):
    """Get log2FC and padj from DESeq2 for a gene symbol."""
    gids = name_to_ids.get(gene_symbol, [])
    for gid in gids:
        if gid in deseq_genes.index:
            row = deseq_genes.loc[gid]
            return row['log2FoldChange'], row['padj'], row['baseMean']
    return np.nan, np.nan, np.nan

reg[['rnaseq_log2FC', 'rnaseq_padj', 'rnaseq_baseMean']] = reg['gene_symbol'].apply(
    lambda x: pd.Series(get_deseq_fc(x)))

matched = reg.dropna(subset=['rnaseq_log2FC']).copy()
unmatched = reg[reg['rnaseq_log2FC'].isna()]
print(f"  Regulatory L1 host genes: {len(reg)} (expanded from {len(reg_delta)})")
print(f"  Matched to DESeq2: {len(matched)}")
print(f"  Unmatched: {len(unmatched)} ({', '.join(unmatched['gene_symbol'].values[:5])}...)")

# Also load annotated version for enhancer/promoter info
reg_annot_rows = []
for _, row in reg_annot.iterrows():
    for gene in row['host_gene'].split(';'):
        gene = gene.strip()
        r = row.copy()
        r['gene_symbol'] = gene
        reg_annot_rows.append(r)
reg_annot_exp = pd.DataFrame(reg_annot_rows)

# ====================================================================
# 4. Load ALL intronic L1 host genes from L1 summary files
# ====================================================================
print("Loading all intronic L1 host genes for background...")
all_l1_genes = set()
for grp in ['HeLa_1', 'HeLa_2', 'HeLa_3']:
    summary_file = PROJECT / f'results_group/{grp}/g_summary/{grp}_L1_summary.tsv'
    if summary_file.exists():
        df_s = pd.read_csv(summary_file, sep='\t')
        if 'overlapping_genes' in df_s.columns:
            for genes_str in df_s['overlapping_genes'].dropna():
                for g in str(genes_str).split(','):
                    g = g.strip()
                    if g and g != 'nan':
                        all_l1_genes.add(g)

# Match all L1 host genes to DESeq2
all_l1_fc = []
for gene in all_l1_genes:
    fc, padj, bm = get_deseq_fc(gene)
    if not np.isnan(fc):
        all_l1_fc.append({'gene_symbol': gene, 'rnaseq_log2FC': fc, 'rnaseq_padj': padj,
                          'rnaseq_baseMean': bm})
all_l1_df = pd.DataFrame(all_l1_fc) if all_l1_fc else pd.DataFrame(columns=['gene_symbol', 'rnaseq_log2FC', 'rnaseq_padj', 'rnaseq_baseMean'])
print(f"  All intronic L1 host genes matched to DESeq2: {len(all_l1_df)}")

# Non-regulatory L1 host genes (for comparison)
reg_gene_set = set(matched['gene_symbol'])
if len(all_l1_df) > 0:
    nonreg_l1_df = all_l1_df[~all_l1_df['gene_symbol'].isin(reg_gene_set)]
else:
    nonreg_l1_df = pd.DataFrame(columns=['gene_symbol', 'rnaseq_log2FC', 'rnaseq_padj', 'rnaseq_baseMean'])

# ====================================================================
# TEST 2: Regulatory L1 host genes vs background
# ====================================================================
print("\n" + "=" * 70)
print("TEST 2: Regulatory L1 host gene expression under arsenite")
print("=" * 70)

print(f"\n--- Regulatory L1 host genes vs all genes ---")
all_genes_fc = deseq_genes[deseq_genes['baseMean'] > 50]['log2FoldChange']
reg_fc = matched[matched['rnaseq_baseMean'] > 50]['rnaseq_log2FC']
print(f"  All genes (baseMean>50, n={len(all_genes_fc):,}): median log2FC = {all_genes_fc.median():.4f}")
print(f"  Regulatory L1 host (n={len(reg_fc)}): median log2FC = {reg_fc.median():.4f}")
u, p = stats.mannwhitneyu(reg_fc, all_genes_fc, alternative='two-sided')
print(f"  Mann-Whitney P = {p:.2e}")

print(f"\n--- Regulatory vs non-regulatory L1 host genes ---")
nonreg_fc = nonreg_l1_df[nonreg_l1_df['rnaseq_baseMean'] > 50]['rnaseq_log2FC']
print(f"  Non-regulatory L1 host (n={len(nonreg_fc)}): median log2FC = {nonreg_fc.median():.4f}")
print(f"  Regulatory L1 host (n={len(reg_fc)}): median log2FC = {reg_fc.median():.4f}")
u2, p2 = stats.mannwhitneyu(reg_fc, nonreg_fc, alternative='two-sided')
print(f"  Mann-Whitney P = {p2:.2e}")

# Proportion significantly changed
reg_sig = matched[(matched['rnaseq_padj'] < 0.05) & (matched['rnaseq_baseMean'] > 50)]
print(f"\n--- Significant changes among regulatory L1 host genes ---")
print(f"  Total: {len(reg_sig)} / {len(matched[matched['rnaseq_baseMean'] > 50])} "
      f"({len(reg_sig)/len(matched[matched['rnaseq_baseMean'] > 50])*100:.1f}%)")
n_up = (reg_sig['rnaseq_log2FC'] > 0).sum()
n_down = (reg_sig['rnaseq_log2FC'] < 0).sum()
print(f"  UP: {n_up}, DOWN: {n_down}")

# Per-gene detail
print(f"\n--- Per-gene regulatory L1 host (top by |delta| and |log2FC|) ---")
matched_sorted = matched.sort_values('rnaseq_padj')
print(f"  {'Gene':<15} {'L1 Δpoly(A)':<12} {'RNA-seq FC':<12} {'padj':<10} {'L1 response':<12}")
for _, row in matched_sorted.head(20).iterrows():
    sig = '*' if row['rnaseq_padj'] < 0.05 else ''
    print(f"  {row['gene_symbol']:<15} {row['delta']:>+8.1f}nt   "
          f"{row['rnaseq_log2FC']:>+8.3f}{sig}    {row['rnaseq_padj']:.2e}  {row['response']}")

# ====================================================================
# TEST 3: m6A-stratified host gene response
# ====================================================================
print("\n" + "=" * 70)
print("TEST 3: m6A-stratified regulatory L1 host gene expression")
print("=" * 70)

m = matched[matched['rnaseq_baseMean'] > 50].copy()
m6a_med = m['m6a_avg'].median()
m['m6a_group'] = np.where(m['m6a_avg'] >= m6a_med, 'high_m6A', 'low_m6A')

for grp in ['low_m6A', 'high_m6A']:
    sub = m[m['m6a_group'] == grp]
    print(f"\n  {grp} (n={len(sub)}, median m6A/kb={sub['m6a_avg'].median():.1f}):")
    print(f"    L1 poly(A) delta: {sub['delta'].median():+.1f}nt")
    print(f"    RNA-seq log2FC:   {sub['rnaseq_log2FC'].median():+.4f}")
    sig_sub = sub[sub['rnaseq_padj'] < 0.05]
    print(f"    Sig changed: {len(sig_sub)}/{len(sub)} ({len(sig_sub)/len(sub)*100:.0f}%)")
    print(f"    UP: {(sig_sub['rnaseq_log2FC'] > 0).sum()}, DOWN: {(sig_sub['rnaseq_log2FC'] < 0).sum()}")

u3, p3 = stats.mannwhitneyu(
    m[m['m6a_group'] == 'high_m6A']['rnaseq_log2FC'],
    m[m['m6a_group'] == 'low_m6A']['rnaseq_log2FC'])
print(f"\n  High vs Low m6A: MW P = {p3:.2e}")

# Correlation: L1 m6A/kb ↔ host gene RNA-seq FC
r_m6a, p_m6a = stats.spearmanr(m['m6a_avg'], m['rnaseq_log2FC'])
print(f"  m6A/kb ↔ RNA-seq log2FC: ρ = {r_m6a:.3f}, P = {p_m6a:.2e}")

# ====================================================================
# TEST 4: Bidirectional validation — L1 shortened vs lengthened
# ====================================================================
print("\n" + "=" * 70)
print("TEST 4: Bidirectional poly(A) response → RNA-seq validation")
print("=" * 70)

for resp in ['shortened', 'stable', 'lengthened']:
    sub = matched[(matched['response'] == resp) & (matched['rnaseq_baseMean'] > 50)]
    if len(sub) == 0:
        continue
    print(f"\n  {resp.upper()} L1 poly(A) (n={len(sub)}):")
    print(f"    L1 poly(A) delta: {sub['delta'].median():+.1f}nt")
    print(f"    RNA-seq log2FC:   {sub['rnaseq_log2FC'].median():+.4f} "
          f"(mean: {sub['rnaseq_log2FC'].mean():+.4f})")
    sig_sub = sub[sub['rnaseq_padj'] < 0.05]
    up = (sig_sub['rnaseq_log2FC'] > 0).sum()
    down = (sig_sub['rnaseq_log2FC'] < 0).sum()
    print(f"    Sig: {len(sig_sub)}/{len(sub)} (UP={up}, DOWN={down})")

# Key test: do host genes of shortened L1 vs lengthened L1 show different RNA-seq FC?
short_genes = matched[(matched['response'] == 'shortened') & (matched['rnaseq_baseMean'] > 50)]
long_genes = matched[(matched['response'] == 'lengthened') & (matched['rnaseq_baseMean'] > 50)]
if len(short_genes) > 3 and len(long_genes) > 3:
    u4, p4 = stats.mannwhitneyu(short_genes['rnaseq_log2FC'], long_genes['rnaseq_log2FC'])
    print(f"\n  Shortened vs Lengthened RNA-seq FC: MW P = {p4:.2e}")
    print(f"    Shortened: {short_genes['rnaseq_log2FC'].median():+.4f}")
    print(f"    Lengthened: {long_genes['rnaseq_log2FC'].median():+.4f}")

# Correlation: L1 poly(A) delta ↔ host gene RNA-seq FC
m_all = matched[matched['rnaseq_baseMean'] > 50]
r_delta, p_delta = stats.spearmanr(m_all['delta'], m_all['rnaseq_log2FC'])
print(f"\n  L1 poly(A) delta ↔ RNA-seq log2FC: ρ = {r_delta:.3f}, P = {p_delta:.2e}")

# ====================================================================
# 5. Specific gene highlights
# ====================================================================
print("\n" + "=" * 70)
print("GENE HIGHLIGHTS: Regulatory L1 with concordant/discordant signals")
print("=" * 70)

highlight_genes = ['CKS2', 'GSDMD', 'BRCA1', 'HDAC5', 'PON2', 'TP53BP2',
                   'TTLL4', 'KPNA2', 'AFF1', 'PEX14', 'BIRC6', 'WNK1']
print(f"\n  {'Gene':<12} {'L1 Δpoly(A)':>12} {'L1 m6A/kb':>10} {'RNA-seq FC':>11} {'padj':>10} {'L1 resp':>10}")
for gene in highlight_genes:
    sub = matched[matched['gene_symbol'] == gene]
    if len(sub) == 0:
        continue
    row = sub.iloc[0]
    sig = '***' if row['rnaseq_padj'] < 0.001 else ('**' if row['rnaseq_padj'] < 0.01 else ('*' if row['rnaseq_padj'] < 0.05 else ''))
    print(f"  {gene:<12} {row['delta']:>+9.1f}nt  {row['m6a_avg']:>8.1f}  {row['rnaseq_log2FC']:>+8.3f}{sig:<3}  "
          f"{row['rnaseq_padj']:.1e}  {row['response']:>10}")

# ====================================================================
# 6. Figure
# ====================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
plt.subplots_adjust(hspace=0.40, wspace=0.35)

# (a) Regulatory vs non-regulatory vs all genes — violin/box
ax = axes[0, 0]
data_list = [all_genes_fc.values,
             nonreg_fc.values,
             reg_fc.values]
bp = ax.boxplot(data_list, positions=[1, 2, 3], widths=0.5,
                patch_artist=True, showfliers=False)
colors = ['#E0E0E0', '#90A4AE', '#E65100']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f'All genes\n(n={len(all_genes_fc):,})',
                     f'Non-reg L1\nhost (n={len(nonreg_fc)})',
                     f'Regulatory L1\nhost (n={len(reg_fc)})'], fontsize=8)
ax.set_ylabel('log₂FC (Ars/UN)', fontsize=9)
ax.set_title('Host gene expression\nunder arsenite', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.05, 0.95, f'Reg vs All: P={p:.1e}\nReg vs NonReg: P={p2:.1e}',
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc'))
ax.text(-0.12, 1.05, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (b) Per-gene: L1 poly(A) delta vs RNA-seq log2FC
ax = axes[0, 1]
m_plot = matched[matched['rnaseq_baseMean'] > 50]
colors_resp = {'shortened': '#D32F2F', 'stable': '#757575', 'lengthened': '#1565C0'}
for resp, grp in m_plot.groupby('response'):
    ax.scatter(grp['delta'], grp['rnaseq_log2FC'],
               c=colors_resp.get(resp, '#999'), s=grp['n_total'].clip(upper=50)*3,
               alpha=0.6, edgecolors='white', linewidths=0.3,
               label=f'{resp} (n={len(grp)})')

# Label key genes
for gene in ['CKS2', 'BRCA1', 'HDAC5', 'GSDMD', 'TTLL4', 'PON2', 'KPNA2']:
    sub = m_plot[m_plot['gene_symbol'] == gene]
    if len(sub) > 0:
        row = sub.iloc[0]
        ax.annotate(gene, (row['delta'], row['rnaseq_log2FC']),
                    fontsize=6, ha='left', va='bottom',
                    xytext=(4, 4), textcoords='offset points')

ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.axvline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_xlabel('L1 poly(A) Δ (nt)', fontsize=9)
ax.set_ylabel('Host gene RNA-seq log₂FC', fontsize=9)
ax.set_title(f'L1 poly(A) ↔ host gene expression\n(ρ={r_delta:.3f}, P={p_delta:.1e})',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (c) Bidirectional: shortened vs stable vs lengthened → RNA-seq FC
ax = axes[0, 2]
resp_order = ['shortened', 'stable', 'lengthened']
bp_data = [matched[(matched['response'] == r) & (matched['rnaseq_baseMean'] > 50)]['rnaseq_log2FC'].values
           for r in resp_order]
bp2 = ax.boxplot(bp_data, positions=[1, 2, 3], widths=0.5,
                 patch_artist=True, showfliers=False)
resp_colors = ['#D32F2F', '#757575', '#1565C0']
for patch, c in zip(bp2['boxes'], resp_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_xticks([1, 2, 3])
n_s = len(matched[(matched['response'] == 'shortened') & (matched['rnaseq_baseMean'] > 50)])
n_st = len(matched[(matched['response'] == 'stable') & (matched['rnaseq_baseMean'] > 50)])
n_l = len(matched[(matched['response'] == 'lengthened') & (matched['rnaseq_baseMean'] > 50)])
ax.set_xticklabels([f'Shortened\n(n={n_s})', f'Stable\n(n={n_st})', f'Lengthened\n(n={n_l})'],
                    fontsize=8)
ax.set_ylabel('Host gene RNA-seq log₂FC', fontsize=9)
ax.set_title(f'Bidirectional L1 response\n→ host gene expression', fontsize=10, fontweight='bold')
if len(short_genes) > 3 and len(long_genes) > 3:
    ax.text(0.05, 0.95, f'Short vs Long: P={p4:.1e}', transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (d) m6A stratification
ax = axes[1, 0]
low_m6a = m[m['m6a_group'] == 'low_m6A']['rnaseq_log2FC'].values
high_m6a = m[m['m6a_group'] == 'high_m6A']['rnaseq_log2FC'].values
bp3 = ax.boxplot([low_m6a, high_m6a], positions=[1, 2], widths=0.5,
                 patch_artist=True, showfliers=False)
bp3['boxes'][0].set_facecolor('#FFB74D')
bp3['boxes'][0].set_alpha(0.7)
bp3['boxes'][1].set_facecolor('#E65100')
bp3['boxes'][1].set_alpha(0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_xticks([1, 2])
ax.set_xticklabels([f'Low m6A\n(n={len(low_m6a)})', f'High m6A\n(n={len(high_m6a)})'], fontsize=8)
ax.set_ylabel('Host gene RNA-seq log₂FC', fontsize=9)
ax.set_title(f'm6A-stratified host gene\nexpression (P={p3:.1e})', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (e) m6A/kb vs RNA-seq log2FC scatter
ax = axes[1, 1]
ax.scatter(m_plot['m6a_avg'], m_plot['rnaseq_log2FC'],
           c=[colors_resp.get(r, '#999') for r in m_plot['response']],
           s=30, alpha=0.6, edgecolors='white', linewidths=0.3)
# Regression line
slope, intercept, r_val, p_val, se = stats.linregress(m_plot['m6a_avg'], m_plot['rnaseq_log2FC'])
x_line = np.linspace(m_plot['m6a_avg'].min(), m_plot['m6a_avg'].max(), 50)
ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.5, alpha=0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_xlabel('L1 m6A/kb', fontsize=9)
ax.set_ylabel('Host gene RNA-seq log₂FC', fontsize=9)
ax.set_title(f'm6A/kb ↔ host gene expression\n(ρ={r_m6a:.3f}, P={p_m6a:.1e})',
             fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'e', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (f) Gene-level heatmap: top genes by significance
ax = axes[1, 2]
top_genes = matched.dropna(subset=['rnaseq_padj']).nsmallest(15, 'rnaseq_padj')
top_genes = top_genes.sort_values('rnaseq_log2FC')
y_pos = range(len(top_genes))
colors_bar = ['#D32F2F' if r == 'shortened' else ('#1565C0' if r == 'lengthened' else '#757575')
              for r in top_genes['response']]
ax.barh(y_pos, top_genes['rnaseq_log2FC'], color=colors_bar, alpha=0.8,
        edgecolor='black', linewidth=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['gene_symbol']} (Δ{row['delta']:+.0f}nt)"
                     for _, row in top_genes.iterrows()], fontsize=7)
ax.axvline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_xlabel('RNA-seq log₂FC (Ars/UN)', fontsize=9)
ax.set_title('Top regulatory L1 host genes\n(by RNA-seq significance)', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.15, 1.05, 'f', transform=ax.transAxes, fontsize=16, fontweight='bold')

for fmt in ['pdf', 'png']:
    fig.savefig(OUTPUT / f'regulatory_l1_rnaseq_tests.{fmt}', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUTPUT / 'regulatory_l1_rnaseq_tests.pdf'}")

# ====================================================================
# Summary table
# ====================================================================
matched.to_csv(OUTPUT / 'regulatory_l1_host_gene_rnaseq.tsv', sep='\t', index=False)
print(f"Data saved: {OUTPUT / 'regulatory_l1_host_gene_rnaseq.tsv'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Test 2: Regulatory L1 host genes vs all — MW P = {p:.2e}")
print(f"        Regulatory vs non-reg L1 host — MW P = {p2:.2e}")
print(f"Test 3: m6A high vs low — MW P = {p3:.2e}")
print(f"        m6A/kb ↔ RNA-seq FC: ρ = {r_m6a:.3f}, P = {p_m6a:.2e}")
print(f"Test 4: L1 poly(A) Δ ↔ RNA-seq FC: ρ = {r_delta:.3f}, P = {p_delta:.2e}")
if len(short_genes) > 3 and len(long_genes) > 3:
    print(f"        Shortened vs Lengthened host FC: MW P = {p4:.2e}")
