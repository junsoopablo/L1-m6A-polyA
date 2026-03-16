#!/usr/bin/env python3
"""
Test 1: Does intronic L1 poly(A) shortening correlate with host gene expression change?

Logic:
- DRS: per-locus L1 poly(A) change (HeLa vs HeLa-Ars)
- RNA-seq: DESeq2 gene-level log2FC (GSE278916)
- If L1 decay has cis-regulatory consequences, host genes with strong
  L1 shortening should show different expression changes.

Additional: stratify by m6A level (Test 3)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC = PROJECT / 'analysis/01_exploration/topic_10_rnaseq_validation'
CACHE = PROJECT / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
OUTPUT = TOPIC / 'l1_host_gene_correlation'
OUTPUT.mkdir(exist_ok=True)

YOUNG_L1 = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# ====================================================================
# 1. Load HeLa and HeLa-Ars L1 per-read data
# ====================================================================
print("Loading L1 summary data...")

dfs = []
for condition, groups in [('HeLa', ['HeLa_1', 'HeLa_2', 'HeLa_3']),
                           ('HeLa-Ars', ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'])]:
    for g in groups:
        fpath = RESULTS / g / 'g_summary' / f'{g}_L1_summary.tsv'
        if not fpath.exists():
            print(f"  SKIP: {fpath}")
            continue
        df = pd.read_csv(fpath, sep='\t')
        df['condition'] = condition
        df['group'] = g
        dfs.append(df)

l1 = pd.concat(dfs, ignore_index=True)
print(f"Total L1 reads: {len(l1)} ({l1['condition'].value_counts().to_dict()})")

# Filter: PASS reads with valid poly(A)
l1 = l1[(l1['qc_tag'] == 'PASS') & (l1['polya_length'] > 0)].copy()
print(f"After PASS + poly(A) > 0: {len(l1)}")

# Classify age
l1['subfamily'] = l1['transcript_id'].str.replace(r'_dup\d+', '', regex=True)
l1['age'] = l1['subfamily'].apply(lambda x: 'young' if x in YOUNG_L1 else 'ancient')

# Compute m6A/kb from Part3 cache
print("Loading Part3 cache for m6A data...")
m6a_data = {}
for g in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    cache_file = CACHE / f'{g}_l1_per_read.tsv'
    if cache_file.exists():
        c = pd.read_csv(cache_file, sep='\t')
        for _, row in c.iterrows():
            m6a_data[row['read_id']] = row['m6a_sites_high']

l1['m6a_sites'] = l1['read_id'].map(m6a_data).fillna(0)
l1['m6a_per_kb'] = l1['m6a_sites'] / (l1['read_length'] / 1000)
print(f"Reads with m6A data: {(l1['m6a_sites'] > 0).sum()}")

# ====================================================================
# 2. Compute per-locus poly(A) statistics
# ====================================================================
print("\nComputing per-locus poly(A) statistics...")

# Focus on intronic ancient L1
intronic = l1[(l1['class'] == '3UTR') | (l1['overlapping_genes'].notna() & (l1['overlapping_genes'] != ''))].copy()
# Actually, use the 'class' column — '3UTR' means intronic to a gene
# Let's check what classes exist
print(f"L1 classes: {l1['class'].value_counts().to_dict()}")

# Intronic L1 = reads with overlapping_genes
intronic = l1[l1['overlapping_genes'].notna() & (l1['overlapping_genes'] != '') & (l1['overlapping_genes'] != 'nan')].copy()
intronic = intronic[intronic['age'] == 'ancient']  # Focus on ancient
print(f"Intronic ancient L1 reads: {len(intronic)}")

# Per-locus, per-condition statistics
locus_stats = []
for locus, grp in intronic.groupby('transcript_id'):
    hela = grp[grp['condition'] == 'HeLa']
    ars = grp[grp['condition'] == 'HeLa-Ars']

    if len(hela) < 2 or len(ars) < 2:
        continue

    host_gene = grp['overlapping_genes'].mode().iloc[0] if len(grp['overlapping_genes'].mode()) > 0 else None
    if host_gene is None or str(host_gene) == 'nan':
        continue

    locus_stats.append({
        'locus': locus,
        'host_gene': host_gene,
        'subfamily': grp['subfamily'].iloc[0],
        'n_hela': len(hela),
        'n_ars': len(ars),
        'polya_hela': hela['polya_length'].median(),
        'polya_ars': ars['polya_length'].median(),
        'polya_delta': ars['polya_length'].median() - hela['polya_length'].median(),
        'm6a_kb_hela': hela['m6a_per_kb'].median(),
        'm6a_kb_ars': ars['m6a_per_kb'].median(),
        'm6a_kb_mean': grp['m6a_per_kb'].median(),
    })

ldf = pd.DataFrame(locus_stats)
print(f"\nLoci with ≥2 reads per condition: {len(ldf)}")
print(f"Unique host genes: {ldf['host_gene'].nunique()}")
print(f"Poly(A) delta: median={ldf['polya_delta'].median():.1f}, mean={ldf['polya_delta'].mean():.1f}")

# ====================================================================
# 3. Load RNA-seq DESeq2 results
# ====================================================================
print("\nLoading RNA-seq DESeq2 results...")

deseq = pd.read_csv(TOPIC / 'tetranscripts_output/HeLa_SA_vs_UN_gene_TE_analysis.txt',
                     sep='\t', index_col=0)
# Filter to genes only (exclude TEs)
deseq_genes = deseq[~deseq.index.str.contains(':')].copy()
deseq_genes = deseq_genes.dropna(subset=['padj'])
print(f"DESeq2 genes with padj: {len(deseq_genes)}")

# Map gene names to ENSEMBL IDs
gtf_file = PROJECT / 'reference/Human.gtf'
gene_name_to_id = {}
gene_id_to_name = {}
with open(gtf_file) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene':
            continue
        attrs = parts[8]
        if 'gene_id' in attrs and 'gene_name' in attrs:
            gid = attrs.split('gene_id "')[1].split('"')[0]
            gname = attrs.split('gene_name "')[1].split('"')[0]
            gene_name_to_id[gname] = gid
            gene_id_to_name[gid] = gname

# ====================================================================
# 4. Merge L1 locus data with gene expression
# ====================================================================
print("\nMerging L1 locus data with gene expression...")

# For each locus, find host gene's ENSEMBL ID and RNA-seq log2FC
merged = []
for _, row in ldf.iterrows():
    gene_name = str(row['host_gene']).split(',')[0].strip()  # Take first gene if multiple
    gene_id = gene_name_to_id.get(gene_name)
    if gene_id and gene_id in deseq_genes.index:
        gene_row = deseq_genes.loc[gene_id]
        merged.append({
            **row.to_dict(),
            'gene_name': gene_name,
            'gene_id': gene_id,
            'gene_log2FC': gene_row['log2FoldChange'],
            'gene_padj': gene_row['padj'],
            'gene_baseMean': gene_row['baseMean'],
        })

mdf = pd.DataFrame(merged)
print(f"Matched loci with RNA-seq data: {len(mdf)}")
print(f"Unique host genes: {mdf['gene_name'].nunique()}")

# Per-gene aggregation (one gene can have multiple L1 loci)
gene_agg = mdf.groupby('gene_name').agg({
    'polya_delta': 'median',
    'm6a_kb_mean': 'median',
    'gene_log2FC': 'first',
    'gene_padj': 'first',
    'gene_baseMean': 'first',
    'n_hela': 'sum',
    'n_ars': 'sum',
    'locus': 'count',
}).rename(columns={'locus': 'n_loci'})
gene_agg = gene_agg[gene_agg['gene_baseMean'] > 50]  # Filter low-expression genes
print(f"\nGenes after baseMean > 50 filter: {len(gene_agg)}")

# ====================================================================
# 5. Correlation analysis
# ====================================================================
print("\n" + "=" * 70)
print("TEST 1: L1 poly(A) shortening ↔ Host gene expression change")
print("=" * 70)

# 5a. Overall correlation
r_s, p_s = stats.spearmanr(gene_agg['polya_delta'], gene_agg['gene_log2FC'])
r_p, p_p = stats.pearsonr(gene_agg['polya_delta'], gene_agg['gene_log2FC'])
print(f"\nOverall (n={len(gene_agg)} genes):")
print(f"  Spearman ρ = {r_s:.3f}, P = {p_s:.2e}")
print(f"  Pearson  r = {r_p:.3f}, P = {p_p:.2e}")

# 5b. Stratify by L1 poly(A) shortening intensity
print(f"\n--- Stratification by L1 poly(A) delta ---")
q_labels = ['Strong shortening', 'Moderate', 'Mild/Lengthening']
gene_agg['delta_tertile'] = pd.qcut(gene_agg['polya_delta'], 3, labels=q_labels)
for t in q_labels:
    sub = gene_agg[gene_agg['delta_tertile'] == t]
    print(f"  {t} (n={len(sub)}): "
          f"L1 Δpoly(A)={sub['polya_delta'].median():.1f}nt, "
          f"gene log2FC={sub['gene_log2FC'].median():.3f} ({sub['gene_log2FC'].mean():.3f})")

# KW test across tertiles
groups_kw = [gene_agg[gene_agg['delta_tertile'] == t]['gene_log2FC'].values for t in q_labels]
kw_stat, kw_p = stats.kruskal(*groups_kw)
print(f"  Kruskal-Wallis P = {kw_p:.2e}")

# MW between extremes
strong = gene_agg[gene_agg['delta_tertile'] == 'Strong shortening']['gene_log2FC']
mild = gene_agg[gene_agg['delta_tertile'] == 'Mild/Lengthening']['gene_log2FC']
mw_u, mw_p = stats.mannwhitneyu(strong, mild, alternative='two-sided')
print(f"  Strong vs Mild MW P = {mw_p:.2e}")

# ====================================================================
# 6. TEST 3: m6A-stratified host gene response
# ====================================================================
print(f"\n{'=' * 70}")
print("TEST 3: m6A-stratified host gene expression")
print("=" * 70)

# Quartile by m6A
gene_agg['m6a_quartile'] = pd.qcut(gene_agg['m6a_kb_mean'], 4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'])
for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
    sub = gene_agg[gene_agg['m6a_quartile'] == q]
    print(f"  {q} (n={len(sub)}): "
          f"m6A/kb={sub['m6a_kb_mean'].median():.2f}, "
          f"gene log2FC={sub['gene_log2FC'].median():.3f} ({sub['gene_log2FC'].mean():.3f}), "
          f"L1 Δpoly(A)={sub['polya_delta'].median():.1f}nt")

groups_m6a = [gene_agg[gene_agg['m6a_quartile'] == q]['gene_log2FC'].values
              for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']]
kw_stat2, kw_p2 = stats.kruskal(*groups_m6a)
print(f"  Kruskal-Wallis P = {kw_p2:.2e}")

r_m6a, p_m6a = stats.spearmanr(gene_agg['m6a_kb_mean'], gene_agg['gene_log2FC'])
print(f"  m6A/kb ↔ gene log2FC: Spearman ρ = {r_m6a:.3f}, P = {p_m6a:.2e}")

# ====================================================================
# 7. Absolute expression change (up or down = more change)
# ====================================================================
print(f"\n{'=' * 70}")
print("Alternative: |gene log2FC| (magnitude of change)")
print("=" * 70)

gene_agg['abs_log2FC'] = gene_agg['gene_log2FC'].abs()

r_abs, p_abs = stats.spearmanr(gene_agg['polya_delta'], gene_agg['abs_log2FC'])
print(f"  L1 Δpoly(A) ↔ |gene log2FC|: Spearman ρ = {r_abs:.3f}, P = {p_abs:.2e}")

for t in q_labels:
    sub = gene_agg[gene_agg['delta_tertile'] == t]
    print(f"  {t}: |gene log2FC| median={sub['abs_log2FC'].median():.3f}, mean={sub['abs_log2FC'].mean():.3f}")

# ====================================================================
# 8. Control: genes WITHOUT intronic L1
# ====================================================================
print(f"\n{'=' * 70}")
print("Control: genes with vs without intronic L1")
print("=" * 70)

genes_with_l1 = set(gene_agg.index)
all_gene_names = set(gene_id_to_name.values())

# Get expression changes for genes without L1
no_l1_fc = []
for gname in all_gene_names - genes_with_l1:
    gid = gene_name_to_id.get(gname)
    if gid and gid in deseq_genes.index:
        row = deseq_genes.loc[gid]
        if row['baseMean'] > 50:
            no_l1_fc.append(row['log2FoldChange'])

no_l1_fc = np.array(no_l1_fc)
with_l1_fc = gene_agg['gene_log2FC'].values

print(f"  Genes with intronic L1: n={len(with_l1_fc)}, median log2FC={np.median(with_l1_fc):.4f}")
print(f"  Genes without intronic L1: n={len(no_l1_fc)}, median log2FC={np.median(no_l1_fc):.4f}")
u, p = stats.mannwhitneyu(with_l1_fc, no_l1_fc, alternative='two-sided')
print(f"  Mann-Whitney P = {p:.2e}")
print(f"  |log2FC| with L1: {np.median(np.abs(with_l1_fc)):.4f} vs without: {np.median(np.abs(no_l1_fc)):.4f}")
u2, p2 = stats.mannwhitneyu(np.abs(with_l1_fc), np.abs(no_l1_fc), alternative='two-sided')
print(f"  |log2FC| Mann-Whitney P = {p2:.2e}")

# ====================================================================
# 9. Figure
# ====================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
plt.subplots_adjust(hspace=0.35, wspace=0.35)

# (a) Scatter: L1 poly(A) delta vs host gene log2FC
ax = axes[0, 0]
ax.scatter(gene_agg['polya_delta'], gene_agg['gene_log2FC'],
           c='#8D6E63', alpha=0.4, s=20, edgecolors='white', linewidths=0.3)
# Regression line
slope, intercept, r_val, p_val, se = stats.linregress(gene_agg['polya_delta'], gene_agg['gene_log2FC'])
x_range = np.linspace(gene_agg['polya_delta'].min(), gene_agg['polya_delta'].max(), 100)
ax.plot(x_range, slope * x_range + intercept, 'r-', linewidth=1.5, alpha=0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.axvline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_xlabel('L1 poly(A) Δ (Ars - HeLa, nt)', fontsize=9)
ax.set_ylabel('Host gene log₂FC (Ars/UN)', fontsize=9)
ax.set_title(f'L1 poly(A) shortening vs\nhost gene expression (n={len(gene_agg)})', fontsize=10, fontweight='bold')
ax.text(0.05, 0.95, f'ρ = {r_s:.3f}\nP = {p_s:.2e}', transform=ax.transAxes,
        fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (b) Boxplot: host gene log2FC by L1 shortening tertile
ax = axes[0, 1]
bp_data = [gene_agg[gene_agg['delta_tertile'] == t]['gene_log2FC'].values for t in q_labels]
bp = ax.boxplot(bp_data, labels=['Strong\nshortening', 'Moderate', 'Mild/\nLengthening'],
                patch_artist=True, widths=0.6, showfliers=False)
colors_bp = ['#E53935', '#FF9800', '#4CAF50']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_ylabel('Host gene log₂FC', fontsize=9)
ax.set_title(f'Host gene expression by\nL1 shortening tertile', fontsize=10, fontweight='bold')
ax.text(0.05, 0.95, f'KW P = {kw_p:.2e}', transform=ax.transAxes,
        fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (c) m6A quartile vs host gene log2FC
ax = axes[1, 0]
bp_data2 = [gene_agg[gene_agg['m6a_quartile'] == q]['gene_log2FC'].values
            for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']]
bp2 = ax.boxplot(bp_data2, labels=['Q1\n(low m6A)', 'Q2', 'Q3', 'Q4\n(high m6A)'],
                 patch_artist=True, widths=0.6, showfliers=False)
colors_m = ['#BBDEFB', '#64B5F6', '#1E88E5', '#0D47A1']
for patch, color in zip(bp2['boxes'], colors_m):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_ylabel('Host gene log₂FC', fontsize=9)
ax.set_title(f'Host gene expression by\nL1 m6A quartile', fontsize=10, fontweight='bold')
ax.text(0.05, 0.95, f'ρ = {r_m6a:.3f}\nP = {p_m6a:.2e}', transform=ax.transAxes,
        fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (d) With L1 vs without L1 comparison
ax = axes[1, 1]
bp_data3 = [with_l1_fc, no_l1_fc]
bp3 = ax.boxplot(bp_data3, labels=['With intronic\nancient L1', 'Without\nintronic L1'],
                 patch_artist=True, widths=0.5, showfliers=False)
bp3['boxes'][0].set_facecolor('#8D6E63')
bp3['boxes'][0].set_alpha(0.7)
bp3['boxes'][1].set_facecolor('#BDBDBD')
bp3['boxes'][1].set_alpha(0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_ylabel('Gene log₂FC (Ars/UN)', fontsize=9)
ax.set_title('Gene expression change:\nwith vs without intronic L1', fontsize=10, fontweight='bold')
ax.text(0.05, 0.95, f'MW P = {p:.2e}', transform=ax.transAxes,
        fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold')

fig.savefig(OUTPUT / 'l1_host_gene_correlation.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT / 'l1_host_gene_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUTPUT / 'l1_host_gene_correlation.pdf'}")

# Save data
gene_agg.to_csv(OUTPUT / 'gene_agg_l1_polya_rnaseq.tsv', sep='\t')
mdf_out = pd.DataFrame(merged)
mdf_out.to_csv(OUTPUT / 'l1_locus_host_gene_merged.tsv', sep='\t', index=False)
print(f"Data saved: {OUTPUT}")
