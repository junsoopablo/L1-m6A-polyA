#!/usr/bin/env python3
"""
Regulatory L1 nearby gene analysis
===================================
Identify genes near/within regulatory (Enhancer/Promoter) L1 elements detected
in ChromHMM analysis, with focus on stress-related genes.

Key questions:
1. Which specific genes host or are near regulatory L1 elements?
2. Are any of these genes stress-related?
3. What are the m6A/kb and poly(A) values for these reads?
"""

import os
import sys
import re
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# Paths
# ============================================================================
BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
CHROMHMM_FILE = f'{BASE}/analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv'
RESULTS_DIR = f'{BASE}/results_group'
GTF_FILE = f'{BASE}/reference/Human.gtf'
OUT_DIR = f'{BASE}/analysis/01_exploration/topic_08_regulatory_chromatin/stress_gene_analysis'

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# Stress gene categories
# ============================================================================
STRESS_GENES = {
    'DDR': ['BRCA1', 'BRCA2', 'ATM', 'ATR', 'RAD51', 'TP53', 'CHEK1', 'CHEK2',
            'XRCC1', 'XRCC2', 'XRCC3', 'XRCC4', 'XRCC5', 'XRCC6',
            'PARP1', 'PARP2', 'PARP3', 'PARP4',
            'FANCA', 'FANCB', 'FANCC', 'FANCD2', 'FANCE', 'FANCF', 'FANCG',
            'FANCI', 'FANCL', 'FANCM',
            'BLM', 'WRN', 'NBN', 'MRE11', 'RAD50', 'H2AFX', 'MDC1', 'RNF8',
            'RNF168', 'TP53BP1', 'PALB2', 'RAD51B', 'RAD51C', 'RAD51D',
            'RAD52', 'RAD54L', 'RAD54B'],
    'Heat_shock': ['HSPA1A', 'HSPA1B', 'HSPA2', 'HSPA4', 'HSPA5', 'HSPA6',
                   'HSPA7', 'HSPA8', 'HSPA9', 'HSPA12A', 'HSPA12B', 'HSPA13', 'HSPA14',
                   'HSP90AA1', 'HSP90AB1', 'HSP90B1',
                   'HSPB1', 'HSPB2', 'HSPB3', 'HSPB6', 'HSPB7', 'HSPB8',
                   'DNAJA1', 'DNAJA2', 'DNAJA3', 'DNAJB1', 'DNAJB2', 'DNAJB4',
                   'DNAJB6', 'DNAJB9', 'DNAJB11', 'DNAJC1', 'DNAJC2', 'DNAJC3',
                   'HSF1', 'HSF2'],
    'Oxidative': ['NFE2L2', 'KEAP1', 'SOD1', 'SOD2', 'SOD3', 'CAT',
                  'GPX1', 'GPX2', 'GPX3', 'GPX4', 'GPX5', 'GPX6', 'GPX7', 'GPX8',
                  'NQO1', 'HMOX1', 'HMOX2', 'TXN', 'TXNRD1', 'TXNRD2',
                  'PRDX1', 'PRDX2', 'PRDX3', 'PRDX4', 'PRDX5', 'PRDX6',
                  'GSTP1', 'GSTM1', 'GSTT1'],
    'Apoptosis': ['BCL2', 'BAX', 'BAK1', 'MCL1', 'BCL2L1', 'BCL2L11', 'BID', 'BAD',
                  'CASP1', 'CASP2', 'CASP3', 'CASP4', 'CASP5', 'CASP6', 'CASP7',
                  'CASP8', 'CASP9', 'CASP10', 'APAF1', 'CYCS', 'DIABLO', 'XIAP',
                  'BIRC2', 'BIRC3', 'BIRC5'],
    'Autophagy': ['BECN1', 'ATG3', 'ATG4A', 'ATG4B', 'ATG5', 'ATG7', 'ATG9A',
                  'ATG10', 'ATG12', 'ATG13', 'ATG14', 'ATG16L1', 'ATG16L2',
                  'SQSTM1', 'ULK1', 'ULK2', 'MAP1LC3A', 'MAP1LC3B', 'LAMP1', 'LAMP2'],
    'UPR': ['EIF2AK3', 'ERN1', 'ATF4', 'ATF6', 'XBP1', 'DDIT3', 'HSPA5',
            'EIF2AK1', 'EIF2AK2', 'EIF2AK4', 'PDIA3', 'PDIA4', 'CALR', 'CANX'],
    'Inflammation': ['NFKB1', 'NFKB2', 'RELA', 'RELB', 'REL',
                     'IKBKA', 'IKBKB', 'IKBKG', 'IKBKE',
                     'TNFAIP1', 'TNFAIP2', 'TNFAIP3', 'TNFAIP6', 'TNFAIP8',
                     'TRAF1', 'TRAF2', 'TRAF3', 'TRAF4', 'TRAF5', 'TRAF6',
                     'IL6', 'IL1B', 'TNF', 'CXCL8'],
    'Cell_cycle': ['CDKN1A', 'CDKN2A', 'CDKN1B', 'CDKN2B', 'RB1',
                   'GADD45A', 'GADD45B', 'GADD45G',
                   'TP53', 'MDM2', 'MDM4', 'CCND1', 'CCNE1', 'CDK4', 'CDK6'],
    'Stress_kinase': ['MAPK8', 'MAPK9', 'MAPK10', 'MAPK14', 'MAPK11', 'MAPK12', 'MAPK13',
                      'MAP3K5', 'MAP2K4', 'MAP2K7', 'MAP2K3', 'MAP2K6',
                      'HIF1A', 'EPAS1', 'ARNT', 'VHL'],
    'RNA_stress': ['G3BP1', 'G3BP2', 'TIA1', 'TIAL1',
                   'EIF2S1', 'EIF4E', 'EIF4G1', 'PABPC1',
                   'DCP1A', 'DCP1B', 'DCP2', 'XRN1', 'XRN2',
                   'EDC3', 'EDC4', 'LSM1', 'DDX6'],
    'TE_defense': ['MOV10', 'SAMHD1', 'ZAP', 'ZC3HAV1', 'ADAR', 'ADARB1',
                   'APOBEC3A', 'APOBEC3B', 'APOBEC3C', 'APOBEC3F', 'APOBEC3G',
                   'TRIM28', 'SETDB1', 'MORC2', 'TASOR', 'MPP8', 'PPHLN1',
                   'DNMT1', 'DNMT3A', 'DNMT3B', 'DNMT3L',
                   'PIWIL1', 'PIWIL2', 'PIWIL4', 'MAEL'],
    'm6A_machinery': ['METTL3', 'METTL14', 'WTAP', 'RBM15', 'RBM15B', 'VIRMA',
                      'YTHDC1', 'YTHDC2', 'YTHDF1', 'YTHDF2', 'YTHDF3',
                      'FTO', 'ALKBH5', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3',
                      'HNRNPA2B1', 'HNRNPC'],
    'Psi_machinery': ['DKC1', 'PUS1', 'PUS3', 'PUS7', 'PUS7L', 'PUS10',
                      'TRUB1', 'TRUB2', 'RPUSD1', 'RPUSD2', 'RPUSD3', 'RPUSD4'],
}

# Build flat lookup: gene_name -> set of categories
STRESS_LOOKUP = {}
for cat, genes in STRESS_GENES.items():
    for g in genes:
        if g not in STRESS_LOOKUP:
            STRESS_LOOKUP[g] = set()
        STRESS_LOOKUP[g].add(cat)

# Also build pattern-based matches (prefix matching)
STRESS_PREFIXES = {
    'HSPA': 'Heat_shock', 'HSPB': 'Heat_shock', 'HSP90': 'Heat_shock',
    'DNAJ': 'Heat_shock',
    'XRCC': 'DDR', 'PARP': 'DDR', 'FANC': 'DDR', 'RAD5': 'DDR',
    'SOD': 'Oxidative', 'GPX': 'Oxidative', 'PRDX': 'Oxidative',
    'CASP': 'Apoptosis', 'BCL2': 'Apoptosis',
    'ATG': 'Autophagy',
    'NFKB': 'Inflammation', 'IKBK': 'Inflammation', 'TNFAIP': 'Inflammation',
    'TRAF': 'Inflammation',
    'GADD45': 'Cell_cycle', 'CDKN': 'Cell_cycle',
    'MAPK': 'Stress_kinase',
    'APOBEC': 'TE_defense', 'DNMT': 'TE_defense',
    'YTHD': 'm6A_machinery', 'IGF2BP': 'm6A_machinery',
    'PUS': 'Psi_machinery', 'TRUB': 'Psi_machinery', 'RPUSD': 'Psi_machinery',
}


def get_stress_categories(gene_name):
    """Return set of stress categories for a gene name."""
    if pd.isna(gene_name) or gene_name == '' or gene_name == 'intergenic':
        return set()
    cats = set()
    # Direct lookup
    if gene_name in STRESS_LOOKUP:
        cats.update(STRESS_LOOKUP[gene_name])
    # Prefix matching
    for prefix, cat in STRESS_PREFIXES.items():
        if gene_name.startswith(prefix):
            cats.add(cat)
    return cats


# ============================================================================
# Step 1: Load ChromHMM-annotated L1 reads
# ============================================================================
print("=" * 80)
print("Step 1: Loading ChromHMM-annotated L1 reads")
print("=" * 80)

chromhmm = pd.read_csv(CHROMHMM_FILE, sep='\t')
print(f"Total ChromHMM-annotated reads: {len(chromhmm):,}")

# Filter for regulatory
reg = chromhmm[chromhmm['chromhmm_group'].isin(['Enhancer', 'Promoter'])].copy()
print(f"Regulatory (Enhancer + Promoter) reads: {len(reg):,}")
print(f"  Enhancer: {(reg['chromhmm_group'] == 'Enhancer').sum():,}")
print(f"  Promoter: {(reg['chromhmm_group'] == 'Promoter').sum():,}")
print(f"  Cell lines: {reg['cellline'].nunique()}")
print(f"  Conditions: {reg['condition'].value_counts().to_dict()}")

# ============================================================================
# Step 2: Load L1 summaries and merge to get host gene info
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: Loading L1 summary files for host gene info")
print("=" * 80)

groups = sorted([d for d in os.listdir(RESULTS_DIR)
                 if os.path.isdir(os.path.join(RESULTS_DIR, d))])

summary_dfs = []
for grp in groups:
    summary_file = os.path.join(RESULTS_DIR, grp, 'g_summary', f'{grp}_L1_summary.tsv')
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file, sep='\t',
                         usecols=['read_id', 'chr', 'start', 'end', 'read_length',
                                  'gene_id', 'te_start', 'te_end',
                                  'overlapping_genes', 'TE_group', 'read_strand'])
        df['group_name'] = grp
        summary_dfs.append(df)

summary = pd.concat(summary_dfs, ignore_index=True)
print(f"Total L1 summary reads loaded: {len(summary):,} from {len(summary_dfs)} groups")

# Merge chromhmm regulatory reads with summary
reg_merged = reg.merge(summary[['read_id', 'overlapping_genes', 'read_length',
                                 'te_start', 'te_end', 'read_strand']],
                       on='read_id', how='left')
print(f"Merged regulatory reads: {len(reg_merged):,}")
print(f"  With host gene (intronic): {(reg_merged['genomic_context'] == 'intronic').sum():,}")
print(f"  Intergenic: {(reg_merged['genomic_context'] == 'intergenic').sum():,}")

# ============================================================================
# Step 3: For intergenic reads, find nearest gene using GTF
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: Finding nearest genes for intergenic regulatory L1")
print("=" * 80)

# Parse gene-level records from GTF
print("Parsing GENCODE v38 GTF for gene coordinates...")
gene_records = []
with open(GTF_FILE) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if fields[2] != 'gene':
            continue
        chrom = fields[0]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        attrs = fields[8]
        # Extract gene_name
        m = re.search(r'gene_name "([^"]+)"', attrs)
        gene_name = m.group(1) if m else 'unknown'
        m2 = re.search(r'gene_type "([^"]+)"', attrs)
        gene_type = m2.group(1) if m2 else 'unknown'
        gene_records.append({
            'chr': chrom, 'gene_start': start, 'gene_end': end,
            'gene_strand': strand, 'gene_name': gene_name, 'gene_type': gene_type
        })

genes_df = pd.DataFrame(gene_records)
print(f"Loaded {len(genes_df):,} genes from GTF")

# For intergenic reads, find nearest gene
intergenic_mask = reg_merged['genomic_context'] == 'intergenic'
intergenic_reads = reg_merged[intergenic_mask].copy()
print(f"Finding nearest gene for {len(intergenic_reads):,} intergenic regulatory L1 reads...")

# Build a per-chromosome lookup for faster nearest-gene search
genes_by_chr = {}
for chrom, gdf in genes_df.groupby('chr'):
    starts = gdf['gene_start'].values
    ends = gdf['gene_end'].values
    names = gdf['gene_name'].values
    types = gdf['gene_type'].values
    # Sort by start
    idx = np.argsort(starts)
    genes_by_chr[chrom] = {
        'starts': starts[idx],
        'ends': ends[idx],
        'names': names[idx],
        'types': types[idx],
    }

nearest_genes = []
nearest_distances = []
nearest_gene_types = []

for _, row in intergenic_reads.iterrows():
    chrom = row['chr']
    read_mid = (row['start'] + row['end']) / 2
    if chrom not in genes_by_chr:
        nearest_genes.append('unknown')
        nearest_distances.append(np.nan)
        nearest_gene_types.append('unknown')
        continue
    gdata = genes_by_chr[chrom]
    # Distance to each gene: min of abs(mid - gene_start), abs(mid - gene_end)
    d_start = np.abs(gdata['starts'] - read_mid)
    d_end = np.abs(gdata['ends'] - read_mid)
    dists = np.minimum(d_start, d_end)
    # For genes that overlap the midpoint, distance = 0
    overlap = (gdata['starts'] <= read_mid) & (gdata['ends'] >= read_mid)
    dists[overlap] = 0
    best_idx = np.argmin(dists)
    nearest_genes.append(gdata['names'][best_idx])
    nearest_distances.append(int(dists[best_idx]))
    nearest_gene_types.append(gdata['types'][best_idx])

intergenic_reads = intergenic_reads.copy()
intergenic_reads['nearest_gene'] = nearest_genes
intergenic_reads['nearest_gene_dist'] = nearest_distances
intergenic_reads['nearest_gene_type'] = nearest_gene_types

print(f"  Median distance to nearest gene: {intergenic_reads['nearest_gene_dist'].median():,.0f} bp")
print(f"  < 10kb from gene: {(intergenic_reads['nearest_gene_dist'] < 10000).sum():,}")
print(f"  < 50kb from gene: {(intergenic_reads['nearest_gene_dist'] < 50000).sum():,}")

# ============================================================================
# Step 4: Compile master table
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Compiling master regulatory L1 table")
print("=" * 80)

# For intronic reads, the host gene is overlapping_genes
# For intergenic reads, use nearest gene
reg_merged['host_gene'] = reg_merged['overlapping_genes']
reg_merged['gene_source'] = 'intronic'
reg_merged.loc[intergenic_mask, 'gene_source'] = 'nearest'

# Update intergenic host_gene with nearest gene
for idx in intergenic_reads.index:
    reg_merged.loc[idx, 'host_gene'] = intergenic_reads.loc[idx, 'nearest_gene']
    reg_merged.loc[idx, 'nearest_gene_dist'] = intergenic_reads.loc[idx, 'nearest_gene_dist']
    reg_merged.loc[idx, 'nearest_gene_type'] = intergenic_reads.loc[idx, 'nearest_gene_type']

# Fill missing nearest_gene_dist for intronic (=0)
reg_merged['nearest_gene_dist'] = reg_merged.get('nearest_gene_dist', np.nan)
reg_merged.loc[reg_merged['gene_source'] == 'intronic', 'nearest_gene_dist'] = 0

# Classify stress categories
reg_merged['stress_categories'] = reg_merged['host_gene'].apply(
    lambda g: ';'.join(sorted(get_stress_categories(g))) if get_stress_categories(g) else ''
)
reg_merged['is_stress_gene'] = reg_merged['stress_categories'] != ''

print(f"Total regulatory L1 reads: {len(reg_merged):,}")
print(f"  With identified gene: {reg_merged['host_gene'].notna().sum():,}")
print(f"  In stress-related genes: {reg_merged['is_stress_gene'].sum():,}")

# ============================================================================
# Step 5: Analysis
# ============================================================================
print("\n" + "=" * 80)
print("Step 5a: All genes with regulatory L1 reads")
print("=" * 80)

# Gene-level stats
gene_stats = []
for gene, gdf in reg_merged.groupby('host_gene'):
    if pd.isna(gene):
        continue
    n_total = len(gdf)
    n_hela = len(gdf[(gdf['cellline'] == 'HeLa') & (gdf['condition'] == 'normal')])
    n_ars = len(gdf[gdf['cellline'] == 'HeLa-Ars'])
    n_other = n_total - n_hela - n_ars

    # Poly(A)
    polya_hela = gdf.loc[(gdf['cellline'] == 'HeLa') & (gdf['condition'] == 'normal'), 'polya_length']
    polya_ars = gdf.loc[gdf['cellline'] == 'HeLa-Ars', 'polya_length']
    polya_all = gdf['polya_length']

    # m6A
    m6a_all = gdf['m6a_per_kb']

    # Chromatin
    n_enh = (gdf['chromhmm_group'] == 'Enhancer').sum()
    n_prom = (gdf['chromhmm_group'] == 'Promoter').sum()

    # L1 subfamilies
    subfamilies = gdf['gene_id'].value_counts().to_dict()
    top_subfamily = gdf['gene_id'].mode().iloc[0] if len(gdf) > 0 else ''
    n_young = (gdf['l1_age'] == 'young').sum()
    n_ancient = (gdf['l1_age'] == 'ancient').sum()

    # Cell lines
    cell_lines = sorted(gdf['cellline'].unique())

    # Genomic context
    n_intronic = (gdf['genomic_context'] == 'intronic').sum()

    # Stress
    cats = get_stress_categories(gene)

    gene_stats.append({
        'gene': gene,
        'n_reads': n_total,
        'n_HeLa': n_hela,
        'n_HeLa_Ars': n_ars,
        'n_other_CL': n_other,
        'n_enhancer': n_enh,
        'n_promoter': n_prom,
        'median_polya': polya_all.median(),
        'median_polya_HeLa': polya_hela.median() if len(polya_hela) > 0 else np.nan,
        'median_polya_Ars': polya_ars.median() if len(polya_ars) > 0 else np.nan,
        'delta_polya': (polya_ars.median() - polya_hela.median()) if len(polya_hela) > 0 and len(polya_ars) > 0 else np.nan,
        'median_m6a_per_kb': m6a_all.median(),
        'n_young': n_young,
        'n_ancient': n_ancient,
        'top_subfamily': top_subfamily,
        'n_celllines': len(cell_lines),
        'celllines': ','.join(cell_lines),
        'n_intronic': n_intronic,
        'stress_categories': ';'.join(sorted(cats)) if cats else '',
        'is_stress': bool(cats),
    })

gene_stats_df = pd.DataFrame(gene_stats).sort_values('n_reads', ascending=False)
print(f"Total unique genes with regulatory L1: {len(gene_stats_df):,}")
print(f"Stress-related genes: {gene_stats_df['is_stress'].sum():,}")

# Top genes by read count
print(f"\n--- Top 30 genes by regulatory L1 read count ---")
cols_show = ['gene', 'n_reads', 'n_HeLa', 'n_HeLa_Ars', 'n_other_CL',
             'n_enhancer', 'n_promoter', 'median_polya', 'median_m6a_per_kb',
             'n_young', 'n_ancient', 'stress_categories']
top30 = gene_stats_df.head(30)[cols_show]
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 40)
print(top30.to_string(index=False))

# ============================================================================
print("\n" + "=" * 80)
print("Step 5b: Stress-related genes in regulatory L1")
print("=" * 80)

stress_df = gene_stats_df[gene_stats_df['is_stress']].copy()
if len(stress_df) > 0:
    print(f"\n{len(stress_df)} stress-related genes found:")
    print(stress_df[['gene', 'n_reads', 'n_HeLa', 'n_HeLa_Ars', 'n_enhancer',
                     'n_promoter', 'median_polya', 'median_m6a_per_kb',
                     'delta_polya', 'stress_categories']].to_string(index=False))

    # Breakdown by stress category
    print(f"\n--- Stress category breakdown ---")
    cat_counts = defaultdict(int)
    cat_reads = defaultdict(int)
    for _, row in stress_df.iterrows():
        for cat in row['stress_categories'].split(';'):
            if cat:
                cat_counts[cat] += 1
                cat_reads[cat] += row['n_reads']
    for cat in sorted(cat_counts, key=lambda x: -cat_reads[x]):
        print(f"  {cat:20s}: {cat_counts[cat]:3d} genes, {cat_reads[cat]:5d} reads")
else:
    print("No stress-related genes found in regulatory L1 regions.")

# ============================================================================
print("\n" + "=" * 80)
print("Step 5c: Regulatory L1 reads: HeLa vs HeLa-Ars comparison")
print("=" * 80)

# Subset to HeLa (normal) and HeLa-Ars (stress)
# NOTE: cellline='HeLa-Ars' with condition='stress' represents arsenite-treated HeLa
hela_norm_reg = reg_merged[(reg_merged['cellline'] == 'HeLa') & (reg_merged['condition'] == 'normal')].copy()
hela_ars_reg = reg_merged[reg_merged['cellline'] == 'HeLa-Ars'].copy()
print(f"HeLa regulatory L1 reads:")
print(f"  Normal (HeLa): {len(hela_norm_reg):,}")
print(f"  Arsenite (HeLa-Ars): {len(hela_ars_reg):,}")

for cond_name, sub in [('HeLa Normal', hela_norm_reg), ('HeLa-Ars Stress', hela_ars_reg)]:
    print(f"\n  {cond_name} (n={len(sub):,}):")
    if len(sub) > 0:
        print(f"    Enhancer: {(sub['chromhmm_group'] == 'Enhancer').sum():,}")
        print(f"    Promoter: {(sub['chromhmm_group'] == 'Promoter').sum():,}")
        print(f"    Median poly(A): {sub['polya_length'].median():.1f} nt")
        print(f"    Median m6A/kb: {sub['m6a_per_kb'].median():.2f}")
        print(f"    Young: {(sub['l1_age'] == 'young').sum():,}")
        print(f"    Intronic: {(sub['genomic_context'] == 'intronic').sum():,}")
        print(f"    Stress genes: {sub['is_stress_gene'].sum():,}")
        # List genes
        genes_in_cond = sub['host_gene'].value_counts()
        print(f"    Top genes: {', '.join(f'{g}({n})' for g, n in genes_in_cond.head(10).items())}")

# Delta poly(A) for regulatory
hela_norm_polya = hela_norm_reg['polya_length']
hela_ars_polya = hela_ars_reg['polya_length']
if len(hela_norm_polya) > 0 and len(hela_ars_polya) > 0:
    delta = hela_ars_polya.median() - hela_norm_polya.median()
    u_stat, u_p = stats.mannwhitneyu(hela_ars_polya, hela_norm_polya, alternative='two-sided')
    print(f"\n  Regulatory L1 poly(A) delta (Ars - Normal): {delta:.1f} nt (U-test p={u_p:.2e})")

# By chromhmm_group
hela_both = pd.concat([hela_norm_reg.assign(is_ars=False), hela_ars_reg.assign(is_ars=True)])
for grp in ['Enhancer', 'Promoter']:
    n_sub = hela_both[hela_both['chromhmm_group'] == grp]
    norm = n_sub[~n_sub['is_ars']]['polya_length']
    ars = n_sub[n_sub['is_ars']]['polya_length']
    if len(norm) > 0 and len(ars) > 0:
        delta = ars.median() - norm.median()
        _, p = stats.mannwhitneyu(ars, norm, alternative='two-sided')
        print(f"    {grp}: Normal {norm.median():.1f} -> Ars {ars.median():.1f}, "
              f"delta={delta:.1f} nt (p={p:.2e}, n_norm={len(norm)}, n_ars={len(ars)})")

# ============================================================================
print("\n" + "=" * 80)
print("Step 5d: Stress gene reads — per-read detail")
print("=" * 80)

stress_reads = reg_merged[reg_merged['is_stress_gene']].copy()
if len(stress_reads) > 0:
    print(f"\n{len(stress_reads)} reads in stress-related genes:")
    display_cols = ['read_id', 'host_gene', 'stress_categories', 'chromhmm_group',
                    'cellline', 'condition', 'polya_length', 'm6a_per_kb',
                    'gene_id', 'l1_age', 'genomic_context']
    # Show only columns that exist
    display_cols = [c for c in display_cols if c in stress_reads.columns]
    print(stress_reads[display_cols].sort_values('host_gene').to_string(index=False))

# ============================================================================
print("\n" + "=" * 80)
print("Step 5e: Gene category summary — regulatory L1")
print("=" * 80)

# Total reads by gene presence
total_reads = len(reg_merged)
stress_gene_reads = reg_merged['is_stress_gene'].sum()
non_stress_reads = total_reads - stress_gene_reads

print(f"Total regulatory L1 reads: {total_reads:,}")
print(f"  In stress genes: {stress_gene_reads:,} ({100*stress_gene_reads/total_reads:.1f}%)")
print(f"  In non-stress genes: {non_stress_reads:,} ({100*non_stress_reads/total_reads:.1f}%)")

# Intronic vs intergenic split
intronic = reg_merged[reg_merged['genomic_context'] == 'intronic']
intergenic = reg_merged[reg_merged['genomic_context'] == 'intergenic']
print(f"\n  Intronic: {len(intronic):,} ({100*len(intronic)/total_reads:.1f}%)")
print(f"    In stress genes: {intronic['is_stress_gene'].sum():,}")
print(f"  Intergenic: {len(intergenic):,} ({100*len(intergenic)/total_reads:.1f}%)")
print(f"    Near stress genes: {intergenic['is_stress_gene'].sum():,}")

# ============================================================================
print("\n" + "=" * 80)
print("Step 5f: Regulatory vs non-regulatory L1 in same genes")
print("=" * 80)

# Get all non-regulatory reads in genes that have regulatory L1
reg_genes = set(gene_stats_df['gene'].dropna().unique())
all_chromhmm = chromhmm.copy()
all_chromhmm_with_genes = all_chromhmm.merge(
    summary[['read_id', 'overlapping_genes']], on='read_id', how='left')

# Non-regulatory reads in regulatory genes
non_reg_in_genes = all_chromhmm_with_genes[
    (~all_chromhmm_with_genes['chromhmm_group'].isin(['Enhancer', 'Promoter'])) &
    (all_chromhmm_with_genes['overlapping_genes'].isin(reg_genes))
].copy()

print(f"Non-regulatory L1 reads in genes that also have regulatory L1: {len(non_reg_in_genes):,}")

# Compare poly(A) and m6A
if len(non_reg_in_genes) > 0:
    reg_polya = reg_merged['polya_length']
    nonreg_polya = non_reg_in_genes['polya_length']
    reg_m6a = reg_merged['m6a_per_kb']
    nonreg_m6a = non_reg_in_genes['m6a_per_kb']

    print(f"\n  Regulatory L1 in these genes:")
    print(f"    Poly(A): median={reg_polya.median():.1f}, mean={reg_polya.mean():.1f}")
    print(f"    m6A/kb:  median={reg_m6a.median():.2f}, mean={reg_m6a.mean():.2f}")
    print(f"  Non-regulatory L1 in same genes:")
    print(f"    Poly(A): median={nonreg_polya.median():.1f}, mean={nonreg_polya.mean():.1f}")
    print(f"    m6A/kb:  median={nonreg_m6a.median():.2f}, mean={nonreg_m6a.mean():.2f}")

    _, p_polya = stats.mannwhitneyu(reg_polya, nonreg_polya, alternative='two-sided')
    _, p_m6a = stats.mannwhitneyu(reg_m6a, nonreg_m6a, alternative='two-sided')
    print(f"  MWU poly(A): p={p_polya:.2e}")
    print(f"  MWU m6A/kb:  p={p_m6a:.2e}")

# ============================================================================
print("\n" + "=" * 80)
print("Step 5g: Cross-cell-line regulatory L1 gene overlap")
print("=" * 80)

# Genes with regulatory L1 per cell line
cl_genes = {}
for cl, cdf in reg_merged.groupby('cellline'):
    cl_genes[cl] = set(cdf['host_gene'].dropna().unique())
    print(f"  {cl:15s}: {len(cl_genes[cl]):4d} genes with regulatory L1")

# Shared across all
if cl_genes:
    all_cls = list(cl_genes.keys())
    shared = cl_genes[all_cls[0]]
    for cl in all_cls[1:]:
        shared = shared & cl_genes[cl]
    print(f"\n  Genes with regulatory L1 in ALL {len(all_cls)} cell lines: {len(shared)}")
    if shared:
        for g in sorted(shared):
            n = reg_merged[reg_merged['host_gene'] == g].shape[0]
            cats = get_stress_categories(g)
            cats_str = f" [{';'.join(sorted(cats))}]" if cats else ""
            print(f"    {g} (n={n}){cats_str}")

# ============================================================================
print("\n" + "=" * 80)
print("Step 5h: Intergenic regulatory L1 — distance distribution")
print("=" * 80)

intergenic_with_dist = reg_merged[
    (reg_merged['gene_source'] == 'nearest') &
    reg_merged['nearest_gene_dist'].notna()
].copy()

if len(intergenic_with_dist) > 0:
    dists = intergenic_with_dist['nearest_gene_dist']
    print(f"Intergenic regulatory L1 reads: {len(intergenic_with_dist):,}")
    print(f"  Distance to nearest gene:")
    print(f"    Median: {dists.median():,.0f} bp")
    print(f"    Mean:   {dists.mean():,.0f} bp")
    print(f"    <1kb:   {(dists < 1000).sum():,}")
    print(f"    <5kb:   {(dists < 5000).sum():,}")
    print(f"    <10kb:  {(dists < 10000).sum():,}")
    print(f"    <50kb:  {(dists < 50000).sum():,}")
    print(f"    >100kb: {(dists > 100000).sum():,}")

    # Top intergenic regulatory L1 genes
    print(f"\n  Top 20 nearest genes for intergenic regulatory L1:")
    ig_gene_stats = intergenic_with_dist.groupby('host_gene').agg(
        n=('read_id', 'count'),
        median_dist=('nearest_gene_dist', 'median'),
        median_polya=('polya_length', 'median'),
        median_m6a=('m6a_per_kb', 'median'),
    ).sort_values('n', ascending=False)
    for gene, row in ig_gene_stats.head(20).iterrows():
        cats = get_stress_categories(gene)
        cats_str = f" [{';'.join(sorted(cats))}]" if cats else ""
        print(f"    {gene:20s} n={row['n']:3.0f}  dist={row['median_dist']:8,.0f}bp  "
              f"polyA={row['median_polya']:.1f}  m6A/kb={row['median_m6a']:.2f}{cats_str}")

# ============================================================================
# Save results
# ============================================================================
print("\n" + "=" * 80)
print("Saving results")
print("=" * 80)

# 1. Per-read table
per_read_cols = ['read_id', 'chr', 'start', 'end', 'gene_id', 'l1_age',
                 'polya_length', 'm6a_per_kb', 'm6a_sites_high',
                 'cellline', 'condition', 'genomic_context',
                 'chromhmm_state', 'chromhmm_group', 'group', 'sample',
                 'host_gene', 'gene_source', 'nearest_gene_dist',
                 'stress_categories', 'is_stress_gene']
per_read_cols = [c for c in per_read_cols if c in reg_merged.columns]
out1 = os.path.join(OUT_DIR, 'regulatory_l1_per_read.tsv')
reg_merged[per_read_cols].to_csv(out1, sep='\t', index=False)
print(f"Saved: {out1} ({len(reg_merged):,} reads)")

# 2. Gene-level stats
out2 = os.path.join(OUT_DIR, 'regulatory_l1_genes.tsv')
gene_stats_df.to_csv(out2, sep='\t', index=False)
print(f"Saved: {out2} ({len(gene_stats_df):,} genes)")

# 3. Stress gene per-read
out3 = os.path.join(OUT_DIR, 'stress_gene_per_read.tsv')
stress_reads_out = reg_merged[reg_merged['is_stress_gene']]
stress_reads_out_cols = [c for c in per_read_cols if c in stress_reads_out.columns]
stress_reads_out[stress_reads_out_cols].to_csv(out3, sep='\t', index=False)
print(f"Saved: {out3} ({len(stress_reads_out):,} reads)")

# 4. Stress gene summary
out4 = os.path.join(OUT_DIR, 'stress_gene_stats.tsv')
if len(stress_df) > 0:
    stress_df.to_csv(out4, sep='\t', index=False)
    print(f"Saved: {out4} ({len(stress_df):,} genes)")

# 5. All host gene stats
out5 = os.path.join(OUT_DIR, 'all_host_gene_stats.tsv')
gene_stats_df.to_csv(out5, sep='\t', index=False)
print(f"Saved: {out5} ({len(gene_stats_df):,} genes)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Regulatory L1 Analysis Summary
===============================
Total regulatory L1 reads: {len(reg_merged):,}
  Enhancer: {(reg_merged['chromhmm_group'] == 'Enhancer').sum():,}
  Promoter: {(reg_merged['chromhmm_group'] == 'Promoter').sum():,}

Unique genes hosting/near regulatory L1: {len(gene_stats_df):,}
  Intronic (host gene): {len(reg_merged[reg_merged['gene_source'] == 'intronic']):,} reads
  Intergenic (nearest gene): {len(reg_merged[reg_merged['gene_source'] == 'nearest']):,} reads

Stress-related genes: {gene_stats_df['is_stress'].sum():,} ({100*gene_stats_df['is_stress'].sum()/len(gene_stats_df):.1f}%)
Stress gene reads: {reg_merged['is_stress_gene'].sum():,} ({100*reg_merged['is_stress_gene'].sum()/len(reg_merged):.1f}%)

Cell lines: {reg_merged['cellline'].nunique()}
HeLa normal / HeLa-Ars: {len(reg_merged[(reg_merged['cellline']=='HeLa') & (reg_merged['condition']=='normal')]):,} / {len(reg_merged[reg_merged['cellline']=='HeLa-Ars']):,}
""")

print("Done!")
