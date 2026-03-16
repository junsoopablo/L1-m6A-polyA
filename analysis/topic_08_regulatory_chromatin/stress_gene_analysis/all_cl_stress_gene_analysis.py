#!/usr/bin/env python3
"""
Stress-related host gene analysis across ALL 29 sample groups (11 cell lines).

Identifies L1 elements residing within stress-related genes and characterizes them
by read count, cell line coverage, L1 age, m6A density, poly(A) length, and
arsenite response.
"""

import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
from pathlib import Path

# ==============================================================================
# Configuration
# ==============================================================================

BASE_DIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
SUMMARY_DIR = os.path.join(BASE_DIR, "results_group")
CACHE_DIR = os.path.join(BASE_DIR, "analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache")
OUT_DIR = os.path.join(BASE_DIR, "analysis/01_exploration/topic_08_regulatory_chromatin/stress_gene_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = [
    "A549_4", "A549_5", "A549_6",
    "H9_2", "H9_3", "H9_4",
    "Hct116_3", "Hct116_4",
    "HeLa_1", "HeLa_2", "HeLa_3",
    "HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3",
    "HepG2_5", "HepG2_6",
    "HEYA8_1", "HEYA8_2", "HEYA8_3",
    "K562_4", "K562_5", "K562_6",
    "MCF7_2", "MCF7_3", "MCF7_4",
    "MCF7-EV_1",
    "SHSY5Y_1", "SHSY5Y_2", "SHSY5Y_3",
]

YOUNG_PREFIXES = ("L1HS", "L1PA1", "L1PA2", "L1PA3")

# Stress gene categories with gene names/prefixes
# For families (FANC*, HSP*, etc.), we use prefix matching
STRESS_GENES = {
    "DDR/DNA repair": [
        "BRCA1", "BRCA2", "ATM", "ATR", "RAD51", "RAD50", "TP53",
        "CHEK1", "CHEK2", "XRCC", "PARP", "FANC",  # prefix match
        "NBN", "MRE11", "BLM", "WRN", "ERCC", "XPC", "XPA",
        "PALB2", "BRIP1", "SMARCAL1", "ZRANB3", "ATRX", "RB1",
    ],
    "Heat shock/chaperone": [
        "HSPA", "HSP90", "HSPB", "HSPH", "DNAJ", "HSF1", "HSF2",
        "BAG3", "SERPINH1",
    ],
    "Oxidative stress": [
        "NFE2L2", "KEAP1", "SOD1", "SOD2", "CAT", "GPX",
        "NQO1", "HMOX1", "TXN", "PRDX",
    ],
    "Apoptosis": [
        "BCL2", "BAX", "BAK1", "MCL1", "BID", "CASP",
        "APAF1", "XIAP", "BIRC",
    ],
    "Autophagy": [
        "BECN1", "ATG", "SQSTM1", "MAP1LC3B", "BNIP3", "ULK1",
    ],
    "UPR/ER stress": [
        "EIF2AK3", "ERN1", "ATF4", "ATF6", "XBP1", "DDIT3", "HSPA5",
    ],
    "Inflammation/NF-kB": [
        "NFKB", "RELA", "IKBK", "TNFAIP", "TRAF",
    ],
    "Cell cycle checkpoint": [
        "CDKN1A", "CDKN2A", "RB1", "CDC25", "GADD45",
    ],
    "Stress kinase": [
        "MAPK8", "MAPK14", "MAP3K5", "HIF1A", "EGLN", "VHL", "EPAS1",
    ],
    "RNA stress": [
        "G3BP1", "G3BP2", "TIA1", "TIAL1", "EIF2AK2",
    ],
    "Cancer/tumor suppressor": [
        "APC", "PTEN", "NF1", "NF2", "VHL", "WT1", "SMAD4", "STK11",
    ],
    "MIL (Xiong 2021)": [
        "ZRANB3", "SMARCAL1", "ATR", "ATRX", "FANCC", "FANCD2",
        "FANCI", "BRIP1", "SPIDR", "ERCC6L2",
    ],
    "m6A machinery": [
        "METTL3", "METTL14", "WTAP", "VIRMA", "ALKBH5", "FTO",
        "YTHDF1", "YTHDF2", "YTHDF3", "YTHDC1", "YTHDC2",
    ],
    "L1 restriction": [
        "MOV10", "SAMHD1", "APOBEC3", "ADAR", "ZC3HAV1",
        "TREX1", "RNASEH2",
    ],
}

# Xiong 2021 MIL gene list for flagging
MIL_GENES = {"ZRANB3", "SMARCAL1", "ATR", "ATRX", "FANCC", "FANCD2",
             "FANCI", "BRIP1", "SPIDR", "ERCC6L2"}


def get_cell_line(group):
    """Extract cell line name from group (e.g., HeLa_1 -> HeLa)."""
    # Handle MCF7-EV, HeLa-Ars specially
    if group.startswith("MCF7-EV"):
        return "MCF7-EV"
    elif group.startswith("HeLa-Ars"):
        return "HeLa-Ars"
    else:
        return re.sub(r'_\d+$', '', group)


def classify_gene(gene_name):
    """
    Classify a gene into stress categories.
    Returns list of (category, matched_pattern) tuples.
    Uses prefix matching for family genes.
    """
    matches = []
    gene_upper = gene_name.upper()

    for category, genes in STRESS_GENES.items():
        for pattern in genes:
            pattern_upper = pattern.upper()
            # Exact match first
            if gene_upper == pattern_upper:
                matches.append((category, pattern))
                break
            # Prefix match for gene families (e.g., FANC* matches FANCC, FANCD2)
            # Only do prefix match for known family patterns
            if gene_upper.startswith(pattern_upper) and len(pattern) >= 3:
                # Avoid overly broad matches: ATG should match ATG5 but not ATGR
                # CAT should match CAT exactly but not CATSPER
                # Check: if pattern is a known exact gene (like CAT, VHL, BID),
                # only do exact match
                exact_only = {"CAT", "VHL", "BID", "BLM", "WRN", "NBN",
                              "BAX", "FTO", "RB1", "APC", "NF1", "NF2",
                              "WT1", "XPA", "XPC", "SOD1", "SOD2", "NQO1",
                              "TIA1", "RELA", "BAG3"}
                if pattern_upper in {x.upper() for x in exact_only}:
                    continue  # skip prefix match for exact-only genes
                matches.append((category, pattern))
                break

    # Check MIL membership
    is_mil = gene_name.upper() in {x.upper() for x in MIL_GENES}

    return matches, is_mil


# ==============================================================================
# Step 1: Load all L1 summary files
# ==============================================================================

print("=" * 80)
print("STEP 1: Loading L1 summary files from 29 groups")
print("=" * 80)

all_summaries = []
for group in GROUPS:
    fpath = os.path.join(SUMMARY_DIR, group, "g_summary", f"{group}_L1_summary.tsv")
    if not os.path.exists(fpath):
        print(f"  WARNING: Missing {fpath}")
        continue
    df = pd.read_csv(fpath, sep='\t')
    df['group'] = group
    df['cell_line'] = get_cell_line(group)
    all_summaries.append(df)
    print(f"  {group}: {len(df)} reads ({(df['qc_tag']=='PASS').sum()} PASS)")

summary = pd.concat(all_summaries, ignore_index=True)
print(f"\nTotal reads loaded: {len(summary)}")

# Filter for PASS
summary = summary[summary['qc_tag'] == 'PASS'].copy()
print(f"After PASS filter: {len(summary)}")

# Determine L1 age
summary['l1_age'] = summary['gene_id'].apply(
    lambda x: 'young' if any(x.startswith(p) for p in YOUNG_PREFIXES) else 'ancient'
)

# Identify intronic reads (has overlapping_genes)
summary['is_intronic'] = summary['overlapping_genes'].notna() & (summary['overlapping_genes'] != '')
intronic = summary[summary['is_intronic']].copy()
print(f"Intronic reads (with host gene): {len(intronic)}")
print(f"Intergenic reads: {len(summary) - len(intronic)}")
print(f"Young L1: {(summary['l1_age']=='young').sum()}, Ancient: {(summary['l1_age']=='ancient').sum()}")

# ==============================================================================
# Step 2: Load Part3 m6A cache and merge
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 2: Loading Part3 m6A cache files")
print("=" * 80)

all_cache = []
for group in GROUPS:
    fpath = os.path.join(CACHE_DIR, f"{group}_l1_per_read.tsv")
    if not os.path.exists(fpath):
        print(f"  WARNING: Missing cache for {group}")
        continue
    df = pd.read_csv(fpath, sep='\t')
    all_cache.append(df[['read_id', 'read_length', 'm6a_sites_high']])

cache = pd.concat(all_cache, ignore_index=True)
cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)
print(f"Cache loaded: {len(cache)} reads")

# Merge m6A data into intronic reads
intronic = intronic.merge(
    cache[['read_id', 'm6a_per_kb', 'm6a_sites_high']],
    on='read_id', how='left'
)
m6a_available = intronic['m6a_per_kb'].notna().sum()
print(f"Intronic reads with m6A data: {m6a_available} ({m6a_available/len(intronic)*100:.1f}%)")

# ==============================================================================
# Step 3: Classify host genes
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 3: Classifying stress-related host genes")
print("=" * 80)

# Some reads may have multiple overlapping genes (comma-separated)
# Expand them
def expand_genes(row):
    """Split overlapping_genes and return one row per gene."""
    genes = str(row['overlapping_genes']).split(',')
    return [g.strip() for g in genes if g.strip()]

# Get unique host genes
all_host_genes = set()
for genes_str in intronic['overlapping_genes'].dropna().unique():
    for g in str(genes_str).split(','):
        g = g.strip()
        if g:
            all_host_genes.add(g)

print(f"Unique host genes: {len(all_host_genes)}")

# Classify each gene
gene_categories = {}
gene_is_mil = {}
stress_gene_set = set()

for gene in all_host_genes:
    cats, is_mil = classify_gene(gene)
    if cats:
        gene_categories[gene] = cats
        stress_gene_set.add(gene)
    gene_is_mil[gene] = is_mil

print(f"Stress-related host genes found: {len(stress_gene_set)}")

# Print categories
cat_counts = defaultdict(list)
for gene, cats in gene_categories.items():
    for cat, pattern in cats:
        cat_counts[cat].append(gene)

for cat in STRESS_GENES.keys():
    genes = sorted(set(cat_counts.get(cat, [])))
    if genes:
        print(f"  {cat}: {len(genes)} genes - {', '.join(genes)}")

# ==============================================================================
# Step 4: Per-gene statistics for stress genes
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 4: Computing per-gene statistics for stress-related genes")
print("=" * 80)

# For each intronic read, assign to primary gene (first in list)
# But also track multi-gene reads
intronic['primary_gene'] = intronic['overlapping_genes'].apply(
    lambda x: str(x).split(',')[0].strip() if pd.notna(x) else ''
)

# Group by primary gene
gene_stats = []

for gene in sorted(stress_gene_set):
    # Get reads for this gene (check all genes in overlapping_genes)
    mask = intronic['overlapping_genes'].str.contains(
        r'(?:^|,)\s*' + re.escape(gene) + r'\s*(?:,|$)',
        na=False, regex=True
    )
    gene_reads = intronic[mask].copy()

    if len(gene_reads) == 0:
        continue

    # Basic stats
    n_reads = len(gene_reads)
    cell_lines = sorted(gene_reads['cell_line'].unique())
    n_cell_lines = len(cell_lines)
    subfamilies = sorted(gene_reads['gene_id'].unique())
    n_young = (gene_reads['l1_age'] == 'young').sum()
    n_ancient = (gene_reads['l1_age'] == 'ancient').sum()

    # m6A stats
    m6a_vals = gene_reads['m6a_per_kb'].dropna()
    median_m6a = m6a_vals.median() if len(m6a_vals) > 0 else np.nan
    mean_m6a = m6a_vals.mean() if len(m6a_vals) > 0 else np.nan

    # Poly(A) stats
    polya_vals = gene_reads['polya_length'].dropna()
    polya_vals = polya_vals[polya_vals > 0]  # exclude 0 (undetected)
    median_polya = polya_vals.median() if len(polya_vals) > 0 else np.nan

    # HeLa vs HeLa-Ars
    hela_reads = gene_reads[gene_reads['cell_line'] == 'HeLa']
    ars_reads = gene_reads[gene_reads['cell_line'] == 'HeLa-Ars']
    n_hela = len(hela_reads)
    n_ars = len(ars_reads)

    hela_polya = hela_reads['polya_length'].dropna()
    hela_polya = hela_polya[hela_polya > 0]
    ars_polya = ars_reads['polya_length'].dropna()
    ars_polya = ars_polya[ars_polya > 0]

    hela_median_polya = hela_polya.median() if len(hela_polya) > 0 else np.nan
    ars_median_polya = ars_polya.median() if len(ars_polya) > 0 else np.nan
    polya_delta = ars_median_polya - hela_median_polya if (
        pd.notna(hela_median_polya) and pd.notna(ars_median_polya)
    ) else np.nan

    # Categories
    cats = gene_categories.get(gene, [])
    cat_str = "; ".join(sorted(set(c for c, _ in cats)))
    is_mil = gene_is_mil.get(gene, False)

    gene_stats.append({
        'gene': gene,
        'category': cat_str,
        'is_MIL': is_mil,
        'total_reads': n_reads,
        'n_cell_lines': n_cell_lines,
        'cell_lines': ','.join(cell_lines),
        'n_young': n_young,
        'n_ancient': n_ancient,
        'young_frac': n_young / n_reads if n_reads > 0 else 0,
        'n_subfamilies': len(subfamilies),
        'subfamilies': ','.join(subfamilies[:10]),  # top 10
        'median_m6a_per_kb': median_m6a,
        'mean_m6a_per_kb': mean_m6a,
        'median_polya': median_polya,
        'n_hela': n_hela,
        'n_ars': n_ars,
        'hela_median_polya': hela_median_polya,
        'ars_median_polya': ars_median_polya,
        'polya_delta_ars_hela': polya_delta,
    })

stress_df = pd.DataFrame(gene_stats)
stress_df = stress_df.sort_values('total_reads', ascending=False)

# Print results
print(f"\nStress genes with L1 reads: {len(stress_df)}")
print(f"Total reads in stress genes: {stress_df['total_reads'].sum()}")

print("\n" + "-" * 120)
print(f"{'Gene':<15} {'Category':<25} {'Reads':>6} {'CLs':>4} {'Young':>6} {'m6A/kb':>7} "
      f"{'polyA':>6} {'HeLa':>5} {'Ars':>5} {'dPolyA':>7} {'MIL':>4}")
print("-" * 120)

for _, row in stress_df.iterrows():
    mil_flag = "YES" if row['is_MIL'] else ""
    delta_str = f"{row['polya_delta_ars_hela']:+.1f}" if pd.notna(row['polya_delta_ars_hela']) else "n/a"
    m6a_str = f"{row['median_m6a_per_kb']:.2f}" if pd.notna(row['median_m6a_per_kb']) else "n/a"
    polya_str = f"{row['median_polya']:.1f}" if pd.notna(row['median_polya']) else "n/a"
    young_str = f"{row['n_young']}/{row['total_reads']}"
    cat_short = row['category'][:24]

    print(f"{row['gene']:<15} {cat_short:<25} {row['total_reads']:>6} {row['n_cell_lines']:>4} "
          f"{young_str:>6} {m6a_str:>7} {polya_str:>6} {row['n_hela']:>5} {row['n_ars']:>5} "
          f"{delta_str:>7} {mil_flag:>4}")

# ==============================================================================
# Step 5: Top 100 host genes by read count (all genes)
# ==============================================================================

print("\n\n" + "=" * 80)
print("STEP 5: Top 100 host genes by total read count (ALL genes)")
print("=" * 80)

# Count reads per primary gene
gene_read_counts = intronic.groupby('primary_gene').agg(
    total_reads=('read_id', 'count'),
    n_cell_lines=('cell_line', 'nunique'),
    cell_lines=('cell_line', lambda x: ','.join(sorted(x.unique()))),
    n_young=('l1_age', lambda x: (x == 'young').sum()),
    n_ancient=('l1_age', lambda x: (x == 'ancient').sum()),
    median_m6a_per_kb=('m6a_per_kb', 'median'),
    mean_m6a_per_kb=('m6a_per_kb', 'mean'),
    median_polya=('polya_length', lambda x: x[x > 0].median() if (x > 0).any() else np.nan),
    n_subfamilies=('gene_id', 'nunique'),
    subfamilies=('gene_id', lambda x: ','.join(sorted(x.unique())[:10])),
).reset_index()

# Add HeLa/Ars info
hela_counts = intronic[intronic['cell_line'] == 'HeLa'].groupby('primary_gene').size().rename('n_hela')
ars_counts = intronic[intronic['cell_line'] == 'HeLa-Ars'].groupby('primary_gene').size().rename('n_ars')
hela_polya_med = intronic[
    (intronic['cell_line'] == 'HeLa') & (intronic['polya_length'] > 0)
].groupby('primary_gene')['polya_length'].median().rename('hela_median_polya')
ars_polya_med = intronic[
    (intronic['cell_line'] == 'HeLa-Ars') & (intronic['polya_length'] > 0)
].groupby('primary_gene')['polya_length'].median().rename('ars_median_polya')

gene_read_counts = gene_read_counts.merge(hela_counts, left_on='primary_gene', right_index=True, how='left')
gene_read_counts = gene_read_counts.merge(ars_counts, left_on='primary_gene', right_index=True, how='left')
gene_read_counts = gene_read_counts.merge(hela_polya_med, left_on='primary_gene', right_index=True, how='left')
gene_read_counts = gene_read_counts.merge(ars_polya_med, left_on='primary_gene', right_index=True, how='left')
gene_read_counts['n_hela'] = gene_read_counts['n_hela'].fillna(0).astype(int)
gene_read_counts['n_ars'] = gene_read_counts['n_ars'].fillna(0).astype(int)
gene_read_counts['polya_delta_ars_hela'] = gene_read_counts['ars_median_polya'] - gene_read_counts['hela_median_polya']

# Add stress category
gene_read_counts['stress_category'] = gene_read_counts['primary_gene'].apply(
    lambda g: "; ".join(sorted(set(c for c, _ in gene_categories.get(g, [])))) if g in gene_categories else ""
)
gene_read_counts['is_MIL'] = gene_read_counts['primary_gene'].apply(
    lambda g: gene_is_mil.get(g, False)
)

gene_read_counts = gene_read_counts.sort_values('total_reads', ascending=False)
top100 = gene_read_counts.head(100)

print(f"\nTotal unique host genes: {len(gene_read_counts)}")
print(f"Total intronic reads: {gene_read_counts['total_reads'].sum()}")
print(f"\nTop 100 host genes:")
print("-" * 140)
print(f"{'Rank':>4} {'Gene':<20} {'Cat':<20} {'Reads':>6} {'CLs':>4} {'Young':>6} "
      f"{'m6A/kb':>7} {'polyA':>6} {'HeLa':>5} {'Ars':>5} {'dPolyA':>7} {'MIL':>4} {'Subfamilies':<30}")
print("-" * 140)

for rank, (_, row) in enumerate(top100.iterrows(), 1):
    mil_flag = "YES" if row['is_MIL'] else ""
    delta_str = f"{row['polya_delta_ars_hela']:+.1f}" if pd.notna(row['polya_delta_ars_hela']) else "n/a"
    m6a_str = f"{row['median_m6a_per_kb']:.2f}" if pd.notna(row['median_m6a_per_kb']) else "n/a"
    polya_str = f"{row['median_polya']:.1f}" if pd.notna(row['median_polya']) else "n/a"
    young_str = f"{row['n_young']}/{row['total_reads']}"
    cat_short = row['stress_category'][:19] if row['stress_category'] else ""
    subs = str(row['subfamilies'])[:29]

    print(f"{rank:>4} {row['primary_gene']:<20} {cat_short:<20} {row['total_reads']:>6} "
          f"{row['n_cell_lines']:>4} {young_str:>6} {m6a_str:>7} {polya_str:>6} "
          f"{row['n_hela']:>5} {row['n_ars']:>5} {delta_str:>7} {mil_flag:>4} {subs:<30}")

# ==============================================================================
# Step 6: Summary statistics
# ==============================================================================

print("\n\n" + "=" * 80)
print("STEP 6: Summary statistics")
print("=" * 80)

# How many stress gene reads total?
stress_gene_names = set(stress_df['gene'].values)
stress_mask = intronic['overlapping_genes'].apply(
    lambda x: any(g.strip() in stress_gene_names for g in str(x).split(',')) if pd.notna(x) else False
)
n_stress_reads = stress_mask.sum()
print(f"\nTotal intronic reads: {len(intronic)}")
print(f"Reads in stress-related genes: {n_stress_reads} ({n_stress_reads/len(intronic)*100:.1f}%)")

# Category breakdown
print(f"\nCategory breakdown:")
for cat in STRESS_GENES.keys():
    cat_genes = set(cat_counts.get(cat, []))
    if not cat_genes:
        continue
    cat_mask = intronic['overlapping_genes'].apply(
        lambda x: any(g.strip() in cat_genes for g in str(x).split(',')) if pd.notna(x) else False
    )
    n = cat_mask.sum()
    print(f"  {cat:<30} {len(cat_genes):>3} genes, {n:>5} reads")

# Cell line breakdown for stress genes
print(f"\nStress gene reads by cell line:")
stress_reads = intronic[stress_mask]
for cl in sorted(stress_reads['cell_line'].unique()):
    n = (stress_reads['cell_line'] == cl).sum()
    print(f"  {cl:<15} {n:>5} reads")

# m6A comparison: stress genes vs all intronic
print(f"\nm6A/kb comparison:")
stress_m6a = intronic.loc[stress_mask, 'm6a_per_kb'].dropna()
all_m6a = intronic['m6a_per_kb'].dropna()
nonstress_m6a = intronic.loc[~stress_mask, 'm6a_per_kb'].dropna()
print(f"  All intronic L1:      median={all_m6a.median():.2f}, mean={all_m6a.mean():.2f} (n={len(all_m6a)})")
print(f"  Stress gene L1:       median={stress_m6a.median():.2f}, mean={stress_m6a.mean():.2f} (n={len(stress_m6a)})")
print(f"  Non-stress gene L1:   median={nonstress_m6a.median():.2f}, mean={nonstress_m6a.mean():.2f} (n={len(nonstress_m6a)})")

# Poly(A) comparison
print(f"\nPoly(A) comparison:")
stress_polya = intronic.loc[stress_mask, 'polya_length'].dropna()
stress_polya = stress_polya[stress_polya > 0]
all_polya = intronic['polya_length'].dropna()
all_polya = all_polya[all_polya > 0]
nonstress_polya = intronic.loc[~stress_mask, 'polya_length'].dropna()
nonstress_polya = nonstress_polya[nonstress_polya > 0]
print(f"  All intronic L1:      median={all_polya.median():.1f} (n={len(all_polya)})")
print(f"  Stress gene L1:       median={stress_polya.median():.1f} (n={len(stress_polya)})")
print(f"  Non-stress gene L1:   median={nonstress_polya.median():.1f} (n={len(nonstress_polya)})")

# Arsenite response in stress genes
print(f"\nArsenite response (HeLa vs HeLa-Ars) in stress genes:")
hela_stress = stress_reads[stress_reads['cell_line'] == 'HeLa']['polya_length']
hela_stress = hela_stress[hela_stress > 0]
ars_stress = stress_reads[stress_reads['cell_line'] == 'HeLa-Ars']['polya_length']
ars_stress = ars_stress[ars_stress > 0]
if len(hela_stress) > 0 and len(ars_stress) > 0:
    from scipy import stats
    delta = ars_stress.median() - hela_stress.median()
    u_stat, u_p = stats.mannwhitneyu(ars_stress, hela_stress, alternative='two-sided')
    print(f"  HeLa: median={hela_stress.median():.1f} (n={len(hela_stress)})")
    print(f"  HeLa-Ars: median={ars_stress.median():.1f} (n={len(ars_stress)})")
    print(f"  Delta: {delta:+.1f} nt (Mann-Whitney p={u_p:.2e})")

# Genes with strongest arsenite shortening
print(f"\nStress genes with strongest arsenite poly(A) shortening (n_hela>=3 & n_ars>=3):")
strong = stress_df[(stress_df['n_hela'] >= 3) & (stress_df['n_ars'] >= 3)].copy()
if len(strong) > 0:
    strong = strong.sort_values('polya_delta_ars_hela')
    for _, row in strong.head(20).iterrows():
        delta_str = f"{row['polya_delta_ars_hela']:+.1f}" if pd.notna(row['polya_delta_ars_hela']) else "n/a"
        print(f"  {row['gene']:<15} {row['category']:<25} HeLa={row['hela_median_polya']:.0f} "
              f"Ars={row['ars_median_polya']:.0f} Delta={delta_str} "
              f"(n_hela={row['n_hela']}, n_ars={row['n_ars']})")

# ==============================================================================
# Step 7: Genes with >= 5 reads in multiple stress categories
# ==============================================================================

print("\n\n" + "=" * 80)
print("STEP 7: Multi-category stress genes with >= 5 reads")
print("=" * 80)

multi_cat = stress_df[stress_df['category'].str.contains(';')].copy()
multi_cat = multi_cat[multi_cat['total_reads'] >= 5]
if len(multi_cat) > 0:
    for _, row in multi_cat.iterrows():
        print(f"  {row['gene']:<15} {row['category']:<40} reads={row['total_reads']}")
else:
    print("  No multi-category genes with >= 5 reads")

# Genes that appear in both DDR and MIL
print("\nMIL genes found:")
mil_found = stress_df[stress_df['is_MIL']].copy()
for _, row in mil_found.iterrows():
    print(f"  {row['gene']:<15} reads={row['total_reads']}, CLs={row['n_cell_lines']}, "
          f"m6A/kb={row['median_m6a_per_kb']:.2f}" if pd.notna(row['median_m6a_per_kb']) else
          f"  {row['gene']:<15} reads={row['total_reads']}, CLs={row['n_cell_lines']}, m6A/kb=n/a")

# ==============================================================================
# Step 8: Save results
# ==============================================================================

print("\n\n" + "=" * 80)
print("STEP 8: Saving results")
print("=" * 80)

out1 = os.path.join(OUT_DIR, "all_cl_stress_gene_stats.tsv")
stress_df.to_csv(out1, sep='\t', index=False)
print(f"Saved stress gene stats: {out1}")

out2 = os.path.join(OUT_DIR, "all_cl_top_genes.tsv")
# Save full gene stats, not just top 100
gene_read_counts.to_csv(out2, sep='\t', index=False)
print(f"Saved all host gene stats: {out2} ({len(gene_read_counts)} genes)")

# Also save a focused version: top 100
out3 = os.path.join(OUT_DIR, "all_cl_top100_genes.tsv")
top100.to_csv(out3, sep='\t', index=False)
print(f"Saved top 100 genes: {out3}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
