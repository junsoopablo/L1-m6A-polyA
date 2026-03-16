#!/usr/bin/env python3
"""
Stress-related host gene L1 analysis.
Find ancient L1 elements in stress-related host genes from HeLa/HeLa-Ars ONT DRS data.
Examine m6A-poly(A) coupling under arsenite stress in biologically relevant contexts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

BASE_DIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
SUMMARY_DIR = BASE_DIR / 'results_group'
CACHE_DIR = BASE_DIR / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
CHROMHMM_FILE = BASE_DIR / 'analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv'
OUTPUT_DIR = BASE_DIR / 'analysis/01_exploration/topic_08_regulatory_chromatin/stress_gene_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3']
ARS_GROUPS = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Stress-related gene categories with patterns
STRESS_CATEGORIES = {
    'DDR/DNA repair': [
        'BRCA1', 'BRCA2', 'ATM', 'ATR', 'RAD51', 'RAD50', 'TP53',
        'CHEK1', 'CHEK2', 'PARP1', 'PARP2', 'PARP3', 'NBN', 'MRE11',
        'BLM', 'WRN', 'XPC', 'XPA',
    ],
    'DDR/DNA repair (prefix)': [
        'XRCC', 'FANC', 'ERCC',
    ],
    'Heat shock/chaperone': [
        'HSF1', 'HSF2', 'BAG3',
    ],
    'Heat shock/chaperone (prefix)': [
        'HSPA', 'HSP90', 'HSPB', 'HSPH', 'DNAJ',
    ],
    'Oxidative stress': [
        'NFE2L2', 'KEAP1', 'SOD1', 'SOD2', 'CAT', 'NQO1', 'HMOX1', 'TXN',
    ],
    'Oxidative stress (prefix)': [
        'GPX', 'PRDX',
    ],
    'Apoptosis': [
        'BCL2', 'BAX', 'BAK1', 'MCL1', 'BID', 'APAF1', 'XIAP',
    ],
    'Apoptosis (prefix)': [
        'CASP', 'BIRC',
    ],
    'Autophagy': [
        'BECN1', 'SQSTM1', 'MAP1LC3B', 'BNIP3', 'ULK1',
    ],
    'Autophagy (prefix)': [
        'ATG',
    ],
    'UPR/ER stress': [
        'EIF2AK3', 'ERN1', 'ATF4', 'ATF6', 'XBP1', 'DDIT3', 'HSPA5',
    ],
    'Inflammation/NF-kB': [
        'NFKB1', 'RELA',
    ],
    'Inflammation/NF-kB (prefix)': [
        'IKBK', 'TNFAIP', 'TRAF',
    ],
    'Cell cycle checkpoint': [
        'CDKN1A', 'CDKN2A', 'RB1',
    ],
    'Cell cycle checkpoint (prefix)': [
        'CDC25', 'GADD45',
    ],
    'Stress signaling': [
        'MAPK8', 'MAPK14', 'MAP3K5', 'HIF1A', 'VHL',
    ],
    'Stress signaling (prefix)': [
        'EGLN',
    ],
    'RNA stress': [
        'G3BP1', 'G3BP2', 'TIA1', 'TIAL1', 'EIF2AK2',
    ],
}

# Alternative gene names mapping
ALT_NAMES = {
    'NFE2L2': 'NRF2',
    'EIF2AK3': 'PERK',
    'ERN1': 'IRE1',
    'DDIT3': 'CHOP',
    'HSPA5': 'BiP',
    'CDKN1A': 'p21',
    'CDKN2A': 'p16',
    'MAPK8': 'JNK',
    'MAPK14': 'p38',
    'MAP3K5': 'ASK1',
    'EIF2AK2': 'PKR',
}


def classify_l1_age(subfamily):
    """Classify L1 subfamily as young or ancient."""
    if subfamily in YOUNG_SUBFAMILIES:
        return 'young'
    return 'ancient'


def match_stress_gene(gene_name):
    """Check if a gene matches any stress category. Return (category, matched_pattern) or None."""
    if not gene_name or gene_name == '.' or pd.isna(gene_name):
        return None

    gene_upper = gene_name.upper()

    # Exact matches first
    for cat, genes in STRESS_CATEGORIES.items():
        if '(prefix)' in cat:
            continue
        for g in genes:
            if gene_upper == g.upper():
                clean_cat = cat
                alt = ALT_NAMES.get(g, '')
                return (clean_cat, g, alt)

    # Prefix matches
    for cat, prefixes in STRESS_CATEGORIES.items():
        if '(prefix)' not in cat:
            continue
        clean_cat = cat.replace(' (prefix)', '')
        for prefix in prefixes:
            if gene_upper.startswith(prefix.upper()):
                return (clean_cat, prefix + '*', '')

    return None


def load_data():
    """Load and merge L1 summary + Part3 cache for HeLa and HeLa-Ars."""
    all_dfs = []

    for group_list, condition in [(HELA_GROUPS, 'HeLa'), (ARS_GROUPS, 'HeLa-Ars')]:
        for group in group_list:
            # L1 summary
            summary_path = SUMMARY_DIR / group / 'g_summary' / f'{group}_L1_summary.tsv'
            if not summary_path.exists():
                print(f"  WARNING: {summary_path} not found")
                continue

            df_sum = pd.read_csv(summary_path, sep='\t')
            # Filter PASS only
            df_sum = df_sum[df_sum['qc_tag'] == 'PASS'].copy()

            # Part3 cache
            cache_path = CACHE_DIR / f'{group}_l1_per_read.tsv'
            if not cache_path.exists():
                print(f"  WARNING: {cache_path} not found")
                continue

            df_cache = pd.read_csv(cache_path, sep='\t')

            # Merge on read_id
            df_merged = df_sum.merge(df_cache[['read_id', 'read_length', 'm6a_sites_high', 'psi_sites_high']],
                                     on='read_id', how='left', suffixes=('_sum', '_cache'))

            # Compute m6a_per_kb from cache
            rl = df_merged['read_length_cache'].fillna(df_merged['read_length_sum'])
            df_merged['m6a_per_kb'] = df_merged['m6a_sites_high'].fillna(0) / (rl / 1000)
            df_merged['psi_per_kb'] = df_merged['psi_sites_high'].fillna(0) / (rl / 1000)
            df_merged['read_len'] = rl

            # Add metadata
            df_merged['group'] = group
            df_merged['condition'] = condition
            df_merged['l1_age'] = df_merged['gene_id'].apply(classify_l1_age)

            # Rename for clarity
            df_merged.rename(columns={
                'gene_id': 'l1_subfamily',
                'TE_group': 'genomic_context',
            }, inplace=True)

            all_dfs.append(df_merged)
            print(f"  Loaded {group}: {len(df_merged)} PASS reads")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal merged reads: {len(df)}")
    print(f"  HeLa: {(df['condition']=='HeLa').sum()}")
    print(f"  HeLa-Ars: {(df['condition']=='HeLa-Ars').sum()}")
    return df


def expand_genes(df):
    """Expand rows with multiple overlapping genes (semicolon-separated)."""
    # Some reads overlap multiple genes
    rows = []
    for _, row in df.iterrows():
        genes = str(row['overlapping_genes'])
        if genes == 'nan' or genes == '.' or genes == '':
            continue
        for gene in genes.split(';'):
            gene = gene.strip()
            if gene:
                rows.append({**row.to_dict(), 'host_gene': gene})
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("STRESS-RELATED HOST GENE L1 ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading L1 summary + Part3 cache...")
    df = load_data()

    # Filter intronic only (intergenic has no host gene)
    df_intronic = df[df['genomic_context'] == 'intronic'].copy()
    print(f"\nIntronic reads (have host genes): {len(df_intronic)}")

    # ------------------------------------------------------------------
    # 2. Expand genes and annotate stress categories
    # ------------------------------------------------------------------
    print("\n[2] Expanding gene annotations and matching stress categories...")
    df_genes = expand_genes(df_intronic)
    print(f"Expanded gene-read pairs: {len(df_genes)}")

    unique_genes = df_genes['host_gene'].unique()
    print(f"Unique host genes: {len(unique_genes)}")

    # Match stress genes
    stress_matches = {}
    for gene in unique_genes:
        match = match_stress_gene(gene)
        if match:
            stress_matches[gene] = match

    print(f"Stress-related host genes found: {len(stress_matches)}")
    for gene, (cat, pattern, alt) in sorted(stress_matches.items(), key=lambda x: x[1][0]):
        alt_str = f" ({alt})" if alt else ""
        print(f"  {gene}{alt_str} -> {cat} [{pattern}]")

    # ------------------------------------------------------------------
    # 3. Compute per-gene statistics
    # ------------------------------------------------------------------
    print("\n[3] Computing per-gene statistics...")

    gene_stats = []
    for gene in unique_genes:
        gdf = df_genes[df_genes['host_gene'] == gene]

        hela_reads = gdf[gdf['condition'] == 'HeLa']
        ars_reads = gdf[gdf['condition'] == 'HeLa-Ars']

        n_total = len(gdf)
        n_hela = len(hela_reads)
        n_ars = len(ars_reads)

        # Poly(A)
        polya_hela = hela_reads['polya_length'].median() if n_hela > 0 else np.nan
        polya_ars = ars_reads['polya_length'].median() if n_ars > 0 else np.nan
        polya_delta = polya_ars - polya_hela if n_hela > 0 and n_ars > 0 else np.nan

        # m6A
        m6a_median = gdf['m6a_per_kb'].median()
        m6a_hela = hela_reads['m6a_per_kb'].median() if n_hela > 0 else np.nan
        m6a_ars = ars_reads['m6a_per_kb'].median() if n_ars > 0 else np.nan

        # Read length
        rl_median = gdf['read_len'].median()

        # L1 subfamilies
        subfamilies = gdf['l1_subfamily'].value_counts()
        top_subfamily = subfamilies.index[0]
        age_counts = gdf['l1_age'].value_counts()
        n_young = age_counts.get('young', 0)
        n_ancient = age_counts.get('ancient', 0)

        # Stress category
        stress_info = stress_matches.get(gene, None)
        is_stress = stress_info is not None
        stress_cat = stress_info[0] if stress_info else ''
        stress_pattern = stress_info[1] if stress_info else ''
        stress_alt = stress_info[2] if stress_info else ''

        # Poly(A) p-value if both conditions have >=3 reads
        polya_pval = np.nan
        if n_hela >= 3 and n_ars >= 3:
            try:
                _, polya_pval = stats.mannwhitneyu(
                    hela_reads['polya_length'].dropna(),
                    ars_reads['polya_length'].dropna(),
                    alternative='two-sided'
                )
            except:
                pass

        gene_stats.append({
            'host_gene': gene,
            'is_stress_gene': is_stress,
            'stress_category': stress_cat,
            'stress_pattern': stress_pattern,
            'alt_name': stress_alt,
            'n_total': n_total,
            'n_hela': n_hela,
            'n_ars': n_ars,
            'polya_hela': polya_hela,
            'polya_ars': polya_ars,
            'polya_delta': polya_delta,
            'polya_pval': polya_pval,
            'm6a_overall': m6a_median,
            'm6a_hela': m6a_hela,
            'm6a_ars': m6a_ars,
            'read_length': rl_median,
            'top_subfamily': top_subfamily,
            'n_young': n_young,
            'n_ancient': n_ancient,
            'pct_young': 100 * n_young / n_total if n_total > 0 else 0,
        })

    df_stats = pd.DataFrame(gene_stats)
    df_stats.sort_values('n_total', ascending=False, inplace=True)

    # ------------------------------------------------------------------
    # 4. Print stress-related host genes
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STRESS-RELATED HOST GENES WITH L1 ELEMENTS")
    print("=" * 80)

    df_stress = df_stats[df_stats['is_stress_gene']].sort_values('n_total', ascending=False)
    print(f"\nFound {len(df_stress)} stress-related host genes\n")

    if len(df_stress) > 0:
        print(f"{'Gene':<16} {'Category':<22} {'Alt':<8} {'Total':>5} {'HeLa':>5} {'Ars':>5} "
              f"{'pA_HeLa':>8} {'pA_Ars':>8} {'Delta':>7} {'p-val':>10} "
              f"{'m6A/kb':>7} {'RL':>6} {'TopSF':<12} {'Yng':>3} {'Anc':>3}")
        print("-" * 160)

        for _, row in df_stress.iterrows():
            pval_str = f"{row['polya_pval']:.2e}" if not np.isnan(row['polya_pval']) else 'n/a'
            delta_str = f"{row['polya_delta']:+.1f}" if not np.isnan(row['polya_delta']) else 'n/a'
            pa_hela = f"{row['polya_hela']:.1f}" if not np.isnan(row['polya_hela']) else 'n/a'
            pa_ars = f"{row['polya_ars']:.1f}" if not np.isnan(row['polya_ars']) else 'n/a'

            print(f"{row['host_gene']:<16} {row['stress_category']:<22} {row['alt_name']:<8} "
                  f"{row['n_total']:>5} {row['n_hela']:>5} {row['n_ars']:>5} "
                  f"{pa_hela:>8} {pa_ars:>8} {delta_str:>7} {pval_str:>10} "
                  f"{row['m6a_overall']:>7.2f} {row['read_length']:>6.0f} "
                  f"{row['top_subfamily']:<12} {row['n_young']:>3} {row['n_ancient']:>3}")

    # Category summary
    print(f"\n--- Stress gene summary by category ---")
    cat_summary = df_stress.groupby('stress_category').agg(
        n_genes=('host_gene', 'nunique'),
        n_reads=('n_total', 'sum'),
        avg_m6a=('m6a_overall', 'mean'),
        avg_polya_delta=('polya_delta', lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan),
    ).sort_values('n_reads', ascending=False)
    print(cat_summary.to_string())

    # ------------------------------------------------------------------
    # 5. Print top 50 host genes by read count
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TOP 50 HOST GENES BY READ COUNT (ALL GENES)")
    print("=" * 80)

    df_top50 = df_stats.head(50)
    print(f"\n{'#':>3} {'Gene':<20} {'Stress?':<8} {'Category':<22} {'Total':>5} {'HeLa':>5} {'Ars':>5} "
          f"{'pA_HeLa':>8} {'pA_Ars':>8} {'Delta':>7} {'p-val':>10} "
          f"{'m6A/kb':>7} {'RL':>6} {'TopSF':<12} {'%Yng':>5}")
    print("-" * 170)

    for i, (_, row) in enumerate(df_top50.iterrows(), 1):
        pval_str = f"{row['polya_pval']:.2e}" if not np.isnan(row['polya_pval']) else 'n/a'
        delta_str = f"{row['polya_delta']:+.1f}" if not np.isnan(row['polya_delta']) else 'n/a'
        pa_hela = f"{row['polya_hela']:.1f}" if not np.isnan(row['polya_hela']) else 'n/a'
        pa_ars = f"{row['polya_ars']:.1f}" if not np.isnan(row['polya_ars']) else 'n/a'
        stress_flag = '*' if row['is_stress_gene'] else ''

        print(f"{i:>3} {row['host_gene']:<20} {stress_flag:<8} {row['stress_category']:<22} "
              f"{row['n_total']:>5} {row['n_hela']:>5} {row['n_ars']:>5} "
              f"{pa_hela:>8} {pa_ars:>8} {delta_str:>7} {pval_str:>10} "
              f"{row['m6a_overall']:>7.2f} {row['read_length']:>6.0f} "
              f"{row['top_subfamily']:<12} {row['pct_young']:>5.1f}")

    # ------------------------------------------------------------------
    # 6. ChromHMM integration
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CHROMHMM STATE ANALYSIS FOR STRESS-RELATED HOST GENES")
    print("=" * 80)

    print("\nLoading ChromHMM annotations...")
    df_chromhmm = pd.read_csv(CHROMHMM_FILE, sep='\t')
    # Filter HeLa / HeLa-Ars
    df_chromhmm_hela = df_chromhmm[df_chromhmm['cellline'].isin(['HeLa', 'HeLa-Ars'])].copy()
    print(f"ChromHMM annotated reads (HeLa/HeLa-Ars): {len(df_chromhmm_hela)}")

    # Merge chromHMM with our gene-expanded df (using read_id)
    # df_genes has host_gene; df_chromhmm_hela has chromhmm_group
    df_chrom_genes = df_genes.merge(
        df_chromhmm_hela[['read_id', 'chromhmm_state', 'chromhmm_group']],
        on='read_id', how='inner'
    )
    print(f"Reads with both gene and chromHMM annotation: {len(df_chrom_genes)}")

    # Stress genes only
    stress_gene_set = set(stress_matches.keys())
    df_chrom_stress = df_chrom_genes[df_chrom_genes['host_gene'].isin(stress_gene_set)]
    print(f"Stress gene reads with chromHMM: {len(df_chrom_stress)}")

    if len(df_chrom_stress) > 0:
        print(f"\n--- ChromHMM distribution of stress-gene L1 reads ---")
        chrom_dist = df_chrom_stress.groupby(['chromhmm_group', 'condition']).size().unstack(fill_value=0)
        print(chrom_dist.to_string())

        print(f"\n--- Per stress gene × chromHMM state ---")
        gene_chrom = df_chrom_stress.groupby(['host_gene', 'chromhmm_group']).agg(
            n=('read_id', 'count'),
            m6a_median=('m6a_per_kb', 'median'),
            polya_median=('polya_length', 'median'),
        ).reset_index()

        for gene in df_chrom_stress['host_gene'].unique():
            gc = gene_chrom[gene_chrom['host_gene'] == gene]
            cat_info = stress_matches.get(gene, ('', '', ''))
            print(f"\n  {gene} ({cat_info[0]}):")
            for _, r in gc.iterrows():
                print(f"    {r['chromhmm_group']:<15} n={r['n']:>3}  m6A/kb={r['m6a_median']:.2f}  poly(A)={r['polya_median']:.1f}")

    # ------------------------------------------------------------------
    # 7. Overall chromHMM distribution for all intronic L1 reads (reference)
    # ------------------------------------------------------------------
    print(f"\n--- Reference: ChromHMM distribution for ALL intronic L1 reads ---")
    df_chrom_all = df_chrom_genes.groupby('chromhmm_group').size()
    df_chrom_all_pct = 100 * df_chrom_all / df_chrom_all.sum()
    for state, pct in df_chrom_all_pct.sort_values(ascending=False).items():
        print(f"  {state:<15} {pct:>5.1f}%  (n={df_chrom_all[state]})")

    # ------------------------------------------------------------------
    # 8. Detailed case studies: stress genes with >=5 reads in both conditions
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DETAILED CASE STUDIES: STRESS GENES WITH >=5 READS IN BOTH CONDITIONS")
    print("=" * 80)

    df_stress_cases = df_stress[(df_stress['n_hela'] >= 5) & (df_stress['n_ars'] >= 5)]
    print(f"\nGenes with >=5 reads in both conditions: {len(df_stress_cases)}")

    for _, row in df_stress_cases.iterrows():
        gene = row['host_gene']
        cat_info = stress_matches.get(gene, ('', '', ''))
        gdf = df_genes[df_genes['host_gene'] == gene]

        hela_reads = gdf[gdf['condition'] == 'HeLa']
        ars_reads = gdf[gdf['condition'] == 'HeLa-Ars']

        print(f"\n{'='*60}")
        alt_str = f" ({cat_info[2]})" if cat_info[2] else ""
        print(f"  {gene}{alt_str} | {cat_info[0]}")
        print(f"  HeLa: n={len(hela_reads)}, HeLa-Ars: n={len(ars_reads)}")
        print(f"{'='*60}")

        # Poly(A) distribution
        print(f"  Poly(A):")
        print(f"    HeLa:     median={hela_reads['polya_length'].median():.1f}, "
              f"mean={hela_reads['polya_length'].mean():.1f}, "
              f"IQR=[{hela_reads['polya_length'].quantile(0.25):.1f}-{hela_reads['polya_length'].quantile(0.75):.1f}]")
        print(f"    HeLa-Ars: median={ars_reads['polya_length'].median():.1f}, "
              f"mean={ars_reads['polya_length'].mean():.1f}, "
              f"IQR=[{ars_reads['polya_length'].quantile(0.25):.1f}-{ars_reads['polya_length'].quantile(0.75):.1f}]")
        print(f"    Delta: {row['polya_delta']:+.1f}nt")

        # m6A
        print(f"  m6A/kb:")
        print(f"    HeLa:     median={hela_reads['m6a_per_kb'].median():.2f}, "
              f"mean={hela_reads['m6a_per_kb'].mean():.2f}")
        print(f"    HeLa-Ars: median={ars_reads['m6a_per_kb'].median():.2f}, "
              f"mean={ars_reads['m6a_per_kb'].mean():.2f}")

        # Read length
        print(f"  Read length:")
        print(f"    HeLa:     median={hela_reads['read_len'].median():.0f}")
        print(f"    HeLa-Ars: median={ars_reads['read_len'].median():.0f}")

        # L1 subfamilies
        print(f"  L1 subfamilies: {dict(gdf['l1_subfamily'].value_counts().head(5))}")
        print(f"  Age: ancient={row['n_ancient']}, young={row['n_young']}")

        # Genomic loci
        loci = gdf[['chr', 'l1_subfamily', 'condition']].drop_duplicates()
        print(f"  Unique loci: {len(gdf[['chr', 'start', 'end']].drop_duplicates())}")

    # ------------------------------------------------------------------
    # 9. Aggregate: stress genes vs all intronic genes
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGGREGATE COMPARISON: STRESS GENES VS ALL INTRONIC GENES")
    print("=" * 80)

    df_genes['is_stress'] = df_genes['host_gene'].isin(stress_gene_set)

    for condition in ['HeLa', 'HeLa-Ars']:
        sub = df_genes[df_genes['condition'] == condition]
        stress = sub[sub['is_stress']]
        nonstress = sub[~sub['is_stress']]

        print(f"\n--- {condition} ---")
        print(f"  Stress genes: n={len(stress)} reads in {stress['host_gene'].nunique()} genes")
        print(f"  Non-stress:   n={len(nonstress)} reads in {nonstress['host_gene'].nunique()} genes")

        if len(stress) > 0:
            print(f"  Poly(A):  stress={stress['polya_length'].median():.1f} vs non-stress={nonstress['polya_length'].median():.1f}")
            print(f"  m6A/kb:   stress={stress['m6a_per_kb'].median():.2f} vs non-stress={nonstress['m6a_per_kb'].median():.2f}")
            print(f"  RL:       stress={stress['read_len'].median():.0f} vs non-stress={nonstress['read_len'].median():.0f}")

    # Poly(A) delta for stress vs non-stress genes
    print(f"\n--- Arsenite poly(A) delta: stress vs non-stress host genes ---")
    for label, gene_set in [('Stress genes', stress_gene_set), ('Non-stress genes', None)]:
        if gene_set is not None:
            reads_hela = df_genes[(df_genes['host_gene'].isin(gene_set)) & (df_genes['condition'] == 'HeLa')]['polya_length']
            reads_ars = df_genes[(df_genes['host_gene'].isin(gene_set)) & (df_genes['condition'] == 'HeLa-Ars')]['polya_length']
        else:
            reads_hela = df_genes[(~df_genes['host_gene'].isin(stress_gene_set)) & (df_genes['condition'] == 'HeLa')]['polya_length']
            reads_ars = df_genes[(~df_genes['host_gene'].isin(stress_gene_set)) & (df_genes['condition'] == 'HeLa-Ars')]['polya_length']

        if len(reads_hela) > 0 and len(reads_ars) > 0:
            delta = reads_ars.median() - reads_hela.median()
            stat, pval = stats.mannwhitneyu(reads_hela.dropna(), reads_ars.dropna(), alternative='two-sided')
            print(f"  {label:<20}: HeLa median={reads_hela.median():.1f}, Ars median={reads_ars.median():.1f}, "
                  f"delta={delta:+.1f}, MWU p={pval:.2e}")

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save full gene stats
    out_path1 = OUTPUT_DIR / 'all_host_gene_stats.tsv'
    df_stats.to_csv(out_path1, sep='\t', index=False)
    print(f"Saved: {out_path1}")

    # Save stress gene stats
    out_path2 = OUTPUT_DIR / 'stress_gene_stats.tsv'
    df_stress.to_csv(out_path2, sep='\t', index=False)
    print(f"Saved: {out_path2}")

    # Save stress gene per-read data
    df_stress_reads = df_genes[df_genes['host_gene'].isin(stress_gene_set)]
    out_path3 = OUTPUT_DIR / 'stress_gene_per_read.tsv'
    df_stress_reads[['read_id', 'host_gene', 'l1_subfamily', 'l1_age', 'condition',
                     'polya_length', 'm6a_per_kb', 'psi_per_kb', 'read_len',
                     'genomic_context', 'chr', 'start', 'end']].to_csv(out_path3, sep='\t', index=False)
    print(f"Saved: {out_path3}")

    print("\nDone!")


if __name__ == '__main__':
    main()
