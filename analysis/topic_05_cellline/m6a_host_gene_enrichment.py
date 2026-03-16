#!/usr/bin/env python3
"""Host gene functional enrichment: high-m6A vs low-m6A ancient L1.

Question: Do ancient L1 loci with high m6A reside in functionally distinct
host genes compared to low-m6A loci?

If high-m6A L1 are in stress response / essential genes → selective protection.

Approach:
  1. Load all ancient intronic L1 reads (normal conditions only)
  2. Aggregate by locus (transcript_id) to get per-locus m6A/kb
  3. Split into high vs low m6A quartiles
  4. Compare host gene lists using gseapy enrichr
  5. Also: stress-specific — compare "protected" (poly(A) maintained) vs "degraded" loci
"""

import pandas as pd
import numpy as np
import gseapy as gp
from pathlib import Path
from scipy import stats
from collections import Counter

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
CACHE_DIR = TOPIC_05 / 'part3_l1_per_read_cache'
OUTDIR = TOPIC_05 / 'm6a_host_gene_enrichment'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
EXCLUDE_PREFIXES = ('HeLa-Ars', 'MCF7-EV')

ALL_GROUPS = sorted([
    f.stem.replace('_l1_per_read', '')
    for f in CACHE_DIR.glob('*_l1_per_read.tsv')
    if not any(f.stem.startswith(ex) for ex in EXCLUDE_PREFIXES)
])

# Also load HeLa-Ars for stress comparison
ARS_GROUPS = sorted([
    f.stem.replace('_l1_per_read', '')
    for f in CACHE_DIR.glob('HeLa-Ars*_l1_per_read.tsv')
])
HELA_GROUPS = sorted([
    f.stem.replace('_l1_per_read', '')
    for f in CACHE_DIR.glob('*_l1_per_read.tsv')
    if f.stem.startswith('HeLa_')
])

# =====================================================================
# 1. Load reads
# =====================================================================
def load_groups(group_list):
    all_reads = []
    for grp in group_list:
        summary_f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        cache_f = CACHE_DIR / f'{grp}_l1_per_read.tsv'
        if not summary_f.exists() or not cache_f.exists():
            continue

        summary = pd.read_csv(summary_f, sep='\t', usecols=[
            'read_id', 'chr', 'start', 'end', 'read_length',
            'te_start', 'te_end', 'gene_id', 'te_strand',
            'transcript_id', 'qc_tag', 'TE_group', 'overlapping_genes',
            'polya_length',
        ])
        summary = summary[summary['qc_tag'] == 'PASS'].copy()

        cache = pd.read_csv(cache_f, sep='\t', usecols=[
            'read_id', 'read_length', 'm6a_sites_high',
        ])

        merged = summary.merge(cache[['read_id', 'm6a_sites_high']],
                               on='read_id', how='inner')
        merged['group'] = grp
        merged['is_young'] = merged['gene_id'].isin(YOUNG)
        merged['m6a_per_kb'] = merged['m6a_sites_high'] / merged['read_length'] * 1000
        all_reads.append(merged)

    return pd.concat(all_reads, ignore_index=True) if all_reads else pd.DataFrame()

print("Loading normal condition reads...")
df_normal = load_groups(ALL_GROUPS)
print(f"  Total: {len(df_normal)}")

print("Loading HeLa reads...")
df_hela = load_groups(HELA_GROUPS)
print(f"  HeLa: {len(df_hela)}")

print("Loading HeLa-Ars reads...")
df_ars = load_groups(ARS_GROUPS)
print(f"  HeLa-Ars: {len(df_ars)}")

# =====================================================================
# 2. Filter to ancient intronic L1
# =====================================================================
print("\nFiltering to ancient intronic L1...")

def filter_ancient_intronic(df):
    mask = (~df['is_young']) & (df['TE_group'] == 'intronic') & (df['overlapping_genes'].notna())
    return df[mask].copy()

normal = filter_ancient_intronic(df_normal)
hela = filter_ancient_intronic(df_hela)
ars = filter_ancient_intronic(df_ars)

print(f"  Normal ancient intronic: {len(normal)}, unique genes: {normal['overlapping_genes'].nunique()}")
print(f"  HeLa ancient intronic: {len(hela)}, unique genes: {hela['overlapping_genes'].nunique()}")
print(f"  HeLa-Ars ancient intronic: {len(ars)}, unique genes: {ars['overlapping_genes'].nunique()}")

# =====================================================================
# 3. Per-locus m6A aggregation (normal conditions)
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 1: PER-LOCUS m6A/kb — HIGH vs LOW QUARTILE HOST GENES")
print("=" * 80)

# Aggregate by locus (transcript_id = L1 element ID)
locus_agg = normal.groupby('transcript_id').agg(
    n_reads=('read_id', 'count'),
    mean_m6a_per_kb=('m6a_per_kb', 'mean'),
    median_m6a_per_kb=('m6a_per_kb', 'median'),
    host_gene=('overlapping_genes', 'first'),
    gene_id=('gene_id', 'first'),
    mean_read_length=('read_length', 'mean'),
).reset_index()

# Also aggregate by host gene (multiple L1 loci per gene)
gene_agg = normal.groupby('overlapping_genes').agg(
    n_reads=('read_id', 'count'),
    n_loci=('transcript_id', 'nunique'),
    mean_m6a_per_kb=('m6a_per_kb', 'mean'),
    median_m6a_per_kb=('m6a_per_kb', 'median'),
    mean_read_length=('read_length', 'mean'),
).reset_index()

# Filter to genes with ≥3 reads for reliable m6A estimates
gene_reliable = gene_agg[gene_agg['n_reads'] >= 3].copy()
print(f"\n  Genes with ≥3 reads: {len(gene_reliable)}")
print(f"  m6A/kb distribution: median={gene_reliable['median_m6a_per_kb'].median():.2f}, "
      f"mean={gene_reliable['mean_m6a_per_kb'].mean():.2f}")

# Quartile split
gene_reliable['m6a_quartile'] = pd.qcut(gene_reliable['mean_m6a_per_kb'], 4,
                                          labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'],
                                          duplicates='drop')

for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
    qd = gene_reliable[gene_reliable['m6a_quartile'] == q]
    print(f"  {q}: {len(qd)} genes, m6A/kb range: {qd['mean_m6a_per_kb'].min():.2f}-{qd['mean_m6a_per_kb'].max():.2f}")

# Gene lists
high_m6a_genes = gene_reliable[gene_reliable['m6a_quartile'] == 'Q4_high']['overlapping_genes'].tolist()
low_m6a_genes = gene_reliable[gene_reliable['m6a_quartile'] == 'Q1_low']['overlapping_genes'].tolist()
all_bg_genes = gene_reliable['overlapping_genes'].tolist()

print(f"\n  High-m6A (Q4) genes: {len(high_m6a_genes)}")
print(f"  Low-m6A (Q1) genes: {len(low_m6a_genes)}")
print(f"  Background genes: {len(all_bg_genes)}")

# Save gene lists
pd.DataFrame({'gene': high_m6a_genes}).to_csv(OUTDIR / 'high_m6a_genes_Q4.txt',
                                                index=False, header=False)
pd.DataFrame({'gene': low_m6a_genes}).to_csv(OUTDIR / 'low_m6a_genes_Q1.txt',
                                               index=False, header=False)
pd.DataFrame({'gene': all_bg_genes}).to_csv(OUTDIR / 'background_genes.txt',
                                              index=False, header=False)

# =====================================================================
# 4. Enrichr analysis — high m6A genes
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 2: ENRICHR — HIGH-m6A HOST GENES (Q4)")
print("=" * 80)

gene_set_libraries = [
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'GO_Cellular_Component_2023',
    'KEGG_2021_Human',
    'Reactome_2022',
]

def run_enrichr(gene_list, label, outdir):
    """Run enrichr and return results."""
    results = {}
    for lib in gene_set_libraries:
        try:
            enr = gp.enrichr(gene_list=gene_list,
                             gene_sets=lib,
                             organism='human',
                             outdir=str(outdir / f'{label}_{lib}'),
                             no_plot=True,
                             cutoff=0.05)
            df_res = enr.results
            sig = df_res[df_res['Adjusted P-value'] < 0.05]
            results[lib] = sig
            if len(sig) > 0:
                print(f"\n  {lib}: {len(sig)} significant terms (FDR<0.05)")
                for _, row in sig.head(10).iterrows():
                    genes_str = row['Genes']
                    n_genes = len(genes_str.split(';'))
                    print(f"    {row['Term'][:60]:<60s} p={row['Adjusted P-value']:.2e} "
                          f"({n_genes} genes)")
            else:
                print(f"\n  {lib}: no significant terms")
        except Exception as e:
            print(f"\n  {lib}: ERROR - {e}")
            results[lib] = pd.DataFrame()
    return results

print("\n--- High-m6A genes (Q4) ---")
high_results = run_enrichr(high_m6a_genes, 'high_m6a', OUTDIR)

print("\n\n--- Low-m6A genes (Q1) ---")
low_results = run_enrichr(low_m6a_genes, 'low_m6a', OUTDIR)

# =====================================================================
# 5. Stress-specific: protected vs degraded loci
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 3: STRESS PROTECTED vs DEGRADED LOCI (HeLa vs HeLa-Ars)")
print("=" * 80)

# Combine HeLa + HeLa-Ars
hela_combined = pd.concat([hela.assign(condition='Normal'),
                            ars.assign(condition='Arsenite')], ignore_index=True)

# Per-locus: compare poly(A) between conditions
locus_stress = hela_combined.groupby(['transcript_id', 'condition']).agg(
    n_reads=('read_id', 'count'),
    median_polya=('polya_length', 'median'),
    mean_m6a_per_kb=('m6a_per_kb', 'mean'),
    host_gene=('overlapping_genes', 'first'),
).reset_index()

# Pivot to get normal vs arsenite side by side
locus_wide = locus_stress.pivot_table(
    index=['transcript_id', 'host_gene'],
    columns='condition',
    values=['median_polya', 'mean_m6a_per_kb', 'n_reads']
).reset_index()
locus_wide.columns = ['_'.join(col).strip('_') for col in locus_wide.columns]

# Filter to loci with reads in both conditions
both = locus_wide[
    (locus_wide.get('n_reads_Normal', 0) >= 1) &
    (locus_wide.get('n_reads_Arsenite', 0) >= 1)
].copy()

if len(both) > 0 and 'median_polya_Normal' in both.columns and 'median_polya_Arsenite' in both.columns:
    both['delta_polya'] = both['median_polya_Arsenite'] - both['median_polya_Normal']

    # Protected = delta_polya > -10 (poly(A) maintained)
    # Degraded = delta_polya < -30 (poly(A) shortened)
    protected = both[both['delta_polya'] > -10]
    degraded = both[both['delta_polya'] < -30]

    print(f"\n  Loci in both conditions: {len(both)}")
    print(f"  Protected (Δ > -10nt): {len(protected)} loci")
    print(f"  Degraded (Δ < -30nt): {len(degraded)} loci")

    if len(protected) > 0 and len(degraded) > 0:
        protected_genes = protected['host_gene'].dropna().unique().tolist()
        degraded_genes = degraded['host_gene'].dropna().unique().tolist()

        print(f"  Protected unique genes: {len(protected_genes)}")
        print(f"  Degraded unique genes: {len(degraded_genes)}")

        # Check m6A/kb difference
        prot_m6a = both[both['delta_polya'] > -10].get('mean_m6a_per_kb_Normal')
        deg_m6a = both[both['delta_polya'] < -30].get('mean_m6a_per_kb_Normal')
        if prot_m6a is not None and deg_m6a is not None and len(prot_m6a) > 0 and len(deg_m6a) > 0:
            print(f"\n  Protected loci m6A/kb: {prot_m6a.median():.2f}")
            print(f"  Degraded loci m6A/kb: {deg_m6a.median():.2f}")
            _, p = stats.mannwhitneyu(prot_m6a.dropna(), deg_m6a.dropna(), alternative='two-sided')
            print(f"  Mann-Whitney p = {p:.2e}")

        if len(protected_genes) >= 10:
            print("\n--- Protected loci host genes ---")
            run_enrichr(protected_genes, 'stress_protected', OUTDIR)

        if len(degraded_genes) >= 10:
            print("\n--- Degraded loci host genes ---")
            run_enrichr(degraded_genes, 'stress_degraded', OUTDIR)
else:
    print("\n  Insufficient data for stress comparison")

# =====================================================================
# 6. Top high-m6A host genes — what are they?
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 4: TOP HIGH-m6A HOST GENES")
print("=" * 80)

top_genes = gene_reliable.nlargest(30, 'mean_m6a_per_kb')
print(f"\n{'Gene':<15s} {'n_reads':>8s} {'n_loci':>7s} {'m6A/kb':>8s} {'RL':>6s}")
print("-" * 50)
for _, row in top_genes.iterrows():
    print(f"  {row['overlapping_genes']:<15s} {row['n_reads']:>8.0f} {row['n_loci']:>7.0f} "
          f"{row['mean_m6a_per_kb']:>8.2f} {row['mean_read_length']:>6.0f}")

# Bottom m6A genes
print(f"\nBottom 30 (lowest m6A/kb):")
bottom_genes = gene_reliable.nsmallest(30, 'mean_m6a_per_kb')
print(f"\n{'Gene':<15s} {'n_reads':>8s} {'n_loci':>7s} {'m6A/kb':>8s} {'RL':>6s}")
print("-" * 50)
for _, row in bottom_genes.iterrows():
    print(f"  {row['overlapping_genes']:<15s} {row['n_reads']:>8.0f} {row['n_loci']:>7.0f} "
          f"{row['mean_m6a_per_kb']:>8.2f} {row['mean_read_length']:>6.0f}")

# =====================================================================
# 7. Gene length as confound check
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 5: CONFOUND CHECK — GENE LENGTH vs m6A/kb")
print("=" * 80)

# Number of L1 loci per gene correlates with gene length
# Check if m6A/kb correlates with n_loci (proxy for gene length)
r, p = stats.spearmanr(gene_reliable['n_loci'], gene_reliable['mean_m6a_per_kb'])
print(f"\n  n_loci vs m6A/kb: Spearman r={r:.3f}, p={p:.3f}")

r2, p2 = stats.spearmanr(gene_reliable['n_reads'], gene_reliable['mean_m6a_per_kb'])
print(f"  n_reads vs m6A/kb: Spearman r={r2:.3f}, p={p2:.3f}")

r3, p3 = stats.spearmanr(gene_reliable['mean_read_length'], gene_reliable['mean_m6a_per_kb'])
print(f"  read_length vs m6A/kb: Spearman r={r3:.3f}, p={p3:.3f}")

# Save full gene table
gene_reliable.to_csv(OUTDIR / 'host_gene_m6a_summary.tsv', sep='\t', index=False)

print(f"\nResults saved to: {OUTDIR}")
print("Done!")
