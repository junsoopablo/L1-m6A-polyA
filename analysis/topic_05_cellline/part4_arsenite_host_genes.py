#!/usr/bin/env python3
"""
Part 4 Analysis 2: Arsenite-responsive L1 loci → host gene function
- Compare host genes of L1 loci affected by arsenite poly(A) shortening
- Ancient (affected) vs Young (immune) host gene function
- High-psi vs Low-psi L1 host gene function
- GO enrichment of arsenite-responsive L1 host genes
"""

import pandas as pd
import numpy as np
import os
import gseapy
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{BASE}/results_group"
TOPICDIR = f"{BASE}/analysis/01_exploration"
OUTDIR = f"{TOPICDIR}/topic_05_cellline/part4_host_gene_enrichment"
os.makedirs(OUTDIR, exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
PSI_THRESHOLD = 128  # 128/255 = 50%

# ─── Load HeLa + HeLa-Ars L1 summaries ───────────────────────────────────────
print("=" * 70)
print("Part 4 Analysis 2: Arsenite-Responsive L1 Host Gene Function")
print("=" * 70)

hela_reps = ['HeLa_1', 'HeLa_2', 'HeLa_3']
ars_reps = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

dfs = []
for rep in hela_reps + ars_reps:
    fpath = f"{RESULTS}/{rep}/g_summary/{rep}_L1_summary.tsv"
    df = pd.read_csv(fpath, sep='\t')
    df['condition'] = 'HeLa-Ars' if 'Ars' in rep else 'HeLa'
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all['l1_age'] = df_all['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')

print(f"Total reads: {len(df_all):,}")
print(f"  HeLa: {(df_all['condition']=='HeLa').sum():,}")
print(f"  HeLa-Ars: {(df_all['condition']=='HeLa-Ars').sum():,}")

# ─── Also load state classification for psi info ─────────────────────────────
state_df = pd.read_csv(f"{TOPICDIR}/topic_04_state/l1_state_classification.tsv", sep='\t')
state_df['condition'] = state_df['group'].apply(lambda x: 'HeLa-Ars' if 'Ars' in x else 'HeLa')

# Merge psi/kb and polya from state classification into main df
# Use read_id as key
state_cols = ['read_id', 'psi_per_kb', 'polya_length', 'l1_age', 'has_psi']
state_sub = state_df[state_cols].rename(columns={
    'psi_per_kb': 'psi_per_kb_state',
    'polya_length': 'polya_state',
    'l1_age': 'l1_age_state',
    'has_psi': 'has_psi_state',
})

df_merged = df_all.merge(state_sub, on='read_id', how='inner')
print(f"Merged with state data: {len(df_merged):,} reads")

# ─── Focus on intronic L1 only ────────────────────────────────────────────────
df_intr = df_merged[df_merged['TE_group'] == 'intronic'].copy()
print(f"Intronic reads: {len(df_intr):,}")

# Extract host genes
def extract_genes(genes_str):
    if pd.isna(genes_str) or genes_str == '':
        return []
    return [g.strip() for g in str(genes_str).split(';') if g.strip()]

df_intr['host_gene_list'] = df_intr['overlapping_genes'].apply(extract_genes)

# ─── 1. Locus-level poly(A) comparison ────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 1: Per-Locus Poly(A) Change Under Arsenite")
print("=" * 70)

# Group by transcript_id (locus) × condition
locus_stats = df_intr.groupby(['transcript_id', 'condition', 'gene_id', 'l1_age']).agg(
    n_reads=('read_id', 'count'),
    polya_median=('polya_length', 'median'),
    psi_mean=('psi', 'mean'),
    host_genes=('overlapping_genes', 'first'),
).reset_index()

# Pivot to get HeLa vs HeLa-Ars side by side
hela_loci = locus_stats[locus_stats['condition'] == 'HeLa'].set_index('transcript_id')
ars_loci = locus_stats[locus_stats['condition'] == 'HeLa-Ars'].set_index('transcript_id')

# Shared loci
shared = sorted(set(hela_loci.index) & set(ars_loci.index))
print(f"\nShared loci (both HeLa & HeLa-Ars): {len(shared)}")

shared_df = pd.DataFrame({
    'locus': shared,
    'gene_id': [hela_loci.loc[l, 'gene_id'] for l in shared],
    'l1_age': [hela_loci.loc[l, 'l1_age'] for l in shared],
    'host_genes': [hela_loci.loc[l, 'host_genes'] for l in shared],
    'n_hela': [hela_loci.loc[l, 'n_reads'] for l in shared],
    'n_ars': [ars_loci.loc[l, 'n_reads'] for l in shared],
    'polya_hela': [hela_loci.loc[l, 'polya_median'] for l in shared],
    'polya_ars': [ars_loci.loc[l, 'polya_median'] for l in shared],
    'psi_hela': [hela_loci.loc[l, 'psi_mean'] for l in shared],
    'psi_ars': [ars_loci.loc[l, 'psi_mean'] for l in shared],
})

shared_df['polya_delta'] = shared_df['polya_ars'] - shared_df['polya_hela']
shared_df['host_gene_clean'] = shared_df['host_genes'].apply(
    lambda x: str(x).split(';')[0] if pd.notna(x) else ''
)

# Filter to loci with >=2 reads in both conditions for robust comparison
robust = shared_df[(shared_df['n_hela'] >= 2) & (shared_df['n_ars'] >= 2)]
print(f"Robust shared loci (>=2 reads each): {len(robust)}")

# Summary
ancient_rob = robust[robust['l1_age'] == 'ancient']
young_rob = robust[robust['l1_age'] == 'young']
print(f"\n  Ancient: n={len(ancient_rob)}, median Δpoly(A)={ancient_rob['polya_delta'].median():.1f}nt")
print(f"  Young: n={len(young_rob)}, median Δpoly(A)={young_rob['polya_delta'].median():.1f}nt")

# Top shortening loci
top_shortened = robust.nsmallest(20, 'polya_delta')
print(f"\nTop 20 most-shortened loci (shared, robust):")
for _, row in top_shortened.iterrows():
    print(f"  {row['locus'][:25]:25s} | {row['host_gene_clean']:15s} | {row['l1_age']:7s} | "
          f"HeLa={row['polya_hela']:.0f} → Ars={row['polya_ars']:.0f} (Δ={row['polya_delta']:.0f}nt) | "
          f"n={row['n_hela']},{row['n_ars']}")

# ─── 2. Host genes of arsenite-affected vs unaffected loci ────────────────────
print("\n" + "=" * 70)
print("Section 2: Host Genes of Affected vs Unaffected Loci")
print("=" * 70)

# Define affected: Δpoly(A) < -20nt; unaffected: |Δ| < 10nt
AFFECTED_THRESHOLD = -20
UNAFFECTED_THRESHOLD = 10

robust['effect'] = 'neutral'
robust.loc[robust['polya_delta'] < AFFECTED_THRESHOLD, 'effect'] = 'shortened'
robust.loc[robust['polya_delta'] > UNAFFECTED_THRESHOLD, 'effect'] = 'lengthened'

for eff in ['shortened', 'neutral', 'lengthened']:
    n = (robust['effect'] == eff).sum()
    pct = n / len(robust) * 100
    print(f"  {eff}: {n} loci ({pct:.1f}%)")

# Extract host genes for each group
def get_host_genes(df_sub):
    genes = set()
    for genes_str in df_sub['host_genes']:
        if pd.notna(genes_str):
            for g in str(genes_str).split(';'):
                g = g.strip()
                if g:
                    genes.add(g)
    return sorted(genes)

shortened_genes = get_host_genes(robust[robust['effect'] == 'shortened'])
neutral_genes = get_host_genes(robust[robust['effect'] == 'neutral'])
lengthened_genes = get_host_genes(robust[robust['effect'] == 'lengthened'])

print(f"\nHost genes: shortened={len(shortened_genes)}, neutral={len(neutral_genes)}, lengthened={len(lengthened_genes)}")

# ─── 3. Read-level analysis: All HeLa-Ars intronic reads ─────────────────────
print("\n" + "=" * 70)
print("Section 3: All Intronic L1 Host Genes — HeLa vs HeLa-Ars")
print("=" * 70)

# Explode to per-gene rows
df_exploded = df_intr.explode('host_gene_list')
df_exploded = df_exploded[df_exploded['host_gene_list'].notna()]
df_exploded.rename(columns={'host_gene_list': 'host_gene'}, inplace=True)

# Per-gene poly(A) by condition
gene_polya = df_exploded.groupby(['host_gene', 'condition']).agg(
    n_reads=('read_id', 'count'),
    polya_median=('polya_length', 'median'),
    psi_mean=('psi', 'mean'),
).reset_index()

# Pivot
hela_genes = gene_polya[gene_polya['condition'] == 'HeLa'].set_index('host_gene')
ars_genes = gene_polya[gene_polya['condition'] == 'HeLa-Ars'].set_index('host_gene')

shared_genes = sorted(set(hela_genes.index) & set(ars_genes.index))
print(f"Shared host genes (both conditions): {len(shared_genes)}")

gene_comparison = pd.DataFrame({
    'host_gene': shared_genes,
    'n_hela': [hela_genes.loc[g, 'n_reads'] for g in shared_genes],
    'n_ars': [ars_genes.loc[g, 'n_reads'] for g in shared_genes],
    'polya_hela': [hela_genes.loc[g, 'polya_median'] for g in shared_genes],
    'polya_ars': [ars_genes.loc[g, 'polya_median'] for g in shared_genes],
    'psi_hela': [hela_genes.loc[g, 'psi_mean'] for g in shared_genes],
    'psi_ars': [ars_genes.loc[g, 'psi_mean'] for g in shared_genes],
})
gene_comparison['polya_delta'] = gene_comparison['polya_ars'] - gene_comparison['polya_hela']
gene_comparison = gene_comparison.sort_values('polya_delta')
gene_comparison.to_csv(f"{OUTDIR}/arsenite_host_gene_polya.tsv", sep='\t', index=False)

# Show top affected genes
print(f"\nTop 20 most-shortened host genes:")
top20 = gene_comparison[(gene_comparison['n_hela'] >= 3) & (gene_comparison['n_ars'] >= 3)].head(20)
for _, row in top20.iterrows():
    print(f"  {row['host_gene']:20s} | HeLa={row['polya_hela']:.0f} → Ars={row['polya_ars']:.0f} "
          f"(Δ={row['polya_delta']:.0f}nt) | n={int(row['n_hela'])},{int(row['n_ars'])} | "
          f"psi={row['psi_hela']:.2f}/{row['psi_ars']:.2f}")

# ─── 4. Enrichment: Arsenite-affected host genes ─────────────────────────────
print("\n" + "=" * 70)
print("Section 4: GO Enrichment — Arsenite-Affected Host Genes")
print("=" * 70)

gene_sets_to_test = [
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'KEGG_2021_Human',
    'MSigDB_Hallmark_2020',
]

def is_standard_gene(g):
    if g.startswith(('RP11-', 'RP4-', 'RP5-', 'RP3-', 'RP1-', 'AC0', 'AL0', 'AL1',
                     'AL3', 'AL4', 'AL5', 'AL6', 'AL7', 'AL8', 'AL9',
                     'AP0', 'CTC-', 'CTB-', 'CTD-', 'KB-', 'LA16')):
        return False
    return True

def run_enrichr(gene_list, label, gene_sets_list):
    all_results = []
    for gs in gene_sets_list:
        try:
            enr = gseapy.enrichr(
                gene_list=gene_list,
                gene_sets=gs,
                organism='human',
                outdir=None,
                no_plot=True,
            )
            res = enr.results.copy()
            res['database'] = gs
            res['gene_list'] = label
            all_results.append(res)
        except Exception as e:
            print(f"  WARNING: {gs} failed for {label}: {e}")
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

# Genes where L1 poly(A) shortened by >=20nt (robust: >=3 reads each)
robust_gene_comp = gene_comparison[(gene_comparison['n_hela'] >= 3) & (gene_comparison['n_ars'] >= 3)]
affected_genes = robust_gene_comp[robust_gene_comp['polya_delta'] < -20]['host_gene'].tolist()
unaffected_genes = robust_gene_comp[robust_gene_comp['polya_delta'].abs() < 10]['host_gene'].tolist()

affected_filt = [g for g in affected_genes if is_standard_gene(g)]
unaffected_filt = [g for g in unaffected_genes if is_standard_gene(g)]

print(f"Affected genes (Δ<-20, >=3 reads each): {len(affected_filt)}")
print(f"Unaffected genes (|Δ|<10, >=3 reads each): {len(unaffected_filt)}")

if len(affected_filt) >= 5:
    results_affected = run_enrichr(affected_filt, 'affected', gene_sets_to_test)
    sig = results_affected[results_affected['Adjusted P-value'] < 0.05]
    print(f"\nAffected genes enrichment: {len(sig)} significant terms")
    if len(sig) > 0:
        for _, row in sig.nsmallest(10, 'Adjusted P-value').iterrows():
            print(f"  [{row['database']}] {row['Term'][:65]} | {row['Overlap']} | adj.p={row['Adjusted P-value']:.2e}")
    else:
        print("Top 10 by raw p-value:")
        for _, row in results_affected.nsmallest(10, 'P-value').iterrows():
            print(f"  [{row['database']}] {row['Term'][:65]} | {row['Overlap']} | p={row['P-value']:.2e}")
    results_affected.to_csv(f"{OUTDIR}/enrichr_arsenite_affected.tsv", sep='\t', index=False)
else:
    print("Too few affected genes for enrichment")

# ─── 5. Young vs Ancient host genes ──────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 5: Young vs Ancient L1 Host Genes in HeLa/HeLa-Ars")
print("=" * 70)

# Young L1 host genes (from HeLa + HeLa-Ars)
young_mask = df_exploded['l1_age'] == 'young'
ancient_mask = df_exploded['l1_age'] == 'ancient'

young_genes_hela = set(df_exploded[(young_mask) & (df_exploded['condition'] == 'HeLa')]['host_gene'].unique())
young_genes_ars = set(df_exploded[(young_mask) & (df_exploded['condition'] == 'HeLa-Ars')]['host_gene'].unique())
ancient_genes_hela = set(df_exploded[(ancient_mask) & (df_exploded['condition'] == 'HeLa')]['host_gene'].unique())
ancient_genes_ars = set(df_exploded[(ancient_mask) & (df_exploded['condition'] == 'HeLa-Ars')]['host_gene'].unique())

print(f"Young L1 host genes: HeLa={len(young_genes_hela)}, HeLa-Ars={len(young_genes_ars)}")
print(f"Ancient L1 host genes: HeLa={len(ancient_genes_hela)}, HeLa-Ars={len(ancient_genes_ars)}")
print(f"Overlap: Young={len(young_genes_hela & young_genes_ars)}, Ancient={len(ancient_genes_hela & ancient_genes_ars)}")

# Young-specific enrichment (if enough genes)
young_all = sorted(young_genes_hela | young_genes_ars)
young_all_filt = [g for g in young_all if is_standard_gene(g)]
ancient_all = sorted(ancient_genes_hela | ancient_genes_ars)
ancient_all_filt = [g for g in ancient_all if is_standard_gene(g)]

print(f"\nYoung L1 host genes (filtered): {len(young_all_filt)}")
print(f"Ancient L1 host genes (filtered): {len(ancient_all_filt)}")

# Young-only genes (not in ancient)
young_only = sorted(set(young_all_filt) - set(ancient_all_filt))
print(f"Young-only host genes: {len(young_only)}")

if len(young_all_filt) >= 10:
    results_young = run_enrichr(young_all_filt, 'young_hela_ars', gene_sets_to_test)
    sig_y = results_young[results_young['Adjusted P-value'] < 0.05]
    print(f"\nYoung L1 host genes enrichment: {len(sig_y)} significant terms")
    for _, row in results_young.nsmallest(10, 'P-value').iterrows():
        sig_mark = "***" if row['Adjusted P-value'] < 0.05 else ""
        print(f"  [{row['database']}] {row['Term'][:60]} | {row['Overlap']} | p={row['P-value']:.2e} {sig_mark}")
    results_young.to_csv(f"{OUTDIR}/enrichr_young_hela_ars.tsv", sep='\t', index=False)

# ─── 6. Psi-stratified host gene analysis ────────────────────────────────────
print("\n" + "=" * 70)
print("Section 6: High-Psi vs Low-Psi L1 Host Gene Comparison")
print("=" * 70)

# Split by psi (using psi column from summary = binary 0/1)
# Use psi_per_kb from state data for more granular split
ars_intr = df_intr[df_intr['condition'] == 'HeLa-Ars'].copy()

# Merge psi_per_kb
ars_with_psi = ars_intr.merge(
    state_df[['read_id', 'psi_per_kb']].drop_duplicates(),
    on='read_id', how='left'
)

psi_median = ars_with_psi['psi_per_kb'].median()
ars_with_psi['psi_group'] = np.where(ars_with_psi['psi_per_kb'] > psi_median, 'high_psi', 'low_psi')

# Explode and get host genes per group
ars_exp = ars_with_psi.explode('host_gene_list')
ars_exp = ars_exp[ars_exp['host_gene_list'].notna()]
ars_exp.rename(columns={'host_gene_list': 'host_gene'}, inplace=True)

# Host gene poly(A) by psi group
psi_gene = ars_exp.groupby(['host_gene', 'psi_group']).agg(
    n_reads=('read_id', 'count'),
    polya_median=('polya_length', 'median'),
).reset_index()

high_psi_genes = set(psi_gene[psi_gene['psi_group'] == 'high_psi']['host_gene'])
low_psi_genes = set(psi_gene[psi_gene['psi_group'] == 'low_psi']['host_gene'])

print(f"High-psi L1 host genes: {len(high_psi_genes)}")
print(f"Low-psi L1 host genes: {len(low_psi_genes)}")
print(f"Overlap: {len(high_psi_genes & low_psi_genes)}")

# Check if high-psi reads have different poly(A) per host gene
both_psi = sorted(high_psi_genes & low_psi_genes)
if len(both_psi) > 10:
    psi_pivot = psi_gene[psi_gene['host_gene'].isin(both_psi)].pivot_table(
        index='host_gene', columns='psi_group', values='polya_median'
    ).dropna()
    psi_pivot['delta'] = psi_pivot['high_psi'] - psi_pivot['low_psi']
    mean_delta = psi_pivot['delta'].mean()
    med_delta = psi_pivot['delta'].median()
    t, p = stats.ttest_1samp(psi_pivot['delta'], 0)
    print(f"\nPer-gene poly(A) delta (high_psi - low_psi): mean={mean_delta:.1f}, median={med_delta:.1f}, p={p:.2e}")
    print(f"  High-psi L1 reads → {'longer' if mean_delta > 0 else 'shorter'} poly(A) within same host gene")

# ─── 7. Key DDR/Stress/Immune genes under arsenite ───────────────────────────
print("\n" + "=" * 70)
print("Section 7: Key Functional Genes — L1 Poly(A) Under Arsenite")
print("=" * 70)

# Check specific genes
KEY_GENES = {
    'DDR': ['BRCA1', 'FANCC', 'ATR', 'ATM', 'RAD51', 'RAD50', 'ZRANB3', 'SMARCAL1', 'FANCI', 'FANCD2'],
    'Immune': ['STAT1', 'JAK1', 'JAK2', 'TRIM5', 'IFNGR2', 'IRF1'],
    'Stress': ['NFE2L2', 'SOD2', 'HSF1', 'HSP90AA1', 'EIF2AK3', 'EIF2AK4'],
    'Neuronal': ['GPHN', 'DLG2', 'GABRG3', 'CTNND2'],
}

for category, genes in KEY_GENES.items():
    print(f"\n  [{category}]")
    for g in genes:
        hela_sub = df_exploded[(df_exploded['host_gene'] == g) & (df_exploded['condition'] == 'HeLa')]
        ars_sub = df_exploded[(df_exploded['host_gene'] == g) & (df_exploded['condition'] == 'HeLa-Ars')]
        n_h = len(hela_sub)
        n_a = len(ars_sub)
        if n_h + n_a == 0:
            continue
        polya_h = hela_sub['polya_length'].median() if n_h > 0 else np.nan
        polya_a = ars_sub['polya_length'].median() if n_a > 0 else np.nan
        psi_h = hela_sub['psi'].mean() if n_h > 0 else np.nan
        psi_a = ars_sub['psi'].mean() if n_a > 0 else np.nan
        delta = polya_a - polya_h if (n_h > 0 and n_a > 0) else np.nan
        print(f"    {g:15s} | HeLa: n={n_h}, polyA={polya_h:.0f}nt, psi={psi_h:.2f} | "
              f"Ars: n={n_a}, polyA={polya_a:.0f}nt, psi={psi_a:.2f} | Δ={delta:+.0f}nt" if not np.isnan(delta)
              else f"    {g:15s} | HeLa: n={n_h} | Ars: n={n_a}")

# ─── 8. Summary statistics ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Overall numbers
hela_intr = df_intr[df_intr['condition'] == 'HeLa']
ars_intr_df = df_intr[df_intr['condition'] == 'HeLa-Ars']

hela_genes_all = set()
ars_genes_all = set()
for genes_str in hela_intr['overlapping_genes']:
    for g in extract_genes(genes_str):
        hela_genes_all.add(g)
for genes_str in ars_intr_df['overlapping_genes']:
    for g in extract_genes(genes_str):
        ars_genes_all.add(g)

print(f"\nIntronic L1 host genes:")
print(f"  HeLa: {len(hela_genes_all)} genes from {len(hela_intr)} reads")
print(f"  HeLa-Ars: {len(ars_genes_all)} genes from {len(ars_intr_df)} reads")
print(f"  Shared: {len(hela_genes_all & ars_genes_all)}")
print(f"  HeLa-only: {len(hela_genes_all - ars_genes_all)}")
print(f"  Ars-only: {len(ars_genes_all - hela_genes_all)}")

# Poly(A) summary
print(f"\nPoly(A) by condition (all intronic L1):")
for cond in ['HeLa', 'HeLa-Ars']:
    sub = df_intr[df_intr['condition'] == cond]
    print(f"  {cond}: median={sub['polya_length'].median():.1f}nt, n={len(sub)}")

# Save robust locus-level data
robust.to_csv(f"{OUTDIR}/arsenite_locus_polya_comparison.tsv", sep='\t', index=False)
print(f"\nResults saved to: {OUTDIR}")
