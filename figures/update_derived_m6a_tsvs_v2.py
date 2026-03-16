#!/usr/bin/env python3
"""
Update m6A columns in derived TSVs after threshold change (v2).
Uses Part3 caches (already at threshold 204) to update read_id-based m6A values.
"""
import os, glob
import pandas as pd
import numpy as np

BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration"
RESULTS = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group"

# Build global read_id -> m6A lookup from all Part3 caches
print("Loading Part3 L1 caches...")
frames = []
for f in sorted(glob.glob(f'{BASE}/topic_05_cellline/part3_l1_per_read_cache/*_l1_per_read.tsv')):
    frames.append(pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high']))
df_lookup = pd.concat(frames, ignore_index=True)
df_lookup['m6a_per_kb'] = df_lookup['m6a_sites_high'] / (df_lookup['read_length'] / 1000)
lookup_m6a = df_lookup.set_index('read_id')['m6a_per_kb']
lookup_sites = df_lookup.set_index('read_id')['m6a_sites_high']
print(f"  Loaded {len(lookup_m6a)} reads")

# =====================================================================
# 1. Update regulatory_l1_per_read.tsv (Fig S7)
# =====================================================================
print("\n--- 1. regulatory_l1_per_read.tsv ---")
per_read_f = f'{BASE}/topic_08_regulatory_chromatin/stress_gene_analysis/regulatory_l1_per_read.tsv'
df_pr = pd.read_csv(per_read_f, sep='\t')
matched = df_pr['read_id'].isin(lookup_m6a.index)
df_pr.loc[matched, 'm6a_per_kb'] = df_pr.loc[matched, 'read_id'].map(lookup_m6a).values
df_pr.loc[matched, 'm6a_sites_high'] = df_pr.loc[matched, 'read_id'].map(lookup_sites).values
df_pr.to_csv(per_read_f, sep='\t', index=False)
print(f"  Updated: {matched.sum()}/{len(df_pr)} matched. m6A/kb median={df_pr['m6a_per_kb'].median():.3f}")

# =====================================================================
# 2. Reaggregate gene_polya_delta.tsv (Fig S7)
# =====================================================================
print("\n--- 2. gene_polya_delta.tsv ---")
gene_delta_f = f'{BASE}/topic_08_regulatory_chromatin/regulatory_stress_response/gene_polya_delta.tsv'
df_gd = pd.read_csv(gene_delta_f, sep='\t')

# Reaggregate from updated per-read file
# For each host_gene, compute m6a_hela (mean of HeLa reads), m6a_ars (mean of HeLa-Ars reads)
# Get host gene from overlapping_genes in L1 summary
hela_pr = df_pr[df_pr['condition'] == 'normal']
ars_pr = df_pr[df_pr['condition'] == 'stress']

# We need to map reads to host genes. Use L1 summary overlapping_genes.
l1_frames = []
for group in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    f = f'{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv'
    if os.path.exists(f):
        l1_frames.append(pd.read_csv(f, sep='\t', usecols=['read_id', 'overlapping_genes', 'qc_tag']))
df_l1 = pd.concat(l1_frames, ignore_index=True)
df_l1 = df_l1[df_l1['qc_tag'] == 'PASS']

# Join per_read with L1 summary to get overlapping_genes
df_pr_genes = df_pr.merge(df_l1[['read_id', 'overlapping_genes']], on='read_id', how='left')
df_pr_genes = df_pr_genes[df_pr_genes['overlapping_genes'].notna() & (df_pr_genes['overlapping_genes'] != '.')]

# Explode overlapping_genes (may be comma-separated)
df_pr_genes['host_gene'] = df_pr_genes['overlapping_genes'].str.split(',')
df_pr_genes = df_pr_genes.explode('host_gene')
df_pr_genes['host_gene'] = df_pr_genes['host_gene'].str.strip()

# Reaggregate by host_gene
updated_rows = []
for gene in df_gd['host_gene'].unique():
    gene_reads = df_pr_genes[df_pr_genes['host_gene'] == gene]
    hela_reads = gene_reads[gene_reads['condition'] == 'normal']
    ars_reads = gene_reads[gene_reads['condition'] == 'stress']

    old_row = df_gd[df_gd['host_gene'] == gene].iloc[0]
    row = old_row.to_dict()

    if len(hela_reads) > 0:
        row['m6a_hela'] = hela_reads['m6a_per_kb'].mean()
    if len(ars_reads) > 0:
        row['m6a_ars'] = ars_reads['m6a_per_kb'].mean()
    if len(hela_reads) > 0 or len(ars_reads) > 0:
        all_m6a = gene_reads['m6a_per_kb'].values
        row['m6a_avg'] = np.mean(all_m6a) if len(all_m6a) > 0 else old_row['m6a_avg']

    updated_rows.append(row)

df_gd_new = pd.DataFrame(updated_rows)
df_gd_new.to_csv(gene_delta_f, sep='\t', index=False)
print(f"  Updated: {len(df_gd_new)} genes. m6A avg median={df_gd_new['m6a_avg'].median():.3f}")

# =====================================================================
# 3. DDR gene m6A stats (Fig S4)
# =====================================================================
print("\n--- 3. ddr_gene_m6a_stats.tsv ---")
ddr_f = f'{BASE}/topic_05_cellline/part4_ddr_m6a_integration/ddr_gene_m6a_stats.tsv'
df_ddr = pd.read_csv(ddr_f, sep='\t')

# Load ALL L1 summaries to find reads per DDR gene
all_l1_frames = []
for f in sorted(glob.glob(f'{RESULTS}/*/g_summary/*_L1_summary.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'qc_tag', 'overlapping_genes'])
    all_l1_frames.append(tmp)
df_all_l1 = pd.concat(all_l1_frames, ignore_index=True)
df_all_l1 = df_all_l1[df_all_l1['qc_tag'] == 'PASS']

# Explode overlapping_genes
df_all_l1 = df_all_l1[df_all_l1['overlapping_genes'].notna() & (df_all_l1['overlapping_genes'] != '.')]
df_all_l1['gene'] = df_all_l1['overlapping_genes'].str.split(',')
df_all_l1 = df_all_l1.explode('gene')
df_all_l1['gene'] = df_all_l1['gene'].str.strip()

# Add m6A from Part3 cache
df_all_l1 = df_all_l1.merge(df_lookup[['read_id', 'm6a_per_kb']].drop_duplicates('read_id'),
                              on='read_id', how='left')

# Reaggregate for DDR genes
updated_ddr = []
for _, row in df_ddr.iterrows():
    gene_name = row['gene']
    gene_reads = df_all_l1[df_all_l1['gene'] == gene_name]
    new_row = row.to_dict()
    if len(gene_reads) > 0:
        valid = gene_reads['m6a_per_kb'].dropna()
        if len(valid) > 0:
            new_row['m6a_kb_median'] = valid.median()
            new_row['m6a_kb_mean'] = valid.mean()
    updated_ddr.append(new_row)

df_ddr_new = pd.DataFrame(updated_ddr)
df_ddr_new.to_csv(ddr_f, sep='\t', index=False)
print(f"  Updated: {len(df_ddr_new)} genes. m6A/kb median of medians={df_ddr_new['m6a_kb_median'].median():.3f}")

print("\nAll derived TSV updates complete.")
