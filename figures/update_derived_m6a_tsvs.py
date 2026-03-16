#!/usr/bin/env python3
"""
Update m6A columns in derived TSVs after threshold change.
Uses Part3 caches (already at threshold 204) to update read_id-based m6A values.
"""
import os, glob
import pandas as pd
import numpy as np

BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration"

# Build global read_id -> m6A lookup from all Part3 caches
print("Loading Part3 L1 caches...")
frames = []
for f in sorted(glob.glob(f'{BASE}/topic_05_cellline/part3_l1_per_read_cache/*_l1_per_read.tsv')):
    frames.append(pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high']))
df_lookup = pd.concat(frames, ignore_index=True)
df_lookup['m6a_per_kb'] = df_lookup['m6a_sites_high'] / (df_lookup['read_length'] / 1000)
lookup = df_lookup.set_index('read_id')[['m6a_per_kb', 'm6a_sites_high']]
print(f"  Loaded {len(lookup)} reads")


def update_file(fpath, m6a_col='m6a_per_kb', sites_col=None, id_col='read_id'):
    """Update m6A column(s) in a TSV by joining with Part3 cache lookup."""
    if not os.path.exists(fpath):
        print(f"  SKIP (not found): {fpath}")
        return
    df = pd.read_csv(fpath, sep='\t')
    if id_col not in df.columns:
        print(f"  SKIP (no {id_col}): {fpath}")
        return
    if m6a_col not in df.columns:
        print(f"  SKIP (no {m6a_col}): {fpath}")
        return

    n = len(df)
    matched = df[id_col].isin(lookup.index)
    n_match = matched.sum()

    # Update m6A/kb
    df.loc[matched, m6a_col] = df.loc[matched, id_col].map(lookup['m6a_per_kb']).values
    if sites_col and sites_col in df.columns:
        df.loc[matched, sites_col] = df.loc[matched, id_col].map(lookup['m6a_sites_high']).values

    df.to_csv(fpath, sep='\t', index=False)
    print(f"  Updated {fpath}: {n_match}/{n} matched. m6A/kb median={df[m6a_col].median():.3f}")


# 1. DDR gene m6A stats — per-gene aggregated, has read_id? No, it's per-gene.
# Need to regenerate from per-read data.
ddr_f = f'{BASE}/topic_05_cellline/part4_ddr_m6a_integration/ddr_gene_m6a_stats.tsv'
if os.path.exists(ddr_f):
    df_ddr = pd.read_csv(ddr_f, sep='\t')
    print(f"\n--- DDR gene stats ---")
    print(f"  Original columns: {list(df_ddr.columns)}")

    # This file is per-gene, not per-read. Check if there's source per-read data.
    ddr_reads_f = f'{BASE}/topic_05_cellline/part4_ddr_m6a_integration/ddr_l1_reads.tsv'
    if os.path.exists(ddr_reads_f):
        df_ddr_reads = pd.read_csv(ddr_reads_f, sep='\t')
        update_file(ddr_reads_f, 'm6a_per_kb')

        # Reaggregate per gene
        matched = df_ddr_reads['read_id'].isin(lookup.index)
        df_ddr_reads.loc[matched, 'm6a_per_kb'] = df_ddr_reads.loc[matched, 'read_id'].map(lookup['m6a_per_kb']).values

        if 'gene' in df_ddr_reads.columns:
            gene_stats = df_ddr_reads.groupby('gene').agg(
                n_reads=('read_id', 'count'),
                m6a_kb_median=('m6a_per_kb', 'median'),
                m6a_kb_mean=('m6a_per_kb', 'mean'),
            ).reset_index()

            # Preserve other columns from original
            if 'pathway' in df_ddr.columns:
                gene_stats = gene_stats.merge(df_ddr[['gene', 'pathway']].drop_duplicates(), on='gene', how='left')

            gene_stats.to_csv(ddr_f, sep='\t', index=False)
            print(f"  DDR stats regenerated: {len(gene_stats)} genes, m6A/kb median of medians={gene_stats['m6a_kb_median'].median():.3f}")
    else:
        print(f"  No per-read DDR data found; skipping DDR stats update")

# 2. Regulatory stress response — gene_polya_delta.tsv
print(f"\n--- Regulatory stress gene ---")
gene_delta_f = f'{BASE}/topic_08_regulatory_chromatin/regulatory_stress_response/gene_polya_delta.tsv'
if os.path.exists(gene_delta_f):
    df_gd = pd.read_csv(gene_delta_f, sep='\t')
    if 'm6a_avg' in df_gd.columns and 'gene_id' in df_gd.columns:
        # Per-gene aggregated. Need per-read source to update.
        # Check for per-read file
        per_read_f = f'{BASE}/topic_08_regulatory_chromatin/stress_gene_analysis/regulatory_l1_per_read.tsv'
        if os.path.exists(per_read_f):
            print(f"  Updating per-read file first...")
            update_file(per_read_f, 'm6a_per_kb')

            # Reaggregate gene-level m6A
            df_pr = pd.read_csv(per_read_f, sep='\t')
            if 'gene_id' in df_pr.columns and 'm6a_per_kb' in df_pr.columns:
                gene_m6a = df_pr.groupby('gene_id')['m6a_per_kb'].mean()
                matched = df_gd['gene_id'].isin(gene_m6a.index)
                df_gd.loc[matched, 'm6a_avg'] = df_gd.loc[matched, 'gene_id'].map(gene_m6a).values
                df_gd.to_csv(gene_delta_f, sep='\t', index=False)
                print(f"  Gene delta updated: {matched.sum()} genes, m6A avg median={df_gd['m6a_avg'].median():.3f}")
        else:
            print(f"  No per-read file found; skipping")
    else:
        print(f"  Columns: {list(df_gd.columns)}, no m6a_avg or gene_id found")

print("\nAll derived TSV updates complete.")
