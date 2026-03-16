#!/usr/bin/env python3
"""
Generate Supplementary Table: Intergenic vs Intronic L1 comparison.
Uses ChromHMM annotated file (has genomic_context) + L1 summaries + Part3 cache.
Also extracts MAPQ from BAM files for a subset.
"""

import pandas as pd
import numpy as np
from scipy import stats
import pysam
import os, glob

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
OUTDIR = f'{BASE}/analysis/01_exploration/topic_09_disease_variants'

# ── Load all L1 summaries and derive genomic context ──
print("Loading L1 summaries...")
summary_files = glob.glob(f'{BASE}/results_group/*/g_summary/*_L1_summary.tsv')
# Exclude locus summaries
summary_files = [f for f in summary_files if 'locus_summary' not in f]

dfs = []
for f in sorted(summary_files):
    df = pd.read_csv(f, sep='\t')
    group = os.path.basename(f).replace('_L1_summary.tsv', '')
    df['group'] = group
    dfs.append(df)

all_l1 = pd.concat(dfs, ignore_index=True)
print(f"  Total reads: {len(all_l1):,}")

# Derive genomic context from overlapping_genes
all_l1['context'] = all_l1['overlapping_genes'].apply(
    lambda x: 'intergenic' if pd.isna(x) or str(x).strip() == '' else 'intronic')

# Filter to PASS L1 only (comments MAU=matched, YAY=good, etc; qc_tag=PASS from nanopolish)
# Actually keep all reads — PASS filter is already applied in the summary pipeline
print(f"  Intronic: {(all_l1['context']=='intronic').sum():,}")
print(f"  Intergenic: {(all_l1['context']=='intergenic').sum():,}")

# Young vs ancient
young_subfamilies = ['L1HS', 'L1PA2', 'L1PA3']
all_l1['is_young'] = all_l1['gene_id'].isin(young_subfamilies)
all_l1['age'] = all_l1['is_young'].map({True: 'young', False: 'ancient'})

# Cell line and condition
def get_cellline_condition(group):
    g = group.lower()
    if 'ars' in g:
        cl = group.split('_Ars')[0].split('-Ars')[0]
        return cl + '-Ars', 'stress'
    elif 'ev' in g and 'mcf7' in g:
        return 'MCF7-EV', 'normal'
    else:
        # Remove trailing _N (replicate number)
        parts = group.rsplit('_', 1)
        return parts[0], 'normal'

all_l1[['cellline', 'condition']] = all_l1['group'].apply(
    lambda g: pd.Series(get_cellline_condition(g)))

# ── Load Part3 cache for m6A/kb ──
print("Loading Part3 m6A cache...")
cache_dir = f'{BASE}/analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
cache_files = glob.glob(f'{cache_dir}/*.tsv')

cache_dfs = []
for f in sorted(cache_files):
    df = pd.read_csv(f, sep='\t')
    cache_dfs.append(df)

if cache_dfs:
    cache = pd.concat(cache_dfs, ignore_index=True)
    # Find m6A column
    m6a_col = None
    for c in cache.columns:
        if 'm6a' in c.lower() and 'kb' in c.lower():
            m6a_col = c
            break
    if m6a_col is None:
        # Try m6a_sites_high / read_length
        print("  No m6A/kb column found, computing from sites and length")
        if 'm6a_sites_high' in cache.columns and 'read_length' in cache.columns:
            cache['m6a_per_kb'] = cache['m6a_sites_high'] / cache['read_length'] * 1000
            m6a_col = 'm6a_per_kb'

    if m6a_col:
        print(f"  Using m6A column: {m6a_col}")
        all_l1 = all_l1.merge(
            cache[['read_id', m6a_col]].rename(columns={m6a_col: 'm6a_per_kb'}),
            on='read_id', how='left')
        print(f"  m6A data: {all_l1['m6a_per_kb'].notna().sum():,} / {len(all_l1):,} reads")

# ── Extract MAPQ from BAM files (HeLa replicates only for speed) ──
print("\nExtracting MAPQ from BAM files (HeLa)...")
hela_groups = [g for g in all_l1['group'].unique() if g.startswith('HeLa')]
mapq_data = {}

for group in sorted(hela_groups):
    bam_path = f'{BASE}/results_group/{group}/h_mafia/{group}.mAFiA.reads.bam'
    if not os.path.exists(bam_path):
        # Try alternative BAM paths
        alt_bams = glob.glob(f'{BASE}/results_group/{group}/*/*.bam')
        bam_path = alt_bams[0] if alt_bams else None

    if bam_path and os.path.exists(bam_path):
        try:
            bam = pysam.AlignmentFile(bam_path, 'rb')
            for read in bam.fetch(until_eof=True):
                if not read.is_unmapped and not read.is_secondary:
                    mapq_data[read.query_name] = read.mapping_quality
            bam.close()
            print(f"  {group}: {sum(1 for r in all_l1[all_l1['group']==group]['read_id'] if r in mapq_data)} reads with MAPQ")
        except Exception as e:
            print(f"  {group}: BAM error - {e}")

all_l1['mapq'] = all_l1['read_id'].map(mapq_data)
n_mapq = all_l1['mapq'].notna().sum()
print(f"  Total reads with MAPQ: {n_mapq:,}")

# ── Build comparison table ──
print("\nBuilding comparison table...")

rows = []

def add_row(metric, intronic_val, intergenic_val, p_val='', note=''):
    rows.append({
        'Metric': metric,
        'Intronic': str(intronic_val),
        'Intergenic': str(intergenic_val),
        'P': str(p_val),
        'Note': note
    })

intr = all_l1[all_l1['context'] == 'intronic']
inter = all_l1[all_l1['context'] == 'intergenic']
intr_anc = intr[intr['age'] == 'ancient']
inter_anc = inter[inter['age'] == 'ancient']

# 1. Read counts
add_row('Total reads',
        f'{len(intr):,} ({len(intr)/len(all_l1)*100:.1f}%)',
        f'{len(inter):,} ({len(inter)/len(all_l1)*100:.1f}%)')

# 2. Young L1 %
add_row('Young L1 (%)',
        f'{intr["is_young"].mean()*100:.1f}',
        f'{inter["is_young"].mean()*100:.1f}')

# 3. Unique loci
if 'chr' in all_l1.columns:
    # Use te_chr, te_start, te_end for locus identity
    intr_loci = intr.drop_duplicates(subset=['te_chr', 'te_start', 'te_end']).shape[0]
    inter_loci = inter.drop_duplicates(subset=['te_chr', 'te_start', 'te_end']).shape[0]
    add_row('Unique L1 loci',
            f'{intr_loci:,}', f'{inter_loci:,}')

# 4. MAPQ (HeLa + HeLa-Ars, ancient only)
mapq_intr = intr_anc['mapq'].dropna()
mapq_inter = inter_anc['mapq'].dropna()
if len(mapq_intr) > 10 and len(mapq_inter) > 10:
    u_stat, u_p = stats.mannwhitneyu(mapq_intr, mapq_inter)
    add_row('MAPQ, median (ancient, HeLa)',
            f'{mapq_intr.median():.0f}', f'{mapq_inter.median():.0f}',
            f'{u_p:.2e}')
    add_row('MAPQ ≥ 60 (%, ancient, HeLa)',
            f'{(mapq_intr >= 60).mean()*100:.1f}',
            f'{(mapq_inter >= 60).mean()*100:.1f}')
    add_row('MAPQ = 0 (%, ancient, HeLa)',
            f'{(mapq_intr == 0).mean()*100:.1f}',
            f'{(mapq_inter == 0).mean()*100:.1f}')

# 5. Read length (ancient)
rl_intr = intr_anc['read_length'].dropna()
rl_inter = inter_anc['read_length'].dropna()
u_stat, u_p = stats.mannwhitneyu(rl_intr, rl_inter)
add_row('Read length, median bp (ancient)',
        f'{rl_intr.median():.0f}', f'{rl_inter.median():.0f}',
        f'{u_p:.2e}')

# 6. Poly(A) (ancient)
pa_intr = intr_anc['polya_length'].dropna()
pa_inter = inter_anc['polya_length'].dropna()
u_stat, u_p = stats.mannwhitneyu(pa_intr, pa_inter)
add_row('Poly(A) length, median nt (ancient)',
        f'{pa_intr.median():.1f}', f'{pa_inter.median():.1f}',
        f'{u_p:.2e}')

# 7. m6A/kb (ancient)
if 'm6a_per_kb' in all_l1.columns:
    m6a_intr = intr_anc['m6a_per_kb'].dropna()
    m6a_inter = inter_anc['m6a_per_kb'].dropna()
    if len(m6a_intr) > 10 and len(m6a_inter) > 10:
        u_stat, u_p = stats.mannwhitneyu(m6a_intr, m6a_inter)
        add_row('m6A/kb, median (ancient)',
                f'{m6a_intr.median():.2f}', f'{m6a_inter.median():.2f}',
                f'{u_p:.2e}')

# 8. Arsenite response: ancient intronic vs intergenic
hela_n = all_l1[(all_l1['cellline'] == 'HeLa') & (all_l1['age'] == 'ancient')]
hela_s = all_l1[(all_l1['cellline'] == 'HeLa-Ars') & (all_l1['age'] == 'ancient')]

for ctx in ['intronic', 'intergenic']:
    norm = hela_n[hela_n['context'] == ctx]['polya_length'].dropna()
    stress = hela_s[hela_s['context'] == ctx]['polya_length'].dropna()
    if len(norm) > 5 and len(stress) > 5:
        delta = stress.median() - norm.median()
        u_stat, u_p = stats.mannwhitneyu(norm, stress)
        add_row(f'Arsenite Δpoly(A) (ancient {ctx})',
                f'{norm.median():.1f} → {stress.median():.1f}',
                f'Δ = {delta:+.1f} nt',
                f'{u_p:.2e}',
                f'n = {len(norm)}, {len(stress)}')

# 9. Cross-CL intergenic fraction
cl_fracs = []
for cl in sorted(all_l1['cellline'].unique()):
    cl_data = all_l1[all_l1['cellline'] == cl]
    if len(cl_data) < 100:
        continue
    frac = (cl_data['context'] == 'intergenic').mean() * 100
    cl_fracs.append(frac)

add_row('Intergenic fraction across cell lines',
        f'median {np.median(cl_fracs):.1f}%',
        f'range {min(cl_fracs):.1f}--{max(cl_fracs):.1f}%')

# ── Print and save ──
table_df = pd.DataFrame(rows)
outpath = f'{OUTDIR}/supplementary_table_intergenic_vs_intronic.tsv'
table_df.to_csv(outpath, sep='\t', index=False)

print("\n" + "=" * 90)
print("SUPPLEMENTARY TABLE: Intergenic vs Intronic L1 Comparison")
print("=" * 90)
for _, row in table_df.iterrows():
    print(f"  {row['Metric']:45s} | {row['Intronic']:25s} | {row['Intergenic']:25s} | {row['P']}")
print(f"\nSaved to {outpath}")
