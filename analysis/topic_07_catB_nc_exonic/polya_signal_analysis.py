#!/usr/bin/env python3
"""Poly(A) signal motif analysis for L1 elements.

Extracts 3' end sequences of L1 elements, classifies poly(A) signal motifs,
and compares young vs ancient, PASS vs Cat B, shortened vs not shortened.
"""

import re
import pysam
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
OUTDIR = TOPIC_07 / 'polya_signal_analysis'
OUTDIR.mkdir(exist_ok=True)

GENOME = PROJECT / 'reference/Human.fasta'
TE_GTF = PROJECT / 'reference/hg38_rmsk_TE.gtf'

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Canonical and variant poly(A) signals (Beaudoing et al. 2000, Tian et al. 2005)
# Ranked by strength/frequency in human genes
PAS_CANONICAL = 'AATAAA'
PAS_VARIANTS = [
    'ATTAAA',  # most common variant (~15%)
    'AGTAAA',  # ~4%
    'TATAAA',  # ~3%
    'CATAAA',  # ~1.5%
    'GATAAA',  # ~1%
    'AATATA',  # ~1%
    'AATACA',  # ~1%
    'AATAGA',  # ~0.5%
    'AAAAAG',  # ~0.5%
    'ACTAAA',  # ~0.5%
    'AATGAA',  # ~0.5%
    'AATAAT',  # rare
]
ALL_PAS = [PAS_CANONICAL] + PAS_VARIANTS

CELL_LINES = {
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
}

# =============================================================================
# 1. Build locus → PAS mapping from genome
# =============================================================================
print("Step 1: Loading L1 element annotations with strand info...")

# Parse TE GTF for L1 family elements
l1_elements = {}
with open(TE_GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.rstrip('\n').split('\t')
        if len(fields) < 9:
            continue
        if 'family_id "L1"' not in fields[8]:
            continue
        chrom = fields[0]
        start = int(fields[3]) - 1  # 0-based
        end = int(fields[4])
        strand = fields[6]
        m_tid = re.search(r'transcript_id "([^"]*)"', fields[8])
        m_gid = re.search(r'gene_id "([^"]*)"', fields[8])
        if m_tid:
            tid = m_tid.group(1)
            gid = m_gid.group(1) if m_gid else ''
            l1_elements[tid] = {
                'chr': chrom, 'start': start, 'end': end,
                'strand': strand, 'subfamily': gid, 'locus_id': tid,
            }

print(f"  Loaded {len(l1_elements)} L1 elements")

# =============================================================================
# 2. Extract 3' end sequences and scan for PAS
# =============================================================================
print("\nStep 2: Extracting 3' end sequences and scanning for PAS...")

# Open genome
genome = pysam.FastaFile(str(GENOME))

SCAN_WINDOW = 50  # bp upstream of 3' end to scan for PAS

pas_results = []
n_processed = 0
n_found = 0

for tid, info in l1_elements.items():
    chrom = info['chr']
    start = info['start']
    end = info['end']
    strand = info['strand']
    length = end - start

    # Define 3' end region
    if strand == '+':
        # 3' end is at 'end'
        scan_start = max(end - SCAN_WINDOW, start)
        scan_end = end
    else:
        # 3' end is at 'start'
        scan_start = start
        scan_end = min(start + SCAN_WINDOW, end)

    try:
        seq = genome.fetch(chrom, scan_start, scan_end).upper()
    except Exception:
        continue

    if strand == '-':
        # Reverse complement
        comp = str.maketrans('ACGT', 'TGCA')
        seq = seq.translate(comp)[::-1]

    # Scan for PAS motifs
    best_pas = 'none'
    best_pas_rank = len(ALL_PAS) + 1

    for rank, pas in enumerate(ALL_PAS):
        if pas in seq:
            if rank < best_pas_rank:
                best_pas = pas
                best_pas_rank = rank

    is_young = info['subfamily'] in YOUNG
    pas_results.append({
        'locus_id': tid,
        'subfamily': info['subfamily'],
        'chr': chrom,
        'start': start,
        'end': end,
        'strand': strand,
        'length': length,
        'is_young': is_young,
        'age': 'young' if is_young else 'ancient',
        'best_pas': best_pas,
        'has_canonical': PAS_CANONICAL in seq,
        'has_any_pas': best_pas != 'none',
        'seq_3prime': seq,
    })

    n_processed += 1
    if best_pas != 'none':
        n_found += 1

genome.close()

pas_df = pd.DataFrame(pas_results)
print(f"  Processed: {n_processed}")
print(f"  With any PAS: {n_found} ({n_found/n_processed*100:.1f}%)")

# =============================================================================
# 3. Overall PAS distribution: young vs ancient
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: PAS MOTIF — YOUNG vs ANCIENT (all 1M L1 elements)")
print("=" * 70)

for age in ['young', 'ancient']:
    sub = pas_df[pas_df['age'] == age]
    n = len(sub)
    canon = sub['has_canonical'].sum()
    any_pas = sub['has_any_pas'].sum()
    print(f"\n  {age} (n={n}):")
    print(f"    Canonical AATAAA: {canon} ({canon/n*100:.1f}%)")
    print(f"    Any PAS variant:  {any_pas} ({any_pas/n*100:.1f}%)")
    print(f"    No PAS:           {n-any_pas} ({(n-any_pas)/n*100:.1f}%)")
    # Top PAS
    top = sub['best_pas'].value_counts().head(8)
    for pas, cnt in top.items():
        print(f"      {pas}: {cnt} ({cnt/n*100:.1f}%)")

# Chi-square
ct = pd.crosstab(pas_df['age'], pas_df['has_canonical'])
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"\n  Young vs Ancient canonical PAS: chi2={chi2:.1f}, p={p:.2e}")

ct2 = pd.crosstab(pas_df['age'], pas_df['has_any_pas'])
chi2_2, p2, _, _ = stats.chi2_contingency(ct2)
print(f"  Young vs Ancient any PAS: chi2={chi2_2:.1f}, p={p2:.2e}")

# =============================================================================
# 4. PAS in detected loci (HeLa + HeLa-Ars)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: PAS IN DETECTED LOCI (HeLa/HeLa-Ars)")
print("=" * 70)

# Load L1 summary for HeLa/Ars
all_reads = []
for cl, grps in CELL_LINES.items():
    for grp in grps:
        f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['group'] = grp
            df['cell_line'] = cl
            df['locus_id'] = df['transcript_id']
            df['subfamily'] = df['gene_id']
            df['is_young'] = df['subfamily'].isin(YOUNG)
            df['age'] = np.where(df['is_young'], 'young', 'ancient')
            df = df.rename(columns={'polya_length': 'polya'})
            all_reads.append(df)

reads_df = pd.concat(all_reads, ignore_index=True)
reads_df = reads_df[reads_df['polya'].notna()].copy()

# Merge with PAS info
reads_with_pas = reads_df.merge(
    pas_df[['locus_id', 'best_pas', 'has_canonical', 'has_any_pas']],
    on='locus_id', how='left'
)

# Categorize loci
hela_loci = set(reads_df[reads_df['cell_line'] == 'HeLa']['locus_id'])
ars_loci = set(reads_df[reads_df['cell_line'] == 'HeLa-Ars']['locus_id'])
shared = hela_loci & ars_loci
hela_only = hela_loci - ars_loci
ars_only = ars_loci - hela_loci

reads_with_pas['locus_cat'] = reads_with_pas['locus_id'].map(
    lambda x: 'shared' if x in shared else ('hela_only' if x in hela_only else 'ars_only')
)

# PAS by locus category
print("\n--- PAS by Locus Category ---")
for cat in ['hela_only', 'shared', 'ars_only']:
    loci = reads_with_pas[reads_with_pas['locus_cat'] == cat]['locus_id'].unique()
    loci_pas = pas_df[pas_df['locus_id'].isin(loci)]
    n = len(loci_pas)
    if n == 0:
        continue
    canon = loci_pas['has_canonical'].sum()
    any_p = loci_pas['has_any_pas'].sum()
    print(f"\n  {cat} ({n} loci):")
    print(f"    Canonical: {canon} ({canon/n*100:.1f}%)")
    print(f"    Any PAS:   {any_p} ({any_p/n*100:.1f}%)")
    print(f"    No PAS:    {n-any_p} ({(n-any_p)/n*100:.1f}%)")

# =============================================================================
# 5. PAS vs poly(A) shortening
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: PAS vs POLY(A) SHORTENING")
print("=" * 70)

# Per-read poly(A) by PAS status
hela_reads = reads_with_pas[reads_with_pas['cell_line'] == 'HeLa']
ars_reads = reads_with_pas[reads_with_pas['cell_line'] == 'HeLa-Ars']

print("\n--- HeLa poly(A) by PAS ---")
for pas_status, label in [(True, 'Canonical AATAAA'), (False, 'No canonical')]:
    h = hela_reads[hela_reads['has_canonical'] == pas_status]['polya']
    a = ars_reads[ars_reads['has_canonical'] == pas_status]['polya']
    if len(h) > 5 and len(a) > 5:
        _, p = stats.mannwhitneyu(h, a, alternative='two-sided')
        print(f"  {label:20s}: HeLa={h.median():.1f}(n={len(h)}), "
              f"Ars={a.median():.1f}(n={len(a)}), Δ={a.median()-h.median():+.1f}, p={p:.2e}")

print("\n--- HeLa poly(A) by any PAS ---")
for pas_status, label in [(True, 'Has PAS variant'), (False, 'No PAS')]:
    h = hela_reads[hela_reads['has_any_pas'] == pas_status]['polya']
    a = ars_reads[ars_reads['has_any_pas'] == pas_status]['polya']
    if len(h) > 5 and len(a) > 5:
        _, p = stats.mannwhitneyu(h, a, alternative='two-sided')
        print(f"  {label:20s}: HeLa={h.median():.1f}(n={len(h)}), "
              f"Ars={a.median():.1f}(n={len(a)}), Δ={a.median()-h.median():+.1f}, p={p:.2e}")

# Ancient only
print("\n--- Ancient L1 only: PAS vs shortening ---")
hela_anc = hela_reads[hela_reads['age'] == 'ancient']
ars_anc = ars_reads[ars_reads['age'] == 'ancient']

for pas_status, label in [(True, 'Canonical AATAAA'), (False, 'No canonical')]:
    h = hela_anc[hela_anc['has_canonical'] == pas_status]['polya']
    a = ars_anc[ars_anc['has_canonical'] == pas_status]['polya']
    if len(h) > 5 and len(a) > 5:
        _, p = stats.mannwhitneyu(h, a, alternative='two-sided')
        print(f"  {label:20s}: HeLa={h.median():.1f}(n={len(h)}), "
              f"Ars={a.median():.1f}(n={len(a)}), Δ={a.median()-h.median():+.1f}, p={p:.2e}")

# =============================================================================
# 6. PAS in Cat B loci
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: PAS IN CAT B LOCI")
print("=" * 70)

catB_meta = []
for cl_name, grps in {**CELL_LINES, 'A549': ['A549_4','A549_5','A549_6'],
                        'MCF7': ['MCF7_2','MCF7_3','MCF7_4'],
                        'K562': ['K562_4','K562_5','K562_6']}.items():
    for grp in grps:
        f = TOPIC_07 / f'catB_reads_{grp}.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['cell_line'] = cl_name
            catB_meta.append(df)

if catB_meta:
    catB_df = pd.concat(catB_meta, ignore_index=True)
    catB_loci = catB_df['locus_id'].unique()
    catB_pas = pas_df[pas_df['locus_id'].isin(catB_loci)]

    n = len(catB_pas)
    if n > 0:
        canon = catB_pas['has_canonical'].sum()
        any_p = catB_pas['has_any_pas'].sum()
        print(f"\n  Cat B ({n} unique loci):")
        print(f"    Canonical: {canon} ({canon/n*100:.1f}%)")
        print(f"    Any PAS:   {any_p} ({any_p/n*100:.1f}%)")
        print(f"    No PAS:    {n-any_p} ({(n-any_p)/n*100:.1f}%)")

# Comparison: PASS vs Cat B (detected loci only)
pass_loci = reads_df['locus_id'].unique()
pass_pas = pas_df[pas_df['locus_id'].isin(pass_loci)]
n_pass = len(pass_pas)
if n_pass > 0:
    canon_pass = pass_pas['has_canonical'].sum()
    any_pass = pass_pas['has_any_pas'].sum()
    print(f"\n  PASS ({n_pass} unique loci):")
    print(f"    Canonical: {canon_pass} ({canon_pass/n_pass*100:.1f}%)")
    print(f"    Any PAS:   {any_pass} ({any_pass/n_pass*100:.1f}%)")

# =============================================================================
# 7. Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary_rows = []
for label, loci_set in [('All L1 (genome)', pas_df['locus_id'].unique()),
                          ('Young (genome)', pas_df[pas_df['age']=='young']['locus_id'].unique()),
                          ('Ancient (genome)', pas_df[pas_df['age']=='ancient']['locus_id'].unique()),
                          ('PASS detected', pass_loci),
                          ('Cat B detected', catB_loci if catB_meta else []),
                          ('Ars-only loci', list(ars_only)),
                          ('Shared loci', list(shared)),
                          ('HeLa-only loci', list(hela_only))]:
    sub = pas_df[pas_df['locus_id'].isin(loci_set)]
    n = len(sub)
    if n == 0:
        continue
    summary_rows.append({
        'category': label, 'n_loci': n,
        'canonical_pct': f"{sub['has_canonical'].mean()*100:.1f}%",
        'any_pas_pct': f"{sub['has_any_pas'].mean()*100:.1f}%",
        'no_pas_pct': f"{(~sub['has_any_pas']).mean()*100:.1f}%",
    })

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
summary.to_csv(OUTDIR / 'pas_summary.tsv', sep='\t', index=False)

print(f"\nAll results saved to: {OUTDIR}")
print("Done!")
