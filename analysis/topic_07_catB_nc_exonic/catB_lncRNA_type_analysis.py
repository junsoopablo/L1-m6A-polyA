#!/usr/bin/env python3
"""
Investigate whether Cat B lncRNA-overlapping L1 reads are:
  Type 1: L1 embedded within a multi-exon lncRNA (host read-through)
  Type 2: L1 element that GENCODE happened to annotate as a single-exon lncRNA
          (biologically same as PASS intergenic L1)

If Type 2 reads behave like PASS L1 (arsenite-sensitive), then the PASS vs Cat B
distinction is partly an annotation artifact for those reads.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
OUTDIR = TOPIC_07 / 'catB_lncRNA_type_analysis'
OUTDIR.mkdir(exist_ok=True)

GTF = PROJECT / 'reference/Human.gtf'
TE_GTF = PROJECT / 'reference/hg38_rmsk_TE.gtf'

CELL_LINES = {
    'A549': ['A549_4','A549_5','A549_6'],
    'H9': ['H9_2','H9_3','H9_4'],
    'Hct116': ['Hct116_3','Hct116_4'],
    'HeLa': ['HeLa_1','HeLa_2','HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1','HeLa-Ars_2','HeLa-Ars_3'],
    'HepG2': ['HepG2_5','HepG2_6'],
    'HEYA8': ['HEYA8_1','HEYA8_2','HEYA8_3'],
    'K562': ['K562_4','K562_5','K562_6'],
    'MCF7': ['MCF7_2','MCF7_3','MCF7_4'],
    'MCF7-EV': ['MCF7-EV_1'],
    'SHSY5Y': ['SHSY5Y_1','SHSY5Y_2','SHSY5Y_3'],
}

# =========================================================================
# Step 1: Parse GTF → per-gene: n_exons, gene_body_span, total_exon_bp
# =========================================================================
print("Step 1: Parsing GENCODE GTF for lncRNA gene structure...")

gene_info = {}  # gene_name → {chr, start, end, n_exons, exon_bp, strand}
exon_per_gene = defaultdict(list)  # gene_name → [(start, end), ...]

with open(GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.rstrip('\n').split('\t')
        if len(fields) < 9:
            continue

        m_type = re.search(r'gene_type "([^"]*)"', fields[8])
        if not m_type or m_type.group(1) != 'lncRNA':
            continue

        m_name = re.search(r'gene_name "([^"]*)"', fields[8])
        if not m_name:
            continue
        gene_name = m_name.group(1)
        chrom = fields[0]
        start = int(fields[3]) - 1  # 0-based
        end = int(fields[4])
        strand = fields[6]

        if fields[2] == 'gene':
            gene_info[gene_name] = {
                'chr': chrom, 'start': start, 'end': end,
                'strand': strand, 'gene_body_span': end - start
            }
        elif fields[2] == 'exon':
            exon_per_gene[gene_name].append((start, end))

# Compute per-gene exon stats
for gene_name in gene_info:
    exons = exon_per_gene.get(gene_name, [])
    # Merge overlapping exons for accurate count
    if exons:
        exons_sorted = sorted(set(exons))
        merged = [exons_sorted[0]]
        for s, e in exons_sorted[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        gene_info[gene_name]['n_unique_exons'] = len(merged)
        gene_info[gene_name]['total_exon_bp'] = sum(e - s for s, e in merged)
        # N transcripts = number of raw exons / merged exons gives rough idea
        gene_info[gene_name]['n_raw_exon_entries'] = len(exons)
    else:
        gene_info[gene_name]['n_unique_exons'] = 0
        gene_info[gene_name]['total_exon_bp'] = 0
        gene_info[gene_name]['n_raw_exon_entries'] = 0

gene_df = pd.DataFrame.from_dict(gene_info, orient='index')
gene_df.index.name = 'gene_name'
gene_df = gene_df.reset_index()

print(f"  Total lncRNA genes: {len(gene_df)}")
print(f"  Single-exon (1 unique exon): {(gene_df['n_unique_exons'] == 1).sum()}")
print(f"  Multi-exon (≥2 unique exons): {(gene_df['n_unique_exons'] >= 2).sum()}")
print(f"  Gene body span: median={gene_df['gene_body_span'].median():.0f}, "
      f"mean={gene_df['gene_body_span'].mean():.0f}")

# =========================================================================
# Step 2: Load Cat B reads with biotype
# =========================================================================
print("\nStep 2: Loading Cat B reads...")
catB = pd.read_csv(TOPIC_07 / 'catB_vs_pass_analysis/catB_with_biotype.tsv', sep='\t')
catB_lnc = catB[catB['biotype'].isin(['lncRNA', 'both'])].copy()
print(f"  Cat B lncRNA reads: {len(catB_lnc)}")

# Merge with gene structure info
catB_lnc = catB_lnc.merge(
    gene_df[['gene_name', 'n_unique_exons', 'gene_body_span', 'total_exon_bp']],
    left_on='host_gene', right_on='gene_name', how='left'
)

matched = catB_lnc['n_unique_exons'].notna().sum()
print(f"  Matched to gene structure: {matched}/{len(catB_lnc)} ({matched/len(catB_lnc)*100:.1f}%)")

# =========================================================================
# Step 3: Classify Type 1 vs Type 2
# =========================================================================
print("\nStep 3: Classifying Type 1 vs Type 2...")

# Type 2 criteria: single-exon lncRNA where gene body is similar size to read/L1
# Type 1: multi-exon lncRNA (L1 is part of a larger transcript)

catB_lnc['l1_type'] = 'unknown'
mask_single = catB_lnc['n_unique_exons'] == 1
mask_multi = catB_lnc['n_unique_exons'] >= 2

# For single-exon: check if gene body ≈ L1 read span
# If gene_body < 2x read_span, likely the lncRNA IS the L1
catB_lnc.loc[mask_single, 'l1_type'] = 'Type2_single_exon'
catB_lnc.loc[mask_multi, 'l1_type'] = 'Type1_multi_exon'

# Further split Type 2 by size ratio
catB_lnc['size_ratio'] = catB_lnc['gene_body_span'] / catB_lnc['read_span']

type2 = catB_lnc[catB_lnc['l1_type'] == 'Type2_single_exon']
type1 = catB_lnc[catB_lnc['l1_type'] == 'Type1_multi_exon']
unknown = catB_lnc[catB_lnc['l1_type'] == 'unknown']

print(f"\n  Type 1 (multi-exon lncRNA host): {len(type1)} reads ({len(type1)/len(catB_lnc)*100:.1f}%)")
print(f"    Gene body span: median={type1['gene_body_span'].median():.0f} bp")
print(f"    N exons: median={type1['n_unique_exons'].median():.0f}")
print(f"    Size ratio (gene/read): median={type1['size_ratio'].median():.1f}")

print(f"\n  Type 2 (single-exon lncRNA ≈ L1): {len(type2)} reads ({len(type2)/len(catB_lnc)*100:.1f}%)")
print(f"    Gene body span: median={type2['gene_body_span'].median():.0f} bp")
print(f"    Size ratio (gene/read): median={type2['size_ratio'].median():.1f}")

if len(unknown) > 0:
    print(f"\n  Unknown: {len(unknown)} reads")

# Size ratio distribution for Type 2
print(f"\n  Type 2 size ratio distribution:")
for cutoff in [1.5, 2, 3, 5, 10]:
    n_below = (type2['size_ratio'] < cutoff).sum()
    print(f"    gene_body < {cutoff}x read_span: {n_below} ({n_below/len(type2)*100:.1f}%)")

# =========================================================================
# Step 4: Top host genes by type
# =========================================================================
print("\n" + "=" * 70)
print("Top host genes by type:")
print("=" * 70)

for ltype, sub in [("Type1_multi_exon", type1), ("Type2_single_exon", type2)]:
    top = sub['host_gene'].value_counts().head(10)
    print(f"\n  {ltype}:")
    for gene, cnt in top.items():
        gi = gene_info.get(gene, {})
        n_ex = gi.get('n_unique_exons', '?')
        span = gi.get('gene_body_span', 0)
        print(f"    {gene:25s}: {cnt:5d} reads, {n_ex} exons, gene={span:>8,d} bp")

# =========================================================================
# Step 5: Load poly(A) for Cat B reads
# =========================================================================
print("\n" + "=" * 70)
print("Step 5: Loading poly(A) for Cat B reads...")
print("=" * 70)

polya_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = RESULTS / grp / 'j_catB' / f'{grp}.catB.nanopolish.polya.tsv.gz'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df = df[df['qc_tag'] == 'PASS'].copy()
            df['group'] = grp
            df['cell_line'] = group
            df = df.rename(columns={'readname': 'read_id', 'polya_length': 'polya'})
            polya_list.append(df[['read_id', 'group', 'cell_line', 'polya']])

polya = pd.concat(polya_list, ignore_index=True)

# Merge with type classification
catB_lnc_slim = catB_lnc[['read_id', 'group', 'l1_type', 'host_gene', 'n_unique_exons',
                            'gene_body_span', 'size_ratio', 'subfamily', 'age', 'is_young',
                            'locus_id']].drop_duplicates()

polya_typed = polya.merge(catB_lnc_slim, on=['read_id', 'group'], how='inner')
print(f"  Poly(A) with type info: {len(polya_typed)}")

# =========================================================================
# Step 6: ARSENITE RESPONSE by Type (KEY ANALYSIS)
# =========================================================================
print("\n" + "=" * 70)
print("Step 6: ARSENITE RESPONSE — Type 1 vs Type 2")
print("=" * 70)

# Also load PASS L1 poly(A) for comparison
pass_polya_list = []
for grp in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
    if f.exists():
        df = pd.read_csv(f, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        cl = 'HeLa-Ars' if 'Ars' in grp else 'HeLa'
        pass_polya_list.append(pd.DataFrame({
            'read_id': df['read_id'],
            'cell_line': cl,
            'polya': df['polya_length'],
            'age': df['class'].map(lambda x: 'young' if x in ['L1HS','L1PA1','L1PA2','L1PA3'] else 'ancient'),
            'l1_type': 'PASS_L1'
        }))

pass_polya = pd.concat(pass_polya_list, ignore_index=True)

# Compare
print(f"\n{'Category':30s} {'HeLa med':>10s} {'HeLa n':>8s} {'Ars med':>10s} {'Ars n':>8s} {'Δ':>8s} {'p':>12s}")
print("-" * 90)

for label, data in [
    ('PASS L1 (all)', pass_polya),
    ('PASS L1 (ancient only)', pass_polya[pass_polya['age'] == 'ancient']),
    ('Cat B Type1 multi-exon (all)', polya_typed[polya_typed['l1_type'] == 'Type1_multi_exon']),
    ('Cat B Type1 multi-exon (anc)', polya_typed[(polya_typed['l1_type'] == 'Type1_multi_exon') & (polya_typed['age'] == 'ancient')]),
    ('Cat B Type2 single-exon (all)', polya_typed[polya_typed['l1_type'] == 'Type2_single_exon']),
    ('Cat B Type2 single-exon (anc)', polya_typed[(polya_typed['l1_type'] == 'Type2_single_exon') & (polya_typed['age'] == 'ancient')]),
]:
    hela = data[data['cell_line'] == 'HeLa']['polya'].dropna()
    ars = data[data['cell_line'] == 'HeLa-Ars']['polya'].dropna()
    if len(hela) >= 3 and len(ars) >= 3:
        _, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
        delta = ars.median() - hela.median()
        print(f"{label:30s} {hela.median():10.1f} {len(hela):8d} {ars.median():10.1f} {len(ars):8d} {delta:+8.1f} {p:12.2e}")
    else:
        print(f"{label:30s}  (too few data)")

# =========================================================================
# Step 7: Type 2 further stratified by size ratio
# =========================================================================
print("\n" + "=" * 70)
print("Step 7: Type 2 single-exon by gene/read size ratio")
print("=" * 70)

# Split Type 2 into: gene_body < 2x read (very L1-like) vs gene_body ≥ 2x read
print(f"\n{'Subgroup':40s} {'HeLa med':>10s} {'HeLa n':>8s} {'Ars med':>10s} {'Ars n':>8s} {'Δ':>8s} {'p':>12s}")
print("-" * 100)

for ratio_label, ratio_mask in [
    ('Type2: gene < 2x read (≈ L1)', polya_typed['size_ratio'] < 2),
    ('Type2: gene 2-5x read', (polya_typed['size_ratio'] >= 2) & (polya_typed['size_ratio'] < 5)),
    ('Type2: gene ≥ 5x read', polya_typed['size_ratio'] >= 5),
]:
    sub = polya_typed[(polya_typed['l1_type'] == 'Type2_single_exon') & ratio_mask]
    hela = sub[sub['cell_line'] == 'HeLa']['polya'].dropna()
    ars = sub[sub['cell_line'] == 'HeLa-Ars']['polya'].dropna()
    if len(hela) >= 3 and len(ars) >= 3:
        _, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
        delta = ars.median() - hela.median()
        print(f"{ratio_label:40s} {hela.median():10.1f} {len(hela):8d} {ars.median():10.1f} {len(ars):8d} {delta:+8.1f} {p:12.2e}")
    else:
        print(f"{ratio_label:40s}  HeLa={len(hela)}, Ars={len(ars)} (too few)")

# =========================================================================
# Step 8: Compare with PASS intergenic L1
# =========================================================================
print("\n" + "=" * 70)
print("Step 8: PASS intergenic vs intronic L1 arsenite response")
print("=" * 70)

# Load overlapping_genes column from PASS L1 summaries
pass_context_list = []
for grp in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
    if f.exists():
        df = pd.read_csv(f, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        cl = 'HeLa-Ars' if 'Ars' in grp else 'HeLa'
        is_intergenic = df['overlapping_genes'].str.lower().str.contains('intergenic', na=False)
        pass_context_list.append(pd.DataFrame({
            'read_id': df['read_id'],
            'cell_line': cl,
            'polya': df['polya_length'],
            'context': np.where(is_intergenic, 'intergenic', 'intronic'),
            'age': df['class'].map(lambda x: 'young' if x in ['L1HS','L1PA1','L1PA2','L1PA3'] else 'ancient'),
        }))

pass_context = pd.concat(pass_context_list, ignore_index=True)

print(f"\n{'Subgroup':40s} {'HeLa med':>10s} {'HeLa n':>8s} {'Ars med':>10s} {'Ars n':>8s} {'Δ':>8s} {'p':>12s}")
print("-" * 100)

for label, mask in [
    ('PASS intergenic (ancient)', (pass_context['context'] == 'intergenic') & (pass_context['age'] == 'ancient')),
    ('PASS intronic (ancient)', (pass_context['context'] == 'intronic') & (pass_context['age'] == 'ancient')),
    ('PASS intergenic (all)', pass_context['context'] == 'intergenic'),
    ('PASS intronic (all)', pass_context['context'] == 'intronic'),
]:
    sub = pass_context[mask]
    hela = sub[sub['cell_line'] == 'HeLa']['polya'].dropna()
    ars = sub[sub['cell_line'] == 'HeLa-Ars']['polya'].dropna()
    if len(hela) >= 3 and len(ars) >= 3:
        _, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
        delta = ars.median() - hela.median()
        print(f"{label:40s} {hela.median():10.1f} {len(hela):8d} {ars.median():10.1f} {len(ars):8d} {delta:+8.1f} {p:12.2e}")
    else:
        print(f"{label:40s}  HeLa={len(hela)}, Ars={len(ars)} (too few)")

# =========================================================================
# Step 9: Baseline poly(A) comparison
# =========================================================================
print("\n" + "=" * 70)
print("Step 9: Baseline poly(A) comparison (non-Ars cell lines only)")
print("=" * 70)

non_ars = polya_typed[~polya_typed['cell_line'].isin(['HeLa-Ars', 'MCF7-EV'])]
pass_non_ars = pass_polya[~pass_polya['cell_line'].isin(['HeLa-Ars', 'MCF7-EV'])]

for label, data in [
    ('PASS L1', pass_non_ars),
    ('Cat B Type1 multi-exon', non_ars[non_ars['l1_type'] == 'Type1_multi_exon']),
    ('Cat B Type2 single-exon', non_ars[non_ars['l1_type'] == 'Type2_single_exon']),
]:
    vals = data['polya'].dropna()
    print(f"  {label:30s}: median={vals.median():.1f} nt, mean={vals.mean():.1f}, n={len(vals)}")

# Mann-Whitney: PASS vs Type2
pass_vals = pass_non_ars['polya'].dropna()
type2_vals = non_ars[non_ars['l1_type'] == 'Type2_single_exon']['polya'].dropna()
if len(pass_vals) >= 3 and len(type2_vals) >= 3:
    _, p = stats.mannwhitneyu(pass_vals, type2_vals, alternative='two-sided')
    print(f"\n  PASS vs Type2 baseline: Δ={type2_vals.median() - pass_vals.median():+.1f} nt, p={p:.2e}")

# =========================================================================
# Step 10: Save results
# =========================================================================
print("\n" + "=" * 70)
print("Saving results...")

catB_lnc[['read_id', 'sample', 'chr', 'start', 'end', 'read_span', 'subfamily',
           'locus_id', 'age', 'group', 'cell_line', 'is_young', 'host_gene',
           'l1_type', 'n_unique_exons', 'gene_body_span', 'size_ratio']].to_csv(
    OUTDIR / 'catB_lncRNA_typed.tsv', sep='\t', index=False
)

polya_typed.to_csv(OUTDIR / 'catB_lncRNA_polya_typed.tsv', sep='\t', index=False)

print(f"Output saved to: {OUTDIR}")
print("Done!")
