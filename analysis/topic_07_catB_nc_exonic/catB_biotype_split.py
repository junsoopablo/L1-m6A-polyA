#!/usr/bin/env python3
"""Split Cat B reads into lncRNA vs pseudogene, then compare poly(A) & modification."""

import re
import subprocess
import tempfile
import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from scipy import stats
from collections import defaultdict

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
OUTDIR = TOPIC_07 / 'catB_vs_pass_analysis'
OUTDIR.mkdir(exist_ok=True)

GTF = PROJECT / 'reference/Human.gtf'
OVERLAP_MIN = 100
PROB_THRESHOLD = 128
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

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
CL_ORDER = ['A549','H9','Hct116','HeLa','HeLa-Ars','HepG2','HEYA8','K562','MCF7','MCF7-EV','SHSY5Y']

PSEUDO_TYPES = {
    'transcribed_unprocessed_pseudogene', 'processed_pseudogene',
    'unprocessed_pseudogene', 'transcribed_unitary_pseudogene',
    'transcribed_processed_pseudogene', 'unitary_pseudogene',
    'translated_processed_pseudogene', 'polymorphic_pseudogene',
}

# =============================================================================
# Step 1: Create separate lncRNA and pseudogene BED files from GTF
# =============================================================================
print("Step 1: Creating biotype-specific exon BEDs...")
lnc_bed_path = OUTDIR / 'lncRNA_exons.bed'
pseudo_bed_path = OUTDIR / 'pseudo_exons.bed'

if not lnc_bed_path.exists() or not pseudo_bed_path.exists():
    lnc_lines, pseudo_lines = [], []
    with open(GTF) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9 or fields[2] != 'exon':
                continue
            m = re.search(r'gene_type "([^"]*)"', fields[8])
            if not m:
                continue
            gene_type = m.group(1)
            chrom = fields[0]
            start = max(int(fields[3]) - 1, 0)
            end = int(fields[4])
            # Also extract gene_name for annotation
            gn = re.search(r'gene_name "([^"]*)"', fields[8])
            gene_name = gn.group(1) if gn else 'unknown'
            entry = f"{chrom}\t{start}\t{end}\t{gene_type}|{gene_name}\n"
            if gene_type == 'lncRNA':
                lnc_lines.append(entry)
            elif gene_type in PSEUDO_TYPES:
                pseudo_lines.append(entry)

    for bed_path, lines, label in [(lnc_bed_path, lnc_lines, 'lncRNA'),
                                    (pseudo_bed_path, pseudo_lines, 'pseudogene')]:
        tmp = str(bed_path) + '.unsorted'
        with open(tmp, 'w') as f:
            f.writelines(lines)
        subprocess.run(f"sort -k1,1 -k2,2n {tmp} > {bed_path} && rm {tmp}",
                       shell=True, check=True)
        print(f"  {label}: {len(lines)} exon entries")
else:
    print("  BED files already exist, skipping.")

# =============================================================================
# Step 2: Load Cat B reads and intersect with biotype BEDs
# =============================================================================
print("\nStep 2: Loading Cat B metadata and annotating biotype...")

catB_all = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = TOPIC_07 / f'catB_reads_{grp}.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['cell_line'] = group
            catB_all.append(df)

catB = pd.concat(catB_all, ignore_index=True)
catB['is_young'] = catB['subfamily'].isin(YOUNG)
print(f"  Total Cat B reads: {len(catB)}")

# Write Cat B reads as BED for intersection
tmp_catb = tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False, prefix='catB_')
for _, row in catB.iterrows():
    tmp_catb.write(f"{row['chr']}\t{row['start']}\t{row['end']}\t{row['read_id']}\n")
tmp_catb.close()

# Sort
tmp_sorted = tmp_catb.name + '.sorted'
subprocess.run(f"sort -k1,1 -k2,2n {tmp_catb.name} > {tmp_sorted}", shell=True, check=True)

# Intersect with lncRNA
lnc_overlap = defaultdict(lambda: {'max_ov': 0, 'gene_name': ''})
result = subprocess.run(
    f"bedtools intersect -a {tmp_sorted} -b {lnc_bed_path} -wo",
    shell=True, capture_output=True, text=True)
for line in result.stdout.strip().split('\n'):
    if not line:
        continue
    fields = line.split('\t')
    rid = fields[3]
    info = fields[7]  # gene_type|gene_name
    ov = int(fields[-1])
    if ov > lnc_overlap[rid]['max_ov']:
        lnc_overlap[rid]['max_ov'] = ov
        lnc_overlap[rid]['gene_name'] = info.split('|')[1] if '|' in info else ''

# Intersect with pseudogene
pseudo_overlap = defaultdict(lambda: {'max_ov': 0, 'gene_name': ''})
result = subprocess.run(
    f"bedtools intersect -a {tmp_sorted} -b {pseudo_bed_path} -wo",
    shell=True, capture_output=True, text=True)
for line in result.stdout.strip().split('\n'):
    if not line:
        continue
    fields = line.split('\t')
    rid = fields[3]
    info = fields[7]
    ov = int(fields[-1])
    if ov > pseudo_overlap[rid]['max_ov']:
        pseudo_overlap[rid]['max_ov'] = ov
        pseudo_overlap[rid]['gene_name'] = info.split('|')[1] if '|' in info else ''

import os
os.unlink(tmp_catb.name)
os.unlink(tmp_sorted)

# Assign biotype
def assign_biotype(rid):
    lnc_ov = lnc_overlap[rid]['max_ov'] if rid in lnc_overlap else 0
    psd_ov = pseudo_overlap[rid]['max_ov'] if rid in pseudo_overlap else 0
    if lnc_ov >= OVERLAP_MIN and psd_ov >= OVERLAP_MIN:
        return 'both' if lnc_ov >= psd_ov else 'both'
    elif lnc_ov >= OVERLAP_MIN:
        return 'lncRNA'
    elif psd_ov >= OVERLAP_MIN:
        return 'pseudogene'
    else:
        return 'unknown'

def get_host_gene(rid):
    lnc_ov = lnc_overlap[rid]['max_ov'] if rid in lnc_overlap else 0
    psd_ov = pseudo_overlap[rid]['max_ov'] if rid in pseudo_overlap else 0
    if lnc_ov >= psd_ov and lnc_ov >= OVERLAP_MIN:
        return lnc_overlap[rid]['gene_name']
    elif psd_ov >= OVERLAP_MIN:
        return pseudo_overlap[rid]['gene_name']
    return ''

catB['biotype'] = catB['read_id'].map(assign_biotype)
catB['host_gene'] = catB['read_id'].map(get_host_gene)

print(f"\nBiotype distribution:")
for bt, cnt in catB['biotype'].value_counts().items():
    print(f"  {bt:15s}: {cnt:6d} ({cnt/len(catB)*100:.1f}%)")

catB.to_csv(OUTDIR / 'catB_with_biotype.tsv', sep='\t', index=False)

# =============================================================================
# Step 3: Load poly(A) and merge biotype
# =============================================================================
print("\nStep 3: Loading poly(A)...")
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
polya = polya.merge(
    catB[['read_id', 'group', 'biotype', 'host_gene', 'subfamily', 'is_young', 'age', 'locus_id']].drop_duplicates(),
    on=['read_id', 'group'], how='inner'
)
print(f"  Poly(A) with biotype: {len(polya)}")

# =============================================================================
# Step 4: Load MAFIA modification and merge biotype
# =============================================================================
print("\nStep 4: Loading MAFIA modification...")

def parse_mm_ml_tags(mm_tag, ml_tag):
    result = {'m6A': [], 'psi': []}
    if mm_tag is None or ml_tag is None:
        return result
    mod_blocks = mm_tag.rstrip(';').split(';')
    ml_idx = 0
    for block in mod_blocks:
        if not block:
            continue
        parts = block.split(',')
        mod_type = parts[0]
        if '17802' in mod_type or 'U+p' in mod_type or 'T+p' in mod_type:
            mod_key = 'psi'
        elif '21891' in mod_type or 'A+a' in mod_type or 'A+m' in mod_type:
            mod_key = 'm6A'
        else:
            ml_idx += len(parts) - 1
            continue
        for pos_str in parts[1:]:
            if pos_str:
                if ml_idx < len(ml_tag):
                    result[mod_key].append(ml_tag[ml_idx])
                ml_idx += 1
    return result

mod_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        bam = RESULTS / grp / 'j_catB' / 'mafia' / f'{grp}.catB.mAFiA.reads.bam'
        if bam.exists():
            with pysam.AlignmentFile(str(bam), "rb") as bf:
                for read in bf:
                    mm = read.get_tag("MM") if read.has_tag("MM") else None
                    ml = read.get_tag("ML") if read.has_tag("ML") else None
                    mods = parse_mm_ml_tags(mm, ml)
                    rl = read.query_length or read.infer_query_length() or 0
                    if rl < 50:
                        continue
                    mod_list.append({
                        'read_id': read.query_name,
                        'group': grp,
                        'cell_line': group,
                        'read_length': rl,
                        'm6a_sites': sum(1 for p in mods['m6A'] if p >= PROB_THRESHOLD),
                        'psi_sites': sum(1 for p in mods['psi'] if p >= PROB_THRESHOLD),
                    })

mod_df = pd.DataFrame(mod_list)
mod_df['m6a_per_kb'] = mod_df['m6a_sites'] / (mod_df['read_length'] / 1000)
mod_df['psi_per_kb'] = mod_df['psi_sites'] / (mod_df['read_length'] / 1000)
mod_df = mod_df.merge(
    catB[['read_id', 'group', 'biotype', 'host_gene', 'subfamily', 'is_young', 'age']].drop_duplicates(),
    on=['read_id', 'group'], how='inner'
)
print(f"  Mod with biotype: {len(mod_df)}")

# =============================================================================
# ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: BIOTYPE BREAKDOWN")
print("=" * 70)

for bt in ['lncRNA', 'pseudogene', 'both', 'unknown']:
    sub = catB[catB['biotype'] == bt]
    if len(sub) == 0:
        continue
    print(f"\n  {bt}: {len(sub)} reads")
    print(f"    Young: {sub['is_young'].sum()} ({sub['is_young'].mean()*100:.1f}%)")
    print(f"    Unique loci: {sub['locus_id'].nunique()}")
    top_sub = sub['subfamily'].value_counts().head(5)
    print(f"    Top subfamilies: {dict(top_sub)}")

# Top host genes
print("\n--- Top Host Genes ---")
for bt in ['lncRNA', 'pseudogene']:
    sub = catB[catB['biotype'] == bt]
    if len(sub) == 0:
        continue
    top_genes = sub['host_gene'].value_counts().head(10)
    print(f"\n  {bt}:")
    for gene, cnt in top_genes.items():
        print(f"    {gene:25s}: {cnt:5d} reads")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: POLY(A) BY BIOTYPE")
print("=" * 70)

print("\n--- Overall ---")
for bt in ['lncRNA', 'pseudogene', 'both', 'unknown']:
    sub = polya[polya['biotype'] == bt]
    if len(sub) < 5:
        continue
    print(f"  {bt:15s}: median={sub['polya'].median():.1f} nt, n={len(sub)}")

# Per cell line
print("\n--- Per Cell Line ---")
rows = []
for cl in CL_ORDER:
    for bt in ['lncRNA', 'pseudogene']:
        sub = polya[(polya['cell_line'] == cl) & (polya['biotype'] == bt)]
        if len(sub) >= 5:
            rows.append({'cell_line': cl, 'biotype': bt, 'n': len(sub),
                         'median_polya': sub['polya'].median()})
bt_cl = pd.DataFrame(rows)
if len(bt_cl) > 0:
    pivot = bt_cl.pivot(index='cell_line', columns='biotype', values='median_polya')
    print(pivot.to_string())

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MODIFICATION BY BIOTYPE")
print("=" * 70)

print("\n--- Overall ---")
for bt in ['lncRNA', 'pseudogene', 'both', 'unknown']:
    sub = mod_df[mod_df['biotype'] == bt]
    if len(sub) < 5:
        continue
    print(f"  {bt:15s}: m6A/kb={sub['m6a_per_kb'].median():.2f}, "
          f"psi/kb={sub['psi_per_kb'].median():.2f}, n={len(sub)}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: ARSENITE BY BIOTYPE (HeLa vs HeLa-Ars)")
print("=" * 70)

for bt in ['lncRNA', 'pseudogene']:
    print(f"\n--- {bt} ---")
    hela = polya[(polya['cell_line'] == 'HeLa') & (polya['biotype'] == bt)]['polya']
    ars = polya[(polya['cell_line'] == 'HeLa-Ars') & (polya['biotype'] == bt)]['polya']
    if len(hela) >= 5 and len(ars) >= 5:
        _, p = stats.mannwhitneyu(hela.dropna(), ars.dropna(), alternative='two-sided')
        delta = ars.median() - hela.median()
        print(f"  All: HeLa={hela.median():.1f}(n={len(hela)}), "
              f"Ars={ars.median():.1f}(n={len(ars)}), Δ={delta:+.1f}, p={p:.2e}")
    else:
        print(f"  All: too few (HeLa={len(hela)}, Ars={len(ars)})")

    # By age
    for age_val in ['ancient', 'young']:
        h = polya[(polya['cell_line'] == 'HeLa') & (polya['biotype'] == bt) & (polya['age'] == age_val)]['polya']
        a = polya[(polya['cell_line'] == 'HeLa-Ars') & (polya['biotype'] == bt) & (polya['age'] == age_val)]['polya']
        if len(h) >= 3 and len(a) >= 3:
            _, p = stats.mannwhitneyu(h.dropna(), a.dropna(), alternative='two-sided')
            delta = a.median() - h.median()
            print(f"  {age_val:8s}: HeLa={h.median():.1f}(n={len(h)}), "
                  f"Ars={a.median():.1f}(n={len(a)}), Δ={delta:+.1f}, p={p:.2e}")
        else:
            print(f"  {age_val:8s}: too few (HeLa={len(h)}, Ars={len(a)})")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: SUMMARY TABLE")
print("=" * 70)

rows = []
for bt in ['lncRNA', 'pseudogene']:
    sub_meta = catB[catB['biotype'] == bt]
    sub_polya = polya[polya['biotype'] == bt]
    sub_mod = mod_df[mod_df['biotype'] == bt]

    hela_p = sub_polya[sub_polya['cell_line'] == 'HeLa']['polya']
    ars_p = sub_polya[sub_polya['cell_line'] == 'HeLa-Ars']['polya']
    ars_delta = ars_p.median() - hela_p.median() if len(hela_p) > 0 and len(ars_p) > 0 else np.nan

    rows.append({
        'biotype': bt,
        'n_reads': len(sub_meta),
        'young_pct': f"{sub_meta['is_young'].mean()*100:.1f}%",
        'polya_median': f"{sub_polya['polya'].median():.1f}" if len(sub_polya) > 0 else '-',
        'm6a_per_kb': f"{sub_mod['m6a_per_kb'].median():.2f}" if len(sub_mod) > 0 else '-',
        'psi_per_kb': f"{sub_mod['psi_per_kb'].median():.2f}" if len(sub_mod) > 0 else '-',
        'ars_delta': f"{ars_delta:+.1f}" if not np.isnan(ars_delta) else '-',
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(OUTDIR / 'catB_biotype_summary.tsv', sep='\t', index=False)

print(f"\nAll results saved to: {OUTDIR}")
print("Done!")
