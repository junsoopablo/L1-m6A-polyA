#!/usr/bin/env python3
"""Classify filtered-out L1 reads by exon overlap type.

Categories:
  A: Overlap protein_coding exon >= 100bp (correctly filtered)
  B: Overlap lncRNA/pseudogene exon >= 100bp but NOT protein_coding (potentially recoverable)
  C: Filtered for other reasons (no TE match, strand mismatch)

Usage: conda run -n research python classify_filtered_l1.py
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──
BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = BASE / 'results'
GTF = BASE / 'reference/Human.gtf'
L1_TE_BED = BASE / 'reference/L1_TE_L1_family.bed'
OVERLAP_MIN = 100
OUT_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/exon_type_analysis'

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Non-coding gene types
NC_TYPES = {
    'lncRNA',
    'transcribed_unprocessed_pseudogene', 'processed_pseudogene',
    'unprocessed_pseudogene', 'transcribed_unitary_pseudogene',
    'transcribed_processed_pseudogene', 'unitary_pseudogene',
    'translated_processed_pseudogene', 'polymorphic_pseudogene',
}

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

GROUP_TO_CL = {}
for cl, groups in CELL_LINES.items():
    for g in groups:
        GROUP_TO_CL[g] = cl


def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def create_typed_exon_beds():
    """Create protein_coding and non-coding exon BED files (sorted & merged)."""
    pc_bed = OUT_DIR / 'pc_exons.bed'
    nc_bed = OUT_DIR / 'nc_exons.bed'

    if pc_bed.exists() and nc_bed.exists():
        print("Typed exon BEDs already exist, skipping creation.")
        return pc_bed, nc_bed

    pc_lines, nc_lines = [], []

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
            entry = f"{chrom}\t{start}\t{end}\n"
            if gene_type == 'protein_coding':
                pc_lines.append(entry)
            elif gene_type in NC_TYPES:
                nc_lines.append(entry)

    for bed_path, lines, label in [(pc_bed, pc_lines, 'protein_coding'),
                                    (nc_bed, nc_lines, 'non-coding')]:
        with open(bed_path, 'w') as f:
            f.writelines(lines)
        # Sort and merge
        sorted_path = str(bed_path) + '.tmp'
        run_cmd(f"sort -k1,1 -k2,2n {bed_path} > {sorted_path} && "
                f"bedtools merge -i {sorted_path} > {bed_path} && rm {sorted_path}")
        n_merged = sum(1 for _ in open(bed_path))
        print(f"  {label}: {len(lines)} exons → {n_merged} merged regions")

    return pc_bed, nc_bed


def classify_sample(sample, pc_bed, nc_bed):
    """Classify filtered-out L1 reads for one sample."""
    d = RESULTS / sample / 'd_LINE_quantification'
    l1_bed = d / 'L1_reads.bed'
    pass_ids_file = d / f'{sample}_L1_pass_readIDs.txt'

    if not l1_bed.exists() or not pass_ids_file.exists():
        return None

    # Load pass IDs
    with open(pass_ids_file) as f:
        pass_ids = {line.strip() for line in f if line.strip()}

    # Load all L1 reads from BED (read_id → list of blocks)
    all_reads = {}
    with open(l1_bed) as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            rid = fields[3]
            if rid not in all_reads:
                all_reads[rid] = []
            all_reads[rid].append((fields[0], int(fields[1]), int(fields[2])))

    filtered_ids = set(all_reads.keys()) - pass_ids
    if not filtered_ids:
        return {
            'sample': sample, 'n_total': len(all_reads), 'n_pass': len(pass_ids),
            'n_filtered': 0, 'n_catA': 0, 'n_catB': 0, 'n_catC': 0,
            'catB_reads': [],
        }

    # Write filtered-out reads BED
    tmp_bed = tempfile.NamedTemporaryFile(
        mode='w', suffix='.bed', delete=False, dir='/tmp', prefix=f'filt_{sample}_')
    for rid in filtered_ids:
        for chrom, start, end in all_reads[rid]:
            tmp_bed.write(f"{chrom}\t{start}\t{end}\t{rid}\n")
    tmp_bed.close()

    # Intersect with protein_coding exons → max overlap per read
    pc_overlap = defaultdict(int)
    result = run_cmd(f"bedtools intersect -a {tmp_bed.name} -b {pc_bed} -wo")
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        fields = line.split('\t')
        rid = fields[3]
        ov = int(fields[-1])
        pc_overlap[rid] = max(pc_overlap[rid], ov)

    cat_A = {rid for rid, ov in pc_overlap.items() if ov >= OVERLAP_MIN}

    # Intersect with non-coding exons → max overlap per read
    nc_overlap = defaultdict(int)
    result = run_cmd(f"bedtools intersect -a {tmp_bed.name} -b {nc_bed} -wo")
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        fields = line.split('\t')
        rid = fields[3]
        ov = int(fields[-1])
        nc_overlap[rid] = max(nc_overlap[rid], ov)

    cat_B_ids = {rid for rid, ov in nc_overlap.items()
                 if ov >= OVERLAP_MIN and rid not in cat_A}
    cat_C = filtered_ids - cat_A - cat_B_ids

    os.unlink(tmp_bed.name)

    # Collect Category B read info
    catB_reads = []
    for rid in cat_B_ids:
        blocks = all_reads[rid]
        chrom = blocks[0][0]
        start = min(b[1] for b in blocks)
        end = max(b[2] for b in blocks)
        catB_reads.append({
            'read_id': rid, 'sample': sample,
            'chr': chrom, 'start': start, 'end': end,
            'read_span': end - start,
        })

    return {
        'sample': sample,
        'n_total': len(all_reads),
        'n_pass': len(pass_ids),
        'n_filtered': len(filtered_ids),
        'n_catA': len(cat_A),
        'n_catB': len(cat_B_ids),
        'n_catC': len(cat_C),
        'catB_reads': catB_reads,
    }


def annotate_catB_with_te(catB_df):
    """Annotate Category B reads with L1 subfamily using bedtools intersect."""
    if catB_df.empty:
        return catB_df

    tmp_bed = '/tmp/catB_reads.bed'
    catB_df[['chr', 'start', 'end', 'read_id']].to_csv(
        tmp_bed, sep='\t', header=False, index=False)

    result = run_cmd(f"bedtools intersect -a {tmp_bed} -b {L1_TE_BED} -wo")

    te_anno = {}
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        fields = line.split('\t')
        rid = fields[3]
        subfamily = fields[7]
        locus_id = fields[8]
        overlap = int(fields[-1])
        if rid not in te_anno or overlap > te_anno[rid][2]:
            te_anno[rid] = (subfamily, locus_id, overlap)

    catB_df['subfamily'] = catB_df['read_id'].map(
        lambda x: te_anno.get(x, (None, None, 0))[0])
    catB_df['locus_id'] = catB_df['read_id'].map(
        lambda x: te_anno.get(x, (None, None, 0))[1])
    catB_df['age'] = catB_df['subfamily'].apply(
        lambda x: 'young' if x in YOUNG else ('ancient' if x else 'unknown'))

    os.unlink(tmp_bed)
    return catB_df


def find_group(sample_name):
    """Map sample name (e.g. HeLa_1_1) to group (e.g. HeLa_1)."""
    # Sample format: {group}_{run_number}
    # Group format varies: HeLa_1, HeLa-Ars_1, MCF7-EV_1, etc.
    for cl, groups in CELL_LINES.items():
        for g in groups:
            if sample_name.startswith(g + '_'):
                return g
    return None


def generate_figures(summary_df, catB_df, pass_df):
    """Generate comparison figures."""
    fig_dir = OUT_DIR
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Fig A: Classification breakdown (aggregated) ──
    ax = axes[0, 0]
    totals = summary_df[['n_pass', 'n_catA', 'n_catB', 'n_catC']].sum()
    labels = ['PASS\n(current)', 'Cat A\n(pc exon)', 'Cat B\n(ncRNA/pseudo)', 'Cat C\n(other)']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
    bars = ax.bar(labels, totals.values, color=colors)
    for bar, val in zip(bars, totals.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Read count')
    ax.set_title('A. L1 Read Classification')

    # ── Fig B: Cat B / PASS ratio per cell line ──
    ax = axes[0, 1]
    summary_df['group'] = summary_df['sample'].apply(find_group)
    summary_df['cell_line'] = summary_df['group'].map(GROUP_TO_CL)
    cl_agg = summary_df.dropna(subset=['cell_line']).groupby('cell_line').agg(
        {'n_pass': 'sum', 'n_catB': 'sum'}).reset_index()
    cl_agg['ratio_pct'] = cl_agg['n_catB'] / cl_agg['n_pass'] * 100
    cl_agg = cl_agg.sort_values('ratio_pct', ascending=False)
    bars = ax.bar(cl_agg['cell_line'], cl_agg['ratio_pct'], color='#3498db')
    ax.set_ylabel('Cat B / PASS (%)')
    ax.set_title('B. Additional Reads per Cell Line')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, cl_agg['ratio_pct']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # ── Fig C: Age distribution (PASS vs Cat B) ──
    ax = axes[1, 0]
    if not catB_df.empty and not pass_df.empty:
        pass_df['age'] = pass_df['gene_id'].apply(
            lambda x: 'young' if x in YOUNG else 'ancient')
        pass_age = pass_df['age'].value_counts(normalize=True) * 100
        catB_age = catB_df['age'].value_counts(normalize=True) * 100

        x = np.arange(2)
        w = 0.35
        pass_vals = [pass_age.get('young', 0), pass_age.get('ancient', 0)]
        catB_vals = [catB_age.get('young', 0), catB_age.get('ancient', 0)]
        ax.bar(x - w/2, pass_vals, w, label='PASS', color='#2ecc71')
        ax.bar(x + w/2, catB_vals, w, label='Cat B', color='#3498db')
        ax.set_xticks(x)
        ax.set_xticklabels(['Young L1', 'Ancient L1'])
        ax.set_ylabel('Proportion (%)')
        ax.set_title('C. Age Distribution')
        ax.legend()

    # ── Fig D: Read span distribution ──
    ax = axes[1, 1]
    if not catB_df.empty and not pass_df.empty:
        bins = np.arange(0, 5001, 200)
        ax.hist(pass_df['read_length'].clip(upper=5000), bins=bins,
                alpha=0.6, density=True, label='PASS', color='#2ecc71')
        ax.hist(catB_df['read_span'].clip(upper=5000), bins=bins,
                alpha=0.6, density=True, label='Cat B', color='#3498db')
        ax.set_xlabel('Read length / span (nt)')
        ax.set_ylabel('Density')
        ax.set_title('D. Read Length Distribution')
        ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / 'exon_type_classification.png', dpi=150)
    plt.close()
    print(f"Figure saved: {fig_dir / 'exon_type_classification.png'}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Create typed exon BEDs
    print("Step 1: Creating typed exon BEDs...")
    pc_bed, nc_bed = create_typed_exon_beds()

    # Step 2: Classify filtered reads per sample
    print("\nStep 2: Classifying filtered reads per sample...")
    all_results = []
    all_catB = []

    for sample_dir in sorted(RESULTS.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        l1_bed = sample_dir / 'd_LINE_quantification' / 'L1_reads.bed'
        if not l1_bed.exists():
            continue

        result = classify_sample(sample, pc_bed, nc_bed)
        if result is None:
            continue

        all_results.append(result)
        all_catB.extend(result['catB_reads'])
        pct_B = result['n_catB'] / result['n_total'] * 100 if result['n_total'] else 0
        print(f"  {sample}: total={result['n_total']}, pass={result['n_pass']}, "
              f"A={result['n_catA']}, B={result['n_catB']} ({pct_B:.1f}%), C={result['n_catC']}")

    # Save summary
    summary_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != 'catB_reads'} for r in all_results])
    summary_df.to_csv(OUT_DIR / 'filter_classification_summary.tsv', sep='\t', index=False)

    # Step 3: Annotate Category B reads
    catB_df = pd.DataFrame(all_catB)
    print(f"\nStep 3: Annotating {len(catB_df)} Category B reads with TE info...")
    if not catB_df.empty:
        catB_df = annotate_catB_with_te(catB_df)
        catB_df.to_csv(OUT_DIR / 'catB_reads_detail.tsv', sep='\t', index=False)

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)

    total_all = summary_df['n_total'].sum()
    total_pass = summary_df['n_pass'].sum()
    total_filtered = summary_df['n_filtered'].sum()
    total_A = summary_df['n_catA'].sum()
    total_B = summary_df['n_catB'].sum()
    total_C = summary_df['n_catC'].sum()

    print(f"\nTotal L1 reads (pre-filter): {total_all:,}")
    print(f"  PASS (current pipeline):   {total_pass:,} ({total_pass/total_all*100:.1f}%)")
    print(f"  Filtered out:              {total_filtered:,} ({total_filtered/total_all*100:.1f}%)")
    print(f"    Cat A (protein_coding):  {total_A:,} ({total_A/total_all*100:.1f}%)")
    print(f"    Cat B (ncRNA/pseudo):    {total_B:,} ({total_B/total_all*100:.1f}%)")
    print(f"    Cat C (other reasons):   {total_C:,} ({total_C/total_all*100:.1f}%)")
    print(f"\n  Cat B / current PASS:      {total_B/total_pass*100:.1f}% additional reads")

    if not catB_df.empty:
        print(f"\n--- Category B Characteristics ---")
        print(f"Total Cat B reads: {len(catB_df):,}")

        sf_counts = catB_df['subfamily'].value_counts().head(15)
        print(f"\nTop subfamilies:")
        for sf, cnt in sf_counts.items():
            age_tag = '(young)' if sf in YOUNG else ''
            print(f"  {sf} {age_tag}: {cnt} ({cnt/len(catB_df)*100:.1f}%)")

        young_n = (catB_df['age'] == 'young').sum()
        ancient_n = (catB_df['age'] == 'ancient').sum()
        unknown_n = (catB_df['age'] == 'unknown').sum()
        print(f"\nAge: young={young_n} ({young_n/len(catB_df)*100:.1f}%), "
              f"ancient={ancient_n} ({ancient_n/len(catB_df)*100:.1f}%), "
              f"unknown={unknown_n}")
        print(f"Read span: median={catB_df['read_span'].median():.0f}, "
              f"mean={catB_df['read_span'].mean():.0f}")

    # Step 4: Load PASS reads for comparison
    pass_data = []
    for cl, groups in CELL_LINES.items():
        for g in groups:
            summary_path = BASE / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
            if summary_path.exists():
                df = pd.read_csv(summary_path, sep='\t')
                df['cell_line'] = cl
                df['group'] = g
                pass_data.append(df)

    pass_df = pd.concat(pass_data, ignore_index=True) if pass_data else pd.DataFrame()

    if not pass_df.empty:
        pass_df_pass = pass_df[pass_df['qc_tag'] == 'PASS'].copy()
        pass_young = (pass_df_pass['gene_id'].isin(YOUNG)).sum()
        pass_total = len(pass_df_pass)

        print(f"\n--- PASS vs Category B Comparison ---")
        print(f"PASS: {pass_total:,} reads, young {pass_young/pass_total*100:.1f}%, "
              f"median read_length={pass_df_pass['read_length'].median():.0f}")
        if not catB_df.empty:
            print(f"Cat B: {len(catB_df):,} reads, young {young_n/len(catB_df)*100:.1f}%, "
                  f"median read_span={catB_df['read_span'].median():.0f}")

        # Generate figures
        print("\nStep 4: Generating figures...")
        generate_figures(summary_df, catB_df, pass_df_pass)

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
