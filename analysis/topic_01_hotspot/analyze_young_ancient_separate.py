#!/usr/bin/env python3
"""
Separate analysis for Young vs Ancient L1

Young L1 (L1HS, L1PA1-2): Subfamily-level aggregation (multi-mapping OK)
Ancient L1 (L1M, etc.): Locus-level analysis (unique mapping preferred)
"""

import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from collections import defaultdict

def get_mapq_from_bam(bam_path):
    """Extract read_id and MAPQ from BAM"""
    result = subprocess.run(
        ['samtools', 'view', str(bam_path)],
        capture_output=True, text=True
    )
    mapq_dict = {}
    for line in result.stdout.strip().split('\n'):
        if line:
            fields = line.split('\t')
            read_id = fields[0]
            mapq = int(fields[4])
            mapq_dict[read_id] = mapq
    return mapq_dict

def categorize_l1(gene_id):
    """Categorize L1 as Young (active) or Ancient"""
    if gene_id.startswith('L1HS'):
        return 'young', 'L1HS'
    elif gene_id.startswith('L1PA'):
        rest = gene_id[4:].split('_')[0]
        if rest.isdigit():
            num = int(rest)
            if num <= 2:
                return 'young', 'L1PA1-2'
            elif num <= 4:
                return 'young', 'L1PA3-4'
            else:
                return 'ancient', f'L1PA{num}'
        return 'ancient', 'L1PA_other'
    elif gene_id.startswith('L1PB'):
        return 'ancient', 'L1PB'
    elif gene_id.startswith('L1M'):
        # Extract L1M subfamily (e.g., L1MC4, L1ME2, L1MA8)
        parts = gene_id.split('_')
        subfamily = parts[0] if parts else gene_id
        return 'ancient', subfamily
    elif gene_id.startswith('HAL1'):
        return 'ancient', 'HAL1'
    return 'ancient', 'Other'

def parse_cell_line(group):
    """Extract cell line from group name"""
    parts = group.replace("-", "_").split("_")
    cell_line = parts[0]
    if len(parts) > 1 and parts[1] in ["Ars", "EV", "Kasumi3", "HFF"]:
        cell_line = f"{parts[0]}-{parts[1]}"
    return cell_line

def main():
    base = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group")

    # Load all data
    all_data = []

    for summary_file in base.glob("*/g_summary/*_L1_summary.tsv"):
        group = summary_file.stem.replace("_L1_summary", "")
        if "THP1" in group:
            continue

        bam_path = base / group / "d_LINE_quantification" / f"{group}_L1_reads.bam"

        # Load summary
        df = pd.read_csv(summary_file, sep='\t')
        df['group'] = group
        df['cell_line'] = parse_cell_line(group)

        # Get MAPQ if BAM exists
        if bam_path.exists():
            mapq_dict = get_mapq_from_bam(bam_path)
            df['mapq'] = df['read_id'].map(mapq_dict)
        else:
            df['mapq'] = np.nan

        # Categorize L1
        df[['age_category', 'subfamily']] = df['gene_id'].apply(
            lambda x: pd.Series(categorize_l1(x))
        )

        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    print("=" * 70)
    print("YOUNG vs ANCIENT L1 SEPARATE ANALYSIS")
    print("=" * 70)

    # Split into young and ancient
    young_df = full_df[full_df['age_category'] == 'young'].copy()
    ancient_df = full_df[full_df['age_category'] == 'ancient'].copy()

    print(f"\nTotal reads: {len(full_df):,}")
    print(f"Young L1 (L1HS, L1PA1-4): {len(young_df):,} ({len(young_df)/len(full_df)*100:.1f}%)")
    print(f"Ancient L1: {len(ancient_df):,} ({len(ancient_df)/len(full_df)*100:.1f}%)")

    # =========================================================
    # YOUNG L1: Subfamily-level analysis (include all MAPQ)
    # =========================================================
    print("\n" + "=" * 70)
    print("YOUNG L1 ANALYSIS (Subfamily-level, all MAPQ included)")
    print("=" * 70)

    # By cell line and subfamily
    young_by_cellline = young_df.groupby(['cell_line', 'subfamily']).agg({
        'read_id': 'count',
        'polya_length': 'median',
        'mapq': 'mean'
    }).rename(columns={'read_id': 'read_count', 'polya_length': 'median_polya'})

    young_pivot = young_by_cellline['read_count'].unstack(fill_value=0)

    # Sort by total young L1
    young_pivot['Total'] = young_pivot.sum(axis=1)
    young_pivot = young_pivot.sort_values('Total', ascending=False)

    print("\nYoung L1 read counts by cell line and subfamily:")
    print(young_pivot.to_string())

    # Normalize by total L1 to get "young L1 fraction"
    total_by_cellline = full_df.groupby('cell_line')['read_id'].count()
    young_total = young_df.groupby('cell_line')['read_id'].count()
    young_fraction = (young_total / total_by_cellline * 100).sort_values(ascending=False)

    print("\n\nYoung L1 fraction by cell line:")
    print("-" * 40)
    for cl, frac in young_fraction.items():
        marker = " ← ESC" if cl == "H9" else (" ← Neuronal" if cl == "SHSY5Y" else "")
        print(f"  {cl:<15} {frac:>6.2f}%{marker}")

    # =========================================================
    # ANCIENT L1: Locus-level analysis (MAPQ >= 20)
    # =========================================================
    print("\n" + "=" * 70)
    print("ANCIENT L1 ANALYSIS (Locus-level, MAPQ >= 20)")
    print("=" * 70)

    # Filter for unique/high-confidence mapping
    ancient_unique = ancient_df[ancient_df['mapq'] >= 20].copy()

    print(f"\nAncient L1 with MAPQ >= 20: {len(ancient_unique):,} / {len(ancient_df):,} ({len(ancient_unique)/len(ancient_df)*100:.1f}%)")

    # Top hotspots (locus-level)
    ancient_hotspots = ancient_unique.groupby('transcript_id').agg({
        'read_id': 'count',
        'gene_id': 'first',
        'cell_line': lambda x: x.nunique(),
        'polya_length': 'median'
    }).rename(columns={'read_id': 'read_count', 'cell_line': 'n_celllines', 'polya_length': 'median_polya'})

    ancient_hotspots = ancient_hotspots.sort_values('read_count', ascending=False)

    print("\nTop 15 Ancient L1 hotspots (unique mapping):")
    print("-" * 70)
    print(f"{'Locus':<25} {'Reads':>8} {'Subfamily':<12} {'Cell Lines':>10}")
    print("-" * 70)
    for tid, row in ancient_hotspots.head(15).iterrows():
        print(f"{tid[:24]:<25} {row['read_count']:>8} {row['gene_id'][:11]:<12} {row['n_celllines']:>10}")

    # =========================================================
    # COMPARISON: H9 and SHSY5Y Young L1
    # =========================================================
    print("\n" + "=" * 70)
    print("H9 (ESC) and SHSY5Y (Neuronal) - Young L1 Focus")
    print("=" * 70)

    for cl in ['H9', 'SHSY5Y']:
        cl_young = young_df[young_df['cell_line'] == cl]
        cl_total = full_df[full_df['cell_line'] == cl]

        print(f"\n{cl}:")
        print(f"  Total L1 reads: {len(cl_total):,}")
        print(f"  Young L1 reads: {len(cl_young):,} ({len(cl_young)/len(cl_total)*100:.1f}%)")

        if len(cl_young) > 0:
            subfamily_dist = cl_young.groupby('subfamily')['read_id'].count()
            print(f"  Subfamily distribution:")
            for sf, count in subfamily_dist.sort_values(ascending=False).items():
                print(f"    {sf}: {count}")

    # =========================================================
    # Save results
    # =========================================================
    out_dir = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_01_hotspot")

    # Young L1 summary
    young_summary = young_df.groupby(['cell_line', 'subfamily']).agg({
        'read_id': 'count',
        'polya_length': ['median', 'mean', 'std'],
        'mapq': 'mean'
    })
    young_summary.columns = ['read_count', 'median_polya', 'mean_polya', 'std_polya', 'mean_mapq']
    young_summary.to_csv(out_dir / "young_l1_by_cellline.tsv", sep='\t')

    # Ancient hotspots
    ancient_hotspots.to_csv(out_dir / "ancient_l1_hotspots.tsv", sep='\t')

    print(f"\n\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
