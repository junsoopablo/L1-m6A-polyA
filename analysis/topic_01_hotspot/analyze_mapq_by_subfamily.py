#!/usr/bin/env python3
"""
Analyze mapping quality by L1 subfamily
Check if young L1 (L1HS, L1PA1-2) have lower MAPQ due to sequence similarity
"""

import pandas as pd
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

def categorize_subfamily(gene_id):
    """Categorize L1 by retrotransposition competence"""
    if gene_id.startswith('L1HS'):
        return 'L1HS (active)'
    elif gene_id.startswith('L1PA'):
        rest = gene_id[4:].split('_')[0]
        if rest.isdigit():
            num = int(rest)
            if num <= 2:
                return 'L1PA1-2 (active)'
            elif num <= 4:
                return 'L1PA3-4 (recent)'
            else:
                return 'L1PA5+ (ancient)'
    elif gene_id.startswith('L1PB'):
        return 'L1PB (primate)'
    elif gene_id.startswith('L1M'):
        return 'L1M (mammalian)'
    elif gene_id.startswith('HAL1'):
        return 'HAL1'
    return 'Other'

def main():
    base = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group")

    # Analyze multiple samples
    samples = ['HeLa_1', 'MCF7_2', 'H9_2', 'HepG2_5', 'K562_4']

    all_results = []

    for group in samples:
        bam_path = base / group / "d_LINE_quantification" / f"{group}_L1_reads.bam"
        summary_path = base / group / "g_summary" / f"{group}_L1_summary.tsv"

        if not bam_path.exists() or not summary_path.exists():
            print(f"Skipping {group}: files not found")
            continue

        print(f"Processing {group}...")

        # Get MAPQ from BAM
        mapq_dict = get_mapq_from_bam(bam_path)

        # Load L1 summary
        summary = pd.read_csv(summary_path, sep='\t')

        # Add MAPQ
        summary['mapq'] = summary['read_id'].map(mapq_dict)
        summary = summary.dropna(subset=['mapq'])

        # Categorize subfamily
        summary['subfamily_cat'] = summary['gene_id'].apply(categorize_subfamily)
        summary['group'] = group

        all_results.append(summary[['group', 'read_id', 'gene_id', 'subfamily_cat', 'mapq']])

    if not all_results:
        print("No data found")
        return

    df = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 70)
    print("MAPQ DISTRIBUTION BY SUBFAMILY CATEGORY")
    print("=" * 70)

    # Overall stats
    stats = df.groupby('subfamily_cat').agg({
        'mapq': ['count', 'mean', 'median', 'std'],
        'read_id': 'count'
    })
    stats.columns = ['count', 'mean_mapq', 'median_mapq', 'std_mapq', 'n_reads']
    stats = stats.drop('n_reads', axis=1)
    stats['pct_mapq60'] = df[df['mapq'] == 60].groupby('subfamily_cat').size() / stats['count'] * 100
    stats['pct_mapq0'] = df[df['mapq'] == 0].groupby('subfamily_cat').size() / stats['count'] * 100
    stats = stats.fillna(0)
    stats = stats.sort_values('count', ascending=False)

    print("\n{:<20} {:>8} {:>10} {:>10} {:>10} {:>10}".format(
        "Subfamily", "Count", "Mean MAPQ", "Med MAPQ", "%MAPQ=60", "%MAPQ=0"))
    print("-" * 75)
    for cat, row in stats.iterrows():
        print("{:<20} {:>8} {:>10.1f} {:>10.0f} {:>9.1f}% {:>9.1f}%".format(
            cat, int(row['count']), row['mean_mapq'], row['median_mapq'],
            row['pct_mapq60'], row['pct_mapq0']))

    # Focus on active vs inactive
    print("\n" + "=" * 70)
    print("ACTIVE (retrotransposition-competent) vs INACTIVE L1")
    print("=" * 70)

    active_cats = ['L1HS (active)', 'L1PA1-2 (active)']
    df['is_active'] = df['subfamily_cat'].isin(active_cats)

    active_stats = df.groupby('is_active').agg({
        'mapq': ['count', 'mean', 'median'],
        'read_id': 'count'
    })

    active_df = df[df['is_active']]
    inactive_df = df[~df['is_active']]

    print(f"\nActive L1 (L1HS + L1PA1-2):")
    print(f"  Count: {len(active_df)}")
    print(f"  Mean MAPQ: {active_df['mapq'].mean():.1f}")
    print(f"  % MAPQ=60 (unique): {(active_df['mapq'] == 60).sum() / len(active_df) * 100:.1f}%")
    print(f"  % MAPQ=0 (multi): {(active_df['mapq'] == 0).sum() / len(active_df) * 100:.1f}%")

    print(f"\nInactive L1 (ancient):")
    print(f"  Count: {len(inactive_df)}")
    print(f"  Mean MAPQ: {inactive_df['mapq'].mean():.1f}")
    print(f"  % MAPQ=60 (unique): {(inactive_df['mapq'] == 60).sum() / len(inactive_df) * 100:.1f}%")
    print(f"  % MAPQ=0 (multi): {(inactive_df['mapq'] == 0).sum() / len(inactive_df) * 100:.1f}%")

    # Save results
    out_dir = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_01_hotspot")
    stats.to_csv(out_dir / "mapq_by_subfamily.tsv", sep='\t')
    print(f"\n\nResults saved to {out_dir / 'mapq_by_subfamily.tsv'}")

if __name__ == "__main__":
    main()
