#!/usr/bin/env python3
"""
Topic 1: L1 Hotspot Analysis
- Calculate concentration metrics (Gini, top N contribution)
- Identify hotspots per cell line
- Check for cell-type specific vs shared hotspots
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def gini(x):
    """Calculate Gini coefficient"""
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def parse_cell_line(group):
    """Extract cell line from group name"""
    parts = group.replace("-", "_").split("_")
    cell_line = parts[0]
    if len(parts) > 1 and parts[1] in ["Ars", "EV", "Kasumi3", "HFF"]:
        cell_line = f"{parts[0]}-{parts[1]}"
    return cell_line

def main():
    base = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group")

    # Load all locus summaries
    all_locus = []
    for f in base.glob("*/g_summary/*_L1_locus_summary.tsv"):
        group = f.stem.replace("_L1_locus_summary", "")
        if "THP1" in group:  # Skip THP1
            continue
        df = pd.read_csv(f, sep='\t')
        df['group'] = group
        df['cell_line'] = parse_cell_line(group)
        all_locus.append(df)

    locus_df = pd.concat(all_locus, ignore_index=True)

    print("=" * 60)
    print("L1 HOTSPOT ANALYSIS - Topic 1")
    print("=" * 60)
    print(f"\nTotal loci entries: {len(locus_df):,}")
    print(f"Unique loci (transcript_id): {locus_df['transcript_id'].nunique():,}")
    print(f"Groups: {locus_df['group'].nunique()}")
    print(f"Cell lines: {locus_df['cell_line'].nunique()}")

    # Calculate concentration metrics per group
    print("\n" + "=" * 60)
    print("CONCENTRATION METRICS BY GROUP")
    print("=" * 60)

    concentration = []
    for group, gdf in locus_df.groupby('group'):
        total_reads = gdf['read_count'].sum()
        n_loci = len(gdf)
        top1 = gdf.nlargest(1, 'read_count')['read_count'].sum() / total_reads * 100
        top5 = gdf.nlargest(5, 'read_count')['read_count'].sum() / total_reads * 100
        top10 = gdf.nlargest(10, 'read_count')['read_count'].sum() / total_reads * 100
        g = gini(gdf['read_count'].values)
        cell_line = gdf['cell_line'].iloc[0]
        concentration.append({
            'group': group, 'cell_line': cell_line, 'total_reads': total_reads,
            'n_loci': n_loci, 'top1_pct': top1, 'top5_pct': top5,
            'top10_pct': top10, 'gini': g
        })

    conc_df = pd.DataFrame(concentration).sort_values('total_reads', ascending=False)

    print("\n{:<20} {:>10} {:>8} {:>8} {:>8} {:>8} {:>6}".format(
        "Group", "Reads", "Loci", "Top1%", "Top5%", "Top10%", "Gini"))
    print("-" * 75)
    for _, row in conc_df.iterrows():
        print("{:<20} {:>10} {:>8} {:>7.1f}% {:>7.1f}% {:>7.1f}% {:>6.2f}".format(
            row['group'], row['total_reads'], row['n_loci'],
            row['top1_pct'], row['top5_pct'], row['top10_pct'], row['gini']))

    # Summary by cell line
    print("\n" + "=" * 60)
    print("SUMMARY BY CELL LINE")
    print("=" * 60)

    cell_summary = conc_df.groupby('cell_line').agg({
        'total_reads': 'sum',
        'top5_pct': 'mean',
        'top10_pct': 'mean',
        'gini': 'mean'
    }).sort_values('total_reads', ascending=False)

    print("\n{:<15} {:>12} {:>10} {:>10} {:>8}".format(
        "Cell Line", "Total Reads", "Avg Top5%", "Avg Top10%", "Avg Gini"))
    print("-" * 60)
    for cl, row in cell_summary.iterrows():
        print("{:<15} {:>12,} {:>9.1f}% {:>9.1f}% {:>8.2f}".format(
            cl, int(row['total_reads']), row['top5_pct'], row['top10_pct'], row['gini']))

    # Identify top hotspots across all samples
    print("\n" + "=" * 60)
    print("TOP 20 HOTSPOTS (by total read count across all samples)")
    print("=" * 60)

    hotspots = locus_df.groupby('transcript_id').agg({
        'read_count': 'sum',
        'gene_id': 'first',
        'te_strand': 'first',
        'TE_group': 'first',
        'group': 'nunique'
    }).rename(columns={'group': 'n_groups'})
    hotspots = hotspots.sort_values('read_count', ascending=False)

    print("\n{:<25} {:>10} {:>10} {:>8} {:>12} {:>10}".format(
        "Locus (transcript_id)", "Reads", "Subfamily", "Strand", "TE_group", "N_groups"))
    print("-" * 80)
    for tid, row in hotspots.head(20).iterrows():
        print("{:<25} {:>10} {:>10} {:>8} {:>12} {:>10}".format(
            tid[:25], row['read_count'], row['gene_id'][:10],
            row['te_strand'], row['TE_group'], row['n_groups']))

    # Check cell-type specificity of top hotspots
    print("\n" + "=" * 60)
    print("HOTSPOT CELL-TYPE SPECIFICITY")
    print("=" * 60)

    # For each hotspot, calculate what % of reads come from each cell line
    top_hotspots = hotspots.head(20).index.tolist()

    hotspot_celltype = locus_df[locus_df['transcript_id'].isin(top_hotspots)].groupby(
        ['transcript_id', 'cell_line'])['read_count'].sum().unstack(fill_value=0)

    # Normalize to percentage
    hotspot_pct = hotspot_celltype.div(hotspot_celltype.sum(axis=1), axis=0) * 100

    print("\nTop hotspots - % reads by cell line:")
    print(hotspot_pct.round(1).to_string())

    # Save results
    out_dir = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_01_hotspot")
    conc_df.to_csv(out_dir / "concentration_by_group.tsv", sep='\t', index=False)
    hotspots.to_csv(out_dir / "hotspots_ranked.tsv", sep='\t')
    hotspot_pct.to_csv(out_dir / "hotspot_celltype_pct.tsv", sep='\t')

    print(f"\n\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
