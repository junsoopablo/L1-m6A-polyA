#!/usr/bin/env python3
"""
Analyze subfamily distribution across cell lines
Focus on young vs ancient L1 elements
"""

import pandas as pd
import numpy as np
from pathlib import Path

def categorize_subfamily(gene_id):
    """Categorize L1 subfamily by evolutionary age"""
    if gene_id.startswith('L1HS'):
        return 'L1HS (youngest)'
    elif gene_id.startswith('L1PA') and len(gene_id) > 4:
        # Extract number after L1PA
        rest = gene_id[4:].split('_')[0]
        if rest.isdigit():
            num = int(rest)
            if num <= 4:
                return 'L1PA1-4 (young)'
            elif num <= 8:
                return 'L1PA5-8 (intermediate)'
            else:
                return 'L1PA9+ (older)'
        return 'L1PA (other)'
    elif gene_id.startswith('L1PB'):
        return 'L1PB (primate)'
    elif gene_id.startswith('L1M'):
        return 'L1M (ancient mammalian)'
    elif gene_id.startswith('HAL1'):
        return 'HAL1 (half-L1)'
    else:
        return 'Other'

def main():
    base = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group")

    # Load all locus summaries
    all_locus = []
    for f in base.glob("*/g_summary/*_L1_locus_summary.tsv"):
        group = f.stem.replace("_L1_locus_summary", "")
        if "THP1" in group:
            continue
        df = pd.read_csv(f, sep='\t')
        df['group'] = group
        parts = group.replace("-", "_").split("_")
        cell_line = parts[0]
        if len(parts) > 1 and parts[1] in ["Ars", "EV", "Kasumi3", "HFF"]:
            cell_line = f"{parts[0]}-{parts[1]}"
        df['cell_line'] = cell_line
        all_locus.append(df)

    locus_df = pd.concat(all_locus, ignore_index=True)
    locus_df['subfamily_cat'] = locus_df['gene_id'].apply(categorize_subfamily)

    print("=" * 60)
    print("SUBFAMILY DISTRIBUTION")
    print("=" * 60)

    # Overall distribution
    subfamily_totals = locus_df.groupby('subfamily_cat')['read_count'].sum().sort_values(ascending=False)
    print("\nOverall (all samples):")
    total = subfamily_totals.sum()
    for cat, count in subfamily_totals.items():
        print(f"  {cat:30} {count:>8,} ({count/total*100:5.1f}%)")

    # By cell line
    print("\n" + "=" * 60)
    print("SUBFAMILY DISTRIBUTION BY CELL LINE (% of reads)")
    print("=" * 60)

    cellline_subfamily = locus_df.groupby(['cell_line', 'subfamily_cat'])['read_count'].sum().unstack(fill_value=0)
    cellline_pct = cellline_subfamily.div(cellline_subfamily.sum(axis=1), axis=0) * 100

    # Sort by total reads
    cellline_totals = locus_df.groupby('cell_line')['read_count'].sum().sort_values(ascending=False)
    cellline_pct = cellline_pct.loc[cellline_totals.index]

    # Reorder columns by age
    col_order = ['L1HS (youngest)', 'L1PA1-4 (young)', 'L1PA5-8 (intermediate)',
                 'L1PA9+ (older)', 'L1PA (other)', 'L1PB (primate)',
                 'L1M (ancient mammalian)', 'HAL1 (half-L1)', 'Other']
    col_order = [c for c in col_order if c in cellline_pct.columns]
    cellline_pct = cellline_pct[col_order]

    print("\n" + cellline_pct.round(1).to_string())

    # Young L1 (L1HS + L1PA1-4) focus
    print("\n" + "=" * 60)
    print("YOUNG L1 (L1HS + L1PA1-4) PERCENTAGE BY CELL LINE")
    print("=" * 60)

    young_cols = ['L1HS (youngest)', 'L1PA1-4 (young)']
    young_cols = [c for c in young_cols if c in cellline_pct.columns]
    if young_cols:
        young_pct = cellline_pct[young_cols].sum(axis=1).sort_values(ascending=False)
        print("\n{:<15} {:>10}".format("Cell Line", "Young L1 %"))
        print("-" * 30)
        for cl, pct in young_pct.items():
            print(f"{cl:<15} {pct:>9.1f}%")

    # Check specific subfamilies
    print("\n" + "=" * 60)
    print("TOP 10 SUBFAMILIES (by total read count)")
    print("=" * 60)

    subfamily_totals = locus_df.groupby('gene_id')['read_count'].sum().sort_values(ascending=False)
    print("\n{:<15} {:>10}".format("Subfamily", "Reads"))
    print("-" * 30)
    for sf, count in subfamily_totals.head(10).items():
        print(f"{sf:<15} {count:>10,}")

    # Save
    out_dir = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_01_hotspot")
    cellline_pct.to_csv(out_dir / "subfamily_by_cellline.tsv", sep='\t')
    print(f"\n\nSaved to {out_dir / 'subfamily_by_cellline.tsv'}")

if __name__ == "__main__":
    main()
