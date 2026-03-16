#!/usr/bin/env python3
"""
Validation: Check replicate consistency for L1 hotspots
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

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

    print("=" * 60)
    print("REPLICATE CONSISTENCY CHECK")
    print("=" * 60)

    # For each cell line with multiple replicates
    cell_lines_multi = locus_df.groupby('cell_line')['group'].nunique()
    cell_lines_multi = cell_lines_multi[cell_lines_multi >= 2].index.tolist()

    print(f"\nCell lines with ≥2 replicates: {len(cell_lines_multi)}")

    results = []
    for cl in sorted(cell_lines_multi):
        groups = sorted(locus_df[locus_df['cell_line'] == cl]['group'].unique())

        # Create locus x group matrix
        cl_data = locus_df[locus_df['cell_line'] == cl].pivot_table(
            index='transcript_id', columns='group', values='read_count', fill_value=0
        )

        print(f"\n{cl} ({len(groups)} replicates):")

        corrs = []
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i >= j:
                    continue
                common = cl_data[(cl_data[g1] > 0) & (cl_data[g2] > 0)]
                if len(common) > 5:
                    rho, p = spearmanr(common[g1], common[g2])
                    corrs.append(rho)
                    print(f"  {g1} vs {g2}: rho = {rho:.3f} (n={len(common)} common loci)")
                    results.append({'cell_line': cl, 'g1': g1, 'g2': g2, 'rho': rho, 'n': len(common)})
                else:
                    print(f"  {g1} vs {g2}: Too few common loci (n={len(common)})")

        if corrs:
            print(f"  Mean rho: {np.mean(corrs):.3f}")

    # Check if top hotspots are consistent
    print("\n" + "=" * 60)
    print("TOP HOTSPOT CONSISTENCY ACROSS REPLICATES")
    print("=" * 60)

    for cl in ['MCF7', 'HeLa', 'HeLa-Ars', 'K562', 'HepG2', 'A549', 'H9']:
        if cl not in cell_lines_multi:
            continue

        cl_data = locus_df[locus_df['cell_line'] == cl]
        groups = sorted(cl_data['group'].unique())

        print(f"\n{cl}:")

        # Collect top 5 per replicate
        all_top5 = set()
        rep_top5 = {}
        for g in groups:
            top5 = cl_data[cl_data['group'] == g].nlargest(5, 'read_count')['transcript_id'].tolist()
            rep_top5[g] = set(top5)
            all_top5.update(top5)
            print(f"  {g}: {', '.join(top5)}")

        # Check overlap
        if len(groups) >= 2:
            common = set.intersection(*rep_top5.values())
            print(f"  → Common in all replicates: {len(common)}/5 ({', '.join(common) if common else 'none'})")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        out_path = Path(__file__).parent / "replicate_correlations.tsv"
        results_df.to_csv(out_path, sep='\t', index=False)
        print(f"\n\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
