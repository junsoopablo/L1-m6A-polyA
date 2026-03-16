#!/usr/bin/env python3
"""
Analyze H9 (ESC) and SHSY5Y (Neuronal) L1 expression patterns
Literature suggests high L1 in stem cells and neurons
"""

import pandas as pd
from pathlib import Path

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

    print("=" * 70)
    print("H9 (ESC) and SHSY5Y (Neuronal) Analysis")
    print("=" * 70)

    # 1. Read counts comparison
    print("\n1. Total L1 Read Counts by Cell Line")
    print("-" * 50)
    cell_totals = locus_df.groupby('cell_line')['read_count'].sum().sort_values(ascending=False)
    for cl, count in cell_totals.items():
        marker = " ← ESC" if cl == "H9" else (" ← Neuronal" if cl == "SHSY5Y" else "")
        print(f"  {cl:<15} {count:>8,}{marker}")

    # 2. H9-specific hotspots
    print("\n" + "=" * 70)
    print("2. H9-Specific Hotspots")
    print("-" * 70)

    h9_data = locus_df[locus_df['cell_line'] == 'H9'].groupby('transcript_id')['read_count'].sum()
    other_data = locus_df[locus_df['cell_line'] != 'H9'].groupby('transcript_id')['read_count'].sum()

    h9_specific = []
    for tid in h9_data.index:
        h9_count = h9_data[tid]
        other_count = other_data.get(tid, 0)
        total = h9_count + other_count
        if total >= 10 and h9_count / total >= 0.5:
            h9_specific.append({
                'transcript_id': tid,
                'h9_reads': h9_count,
                'other_reads': other_count,
                'h9_pct': h9_count / total * 100,
                'subfamily': locus_df[locus_df['transcript_id'] == tid]['gene_id'].iloc[0]
            })

    h9_df = pd.DataFrame(h9_specific).sort_values('h9_reads', ascending=False)
    print(f"\nLoci with >50% reads from H9 (N={len(h9_df)}):")
    if len(h9_df) > 0:
        print(f"{'Locus':<25} {'H9':>8} {'Other':>8} {'H9%':>8} {'Subfamily':<10}")
        for _, row in h9_df.head(10).iterrows():
            print(f"{row['transcript_id'][:24]:<25} {int(row['h9_reads']):>8} {int(row['other_reads']):>8} {row['h9_pct']:>7.1f}% {row['subfamily']:<10}")

    # 3. SHSY5Y-specific hotspots
    print("\n" + "=" * 70)
    print("3. SHSY5Y-Specific Hotspots")
    print("-" * 70)

    shsy_data = locus_df[locus_df['cell_line'] == 'SHSY5Y'].groupby('transcript_id')['read_count'].sum()
    other_data2 = locus_df[locus_df['cell_line'] != 'SHSY5Y'].groupby('transcript_id')['read_count'].sum()

    shsy_specific = []
    for tid in shsy_data.index:
        shsy_count = shsy_data[tid]
        other_count = other_data2.get(tid, 0)
        total = shsy_count + other_count
        if total >= 10 and shsy_count / total >= 0.3:
            shsy_specific.append({
                'transcript_id': tid,
                'shsy_reads': shsy_count,
                'other_reads': other_count,
                'shsy_pct': shsy_count / total * 100,
                'subfamily': locus_df[locus_df['transcript_id'] == tid]['gene_id'].iloc[0]
            })

    shsy_df = pd.DataFrame(shsy_specific).sort_values('shsy_reads', ascending=False)
    print(f"\nLoci with >30% reads from SHSY5Y (N={len(shsy_df)}):")
    if len(shsy_df) > 0:
        print(f"{'Locus':<25} {'SHSY5Y':>8} {'Other':>8} {'SHSY%':>8} {'Subfamily':<10}")
        for _, row in shsy_df.head(10).iterrows():
            print(f"{row['transcript_id'][:24]:<25} {int(row['shsy_reads']):>8} {int(row['other_reads']):>8} {row['shsy_pct']:>7.1f}% {row['subfamily']:<10}")
    else:
        print("  No SHSY5Y-enriched loci found")

    # 4. Young L1 in H9/SHSY5Y vs others
    print("\n" + "=" * 70)
    print("4. Young L1 (L1HS + L1PA1-2) Distribution")
    print("-" * 70)

    def is_young(gene_id):
        if gene_id.startswith('L1HS'):
            return True
        if gene_id.startswith('L1PA'):
            rest = gene_id[4:].split('_')[0]
            if rest.isdigit() and int(rest) <= 2:
                return True
        return False

    locus_df['is_young'] = locus_df['gene_id'].apply(is_young)

    young_by_cell = locus_df.groupby('cell_line').apply(
        lambda x: x[x['is_young']]['read_count'].sum() / x['read_count'].sum() * 100 if x['read_count'].sum() > 0 else 0
    ).sort_values(ascending=False)

    print("\nYoung L1 (L1HS + L1PA1-2) percentage:")
    for cl, pct in young_by_cell.items():
        marker = " ← ESC" if cl == "H9" else (" ← Neuronal" if cl == "SHSY5Y" else "")
        print(f"  {cl:<15} {pct:>6.2f}%{marker}")

    # 5. Unique loci count
    print("\n" + "=" * 70)
    print("5. Number of Unique L1 Loci Expressed")
    print("-" * 70)

    loci_by_cell = locus_df.groupby('cell_line')['transcript_id'].nunique().sort_values(ascending=False)
    for cl, count in loci_by_cell.items():
        marker = " ← ESC" if cl == "H9" else (" ← Neuronal" if cl == "SHSY5Y" else "")
        print(f"  {cl:<15} {count:>6} loci{marker}")

if __name__ == "__main__":
    main()
