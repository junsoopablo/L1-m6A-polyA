#!/usr/bin/env python3
"""Prepare Cat B read ID files per group for the nc_exonic pipeline.

Reads catB_reads_detail.tsv, maps sample→group, outputs:
  - catB_readIDs_{group}.txt (one read_id per line)
  - catB_reads_{group}.tsv (full detail for downstream analysis)
  - catB_group_summary.tsv (read counts per group)

Usage:
    conda run -n research python prepare_catB_readids.py
"""

import os
import sys
from collections import defaultdict
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
CATB_DETAIL = os.path.join(
    PROJECT_DIR,
    "analysis/01_exploration/topic_05_cellline/exon_type_analysis/catB_reads_detail.tsv",
)
OUT_DIR = os.path.join(
    PROJECT_DIR, "analysis/01_exploration/topic_07_catB_nc_exonic"
)

# Only active samples from config.yaml (exclude THP1, HCMV, SCV2)
ACTIVE_SAMPLES = [
    "A549_4_1", "A549_5_1", "A549_6_1",
    "H9_2_1", "H9_2_2", "H9_3_1", "H9_3_2", "H9_4_1", "H9_4_2",
    "Hct116_3_1", "Hct116_3_4", "Hct116_4_3",
    "HeLa_1_1", "HeLa_2_1", "HeLa_3_1",
    "HeLa-Ars_1_1", "HeLa-Ars_2_1", "HeLa-Ars_3_1",
    "HepG2_5_1", "HepG2_5_2", "HepG2_6_1",
    "HEYA8_1_1", "HEYA8_1_2", "HEYA8_2_1", "HEYA8_2_2", "HEYA8_3_1",
    "K562_4_1", "K562_5_1", "K562_6_1",
    "MCF7_2_3", "MCF7_3_1", "MCF7_4_1",
    "MCF7-EV_1_1",
    "SHSY5Y_1_1", "SHSY5Y_2_1", "SHSY5Y_3_1",
    "Hek293T_3_1", "Hek293T_4_1",
    "Hek293_1_1",
]


def sample_group(sample):
    """Same logic as Snakemake: HeLa_1_1 → HeLa_1."""
    parts = sample.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return sample


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    active_set = set(ACTIVE_SAMPLES)

    print(f"Reading {CATB_DETAIL} ...")
    df = pd.read_csv(CATB_DETAIL, sep="\t")
    print(f"  Total Cat B reads: {len(df):,}")

    # Filter to active samples
    df = df[df["sample"].isin(active_set)].copy()
    print(f"  After filtering to active samples: {len(df):,}")

    # Add group column
    df["group"] = df["sample"].apply(sample_group)
    groups = sorted(df["group"].unique())
    print(f"  Groups: {len(groups)}")

    # ── Per-group output ──
    summary_rows = []
    for group in groups:
        gdf = df[df["group"] == group]
        n_reads = len(gdf)
        n_samples = gdf["sample"].nunique()
        n_young = (gdf["age"] == "young").sum()
        n_ancient = (gdf["age"] == "ancient").sum()

        # Write read IDs (for POD5 subset)
        readid_file = os.path.join(OUT_DIR, f"catB_readIDs_{group}.txt")
        gdf["read_id"].to_csv(readid_file, index=False, header=False)

        # Write full detail (for analysis)
        detail_file = os.path.join(OUT_DIR, f"catB_reads_{group}.tsv")
        gdf.to_csv(detail_file, sep="\t", index=False)

        summary_rows.append({
            "group": group,
            "n_reads": n_reads,
            "n_samples": n_samples,
            "n_young": n_young,
            "n_ancient": n_ancient,
            "pct_young": round(100 * n_young / n_reads, 1) if n_reads > 0 else 0,
            "samples": ",".join(sorted(gdf["sample"].unique())),
        })
        print(f"  {group}: {n_reads:,} reads ({n_young} young, {n_ancient} ancient)")

    # ── Per-sample read IDs (for POD5 subset, which is per-sample) ──
    samples_with_data = sorted(df["sample"].unique())
    sample_dir = os.path.join(OUT_DIR, "_by_sample")
    os.makedirs(sample_dir, exist_ok=True)
    for sample in samples_with_data:
        sdf = df[df["sample"] == sample]
        readid_file = os.path.join(sample_dir, f"catB_readIDs_{sample}.txt")
        sdf["read_id"].to_csv(readid_file, index=False, header=False)

    # ── Group summary ──
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(OUT_DIR, "catB_group_summary.tsv")
    summary_df.to_csv(summary_file, sep="\t", index=False)
    print(f"\nGroup summary written to {summary_file}")
    print(f"Total: {summary_df['n_reads'].sum():,} reads across {len(groups)} groups")

    # ── Print groups list for sbatch scripts ──
    print("\n# Groups for sbatch scripts (copy-paste):")
    print("GROUPS=(")
    for g in groups:
        print(f'    "{g}"')
    print(")")


if __name__ == "__main__":
    main()
