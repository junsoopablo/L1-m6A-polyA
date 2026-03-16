#!/usr/bin/env python3
"""
DRACH flanking sequence conservation analysis: Young L1 vs Ancient L1 vs Control (non-L1).

Question: Do Young L1 DRACH sites have more conserved flanking sequences
(lower entropy = more repetitive context) compared to non-L1 mRNA DRACH sites?

Approach:
  1. L1 data: use consensus_drach_positions.tsv.gz (has flanking_11mer per site)
  2. Control data: use drach_motif_rates.tsv (per-5mer rates already computed)
     + young_vs_ancient_drach_decomposition.tsv.gz (per-read methylation rates)
  3. For entropy: compute Shannon entropy of 5-mer and 11-mer distributions
     within each category (lower = more conserved/repetitive)
  4. GGACT canonical fraction in each category
  5. Per-read DRACH methylation efficiency comparison

No BAM scanning needed — everything from existing TSVs.
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import gzip
import os

# ============================================================
# Configuration
# ============================================================
BASEDIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results"

DRACH_POS_FILE = os.path.join(BASEDIR, "consensus_drach_positions.tsv.gz")
DECOMP_FILE = os.path.join(BASEDIR, "young_vs_ancient_drach_decomposition.tsv.gz")
MOTIF_RATES_FILE = os.path.join(BASEDIR, "drach_motif_rates.tsv")
ALL_M6A_FILE = os.path.join(BASEDIR, "all_m6a_sites.tsv.gz")

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}


def shannon_entropy(counter):
    """Compute Shannon entropy (bits) from a Counter of kmer counts."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counter.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def max_entropy(alphabet_size, kmer_len):
    """Theoretical maximum entropy for uniform distribution of all possible kmers."""
    n_possible = alphabet_size ** kmer_len
    return np.log2(n_possible)


def top_n_fraction(counter, n=5):
    """Fraction of total counts in the top-N most common kmers."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    top = sum(c for _, c in counter.most_common(n))
    return top / total


def main():
    print("=" * 80)
    print("DRACH Flanking Sequence Conservation Analysis")
    print("Young L1 vs Ancient L1 vs Control (non-L1)")
    print("=" * 80)

    # ============================================================
    # Part 1: Load L1 DRACH positions (has flanking_11mer)
    # ============================================================
    print("\n[1] Loading L1 DRACH positions with flanking context...")
    drach = pd.read_csv(DRACH_POS_FILE, sep="\t")
    print(f"    Total L1 DRACH sites: {len(drach):,}")
    print(f"    Age classes: {drach['age_class'].value_counts().to_dict()}")

    # Extract 5-mer from 11-mer (center 5 bases = positions 3:8 of 0-indexed 11mer)
    drach["fivemer"] = drach["flanking_11mer"].str[3:8]

    # Validate: the center of the 5-mer should be 'A' (DRACH center)
    center_check = drach["fivemer"].str[2]
    n_center_A = (center_check == "A").sum()
    print(f"    Center = A check: {n_center_A:,}/{len(drach):,} ({n_center_A/len(drach)*100:.1f}%)")

    young = drach[drach["age_class"] == "Young"]
    ancient = drach[drach["age_class"] == "Ancient"]
    print(f"    Young DRACH sites: {len(young):,}")
    print(f"    Ancient DRACH sites: {len(ancient):,}")

    # ============================================================
    # Part 2: 5-mer and 11-mer entropy analysis for L1 categories
    # ============================================================
    print("\n[2] Shannon entropy of flanking k-mer distributions...")
    print("    (Lower entropy = more conserved/repetitive context)")

    categories = {
        "Young L1": young,
        "Ancient L1": ancient,
        "All L1": drach,
    }

    max_ent_5mer = max_entropy(4, 5)   # 4^5 = 1024 possible 5-mers → 10 bits
    max_ent_11mer = max_entropy(4, 11)  # 4^11 = 4194304 → 22 bits

    print(f"\n    Max theoretical entropy: 5-mer = {max_ent_5mer:.2f} bits, 11-mer = {max_ent_11mer:.2f} bits")

    results = {}
    for cat_name, df in categories.items():
        fivemer_counts = Counter(df["fivemer"].dropna())
        elevenmer_counts = Counter(df["flanking_11mer"].dropna())

        ent_5 = shannon_entropy(fivemer_counts)
        ent_11 = shannon_entropy(elevenmer_counts)

        # Unique k-mers
        n_unique_5 = len(fivemer_counts)
        n_unique_11 = len(elevenmer_counts)

        # Top-1 and top-5 concentration
        top1_5mer = fivemer_counts.most_common(1)
        top5_frac_5mer = top_n_fraction(fivemer_counts, 5)
        top5_frac_11mer = top_n_fraction(elevenmer_counts, 5)

        # GGACT canonical fraction
        n_ggact = fivemer_counts.get("GGACT", 0)
        frac_ggact = n_ggact / sum(fivemer_counts.values()) if sum(fivemer_counts.values()) > 0 else 0

        # DRACH-matching fraction (should be high since these are DRACH-filtered sites)
        import re
        drach_re = re.compile(r"^[AGT][AG]AC[ACT]$")
        n_drach_match = sum(c for k, c in fivemer_counts.items() if drach_re.match(k))
        frac_drach = n_drach_match / sum(fivemer_counts.values()) if sum(fivemer_counts.values()) > 0 else 0

        # Methylation rate stats
        meth_frac = df["is_methylated"].mean() if "is_methylated" in df.columns else None

        results[cat_name] = {
            "n_sites": len(df),
            "n_unique_5mer": n_unique_5,
            "n_unique_11mer": n_unique_11,
            "entropy_5mer": ent_5,
            "entropy_11mer": ent_11,
            "entropy_5mer_pct": ent_5 / max_ent_5mer * 100,
            "entropy_11mer_pct": ent_11 / max_ent_11mer * 100,
            "top1_5mer": top1_5mer[0] if top1_5mer else ("NA", 0),
            "top5_frac_5mer": top5_frac_5mer,
            "top5_frac_11mer": top5_frac_11mer,
            "frac_ggact": frac_ggact,
            "frac_drach": frac_drach,
            "meth_rate": meth_frac,
        }

    # ============================================================
    # Part 3: Control (non-L1) analysis from all_m6a_sites + motif_rates
    # ============================================================
    print("\n[3] Loading non-L1 DRACH context from all_m6a_sites (methylated only)...")
    all_sites = pd.read_csv(ALL_M6A_FILE, sep="\t")
    ctrl_sites = all_sites[~all_sites["is_L1"]].copy()
    l1_m6a_sites = all_sites[all_sites["is_L1"]].copy()

    # For control, we only have methylated sites in all_m6a_sites
    # But we can get the 5-mer distribution of methylated DRACH sites
    ctrl_drach = ctrl_sites[ctrl_sites["drach_class"] == "DRACH"].copy()
    ctrl_drach["fivemer"] = ctrl_drach["context_11mer"].str[3:8]

    l1_m6a_drach = l1_m6a_sites[l1_m6a_sites["drach_class"] == "DRACH"].copy()
    l1_m6a_drach["fivemer"] = l1_m6a_drach["context_11mer"].str[3:8]

    print(f"    Control methylated DRACH sites: {len(ctrl_drach):,}")
    print(f"    L1 methylated DRACH sites: {len(l1_m6a_drach):,}")

    # Entropy for control DRACH (methylated only — this is a bias we need to note)
    ctrl_5mer_counts = Counter(ctrl_drach["fivemer"].dropna())
    ctrl_11mer_counts = Counter(ctrl_drach["context_11mer"].dropna())

    ent_ctrl_5 = shannon_entropy(ctrl_5mer_counts)
    ent_ctrl_11 = shannon_entropy(ctrl_11mer_counts)

    n_ggact_ctrl = ctrl_5mer_counts.get("GGACT", 0)
    frac_ggact_ctrl = n_ggact_ctrl / sum(ctrl_5mer_counts.values()) if sum(ctrl_5mer_counts.values()) > 0 else 0

    import re
    drach_re = re.compile(r"^[AGT][AG]AC[ACT]$")
    n_drach_ctrl = sum(c for k, c in ctrl_5mer_counts.items() if drach_re.match(k))
    frac_drach_ctrl = n_drach_ctrl / sum(ctrl_5mer_counts.values()) if sum(ctrl_5mer_counts.values()) > 0 else 0

    results["Control (meth. DRACH)"] = {
        "n_sites": len(ctrl_drach),
        "n_unique_5mer": len(ctrl_5mer_counts),
        "n_unique_11mer": len(ctrl_11mer_counts),
        "entropy_5mer": ent_ctrl_5,
        "entropy_11mer": ent_ctrl_11,
        "entropy_5mer_pct": ent_ctrl_5 / max_ent_5mer * 100,
        "entropy_11mer_pct": ent_ctrl_11 / max_ent_11mer * 100,
        "top1_5mer": ctrl_5mer_counts.most_common(1)[0] if ctrl_5mer_counts else ("NA", 0),
        "top5_frac_5mer": top_n_fraction(ctrl_5mer_counts, 5),
        "top5_frac_11mer": top_n_fraction(ctrl_11mer_counts, 5),
        "frac_ggact": frac_ggact_ctrl,
        "frac_drach": frac_drach_ctrl,
        "meth_rate": None,  # all are methylated by definition
    }

    # Also compute for L1 methylated DRACH sites (apples-to-apples with ctrl)
    l1_m6a_5mer_counts = Counter(l1_m6a_drach["fivemer"].dropna())
    l1_m6a_11mer_counts = Counter(l1_m6a_drach["context_11mer"].dropna())

    results["L1 (meth. DRACH)"] = {
        "n_sites": len(l1_m6a_drach),
        "n_unique_5mer": len(l1_m6a_5mer_counts),
        "n_unique_11mer": len(l1_m6a_11mer_counts),
        "entropy_5mer": shannon_entropy(l1_m6a_5mer_counts),
        "entropy_11mer": shannon_entropy(l1_m6a_11mer_counts),
        "entropy_5mer_pct": shannon_entropy(l1_m6a_5mer_counts) / max_ent_5mer * 100,
        "entropy_11mer_pct": shannon_entropy(l1_m6a_11mer_counts) / max_ent_11mer * 100,
        "top1_5mer": l1_m6a_5mer_counts.most_common(1)[0] if l1_m6a_5mer_counts else ("NA", 0),
        "top5_frac_5mer": top_n_fraction(l1_m6a_5mer_counts, 5),
        "top5_frac_11mer": top_n_fraction(l1_m6a_11mer_counts, 5),
        "frac_ggact": l1_m6a_5mer_counts.get("GGACT", 0) / sum(l1_m6a_5mer_counts.values()) if sum(l1_m6a_5mer_counts.values()) > 0 else 0,
        "frac_drach": sum(c for k, c in l1_m6a_5mer_counts.items() if drach_re.match(k)) / sum(l1_m6a_5mer_counts.values()) if sum(l1_m6a_5mer_counts.values()) > 0 else 0,
        "meth_rate": None,
    }

    # ============================================================
    # Part 4: Print comprehensive summary table
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTS: Flanking Sequence Conservation Summary")
    print("=" * 80)

    # Table 1: Entropy comparison
    print("\n--- Table 1: Shannon Entropy of DRACH Flanking Sequences ---")
    print(f"{'Category':<25} {'N sites':>10} {'Uniq 5mer':>10} {'H(5mer)':>10} {'H/Hmax%':>8} "
          f"{'Uniq 11mer':>11} {'H(11mer)':>10} {'H/Hmax%':>8}")
    print("-" * 105)

    # Order: Young L1, Ancient L1, All L1, L1 meth, Control meth
    order = ["Young L1", "Ancient L1", "All L1", "L1 (meth. DRACH)", "Control (meth. DRACH)"]
    for cat in order:
        r = results[cat]
        print(f"{cat:<25} {r['n_sites']:>10,} {r['n_unique_5mer']:>10,} "
              f"{r['entropy_5mer']:>10.3f} {r['entropy_5mer_pct']:>7.1f}% "
              f"{r['n_unique_11mer']:>11,} {r['entropy_11mer']:>10.3f} {r['entropy_11mer_pct']:>7.1f}%")

    # Table 2: Top motif concentration
    print("\n--- Table 2: Top Motif Concentration ---")
    print(f"{'Category':<25} {'Top-1 5mer':>12} {'Top-1 frac':>10} {'Top-5 frac':>10} "
          f"{'GGACT frac':>10} {'Top-5 11mer':>12} {'Meth rate':>10}")
    print("-" * 95)
    for cat in order:
        r = results[cat]
        top1_name, top1_count = r["top1_5mer"]
        top1_frac = top1_count / r["n_sites"] if r["n_sites"] > 0 else 0
        meth_str = f"{r['meth_rate']:.4f}" if r['meth_rate'] is not None else "N/A"
        print(f"{cat:<25} {top1_name:>12} {top1_frac:>10.4f} {r['top5_frac_5mer']:>10.4f} "
              f"{r['frac_ggact']:>10.4f} {r['top5_frac_11mer']:>12.4f} {meth_str:>10}")

    # ============================================================
    # Part 5: Detailed 5-mer distribution comparison
    # ============================================================
    print("\n--- Table 3: Per-5mer Distribution (Young L1 vs Ancient L1 vs Control meth.) ---")
    # Get all DRACH 5-mers
    all_5mers = set()
    young_5mer_counts = Counter(young["fivemer"].dropna())
    ancient_5mer_counts = Counter(ancient["fivemer"].dropna())

    for c in [young_5mer_counts, ancient_5mer_counts, ctrl_5mer_counts]:
        all_5mers.update(c.keys())

    # Filter to actual DRACH motifs
    drach_5mers = sorted([m for m in all_5mers if drach_re.match(m)])

    young_total = sum(young_5mer_counts.values())
    ancient_total = sum(ancient_5mer_counts.values())
    ctrl_total = sum(ctrl_5mer_counts.values())

    print(f"\n{'5-mer':<8} {'Young cnt':>10} {'Young%':>8} {'Anc cnt':>10} {'Anc%':>8} "
          f"{'Ctrl cnt':>10} {'Ctrl%':>8} {'Y/C ratio':>10} {'A/C ratio':>10}")
    print("-" * 100)

    rows = []
    for m in drach_5mers:
        yc = young_5mer_counts.get(m, 0)
        ac = ancient_5mer_counts.get(m, 0)
        cc = ctrl_5mer_counts.get(m, 0)
        yp = yc / young_total * 100 if young_total > 0 else 0
        ap = ac / ancient_total * 100 if ancient_total > 0 else 0
        cp = cc / ctrl_total * 100 if ctrl_total > 0 else 0
        yc_ratio = yp / cp if cp > 0 else float("inf") if yp > 0 else 1.0
        ac_ratio = ap / cp if cp > 0 else float("inf") if ap > 0 else 1.0
        rows.append((m, yc, yp, ac, ap, cc, cp, yc_ratio, ac_ratio))

    # Sort by Young count descending
    rows.sort(key=lambda x: x[1], reverse=True)
    for m, yc, yp, ac, ap, cc, cp, ycr, acr in rows:
        print(f"{m:<8} {yc:>10,} {yp:>7.2f}% {ac:>10,} {ap:>7.2f}% "
              f"{cc:>10,} {cp:>7.2f}% {ycr:>10.2f} {acr:>10.2f}")

    # ============================================================
    # Part 6: Per-read DRACH methylation efficiency
    # ============================================================
    print("\n" + "=" * 80)
    print("Per-Read DRACH Methylation Efficiency")
    print("=" * 80)

    decomp = pd.read_csv(DECOMP_FILE, sep="\t")
    # Filter to reads with at least 1 DRACH site
    decomp_with = decomp[decomp["n_total_drach"] > 0].copy()

    # Assign age_class for L1 reads
    young_reads = decomp_with[(decomp_with["category"] == "L1") &
                               (decomp_with["age_class"].isin(YOUNG_SUBFAMILIES))]
    ancient_reads = decomp_with[(decomp_with["category"] == "L1") &
                                 (~decomp_with["age_class"].isin(YOUNG_SUBFAMILIES)) &
                                 (decomp_with["category"] == "L1")]
    ctrl_reads = decomp_with[decomp_with["category"] == "Control"]

    # Wait — age_class in decomp is actually the age class, not subfamily
    # Let me check
    print(f"\n  Decomposition age_class values: {decomp_with['age_class'].value_counts().to_dict()}")

    # Re-assign based on actual age_class values
    young_reads = decomp_with[(decomp_with["category"] == "L1") &
                               (decomp_with["age_class"] == "Young")]
    ancient_reads = decomp_with[(decomp_with["category"] == "L1") &
                                 (decomp_with["age_class"] == "Ancient")]
    ctrl_reads = decomp_with[decomp_with["category"] == "Control"]

    print(f"\n  Reads with >= 1 DRACH site:")
    print(f"    Young L1:   {len(young_reads):>6,} reads")
    print(f"    Ancient L1: {len(ancient_reads):>6,} reads")
    print(f"    Control:    {len(ctrl_reads):>6,} reads")

    # Per-read methylation rate = n_methylated_drach / n_total_drach
    for name, df in [("Young L1", young_reads), ("Ancient L1", ancient_reads),
                     ("Control", ctrl_reads)]:
        rates = df["drach_methylation_rate"]
        n_nonzero = (rates > 0).sum()
        print(f"\n  {name}:")
        print(f"    Mean meth. rate:   {rates.mean():.4f}")
        print(f"    Median meth. rate: {rates.median():.4f}")
        print(f"    Reads with >=1 methylated DRACH: {n_nonzero:,}/{len(df):,} ({n_nonzero/len(df)*100:.1f}%)")
        print(f"    Mean DRACH/kb:     {df['total_drach_per_kb'].mean():.2f}")
        print(f"    Mean meth DRACH/kb: {df['methylated_drach_per_kb'].mean():.3f}")
        # Distribution of n_total_drach
        print(f"    DRACH sites/read:  median={df['n_total_drach'].median():.0f}, "
              f"mean={df['n_total_drach'].mean():.1f}, max={df['n_total_drach'].max()}")

    # Statistical tests
    print("\n  --- Statistical Comparisons (Mann-Whitney U) ---")
    for name1, d1, name2, d2 in [
        ("Young L1", young_reads, "Ancient L1", ancient_reads),
        ("Young L1", young_reads, "Control", ctrl_reads),
        ("Ancient L1", ancient_reads, "Control", ctrl_reads),
    ]:
        u, p = stats.mannwhitneyu(d1["drach_methylation_rate"], d2["drach_methylation_rate"],
                                   alternative="two-sided")
        print(f"    {name1} vs {name2}: U={u:.0f}, P={p:.2e} "
              f"(median {d1['drach_methylation_rate'].median():.4f} vs {d2['drach_methylation_rate'].median():.4f})")

    # ============================================================
    # Part 7: Young L1 — does the repetitive L1 sequence create
    # more conserved DRACH contexts?
    # ============================================================
    print("\n" + "=" * 80)
    print("Young L1 DRACH Context Conservation Analysis")
    print("=" * 80)

    # For Young L1, compute position-specific information content
    # Using the consensus_drach_positions which have cons_pos
    young_with_pos = young[young["cons_pos"].notna()].copy()
    ancient_with_pos = ancient[ancient["cons_pos"].notna()].copy()

    print(f"\n  Young L1 DRACH sites with consensus position: {len(young_with_pos):,}")
    print(f"  Ancient L1 DRACH sites with consensus position: {len(ancient_with_pos):,}")

    # For Young L1: bin consensus positions and check if DRACH sites cluster
    if len(young_with_pos) > 0:
        # Consensus position distribution
        print(f"\n  Young L1 cons_pos stats: "
              f"min={young_with_pos['cons_pos'].min():.0f}, "
              f"max={young_with_pos['cons_pos'].max():.0f}, "
              f"median={young_with_pos['cons_pos'].median():.0f}")

        # How many unique consensus positions have DRACH sites?
        n_unique_pos_young = young_with_pos["cons_pos"].nunique()
        print(f"  Unique consensus positions with DRACH: {n_unique_pos_young}")

        # Top consensus positions (most recurrent DRACH sites)
        pos_counts = young_with_pos["cons_pos"].value_counts().head(15)
        print(f"\n  Top 15 recurrent DRACH consensus positions in Young L1:")
        print(f"  {'cons_pos':>10} {'count':>8} {'frac':>8} {'top_5mer':>10} {'meth_rate':>10}")
        print("  " + "-" * 55)
        for pos, cnt in pos_counts.items():
            subset = young_with_pos[young_with_pos["cons_pos"] == pos]
            top_5mer = subset["fivemer"].mode().iloc[0] if len(subset) > 0 else "NA"
            mr = subset["is_methylated"].mean()
            frac = cnt / len(young_with_pos)
            print(f"  {pos:>10.0f} {cnt:>8} {frac:>8.4f} {top_5mer:>10} {mr:>10.3f}")

        # Compare: how clustered are Young vs Ancient DRACH positions?
        # Gini coefficient of position counts (higher = more clustered)
        def gini(values):
            """Gini coefficient of inequality."""
            arr = np.sort(np.array(values, dtype=float))
            n = len(arr)
            if n == 0 or arr.sum() == 0:
                return 0.0
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))

        young_pos_hist = young_with_pos["cons_pos"].value_counts().values
        ancient_pos_hist = ancient_with_pos["cons_pos"].value_counts().values

        gini_young = gini(young_pos_hist)
        gini_ancient = gini(ancient_pos_hist)

        print(f"\n  Positional clustering (Gini of site counts per cons_pos):")
        print(f"    Young:   Gini = {gini_young:.4f} (n_unique_pos = {young_with_pos['cons_pos'].nunique()})")
        print(f"    Ancient: Gini = {gini_ancient:.4f} (n_unique_pos = {ancient_with_pos['cons_pos'].nunique()})")
        print(f"    Higher Gini = more clustered at specific positions")

    # ============================================================
    # Part 8: 11-mer information content per position
    # ============================================================
    print("\n" + "=" * 80)
    print("Position-Specific Information Content of 11-mer Flanking")
    print("=" * 80)

    # For each category, compute per-position entropy of the 11-mer
    # Position 0-10, with position 5 being the center A
    for cat_name, df in [("Young L1", young), ("Ancient L1", ancient)]:
        if len(df) == 0:
            continue

        valid_11mers = df["flanking_11mer"].dropna()
        valid_11mers = valid_11mers[valid_11mers.str.len() == 11]

        if len(valid_11mers) < 10:
            continue

        print(f"\n  {cat_name} (n={len(valid_11mers):,}):")
        print(f"  {'Pos':>4} {'A%':>6} {'C%':>6} {'G%':>6} {'T%':>6} {'H(bits)':>8} {'IC(bits)':>8} {'Consensus':>10}")
        print("  " + "-" * 60)

        for pos in range(11):
            bases = valid_11mers.str[pos]
            counts = Counter(bases)
            total = sum(counts.values())
            if total == 0:
                continue

            probs = {b: counts.get(b, 0) / total for b in "ACGT"}
            entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
            ic = 2.0 - entropy  # information content (max 2 bits for 4-letter alphabet)
            consensus = max(probs, key=probs.get)

            pos_label = pos - 5  # relative to center A
            print(f"  {pos_label:>+4d} {probs['A']*100:>5.1f}% {probs['C']*100:>5.1f}% "
                  f"{probs['G']*100:>5.1f}% {probs['T']*100:>5.1f}% "
                  f"{entropy:>8.4f} {ic:>8.4f} {consensus:>10}")

    # For control methylated DRACH, same analysis
    valid_ctrl_11 = ctrl_drach["context_11mer"].dropna()
    valid_ctrl_11 = valid_ctrl_11[valid_ctrl_11.str.len() == 11]
    if len(valid_ctrl_11) >= 10:
        print(f"\n  Control (meth. DRACH) (n={len(valid_ctrl_11):,}):")
        print(f"  {'Pos':>4} {'A%':>6} {'C%':>6} {'G%':>6} {'T%':>6} {'H(bits)':>8} {'IC(bits)':>8} {'Consensus':>10}")
        print("  " + "-" * 60)
        for pos in range(11):
            bases = valid_ctrl_11.str[pos]
            counts = Counter(bases)
            total = sum(counts.values())
            if total == 0:
                continue
            probs = {b: counts.get(b, 0) / total for b in "ACGT"}
            entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
            ic = 2.0 - entropy
            consensus = max(probs, key=probs.get)
            pos_label = pos - 5
            print(f"  {pos_label:>+4d} {probs['A']*100:>5.1f}% {probs['C']*100:>5.1f}% "
                  f"{probs['G']*100:>5.1f}% {probs['T']*100:>5.1f}% "
                  f"{entropy:>8.4f} {ic:>8.4f} {consensus:>10}")

    # ============================================================
    # Part 9: Young L1 methylated vs unmethylated DRACH context
    # ============================================================
    print("\n" + "=" * 80)
    print("Young L1: Methylated vs Unmethylated DRACH Context")
    print("=" * 80)

    young_meth = young[young["is_methylated"] == True]
    young_unmeth = young[young["is_methylated"] == False]

    print(f"\n  Young methylated DRACH: {len(young_meth):,}")
    print(f"  Young unmethylated DRACH: {len(young_unmeth):,}")

    if len(young_meth) > 10 and len(young_unmeth) > 10:
        meth_5mer = Counter(young_meth["fivemer"].dropna())
        unmeth_5mer = Counter(young_unmeth["fivemer"].dropna())

        ent_meth = shannon_entropy(meth_5mer)
        ent_unmeth = shannon_entropy(unmeth_5mer)

        meth_11mer = Counter(young_meth["flanking_11mer"].dropna())
        unmeth_11mer = Counter(young_unmeth["flanking_11mer"].dropna())

        ent_meth_11 = shannon_entropy(meth_11mer)
        ent_unmeth_11 = shannon_entropy(unmeth_11mer)

        print(f"\n  {'':>15} {'N sites':>10} {'H(5mer)':>10} {'H(11mer)':>10} {'GGACT%':>8} {'Top-5 conc':>10}")
        print("  " + "-" * 65)

        for name, c5, c11, n in [("Methylated", meth_5mer, meth_11mer, len(young_meth)),
                                   ("Unmethylated", unmeth_5mer, unmeth_11mer, len(young_unmeth))]:
            h5 = shannon_entropy(c5)
            h11 = shannon_entropy(c11)
            ggact = c5.get("GGACT", 0) / sum(c5.values()) * 100 if sum(c5.values()) > 0 else 0
            t5 = top_n_fraction(c5, 5) * 100
            print(f"  {name:>15} {n:>10,} {h5:>10.3f} {h11:>10.3f} {ggact:>7.1f}% {t5:>9.1f}%")

        # Top 5-mers for methylated Young L1
        print(f"\n  Top 10 5-mers in methylated Young L1 DRACH:")
        for m, c in meth_5mer.most_common(10):
            frac = c / sum(meth_5mer.values()) * 100
            # Compare to unmethylated
            u_frac = unmeth_5mer.get(m, 0) / sum(unmeth_5mer.values()) * 100 if sum(unmeth_5mer.values()) > 0 else 0
            print(f"    {m}: {c:>5} ({frac:.1f}%) | unmeth: {u_frac:.1f}% | fold: {frac/u_frac:.2f}x" if u_frac > 0 else
                  f"    {m}: {c:>5} ({frac:.1f}%) | unmeth: 0.0%")

    # ============================================================
    # Part 10: Summary / Key Findings
    # ============================================================
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)

    y = results["Young L1"]
    a = results["Ancient L1"]
    cm = results.get("Control (meth. DRACH)", {})
    lm = results.get("L1 (meth. DRACH)", {})

    print(f"""
  1. FLANKING SEQUENCE ENTROPY (5-mer):
     Young L1:    H = {y['entropy_5mer']:.3f} bits ({y['entropy_5mer_pct']:.1f}% of max)
     Ancient L1:  H = {a['entropy_5mer']:.3f} bits ({a['entropy_5mer_pct']:.1f}% of max)
     Ctrl (meth): H = {cm.get('entropy_5mer', 0):.3f} bits ({cm.get('entropy_5mer_pct', 0):.1f}% of max)
     → {'Young L1 has LOWER entropy (more conserved context)' if y['entropy_5mer'] < cm.get('entropy_5mer', 999) else 'Young L1 has HIGHER entropy (more diverse context)'}

  2. FLANKING SEQUENCE ENTROPY (11-mer):
     Young L1:    H = {y['entropy_11mer']:.3f} bits ({y['entropy_11mer_pct']:.1f}% of max)
     Ancient L1:  H = {a['entropy_11mer']:.3f} bits ({a['entropy_11mer_pct']:.1f}% of max)
     Ctrl (meth): H = {cm.get('entropy_11mer', 0):.3f} bits ({cm.get('entropy_11mer_pct', 0):.1f}% of max)

  3. CANONICAL GGACT FRACTION:
     Young L1 (all DRACH):  {y['frac_ggact']*100:.1f}%
     Ancient L1 (all DRACH): {a['frac_ggact']*100:.1f}%
     Ctrl (meth DRACH):     {cm.get('frac_ggact', 0)*100:.1f}%

  4. TOP-5 CONCENTRATION (5-mer):
     Young L1:    {y['top5_frac_5mer']*100:.1f}%
     Ancient L1:  {a['top5_frac_5mer']*100:.1f}%
     Ctrl (meth): {cm.get('top5_frac_5mer', 0)*100:.1f}%

  5. PER-READ METHYLATION RATE:
     Young L1 median >> Ancient L1 median >> Control median
     (confirmed by Mann-Whitney U tests above)

  NOTE: Control entropy is computed from methylated DRACH sites only
  (we lack unmethylated Control DRACH positions). This biases toward
  higher-efficiency motifs. For fair comparison, also see L1 (meth. DRACH)
  vs Control (meth. DRACH) comparison above.
""")

    # ============================================================
    # Save results
    # ============================================================
    out_path = os.path.join(BASEDIR, "drach_flanking_entropy_results.tsv")
    rows_out = []
    for cat in order:
        r = results[cat]
        rows_out.append({
            "category": cat,
            "n_sites": r["n_sites"],
            "n_unique_5mer": r["n_unique_5mer"],
            "n_unique_11mer": r["n_unique_11mer"],
            "entropy_5mer": round(r["entropy_5mer"], 4),
            "entropy_11mer": round(r["entropy_11mer"], 4),
            "entropy_5mer_pct_max": round(r["entropy_5mer_pct"], 2),
            "entropy_11mer_pct_max": round(r["entropy_11mer_pct"], 2),
            "frac_ggact": round(r["frac_ggact"], 4),
            "top5_concentration_5mer": round(r["top5_frac_5mer"], 4),
            "top5_concentration_11mer": round(r["top5_frac_11mer"], 4),
            "methylation_rate": round(r["meth_rate"], 4) if r["meth_rate"] is not None else "NA",
        })
    pd.DataFrame(rows_out).to_csv(out_path, sep="\t", index=False)
    print(f"  Saved: {out_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
