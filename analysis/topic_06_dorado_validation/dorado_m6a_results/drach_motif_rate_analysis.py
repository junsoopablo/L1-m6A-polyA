#!/usr/bin/env python3
"""
DRACH motif-specific m6A methylation rate analysis: L1 vs non-L1.

For each of the 16 canonical DRACH 5-mers, compute:
  - Total A positions in that motif context (denominator)
  - Number of m6A calls above threshold (numerator)
  - Methylation rate = numerator / denominator
  - L1 vs non-L1 comparison + Fisher's exact test

Strategy:
  Phase 1: Parse dorado BAM to enumerate ALL A positions with 5-mer context
           (subsample for memory: 10K L1 + 5K non-L1 reads, then scale)
  Phase 2: From the same reads, extract m6A calls at each 5-mer
  Phase 3: Compute per-motif methylation rate and compare L1 vs non-L1

Output:
  drach_motif_rates.tsv  - per-5mer methylation rates for L1 and non-L1
  stdout summary         - top motifs + L1-specific enrichment
"""

import pysam
import numpy as np
import re
import sys
import os
from collections import defaultdict, Counter
from scipy import stats
from bisect import bisect_left

# ============================================================
# Configuration
# ============================================================
BAM_PATH = "/blaze/junsoopablo/dorado_validation/HeLa_1_1_m6A/HeLa_1_1.dorado.m6A.sorted.bam"
REF_PATH = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta"
L1_BED = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/L1_TE_L1_family.bed"
OUTDIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results"

MIN_L1_OVERLAP = 0.10
M6A_THRESHOLD = 204  # 80% probability

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

# Subsample reads for Part 1 (all-A enumeration): cap at N reads each
# More than dorado_drach_comparison.py's 5K for better per-5mer stats
MAX_L1_READS = 10000
MAX_NONL1_READS = 5000

# All 16 canonical DRACH 5-mers: D=[AGT], R=[AG], A=A(center), C=C, H=[ACT]
DRACH_RE = re.compile(r"^[AGT][AG]AC[ACT]$")

def enumerate_drach_5mers():
    """Generate all 16 canonical DRACH 5-mers."""
    D = "AGT"
    R = "AG"
    H = "ACT"
    motifs = []
    for d in D:
        for r in R:
            for h in H:
                motifs.append(f"{d}{r}AC{h}")
    return sorted(motifs)

ALL_DRACH_5MERS = enumerate_drach_5mers()

os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# L1 annotation loading
# ============================================================
def load_l1_intervals(bed_path):
    """Load L1 BED into sorted interval lists per chromosome."""
    print("Loading L1 annotations...")
    l1_data = defaultdict(list)
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom, start, end, repname = parts[0], int(parts[1]), int(parts[2]), parts[3]
            l1_data[chrom].append((start, end, repname))
    for chrom in l1_data:
        l1_data[chrom].sort()
    n_total = sum(len(v) for v in l1_data.values())
    print(f"  Loaded {n_total:,} L1 elements across {len(l1_data)} chromosomes")
    return l1_data


def find_l1_overlap(l1_data, chrom, read_start, read_end):
    """Find L1 overlap for a read. Returns (overlap_frac, best_repname)."""
    if chrom not in l1_data:
        return 0.0, None
    intervals = l1_data[chrom]
    read_len = read_end - read_start
    if read_len <= 0:
        return 0.0, None
    left = bisect_left(intervals, (read_start,))
    start_idx = max(0, left - 1)
    best_overlap, best_repname = 0, None
    for i in range(start_idx, len(intervals)):
        istart, iend, repname = intervals[i]
        if istart >= read_end:
            break
        ov = min(read_end, iend) - max(read_start, istart)
        if ov > best_overlap:
            best_overlap = ov
            best_repname = repname
    return best_overlap / read_len, best_repname


# ============================================================
# MM/ML parsing (from dorado_drach_comparison.py)
# ============================================================
def parse_m6a_from_read(read):
    """
    Parse A+a modification from MM/ML tags.
    Returns dict: {read_pos: ml_probability} for each called m6A A.
    """
    if not read.has_tag("MM") or not read.has_tag("ML"):
        return {}

    mm_str = read.get_tag("MM")
    ml_arr = list(read.get_tag("ML"))
    seq = read.query_sequence
    if seq is None:
        return {}

    mod_sections = [s.strip() for s in mm_str.rstrip(";").split(";") if s.strip()]
    ml_offset = 0
    m6a_map = {}

    for section in mod_sections:
        tokens = section.split(",")
        header = tokens[0]
        skip_values = [int(x) for x in tokens[1:]] if len(tokens) > 1 else []
        n_mods = len(skip_values)

        if header.startswith("A+a"):
            a_positions = [i for i, base in enumerate(seq) if base == "A"]
            a_idx = 0
            for mod_i, skip in enumerate(skip_values):
                a_idx += skip
                if a_idx < len(a_positions):
                    read_pos = a_positions[a_idx]
                    prob = ml_arr[ml_offset + mod_i]
                    m6a_map[read_pos] = prob
                a_idx += 1

        ml_offset += n_mods

    return m6a_map


# ============================================================
# Reference context helpers
# ============================================================
def get_ref_5mer(ref, chrom, ref_pos, is_reverse):
    """
    Get 5-mer centered on the given ref_pos, oriented to the RNA strand.
    Returns DRACH-convention 5-mer where center position is the A (target adenosine).
    """
    start = ref_pos - 2
    end = ref_pos + 3
    if start < 0:
        return None
    try:
        ctx = ref.fetch(chrom, start, end).upper()
    except (ValueError, KeyError):
        return None
    if len(ctx) != 5 or "N" in ctx:
        return None

    if is_reverse:
        comp = str.maketrans("ACGT", "TGCA")
        ctx = ctx.translate(comp)[::-1]

    return ctx


# ============================================================
# Main analysis
# ============================================================
def main():
    print("=" * 70)
    print("DRACH Motif-Specific m6A Methylation Rate Analysis")
    print("L1 vs non-L1 comparison")
    print("=" * 70)

    l1_data = load_l1_intervals(L1_BED)
    bam = pysam.AlignmentFile(BAM_PATH, "rb")
    ref = pysam.FastaFile(REF_PATH)

    print(f"\nAll 16 DRACH 5-mers: {', '.join(ALL_DRACH_5MERS)}")
    print(f"m6A threshold: >= {M6A_THRESHOLD}")
    print(f"Max reads: L1={MAX_L1_READS}, non-L1={MAX_NONL1_READS}")

    # Accumulators per 5-mer per category
    # category: "L1", "nonL1", "young", "ancient"
    # For each 5-mer: total_A_count, m6a_count (>= threshold)
    fivemer_total = defaultdict(lambda: defaultdict(int))   # {category: {5mer: count}}
    fivemer_m6a = defaultdict(lambda: defaultdict(int))     # {category: {5mer: count}}
    fivemer_prob_sum = defaultdict(lambda: defaultdict(float))  # for mean prob

    n_processed = 0
    n_l1 = 0
    n_nonl1_selected = 0
    n_skipped_nonl1 = 0

    print("\nProcessing BAM reads...")
    for read in bam.fetch():
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if not read.has_tag("MM") or not read.has_tag("ML"):
            continue
        seq = read.query_sequence
        if seq is None:
            continue

        chrom = read.reference_name
        read_start = read.reference_start
        read_end = read.reference_end
        if read_end is None:
            continue
        is_reverse = read.is_reverse

        # Classify L1
        overlap_frac, repname = find_l1_overlap(l1_data, chrom, read_start, read_end)
        is_l1 = overlap_frac >= MIN_L1_OVERLAP

        if is_l1:
            n_l1 += 1
            if n_l1 > MAX_L1_READS:
                if n_nonl1_selected >= MAX_NONL1_READS:
                    break
                continue
        else:
            if n_nonl1_selected >= MAX_NONL1_READS:
                n_skipped_nonl1 += 1
                if n_l1 >= MAX_L1_READS:
                    break
                continue
            n_nonl1_selected += 1

        n_processed += 1
        if n_processed % 5000 == 0:
            print(f"  Processed {n_processed:,} reads (L1={min(n_l1, MAX_L1_READS)}, non-L1={n_nonl1_selected})")

        # Parse m6A calls
        m6a_map = parse_m6a_from_read(read)

        # Get aligned pairs
        pairs = read.get_aligned_pairs()
        read2ref = {}
        for rp, rfp in pairs:
            if rp is not None and rfp is not None:
                read2ref[rp] = rfp

        # Find all A positions in the read
        a_positions = [i for i, base in enumerate(seq) if base == "A"]

        # Determine categories
        if is_l1:
            categories = ["L1"]
            if repname in YOUNG_SUBFAMILIES:
                categories.append("young")
            else:
                categories.append("ancient")
        else:
            categories = ["nonL1"]

        for rp in a_positions:
            ref_pos = read2ref.get(rp)
            if ref_pos is None:
                continue

            fivemer = get_ref_5mer(ref, chrom, ref_pos, is_reverse)
            if fivemer is None:
                continue

            # Count total A positions per 5-mer
            for cat in categories:
                fivemer_total[cat][fivemer] += 1

            # Check m6A call
            prob = m6a_map.get(rp, 0)
            if prob >= M6A_THRESHOLD:
                for cat in categories:
                    fivemer_m6a[cat][fivemer] += 1
            if prob > 0:
                for cat in categories:
                    fivemer_prob_sum[cat][fivemer] += prob

    bam.close()
    ref.close()

    print(f"\nProcessing complete:")
    print(f"  L1 reads used: {min(n_l1, MAX_L1_READS):,}")
    print(f"  non-L1 reads used: {n_nonl1_selected:,}")

    # ============================================================
    # Analysis: DRACH 5-mer methylation rates
    # ============================================================
    print(f"\n{'=' * 70}")
    print("DRACH 5-MER METHYLATION RATES")
    print(f"{'=' * 70}")

    # Build results table
    results = []
    for fivemer in ALL_DRACH_5MERS:
        row = {"fivemer": fivemer}
        for cat in ["L1", "nonL1", "young", "ancient"]:
            tot = fivemer_total[cat].get(fivemer, 0)
            m6a = fivemer_m6a[cat].get(fivemer, 0)
            rate = m6a / tot if tot > 0 else 0.0
            row[f"{cat}_total_A"] = tot
            row[f"{cat}_m6a_count"] = m6a
            row[f"{cat}_methyl_rate"] = rate
        # L1/nonL1 enrichment
        l1_rate = row["L1_methyl_rate"]
        nonl1_rate = row["nonL1_methyl_rate"]
        if nonl1_rate > 0:
            row["L1_vs_nonL1_ratio"] = l1_rate / nonl1_rate
        else:
            row["L1_vs_nonL1_ratio"] = float("inf") if l1_rate > 0 else 1.0
        # Fisher's exact test: L1 vs non-L1
        a = row["L1_m6a_count"]
        b = row["L1_total_A"] - a
        c = row["nonL1_m6a_count"]
        d = row["nonL1_total_A"] - c
        if a + b > 0 and c + d > 0:
            odds, pval = stats.fisher_exact([[a, b], [c, d]])
            row["fisher_OR"] = odds
            row["fisher_pval"] = pval
        else:
            row["fisher_OR"] = float("nan")
            row["fisher_pval"] = float("nan")
        results.append(row)

    # Print results sorted by L1 methylation rate
    results_sorted = sorted(results, key=lambda x: x["L1_methyl_rate"], reverse=True)

    print(f"\n{'5-mer':<8} {'L1_total':>9} {'L1_m6A':>8} {'L1_rate':>9} "
          f"{'nonL1_tot':>10} {'nonL1_m6A':>9} {'nonL1_rate':>10} "
          f"{'ratio':>7} {'OR':>7} {'P':>10}")
    print("-" * 100)
    for r in results_sorted:
        print(f"{r['fivemer']:<8} {r['L1_total_A']:>9,} {r['L1_m6a_count']:>8,} "
              f"{r['L1_methyl_rate']:>9.4f} "
              f"{r['nonL1_total_A']:>10,} {r['nonL1_m6a_count']:>9,} "
              f"{r['nonL1_methyl_rate']:>10.4f} "
              f"{r['L1_vs_nonL1_ratio']:>7.2f} "
              f"{r['fisher_OR']:>7.2f} {r['fisher_pval']:>10.2e}")

    # ============================================================
    # Young vs Ancient comparison
    # ============================================================
    print(f"\n{'=' * 70}")
    print("YOUNG vs ANCIENT L1 DRACH METHYLATION")
    print(f"{'=' * 70}")

    young_sorted = sorted(results, key=lambda x: x.get("young_methyl_rate", 0), reverse=True)

    print(f"\n{'5-mer':<8} {'Y_total':>9} {'Y_m6A':>8} {'Y_rate':>9} "
          f"{'A_total':>9} {'A_m6A':>8} {'A_rate':>9} {'Y/A':>7}")
    print("-" * 80)
    for r in young_sorted:
        y_rate = r.get("young_methyl_rate", 0)
        a_rate = r.get("ancient_methyl_rate", 0)
        ya_ratio = y_rate / a_rate if a_rate > 0 else float("inf") if y_rate > 0 else 1.0
        print(f"{r['fivemer']:<8} {r.get('young_total_A', 0):>9,} {r.get('young_m6a_count', 0):>8,} "
              f"{y_rate:>9.4f} "
              f"{r.get('ancient_total_A', 0):>9,} {r.get('ancient_m6a_count', 0):>8,} "
              f"{a_rate:>9.4f} {ya_ratio:>7.2f}")

    # ============================================================
    # Non-DRACH top 5-mers (for comparison)
    # ============================================================
    print(f"\n{'=' * 70}")
    print("TOP 20 NON-DRACH 5-MERS BY L1 METHYLATION RATE")
    print(f"{'=' * 70}")

    all_5mers = set()
    for cat in ["L1", "nonL1"]:
        all_5mers.update(fivemer_total[cat].keys())

    non_drach_results = []
    for fivemer in all_5mers:
        if bool(DRACH_RE.match(fivemer.upper())):
            continue
        l1_tot = fivemer_total["L1"].get(fivemer, 0)
        l1_m6a = fivemer_m6a["L1"].get(fivemer, 0)
        nl_tot = fivemer_total["nonL1"].get(fivemer, 0)
        nl_m6a = fivemer_m6a["nonL1"].get(fivemer, 0)
        if l1_tot < 100:  # require minimum counts for meaningful rate
            continue
        l1_rate = l1_m6a / l1_tot if l1_tot > 0 else 0
        nl_rate = nl_m6a / nl_tot if nl_tot > 0 else 0
        ratio = l1_rate / nl_rate if nl_rate > 0 else (float("inf") if l1_rate > 0 else 1.0)
        non_drach_results.append({
            "fivemer": fivemer,
            "L1_total": l1_tot, "L1_m6a": l1_m6a, "L1_rate": l1_rate,
            "nonL1_total": nl_tot, "nonL1_m6a": nl_m6a, "nonL1_rate": nl_rate,
            "ratio": ratio,
        })

    non_drach_sorted = sorted(non_drach_results, key=lambda x: x["L1_rate"], reverse=True)[:20]

    print(f"\n{'5-mer':<8} {'L1_total':>9} {'L1_m6A':>8} {'L1_rate':>9} "
          f"{'nonL1_tot':>10} {'nonL1_m6A':>9} {'nonL1_rate':>10} {'ratio':>7}")
    print("-" * 80)
    for r in non_drach_sorted:
        print(f"{r['fivemer']:<8} {r['L1_total']:>9,} {r['L1_m6a']:>8,} "
              f"{r['L1_rate']:>9.4f} "
              f"{r['nonL1_total']:>10,} {r['nonL1_m6a']:>9,} "
              f"{r['nonL1_rate']:>10.4f} {r['ratio']:>7.2f}")

    # ============================================================
    # Summary statistics
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Overall DRACH vs non-DRACH rates
    for cat in ["L1", "nonL1", "young", "ancient"]:
        drach_total = sum(fivemer_total[cat].get(m, 0) for m in ALL_DRACH_5MERS)
        drach_m6a = sum(fivemer_m6a[cat].get(m, 0) for m in ALL_DRACH_5MERS)
        nondrach_total = sum(fivemer_total[cat].get(m, 0) for m in all_5mers if not DRACH_RE.match(m.upper()))
        nondrach_m6a = sum(fivemer_m6a[cat].get(m, 0) for m in all_5mers if not DRACH_RE.match(m.upper()))
        dr_rate = drach_m6a / drach_total if drach_total > 0 else 0
        ndr_rate = nondrach_m6a / nondrach_total if nondrach_total > 0 else 0
        fold = dr_rate / ndr_rate if ndr_rate > 0 else float("inf")
        total_a = drach_total + nondrach_total
        drach_frac = drach_total / total_a if total_a > 0 else 0
        print(f"  {cat:>8}: DRACH A-sites = {drach_total:>10,} ({drach_frac:.1%}), "
              f"rate = {dr_rate:.4f} | non-DRACH rate = {ndr_rate:.4f} | fold = {fold:.2f}x")

    # Most enriched DRACH motif in L1 vs non-L1
    print(f"\n  Most L1-enriched DRACH 5-mer: {results_sorted[0]['fivemer']} "
          f"(L1 rate = {results_sorted[0]['L1_methyl_rate']:.4f}, "
          f"L1/nonL1 = {results_sorted[0]['L1_vs_nonL1_ratio']:.2f}x)")

    # Mean and median of L1 DRACH rates
    l1_rates = [r["L1_methyl_rate"] for r in results]
    nl_rates = [r["nonL1_methyl_rate"] for r in results]
    print(f"\n  Mean DRACH 5-mer rate: L1 = {np.mean(l1_rates):.4f}, non-L1 = {np.mean(nl_rates):.4f}")
    print(f"  Median DRACH 5-mer rate: L1 = {np.median(l1_rates):.4f}, non-L1 = {np.median(nl_rates):.4f}")

    # Correlation between L1 and non-L1 rates across 16 DRACH motifs
    if len(l1_rates) >= 5 and len(nl_rates) >= 5:
        rho, pval = stats.spearmanr(l1_rates, nl_rates)
        print(f"  Spearman correlation of per-5mer rates (L1 vs non-L1): rho = {rho:.3f}, P = {pval:.3e}")

    # ============================================================
    # Reproducible sites motif distribution
    # ============================================================
    repro_path = os.path.join(OUTDIR, "reproducible_sites.tsv")
    if os.path.exists(repro_path):
        print(f"\n{'=' * 70}")
        print("REPRODUCIBLE SITES: DRACH 5-MER DISTRIBUTION")
        print(f"{'=' * 70}")
        import pandas as pd
        repro = pd.read_csv(repro_path, sep="\t")
        print(f"  Total reproducible sites: {len(repro):,}")
        if "context_11mer" in repro.columns:
            repro["fivemer"] = repro["context_11mer"].str[3:8]
            repro["is_drach"] = repro["fivemer"].apply(lambda x: bool(DRACH_RE.match(x.upper())) if isinstance(x, str) and len(x) == 5 else False)
            n_drach = repro["is_drach"].sum()
            print(f"  DRACH sites: {n_drach:,} ({n_drach / len(repro):.1%})")

            # Per DRACH 5-mer in reproducible sites
            drach_repro = repro[repro["is_drach"]]
            if len(drach_repro) > 0:
                repro_motif_counts = drach_repro["fivemer"].value_counts()
                print(f"\n  DRACH 5-mer distribution in reproducible sites:")
                for motif, cnt in repro_motif_counts.items():
                    # Compare to overall rate from our analysis
                    l1_tot = fivemer_total["L1"].get(motif, 0)
                    rate_from_analysis = fivemer_m6a["L1"].get(motif, 0) / l1_tot if l1_tot > 0 else 0
                    print(f"    {motif}: {cnt:>5} reproducible sites "
                          f"(L1 m6A rate = {rate_from_analysis:.4f})")

    # ============================================================
    # Save output: drach_motif_rates.tsv
    # ============================================================
    out_path = os.path.join(OUTDIR, "drach_motif_rates.tsv")
    print(f"\n{'=' * 70}")
    print(f"SAVING: {out_path}")
    print(f"{'=' * 70}")

    with open(out_path, "w") as f:
        header = [
            "fivemer",
            "L1_total_A", "L1_m6a_count", "L1_methyl_rate",
            "nonL1_total_A", "nonL1_m6a_count", "nonL1_methyl_rate",
            "L1_vs_nonL1_ratio", "fisher_OR", "fisher_pval",
            "young_total_A", "young_m6a_count", "young_methyl_rate",
            "ancient_total_A", "ancient_m6a_count", "ancient_methyl_rate",
        ]
        f.write("\t".join(header) + "\n")
        for r in results_sorted:
            vals = [
                r["fivemer"],
                str(r["L1_total_A"]), str(r["L1_m6a_count"]), f"{r['L1_methyl_rate']:.6f}",
                str(r["nonL1_total_A"]), str(r["nonL1_m6a_count"]), f"{r['nonL1_methyl_rate']:.6f}",
                f"{r['L1_vs_nonL1_ratio']:.4f}",
                f"{r['fisher_OR']:.4f}" if not np.isnan(r['fisher_OR']) else "NA",
                f"{r['fisher_pval']:.2e}" if not np.isnan(r['fisher_pval']) else "NA",
                str(r.get("young_total_A", 0)), str(r.get("young_m6a_count", 0)),
                f"{r.get('young_methyl_rate', 0):.6f}",
                str(r.get("ancient_total_A", 0)), str(r.get("ancient_m6a_count", 0)),
                f"{r.get('ancient_methyl_rate', 0):.6f}",
            ]
            f.write("\t".join(vals) + "\n")

    # Also save non-DRACH top motifs
    nondrach_out = os.path.join(OUTDIR, "nondrach_motif_rates_top50.tsv")
    non_drach_top50 = sorted(non_drach_results, key=lambda x: x["L1_rate"], reverse=True)[:50]
    with open(nondrach_out, "w") as f:
        f.write("fivemer\tL1_total_A\tL1_m6a_count\tL1_methyl_rate\t"
                "nonL1_total_A\tnonL1_m6a_count\tnonL1_methyl_rate\tL1_vs_nonL1_ratio\n")
        for r in non_drach_top50:
            f.write(f"{r['fivemer']}\t{r['L1_total']}\t{r['L1_m6a']}\t{r['L1_rate']:.6f}\t"
                    f"{r['nonL1_total']}\t{r['nonL1_m6a']}\t{r['nonL1_rate']:.6f}\t"
                    f"{r['ratio']:.4f}\n")

    print(f"  Saved: {out_path}")
    print(f"  Saved: {nondrach_out}")
    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
