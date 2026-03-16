#!/usr/bin/env python3
"""
ENCODE PUS1/DKC1 KD RNA-seq → L1 expression analysis.

Uses bigWig signal tracks to quantify RNA-seq signal over L1 regions,
comparing KD vs control.

bigWig files: /scratch1/junsoopablo/pus_encode_bigwig/
L1 BED: reference/L1_TE_L1_family.bed

Approach:
  - For each L1 region, compute total signal from bigWig (sum of coverage)
  - L1 regions are unstranded in our BED, so sum plus+minus strand signal
  - Aggregate by L1 subfamily
  - Compare KD vs control (fold change + Mann-Whitney test)
"""
import os
import sys
import numpy as np
from collections import defaultdict

# ── Paths ──
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
L1_BED = f"{BASE}/reference/L1_TE_L1_family.bed"
BW_DIR = "/scratch1/junsoopablo/pus_encode_bigwig"
OUT_DIR = f"{BASE}/analysis/01_exploration/topic_05_cellline/pus_enzyme_analysis"

YOUNG_L1 = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

# ── Sample definitions ──
# Format: {condition: {rep: {strand: accession}}}
SAMPLES = {
    "DKC1_KD_K562": {
        1: {"plus": "ENCFF725BRJ", "minus": "ENCFF051ILL"},
        2: {"plus": "ENCFF883RBN", "minus": "ENCFF949GMZ"},
    },
    "DKC1_KD_HepG2": {
        1: {"plus": "ENCFF532VXO", "minus": "ENCFF232WWV"},
        2: {"plus": "ENCFF316TWR", "minus": "ENCFF536YNR"},
    },
    "PUS1_KD_K562": {
        1: {"plus": "ENCFF917EVO", "minus": "ENCFF468MNF"},
        2: {"plus": "ENCFF652BZV", "minus": "ENCFF870PAI"},
    },
    "PUS1_KD_HepG2": {
        1: {"plus": "ENCFF953STT", "minus": "ENCFF917SFP"},
        2: {"plus": "ENCFF899IVX", "minus": "ENCFF703EZT"},
    },
    "Ctrl_K562": {
        1: {"plus": "ENCFF138YKP", "minus": "ENCFF527WYJ"},
        2: {"plus": "ENCFF828TWZ", "minus": "ENCFF156YYO"},
    },
    "Ctrl_HepG2": {
        1: {"plus": "ENCFF850AXV", "minus": "ENCFF609EDC"},
        2: {"plus": "ENCFF408SQP", "minus": "ENCFF805QXY"},
    },
}

# KD-Control pairings
COMPARISONS = [
    ("DKC1_KD_K562", "Ctrl_K562"),
    ("DKC1_KD_HepG2", "Ctrl_HepG2"),
    ("PUS1_KD_K562", "Ctrl_K562"),
    ("PUS1_KD_HepG2", "Ctrl_HepG2"),
]


def find_bigwig(condition, rep, strand):
    """Find bigWig file path."""
    acc = SAMPLES[condition][rep][strand]
    # Find matching file
    for fn in os.listdir(BW_DIR):
        if acc in fn and fn.endswith('.bigWig'):
            return os.path.join(BW_DIR, fn)
    raise FileNotFoundError(f"bigWig not found: {condition} rep{rep} {strand} ({acc})")


def load_l1_regions():
    """Load L1 BED regions, grouped by subfamily."""
    regions = []
    with open(L1_BED) as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            subfamily = parts[3]
            regions.append((chrom, start, end, subfamily))
    return regions


def compute_signal_over_regions(bw_plus_path, bw_minus_path, regions, max_regions=None):
    """Compute total signal over L1 regions from bigWig files.

    Uses chromosome-grouped batch approach for speed.
    Returns dict: subfamily → list of signal values (signal per kb)
    """
    import pyBigWig

    bw_plus = pyBigWig.open(bw_plus_path)
    bw_minus = pyBigWig.open(bw_minus_path)

    plus_chroms = bw_plus.chroms()
    minus_chroms = bw_minus.chroms()

    # Group regions by chromosome for batch processing
    chrom_regions = defaultdict(list)
    for i, (chrom, start, end, subfamily) in enumerate(regions):
        if max_regions and i >= max_regions:
            break
        if chrom in plus_chroms and chrom in minus_chroms:
            chrom_regions[chrom].append((start, end, subfamily))

    subfamily_signals = defaultdict(list)
    processed = 0

    for chrom, regs in chrom_regions.items():
        chrom_len = min(plus_chroms[chrom], minus_chroms[chrom])

        for start, end, subfamily in regs:
            region_len = end - start
            if region_len <= 0 or start >= chrom_len:
                continue
            end_c = min(end, chrom_len)

            try:
                ps = bw_plus.stats(chrom, start, end_c, type="sum")
                ms = bw_minus.stats(chrom, start, end_c, type="sum")
                total_sig = (ps[0] or 0) + abs(ms[0] or 0)
                sig_per_kb = total_sig / (region_len / 1000)
                subfamily_signals[subfamily].append(sig_per_kb)
            except Exception:
                continue

            processed += 1

    bw_plus.close()
    bw_minus.close()

    return subfamily_signals


def aggregate_by_category(subfamily_signals):
    """Aggregate signals into categories: all_L1, young_L1, ancient_L1, and per-subfamily."""
    result = {}

    all_vals = []
    young_vals = []
    ancient_vals = []

    for subfam, vals in subfamily_signals.items():
        all_vals.extend(vals)
        if subfam in YOUNG_L1:
            young_vals.extend(vals)
        else:
            ancient_vals.extend(vals)
        if len(vals) >= 10:
            result[subfam] = np.array(vals)

    result['ALL_L1'] = np.array(all_vals) if all_vals else np.array([0])
    result['YOUNG_L1'] = np.array(young_vals) if young_vals else np.array([0])
    result['ANCIENT_L1'] = np.array(ancient_vals) if ancient_vals else np.array([0])

    return result


def main():
    print("=" * 70)
    print("ENCODE PUS1/DKC1 KD → L1 Expression (bigWig signal)")
    print("=" * 70)

    # Verify bigWig files exist
    print("\n[1] Checking bigWig files...")
    missing = []
    for cond in SAMPLES:
        for rep in SAMPLES[cond]:
            for strand in ['plus', 'minus']:
                try:
                    path = find_bigwig(cond, rep, strand)
                    size_mb = os.path.getsize(path) / 1e6
                    if size_mb < 1:
                        missing.append(f"{cond} rep{rep} {strand} (too small: {size_mb:.1f}MB)")
                except FileNotFoundError as e:
                    missing.append(str(e))

    if missing:
        print("  Missing/bad files:")
        for m in missing:
            print(f"    {m}")
        print("  Aborting. Please download bigWig files first.")
        return
    print("  All bigWig files present.")

    # Load L1 regions
    print("\n[2] Loading L1 regions...")
    regions = load_l1_regions()
    print(f"  {len(regions):,} L1 regions")

    # Sample 10k regions for speed (full L1 BED has 1M entries)
    sample_step = max(1, len(regions) // 10000)
    sampled_regions = regions[::sample_step]
    print(f"  Sampling every {sample_step}th region → {len(sampled_regions):,} regions")

    # Compute signal for each condition
    print("\n[3] Computing signal over L1 regions...")
    condition_signals = {}

    for cond in SAMPLES:
        print(f"\n  {cond}:")
        rep_signals = {}
        for rep in SAMPLES[cond]:
            sys.stdout.write(f"    Rep {rep}...")
            sys.stdout.flush()
            bw_plus = find_bigwig(cond, rep, 'plus')
            bw_minus = find_bigwig(cond, rep, 'minus')
            signals = compute_signal_over_regions(bw_plus, bw_minus, sampled_regions)
            agg = aggregate_by_category(signals)
            rep_signals[rep] = agg
            print(f" done (ALL_L1 mean={np.mean(agg['ALL_L1']):.2f})")
            sys.stdout.flush()

        # Average across replicates for each category
        all_categories = set()
        for rep_agg in rep_signals.values():
            all_categories.update(rep_agg.keys())

        condition_signals[cond] = {}
        for cat in all_categories:
            # Concatenate across replicates
            vals = []
            for rep_agg in rep_signals.values():
                if cat in rep_agg:
                    vals.extend(rep_agg[cat].tolist())
            condition_signals[cond][cat] = np.array(vals)

    # Compute library size normalization factors
    # Use median signal across ALL regions as proxy for total library size
    print("\n[4] Library size normalization...")
    lib_size = {}
    for cond in condition_signals:
        median_sig = np.median(condition_signals[cond]['ALL_L1'])
        mean_sig = np.mean(condition_signals[cond]['ALL_L1'])
        lib_size[cond] = mean_sig
        print(f"  {cond}: mean={mean_sig:.2f}, median={median_sig:.2f}")

    # Normalize to reference (average of all controls)
    ctrl_ref = np.mean([lib_size[c] for c in lib_size if c.startswith('Ctrl')])
    print(f"  Reference (avg ctrl): {ctrl_ref:.2f}")
    norm_factors = {c: ctrl_ref / lib_size[c] if lib_size[c] > 0 else 1.0 for c in lib_size}
    for cond, nf in norm_factors.items():
        print(f"  {cond}: norm_factor={nf:.3f}")

    # Compare KD vs Control
    print("\n" + "=" * 70)
    print("KD vs Control Comparison (library-size normalized)")
    print("=" * 70)

    from scipy.stats import mannwhitneyu

    results = []

    for kd_cond, ctrl_cond in COMPARISONS:
        print(f"\n{kd_cond} vs {ctrl_cond}:")
        print(f"  {'Category':<15} {'KD_mean':>10} {'Ctrl_mean':>10} {'FC':>8} {'p-value':>12}")
        print(f"  {'-'*55}")

        for cat in ['ALL_L1', 'YOUNG_L1', 'ANCIENT_L1']:
            kd_vals = condition_signals[kd_cond].get(cat, np.array([0])) * norm_factors[kd_cond]
            ctrl_vals = condition_signals[ctrl_cond].get(cat, np.array([0])) * norm_factors[ctrl_cond]

            kd_mean = np.mean(kd_vals)
            ctrl_mean = np.mean(ctrl_vals)
            fc = kd_mean / ctrl_mean if ctrl_mean > 0 else float('inf')

            if len(kd_vals) > 1 and len(ctrl_vals) > 1:
                stat, pval = mannwhitneyu(kd_vals, ctrl_vals, alternative='two-sided')
            else:
                pval = 1.0

            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            print(f"  {cat:<15} {kd_mean:>10.2f} {ctrl_mean:>10.2f} {fc:>7.3f}x {pval:>10.2e} {sig}")

            results.append({
                'comparison': f"{kd_cond}_vs_{ctrl_cond}",
                'category': cat,
                'kd_mean': kd_mean,
                'ctrl_mean': ctrl_mean,
                'fold_change': fc,
                'p_value': pval,
            })

        # Also show top changing subfamilies
        print(f"\n  Top subfamilies by fold change:")
        subfam_fcs = []
        for cat in condition_signals[kd_cond]:
            if cat in ['ALL_L1', 'YOUNG_L1', 'ANCIENT_L1']:
                continue
            kd_vals = condition_signals[kd_cond].get(cat, np.array([0]))
            ctrl_vals = condition_signals[ctrl_cond].get(cat, np.array([0]))
            if len(kd_vals) >= 5 and len(ctrl_vals) >= 5:
                kd_m = np.mean(kd_vals)
                ctrl_m = np.mean(ctrl_vals)
                if ctrl_m > 0:
                    fc = kd_m / ctrl_m
                    stat, pval = mannwhitneyu(kd_vals, ctrl_vals, alternative='two-sided')
                    subfam_fcs.append((cat, fc, pval, len(kd_vals)))

        subfam_fcs.sort(key=lambda x: x[1], reverse=True)
        for subfam, fc, pval, n in subfam_fcs[:5]:
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            print(f"    {subfam:<12} FC={fc:.3f}x  p={pval:.2e} {sig}  (n={n})")
        print(f"  ...")
        for subfam, fc, pval, n in subfam_fcs[-3:]:
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            print(f"    {subfam:<12} FC={fc:.3f}x  p={pval:.2e} {sig}  (n={n})")

    # Save results
    out_tsv = f"{OUT_DIR}/encode_kd_l1_expression.tsv"
    with open(out_tsv, 'w') as f:
        f.write("comparison\tcategory\tkd_mean_signal_per_kb\tctrl_mean_signal_per_kb\t"
                "fold_change\tp_value\n")
        for r in results:
            f.write(f"{r['comparison']}\t{r['category']}\t{r['kd_mean']:.4f}\t"
                    f"{r['ctrl_mean']:.4f}\t{r['fold_change']:.4f}\t{r['p_value']:.2e}\n")
    print(f"\n  Results saved: {out_tsv}")


if __name__ == "__main__":
    main()
