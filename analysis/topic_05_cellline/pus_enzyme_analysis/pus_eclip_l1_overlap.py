#!/usr/bin/env python3
"""
PUS enzyme eCLIP peaks ∩ L1 overlap analysis.
Tests whether PUS1/DKC1 physically bind L1 RNA regions.

eCLIP experiments:
  - DKC1 HepG2: ENCSR301TFY (IDR: ENCFF633LQC)
  - PUS1 K562:  ENCSR291XPT (IDR: ENCFF247NXL)

Background (non-PUS) RBPs:
  - HNRNPC HepG2: ENCFF440ROZ
  - PTBP1 K562:   ENCFF907HNN
  - RBFOX2 HepG2: ENCFF871NYM
"""
import subprocess
import os
import sys
import gzip
import tempfile
from collections import Counter

# ── Paths ──
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
L1_BED = f"{BASE}/reference/L1_TE_L1_family.bed"
OUT_DIR = f"{BASE}/analysis/01_exploration/topic_05_cellline/pus_enzyme_analysis"
ECLIP_DIR = f"{OUT_DIR}/eclip"
BEDTOOLS = "conda run -n research bedtools"

# ── eCLIP datasets ──
ECLIP_DATA = {
    "DKC1_HepG2": {
        "url": "https://www.encodeproject.org/files/ENCFF633LQC/@@download/ENCFF633LQC.bed.gz",
        "is_pus": True,
    },
    "PUS1_K562": {
        "url": "https://www.encodeproject.org/files/ENCFF247NXL/@@download/ENCFF247NXL.bed.gz",
        "is_pus": True,
    },
    "HNRNPC_HepG2": {
        "url": "https://www.encodeproject.org/files/ENCFF440ROZ/@@download/ENCFF440ROZ.bed.gz",
        "is_pus": False,
    },
    "PTBP1_K562": {
        "url": "https://www.encodeproject.org/files/ENCFF907HNN/@@download/ENCFF907HNN.bed.gz",
        "is_pus": False,
    },
    "RBFOX2_HepG2": {
        "url": "https://www.encodeproject.org/files/ENCFF871NYM/@@download/ENCFF871NYM.bed.gz",
        "is_pus": False,
    },
}

# Young L1 subfamilies
YOUNG_L1 = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

# Genome size (hg38, excluding alt contigs)
GENOME_SIZE = 3_088_286_401


def run(cmd, capture=True):
    """Run shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if result.returncode != 0 and capture:
        print(f"  WARN: {cmd}\n  stderr: {result.stderr[:200]}", file=sys.stderr)
    return result.stdout if capture else None


def download_eclip():
    """Download all eCLIP narrowPeak files."""
    os.makedirs(ECLIP_DIR, exist_ok=True)
    for name, info in ECLIP_DATA.items():
        out_gz = f"{ECLIP_DIR}/{name}.narrowPeak.gz"
        out_bed = f"{ECLIP_DIR}/{name}.narrowPeak.bed"
        if os.path.exists(out_bed) and os.path.getsize(out_bed) > 0:
            print(f"  {name}: already downloaded ({out_bed})")
            continue
        print(f"  Downloading {name}...")
        run(f"curl -s -L '{info['url']}' -o '{out_gz}'", capture=False)
        # Decompress
        with gzip.open(out_gz, 'rt') as fin, open(out_bed, 'w') as fout:
            for line in fin:
                fout.write(line)
        print(f"    → {sum(1 for _ in open(out_bed))} peaks")


def get_l1_genome_bp():
    """Calculate total L1 base pairs in genome."""
    total_bp = 0
    with open(L1_BED) as f:
        for line in f:
            parts = line.strip().split('\t')
            total_bp += int(parts[2]) - int(parts[1])
    return total_bp


def intersect_with_l1(peak_bed):
    """Intersect peaks with L1, return overlapping peaks with L1 info."""
    # -wa -wb: write both A and B entries
    cmd = f"{BEDTOOLS} intersect -a {peak_bed} -b {L1_BED} -wa -wb"
    result = run(cmd)
    overlaps = []
    for line in result.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        # narrowPeak has 10 columns, L1 BED has 5 columns
        n_peak_cols = 10
        l1_subfamily = parts[n_peak_cols + 3]  # col 4 of L1 BED (subfamily)
        overlaps.append({
            'peak_chr': parts[0],
            'peak_start': int(parts[1]),
            'peak_end': int(parts[2]),
            'l1_subfamily': l1_subfamily,
        })
    return overlaps


def fisher_test(peaks_in_l1, total_peaks, l1_bp, genome_bp):
    """Fisher exact test for enrichment."""
    from scipy.stats import fisher_exact
    # 2x2 contingency: peaks_in_L1, peaks_not_in_L1, L1_bp, non_L1_bp
    # But peaks are count-based, genome is bp-based
    # Use proportional approach
    peaks_not_in_l1 = total_peaks - peaks_in_l1
    non_l1_bp = genome_bp - l1_bp
    table = [[peaks_in_l1, peaks_not_in_l1],
             [l1_bp, non_l1_bp]]
    odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
    return odds_ratio, p_value


def analyze_eclip():
    """Main eCLIP analysis."""
    print("=" * 60)
    print("PUS eCLIP ∩ L1 Overlap Analysis")
    print("=" * 60)

    # Download data
    print("\n[1] Downloading eCLIP peaks...")
    download_eclip()

    # Get L1 genome coverage
    print("\n[2] Calculating L1 genome coverage...")
    l1_bp = get_l1_genome_bp()
    l1_frac = l1_bp / GENOME_SIZE
    print(f"  L1 total bp: {l1_bp:,} ({l1_frac*100:.2f}% of genome)")

    # Analyze each dataset
    print("\n[3] Intersecting peaks with L1...")
    results = []
    for name, info in ECLIP_DATA.items():
        peak_bed = f"{ECLIP_DIR}/{name}.narrowPeak.bed"
        total_peaks = sum(1 for _ in open(peak_bed))

        overlaps = intersect_with_l1(peak_bed)
        # Unique peaks in L1 (a peak might overlap multiple L1 elements)
        unique_peaks = set()
        subfam_counts = Counter()
        for ov in overlaps:
            pk = (ov['peak_chr'], ov['peak_start'], ov['peak_end'])
            unique_peaks.add(pk)
            subfam_counts[ov['l1_subfamily']] += 1

        peaks_in_l1 = len(unique_peaks)
        pct = peaks_in_l1 / total_peaks * 100 if total_peaks > 0 else 0
        enrichment = (peaks_in_l1 / total_peaks) / l1_frac if total_peaks > 0 else 0

        # Fisher test
        or_val, p_val = fisher_test(peaks_in_l1, total_peaks, l1_bp, GENOME_SIZE)

        # Young vs ancient
        young_count = sum(v for k, v in subfam_counts.items() if k in YOUNG_L1)
        ancient_count = sum(v for k, v in subfam_counts.items() if k not in YOUNG_L1)

        # Top subfamilies
        top_subfam = subfam_counts.most_common(10)

        print(f"\n  {name} ({'PUS' if info['is_pus'] else 'non-PUS'}):")
        print(f"    Total peaks: {total_peaks:,}")
        print(f"    Peaks in L1: {peaks_in_l1:,} ({pct:.1f}%)")
        print(f"    Enrichment:  {enrichment:.2f}x (vs {l1_frac*100:.2f}% genome)")
        print(f"    Fisher OR:   {or_val:.3f}, p={p_val:.2e}")
        print(f"    Young L1 overlaps: {young_count}, Ancient: {ancient_count}")
        print(f"    Top subfamilies: {', '.join(f'{k}({v})' for k, v in top_subfam[:5])}")

        results.append({
            'name': name,
            'is_pus': info['is_pus'],
            'total_peaks': total_peaks,
            'peaks_in_l1': peaks_in_l1,
            'pct_in_l1': pct,
            'enrichment': enrichment,
            'fisher_or': or_val,
            'fisher_p': p_val,
            'young_overlaps': young_count,
            'ancient_overlaps': ancient_count,
            'top_subfamilies': top_subfam,
        })

    # Write results table
    out_tsv = f"{OUT_DIR}/eclip_l1_overlap.tsv"
    with open(out_tsv, 'w') as f:
        f.write("RBP\tis_PUS\ttotal_peaks\tpeaks_in_L1\tpct_in_L1\tenrichment_vs_genome\t"
                "fisher_OR\tfisher_p\tyoung_L1_overlaps\tancient_L1_overlaps\ttop_subfamilies\n")
        for r in results:
            top_str = "; ".join(f"{k}({v})" for k, v in r['top_subfamilies'][:5])
            f.write(f"{r['name']}\t{r['is_pus']}\t{r['total_peaks']}\t{r['peaks_in_l1']}\t"
                    f"{r['pct_in_l1']:.2f}\t{r['enrichment']:.3f}\t"
                    f"{r['fisher_or']:.4f}\t{r['fisher_p']:.2e}\t"
                    f"{r['young_overlaps']}\t{r['ancient_overlaps']}\t{top_str}\n")
    print(f"\n  Results saved: {out_tsv}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary: PUS vs non-PUS enrichment in L1")
    print("=" * 60)
    pus_enrich = [r['enrichment'] for r in results if r['is_pus']]
    non_pus_enrich = [r['enrichment'] for r in results if not r['is_pus']]
    print(f"  PUS enzymes:   mean enrichment = {sum(pus_enrich)/len(pus_enrich):.2f}x")
    print(f"  Non-PUS RBPs:  mean enrichment = {sum(non_pus_enrich)/len(non_pus_enrich):.2f}x")


if __name__ == "__main__":
    analyze_eclip()
