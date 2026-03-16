#!/usr/bin/env python3
"""METTL3 KO: L1-body m6A vs flanking m6A at thr=204.

For each L1 read:
  1. Get read alignment coordinates from MAFIA BAM
  2. Parse MM/ML tags to get m6A site positions (query-relative)
  3. Map query positions to reference positions via CIGAR
  4. Intersect with L1 RepeatMasker annotation to classify body vs flanking
  5. Compute body-m6A/kb and flanking-m6A/kb separately
  6. Compare WT vs KO for each compartment
"""

import pysam
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Paths
# =============================================================================
VAULT = Path('/vault/external-datasets/2026/PRJEB40872_HEK293T_METTL3KO_xPore')
MAFIA_DIR = VAULT / 'mafia_guppy'
L1_FILTER = VAULT / 'l1_filter_guppy'
RMSK_GTF = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/hg38_rmsk_TE.gtf')

OUTDIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/mettl3ko_body_flanking')
OUTDIR.mkdir(exist_ok=True)

PROB_THRESHOLD = 204  # ML >= 204 (80%)

SAMPLES = {
    'WT_rep1': 'WT', 'WT_rep2': 'WT', 'WT_rep3': 'WT',
    'KO_rep1': 'KO', 'KO_rep2': 'KO', 'KO_rep3': 'KO',
}

# =============================================================================
# Load RepeatMasker L1 annotations
# =============================================================================
def load_l1_annotations():
    """Load L1 annotations from RepeatMasker GTF into an interval lookup dict.
    GTF format: chr  hg38_rmsk  exon  start  end  score  strand  .  attributes
    Filters for family_id "L1".
    """
    if not RMSK_GTF.exists():
        raise FileNotFoundError(f"GTF not found: {RMSK_GTF}")

    records = []
    with open(RMSK_GTF) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 9:
                continue
            # Filter for L1 family
            attr = cols[8]
            if 'family_id "L1"' not in attr:
                continue
            chrom = cols[0]
            start = int(cols[3]) - 1  # GTF is 1-based → 0-based
            end = int(cols[4])        # GTF end is inclusive → exclusive
            records.append((chrom, start, end))

    # Build per-chromosome interval list (sorted by start)
    annot = {}
    for chrom, start, end in records:
        annot.setdefault(chrom, []).append((start, end))
    for chrom in annot:
        annot[chrom].sort()
    return annot


def point_in_l1(annot, chrom, pos):
    """Check if a single reference position falls within any L1 annotation.
    Uses binary search for efficiency.
    """
    if chrom not in annot:
        return False
    intervals = annot[chrom]
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if intervals[mid][0] <= pos:
            lo = mid + 1
        else:
            hi = mid - 1
    if hi >= 0 and intervals[hi][0] <= pos < intervals[hi][1]:
        return True
    return False


def compute_l1_overlap_bases(annot, chrom, ref_start, ref_end):
    """Compute how many bases in [ref_start, ref_end) overlap L1 annotations.
    Much faster than checking every base individually.
    """
    if chrom not in annot:
        return 0
    intervals = annot[chrom]
    # Binary search for first interval that could overlap
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if intervals[mid][1] <= ref_start:
            lo = mid + 1
        else:
            hi = mid - 1
    # Walk forward from lo, accumulating overlap
    total = 0
    for i in range(lo, len(intervals)):
        s, e = intervals[i]
        if s >= ref_end:
            break
        overlap_start = max(s, ref_start)
        overlap_end = min(e, ref_end)
        if overlap_start < overlap_end:
            total += overlap_end - overlap_start
    return total


# =============================================================================
# Parse MAFIA BAM — extract m6A positions with reference coordinates
# =============================================================================
def query_to_ref_positions(read):
    """Build query_pos -> ref_pos mapping from CIGAR."""
    mapping = {}
    ref_pos = read.reference_start
    query_pos = 0
    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X: consumes both query and ref
            for i in range(length):
                mapping[query_pos] = ref_pos
                query_pos += 1
                ref_pos += 1
        elif op == 1:  # I: consumes query only
            for i in range(length):
                mapping[query_pos] = None  # insertion, no ref position
                query_pos += 1
        elif op in (2, 3):  # D, N: consumes ref only
            ref_pos += length
        elif op == 4:  # S: soft clip, consumes query only
            for i in range(length):
                mapping[query_pos] = None
                query_pos += 1
        elif op == 5:  # H: hard clip, consumes neither
            pass
    return mapping


def parse_mm_tag_positions(mm_tag, ml_tag, seq_len):
    """Parse MM/ML tags to get m6A site query positions + ML values.
    Returns list of (query_pos, ml_value) for m6A sites.
    """
    if mm_tag is None or ml_tag is None:
        return []

    ml_list = list(ml_tag)
    entries = mm_tag.rstrip(';').split(';')
    ml_idx = 0
    m6a_sites = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(',')
        base_mod = parts[0]
        skip_counts = parts[1:]

        is_m6a = '21891' in base_mod or 'A+a' in base_mod or 'A+m' in base_mod

        # Determine base and strand from base_mod
        # Format: "C+m" or "N+21891" etc.
        if not is_m6a:
            ml_idx += len(skip_counts)
            continue

        # Walk through the sequence to find modified positions
        # The MM tag gives delta positions between modified bases of the target type
        # For m6A: target base is A (or all bases if N+21891)
        query_pos = -1  # will be incremented
        for skip_str in skip_counts:
            if not skip_str:
                continue
            skip = int(skip_str)
            # Skip 'skip' candidate bases, then the next one is modified
            # For simplicity with N+21891, every position is a candidate
            query_pos += skip + 1
            if ml_idx < len(ml_list):
                m6a_sites.append((query_pos, ml_list[ml_idx]))
                ml_idx += 1
            else:
                ml_idx += 1

    return m6a_sites


def analyze_sample(sample, condition, l1_read_ids, annot):
    """Analyze one sample: classify m6A sites as body vs flanking."""
    bam_path = str(MAFIA_DIR / sample / 'mAFiA.reads.bam')
    if not Path(bam_path).exists():
        print(f"  WARNING: {bam_path} not found")
        return []

    records = []
    bam = pysam.AlignmentFile(bam_path, 'rb')

    found = 0
    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.query_name not in l1_read_ids:
            continue

        found += 1
        chrom = read.reference_name
        rlen = read.query_alignment_length or read.infer_query_length() or 0
        if rlen < 100:
            continue

        # Get m6A sites from MM/ML
        mm_tag = ml_tag = None
        for t in ['MM', 'Mm']:
            if read.has_tag(t):
                mm_tag = read.get_tag(t); break
        for t in ['ML', 'Ml']:
            if read.has_tag(t):
                ml_tag = read.get_tag(t); break

        m6a_sites = parse_mm_tag_positions(mm_tag, ml_tag,
                                           read.query_length or rlen)

        # Build query->ref mapping
        if read.cigartuples is None:
            continue
        q2r = query_to_ref_positions(read)

        # Classify each m6A site
        body_sites_high = 0
        flank_sites_high = 0
        body_sites_all = 0
        flank_sites_all = 0

        # Compute L1 body length on this read using efficient interval overlap
        ref_start = read.reference_start
        ref_end = read.reference_end
        # Compute aligned reference length from CIGAR (M, D, N, =, X consume reference)
        aligned_ref_len = sum(
            op_len for op, op_len in read.cigartuples
            if op in (0, 2, 3, 7, 8)
        )
        body_bases = compute_l1_overlap_bases(annot, chrom, ref_start, ref_end)
        flank_bases = aligned_ref_len - body_bases

        for qpos, ml_val in m6a_sites:
            rpos = q2r.get(qpos)
            if rpos is None:
                continue  # insertion position
            is_body = point_in_l1(annot, chrom, rpos)

            if is_body:
                body_sites_all += 1
                if ml_val >= PROB_THRESHOLD:
                    body_sites_high += 1
            else:
                flank_sites_all += 1
                if ml_val >= PROB_THRESHOLD:
                    flank_sites_high += 1

        body_kb = body_bases / 1000.0 if body_bases > 0 else 0
        flank_kb = flank_bases / 1000.0 if flank_bases > 0 else 0
        total_kb = rlen / 1000.0

        records.append({
            'read_id': read.query_name,
            'sample': sample,
            'condition': condition,
            'read_length': rlen,
            'body_bases': body_bases,
            'flank_bases': flank_bases,
            'overlap_fraction': body_bases / (body_bases + flank_bases) if (body_bases + flank_bases) > 0 else 0,
            'body_m6a_high': body_sites_high,
            'flank_m6a_high': flank_sites_high,
            'total_m6a_high': body_sites_high + flank_sites_high,
            'body_m6a_per_kb': body_sites_high / body_kb if body_kb > 0 else 0,
            'flank_m6a_per_kb': flank_sites_high / flank_kb if flank_kb > 0 else 0,
            'total_m6a_per_kb': (body_sites_high + flank_sites_high) / total_kb if total_kb > 0 else 0,
        })

    bam.close()
    print(f"  {sample}: {found} L1 reads found, {len(records)} analyzed")
    return records


# =============================================================================
# Main
# =============================================================================
print("=" * 70)
print("METTL3 KO: L1-body m6A vs flanking m6A (thr=204)")
print("=" * 70)

# Load L1 annotations
print("\nLoading L1 annotations...")
annot = load_l1_annotations()
n_chroms = len(annot)
n_intervals = sum(len(v) for v in annot.values())
print(f"  {n_chroms} chromosomes, {n_intervals:,} L1 intervals")

# Process each sample
all_records = []
for sample, condition in SAMPLES.items():
    print(f"\nProcessing {sample} ({condition})...")

    # Load L1 read IDs
    rid_file = L1_FILTER / sample / 'b_l1_te_filter' / f'{sample}_L1_readIDs.txt'
    if not rid_file.exists():
        print(f"  WARNING: {rid_file} not found")
        continue
    l1_ids = set(open(rid_file).read().strip().split('\n'))
    print(f"  {len(l1_ids)} L1 read IDs")

    records = analyze_sample(sample, condition, l1_ids, annot)
    all_records.extend(records)

if not all_records:
    print("ERROR: No data collected")
    exit(1)

df = pd.DataFrame(all_records)
df.to_csv(OUTDIR / 'body_flanking_per_read.tsv', sep='\t', index=False)
print(f"\nTotal reads: {len(df):,}")

# =============================================================================
# Results
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: Body vs Flanking m6A (thr=204)")
print("=" * 70)

# Overall stats
print(f"\nOverlap fraction: median={df['overlap_fraction'].median():.3f}, "
      f"mean={df['overlap_fraction'].mean():.3f}")
print(f"Body bases: median={df['body_bases'].median():.0f}, "
      f"Flank bases: median={df['flank_bases'].median():.0f}")

# WT vs KO comparison
for compartment, col in [('Body', 'body_m6a_per_kb'),
                          ('Flanking', 'flank_m6a_per_kb'),
                          ('Total', 'total_m6a_per_kb')]:
    wt = df[df['condition'] == 'WT'][col]
    ko = df[df['condition'] == 'KO'][col]

    # Filter zeros for flanking if needed
    wt_med = wt.median()
    ko_med = ko.median()
    fc = ko_med / wt_med if wt_med > 0 else float('inf')

    if len(wt) > 0 and len(ko) > 0:
        stat, pval = stats.mannwhitneyu(wt, ko, alternative='two-sided')
    else:
        pval = float('nan')

    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"\n  {compartment:>10s} m6A/kb:  WT={wt_med:.3f}  KO={ko_med:.3f}  "
          f"FC={fc:.3f}  P={pval:.2e}  {sig}")
    print(f"            n: WT={len(wt)}, KO={len(ko)}")
    print(f"            mean: WT={wt.mean():.3f}, KO={ko.mean():.3f}")

# Per-replicate
print("\n\nPer-replicate medians:")
for compartment, col in [('Body', 'body_m6a_per_kb'),
                          ('Flanking', 'flank_m6a_per_kb'),
                          ('Total', 'total_m6a_per_kb')]:
    print(f"\n  {compartment}:")
    for sample in sorted(SAMPLES.keys()):
        sub = df[df['sample'] == sample]
        if len(sub) > 0:
            print(f"    {sample}: median={sub[col].median():.3f}, n={len(sub)}")

# High-overlap subset (overlap > 0.7 = mostly L1 body)
print("\n" + "=" * 70)
print("Sensitivity: High-overlap reads (>0.7) — mostly L1 body")
print("=" * 70)
hi_ov = df[df['overlap_fraction'] > 0.7]
if len(hi_ov) > 50:
    for compartment, col in [('Body', 'body_m6a_per_kb'),
                              ('Total', 'total_m6a_per_kb')]:
        wt = hi_ov[hi_ov['condition'] == 'WT'][col]
        ko = hi_ov[hi_ov['condition'] == 'KO'][col]
        wt_med = wt.median()
        ko_med = ko.median()
        fc = ko_med / wt_med if wt_med > 0 else float('inf')
        stat, pval = stats.mannwhitneyu(wt, ko, alternative='two-sided')
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        print(f"  {compartment:>10s}: WT={wt_med:.3f} KO={ko_med:.3f} FC={fc:.3f} "
              f"P={pval:.2e} {sig}  (n={len(wt)}+{len(ko)})")
else:
    print(f"  Only {len(hi_ov)} high-overlap reads — insufficient")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
summary = []
for compartment, col in [('L1 body', 'body_m6a_per_kb'),
                          ('Flanking', 'flank_m6a_per_kb'),
                          ('Total (current)', 'total_m6a_per_kb')]:
    wt = df[df['condition'] == 'WT'][col]
    ko = df[df['condition'] == 'KO'][col]
    wt_med, ko_med = wt.median(), ko.median()
    fc = ko_med / wt_med if wt_med > 0 else float('inf')
    _, pval = stats.mannwhitneyu(wt, ko, alternative='two-sided')
    summary.append({
        'compartment': compartment,
        'WT_median': round(wt_med, 4),
        'KO_median': round(ko_med, 4),
        'fold_change': round(fc, 4),
        'P_value': pval,
        'n_WT': len(wt),
        'n_KO': len(ko),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUTDIR / 'body_flanking_summary.tsv', sep='\t', index=False)
print(summary_df.to_string(index=False))

print("\nDone.")
