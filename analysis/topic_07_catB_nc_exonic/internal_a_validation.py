#!/usr/bin/env python3
"""
Internal-A artifact validation for L1 poly(A) estimates.

Concern: L1 3' UTR has A-rich stretches → nanopolish may conflate internal-A
with the actual poly(A) tail → artificial inflation of poly(A) estimates,
especially for high-overlap reads that include L1 3' end.

If internal-A is the cause of high-overlap "immunity":
  1. Baseline poly(A) should be inflated in high-overlap reads
  2. The effect should depend on dist_to_3prime (closer = more internal-A)
  3. Removing reads near L1 3' end should abolish the immunity

Tests:
  T1. L1 3' end A-content: scan reference genome for A-rich stretches
  T2. Baseline poly(A) comparison: high vs low overlap at normal condition
  T3. dist_to_3prime stratified: immunity pattern by distance from L1 3' end
  T4. Exclude dist_to_3prime < 50bp: does immunity persist?
  T5. Nanopolish QC tag distribution by group
  T6. poly(A) distribution shape: internal-A → bimodal/fat right tail?
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
OUTDIR = TOPIC_07 / 'internal_a_validation'
OUTDIR.mkdir(exist_ok=True)

GENOME = PROJECT / 'reference/Human.fasta'
RMSK_GTF = PROJECT / 'reference/hg38_rmsk_TE.gtf'

# =========================================================================
# Load data
# =========================================================================
print("Loading unified dataset...")
df = pd.read_csv(TOPIC_07 / 'unified_classifier/unified_hela_ars_all_features.tsv', sep='\t')
print(f"  Total: {len(df)} reads")

# Define groups
df['ov_group'] = pd.cut(df['overlap_frac'],
                         bins=[0, 0.3, 0.5, 0.7, 1.01],
                         labels=['<0.3', '0.3-0.5', '0.5-0.7', '>0.7'])
df['is_high_ov'] = df['overlap_frac'] > 0.7

def ars_test(data):
    hela = data[data['cell_line'] == 'HeLa']['polya_length'].dropna()
    ars = data[data['cell_line'] == 'HeLa-Ars']['polya_length'].dropna()
    if len(hela) < 5 or len(ars) < 5:
        return np.nan, np.nan, len(hela), len(ars)
    _, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
    return ars.median() - hela.median(), p, len(hela), len(ars)

# =========================================================================
# T1. L1 3' end A-content in reference genome
# =========================================================================
print("\n" + "=" * 80)
print("T1: L1 3' end A-content in reference genome")
print("=" * 80)

# Parse RepeatMasker for elements in our dataset
l1_ids = set(df['transcript_id'].unique())
element_info = {}
with open(RMSK_GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        attrs = fields[8]
        tid = None
        for attr in attrs.split(';'):
            attr = attr.strip()
            if attr.startswith('transcript_id'):
                tid = attr.split('"')[1]
                break
        if tid and tid in l1_ids:
            element_info[tid] = (fields[0], int(fields[3])-1, int(fields[4]), fields[6])
            if len(element_info) == len(l1_ids):
                break

print(f"  Found {len(element_info)}/{len(l1_ids)} elements")

def reverse_complement(seq):
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))

fa = pysam.FastaFile(str(GENOME))

# Scan last 100bp of each L1 element for A-content
a_content_data = []
for tid, (chrom, start, end, strand) in element_info.items():
    # 3' end of L1 element: last 100bp
    if strand == '+':
        scan_start = max(start, end - 100)
        scan_end = end
    else:
        scan_start = start
        scan_end = min(end, start + 100)

    seq = fa.fetch(chrom, scan_start, scan_end).upper()
    if strand == '-':
        seq = reverse_complement(seq)

    # A-content in windows
    total_a = seq.count('A')
    total_len = len(seq)
    a_frac = total_a / total_len if total_len > 0 else 0

    # Check for A-runs (≥5 consecutive A's)
    max_a_run = 0
    current_run = 0
    for base in seq:
        if base == 'A':
            current_run += 1
            max_a_run = max(max_a_run, current_run)
        else:
            current_run = 0

    # Last 30bp (most critical for nanopolish confusion)
    last30 = seq[-30:] if len(seq) >= 30 else seq
    a_frac_last30 = last30.count('A') / len(last30) if len(last30) > 0 else 0

    a_content_data.append({
        'transcript_id': tid,
        'l1_3end_length': total_len,
        'a_frac_100bp': a_frac,
        'a_frac_last30bp': a_frac_last30,
        'max_a_run': max_a_run,
        'seq_last30': last30,
    })

fa.close()

adf = pd.DataFrame(a_content_data)
df = df.merge(adf[['transcript_id', 'a_frac_100bp', 'a_frac_last30bp', 'max_a_run']],
              on='transcript_id', how='left')

print(f"\n  L1 3' end A-content summary (last 100bp):")
print(f"    Mean A-fraction: {adf['a_frac_100bp'].mean():.3f} (expected random: 0.25)")
print(f"    Mean A-fraction (last 30bp): {adf['a_frac_last30bp'].mean():.3f}")
print(f"    Mean max A-run: {adf['max_a_run'].mean():.1f}")
print(f"    Elements with max_a_run ≥ 10: {(adf['max_a_run'] >= 10).sum()} ({(adf['max_a_run'] >= 10).mean()*100:.1f}%)")
print(f"    Elements with max_a_run ≥ 15: {(adf['max_a_run'] >= 15).sum()} ({(adf['max_a_run'] >= 15).mean()*100:.1f}%)")
print(f"    Elements with a_frac_last30 > 0.5: {(adf['a_frac_last30bp'] > 0.5).sum()} ({(adf['a_frac_last30bp'] > 0.5).mean()*100:.1f}%)")

# Compare A-content by overlap group
print(f"\n  A-content by overlap group:")
for grp in ['<0.3', '0.3-0.5', '0.5-0.7', '>0.7']:
    sub = df[df['ov_group'] == grp]
    print(f"    {grp:10s}: a_frac_100bp={sub['a_frac_100bp'].mean():.3f}, "
          f"a_frac_last30={sub['a_frac_last30bp'].mean():.3f}, "
          f"max_a_run={sub['max_a_run'].mean():.1f}, n={len(sub)}")

# =========================================================================
# T2. Baseline poly(A) comparison — high vs low overlap, NORMAL only
# =========================================================================
print("\n" + "=" * 80)
print("T2: Baseline poly(A) by overlap group (HeLa normal only)")
print("=" * 80)
print("  If internal-A inflates poly(A), high-overlap should have longer baseline")

hela_only = df[df['cell_line'] == 'HeLa']
print(f"\n  {'Overlap':>10s} | {'Median':>8s} {'Mean':>8s} {'SD':>8s} {'n':>6s}")
print("  " + "-" * 50)
for grp in ['<0.3', '0.3-0.5', '0.5-0.7', '>0.7']:
    sub = hela_only[hela_only['ov_group'] == grp]['polya_length'].dropna()
    print(f"  {grp:>10s} | {sub.median():8.1f} {sub.mean():8.1f} {sub.std():8.1f} {len(sub):6d}")

# Statistical test: high vs low at baseline
hi_base = hela_only[hela_only['is_high_ov']]['polya_length'].dropna()
lo_base = hela_only[~hela_only['is_high_ov']]['polya_length'].dropna()
stat, p = stats.mannwhitneyu(hi_base, lo_base, alternative='two-sided')
print(f"\n  High-ov vs low-ov at baseline: median {hi_base.median():.1f} vs {lo_base.median():.1f}, "
      f"diff={hi_base.median()-lo_base.median():+.1f}, p={p:.2e}")

# =========================================================================
# T3. dist_to_3prime stratified arsenite response
# =========================================================================
print("\n" + "=" * 80)
print("T3: dist_to_3prime stratified arsenite response")
print("=" * 80)
print("  If internal-A drives immunity, reads ending at L1 3' end (dist≈0) should be most immune")

# Create dist bins
df['d3p_bin'] = pd.cut(df['dist_to_3prime'],
                        bins=[-1, 0, 50, 200, 1000, 100000],
                        labels=['0', '1-50', '51-200', '201-1000', '>1000'])

print(f"\n  {'dist_to_3prime':>15s} | {'All Δ':>8s} {'n':>6s} | {'HiOv Δ':>8s} {'n':>6s} | {'LoOv Δ':>8s} {'n':>6s}")
print("  " + "-" * 80)

for dbin in ['0', '1-50', '51-200', '201-1000', '>1000']:
    sub_all = df[df['d3p_bin'] == dbin]
    sub_hi = sub_all[sub_all['is_high_ov']]
    sub_lo = sub_all[~sub_all['is_high_ov']]

    d_all, p_all, n_h, n_a = ars_test(sub_all)
    d_hi, p_hi, _, _ = ars_test(sub_hi)
    d_lo, p_lo, _, _ = ars_test(sub_lo)

    d_all_s = f"{d_all:+.1f}" if not np.isnan(d_all) else 'n/a'
    d_hi_s = f"{d_hi:+.1f}" if not np.isnan(d_hi) else 'n/a'
    d_lo_s = f"{d_lo:+.1f}" if not np.isnan(d_lo) else 'n/a'
    print(f"  {dbin:>15s} | {d_all_s:>8s} {n_h+n_a:6d} | {d_hi_s:>8s} {sub_hi.shape[0]:6d} | {d_lo_s:>8s} {sub_lo.shape[0]:6d}")

# =========================================================================
# T4. Exclude reads near L1 3' end — does immunity persist?
# =========================================================================
print("\n" + "=" * 80)
print("T4: Exclude reads near L1 3' end (dist_to_3prime > threshold)")
print("=" * 80)
print("  If internal-A is the cause, excluding L1-3'-proximal reads should abolish immunity")

for excl_thr in [0, 50, 100, 200]:
    sub = df[df['dist_to_3prime'] > excl_thr]
    hi = sub[sub['is_high_ov']]
    lo = sub[~sub['is_high_ov']]

    d_hi, p_hi, nh_hela, nh_ars = ars_test(hi)
    d_lo, p_lo, nl_hela, nl_ars = ars_test(lo)
    dd = abs(d_hi - d_lo) if not (np.isnan(d_hi) or np.isnan(d_lo)) else np.nan

    d_hi_s = f"{d_hi:+.1f}" if not np.isnan(d_hi) else 'n/a'
    d_lo_s = f"{d_lo:+.1f}" if not np.isnan(d_lo) else 'n/a'
    dd_s = f"{dd:.1f}" if not np.isnan(dd) else 'n/a'
    print(f"  Exclude d3p ≤ {excl_thr:3d}: High-ov Δ={d_hi_s:>8s} (n={nh_hela+nh_ars}), "
          f"Low-ov Δ={d_lo_s:>8s} (n={nl_hela+nl_ars}), |ΔΔ|={dd_s}")

# =========================================================================
# T5. Exclude reads with high A-content at L1 3' end
# =========================================================================
print("\n" + "=" * 80)
print("T5: Exclude elements with high 3' A-content (internal-A risk)")
print("=" * 80)

for a_thr in [0.4, 0.5, 0.6]:
    sub = df[df['a_frac_last30bp'] <= a_thr]
    hi = sub[sub['is_high_ov']]
    lo = sub[~sub['is_high_ov']]

    d_hi, _, nh, _ = ars_test(hi)
    d_lo, _, nl, _ = ars_test(lo)
    dd = abs(d_hi - d_lo) if not (np.isnan(d_hi) or np.isnan(d_lo)) else np.nan

    d_hi_s = f"{d_hi:+.1f}" if not np.isnan(d_hi) else 'n/a'
    d_lo_s = f"{d_lo:+.1f}" if not np.isnan(d_lo) else 'n/a'
    dd_s = f"{dd:.1f}" if dd and not np.isnan(dd) else 'n/a'
    n_excl = len(df) - len(sub)
    print(f"  Exclude a_frac_last30 > {a_thr}: High-ov Δ={d_hi_s:>8s}, Low-ov Δ={d_lo_s:>8s}, "
          f"|ΔΔ|={dd_s} (excluded {n_excl} reads)")

# Exclude elements with long A-runs
for arun_thr in [10, 15, 20]:
    sub = df[df['max_a_run'] < arun_thr]
    hi = sub[sub['is_high_ov']]
    lo = sub[~sub['is_high_ov']]

    d_hi, _, nh, _ = ars_test(hi)
    d_lo, _, nl, _ = ars_test(lo)
    dd = abs(d_hi - d_lo) if not (np.isnan(d_hi) or np.isnan(d_lo)) else np.nan

    d_hi_s = f"{d_hi:+.1f}" if not np.isnan(d_hi) else 'n/a'
    d_lo_s = f"{d_lo:+.1f}" if not np.isnan(d_lo) else 'n/a'
    dd_s = f"{dd:.1f}" if dd and not np.isnan(dd) else 'n/a'
    n_excl = len(df) - len(sub)
    print(f"  Exclude max_a_run ≥ {arun_thr:2d}: High-ov Δ={d_hi_s:>8s}, Low-ov Δ={d_lo_s:>8s}, "
          f"|ΔΔ|={dd_s} (excluded {n_excl} reads)")

# =========================================================================
# T6. A-content correlation with poly(A) length
# =========================================================================
print("\n" + "=" * 80)
print("T6: A-content correlation with poly(A) length")
print("=" * 80)
print("  If internal-A inflates poly(A), A-content should correlate with poly(A)")

for condition in ['HeLa', 'HeLa-Ars']:
    sub = df[df['cell_line'] == condition].dropna(subset=['a_frac_last30bp', 'polya_length'])
    r_all, p_all = stats.spearmanr(sub['a_frac_last30bp'], sub['polya_length'])
    r_hi, p_hi = stats.spearmanr(sub[sub['is_high_ov']]['a_frac_last30bp'],
                                   sub[sub['is_high_ov']]['polya_length'])
    print(f"\n  {condition}:")
    print(f"    All reads:    Spearman r={r_all:+.3f}, p={p_all:.2e}, n={len(sub)}")
    print(f"    High-ov only: Spearman r={r_hi:+.3f}, p={p_hi:.2e}, n={sub['is_high_ov'].sum()}")

# =========================================================================
# T7. Within high-overlap: low-A vs high-A L1 elements — same immunity?
# =========================================================================
print("\n" + "=" * 80)
print("T7: Within high-overlap: low-A vs high-A elements")
print("=" * 80)
print("  If internal-A drives immunity, high-A elements should be more immune")

hi_ov = df[df['is_high_ov']].copy()
hi_ov['a_group'] = pd.cut(hi_ov['a_frac_last30bp'],
                            bins=[0, 0.3, 0.5, 1.01],
                            labels=['low-A', 'mid-A', 'high-A'])

print(f"\n  Among high-overlap reads (n={len(hi_ov)}):")
print(f"  {'A-group':>10s} | {'Δ':>8s} {'p':>10s} {'n':>6s} | {'base_med':>10s}")
print("  " + "-" * 55)
for grp in ['low-A', 'mid-A', 'high-A']:
    sub = hi_ov[hi_ov['a_group'] == grp]
    d, p, nh, na = ars_test(sub)
    base_med = sub[sub['cell_line'] == 'HeLa']['polya_length'].median()
    d_s = f"{d:+.1f}" if not np.isnan(d) else 'n/a'
    p_s = f"{p:.2e}" if not np.isnan(p) else 'n/a'
    print(f"  {grp:>10s} | {d_s:>8s} {p_s:>10s} {nh+na:6d} | {base_med:10.1f}")

# =========================================================================
# T8. Reference check: control transcripts internal-A
# =========================================================================
print("\n" + "=" * 80)
print("T8: Sanity check — does overlap_frac pattern hold for PASS-only reads?")
print("=" * 80)
print("  Cat B reads removed — does overlap_frac still predict immunity?")

pass_only = df[df['source'] == 'PASS']
for grp in ['<0.3', '0.3-0.5', '0.5-0.7', '>0.7']:
    sub = pass_only[pass_only['ov_group'] == grp]
    d, p, nh, na = ars_test(sub)
    d_s = f"{d:+.1f}" if not np.isnan(d) else 'n/a'
    p_s = f"{p:.2e}" if not np.isnan(p) else 'n/a'
    print(f"  PASS {grp:>10s}: Δ={d_s:>8s}, p={p_s:>10s}, n={nh+na}")

# =========================================================================
# Summary verdict
# =========================================================================
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

# Collect key evidence
print("""
  Internal-A artifact prediction vs observation:

  Prediction 1: High-ov should have inflated baseline poly(A)
  → Result: [see T2 above]

  Prediction 2: Immunity should be strongest at dist_to_3prime ≈ 0
  → Result: [see T3 above]

  Prediction 3: Excluding L1-3'-proximal reads should abolish immunity
  → Result: [see T4 above]

  Prediction 4: A-content should correlate with poly(A) length
  → Result: [see T6 above]

  Prediction 5: Within high-ov, high-A elements should be more immune
  → Result: [see T7 above]
""")

# Save annotated data
df.to_csv(OUTDIR / 'internal_a_annotated.tsv', sep='\t', index=False)
print(f"Saved to: {OUTDIR}/internal_a_annotated.tsv")
print("Done!")
