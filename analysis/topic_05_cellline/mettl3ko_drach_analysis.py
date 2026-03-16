#!/usr/bin/env python3
"""
METTL3 KO m6A analysis stratified by DRACH motif context.

Hypothesis: METTL3-dependent m6A sits on DRACH motifs.
At thr>=204, non-DRACH (METTL3-independent) sites dominate → KO effect masked.
Filtering to DRACH-only should reveal METTL3 KO effect on L1.

Approach:
  1. Parse MAFIA BAMs, extract each m6A site's ML value + 5-mer context
  2. For reverse reads: reverse-complement the context to get original strand motif
  3. Classify: DRACH, relaxed-DRACH, non-DRACH
  4. Compare WT vs KO m6A density for each motif class
"""
import os, sys
import numpy as np
import pandas as pd
import pysam
from scipy import stats
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

VAULT = Path('/vault/external-datasets/2026/PRJEB40872_HEK293T_METTL3KO_xPore')
MAFIA_DIR = VAULT / 'mafia_guppy'

SAMPLES = {
    'WT_rep1': 'WT', 'WT_rep2': 'WT', 'WT_rep3': 'WT',
    'KO_rep1': 'KO', 'KO_rep2': 'KO', 'KO_rep3': 'KO',
}

OUTDIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/mettl3ko_m6a_signal')
OUTDIR.mkdir(exist_ok=True)

COMP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

def revcomp(seq):
    return ''.join(COMP.get(b, 'N') for b in reversed(seq))

def classify_motif(fivemer):
    """Classify a 5-mer context around an A (m6A candidate).
    fivemer = [pos-2, pos-1, A, pos+1, pos+2] on original strand.
    DRACH: D=[AGT] R=[AG] A C H=[ACT]
    """
    if len(fivemer) != 5:
        return 'edge'
    d, r, a, c, h = fivemer.upper()
    if a != 'A':
        return 'non_A'  # shouldn't happen after correction

    is_drach = (d in 'AGT') and (r in 'AG') and (c == 'C') and (h in 'ACT')
    # Relaxed: allow C at pos+1 to be C or T (some papers include DRATH)
    is_dra_x_h = (d in 'AGT') and (r in 'AG') and (c in 'CT') and (h in 'ACT')
    # GGAC specifically (most common METTL3 motif)
    is_ggac = (d in 'AGT') and (r == 'G') and (c == 'C')

    if is_drach:
        return 'DRACH'
    elif is_dra_x_h:
        return 'DRATH'  # relaxed
    else:
        return 'non-DRACH'

###############################################################################
# Parse BAMs — extract per-site data with motif context
###############################################################################
def parse_bam_with_motif(bam_path):
    """Parse MAFIA BAM, extract per-site ML values with 5-mer motif context."""
    if not os.path.exists(bam_path):
        return [], []
    bam = pysam.AlignmentFile(bam_path, 'rb')
    read_data = []
    site_data = []

    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        rlen = read.query_alignment_length
        seq = read.query_sequence
        if rlen is None or rlen < 100 or seq is None:
            continue

        mm_tag = ml_tag = None
        for t in ['MM', 'Mm']:
            if read.has_tag(t):
                mm_tag = read.get_tag(t); break
        for t in ['ML', 'Ml']:
            if read.has_tag(t):
                ml_tag = read.get_tag(t); break

        read_info = {
            'read_id': read.query_name,
            'read_length': rlen,
            'is_reverse': read.is_reverse,
        }

        if mm_tag is None or ml_tag is None:
            read_data.append(read_info)
            continue

        ml_list = list(ml_tag)
        entries = mm_tag.rstrip(';').split(';')
        ml_idx = 0

        for entry in entries:
            parts = entry.strip().split(',')
            base_mod = parts[0]
            skips = [int(x) for x in parts[1:]] if len(parts) > 1 else []
            n_sites = len(skips)

            if '21891' in base_mod:
                strand_char = '+' if '+' in base_mod else '-'

                # Calculate positions in stored sequence
                positions = []
                if strand_char == '+':
                    pos = -1
                    for s in skips:
                        pos += s + 1
                        positions.append(pos)
                else:  # N-
                    pos = len(seq)
                    for s in skips:
                        pos -= (s + 1)
                        positions.append(pos)

                for j, p in enumerate(positions):
                    if ml_idx + j >= len(ml_list):
                        break
                    ml_val = ml_list[ml_idx + j]

                    # Extract 5-mer context
                    if 2 <= p < len(seq) - 2:
                        raw_context = seq[p-2:p+3]
                        if strand_char == '-':
                            # Reverse complement to get original strand
                            orig_context = revcomp(raw_context)
                        else:
                            orig_context = raw_context
                        motif_class = classify_motif(orig_context)
                    else:
                        orig_context = 'EDGE'
                        motif_class = 'edge'

                    site_data.append({
                        'read_id': read.query_name,
                        'read_length': rlen,
                        'site_pos': p,
                        'ml_value': ml_val,
                        'raw_context': raw_context if 2 <= p < len(seq) - 2 else 'EDGE',
                        'orig_context': orig_context,
                        'motif_class': motif_class,
                        'strand': strand_char,
                    })

            ml_idx += n_sites

        read_data.append(read_info)

    bam.close()
    return read_data, site_data

###############################################################################
# Parse all samples
###############################################################################
print("=" * 70)
print("Parsing MAFIA BAMs with motif context extraction")
print("=" * 70)

all_reads = {}  # sample -> read_data
all_sites = []

for sample, condition in SAMPLES.items():
    bam_path = MAFIA_DIR / sample / 'mAFiA.reads.bam'
    print(f"  {sample} ({condition})...", end=' ', flush=True)
    read_data, site_data = parse_bam_with_motif(str(bam_path))
    for s in site_data:
        s['sample'] = sample
        s['condition'] = condition
    all_sites.extend(site_data)
    all_reads[sample] = read_data
    print(f"{len(read_data)} reads, {len(site_data)} m6A sites")

sites_df = pd.DataFrame(all_sites)
print(f"\nTotal m6A candidate sites: {len(sites_df):,}")

###############################################################################
# Motif classification summary
###############################################################################
print("\n" + "=" * 70)
print("Motif classification summary")
print("=" * 70)

for cond in ['WT', 'KO']:
    sub = sites_df[sites_df['condition'] == cond]
    counts = sub['motif_class'].value_counts()
    total = len(sub)
    print(f"\n  {cond} ({total:,} sites):")
    for cls in ['DRACH', 'DRATH', 'non-DRACH', 'non_A', 'edge']:
        n = counts.get(cls, 0)
        print(f"    {cls:12s}: {n:6d} ({100*n/total:.1f}%)")

# Top motifs
print("\n  Top 20 DRACH 5-mers (WT):")
drach_wt = sites_df[(sites_df['condition'] == 'WT') & (sites_df['motif_class'] == 'DRACH')]
top_motifs = drach_wt['orig_context'].value_counts().head(20)
for motif, count in top_motifs.items():
    print(f"    {motif}: {count}")

print("\n  Top 20 non-DRACH 5-mers (WT):")
non_drach_wt = sites_df[(sites_df['condition'] == 'WT') & (sites_df['motif_class'] == 'non-DRACH')]
top_non = non_drach_wt['orig_context'].value_counts().head(20)
for motif, count in top_non.items():
    print(f"    {motif}: {count}")

###############################################################################
# Per-read m6A density by motif class and threshold
###############################################################################
print("\n" + "=" * 70)
print("Per-read m6A density by motif class")
print("=" * 70)

# Build per-read counts for each motif class × threshold combination
motif_classes = ['DRACH', 'non-DRACH']  # main comparison
thresholds = [0, 50, 100, 128, 153, 178, 204, 229]

# Aggregate per read
per_read_list = []
for sample, condition in SAMPLES.items():
    reads = all_reads[sample]
    sample_sites = sites_df[sites_df['sample'] == sample]

    for rd in reads:
        rid = rd['read_id']
        rlen = rd['read_length']
        rkb = rlen / 1000

        rd_sites = sample_sites[sample_sites['read_id'] == rid]

        row = {
            'read_id': rid,
            'read_length': rlen,
            'read_kb': rkb,
            'sample': sample,
            'condition': condition,
        }

        for mc in motif_classes:
            mc_sites = rd_sites[rd_sites['motif_class'] == mc]
            for thr in thresholds:
                n = (mc_sites['ml_value'] >= thr).sum() if len(mc_sites) > 0 else 0
                row[f'{mc}_sites_{thr}'] = n
                row[f'{mc}_per_kb_{thr}'] = n / rkb if rkb > 0 else 0

            # Also DRATH (relaxed DRACH)
            if mc == 'DRACH':
                drath_sites = rd_sites[rd_sites['motif_class'] == 'DRATH']
                combined = pd.concat([mc_sites, drath_sites])
                for thr in thresholds:
                    n = (combined['ml_value'] >= thr).sum() if len(combined) > 0 else 0
                    row[f'DRACH_extended_sites_{thr}'] = n
                    row[f'DRACH_extended_per_kb_{thr}'] = n / rkb if rkb > 0 else 0

        # All sites (any motif)
        for thr in thresholds:
            n = (rd_sites['ml_value'] >= thr).sum() if len(rd_sites) > 0 else 0
            row[f'all_sites_{thr}'] = n
            row[f'all_per_kb_{thr}'] = n / rkb if rkb > 0 else 0

        per_read_list.append(row)

per_read_df = pd.DataFrame(per_read_list)

###############################################################################
# Compare WT vs KO at each threshold, for each motif class
###############################################################################
print(f"\n  {'Motif':>16s} {'Thr':>5s} {'WT med':>8s} {'KO med':>8s} {'KO/WT':>8s} {'MWU P':>12s} {'sig':>4s}")
print("  " + "-" * 65)

comparison_results = []
for mc in ['all', 'DRACH', 'DRACH_extended', 'non-DRACH']:
    for thr in thresholds:
        col = f'{mc}_per_kb_{thr}'
        if col not in per_read_df.columns:
            continue
        wt = per_read_df[per_read_df['condition'] == 'WT'][col]
        ko = per_read_df[per_read_df['condition'] == 'KO'][col]
        wt_med = wt.median()
        ko_med = ko.median()
        ratio = ko_med / wt_med if wt_med > 0 else float('nan')
        _, p = stats.mannwhitneyu(wt, ko, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        if thr in [0, 128, 178, 204]:  # key thresholds only in print
            print(f"  {mc:>16s} {thr:5d} {wt_med:8.3f} {ko_med:8.3f} {ratio:7.3f}x {p:12.2e} {sig:>4s}")

        comparison_results.append({
            'motif_class': mc,
            'threshold': thr,
            'wt_median': round(wt_med, 4),
            'ko_median': round(ko_med, 4),
            'ko_wt_ratio': round(ratio, 4) if not np.isnan(ratio) else 'nan',
            'mwu_p': p,
            'sig': sig,
        })

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(OUTDIR / 'drach_motif_comparison.tsv', sep='\t', index=False)

###############################################################################
# ML value distribution by motif class
###############################################################################
print("\n" + "=" * 70)
print("ML value distribution by motif class (site-level)")
print("=" * 70)

for mc in ['DRACH', 'non-DRACH']:
    wt_mls = sites_df[(sites_df['condition'] == 'WT') & (sites_df['motif_class'] == mc)]['ml_value']
    ko_mls = sites_df[(sites_df['condition'] == 'KO') & (sites_df['motif_class'] == mc)]['ml_value']
    ks_stat, ks_p = stats.ks_2samp(wt_mls, ko_mls)
    _, mwu_p = stats.mannwhitneyu(wt_mls, ko_mls, alternative='two-sided')
    print(f"\n  {mc}:")
    print(f"    WT: n={len(wt_mls):,}, mean={wt_mls.mean():.1f}, median={wt_mls.median():.0f}")
    print(f"    KO: n={len(ko_mls):,}, mean={ko_mls.mean():.1f}, median={ko_mls.median():.0f}")
    print(f"    Mean shift: {ko_mls.mean() - wt_mls.mean():.2f}")
    print(f"    KS P={ks_p:.2e}, MWU P={mwu_p:.2e}")

    # Band analysis
    print(f"    ML bands (fraction of sites):")
    for lo, hi in [(0, 128), (128, 204), (204, 256)]:
        wt_n = ((wt_mls >= lo) & (wt_mls < hi)).sum()
        ko_n = ((ko_mls >= lo) & (ko_mls < hi)).sum()
        wt_pct = 100 * wt_n / len(wt_mls)
        ko_pct = 100 * ko_n / len(ko_mls)
        print(f"      ML {lo:3d}-{hi:3d}: WT {wt_pct:5.1f}%  KO {ko_pct:5.1f}%  (Δ={ko_pct-wt_pct:+.1f}pp)")

###############################################################################
# Also do control reads
###############################################################################
print("\n" + "=" * 70)
print("Control reads — DRACH analysis for comparison")
print("=" * 70)

ctrl_sites_all = []
ctrl_reads_all = {}
for sample, condition in SAMPLES.items():
    bam_path = MAFIA_DIR / f'{sample}_ctrl' / 'mAFiA.reads.bam'
    print(f"  {sample}_ctrl ({condition})...", end=' ', flush=True)
    rd, sd = parse_bam_with_motif(str(bam_path))
    for s in sd:
        s['sample'] = sample
        s['condition'] = condition
    ctrl_sites_all.extend(sd)
    ctrl_reads_all[sample] = rd
    print(f"{len(rd)} reads, {len(sd)} sites")

ctrl_sites_df = pd.DataFrame(ctrl_sites_all)

print(f"\n  Control motif breakdown:")
for cond in ['WT', 'KO']:
    sub = ctrl_sites_df[ctrl_sites_df['condition'] == cond]
    drach_n = (sub['motif_class'] == 'DRACH').sum()
    non_drach_n = (sub['motif_class'] == 'non-DRACH').sum()
    total = len(sub)
    print(f"    {cond}: DRACH {drach_n} ({100*drach_n/total:.1f}%), non-DRACH {non_drach_n} ({100*non_drach_n/total:.1f}%)")

# Build per-read for control
ctrl_per_read = []
for sample, condition in SAMPLES.items():
    reads = ctrl_reads_all[sample]
    sample_sites = ctrl_sites_df[ctrl_sites_df['sample'] == sample]
    for rd in reads:
        rid = rd['read_id']
        rlen = rd['read_length']
        rkb = rlen / 1000
        rd_sites = sample_sites[sample_sites['read_id'] == rid]
        row = {'read_id': rid, 'read_length': rlen, 'condition': condition}
        for mc in ['DRACH', 'non-DRACH']:
            mc_sites = rd_sites[rd_sites['motif_class'] == mc]
            for thr in [128, 204]:
                n = (mc_sites['ml_value'] >= thr).sum() if len(mc_sites) > 0 else 0
                row[f'{mc}_per_kb_{thr}'] = n / rkb if rkb > 0 else 0
        ctrl_per_read.append(row)

ctrl_pr_df = pd.DataFrame(ctrl_per_read)

print(f"\n  Control WT vs KO:")
for mc in ['DRACH', 'non-DRACH']:
    for thr in [128, 204]:
        col = f'{mc}_per_kb_{thr}'
        wt = ctrl_pr_df[ctrl_pr_df['condition'] == 'WT'][col]
        ko = ctrl_pr_df[ctrl_pr_df['condition'] == 'KO'][col]
        wt_med = wt.median()
        ko_med = ko.median()
        ratio = ko_med / wt_med if wt_med > 0 else float('nan')
        _, p = stats.mannwhitneyu(wt, ko, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"    {mc:>12s} ML>={thr}: WT {wt_med:.3f} vs KO {ko_med:.3f} = {ratio:.3f}x P={p:.2e} {sig}")

###############################################################################
# Save site-level data
###############################################################################
sites_df.to_csv(OUTDIR / 'l1_site_level_motif.tsv.gz', sep='\t', index=False,
                compression='gzip')
per_read_df.to_csv(OUTDIR / 'l1_per_read_motif.tsv', sep='\t', index=False)

print(f"\nSaved to {OUTDIR}/")
print("\nDONE.")
