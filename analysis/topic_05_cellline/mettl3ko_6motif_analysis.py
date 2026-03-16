#!/usr/bin/env python3
"""
METTL3 KO m6A analysis: restrict to original mAFiA 6 high-accuracy DRACH motifs.

Original mAFiA (Hendra et al. Nat Commun 2024) trained on 6 motifs:
  GGACT, GGACA, GAACT, AGACT, GGACC, TGACT
These cover ~80% of consensus m6A sites and have the highest detection accuracy.

psi-co-mAFiA expanded to all 18 DRACH motifs, but the additional 12 motifs
have less training data and potentially lower accuracy.

Hypothesis: restricting to 6 high-accuracy motifs enriches for
METTL3-dependent signal and reveals KO effect more clearly.
"""
import os, sys
import numpy as np
import pandas as pd
import pysam
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

VAULT = Path('/vault/external-datasets/2026/PRJEB40872_HEK293T_METTL3KO_xPore')
MAFIA_DIR = VAULT / 'mafia_guppy'
REF_FASTA = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta'

SAMPLES = {
    'WT_rep1': 'WT', 'WT_rep2': 'WT', 'WT_rep3': 'WT',
    'KO_rep1': 'KO', 'KO_rep2': 'KO', 'KO_rep3': 'KO',
}

# Original mAFiA 6 motifs (DNA encoding, on + strand)
ORIGINAL_6 = {'GGACT', 'GGACA', 'GAACT', 'AGACT', 'GGACC', 'TGACT'}

OUTDIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/mettl3ko_m6a_signal')
OUTDIR.mkdir(exist_ok=True)

COMP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
def revcomp(s): return ''.join(COMP.get(b, 'N') for b in reversed(s))

###############################################################################
# Parse BAM with reference-based motif context
###############################################################################
def parse_bam_ref_motif(bam_path, ref):
    """Parse MAFIA BAM, extract m6A sites with reference-based motif context."""
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

        # Build read-to-ref coordinate map
        aligned_pairs = read.get_aligned_pairs()
        read_to_ref = {}
        for rp, rfp in aligned_pairs:
            if rp is not None and rfp is not None:
                read_to_ref[rp] = rfp

        chrom = read.reference_name

        read_info = {'read_id': read.query_name, 'read_length': rlen}

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
                positions = []
                if strand_char == '+':
                    pos = -1
                    for s in skips:
                        pos += s + 1
                        positions.append(pos)
                else:
                    pos = len(seq)
                    for s in skips:
                        pos -= (s + 1)
                        positions.append(pos)

                for j, p in enumerate(positions):
                    if ml_idx + j >= len(ml_list):
                        break
                    ml_val = ml_list[ml_idx + j]

                    # Get reference 5-mer context
                    motif = 'unknown'
                    if p in read_to_ref:
                        ref_pos = read_to_ref[p]
                        try:
                            ref_5mer = ref.fetch(chrom, ref_pos - 2, ref_pos + 3).upper()
                            if len(ref_5mer) == 5:
                                if read.is_reverse:
                                    ref_5mer_orig = revcomp(ref_5mer)
                                else:
                                    ref_5mer_orig = ref_5mer
                                motif = ref_5mer_orig
                        except:
                            pass

                    site_data.append({
                        'read_id': read.query_name,
                        'read_length': rlen,
                        'ml_value': ml_val,
                        'motif': motif,
                        'is_orig6': motif in ORIGINAL_6,
                    })

            ml_idx += n_sites

        read_data.append(read_info)

    bam.close()
    return read_data, site_data

###############################################################################
# Parse all L1 samples
###############################################################################
print("=" * 70)
print("Parsing MAFIA BAMs with reference-based motif context")
print("=" * 70)

ref = pysam.FastaFile(REF_FASTA)

all_reads = {}
all_sites = []

for sample, condition in SAMPLES.items():
    bam_path = MAFIA_DIR / sample / 'mAFiA.reads.bam'
    print(f"  {sample} ({condition})...", end=' ', flush=True)
    rd, sd = parse_bam_ref_motif(str(bam_path), ref)
    for s in sd:
        s['sample'] = sample
        s['condition'] = condition
    all_sites.extend(sd)
    all_reads[sample] = rd
    print(f"{len(rd)} reads, {len(sd)} sites")

sites_df = pd.DataFrame(all_sites)
print(f"\nTotal m6A sites: {len(sites_df):,}")

###############################################################################
# Motif distribution
###############################################################################
print("\n" + "=" * 70)
print("Motif distribution (reference-based)")
print("=" * 70)

motif_counts = sites_df['motif'].value_counts()
total = len(sites_df)
orig6_n = sites_df['is_orig6'].sum()
print(f"  Original 6 motifs: {orig6_n:,} ({100*orig6_n/total:.1f}%)")
print(f"  Extended 12 motifs: {total - orig6_n:,} ({100*(total-orig6_n)/total:.1f}%)")

print(f"\n  All 18 DRACH motifs:")
for motif, count in motif_counts.head(18).items():
    tag = " ← orig6" if motif in ORIGINAL_6 else ""
    print(f"    {motif}: {count:5d} ({100*count/total:.1f}%){tag}")

###############################################################################
# Per-motif ML value distribution
###############################################################################
print("\n" + "=" * 70)
print("Per-motif mean ML value")
print("=" * 70)

print(f"  {'Motif':>8s} {'N sites':>8s} {'Mean ML':>8s} {'Med ML':>8s} {'%>=204':>8s} {'orig6':>6s}")
print("  " + "-" * 50)
for motif in motif_counts.head(18).index:
    sub = sites_df[sites_df['motif'] == motif]
    pct204 = 100 * (sub['ml_value'] >= 204).sum() / len(sub)
    tag = "yes" if motif in ORIGINAL_6 else "no"
    print(f"  {motif:>8s} {len(sub):8d} {sub['ml_value'].mean():8.1f} "
          f"{sub['ml_value'].median():8.0f} {pct204:7.1f}% {tag:>6s}")

# Aggregate: orig6 vs extended12
orig6_ml = sites_df[sites_df['is_orig6']]['ml_value']
ext12_ml = sites_df[~sites_df['is_orig6']]['ml_value']
print(f"\n  Original 6: mean ML = {orig6_ml.mean():.1f}, median = {orig6_ml.median():.0f}, "
      f"%>=204 = {100*(orig6_ml>=204).sum()/len(orig6_ml):.1f}%")
print(f"  Extended 12: mean ML = {ext12_ml.mean():.1f}, median = {ext12_ml.median():.0f}, "
      f"%>=204 = {100*(ext12_ml>=204).sum()/len(ext12_ml):.1f}%")

###############################################################################
# WT vs KO comparison: original 6 vs extended 12 vs all
###############################################################################
print("\n" + "=" * 70)
print("WT vs KO comparison by motif set and threshold")
print("=" * 70)

# Build per-read counts
per_read_list = []
for sample, condition in SAMPLES.items():
    reads = all_reads[sample]
    sample_sites = sites_df[sites_df['sample'] == sample]

    for rd in reads:
        rid = rd['read_id']
        rlen = rd['read_length']
        rkb = rlen / 1000

        rd_sites = sample_sites[sample_sites['read_id'] == rid]
        rd_orig6 = rd_sites[rd_sites['is_orig6']]
        rd_ext12 = rd_sites[~rd_sites['is_orig6']]

        row = {
            'read_id': rid,
            'read_length': rlen,
            'read_kb': rkb,
            'sample': sample,
            'condition': condition,
        }

        for label, subset in [('all', rd_sites), ('orig6', rd_orig6), ('ext12', rd_ext12)]:
            for thr in [0, 128, 153, 178, 204]:
                n = (subset['ml_value'] >= thr).sum() if len(subset) > 0 else 0
                row[f'{label}_sites_{thr}'] = n
                row[f'{label}_per_kb_{thr}'] = n / rkb if rkb > 0 else 0
            # Probability mass
            if len(subset) > 0:
                row[f'{label}_prob_mass_kb'] = subset['ml_value'].sum() / 255.0 / rkb if rkb > 0 else 0
            else:
                row[f'{label}_prob_mass_kb'] = 0
            # Moderate band (128-204)
            n_mod = ((subset['ml_value'] >= 128) & (subset['ml_value'] < 204)).sum() if len(subset) > 0 else 0
            row[f'{label}_moderate_kb'] = n_mod / rkb if rkb > 0 else 0

        per_read_list.append(row)

pr_df = pd.DataFrame(per_read_list)

# Print comparison table
print(f"\n  {'Motif set':>12s} {'Metric':>20s} {'WT med':>8s} {'KO med':>8s} {'KO/WT':>8s} {'P':>12s} {'sig':>4s}")
print("  " + "-" * 75)

results = []
for label in ['all', 'orig6', 'ext12']:
    for metric_name, col in [
        ('sites/kb (>=128)', f'{label}_per_kb_128'),
        ('sites/kb (>=204)', f'{label}_per_kb_204'),
        ('moderate/kb', f'{label}_moderate_kb'),
        ('prob_mass/kb', f'{label}_prob_mass_kb'),
    ]:
        wt = pr_df[pr_df['condition'] == 'WT'][col]
        ko = pr_df[pr_df['condition'] == 'KO'][col]
        wt_med = wt.median()
        ko_med = ko.median()
        ratio = ko_med / wt_med if wt_med > 0 else float('nan')
        _, p = stats.mannwhitneyu(wt, ko, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        print(f"  {label:>12s} {metric_name:>20s} {wt_med:8.3f} {ko_med:8.3f} {ratio:7.3f}x {p:12.2e} {sig:>4s}")
        results.append({
            'motif_set': label, 'metric': metric_name,
            'wt_median': round(wt_med, 4), 'ko_median': round(ko_med, 4),
            'ko_wt_ratio': round(ratio, 4) if not np.isnan(ratio) else 'nan',
            'mwu_p': p, 'sig': sig,
        })

###############################################################################
# Per-motif WT vs KO (site-level ML shift)
###############################################################################
print("\n" + "=" * 70)
print("Per-motif WT vs KO ML value shift (site-level)")
print("=" * 70)

print(f"  {'Motif':>8s} {'WT mean':>8s} {'KO mean':>8s} {'Δ':>8s} {'MWU P':>12s} {'orig6':>6s}")
print("  " + "-" * 55)

motif_results = []
for motif in motif_counts.head(18).index:
    wt_ml = sites_df[(sites_df['condition'] == 'WT') & (sites_df['motif'] == motif)]['ml_value']
    ko_ml = sites_df[(sites_df['condition'] == 'KO') & (sites_df['motif'] == motif)]['ml_value']
    if len(wt_ml) < 10 or len(ko_ml) < 10:
        continue
    _, p = stats.mannwhitneyu(wt_ml, ko_ml, alternative='two-sided')
    delta = ko_ml.mean() - wt_ml.mean()
    tag = "yes" if motif in ORIGINAL_6 else "no"
    sig = '*' if p < 0.05 else ''
    print(f"  {motif:>8s} {wt_ml.mean():8.1f} {ko_ml.mean():8.1f} {delta:+7.1f} {p:12.2e} {tag:>6s} {sig}")
    motif_results.append({
        'motif': motif, 'is_orig6': motif in ORIGINAL_6,
        'wt_mean_ml': round(wt_ml.mean(), 1), 'ko_mean_ml': round(ko_ml.mean(), 1),
        'delta': round(delta, 2), 'mwu_p': p,
        'wt_n': len(wt_ml), 'ko_n': len(ko_ml),
    })

###############################################################################
# Also do control reads
###############################################################################
print("\n" + "=" * 70)
print("Control reads — same analysis")
print("=" * 70)

ctrl_sites = []
ctrl_reads = {}
for sample, condition in SAMPLES.items():
    bam_path = MAFIA_DIR / f'{sample}_ctrl' / 'mAFiA.reads.bam'
    print(f"  {sample}_ctrl ({condition})...", end=' ', flush=True)
    rd, sd = parse_bam_ref_motif(str(bam_path), ref)
    for s in sd:
        s['sample'] = sample
        s['condition'] = condition
    ctrl_sites.extend(sd)
    ctrl_reads[sample] = rd
    print(f"{len(rd)} reads, {len(sd)} sites")

ctrl_df = pd.DataFrame(ctrl_sites)
ctrl_orig6_n = ctrl_df['is_orig6'].sum()
print(f"\n  Control orig6: {ctrl_orig6_n:,} ({100*ctrl_orig6_n/len(ctrl_df):.1f}%)")

# Per-read for control
ctrl_pr = []
for sample, condition in SAMPLES.items():
    reads = ctrl_reads[sample]
    sample_sites = ctrl_df[ctrl_df['sample'] == sample]
    for rd in reads:
        rid = rd['read_id']
        rlen = rd['read_length']
        rkb = rlen / 1000
        rd_sites = sample_sites[sample_sites['read_id'] == rid]
        rd_orig6 = rd_sites[rd_sites['is_orig6']]
        row = {'read_id': rid, 'read_length': rlen, 'condition': condition}
        for label, subset in [('all', rd_sites), ('orig6', rd_orig6)]:
            for thr in [128, 204]:
                n = (subset['ml_value'] >= thr).sum() if len(subset) > 0 else 0
                row[f'{label}_per_kb_{thr}'] = n / rkb if rkb > 0 else 0
            n_mod = ((subset['ml_value'] >= 128) & (subset['ml_value'] < 204)).sum() if len(subset) > 0 else 0
            row[f'{label}_moderate_kb'] = n_mod / rkb if rkb > 0 else 0
            if len(subset) > 0:
                row[f'{label}_prob_mass_kb'] = subset['ml_value'].sum() / 255.0 / rkb if rkb > 0 else 0
            else:
                row[f'{label}_prob_mass_kb'] = 0
        ctrl_pr.append(row)

ctrl_pr_df = pd.DataFrame(ctrl_pr)

print(f"\n  Control WT vs KO:")
for label in ['all', 'orig6']:
    for metric_name, col in [
        ('sites/kb (>=128)', f'{label}_per_kb_128'),
        ('sites/kb (>=204)', f'{label}_per_kb_204'),
        ('moderate/kb', f'{label}_moderate_kb'),
        ('prob_mass/kb', f'{label}_prob_mass_kb'),
    ]:
        wt = ctrl_pr_df[ctrl_pr_df['condition'] == 'WT'][col]
        ko = ctrl_pr_df[ctrl_pr_df['condition'] == 'KO'][col]
        wt_med = wt.median()
        ko_med = ko.median()
        ratio = ko_med / wt_med if wt_med > 0 else float('nan')
        _, p = stats.mannwhitneyu(wt, ko, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"    {label:>8s} {metric_name:>20s}: WT {wt_med:.3f} KO {ko_med:.3f} = {ratio:.3f}x P={p:.2e} {sig}")

###############################################################################
# Save
###############################################################################
pd.DataFrame(results).to_csv(OUTDIR / '6motif_comparison.tsv', sep='\t', index=False)
pd.DataFrame(motif_results).to_csv(OUTDIR / 'per_motif_ml_shift.tsv', sep='\t', index=False)

ref.close()
print(f"\nSaved to {OUTDIR}/")
print("\nDONE.")
