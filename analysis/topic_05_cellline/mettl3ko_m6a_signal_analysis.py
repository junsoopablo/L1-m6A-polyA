#!/usr/bin/env python3
"""
Explore multiple metrics to demonstrate METTL3 KO effect on L1 m6A.

Problem: At ML>=204 (0.80), L1 m6A/kb shows no significant change (0.98x, ns).
         At ML>=128 (0.50), L1 m6A/kb drops 0.93x (P=0.034).
         -> METTL3-dependent sites may concentrate in moderate probability range.

Approaches:
  1. Raw ML value distribution: WT vs KO (all individual site ML values)
  2. Mean/median ML per read (continuous, no threshold)
  3. ML probability band analysis (which band decreases in KO?)
  4. Multiple threshold sweep with proper stats
  5. "METTL3-sensitive" fraction: moderate (128-204) vs high (>204) ratio
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

SAMPLES = {
    'WT_rep1': 'WT', 'WT_rep2': 'WT', 'WT_rep3': 'WT',
    'KO_rep1': 'KO', 'KO_rep2': 'KO', 'KO_rep3': 'KO',
}

OUTDIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/mettl3ko_m6a_signal')
OUTDIR.mkdir(exist_ok=True)

###############################################################################
# Parse MAFIA BAMs — extract ALL individual m6A ML values per read
###############################################################################
def parse_bam_ml_values(bam_path):
    """Extract per-site m6A ML values from MAFIA BAM."""
    if not os.path.exists(bam_path):
        return [], []
    bam = pysam.AlignmentFile(bam_path, 'rb')
    read_rows = []
    site_ml_values = []  # all individual site ML values

    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        rlen = read.query_alignment_length
        if rlen is None or rlen < 100:
            continue

        mm_tag = ml_tag = None
        for t in ['MM', 'Mm']:
            if read.has_tag(t):
                mm_tag = read.get_tag(t); break
        for t in ['ML', 'Ml']:
            if read.has_tag(t):
                ml_tag = read.get_tag(t); break

        if mm_tag is None or ml_tag is None:
            read_rows.append({
                'read_id': read.query_name, 'read_length': rlen,
                'ml_values': [], 'n_candidate_sites': 0
            })
            continue

        # Extract m6A ML values (chemodCode 21891)
        ml_list = list(ml_tag)
        entries = mm_tag.rstrip(';').split(';')
        idx = 0
        m6a_mls = []
        for entry in entries:
            parts = entry.strip().split(',')
            base_mod = parts[0]
            skip_counts = [int(x) for x in parts[1:]] if len(parts) > 1 else []
            n_sites = len(skip_counts)
            if '21891' in base_mod:
                for i in range(n_sites):
                    if idx + i < len(ml_list):
                        m6a_mls.append(ml_list[idx + i])
            idx += n_sites

        read_rows.append({
            'read_id': read.query_name, 'read_length': rlen,
            'ml_values': m6a_mls, 'n_candidate_sites': len(m6a_mls)
        })
        site_ml_values.extend(m6a_mls)

    bam.close()
    return read_rows, site_ml_values

###############################################################################
# Parse all samples
###############################################################################
print("=" * 70)
print("Parsing MAFIA BAMs — extracting ALL m6A ML values")
print("=" * 70)

all_reads = []
all_site_mls = {'WT': [], 'KO': []}

for sample, condition in SAMPLES.items():
    bam_path = MAFIA_DIR / sample / 'mAFiA.reads.bam'
    print(f"  {sample} ({condition})...", end=' ', flush=True)
    read_rows, site_mls = parse_bam_ml_values(str(bam_path))
    for r in read_rows:
        r['sample'] = sample
        r['condition'] = condition
    all_reads.extend(read_rows)
    all_site_mls[condition].extend(site_mls)
    print(f"{len(read_rows)} reads, {len(site_mls)} m6A candidate sites")

###############################################################################
# Approach 1: Raw ML value distribution (all individual sites)
###############################################################################
print("\n" + "=" * 70)
print("Approach 1: Raw ML value distribution (per-site)")
print("=" * 70)

wt_mls = np.array(all_site_mls['WT'])
ko_mls = np.array(all_site_mls['KO'])
print(f"  WT sites: {len(wt_mls):,}  mean ML = {wt_mls.mean():.1f}  median = {np.median(wt_mls):.0f}")
print(f"  KO sites: {len(ko_mls):,}  mean ML = {ko_mls.mean():.1f}  median = {np.median(ko_mls):.0f}")

ks_stat, ks_p = stats.ks_2samp(wt_mls, ko_mls)
mwu_stat, mwu_p = stats.mannwhitneyu(wt_mls, ko_mls, alternative='two-sided')
print(f"  KS test: D={ks_stat:.4f}, P={ks_p:.2e}")
print(f"  MWU test: P={mwu_p:.2e}")
print(f"  Mean shift: {ko_mls.mean() - wt_mls.mean():.2f} (KO - WT)")

# Distribution in probability bands
print("\n  ML probability bands:")
bands = [(0, 50), (50, 100), (100, 128), (128, 153), (153, 178), (178, 204), (204, 229), (229, 256)]
print(f"  {'Band':>12s}  {'WT count':>10s} {'WT %':>8s}  {'KO count':>10s} {'KO %':>8s}  {'KO/WT':>8s}")
print("  " + "-" * 65)
band_results = []
for lo, hi in bands:
    wt_n = np.sum((wt_mls >= lo) & (wt_mls < hi))
    ko_n = np.sum((ko_mls >= lo) & (ko_mls < hi))
    wt_pct = 100 * wt_n / len(wt_mls) if len(wt_mls) > 0 else 0
    ko_pct = 100 * ko_n / len(ko_mls) if len(ko_mls) > 0 else 0
    ratio = ko_pct / wt_pct if wt_pct > 0 else float('nan')
    print(f"  ML {lo:3d}-{hi:3d}  {wt_n:10d} {wt_pct:7.1f}%  {ko_n:10d} {ko_pct:7.1f}%  {ratio:7.2f}x")
    band_results.append({
        'band': f'{lo}-{hi}', 'wt_count': wt_n, 'wt_pct': round(wt_pct, 2),
        'ko_count': ko_n, 'ko_pct': round(ko_pct, 2), 'ko_wt_ratio': round(ratio, 3)
    })

pd.DataFrame(band_results).to_csv(OUTDIR / 'ml_band_distribution.tsv', sep='\t', index=False)

###############################################################################
# Approach 2: Per-read mean ML value (continuous metric)
###############################################################################
print("\n" + "=" * 70)
print("Approach 2: Per-read mean ML value (continuous, no threshold)")
print("=" * 70)

read_df = pd.DataFrame([{
    'read_id': r['read_id'],
    'read_length': r['read_length'],
    'sample': r['sample'],
    'condition': r['condition'],
    'n_candidate_sites': r['n_candidate_sites'],
    'mean_ml': np.mean(r['ml_values']) if len(r['ml_values']) > 0 else np.nan,
    'median_ml': np.median(r['ml_values']) if len(r['ml_values']) > 0 else np.nan,
    'max_ml': max(r['ml_values']) if len(r['ml_values']) > 0 else np.nan,
    # Counts at different thresholds
    'sites_128': sum(1 for v in r['ml_values'] if v >= 128),
    'sites_153': sum(1 for v in r['ml_values'] if v >= 153),
    'sites_178': sum(1 for v in r['ml_values'] if v >= 178),
    'sites_204': sum(1 for v in r['ml_values'] if v >= 204),
    'sites_229': sum(1 for v in r['ml_values'] if v >= 229),
    # Moderate band (METTL3-sensitive?)
    'sites_128_204': sum(1 for v in r['ml_values'] if 128 <= v < 204),
} for r in all_reads])

# Only reads with at least 1 candidate site
has_sites = read_df[read_df['n_candidate_sites'] > 0].copy()
has_sites['read_kb'] = has_sites['read_length'] / 1000

for metric in ['mean_ml', 'median_ml']:
    wt_vals = has_sites[has_sites['condition'] == 'WT'][metric].dropna()
    ko_vals = has_sites[has_sites['condition'] == 'KO'][metric].dropna()
    u_stat, u_p = stats.mannwhitneyu(wt_vals, ko_vals, alternative='two-sided')
    print(f"  {metric}: WT {wt_vals.median():.1f} vs KO {ko_vals.median():.1f}  "
          f"(Δ={ko_vals.median() - wt_vals.median():.1f}, MWU P={u_p:.2e})")

###############################################################################
# Approach 3: Multi-threshold m6A/kb sweep
###############################################################################
print("\n" + "=" * 70)
print("Approach 3: Threshold sweep — m6A/kb at each threshold")
print("=" * 70)

thresholds = [50, 80, 100, 128, 140, 153, 165, 178, 191, 204, 217, 229, 242]
print(f"  {'Threshold':>10s}  {'WT median':>10s} {'KO median':>10s} {'KO/WT':>8s} {'MWU P':>12s} {'sig':>4s}")
print("  " + "-" * 60)

sweep_results = []
for thr in thresholds:
    col = f'sites_{thr}' if f'sites_{thr}' in read_df.columns else None
    if col is None:
        # Compute from raw data
        read_df[f'sites_{thr}'] = [
            sum(1 for v in r['ml_values'] if v >= thr) for r in all_reads
        ]
    read_df[f'm6a_kb_{thr}'] = read_df[f'sites_{thr}'] / (read_df['read_length'] / 1000)

    wt = read_df[read_df['condition'] == 'WT'][f'm6a_kb_{thr}']
    ko = read_df[read_df['condition'] == 'KO'][f'm6a_kb_{thr}']

    wt_med = wt.median()
    ko_med = ko.median()
    ratio = ko_med / wt_med if wt_med > 0 else float('nan')
    _, p = stats.mannwhitneyu(wt, ko, alternative='two-sided')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

    print(f"  ML>={thr:3d}     {wt_med:10.3f} {ko_med:10.3f} {ratio:7.3f}x {p:12.2e} {sig:>4s}")
    sweep_results.append({
        'threshold': thr, 'prob': round(thr / 255, 3),
        'wt_median': round(wt_med, 4), 'ko_median': round(ko_med, 4),
        'ko_wt_ratio': round(ratio, 4), 'mwu_p': p, 'sig': sig,
        'wt_n': len(wt), 'ko_n': len(ko)
    })

pd.DataFrame(sweep_results).to_csv(OUTDIR / 'threshold_sweep_l1.tsv', sep='\t', index=False)

###############################################################################
# Approach 4: "METTL3-sensitive" fraction (moderate ML sites)
###############################################################################
print("\n" + "=" * 70)
print("Approach 4: METTL3-sensitive fraction (ML 128-204)")
print("=" * 70)

read_df['moderate_sites'] = read_df['sites_128_204']
read_df['high_sites'] = read_df['sites_204']
read_df['moderate_per_kb'] = read_df['moderate_sites'] / (read_df['read_length'] / 1000)
read_df['high_per_kb'] = read_df['high_sites'] / (read_df['read_length'] / 1000)

# Moderate sites per kb
wt_mod = read_df[read_df['condition'] == 'WT']['moderate_per_kb']
ko_mod = read_df[read_df['condition'] == 'KO']['moderate_per_kb']
_, p_mod = stats.mannwhitneyu(wt_mod, ko_mod, alternative='two-sided')
print(f"  Moderate (128-204) sites/kb: WT {wt_mod.median():.3f} vs KO {ko_mod.median():.3f}  "
      f"(KO/WT={ko_mod.median()/wt_mod.median():.3f}x, P={p_mod:.2e})")

# High sites per kb
wt_hi = read_df[read_df['condition'] == 'WT']['high_per_kb']
ko_hi = read_df[read_df['condition'] == 'KO']['high_per_kb']
_, p_hi = stats.mannwhitneyu(wt_hi, ko_hi, alternative='two-sided')
print(f"  High (>=204) sites/kb:       WT {wt_hi.median():.3f} vs KO {ko_hi.median():.3f}  "
      f"(KO/WT={ko_hi.median()/wt_hi.median():.3f}x, P={p_hi:.2e})")

# Ratio of moderate to total
read_df['mod_fraction'] = read_df['moderate_sites'] / (read_df['sites_128'] + 1e-10)
has_m6a = read_df[read_df['sites_128'] > 0].copy()
wt_frac = has_m6a[has_m6a['condition'] == 'WT']['mod_fraction']
ko_frac = has_m6a[has_m6a['condition'] == 'KO']['mod_fraction']
_, p_frac = stats.mannwhitneyu(wt_frac, ko_frac, alternative='two-sided')
print(f"  Moderate fraction of total:  WT {wt_frac.median():.3f} vs KO {ko_frac.median():.3f}  "
      f"(P={p_frac:.2e})")

###############################################################################
# Approach 5: Per-replicate analysis (paired)
###############################################################################
print("\n" + "=" * 70)
print("Approach 5: Per-replicate median m6A/kb (for paired-like view)")
print("=" * 70)

for thr in [128, 153, 178, 204]:
    col = f'm6a_kb_{thr}'
    print(f"\n  ML>={thr} ({thr/255:.2f}):")
    rep_data = read_df.groupby(['sample', 'condition'])[col].median().reset_index()
    for _, row in rep_data.iterrows():
        print(f"    {row['sample']:12s}: {row[col]:.3f} m6A/kb")

###############################################################################
# Approach 6: Total m6A "probability mass" per kb (sum of ML values / 255)
###############################################################################
print("\n" + "=" * 70)
print("Approach 6: m6A probability mass per kb (sum of ML/255)")
print("=" * 70)

read_df['prob_mass'] = [
    sum(v / 255.0 for v in r['ml_values']) for r in all_reads
]
read_df['prob_mass_per_kb'] = read_df['prob_mass'] / (read_df['read_length'] / 1000)

wt_pm = read_df[read_df['condition'] == 'WT']['prob_mass_per_kb']
ko_pm = read_df[read_df['condition'] == 'KO']['prob_mass_per_kb']
_, p_pm = stats.mannwhitneyu(wt_pm, ko_pm, alternative='two-sided')
print(f"  Probability mass/kb: WT {wt_pm.median():.3f} vs KO {ko_pm.median():.3f}  "
      f"(KO/WT={ko_pm.median()/wt_pm.median():.3f}x, P={p_pm:.2e})")

# Also try sum of ML >= 128 only
read_df['prob_mass_128'] = [
    sum(v / 255.0 for v in r['ml_values'] if v >= 128) for r in all_reads
]
read_df['prob_mass_128_per_kb'] = read_df['prob_mass_128'] / (read_df['read_length'] / 1000)

wt_pm128 = read_df[read_df['condition'] == 'WT']['prob_mass_128_per_kb']
ko_pm128 = read_df[read_df['condition'] == 'KO']['prob_mass_128_per_kb']
_, p_pm128 = stats.mannwhitneyu(wt_pm128, ko_pm128, alternative='two-sided')
print(f"  Prob mass/kb (ML>=128): WT {wt_pm128.median():.3f} vs KO {ko_pm128.median():.3f}  "
      f"(KO/WT={ko_pm128.median()/wt_pm128.median():.3f}x, P={p_pm128:.2e})")

###############################################################################
# Also do control reads for comparison
###############################################################################
print("\n" + "=" * 70)
print("Control reads — same analysis for comparison")
print("=" * 70)

ctrl_reads = []
ctrl_site_mls = {'WT': [], 'KO': []}

for sample, condition in SAMPLES.items():
    bam_path = MAFIA_DIR / f'{sample}_ctrl' / 'mAFiA.reads.bam'
    print(f"  {sample}_ctrl ({condition})...", end=' ', flush=True)
    read_rows, site_mls = parse_bam_ml_values(str(bam_path))
    for r in read_rows:
        r['sample'] = sample
        r['condition'] = condition
    ctrl_reads.extend(read_rows)
    ctrl_site_mls[condition].extend(site_mls)
    print(f"{len(read_rows)} reads, {len(site_mls)} sites")

ctrl_wt_mls = np.array(ctrl_site_mls['WT'])
ctrl_ko_mls = np.array(ctrl_site_mls['KO'])
ks_ctrl, ks_p_ctrl = stats.ks_2samp(ctrl_wt_mls, ctrl_ko_mls)
print(f"\n  Control ML distribution: WT mean={ctrl_wt_mls.mean():.1f} vs KO mean={ctrl_ko_mls.mean():.1f}")
print(f"  KS test: D={ks_ctrl:.4f}, P={ks_p_ctrl:.2e}")

# Threshold sweep for control
ctrl_df = pd.DataFrame([{
    'read_id': r['read_id'],
    'read_length': r['read_length'],
    'condition': r['condition'],
    'sites_128': sum(1 for v in r['ml_values'] if v >= 128),
    'sites_204': sum(1 for v in r['ml_values'] if v >= 204),
    'sites_128_204': sum(1 for v in r['ml_values'] if 128 <= v < 204),
    'prob_mass': sum(v / 255.0 for v in r['ml_values']),
} for r in ctrl_reads])

ctrl_df['m6a_kb_128'] = ctrl_df['sites_128'] / (ctrl_df['read_length'] / 1000)
ctrl_df['m6a_kb_204'] = ctrl_df['sites_204'] / (ctrl_df['read_length'] / 1000)
ctrl_df['moderate_per_kb'] = ctrl_df['sites_128_204'] / (ctrl_df['read_length'] / 1000)
ctrl_df['prob_mass_per_kb'] = ctrl_df['prob_mass'] / (ctrl_df['read_length'] / 1000)

print("\n  Control threshold comparison:")
for thr, col in [(128, 'm6a_kb_128'), (204, 'm6a_kb_204')]:
    wt_c = ctrl_df[ctrl_df['condition'] == 'WT'][col]
    ko_c = ctrl_df[ctrl_df['condition'] == 'KO'][col]
    _, p_c = stats.mannwhitneyu(wt_c, ko_c, alternative='two-sided')
    ratio_c = ko_c.median() / wt_c.median() if wt_c.median() > 0 else float('nan')
    print(f"    ML>={thr}: WT {wt_c.median():.3f} vs KO {ko_c.median():.3f} (KO/WT={ratio_c:.3f}x, P={p_c:.2e})")

# Moderate sites for control
wt_cm = ctrl_df[ctrl_df['condition'] == 'WT']['moderate_per_kb']
ko_cm = ctrl_df[ctrl_df['condition'] == 'KO']['moderate_per_kb']
_, p_cm = stats.mannwhitneyu(wt_cm, ko_cm, alternative='two-sided')
print(f"    Moderate (128-204)/kb: WT {wt_cm.median():.3f} vs KO {ko_cm.median():.3f} (KO/WT={ko_cm.median()/wt_cm.median():.3f}x, P={p_cm:.2e})")

# Prob mass
wt_cpm = ctrl_df[ctrl_df['condition'] == 'WT']['prob_mass_per_kb']
ko_cpm = ctrl_df[ctrl_df['condition'] == 'KO']['prob_mass_per_kb']
_, p_cpm = stats.mannwhitneyu(wt_cpm, ko_cpm, alternative='two-sided')
print(f"    Prob mass/kb: WT {wt_cpm.median():.3f} vs KO {ko_cpm.median():.3f} (KO/WT={ko_cpm.median()/wt_cpm.median():.3f}x, P={p_cpm:.2e})")

###############################################################################
# Save comprehensive per-read data
###############################################################################
save_cols = ['read_id', 'read_length', 'sample', 'condition', 'n_candidate_sites',
             'mean_ml', 'median_ml', 'sites_128', 'sites_204', 'sites_128_204',
             'moderate_per_kb', 'high_per_kb', 'prob_mass_per_kb']
read_df[save_cols].to_csv(OUTDIR / 'l1_per_read_ml_analysis.tsv', sep='\t', index=False)
print(f"\nSaved to {OUTDIR}/")

print("\nDONE.")
