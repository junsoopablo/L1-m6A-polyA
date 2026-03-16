#!/usr/bin/env python3
"""
Extended classifier testing — multiple GENCODE-independent approaches
to separate autonomous L1 from host gene read-through.

Approaches tested:
A. Intronic/intergenic (gene-based but simpler than exon overlap)
B. Multi-TE / has_non_l1_te (structural, GENCODE-free)
C. flank_frac threshold sweep (structural, GENCODE-free)
D. dist_to_3prime threshold sweep (structural, GENCODE-free)
E. Supervised logistic regression on structural features
F. ChromHMM × structural interactions
G. PAS motif presence per L1 element (GENCODE-free, genome-based)
H. Combined multi-feature scoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
OUTDIR = TOPIC_07 / 'unified_classifier'

# =========================================================================
# Load data
# =========================================================================
print("Loading unified dataset with ChromHMM...")
df = pd.read_csv(OUTDIR / 'unified_hela_ars_chromhmm.tsv', sep='\t')
print(f"  Total reads: {len(df)}")
print(f"  PASS: {(df['source']=='PASS').sum()}, CatB: {(df['source']=='CatB').sum()}")
print(f"  HeLa: {(df['cell_line']=='HeLa').sum()}, HeLa-Ars: {(df['cell_line']=='HeLa-Ars').sum()}")

# Helper
def ars_delta(data):
    """Compute arsenite poly(A) delta (Ars median - HeLa median)."""
    hela = data[data['cell_line'] == 'HeLa']['polya_length'].dropna()
    ars = data[data['cell_line'] == 'HeLa-Ars']['polya_length'].dropna()
    if len(hela) < 5 or len(ars) < 5:
        return np.nan, np.nan, len(hela) + len(ars)
    _, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
    return ars.median() - hela.median(), p, len(hela) + len(ars)

def classifier_summary(df, mask_grp1, mask_grp2, name_grp1='Grp1', name_grp2='Grp2'):
    """Compute |ΔΔ| for a binary classifier."""
    d1, p1, n1 = ars_delta(df[mask_grp1])
    d2, p2, n2 = ars_delta(df[mask_grp2])
    dd = abs(d1 - d2) if not (np.isnan(d1) or np.isnan(d2)) else np.nan
    return d1, p1, n1, d2, p2, n2, dd

# =========================================================================
# Reference: Current GENCODE-based classifier
# =========================================================================
print("\n" + "=" * 90)
print("REFERENCE: Current GENCODE-based PASS vs Cat B")
print("=" * 90)
ref = classifier_summary(df, df['source'] == 'PASS', df['source'] == 'CatB', 'PASS', 'CatB')
print(f"  PASS: Δ={ref[0]:+.1f} (p={ref[1]:.2e}, n={ref[2]})")
print(f"  CatB: Δ={ref[3]:+.1f} (p={ref[4]:.2e}, n={ref[5]})")
print(f"  |ΔΔ| = {ref[6]:.1f}")

results = []
results.append(('GENCODE PASS/CatB', ref[0], ref[2], ref[3], ref[5], ref[6]))

# =========================================================================
# Approach A: Intronic vs Intergenic
# =========================================================================
print("\n" + "=" * 90)
print("APPROACH A: Intronic vs Intergenic (gene-annotation-based)")
print("=" * 90)

for source_label, sub in [('All', df), ('PASS only', df[df['source']=='PASS']), ('CatB only', df[df['source']=='CatB'])]:
    intronic = sub[sub['TE_group'] == 'intronic']
    intergenic = sub[sub['TE_group'] == 'intergenic']
    d_intr, p_intr, n_intr = ars_delta(intronic)
    d_intg, p_intg, n_intg = ars_delta(intergenic)
    dd = abs(d_intr - d_intg) if not (np.isnan(d_intr) or np.isnan(d_intg)) else np.nan
    print(f"\n  {source_label}:")
    print(f"    Intronic:   Δ={d_intr:+.1f}, p={p_intr:.2e}, n={n_intr}")
    print(f"    Intergenic: Δ={d_intg:+.1f}, p={p_intg:.2e}, n={n_intg}")
    print(f"    |ΔΔ| = {dd:.1f}")

# As classifier (intergenic = "autonomous-like")
r = classifier_summary(df, df['TE_group'] == 'intergenic', df['TE_group'] == 'intronic')
results.append(('Intergenic vs Intronic', r[0], r[2], r[3], r[5], r[6]))

# =========================================================================
# Approach B: Multi-TE / has_non_l1_te
# =========================================================================
print("\n" + "=" * 90)
print("APPROACH B: Multi-TE families / has_non_l1_te")
print("=" * 90)

for feature, label in [('has_non_l1_te', 'Has non-L1 TE'), ('n_te_families', 'n_TE_families > 1')]:
    if feature == 'has_non_l1_te':
        mask_yes = df[feature] == True
        mask_no = df[feature] == False
    else:
        mask_yes = df[feature] > 1
        mask_no = df[feature] <= 1

    d_yes, p_yes, n_yes = ars_delta(df[mask_yes])
    d_no, p_no, n_no = ars_delta(df[mask_no])
    dd = abs(d_yes - d_no) if not (np.isnan(d_yes) or np.isnan(d_no)) else np.nan

    print(f"\n  {label}:")
    print(f"    Yes: Δ={d_yes:+.1f}, n={n_yes}")
    print(f"    No:  Δ={d_no:+.1f}, n={n_no}")
    print(f"    |ΔΔ| = {dd:.1f}")
    results.append((label, d_no, n_no, d_yes, n_yes, dd))

# =========================================================================
# Approach C: flank_frac threshold sweep
# =========================================================================
print("\n" + "=" * 90)
print("APPROACH C: flank_frac threshold sweep")
print("=" * 90)

print(f"\n  {'Threshold':>10s} | {'Low Δ':>8s} {'n':>6s} | {'High Δ':>8s} {'n':>6s} | {'|ΔΔ|':>6s}")
print("  " + "-" * 60)

best_ff_dd = 0
best_ff_thr = 0.1
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    mask_low = df['flank_frac'] <= thr
    mask_high = df['flank_frac'] > thr
    d_lo, _, n_lo = ars_delta(df[mask_low])
    d_hi, _, n_hi = ars_delta(df[mask_high])
    dd = abs(d_lo - d_hi) if not (np.isnan(d_lo) or np.isnan(d_hi)) else np.nan
    star = ' <<<' if dd and dd > best_ff_dd else ''
    if dd and dd > best_ff_dd:
        best_ff_dd = dd
        best_ff_thr = thr
    d_lo_s = f"{d_lo:+.1f}" if not np.isnan(d_lo) else 'n/a'
    d_hi_s = f"{d_hi:+.1f}" if not np.isnan(d_hi) else 'n/a'
    dd_s = f"{dd:.1f}" if dd and not np.isnan(dd) else 'n/a'
    print(f"  {thr:10.2f} | {d_lo_s:>8s} {n_lo:6d} | {d_hi_s:>8s} {n_hi:6d} | {dd_s:>6s}{star}")

r = classifier_summary(df, df['flank_frac'] <= best_ff_thr, df['flank_frac'] > best_ff_thr)
results.append((f'flank_frac ≤ {best_ff_thr}', r[0], r[2], r[3], r[5], r[6]))

# =========================================================================
# Approach D: dist_to_3prime threshold sweep
# =========================================================================
print("\n" + "=" * 90)
print("APPROACH D: dist_to_3prime threshold sweep")
print("=" * 90)

print(f"\n  {'Threshold':>10s} | {'Close Δ':>8s} {'n':>6s} | {'Far Δ':>8s} {'n':>6s} | {'|ΔΔ|':>6s}")
print("  " + "-" * 60)

best_d3p_dd = 0
best_d3p_thr = 100
for thr in [0, 50, 100, 200, 500, 1000, 2000, 5000]:
    mask_close = df['dist_to_3prime'] <= thr
    mask_far = df['dist_to_3prime'] > thr
    d_cl, _, n_cl = ars_delta(df[mask_close])
    d_fa, _, n_fa = ars_delta(df[mask_far])
    dd = abs(d_cl - d_fa) if not (np.isnan(d_cl) or np.isnan(d_fa)) else np.nan
    star = ' <<<' if dd and dd > best_d3p_dd else ''
    if dd and dd > best_d3p_dd:
        best_d3p_dd = dd
        best_d3p_thr = thr
    d_cl_s = f"{d_cl:+.1f}" if not np.isnan(d_cl) else 'n/a'
    d_fa_s = f"{d_fa:+.1f}" if not np.isnan(d_fa) else 'n/a'
    dd_s = f"{dd:.1f}" if dd and not np.isnan(dd) else 'n/a'
    print(f"  {thr:10d} | {d_cl_s:>8s} {n_cl:6d} | {d_fa_s:>8s} {n_fa:6d} | {dd_s:>6s}{star}")

r = classifier_summary(df, df['dist_to_3prime'] <= best_d3p_thr, df['dist_to_3prime'] > best_d3p_thr)
results.append((f'dist_to_3prime ≤ {best_d3p_thr}', r[0], r[2], r[3], r[5], r[6]))

# =========================================================================
# Approach E: Supervised logistic regression
# =========================================================================
print("\n" + "=" * 90)
print("APPROACH E: Supervised logistic regression (predict Cat B from structural features)")
print("=" * 90)

# Train logistic on structural features to predict Cat B
features = ['overlap_frac', 'flank_frac', 'dist_to_3prime', 'n_te_families']
df_train = df.dropna(subset=features + ['polya_length']).copy()
df_train['is_catb'] = (df_train['source'] == 'CatB').astype(int)
df_train['is_intronic'] = (df_train['TE_group'] == 'intronic').astype(int)

# Model 1: Structural only (GENCODE-free)
X1 = df_train[features].values
scaler1 = StandardScaler()
X1s = scaler1.fit_transform(X1)
y = df_train['is_catb'].values

lr1 = LogisticRegression(random_state=42, max_iter=1000)
# Cross-validated prediction to avoid overfitting
df_train['lr_prob_struct'] = cross_val_predict(lr1, X1s, y, cv=5, method='predict_proba')[:, 1]

# Model 2: Structural + intronic + ChromHMM
df_train['is_transcribed'] = (df_train['chromhmm_group'] == 'Transcribed').astype(int)
features2 = features + ['is_intronic', 'is_transcribed']
X2 = df_train[features2].values
scaler2 = StandardScaler()
X2s = scaler2.fit_transform(X2)

lr2 = LogisticRegression(random_state=42, max_iter=1000)
df_train['lr_prob_full'] = cross_val_predict(lr2, X2s, y, cv=5, method='predict_proba')[:, 1]

# Fit to get coefficients
lr1.fit(X1s, y)
lr2.fit(X2s, y)

print("\n  Logistic Regression Coefficients (structural only):")
for f, c in zip(features, lr1.coef_[0]):
    print(f"    {f:25s}: {c:+.3f}")

print(f"\n  Logistic Regression Coefficients (full):")
for f, c in zip(features2, lr2.coef_[0]):
    print(f"    {f:25s}: {c:+.3f}")

# Test as classifier at various probability thresholds
print(f"\n  Cross-validated probability as classifier:")
print(f"  {'Model':35s} {'Threshold':>10s} | {'Low Δ':>8s} {'n':>6s} | {'High Δ':>8s} {'n':>6s} | {'|ΔΔ|':>6s}")
print("  " + "-" * 80)

best_lr_dd = 0
best_lr_model = ''
best_lr_thr = 0.5

for model_name, prob_col in [('Structural only', 'lr_prob_struct'),
                              ('Struct + intronic + ChromHMM', 'lr_prob_full')]:
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        mask_lo = df_train[prob_col] <= thr
        mask_hi = df_train[prob_col] > thr
        d_lo, _, n_lo = ars_delta(df_train[mask_lo])
        d_hi, _, n_hi = ars_delta(df_train[mask_hi])
        dd = abs(d_lo - d_hi) if not (np.isnan(d_lo) or np.isnan(d_hi)) else np.nan
        star = ' <<<' if dd and dd > best_lr_dd else ''
        if dd and dd > best_lr_dd:
            best_lr_dd = dd
            best_lr_model = model_name
            best_lr_thr = thr
        d_lo_s = f"{d_lo:+.1f}" if not np.isnan(d_lo) else 'n/a'
        d_hi_s = f"{d_hi:+.1f}" if not np.isnan(d_hi) else 'n/a'
        dd_s = f"{dd:.1f}" if dd and not np.isnan(dd) else 'n/a'
        print(f"  {model_name:35s} {thr:10.1f} | {d_lo_s:>8s} {n_lo:6d} | {d_hi_s:>8s} {n_hi:6d} | {dd_s:>6s}{star}")

results.append((f'LR ({best_lr_model[:20]}) p>{best_lr_thr}',
                 ars_delta(df_train[df_train['lr_prob_struct' if 'Struct' in best_lr_model else 'lr_prob_full'] <= best_lr_thr])[0],
                 int((df_train['lr_prob_struct' if 'Struct' in best_lr_model else 'lr_prob_full'] <= best_lr_thr).sum()),
                 ars_delta(df_train[df_train['lr_prob_struct' if 'Struct' in best_lr_model else 'lr_prob_full'] > best_lr_thr])[0],
                 int((df_train['lr_prob_struct' if 'Struct' in best_lr_model else 'lr_prob_full'] > best_lr_thr).sum()),
                 best_lr_dd))

# =========================================================================
# Approach F: ChromHMM × structural interaction (2D classifier)
# =========================================================================
print("\n" + "=" * 90)
print("APPROACH F: 2D classifiers (ChromHMM × structural feature)")
print("=" * 90)

# Test: Non-Transcribed + high overlap = autonomous
# Transcribed + low overlap = read-through
for ov_thr in [0.3, 0.5, 0.7]:
    mask_auto = (df['chromhmm_group'] != 'Transcribed') & (df['overlap_frac'] > ov_thr)
    mask_rt = ~mask_auto
    d_auto, _, n_auto = ars_delta(df[mask_auto])
    d_rt, _, n_rt = ars_delta(df[mask_rt])
    dd = abs(d_auto - d_rt) if not (np.isnan(d_auto) or np.isnan(d_rt)) else np.nan
    dd_s = f"{dd:.1f}" if dd and not np.isnan(dd) else 'n/a'
    print(f"  non-Tx + ov>{ov_thr}: auto Δ={d_auto:+.1f}(n={n_auto}) vs rest Δ={d_rt:+.1f}(n={n_rt}) → |ΔΔ|={dd_s}")

# Test: intergenic + high overlap = most autonomous
for ov_thr in [0.3, 0.5, 0.7]:
    mask_auto = (df['TE_group'] == 'intergenic') & (df['overlap_frac'] > ov_thr)
    mask_rt = ~mask_auto
    d_auto, _, n_auto = ars_delta(df[mask_auto])
    d_rt, _, n_rt = ars_delta(df[mask_rt])
    dd = abs(d_auto - d_rt) if not (np.isnan(d_auto) or np.isnan(d_rt)) else np.nan
    dd_s = f"{dd:.1f}" if dd and not np.isnan(dd) else 'n/a'
    print(f"  intergenic + ov>{ov_thr}: auto Δ={d_auto:+.1f}(n={n_auto}) vs rest Δ={d_rt:+.1f}(n={n_rt}) → |ΔΔ|={dd_s}")

# Test: overlap_frac > threshold + no non-L1 TE = pure autonomous
for ov_thr in [0.3, 0.5, 0.7]:
    mask_auto = (df['overlap_frac'] > ov_thr) & (~df['has_non_l1_te'])
    mask_rt = ~mask_auto
    d_auto, _, n_auto = ars_delta(df[mask_auto])
    d_rt, _, n_rt = ars_delta(df[mask_rt])
    dd = abs(d_auto - d_rt) if not (np.isnan(d_auto) or np.isnan(d_rt)) else np.nan
    dd_s = f"{dd:.1f}" if dd and not np.isnan(dd) else 'n/a'
    print(f"  ov>{ov_thr} + no_non-L1_TE: auto Δ={d_auto:+.1f}(n={n_auto}) vs rest Δ={d_rt:+.1f}(n={n_rt}) → |ΔΔ|={dd_s}")

# =========================================================================
# Approach G: PAS motif presence per L1 element
# =========================================================================
print("\n" + "=" * 90)
print("APPROACH G: PAS motif per L1 element (GENCODE-free, genome-based)")
print("=" * 90)

import pysam
GENOME = PROJECT / 'reference/Human.fasta'

# Build L1 element lookup from RepeatMasker GTF
RMSK_GTF = PROJECT / 'reference/hg38_rmsk_TE.gtf'

# Canonical and variant PAS motifs
PAS_CANONICAL = 'AATAAA'
PAS_VARIANTS = ['ATTAAA', 'AGTAAA', 'TATAAA', 'CATAAA', 'GATAAA',
                'AATATA', 'AATACA', 'AATAGA', 'ACTAAA', 'AAGAAA', 'AATGAA']
ALL_PAS = [PAS_CANONICAL] + PAS_VARIANTS

# Get unique L1 elements from dataset
l1_elements = df[['transcript_id', 'chr']].drop_duplicates()
print(f"  Unique L1 elements in dataset: {len(l1_elements)}")

# Parse RepeatMasker GTF for these elements to get their coordinates and strand
print("  Parsing RepeatMasker GTF for L1 element coordinates...")
l1_ids = set(l1_elements['transcript_id'].values)
element_info = {}

with open(RMSK_GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        attrs = fields[8]
        # Parse transcript_id
        tid = None
        for attr in attrs.split(';'):
            attr = attr.strip()
            if attr.startswith('transcript_id'):
                tid = attr.split('"')[1]
                break
        if tid and tid in l1_ids:
            chrom = fields[0]
            start = int(fields[3]) - 1  # Convert to 0-based
            end = int(fields[4])
            strand = fields[6]
            element_info[tid] = (chrom, start, end, strand)
            if len(element_info) == len(l1_ids):
                break

print(f"  Found coordinates for {len(element_info)}/{len(l1_ids)} elements")

# Scan 3' end 50bp of each L1 element for PAS
print("  Scanning for PAS motifs in L1 3' ends...")
fa = pysam.FastaFile(str(GENOME))

def reverse_complement(seq):
    comp = {'A':'T', 'T':'A', 'G':'C', 'C':'G', 'N':'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))

element_pas = {}  # tid → {'canonical': bool, 'any_pas': bool, 'pas_motif': str}
for tid, (chrom, start, end, strand) in element_info.items():
    # Get 3' end 50bp
    if strand == '+':
        scan_start = max(0, end - 50)
        scan_end = end
    else:
        scan_start = start
        scan_end = min(start + 50, fa.get_reference_length(chrom))

    seq = fa.fetch(chrom, scan_start, scan_end).upper()
    if strand == '-':
        seq = reverse_complement(seq)

    has_canonical = PAS_CANONICAL in seq
    has_any = any(pas in seq for pas in ALL_PAS)

    element_pas[tid] = {'canonical': has_canonical, 'any_pas': has_any}

fa.close()

# Map to reads
df['has_canonical_pas'] = df['transcript_id'].map(lambda x: element_pas.get(x, {}).get('canonical', False))
df['has_any_pas'] = df['transcript_id'].map(lambda x: element_pas.get(x, {}).get('any_pas', False))

# Test PAS as classifier
for pas_type, col in [('Canonical PAS (AATAAA)', 'has_canonical_pas'), ('Any PAS motif', 'has_any_pas')]:
    mask_yes = df[col] == True
    mask_no = df[col] == False
    d_yes, p_yes, n_yes = ars_delta(df[mask_yes])
    d_no, p_no, n_no = ars_delta(df[mask_no])
    dd = abs(d_yes - d_no) if not (np.isnan(d_yes) or np.isnan(d_no)) else np.nan

    # Also check distribution
    pass_pct = df[df['source'] == 'PASS'][col].mean() * 100
    catb_pct = df[df['source'] == 'CatB'][col].mean() * 100

    print(f"\n  {pas_type}:")
    print(f"    PASS prevalence: {pass_pct:.1f}%, CatB prevalence: {catb_pct:.1f}%")
    print(f"    With PAS:    Δ={d_yes:+.1f}, n={n_yes}")
    print(f"    Without PAS: Δ={d_no:+.1f}, n={n_no}")
    print(f"    |ΔΔ| = {dd:.1f}")
    results.append((pas_type, d_yes, n_yes, d_no, n_no, dd))

# PAS × overlap combination
for ov_thr in [0.5, 0.7]:
    mask_auto = (df['has_any_pas'] == True) & (df['overlap_frac'] > ov_thr)
    mask_rt = ~mask_auto
    d_auto, _, n_auto = ars_delta(df[mask_auto])
    d_rt, _, n_rt = ars_delta(df[mask_rt])
    dd = abs(d_auto - d_rt) if not (np.isnan(d_auto) or np.isnan(d_rt)) else np.nan
    dd_s = f"{dd:.1f}" if dd and not np.isnan(dd) else 'n/a'
    print(f"\n  PAS + ov>{ov_thr}: auto Δ={d_auto:+.1f}(n={n_auto}) vs rest Δ={d_rt:+.1f}(n={n_rt}) → |ΔΔ|={dd_s}")

# =========================================================================
# Approach H: Comprehensive OLS comparison
# =========================================================================
print("\n" + "=" * 90)
print("APPROACH H: Comprehensive OLS model comparison")
print("=" * 90)

df_ols = df.dropna(subset=['overlap_frac', 'polya_length', 'chromhmm_group']).copy()
df_ols['is_ars'] = (df_ols['cell_line'] == 'HeLa-Ars').astype(int)
df_ols['is_catb'] = (df_ols['source'] == 'CatB').astype(int)
df_ols['is_intronic'] = (df_ols['TE_group'] == 'intronic').astype(int)
df_ols['is_transcribed'] = (df_ols['chromhmm_group'] == 'Transcribed').astype(int)
df_ols['has_pas'] = df_ols['has_any_pas'].astype(int)
df_ols['has_nonl1'] = df_ols['has_non_l1_te'].astype(int)

models = {
    'M0: ars only':
        'polya_length ~ is_ars',
    'M1: ars × catb (GENCODE)':
        'polya_length ~ is_ars * is_catb',
    'M2: ars × intronic':
        'polya_length ~ is_ars * is_intronic',
    'M3: ars × overlap_frac':
        'polya_length ~ is_ars * overlap_frac',
    'M4: ars × PAS':
        'polya_length ~ is_ars * has_pas',
    'M5: ars × non-L1-TE':
        'polya_length ~ is_ars * has_nonl1',
    'M6: ars × ChromHMM(tx)':
        'polya_length ~ is_ars * is_transcribed',
    'M7: ars × (ov + PAS)':
        'polya_length ~ is_ars * overlap_frac + is_ars * has_pas',
    'M8: ars × (ov + intronic)':
        'polya_length ~ is_ars * overlap_frac + is_ars * is_intronic',
    'M9: ars × (ov + PAS + intronic)':
        'polya_length ~ is_ars * overlap_frac + is_ars * has_pas + is_ars * is_intronic',
    'M10: ars × (ov + PAS + tx)':
        'polya_length ~ is_ars * overlap_frac + is_ars * has_pas + is_ars * is_transcribed',
    'M11: ars × (ov + PAS + nonL1 + tx)':
        'polya_length ~ is_ars * overlap_frac + is_ars * has_pas + is_ars * has_nonl1 + is_ars * is_transcribed',
    'M12: ars × catb + all GENCODE-free':
        'polya_length ~ is_ars * is_catb + is_ars * overlap_frac + is_ars * has_pas + is_ars * is_transcribed',
    'M13: ars × (ov + PAS + intronic + nonL1)':
        'polya_length ~ is_ars * overlap_frac + is_ars * has_pas + is_ars * is_intronic + is_ars * has_nonl1',
}

print(f"\n  {'Model':50s} {'R²':>8s} {'AIC':>10s} {'ΔAIC':>8s}")
print("  " + "-" * 80)

aic_ref = None
model_results = {}
for name, formula in models.items():
    m = smf.ols(formula, data=df_ols).fit()
    model_results[name] = m
    if aic_ref is None:
        aic_ref = m.aic
    daic = m.aic - aic_ref
    print(f"  {name:50s} {m.rsquared:8.4f} {m.aic:10.0f} {daic:+8.0f}")

# Print key interaction coefficients for best GENCODE-free models
print("\n  Key interaction coefficients:")
for name in ['M9: ars × (ov + PAS + intronic)', 'M11: ars × (ov + PAS + nonL1 + tx)',
             'M12: ars × catb + all GENCODE-free']:
    m = model_results[name]
    print(f"\n  {name}:")
    for var in m.params.index:
        if 'is_ars' in var and var != 'is_ars':
            sig = '***' if m.pvalues[var] < 0.001 else '**' if m.pvalues[var] < 0.01 else '*' if m.pvalues[var] < 0.05 else 'ns'
            print(f"    {var:45s}: {m.params[var]:+8.2f} (p={m.pvalues[var]:.2e}) {sig}")

# =========================================================================
# Approach I: Within-category controlled tests
# =========================================================================
print("\n" + "=" * 90)
print("APPROACH I: Within-category controlled comparisons")
print("=" * 90)
print("  Does Cat B immunity persist within every feature category?")

for feature, bins, bin_labels in [
    ('has_any_pas', [True, False], ['PAS+', 'PAS-']),
    ('TE_group', ['intronic', 'intergenic'], ['Intronic', 'Intergenic']),
]:
    print(f"\n  Feature: {feature}")
    print(f"    {'Category':15s} | {'PASS Δ':>8s} {'n':>6s} | {'CatB Δ':>8s} {'n':>6s} | {'ΔΔ':>8s}")
    print("    " + "-" * 65)
    for val, label in zip(bins, bin_labels):
        pass_sub = df[(df['source'] == 'PASS') & (df[feature] == val)]
        catb_sub = df[(df['source'] == 'CatB') & (df[feature] == val)]
        dp, _, np_ = ars_delta(pass_sub)
        dc, _, nc = ars_delta(catb_sub)
        dd = dp - dc if not (np.isnan(dp) or np.isnan(dc)) else np.nan
        dp_s = f"{dp:+.1f}" if not np.isnan(dp) else 'n/a'
        dc_s = f"{dc:+.1f}" if not np.isnan(dc) else 'n/a'
        dd_s = f"{dd:+.1f}" if not np.isnan(dd) else 'n/a'
        print(f"    {label:15s} | {dp_s:>8s} {np_:6d} | {dc_s:>8s} {nc:6d} | {dd_s:>8s}")

# overlap_frac bins
print(f"\n  Feature: overlap_frac (binned)")
print(f"    {'Bin':15s} | {'PASS Δ':>8s} {'n':>6s} | {'CatB Δ':>8s} {'n':>6s} | {'ΔΔ':>8s}")
print("    " + "-" * 65)
for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]:
    pass_sub = df[(df['source'] == 'PASS') & (df['overlap_frac'] >= lo) & (df['overlap_frac'] < hi)]
    catb_sub = df[(df['source'] == 'CatB') & (df['overlap_frac'] >= lo) & (df['overlap_frac'] < hi)]
    dp, _, np_ = ars_delta(pass_sub)
    dc, _, nc = ars_delta(catb_sub)
    dd = dp - dc if not (np.isnan(dp) or np.isnan(dc)) else np.nan
    dp_s = f"{dp:+.1f}" if not np.isnan(dp) else 'n/a'
    dc_s = f"{dc:+.1f}" if not np.isnan(dc) else 'n/a'
    dd_s = f"{dd:+.1f}" if not np.isnan(dd) else 'n/a'
    print(f"    [{lo:.1f}-{hi:.1f})       | {dp_s:>8s} {np_:6d} | {dc_s:>8s} {nc:6d} | {dd_s:>8s}")

# =========================================================================
# FINAL SUMMARY TABLE
# =========================================================================
print("\n" + "=" * 90)
print("FINAL SUMMARY: All classifiers ranked by |ΔΔ|")
print("=" * 90)

# Collect additional results
# intergenic + ov>0.7
r = classifier_summary(df,
    (df['TE_group'] == 'intergenic') & (df['overlap_frac'] > 0.7),
    ~((df['TE_group'] == 'intergenic') & (df['overlap_frac'] > 0.7)))
results.append(('Intergenic + ov>0.7', r[0], r[2], r[3], r[5], r[6]))

# PAS + ov>0.7
r = classifier_summary(df,
    (df['has_any_pas']) & (df['overlap_frac'] > 0.7),
    ~((df['has_any_pas']) & (df['overlap_frac'] > 0.7)))
results.append(('PAS + ov>0.7', r[0], r[2], r[3], r[5], r[6]))

# ov>0.7 + no nonL1 TE
r = classifier_summary(df,
    (df['overlap_frac'] > 0.7) & (~df['has_non_l1_te']),
    ~((df['overlap_frac'] > 0.7) & (~df['has_non_l1_te'])))
results.append(('ov>0.7 + no_nonL1_TE', r[0], r[2], r[3], r[5], r[6]))

# ov>0.7 alone
r = classifier_summary(df, df['overlap_frac'] > 0.7, df['overlap_frac'] <= 0.7)
results.append(('overlap_frac > 0.7', r[0], r[2], r[3], r[5], r[6]))

# ChromHMM non-Tx
r = classifier_summary(df, df['chromhmm_group'] != 'Transcribed', df['chromhmm_group'] == 'Transcribed')
results.append(('ChromHMM non-Tx', r[0], r[2], r[3], r[5], r[6]))

# Sort by |ΔΔ| descending
results.sort(key=lambda x: x[5] if not np.isnan(x[5]) else 0, reverse=True)

print(f"\n  {'Classifier':40s} | {'Grp1 Δ':>8s} {'n':>6s} | {'Grp2 Δ':>8s} {'n':>6s} | {'|ΔΔ|':>6s} {'GENCODE-free':>12s}")
print("  " + "-" * 95)

gencode_free = {
    'GENCODE PASS/CatB': 'No',
    'Intergenic vs Intronic': 'Partial',
    'Has non-L1 TE': 'Yes',
    'n_TE_families > 1': 'Yes',
    'Canonical PAS (AATAAA)': 'Yes',
    'Any PAS motif': 'Yes',
    'ChromHMM non-Tx': 'Yes',
    'Intergenic + ov>0.7': 'Partial',
    'PAS + ov>0.7': 'Yes',
    'ov>0.7 + no_nonL1_TE': 'Yes',
    'overlap_frac > 0.7': 'Yes',
}

for name, d1, n1, d2, n2, dd in results:
    d1_s = f"{d1:+.1f}" if not np.isnan(d1) else 'n/a'
    d2_s = f"{d2:+.1f}" if not np.isnan(d2) else 'n/a'
    dd_s = f"{dd:.1f}" if not np.isnan(dd) else 'n/a'
    gf = gencode_free.get(name, '?')
    print(f"  {name:40s} | {d1_s:>8s} {n1:6d} | {d2_s:>8s} {n2:6d} | {dd_s:>6s} {gf:>12s}")

# Save full annotated dataset
df.to_csv(OUTDIR / 'unified_hela_ars_all_features.tsv', sep='\t', index=False)
print(f"\nSaved to: {OUTDIR}/unified_hela_ars_all_features.tsv")
print("Done!")
