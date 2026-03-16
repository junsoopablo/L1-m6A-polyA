#!/usr/bin/env python3
"""
Sequence feature analysis: What predicts L1 poly(A) retention under stress?

Goal: Identify sequence-level features in ancient L1 reads that predict
poly(A) tail length under arsenite stress, beyond m6A density.

Approach:
1. Load HeLa (normal) and HeLa-Ars (stress) ancient L1 reads
2. Extract sequences from BAM files
3. Compute sequence features (k-mers, RBP motifs, GC%, UA dinucleotides, PAS)
4. LASSO regression: poly(A) ~ sequence features + m6A/kb + stress + interactions
5. Identify features with stress-specific predictive power
"""

import os
import sys
import numpy as np
import pandas as pd
import pysam
from collections import Counter
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
RESULTS = f'{BASE}/results_group'
CACHE_DIR = f'{BASE}/analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
OUT_DIR = f'{BASE}/analysis/01_exploration/topic_08_sequence_features'
os.makedirs(OUT_DIR, exist_ok=True)

HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3']
ARS_GROUPS = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Known RBP binding motifs (from literature)
# Format: name -> list of motifs
RBP_MOTIFS = {
    # Stabilizing
    'HuR_ARE': ['AUUUA', 'UAUUUAU'],  # HuR/ELAVL1 - stabilizing when bound
    'HuR_Urich': ['UUUUU', 'UUUUUU'],  # U-rich HuR binding
    # Destabilizing
    'TTP_ARE': ['UAUUUAUU', 'AUUUAUUU'],  # TTP/ZFP36 - destabilizing
    'PUM2': ['UGUAAAUA', 'UGUANAUA'],  # Pumilio - deadenylation
    'KSRP': ['GUUUG', 'GGGGU'],  # KSRP/KHSRP - destabilizing
    # Stress-related
    'G4': ['GGGG'],  # G-quadruplex potential
    'YTHDF': ['GGACU', 'GAACU', 'AGACU'],  # m6A reader binding (DRACH-like on RNA)
    # TE-specific
    'SAFB_Arich': ['AAAAA', 'AAAAAA'],  # SAFB binding (A-rich)
    'ZAP_CpG': ['CG'],  # ZAP recognition
}

# PAS variants
PAS_CANONICAL = 'AATAAA'
PAS_VARIANTS = [
    'AATAAA', 'ATTAAA', 'AGTAAA', 'TATAAA', 'CATAAA', 'GATAAA',
    'AATATA', 'AATACA', 'AATAGA', 'AAAAAG', 'ACTAAA', 'AATGAA'
]

# ============================================================
# Step 1: Load L1 metadata + modification data
# ============================================================
print("=" * 70)
print("Step 1: Loading L1 metadata and modification data")
print("=" * 70)

all_reads = []

for group in HELA_GROUPS + ARS_GROUPS:
    condition = 'stress' if 'Ars' in group else 'normal'

    # Load L1 summary
    summary_path = f'{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv'
    if not os.path.exists(summary_path):
        print(f"  WARNING: {summary_path} not found, skipping")
        continue

    df = pd.read_csv(summary_path, sep='\t')

    # Filter: PASS, ancient only, valid poly(A)
    df = df[df['qc_tag'] == 'PASS'].copy()
    df['is_young'] = df['gene_id'].isin(YOUNG_SUBFAMILIES)
    df_ancient = df[~df['is_young']].copy()
    df_ancient = df_ancient[df_ancient['polya_length'] > 0].copy()

    # Load Part3 cache for m6A/kb
    cache_path = f'{CACHE_DIR}/{group}_l1_per_read.tsv'
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, sep='\t')
        df_ancient = df_ancient.merge(
            cache[['read_id', 'm6a_sites_high', 'psi_sites_high']],
            on='read_id', how='left', suffixes=('', '_cache')
        )
        df_ancient['m6a_per_kb'] = df_ancient['m6a_sites_high'] / (df_ancient['read_length'] / 1000)
    else:
        df_ancient['m6a_per_kb'] = np.nan

    df_ancient['condition'] = condition
    df_ancient['group'] = group

    all_reads.append(df_ancient[['read_id', 'chr', 'start', 'end', 'read_length',
                                  'gene_id', 'te_strand', 'read_strand',
                                  'polya_length', 'm6a_per_kb', 'condition',
                                  'group', 'transcript_id', 'class']])

    print(f"  {group}: {len(df_ancient)} ancient PASS reads")

df_all = pd.concat(all_reads, ignore_index=True)
df_all = df_all.dropna(subset=['m6a_per_kb'])

print(f"\nTotal ancient L1 reads: {len(df_all)}")
print(f"  Normal: {(df_all['condition'] == 'normal').sum()}")
print(f"  Stress: {(df_all['condition'] == 'stress').sum()}")
print(f"  Poly(A) normal median: {df_all.loc[df_all['condition']=='normal', 'polya_length'].median():.1f}")
print(f"  Poly(A) stress median: {df_all.loc[df_all['condition']=='stress', 'polya_length'].median():.1f}")

# ============================================================
# Step 2: Extract sequences from BAM
# ============================================================
print("\n" + "=" * 70)
print("Step 2: Extracting sequences from BAM files")
print("=" * 70)

read_sequences = {}

for group in HELA_GROUPS + ARS_GROUPS:
    bam_path = f'{RESULTS}/{group}/h_mafia/{group}.mAFiA.reads.bam'
    if not os.path.exists(bam_path):
        print(f"  WARNING: {bam_path} not found")
        continue

    # Get read IDs for this group
    group_reads = set(df_all[df_all['group'] == group]['read_id'])
    if not group_reads:
        continue

    print(f"  {group}: extracting {len(group_reads)} sequences...", end=' ')
    found = 0

    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        for read in bam.fetch():
            if read.query_name in group_reads:
                # Get the query sequence (as aligned)
                seq = read.query_sequence
                if seq:
                    read_sequences[read.query_name] = seq
                    found += 1
                    if found >= len(group_reads):
                        break

    print(f"found {found}")

print(f"\nTotal reads with sequence: {len(read_sequences)}")

# Filter to reads with sequences
df_all = df_all[df_all['read_id'].isin(read_sequences)].copy()
print(f"Reads after sequence filter: {len(df_all)}")

# ============================================================
# Step 3: Compute sequence features
# ============================================================
print("\n" + "=" * 70)
print("Step 3: Computing sequence features")
print("=" * 70)

def compute_kmer_counts(seq, k=3):
    """Count k-mers in sequence (DNA alphabet: ACGT)."""
    counts = Counter()
    seq_upper = seq.upper()
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i:i+k]
        if all(c in 'ACGT' for c in kmer):
            counts[kmer] += 1
    return counts

def count_motif(seq, motif):
    """Count occurrences of motif in sequence (RNA motifs converted to DNA)."""
    # Convert RNA motif to DNA
    dna_motif = motif.replace('U', 'T')
    seq_upper = seq.upper()

    # Handle IUPAC codes
    if 'N' in dna_motif:
        count = 0
        for i in range(len(seq_upper) - len(dna_motif) + 1):
            match = True
            for j, c in enumerate(dna_motif):
                if c == 'N':
                    continue
                if seq_upper[i+j] != c:
                    match = False
                    break
            if match:
                count += 1
        return count
    else:
        count = 0
        start = 0
        while True:
            pos = seq_upper.find(dna_motif, start)
            if pos == -1:
                break
            count += 1
            start = pos + 1
        return count

def compute_sequence_features(seq):
    """Compute all sequence features for a single read."""
    features = {}
    seq_upper = seq.upper()
    seq_len = len(seq_upper)

    if seq_len == 0:
        return None

    # --- Basic composition ---
    a_count = seq_upper.count('A')
    t_count = seq_upper.count('T')
    g_count = seq_upper.count('G')
    c_count = seq_upper.count('C')

    features['gc_content'] = (g_count + c_count) / seq_len
    features['at_content'] = (a_count + t_count) / seq_len
    features['a_fraction'] = a_count / seq_len
    features['t_fraction'] = t_count / seq_len

    # --- Dinucleotide frequencies (key ones) ---
    dinucs_of_interest = ['TA', 'CG', 'AA', 'TT', 'AT', 'GC', 'GT', 'CA',
                          'AG', 'TC', 'GA', 'CT', 'TG', 'AC', 'GG', 'CC']
    for di in dinucs_of_interest:
        count = 0
        for i in range(seq_len - 1):
            if seq_upper[i:i+2] == di:
                count += 1
        features[f'di_{di}'] = count / max(seq_len - 1, 1)

    # UA (=TA on DNA) is the key destabilizing dinucleotide (Zhang eLife 2024)
    features['ua_density'] = features['di_TA']  # UA on RNA = TA on DNA

    # --- 3' end features (last 100bp) ---
    end_100 = seq_upper[-100:] if seq_len >= 100 else seq_upper
    end_len = len(end_100)
    features['gc_3prime_100'] = (end_100.count('G') + end_100.count('C')) / end_len
    features['a_fraction_3prime'] = end_100.count('A') / end_len

    # --- PAS in last 50bp ---
    end_50 = seq_upper[-50:] if seq_len >= 50 else seq_upper
    features['has_canonical_pas'] = 1 if PAS_CANONICAL in end_50 else 0
    features['has_any_pas'] = 0
    for pas in PAS_VARIANTS:
        if pas in end_50:
            features['has_any_pas'] = 1
            break

    # --- RBP motifs per kb ---
    for rbp_name, motifs in RBP_MOTIFS.items():
        total = 0
        for motif in motifs:
            total += count_motif(seq, motif)
        features[f'rbp_{rbp_name}_per_kb'] = total / (seq_len / 1000)

    # --- Sequence complexity (Shannon entropy of 3-mers) ---
    kmer_counts = compute_kmer_counts(seq, k=3)
    total_kmers = sum(kmer_counts.values())
    if total_kmers > 0:
        probs = np.array(list(kmer_counts.values())) / total_kmers
        features['kmer3_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
    else:
        features['kmer3_entropy'] = 0

    # --- A-runs (consecutive A's) ---
    max_a_run = 0
    current_run = 0
    for c in seq_upper:
        if c == 'A':
            current_run += 1
            max_a_run = max(max_a_run, current_run)
        else:
            current_run = 0
    features['max_a_run'] = max_a_run

    # --- T-runs (potential poly(T) on template) ---
    max_t_run = 0
    current_run = 0
    for c in seq_upper:
        if c == 'T':
            current_run += 1
            max_t_run = max(max_t_run, current_run)
        else:
            current_run = 0
    features['max_t_run'] = max_t_run

    # --- Selected informative 3-mers (per kb) ---
    # Focus on extremes of stability-associated k-mers
    important_3mers = ['AAA', 'TTT', 'ATT', 'TAT', 'ATA', 'TAA', 'AAT',  # AU-rich
                       'GGG', 'CCC', 'GCC', 'CGC', 'GCG', 'CCG',  # GC-rich
                       'TTA', 'TAT',  # ARE-like
                       'GTG', 'TGT', 'GAT',  # variable
                       ]
    for kmer in important_3mers:
        features[f'k3_{kmer}_per_kb'] = kmer_counts.get(kmer, 0) / (seq_len / 1000)

    return features

# Compute features for all reads
print("Computing sequence features for all reads...")
feature_list = []
valid_read_ids = []

for i, (read_id, row) in enumerate(df_all.iterrows()):
    rid = row['read_id']
    if rid not in read_sequences:
        continue

    seq = read_sequences[rid]
    feats = compute_sequence_features(seq)
    if feats is None:
        continue

    feats['read_id'] = rid
    feature_list.append(feats)
    valid_read_ids.append(rid)

    if (i + 1) % 2000 == 0:
        print(f"  Processed {i+1}/{len(df_all)} reads...")

df_features = pd.DataFrame(feature_list)
print(f"\nFeatures computed for {len(df_features)} reads")
print(f"Number of features: {len(df_features.columns) - 1}")  # -1 for read_id

# ============================================================
# Step 4: Merge and prepare for modeling
# ============================================================
print("\n" + "=" * 70)
print("Step 4: Preparing modeling dataset")
print("=" * 70)

# Merge features with metadata
df_model = df_all.merge(df_features, on='read_id', how='inner')
print(f"Merged dataset: {len(df_model)} reads")

# Create stress indicator
df_model['is_stress'] = (df_model['condition'] == 'stress').astype(int)

# Read length (control variable)
df_model['read_length_kb'] = df_model['read_length'] / 1000

# Subfamily family grouping (coarse)
def get_l1_family(subfamily):
    if subfamily.startswith('L1MC'):
        return 'L1MC'
    elif subfamily.startswith('L1ME'):
        return 'L1ME'
    elif subfamily.startswith('L1M'):
        return 'L1M'
    elif subfamily.startswith('L1P'):
        return 'L1P'
    elif subfamily.startswith('HAL'):
        return 'HAL'
    else:
        return 'other'

df_model['l1_family'] = df_model['gene_id'].apply(get_l1_family)

# Select feature columns
exclude_cols = {'read_id', 'chr', 'start', 'end', 'read_length', 'gene_id',
                'te_strand', 'read_strand', 'polya_length', 'm6a_per_kb',
                'condition', 'group', 'transcript_id', 'class', 'is_stress',
                'read_length_kb', 'l1_family', 'is_young'}
seq_feature_cols = [c for c in df_model.columns if c not in exclude_cols]
print(f"Sequence feature columns: {len(seq_feature_cols)}")

# ============================================================
# Step 5: Descriptive comparison — Normal vs Stress
# ============================================================
print("\n" + "=" * 70)
print("Step 5: Descriptive feature comparison (Normal vs Stress)")
print("=" * 70)

print("\nFeature distributions by condition:")
print(f"{'Feature':<30} {'Normal mean':>12} {'Stress mean':>12} {'Diff':>8} {'P-value':>12}")
print("-" * 80)

normal_mask = df_model['is_stress'] == 0
stress_mask = df_model['is_stress'] == 1

feature_pvalues = {}
for col in sorted(seq_feature_cols):
    n_vals = df_model.loc[normal_mask, col]
    s_vals = df_model.loc[stress_mask, col]

    if n_vals.std() == 0 and s_vals.std() == 0:
        continue

    stat, pval = stats.mannwhitneyu(n_vals, s_vals, alternative='two-sided')
    feature_pvalues[col] = pval

    if pval < 0.05:
        print(f"{col:<30} {n_vals.mean():>12.4f} {s_vals.mean():>12.4f} "
              f"{s_vals.mean() - n_vals.mean():>8.4f} {pval:>12.2e} *")

print(f"\n{sum(1 for p in feature_pvalues.values() if p < 0.05)}/{len(feature_pvalues)} features differ between conditions (p<0.05)")

# ============================================================
# Step 6: LASSO regression — predict poly(A) from features
# ============================================================
print("\n" + "=" * 70)
print("Step 6: LASSO regression")
print("=" * 70)

# --- Model A: Normal condition only ---
print("\n--- Model A: Normal condition (HeLa) ---")
df_normal = df_model[df_model['is_stress'] == 0].copy()

X_normal = df_normal[seq_feature_cols + ['m6a_per_kb', 'read_length_kb']].copy()
y_normal = df_normal['polya_length'].values

# Remove any remaining NaN
mask = ~(X_normal.isna().any(axis=1) | np.isnan(y_normal))
X_normal = X_normal[mask]
y_normal = y_normal[mask]

scaler_n = StandardScaler()
X_normal_scaled = scaler_n.fit_transform(X_normal)

lasso_n = LassoCV(cv=5, random_state=42, max_iter=10000, n_alphas=50)
lasso_n.fit(X_normal_scaled, y_normal)

nonzero_n = np.sum(lasso_n.coef_ != 0)
print(f"  N reads: {len(y_normal)}")
print(f"  Best alpha: {lasso_n.alpha_:.4f}")
print(f"  R²: {lasso_n.score(X_normal_scaled, y_normal):.4f}")
print(f"  Non-zero coefficients: {nonzero_n}/{len(lasso_n.coef_)}")

coef_names_n = list(X_normal.columns)
coefs_normal = pd.DataFrame({
    'feature': coef_names_n,
    'coef_normal': lasso_n.coef_
}).sort_values('coef_normal', key=abs, ascending=False)

print("\n  Top features (normal):")
for _, row in coefs_normal[coefs_normal['coef_normal'] != 0].head(15).iterrows():
    print(f"    {row['feature']:<30} {row['coef_normal']:>+8.3f}")

# --- Model B: Stress condition only ---
print("\n--- Model B: Stress condition (HeLa-Ars) ---")
df_stress = df_model[df_model['is_stress'] == 1].copy()

X_stress = df_stress[seq_feature_cols + ['m6a_per_kb', 'read_length_kb']].copy()
y_stress = df_stress['polya_length'].values

mask = ~(X_stress.isna().any(axis=1) | np.isnan(y_stress))
X_stress = X_stress[mask]
y_stress = y_stress[mask]

scaler_s = StandardScaler()
X_stress_scaled = scaler_s.fit_transform(X_stress)

lasso_s = LassoCV(cv=5, random_state=42, max_iter=10000, n_alphas=50)
lasso_s.fit(X_stress_scaled, y_stress)

nonzero_s = np.sum(lasso_s.coef_ != 0)
print(f"  N reads: {len(y_stress)}")
print(f"  Best alpha: {lasso_s.alpha_:.4f}")
print(f"  R²: {lasso_s.score(X_stress_scaled, y_stress):.4f}")
print(f"  Non-zero coefficients: {nonzero_s}/{len(lasso_s.coef_)}")

coef_names_s = list(X_stress.columns)
coefs_stress = pd.DataFrame({
    'feature': coef_names_s,
    'coef_stress': lasso_s.coef_
}).sort_values('coef_stress', key=abs, ascending=False)

print("\n  Top features (stress):")
for _, row in coefs_stress[coefs_stress['coef_stress'] != 0].head(15).iterrows():
    print(f"    {row['feature']:<30} {row['coef_stress']:>+8.3f}")

# --- Model C: Combined with interaction terms ---
print("\n--- Model C: Combined model with stress interactions ---")

# Create interaction features (feature × stress)
X_combined = df_model[seq_feature_cols + ['m6a_per_kb', 'read_length_kb', 'is_stress']].copy()

# Add key interaction terms (stress × each feature)
interaction_cols = []
for col in seq_feature_cols + ['m6a_per_kb']:
    icol = f'stress_x_{col}'
    X_combined[icol] = X_combined['is_stress'] * X_combined[col]
    interaction_cols.append(icol)

y_combined = df_model['polya_length'].values

mask = ~(X_combined.isna().any(axis=1) | np.isnan(y_combined))
X_combined = X_combined[mask]
y_combined = y_combined[mask]

scaler_c = StandardScaler()
X_combined_scaled = scaler_c.fit_transform(X_combined)

lasso_c = LassoCV(cv=5, random_state=42, max_iter=10000, n_alphas=50)
lasso_c.fit(X_combined_scaled, y_combined)

print(f"  N reads: {len(y_combined)}")
print(f"  Best alpha: {lasso_c.alpha_:.4f}")
print(f"  R²: {lasso_c.score(X_combined_scaled, y_combined):.4f}")
print(f"  Non-zero coefficients: {np.sum(lasso_c.coef_ != 0)}/{len(lasso_c.coef_)}")

coef_names_c = list(X_combined.columns)
coefs_combined = pd.DataFrame({
    'feature': coef_names_c,
    'coef': lasso_c.coef_
})

# Separate main effects and interactions
main_effects = coefs_combined[~coefs_combined['feature'].str.startswith('stress_x_')]
interactions = coefs_combined[coefs_combined['feature'].str.startswith('stress_x_')]

print("\n  Top MAIN effects:")
for _, row in main_effects[main_effects['coef'] != 0].sort_values('coef', key=abs, ascending=False).head(10).iterrows():
    print(f"    {row['feature']:<30} {row['coef']:>+8.3f}")

print("\n  Top STRESS INTERACTION effects (stress-specific predictors):")
for _, row in interactions[interactions['coef'] != 0].sort_values('coef', key=abs, ascending=False).head(15).iterrows():
    feat = row['feature'].replace('stress_x_', '')
    direction = "protective" if row['coef'] > 0 else "vulnerability"
    print(f"    {feat:<30} {row['coef']:>+8.3f}  ({direction})")

# ============================================================
# Step 7: OLS for specific features of interest
# ============================================================
print("\n" + "=" * 70)
print("Step 7: OLS validation of top LASSO features")
print("=" * 70)

import statsmodels.api as sm

# Get top interaction features from LASSO
top_interactions = interactions[interactions['coef'] != 0].sort_values('coef', key=abs, ascending=False)

if len(top_interactions) > 0:
    print("\nOLS validation of top stress-interaction features:")
    print(f"{'Feature':<30} {'Main coef':>10} {'Int coef':>10} {'Int P':>12} {'Direction':>12}")
    print("-" * 80)

    for _, row in top_interactions.head(10).iterrows():
        feat = row['feature'].replace('stress_x_', '')
        if feat not in df_model.columns:
            continue

        # OLS: poly(A) ~ feature + stress + feature*stress + read_length + m6a/kb
        X_ols = df_model[['is_stress', 'read_length_kb', 'm6a_per_kb']].copy()
        X_ols[feat] = df_model[feat]
        X_ols[f'{feat}_x_stress'] = X_ols[feat] * X_ols['is_stress']
        X_ols = sm.add_constant(X_ols)

        y_ols = df_model['polya_length']
        mask = ~(X_ols.isna().any(axis=1) | y_ols.isna())

        model = sm.OLS(y_ols[mask], X_ols[mask]).fit()

        main_coef = model.params.get(feat, 0)
        int_coef = model.params.get(f'{feat}_x_stress', 0)
        int_pval = model.pvalues.get(f'{feat}_x_stress', 1)
        direction = "protective" if int_coef > 0 else "vulnerability"

        sig = '***' if int_pval < 0.001 else '**' if int_pval < 0.01 else '*' if int_pval < 0.05 else 'ns'
        print(f"  {feat:<30} {main_coef:>+10.3f} {int_coef:>+10.3f} {int_pval:>12.2e} {direction:>10} {sig}")
else:
    print("No significant stress interactions found in LASSO.")

# ============================================================
# Step 8: Correlation analysis — features vs poly(A) by condition
# ============================================================
print("\n" + "=" * 70)
print("Step 8: Feature-poly(A) correlations by condition")
print("=" * 70)

print(f"\n{'Feature':<30} {'r (normal)':>12} {'r (stress)':>12} {'Δr':>8} {'P (stress)':>12}")
print("-" * 80)

corr_results = []
for col in sorted(seq_feature_cols):
    if df_model[col].std() == 0:
        continue

    # Normal
    n_vals = df_model.loc[normal_mask, col]
    n_polya = df_model.loc[normal_mask, 'polya_length']
    r_n, p_n = stats.pearsonr(n_vals, n_polya)

    # Stress
    s_vals = df_model.loc[stress_mask, col]
    s_polya = df_model.loc[stress_mask, 'polya_length']
    r_s, p_s = stats.pearsonr(s_vals, s_polya)

    delta_r = r_s - r_n
    corr_results.append({
        'feature': col, 'r_normal': r_n, 'r_stress': r_s,
        'delta_r': delta_r, 'p_stress': p_s
    })

    # Show features with largest stress-specific correlation change
    if abs(delta_r) > 0.03 or (p_s < 0.001 and abs(r_s) > 0.05):
        sig = '***' if p_s < 0.001 else '**' if p_s < 0.01 else '*' if p_s < 0.05 else ''
        print(f"  {col:<30} {r_n:>+12.4f} {r_s:>+12.4f} {delta_r:>+8.4f} {p_s:>12.2e} {sig}")

df_corr = pd.DataFrame(corr_results).sort_values('delta_r', key=abs, ascending=False)

print(f"\nTop 10 features with largest stress-specific correlation change (|Δr|):")
for _, row in df_corr.head(10).iterrows():
    direction = "stress-protective" if row['delta_r'] > 0 else "stress-vulnerability"
    print(f"  {row['feature']:<30} Δr={row['delta_r']:>+.4f} ({direction})")

# ============================================================
# Step 9: Save results
# ============================================================
print("\n" + "=" * 70)
print("Step 9: Saving results")
print("=" * 70)

# Save all coefficients
coefs_normal.to_csv(f'{OUT_DIR}/lasso_coefs_normal.tsv', sep='\t', index=False)
coefs_stress.to_csv(f'{OUT_DIR}/lasso_coefs_stress.tsv', sep='\t', index=False)

all_coefs = coefs_normal.merge(coefs_stress, on='feature', how='outer').fillna(0)
all_coefs.to_csv(f'{OUT_DIR}/lasso_coefs_comparison.tsv', sep='\t', index=False)

# Save interaction model
interactions_out = interactions[interactions['coef'] != 0].copy()
interactions_out['feature'] = interactions_out['feature'].str.replace('stress_x_', '')
interactions_out = interactions_out.sort_values('coef', key=abs, ascending=False)
interactions_out.to_csv(f'{OUT_DIR}/lasso_stress_interactions.tsv', sep='\t', index=False)

# Save correlation results
df_corr.to_csv(f'{OUT_DIR}/feature_polya_correlations.tsv', sep='\t', index=False)

# Save per-read features for downstream use
df_model.to_csv(f'{OUT_DIR}/ancient_l1_with_features.tsv', sep='\t', index=False)

# Summary
print(f"\nResults saved to {OUT_DIR}/")
print(f"  lasso_coefs_normal.tsv")
print(f"  lasso_coefs_stress.tsv")
print(f"  lasso_coefs_comparison.tsv")
print(f"  lasso_stress_interactions.tsv")
print(f"  feature_polya_correlations.tsv")
print(f"  ancient_l1_with_features.tsv")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Analysis: Sequence features predicting L1 poly(A) retention under stress

Dataset:
  Normal (HeLa): {(df_all['condition'] == 'normal').sum()} ancient L1 reads
  Stress (HeLa-Ars): {(df_all['condition'] == 'stress').sum()} ancient L1 reads

LASSO Models:
  Normal-only R²: {lasso_n.score(X_normal_scaled, y_normal):.4f} (non-zero: {nonzero_n})
  Stress-only R²: {lasso_s.score(X_stress_scaled, y_stress):.4f} (non-zero: {nonzero_s})
  Combined R²: {lasso_c.score(X_combined_scaled, y_combined):.4f}

Key question: Which sequence features predict poly(A) ONLY under stress
(stress-specific CREs) vs features that always predict poly(A) (constitutive)?
""")

print("Done!")
