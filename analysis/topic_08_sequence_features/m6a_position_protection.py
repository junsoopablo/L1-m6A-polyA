#!/usr/bin/env python3
"""
Does m6A POSITION within the L1 read affect poly(A) protection under stress?

Hypothesis: m6A near the 3' end (close to poly(A) tail) may be more
protective than m6A far from the poly(A) tail, because:
- 3'-proximal m6A could directly block deadenylase access
- YTHDF readers near poly(A) could recruit PABPC1
- YTHDF2 normally recruits CCR4-NOT for deadenylation; stress may
  switch this to a protective mode when m6A is near the 3' end

Analyses:
1. m6A positional distribution on reads (relative to 3' end)
2. 3'-proximal vs 5'-proximal m6A: differential poly(A) protection
3. Distance of nearest m6A to 3' end → poly(A) correlation
4. m6A "coverage" in last 200bp vs rest
5. OLS: 3'-proximal m6A × stress interaction
6. Combined model: m6A position + density + PAS
"""

import os
import ast
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
RESULTS = f'{BASE}/results_group'
CACHE_DIR = f'{BASE}/analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
OUT_DIR = f'{BASE}/analysis/01_exploration/topic_08_sequence_features'

HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3']
ARS_GROUPS = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# ============================================================
# Load data with m6A positions
# ============================================================
print("=" * 70)
print("Loading data with m6A positions")
print("=" * 70)

all_reads = []
for group in HELA_GROUPS + ARS_GROUPS:
    condition = 'stress' if 'Ars' in group else 'normal'

    summary_path = f'{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv'
    cache_path = f'{CACHE_DIR}/{group}_l1_per_read.tsv'

    if not os.path.exists(summary_path) or not os.path.exists(cache_path):
        continue

    s = pd.read_csv(summary_path, sep='\t')
    c = pd.read_csv(cache_path, sep='\t')

    s = s[s['qc_tag'] == 'PASS'].copy()
    s = s[~s['gene_id'].isin(YOUNG)].copy()
    s = s[s['polya_length'] > 0].copy()

    # Merge with cache (includes m6a_positions)
    s = s.merge(c[['read_id', 'm6a_sites_high', 'psi_sites_high',
                    'm6a_positions', 'psi_positions']], on='read_id', how='inner')

    s['condition'] = condition
    s['group'] = group
    s['m6a_per_kb'] = s['m6a_sites_high'] / (s['read_length'] / 1000)
    all_reads.append(s)

df = pd.concat(all_reads, ignore_index=True)
print(f"Total ancient L1 reads: {len(df)}")
print(f"  Normal: {(df['condition']=='normal').sum()}, Stress: {(df['condition']=='stress').sum()}")

# Parse m6A positions
def parse_positions(pos_str):
    """Parse position list from string."""
    if pd.isna(pos_str) or pos_str == '[]':
        return []
    try:
        return ast.literal_eval(pos_str)
    except:
        return []

df['m6a_pos_list'] = df['m6a_positions'].apply(parse_positions)

# ============================================================
# Analysis 1: m6A positional distribution relative to 3' end
# ============================================================
print("\n" + "=" * 70)
print("Analysis 1: m6A positional distribution")
print("=" * 70)

# For each read, compute m6A positions as fraction of read length
# Position 0 = read start (5'-most in alignment), read_length = 3' end
# DRS reads are 3'-biased, so most coverage is near the 3' end
# m6A position / read_length: 0 = 5' end, 1 = 3' end

all_rel_positions = []
for _, row in df.iterrows():
    positions = row['m6a_pos_list']
    rl = row['read_length']
    if rl > 0 and len(positions) > 0:
        for pos in positions:
            rel_pos = pos / rl  # 0=5', 1=3'
            all_rel_positions.append({
                'rel_pos': rel_pos,
                'condition': row['condition'],
                'dist_from_3prime': rl - pos  # bp from 3' end
            })

df_pos = pd.DataFrame(all_rel_positions)
print(f"Total m6A sites: {len(df_pos)}")
print(f"  Mean relative position: {df_pos['rel_pos'].mean():.3f} (0=5', 1=3')")
print(f"  Median distance from 3' end: {df_pos['dist_from_3prime'].median():.0f} bp")

# Distribution by quintiles
print("\nm6A position quintiles (fraction of read):")
for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pct = (df_pos['rel_pos'] <= q).mean()
    print(f"  ≤{q:.1f}: {pct:.1%}")

# ============================================================
# Analysis 2: Compute per-read positional features
# ============================================================
print("\n" + "=" * 70)
print("Analysis 2: Per-read m6A positional features")
print("=" * 70)

def compute_positional_features(row):
    """Compute m6A positional features for a single read."""
    positions = row['m6a_pos_list']
    rl = row['read_length']
    n_m6a = row['m6a_sites_high']

    features = {}

    if n_m6a == 0 or rl == 0:
        features['has_m6a'] = 0
        features['nearest_m6a_to_3prime'] = np.nan
        features['m6a_in_last_200bp'] = 0
        features['m6a_in_last_500bp'] = 0
        features['m6a_frac_in_3prime_half'] = np.nan
        features['m6a_mean_rel_pos'] = np.nan
        features['m6a_3prime_density'] = 0  # sites/kb in last 500bp
        features['m6a_5prime_density'] = 0  # sites/kb in rest
        return features

    features['has_m6a'] = 1

    # Distance of nearest m6A to 3' end
    dists_from_3prime = [rl - p for p in positions]
    features['nearest_m6a_to_3prime'] = min(dists_from_3prime)

    # m6A count in last 200bp and 500bp
    m6a_last_200 = sum(1 for p in positions if rl - p <= 200)
    m6a_last_500 = sum(1 for p in positions if rl - p <= 500)
    features['m6a_in_last_200bp'] = m6a_last_200
    features['m6a_in_last_500bp'] = m6a_last_500

    # Fraction of m6A in 3' half
    m6a_3prime_half = sum(1 for p in positions if p >= rl / 2)
    features['m6a_frac_in_3prime_half'] = m6a_3prime_half / n_m6a

    # Mean relative position
    rel_positions = [p / rl for p in positions]
    features['m6a_mean_rel_pos'] = np.mean(rel_positions)

    # Density in last 500bp vs rest
    region_3prime = min(500, rl)
    region_5prime = max(rl - 500, 0)
    features['m6a_3prime_density'] = m6a_last_500 / (region_3prime / 1000) if region_3prime > 0 else 0
    features['m6a_5prime_density'] = (n_m6a - m6a_last_500) / (region_5prime / 1000) if region_5prime > 0 else 0

    return features

print("Computing positional features...")
pos_features = df.apply(compute_positional_features, axis=1, result_type='expand')
df = pd.concat([df, pos_features], axis=1)

# Filter to reads with m6A
df_m6a = df[df['has_m6a'] == 1].copy()
print(f"Reads with ≥1 m6A site: {len(df_m6a)}")

# ============================================================
# Analysis 3: 3'-proximal m6A vs poly(A) under stress
# ============================================================
print("\n" + "=" * 70)
print("Analysis 3: Does 3'-proximal m6A protect poly(A) better?")
print("=" * 70)

# Among m6A+ reads: nearest m6A distance to 3' end → poly(A) correlation
for cond in ['normal', 'stress']:
    mask = df_m6a['condition'] == cond
    r, p = stats.pearsonr(
        df_m6a.loc[mask, 'nearest_m6a_to_3prime'],
        df_m6a.loc[mask, 'polya_length']
    )
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {cond}: nearest m6A distance to 3' end vs poly(A): r={r:+.4f}, P={p:.2e} {sig}")
    # Negative r means closer m6A → longer poly(A) (protective)

# m6A in last 200bp → poly(A)
print("\nm6A count in last 200bp vs poly(A):")
for cond in ['normal', 'stress']:
    mask = df_m6a['condition'] == cond
    r, p = stats.pearsonr(
        df_m6a.loc[mask, 'm6a_in_last_200bp'],
        df_m6a.loc[mask, 'polya_length']
    )
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {cond}: m6A in last 200bp vs poly(A): r={r:+.4f}, P={p:.2e} {sig}")

# 3' half m6A fraction → poly(A)
print("\nm6A fraction in 3' half vs poly(A):")
for cond in ['normal', 'stress']:
    mask = (df_m6a['condition'] == cond) & (~df_m6a['m6a_frac_in_3prime_half'].isna())
    r, p = stats.pearsonr(
        df_m6a.loc[mask, 'm6a_frac_in_3prime_half'],
        df_m6a.loc[mask, 'polya_length']
    )
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {cond}: m6A 3' half fraction vs poly(A): r={r:+.4f}, P={p:.2e} {sig}")

# ============================================================
# Analysis 4: 3'-proximal vs 5'-proximal m6A density
# ============================================================
print("\n" + "=" * 70)
print("Analysis 4: 3' vs 5' m6A density comparison")
print("=" * 70)

# Only for reads long enough to have both regions (>500bp)
df_long = df_m6a[df_m6a['read_length'] > 500].copy()
print(f"Reads >500bp with m6A: {len(df_long)}")

print(f"\n  3' 500bp m6A density: {df_long['m6a_3prime_density'].median():.2f}/kb")
print(f"  5' rest m6A density:  {df_long['m6a_5prime_density'].median():.2f}/kb")
print(f"  Ratio (3'/5'):        {df_long['m6a_3prime_density'].median() / max(df_long['m6a_5prime_density'].median(), 0.01):.2f}x")

# Does 3' density predict poly(A) better than 5' density under stress?
print("\nDensity region comparison under stress:")
stress_long = df_long[df_long['condition'] == 'stress']
for feat, name in [('m6a_3prime_density', "3' 500bp density"),
                    ('m6a_5prime_density', "5' rest density"),
                    ('m6a_per_kb', 'Overall m6A/kb')]:
    r, p = stats.pearsonr(stress_long[feat], stress_long['polya_length'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {name:<25} r={r:+.4f}, P={p:.2e} {sig}")

# ============================================================
# Analysis 5: OLS — 3'-proximal m6A × stress interaction
# ============================================================
print("\n" + "=" * 70)
print("Analysis 5: OLS — positional m6A × stress")
print("=" * 70)

# Use all reads (including m6A=0)
df['is_stress'] = (df['condition'] == 'stress').astype(int)

# Fill NaN positional features for m6A-negative reads
df['m6a_in_last_200bp'] = df['m6a_in_last_200bp'].fillna(0)
df['m6a_in_last_500bp'] = df['m6a_in_last_500bp'].fillna(0)

# Compute 3'-proximal m6A per kb (in last 500bp region)
df['m6a_3prime_per_kb'] = df['m6a_in_last_500bp'] / 0.5  # per kb in 500bp window
# Non-3'-proximal m6A
df['m6a_rest_count'] = df['m6a_sites_high'] - df['m6a_in_last_500bp']
df['read_rest_kb'] = (df['read_length'] - 500).clip(lower=0) / 1000
df['m6a_rest_per_kb'] = np.where(df['read_rest_kb'] > 0,
                                  df['m6a_rest_count'] / df['read_rest_kb'], 0)

# Load PAS data from the feature dataset
feat_df = pd.read_csv(f'{OUT_DIR}/ancient_l1_with_features.tsv', sep='\t')
pas_data = feat_df[['read_id', 'has_canonical_pas']].drop_duplicates()
df = df.merge(pas_data, on='read_id', how='left')
df['has_canonical_pas'] = df['has_canonical_pas'].fillna(0)

# Model A: Overall m6A/kb
print("\n--- Model A: Overall m6A/kb (baseline) ---")
X_a = df[['is_stress', 'has_canonical_pas', 'm6a_per_kb']].copy()
X_a['read_length_kb'] = df['read_length'] / 1000
X_a['stress_x_m6a'] = X_a['is_stress'] * X_a['m6a_per_kb']
X_a['stress_x_pas'] = X_a['is_stress'] * X_a['has_canonical_pas']
X_a = sm.add_constant(X_a)
y = df['polya_length']
mask = ~(X_a.isna().any(axis=1) | y.isna())
model_a = sm.OLS(y[mask], X_a[mask]).fit()
print(f"  R²: {model_a.rsquared:.4f}")
print(f"  stress×m6A/kb: coef={model_a.params['stress_x_m6a']:+.3f}, P={model_a.pvalues['stress_x_m6a']:.2e}")

# Model B: Split into 3'-proximal and rest
print("\n--- Model B: 3'-proximal vs rest m6A (split) ---")
X_b = df[['is_stress', 'has_canonical_pas']].copy()
X_b['read_length_kb'] = df['read_length'] / 1000
X_b['m6a_3prime_per_kb'] = df['m6a_3prime_per_kb']
X_b['m6a_rest_per_kb'] = df['m6a_rest_per_kb']
X_b['stress_x_m6a_3prime'] = X_b['is_stress'] * X_b['m6a_3prime_per_kb']
X_b['stress_x_m6a_rest'] = X_b['is_stress'] * X_b['m6a_rest_per_kb']
X_b['stress_x_pas'] = X_b['is_stress'] * X_b['has_canonical_pas']
X_b = sm.add_constant(X_b)
mask = ~(X_b.isna().any(axis=1) | y.isna())
model_b = sm.OLS(y[mask], X_b[mask]).fit()
print(f"  R²: {model_b.rsquared:.4f}")
print(f"  stress×m6A_3prime: coef={model_b.params['stress_x_m6a_3prime']:+.3f}, P={model_b.pvalues['stress_x_m6a_3prime']:.2e}")
print(f"  stress×m6A_rest:   coef={model_b.params['stress_x_m6a_rest']:+.3f}, P={model_b.pvalues['stress_x_m6a_rest']:.2e}")

# Is the 3'-proximal coefficient significantly larger?
from scipy.stats import norm
se_3p = model_b.bse['stress_x_m6a_3prime']
se_rest = model_b.bse['stress_x_m6a_rest']
coef_3p = model_b.params['stress_x_m6a_3prime']
coef_rest = model_b.params['stress_x_m6a_rest']
z = (coef_3p - coef_rest) / np.sqrt(se_3p**2 + se_rest**2)
p_diff = 2 * (1 - norm.cdf(abs(z)))
print(f"\n  3' vs rest coefficient difference: z={z:.3f}, P={p_diff:.4f}")
print(f"  3' coef / rest coef: {coef_3p / coef_rest:.2f}x" if coef_rest != 0 else "  rest coef = 0")

# ============================================================
# Analysis 6: Stratify by m6A position pattern
# ============================================================
print("\n" + "=" * 70)
print("Analysis 6: m6A near 3' end stratification under stress")
print("=" * 70)

# Binary: has m6A in last 200bp (yes/no)
df['m6a_near_3prime'] = (df['m6a_in_last_200bp'] > 0).astype(int)

stress_df = df[df['is_stress'] == 1]
normal_df = df[df['is_stress'] == 0]

print(f"\nStress condition:")
for has_near, label in [(1, 'm6A in last 200bp'), (0, 'No m6A in last 200bp')]:
    sub = stress_df[stress_df['m6a_near_3prime'] == has_near]
    print(f"  {label}: n={len(sub)}, median poly(A)={sub['polya_length'].median():.1f}, "
          f"decay zone={( sub['polya_length'] < 30).mean():.1%}")

print(f"\nNormal condition:")
for has_near, label in [(1, 'm6A in last 200bp'), (0, 'No m6A in last 200bp')]:
    sub = normal_df[normal_df['m6a_near_3prime'] == has_near]
    print(f"  {label}: n={len(sub)}, median poly(A)={sub['polya_length'].median():.1f}")

# 4-way: m6A near 3' × stress
print("\n4-way comparison:")
print(f"{'Group':<30} {'N':>6} {'Median polyA':>14} {'Decay%':>8}")
print("-" * 60)
for cond, cond_label in [('normal', 'Normal'), ('stress', 'Stress')]:
    for has_near in [0, 1]:
        mask = (df['condition'] == cond) & (df['m6a_near_3prime'] == has_near)
        sub = df.loc[mask, 'polya_length']
        near_label = '+m6A near 3\'' if has_near else '-m6A near 3\''
        print(f"  {cond_label} / {near_label:<20} {len(sub):>6} {sub.median():>14.1f} {(sub < 30).mean():>8.1%}")

# ============================================================
# Analysis 7: Full combined model
# ============================================================
print("\n" + "=" * 70)
print("Analysis 7: Full combined model (m6A density + position + PAS)")
print("=" * 70)

X_full = df[['is_stress', 'has_canonical_pas']].copy()
X_full['read_length_kb'] = df['read_length'] / 1000
X_full['m6a_per_kb'] = df['m6a_per_kb']
X_full['m6a_near_3prime'] = df['m6a_near_3prime']
X_full['stress_x_m6a'] = X_full['is_stress'] * X_full['m6a_per_kb']
X_full['stress_x_pas'] = X_full['is_stress'] * X_full['has_canonical_pas']
X_full['stress_x_near'] = X_full['is_stress'] * X_full['m6a_near_3prime']
X_full = sm.add_constant(X_full)

model_full = sm.OLS(y[mask], X_full[mask]).fit()
print(f"R²: {model_full.rsquared:.4f}")
print(f"\n{'Term':<25} {'Coef':>8} {'P':>12}")
print("-" * 50)
for term in ['stress_x_m6a', 'stress_x_pas', 'stress_x_near']:
    print(f"  {term:<25} {model_full.params[term]:>+8.3f} {model_full.pvalues[term]:>12.2e}")

# ============================================================
# Analysis 8: Quad stratification (m6A density × position × PAS × stress)
# ============================================================
print("\n" + "=" * 70)
print("Analysis 8: Quad stratification under stress")
print("=" * 70)

stress_df2 = df[df['is_stress'] == 1].copy()
m6a_med = stress_df2['m6a_per_kb'].median()
stress_df2['m6a_high'] = (stress_df2['m6a_per_kb'] >= m6a_med).astype(int)

header_near = "m6A near 3'"
print(f"\n{'m6A density':>12} {header_near:>14} {'PAS':>5} {'N':>6} {'Median polyA':>14} {'Decay%':>8}")
print("-" * 70)

quad_results = []
for m6a_label, m6a_val in [('low', 0), ('high', 1)]:
    for near_label, near_val in [('no', 0), ('yes', 1)]:
        for pas_label, pas_val in [('no', 0), ('yes', 1)]:
            mask = ((stress_df2['m6a_high'] == m6a_val) &
                    (stress_df2['m6a_near_3prime'] == near_val) &
                    (stress_df2['has_canonical_pas'] == pas_val))
            sub = stress_df2.loc[mask, 'polya_length']
            if len(sub) < 10:
                continue
            decay_pct = (sub < 30).mean()
            quad_results.append({
                'm6a_density': m6a_label, 'm6a_near_3prime': near_label,
                'pas': pas_label, 'n': len(sub),
                'median_polya': sub.median(), 'decay_pct': decay_pct
            })
            print(f"{m6a_label:>12} {near_label:>14} {pas_label:>5} {len(sub):>6} {sub.median():>14.1f} {decay_pct:>8.1%}")

df_quad = pd.DataFrame(quad_results).sort_values('median_polya')
if len(df_quad) >= 2:
    best = df_quad.iloc[-1]
    worst = df_quad.iloc[0]
    print(f"\nMost protected:  m6A={best['m6a_density']}, near3'={best['m6a_near_3prime']}, PAS={best['pas']} → {best['median_polya']:.1f} nt, decay={best['decay_pct']:.1%}")
    print(f"Most vulnerable: m6A={worst['m6a_density']}, near3'={worst['m6a_near_3prime']}, PAS={worst['pas']} → {worst['median_polya']:.1f} nt, decay={worst['decay_pct']:.1%}")
    print(f"Range: {best['median_polya'] - worst['median_polya']:.1f} nt")

# ============================================================
# Save results
# ============================================================
pd.DataFrame(quad_results).to_csv(f'{OUT_DIR}/m6a_position_quad_stratification.tsv', sep='\t', index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Key question: Does m6A POSITION on L1 RNA matter for poly(A) protection?

Reads analyzed: {len(df)} ancient L1 (Normal: {(df['condition']=='normal').sum()}, Stress: {(df['condition']=='stress').sum()})
Reads with m6A: {len(df_m6a)}

Position hypothesis: m6A near poly(A) tail (3' end) → more protective
because YTHDF readers near 3' end can directly shield poly(A) from
deadenylases or recruit PABPC1.
""")
print("Done!")
