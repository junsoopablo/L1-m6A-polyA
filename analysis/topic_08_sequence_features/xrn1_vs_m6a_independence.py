#!/usr/bin/env python3
"""
XRN1 decay vs m6A-poly(A) protection: independent pathways?
Are these targeting different L1 populations or the same L1 from different ends?

Key questions:
1. XRN1 KD → which L1 subfamilies/ages gain most? → m6A/kb of gained reads?
2. Under stress, do 5'-truncated reads (short) differ in m6A from long reads?
3. Loci-level: XRN1-sensitive loci vs m6A-protected loci — overlap?
4. Sequence features that predict XRN1 sensitivity vs m6A-poly(A) protection
"""
import pandas as pd
import numpy as np
from scipy import stats

# ── Load XRN1 factorial data ──
XRN1_DIR = '/vault/external-datasets/2026/PRJNA842344_HeLA_under_oxidative-stress_RNA002/xrn1_analysis/analysis'
df = pd.read_csv(f'{XRN1_DIR}/xrn1_per_read_with_subfamily.tsv', sep='\t')
df = df.dropna(subset=['polya_length'])

print("="*70)
print("ANALYSIS: XRN1 decay vs m6A-poly(A) protection independence")
print("="*70)

# ── 1. m6A/kb distribution by condition ──
print("\n1. m6A/kb BY CONDITION (population-level)")
print("-"*50)
for cond in ['mock', 'XRN1', 'Ars', 'Ars_XRN1']:
    sub = df[df['condition'] == cond]
    anc = sub[sub['l1_age'] == 'ancient']
    print(f"  {cond:12s}: n={len(sub):4d}, m6A/kb={sub['m6a_per_kb'].median():.2f} (med), "
          f"mean={sub['m6a_per_kb'].mean():.2f}, ancient n={len(anc)}")

# ── 2. XRN1-gained reads: m6A profile ──
print("\n2. XRN1 KD GAINED READS — m6A PROFILE")
print("-"*50)
# If XRN1 preferentially degrades a specific m6A population,
# the reads present under XRN1 KD (but not mock) should have
# different m6A than mock reads.
# Approach: compare m6A/kb distributions
mock_m6a = df[df['condition'] == 'mock']['m6a_per_kb']
xrn1_m6a = df[df['condition'] == 'XRN1']['m6a_per_kb']
ars_m6a = df[df['condition'] == 'Ars']['m6a_per_kb']
arsxrn1_m6a = df[df['condition'] == 'Ars_XRN1']['m6a_per_kb']

ks_xrn1, p_xrn1 = stats.ks_2samp(mock_m6a, xrn1_m6a)
ks_ars, p_ars = stats.ks_2samp(mock_m6a, ars_m6a)
print(f"  Mock vs XRN1 KD m6A/kb: KS={ks_xrn1:.3f}, P={p_xrn1:.3e}")
print(f"  Mock vs Ars m6A/kb: KS={ks_ars:.3f}, P={p_ars:.3e}")

# Stratify by m6A quartiles — does XRN1 KD change the quartile distribution?
m6a_q = pd.qcut(df['m6a_per_kb'], 4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'],
                duplicates='drop')
df['m6a_quartile'] = m6a_q

print("\n  m6A quartile distribution by condition:")
for cond in ['mock', 'XRN1', 'Ars', 'Ars_XRN1']:
    sub = df[df['condition'] == cond]
    qcounts = sub['m6a_quartile'].value_counts(normalize=True).sort_index()
    pcts = [f"{qcounts.get(q, 0)*100:.1f}" for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']]
    print(f"    {cond:12s}: Q1={pcts[0]}% Q2={pcts[1]}% Q3={pcts[2]}% Q4={pcts[3]}%")

# ── 3. Read length (5' truncation proxy) vs m6A/kb ──
print("\n3. READ LENGTH (5' TRUNCATION PROXY) vs m6A")
print("-"*50)
for cond in ['mock', 'XRN1', 'Ars', 'Ars_XRN1']:
    sub = df[df['condition'] == cond]
    short = sub[sub['qlen'] < 500]
    long = sub[sub['qlen'] >= 1000]
    r, p = stats.spearmanr(sub['qlen'], sub['m6a_per_kb'])
    print(f"  {cond:12s}: rlen~m6A Spearman r={r:.3f} P={p:.3e}")
    if len(short) > 10 and len(long) > 10:
        print(f"    short(<500bp) m6A/kb={short['m6a_per_kb'].median():.2f} (n={len(short)}), "
              f"long(≥1kb) m6A/kb={long['m6a_per_kb'].median():.2f} (n={len(long)})")

# Under Ars: do 5'-truncated reads (short) ALSO have short poly(A)?
print("\n  Under Ars: 5' truncation × poly(A) shortening")
ars = df[df['condition'] == 'Ars'].copy()
ars['short_read'] = ars['qlen'] < 500
ars['short_polya'] = ars['polya_length'] < 50
ct = pd.crosstab(ars['short_read'], ars['short_polya'])
print(f"  Crosstab (short_read × short_polyA):\n{ct}")
if ct.shape == (2, 2):
    or_val = (ct.iloc[1, 1] * ct.iloc[0, 0]) / (ct.iloc[1, 0] * ct.iloc[0, 1]) if ct.iloc[1, 0] * ct.iloc[0, 1] > 0 else np.inf
    print(f"  Odds ratio = {or_val:.2f}")

# ── 4. m6A-stratified poly(A) under each condition ──
print("\n4. m6A-STRATIFIED POLY(A) BY CONDITION")
print("-"*50)
print("  (Does m6A protect poly(A) equally with and without XRN1?)")
for cond in ['mock', 'XRN1', 'Ars', 'Ars_XRN1']:
    sub = df[(df['condition'] == cond) & (df['l1_age'] == 'ancient')]
    if len(sub) < 20:
        continue
    low = sub[sub['m6a_per_kb'] <= sub['m6a_per_kb'].median()]
    high = sub[sub['m6a_per_kb'] > sub['m6a_per_kb'].median()]
    r, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
    print(f"  {cond:12s}: m6A~polyA r={r:.3f} (P={p:.3e}), "
          f"low-m6A med_pA={low['polya_length'].median():.1f} (n={len(low)}), "
          f"high-m6A med_pA={high['polya_length'].median():.1f} (n={len(high)})")

# ── 5. Subfamily-level analysis ──
print("\n5. SUBFAMILY-LEVEL: XRN1 SENSITIVITY vs m6A DENSITY")
print("-"*50)
# For each subfamily, compute:
# (a) XRN1 sensitivity = RPM_XRN1KD / RPM_mock
# (b) mean m6A/kb
subfam_stats = []
for sf in df['subfamily'].unique():
    mock_n = len(df[(df['condition'] == 'mock') & (df['subfamily'] == sf)])
    xrn1_n = len(df[(df['condition'] == 'XRN1') & (df['subfamily'] == sf)])
    ars_n = len(df[(df['condition'] == 'Ars') & (df['subfamily'] == sf)])
    m6a_mean = df[df['subfamily'] == sf]['m6a_per_kb'].mean()
    polya_mock = df[(df['condition'] == 'mock') & (df['subfamily'] == sf)]['polya_length'].median()
    polya_ars = df[(df['condition'] == 'Ars') & (df['subfamily'] == sf)]['polya_length'].median()

    total_n = mock_n + xrn1_n + ars_n
    if total_n >= 10:
        # Normalize by library size
        mock_total = len(df[df['condition'] == 'mock'])
        xrn1_total = len(df[df['condition'] == 'XRN1'])
        ars_total = len(df[df['condition'] == 'Ars'])

        xrn1_fc = (xrn1_n / xrn1_total) / (mock_n / mock_total) if mock_n > 0 else np.nan
        ars_polya_delta = polya_ars - polya_mock if not np.isnan(polya_ars) and not np.isnan(polya_mock) else np.nan

        subfam_stats.append({
            'subfamily': sf,
            'age': df[df['subfamily'] == sf]['l1_age'].mode().values[0],
            'mock_n': mock_n,
            'xrn1_n': xrn1_n,
            'xrn1_fc': xrn1_fc,
            'm6a_per_kb': m6a_mean,
            'polya_mock': polya_mock,
            'ars_polya_delta': ars_polya_delta,
            'total_reads': total_n,
        })

sf_df = pd.DataFrame(subfam_stats)
sf_df = sf_df.sort_values('total_reads', ascending=False)

# Correlation: XRN1 sensitivity vs m6A/kb
valid = sf_df.dropna(subset=['xrn1_fc', 'm6a_per_kb'])
if len(valid) >= 5:
    r, p = stats.spearmanr(valid['xrn1_fc'], valid['m6a_per_kb'])
    print(f"  Subfamily-level: XRN1 FC ~ m6A/kb: r={r:.3f}, P={p:.3e} (n={len(valid)} subfamilies)")

# Correlation: XRN1 sensitivity vs Ars poly(A) delta
valid2 = sf_df.dropna(subset=['xrn1_fc', 'ars_polya_delta'])
if len(valid2) >= 5:
    r2, p2 = stats.spearmanr(valid2['xrn1_fc'], valid2['ars_polya_delta'])
    print(f"  Subfamily-level: XRN1 FC ~ Ars polyA Δ: r={r2:.3f}, P={p2:.3e}")

# Top subfamilies
print("\n  Top 15 subfamilies by read count:")
print(f"  {'Subfamily':15s} {'Age':8s} {'mock':>5s} {'XRN1':>5s} {'FC':>6s} {'m6A/kb':>7s} {'pA_mock':>8s} {'Ars_Δ':>7s}")
for _, row in sf_df.head(15).iterrows():
    fc_str = f"{row['xrn1_fc']:.2f}" if not np.isnan(row['xrn1_fc']) else "N/A"
    delta_str = f"{row['ars_polya_delta']:.0f}" if not np.isnan(row['ars_polya_delta']) else "N/A"
    pA_str = f"{row['polya_mock']:.0f}" if not np.isnan(row['polya_mock']) else "N/A"
    print(f"  {row['subfamily']:15s} {row['age']:8s} {row['mock_n']:5d} {row['xrn1_n']:5d} "
          f"{fc_str:>6s} {row['m6a_per_kb']:7.2f} {pA_str:>8s} {delta_str:>7s}")

# ── 6. Dual vulnerability score ──
print("\n6. DUAL VULNERABILITY: 5' decay × 3' shortening")
print("-"*50)
# Under arsenite: reads that are BOTH short (5' truncated) AND have short poly(A) (3' shortened)
# vs reads that are ONLY one or the other
ars_anc = df[(df['condition'] == 'Ars') & (df['l1_age'] == 'ancient')].copy()
mock_anc = df[(df['condition'] == 'mock') & (df['l1_age'] == 'ancient')].copy()

# Define thresholds
rl_med = mock_anc['qlen'].median()
pa_med = mock_anc['polya_length'].median()

for cond, sub in [('mock', mock_anc), ('Ars', ars_anc)]:
    sub = sub.copy()
    sub['truncated_5p'] = sub['qlen'] < rl_med * 0.5  # severely short reads
    sub['shortened_3p'] = sub['polya_length'] < pa_med * 0.5  # severely short poly(A)

    both = sub['truncated_5p'] & sub['shortened_3p']
    only_5p = sub['truncated_5p'] & ~sub['shortened_3p']
    only_3p = ~sub['truncated_5p'] & sub['shortened_3p']
    neither = ~sub['truncated_5p'] & ~sub['shortened_3p']

    print(f"\n  {cond}: rl_threshold={rl_med*0.5:.0f}bp, pA_threshold={pa_med*0.5:.0f}nt")
    print(f"    Both 5'+3': {both.sum():4d} ({both.mean()*100:.1f}%), m6A/kb={sub.loc[both, 'm6a_per_kb'].median():.2f}")
    print(f"    Only 5':    {only_5p.sum():4d} ({only_5p.mean()*100:.1f}%), m6A/kb={sub.loc[only_5p, 'm6a_per_kb'].median():.2f}")
    print(f"    Only 3':    {only_3p.sum():4d} ({only_3p.mean()*100:.1f}%), m6A/kb={sub.loc[only_3p, 'm6a_per_kb'].median():.2f}")
    print(f"    Neither:    {neither.sum():4d} ({neither.mean()*100:.1f}%), m6A/kb={sub.loc[neither, 'm6a_per_kb'].median():.2f}")

# ── 7. Does XRN1 KD change the m6A-poly(A) relationship? ──
print("\n7. m6A-POLY(A) COUPLING: DOES XRN1 STATUS CHANGE IT?")
print("-"*50)
print("  (If XRN1 and m6A-poly(A) are truly independent,")
print("   m6A-poly(A) correlation should be similar ±XRN1)")

for cond in ['Ars', 'Ars_XRN1']:
    sub = df[(df['condition'] == cond) & (df['l1_age'] == 'ancient')]
    sub = sub.dropna(subset=['m6a_per_kb', 'polya_length'])
    if len(sub) >= 20:
        r, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
        # Quartile analysis
        q1 = sub[sub['m6a_per_kb'] <= sub['m6a_per_kb'].quantile(0.25)]['polya_length'].median()
        q4 = sub[sub['m6a_per_kb'] >= sub['m6a_per_kb'].quantile(0.75)]['polya_length'].median()
        print(f"  {cond:12s}: r={r:.3f} (P={p:.3e}), Q1_pA={q1:.1f}, Q4_pA={q4:.1f}, Δ={q4-q1:.1f}")

# ── 8. Load ancient_l1_with_features for HeLa sequence features ──
print("\n8. SEQUENCE FEATURES: PREDICTORS OF EACH PATHWAY")
print("-"*50)
feat_path = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_sequence_features/ancient_l1_with_features.tsv'
df_feat = pd.read_csv(feat_path, sep='\t')
df_feat = df_feat.dropna(subset=['polya_length'])

# Poly(A) shortening sensitivity score (per-read):
# Ars poly(A) residual after regressing out normal poly(A)
normal = df_feat[df_feat['is_stress'] == 0]
stress = df_feat[df_feat['is_stress'] == 1]

# What sequence features predict poly(A) under stress?
features_to_test = ['gc_content', 'a_fraction', 'has_canonical_pas', 'has_any_pas',
                    'm6a_per_kb', 'rbp_ZAP_CpG_per_kb', 'rbp_HuR_ARE_per_kb',
                    'rbp_YTHDF_per_kb', 'max_a_run', 'read_length']

print("\n  Spearman correlations with poly(A) under STRESS (ancient L1):")
print(f"  {'Feature':25s} {'r':>7s} {'P':>10s}")
for feat in features_to_test:
    if feat in stress.columns:
        valid = stress.dropna(subset=[feat, 'polya_length'])
        if len(valid) > 20:
            r, p = stats.spearmanr(valid[feat], valid['polya_length'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  {feat:25s} {r:+7.3f} {p:10.2e} {sig}")

print("\n  Same under NORMAL (for comparison):")
for feat in features_to_test:
    if feat in normal.columns:
        valid = normal.dropna(subset=[feat, 'polya_length'])
        if len(valid) > 20:
            r, p = stats.spearmanr(valid[feat], valid['polya_length'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  {feat:25s} {r:+7.3f} {p:10.2e} {sig}")

# ── 9. XRN1 sensitivity by sequence features (if available) ──
print("\n9. FEATURES OF XRN1-SENSITIVE L1 (from XRN1 factorial data)")
print("-"*50)
# Can we cross-reference? XRN1 data doesn't have the same features.
# But it has: subfamily, age, qlen, m6a_per_kb, polya_length
# XRN1-sensitive = enriched under XRN1 KD

# Compare XRN1 KD-gained reads vs mock reads
print("  Reads gained under XRN1 KD vs mock (population characteristics):")
mock_reads = df[df['condition'] == 'mock']
xrn1_reads = df[df['condition'] == 'XRN1']

for metric in ['qlen', 'm6a_per_kb', 'polya_length']:
    m_val = mock_reads[metric].median()
    x_val = xrn1_reads[metric].median()
    ks, p = stats.ks_2samp(mock_reads[metric].dropna(), xrn1_reads[metric].dropna())
    print(f"    {metric:15s}: mock={m_val:.1f}, XRN1={x_val:.1f}, KS={ks:.3f}, P={p:.3e}")

# Young vs ancient sensitivity
print("\n  Young vs Ancient sensitivity to XRN1:")
for age in ['young', 'ancient']:
    m_n = len(mock_reads[mock_reads['l1_age'] == age])
    x_n = len(xrn1_reads[xrn1_reads['l1_age'] == age])
    m_frac = m_n / len(mock_reads) if len(mock_reads) > 0 else 0
    x_frac = x_n / len(xrn1_reads) if len(xrn1_reads) > 0 else 0
    fc = x_frac / m_frac if m_frac > 0 else np.nan
    print(f"    {age:8s}: mock {m_n:4d} ({m_frac*100:.1f}%), "
          f"XRN1 {x_n:4d} ({x_frac*100:.1f}%), FC={fc:.2f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
