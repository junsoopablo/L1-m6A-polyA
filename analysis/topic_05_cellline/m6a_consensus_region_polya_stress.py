#!/usr/bin/env python3
"""Test: Do ancient L1 fragments from high-m6A regions (ORF1/2) show less
poly(A) shortening under arsenite stress than those from low-m6A regions (3'UTR)?

Logic:
  - ORF1 m6A/kb = 6.25 (highest), 3'UTR = 4.00 (lowest)
  - m6A dose-dependent poly(A) protection under stress (OLS p=2.7e-05)
  - Therefore: ORF1-derived fragments should be less vulnerable

Needs: HeLa + HeLa-Ars L1 reads with poly(A) + consensus position mapping.
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
CACHE_DIR = TOPIC_05 / 'part3_l1_per_read_cache'
RMSK = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT/reference/rmsk.txt.gz')
OUTDIR = TOPIC_05 / 'm6a_consensus_position'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

REGIONS = [
    ("5'UTR", 0, 0.15),
    ("ORF1", 0.15, 0.32),
    ("ORF2", 0.32, 0.96),
    ("3'UTR", 0.96, 1.0),
]

# HeLa groups (normal + arsenite)
HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3']
ARS_GROUPS = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

# =====================================================================
# 1. Build rmsk lookup
# =====================================================================
print("Loading rmsk...")
rmsk_lookup = {}

with gzip.open(RMSK, 'rt') as f:
    for line in f:
        fields = line.strip().split('\t')
        if len(fields) < 16:
            continue
        if fields[11] != 'LINE' or fields[12] != 'L1':
            continue

        chrom = fields[5]
        geno_start = int(fields[6])
        geno_end = int(fields[7])
        strand = fields[9]
        rep_name = fields[10]
        rep_start = int(fields[13])
        rep_end = int(fields[14])
        rep_left = int(fields[15])

        if strand == '+':
            cons_start = rep_start
            cons_end = rep_end
            cons_length = rep_end + abs(rep_left)
        else:
            cons_start = rep_left
            cons_end = rep_end
            cons_length = rep_end + abs(rep_start)

        if cons_length <= 0:
            continue

        key = (chrom, geno_start, geno_end, strand, rep_name)
        rmsk_lookup[key] = (cons_start, cons_end, cons_length)

print(f"  {len(rmsk_lookup)} entries")

# =====================================================================
# 2. Load HeLa + HeLa-Ars reads with poly(A) and m6A
# =====================================================================
print("\nLoading HeLa and HeLa-Ars reads...")

def load_groups(group_list, condition_label):
    all_reads = []
    for grp in group_list:
        summary_f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        cache_f = CACHE_DIR / f'{grp}_l1_per_read.tsv'
        if not summary_f.exists() or not cache_f.exists():
            print(f"  SKIP {grp}: missing files")
            continue

        summary = pd.read_csv(summary_f, sep='\t', usecols=[
            'read_id', 'chr', 'start', 'end', 'read_length',
            'te_start', 'te_end', 'gene_id', 'te_strand', 'read_strand',
            'polya_length', 'qc_tag',
        ])
        summary = summary[summary['qc_tag'] == 'PASS'].copy()

        cache = pd.read_csv(cache_f, sep='\t', usecols=[
            'read_id', 'read_length', 'm6a_sites_high',
        ])

        merged = summary.merge(cache[['read_id', 'm6a_sites_high']],
                               on='read_id', how='inner')
        merged['group'] = grp
        merged['condition'] = condition_label
        merged['is_young'] = merged['gene_id'].isin(YOUNG)
        merged['age'] = np.where(merged['is_young'], 'young', 'ancient')
        merged['te_length'] = merged['te_end'] - merged['te_start']
        merged['m6a_per_kb'] = merged['m6a_sites_high'] / merged['read_length'] * 1000
        all_reads.append(merged)

    return pd.concat(all_reads, ignore_index=True) if all_reads else pd.DataFrame()

hela = load_groups(HELA_GROUPS, 'Normal')
ars = load_groups(ARS_GROUPS, 'Arsenite')

df = pd.concat([hela, ars], ignore_index=True)
print(f"  HeLa: {len(hela)}, HeLa-Ars: {len(ars)}, Total: {len(df)}")

# =====================================================================
# 3. Map to consensus positions
# =====================================================================
print("\nMapping to consensus positions...")

cons_data = []
for _, row in df.iterrows():
    key = (row['chr'], row['te_start'], row['te_end'], row['te_strand'], row['gene_id'])
    if key in rmsk_lookup:
        cs, ce, cl = rmsk_lookup[key]
        cons_data.append((cs, ce, cl, (cs + ce) / 2 / cl))
    else:
        cons_data.append((np.nan, np.nan, np.nan, np.nan))

df['cons_start'] = [x[0] for x in cons_data]
df['cons_end'] = [x[1] for x in cons_data]
df['cons_length'] = [x[2] for x in cons_data]
df['frac_center'] = [x[3] for x in cons_data]

df = df[df['cons_length'].notna()].copy()

def assign_region(fc):
    for label, lo, hi in REGIONS:
        if lo <= fc < hi:
            return label
    return 'unknown'

df['region'] = df['frac_center'].apply(assign_region)
df = df[df['region'] != 'unknown'].copy()

print(f"  Matched: {len(df)}")

# =====================================================================
# 4. Ancient L1: poly(A) by condition × consensus region
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 1: ANCIENT L1 — POLY(A) BY CONDITION × CONSENSUS REGION")
print("=" * 80)

ancient = df[df['age'] == 'ancient'].copy()
# Filter to reads with valid poly(A)
ancient = ancient[ancient['polya_length'] > 0].copy()

print(f"\nAncient L1 with poly(A): Normal={len(ancient[ancient['condition']=='Normal'])}, "
      f"Arsenite={len(ancient[ancient['condition']=='Arsenite'])}")

print(f"\n{'Region':<10s} {'Norm_n':>7s} {'Norm_pA':>8s} {'Ars_n':>7s} {'Ars_pA':>8s} "
      f"{'Δ_pA':>7s} {'p':>10s} {'m6A/kb_N':>9s} {'m6A/kb_A':>9s}")
print("-" * 85)

region_results = []
for label, lo, hi in REGIONS:
    norm = ancient[(ancient['condition'] == 'Normal') & (ancient['region'] == label)]
    stress = ancient[(ancient['condition'] == 'Arsenite') & (ancient['region'] == label)]

    if len(norm) >= 5 and len(stress) >= 5:
        norm_pa = norm['polya_length'].median()
        ars_pa = stress['polya_length'].median()
        delta = ars_pa - norm_pa
        _, p = stats.mannwhitneyu(norm['polya_length'], stress['polya_length'],
                                   alternative='two-sided')
        norm_m6a = norm['m6a_per_kb'].median()
        ars_m6a = stress['m6a_per_kb'].median()

        print(f"  {label:<10s} {len(norm):>7d} {norm_pa:>8.1f} {len(stress):>7d} {ars_pa:>8.1f} "
              f"{delta:>7.1f} {p:>10.2e} {norm_m6a:>9.2f} {ars_m6a:>9.2f}")

        region_results.append({
            'region': label, 'norm_n': len(norm), 'ars_n': len(stress),
            'norm_polya': norm_pa, 'ars_polya': ars_pa,
            'delta_polya': delta, 'p_value': p,
            'norm_m6a_per_kb': norm_m6a, 'ars_m6a_per_kb': ars_m6a,
        })

# =====================================================================
# 5. Correlation: region m6A/kb vs delta poly(A)
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 2: REGION m6A/kb vs POLY(A) SHORTENING")
print("=" * 80)

if len(region_results) >= 3:
    m6a_vals = [r['norm_m6a_per_kb'] for r in region_results]
    delta_vals = [r['delta_polya'] for r in region_results]
    r, p = stats.spearmanr(m6a_vals, delta_vals)
    print(f"\n  Spearman correlation (region m6A/kb vs Δpoly(A)): r={r:.3f}, p={p:.3f}")
    print(f"  (Positive r means higher m6A → less shortening = more protection)")

    for res in region_results:
        print(f"    {res['region']:<10s}: m6A/kb={res['norm_m6a_per_kb']:.2f}, Δpoly(A)={res['delta_polya']:.1f}")

# =====================================================================
# 6. Read-level OLS within ancient: m6A/kb × condition → poly(A)
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 3: READ-LEVEL OLS — m6A/kb × STRESS → POLY(A) (ANCIENT ONLY)")
print("=" * 80)

import statsmodels.api as sm

ancient_ols = ancient.copy()
ancient_ols['is_stress'] = (ancient_ols['condition'] == 'Arsenite').astype(int)
ancient_ols['stress_x_m6a'] = ancient_ols['is_stress'] * ancient_ols['m6a_per_kb']

X = ancient_ols[['is_stress', 'm6a_per_kb', 'stress_x_m6a', 'read_length']].copy()
X = sm.add_constant(X)
y = ancient_ols['polya_length']

model = sm.OLS(y, X).fit()
print(f"\n  OLS: poly(A) ~ stress + m6A/kb + stress×m6A/kb + read_length")
print(f"  N = {len(ancient_ols)}")
for var in ['is_stress', 'm6a_per_kb', 'stress_x_m6a']:
    coef = model.params[var]
    pval = model.pvalues[var]
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"    {var:<20s}: coef={coef:>8.3f}, p={pval:.2e} {sig}")

# =====================================================================
# 7. Per-region: poly(A) shortening stratified by m6A quartile
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 4: ORF2-DERIVED ANCIENT — m6A QUARTILE × STRESS POLY(A)")
print("=" * 80)

# ORF2 has the most reads, so best powered
orf2 = ancient[ancient['region'] == 'ORF2'].copy()
print(f"\n  ORF2-derived ancient: Normal={len(orf2[orf2['condition']=='Normal'])}, "
      f"Ars={len(orf2[orf2['condition']=='Arsenite'])}")

# Quartiles based on normal condition m6A/kb
for condition in ['Normal', 'Arsenite']:
    cond_data = orf2[orf2['condition'] == condition]
    quartiles = pd.qcut(cond_data['m6a_per_kb'], 4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'],
                         duplicates='drop')
    print(f"\n  {condition}:")
    for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
        qd = cond_data[quartiles == q]
        if len(qd) > 0:
            print(f"    {q}: n={len(qd):>4d}, poly(A)={qd['polya_length'].median():>6.1f}, "
                  f"m6A/kb={qd['m6a_per_kb'].median():.2f}")

# =====================================================================
# 8. Compare ORF1+ORF2 vs 3'UTR shortening directly
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 5: ORF1+ORF2 vs 3'UTR — DIFFERENTIAL SHORTENING")
print("=" * 80)

for region_label, region_filter in [("ORF1+ORF2", ["ORF1", "ORF2"]),
                                      ("3'UTR", ["3'UTR"]),
                                      ("5'UTR", ["5'UTR"])]:
    sub = ancient[ancient['region'].isin(region_filter)]
    norm = sub[sub['condition'] == 'Normal']['polya_length']
    stress = sub[sub['condition'] == 'Arsenite']['polya_length']

    if len(norm) >= 5 and len(stress) >= 5:
        delta = stress.median() - norm.median()
        _, p = stats.mannwhitneyu(norm, stress, alternative='two-sided')
        pct_change = delta / norm.median() * 100
        print(f"\n  {region_label}:")
        print(f"    Normal: {norm.median():.1f} nt (n={len(norm)})")
        print(f"    Arsenite: {stress.median():.1f} nt (n={len(stress)})")
        print(f"    Δ = {delta:.1f} nt ({pct_change:.1f}%), p={p:.2e}")
        print(f"    m6A/kb (Normal): {sub[sub['condition']=='Normal']['m6a_per_kb'].median():.2f}")

# =====================================================================
# 9. Young L1 for reference
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 6: YOUNG L1 (REFERENCE) — BY REGION")
print("=" * 80)

young = df[(df['age'] == 'young') & (df['polya_length'] > 0)].copy()
print(f"\nYoung L1 with poly(A): Normal={len(young[young['condition']=='Normal'])}, "
      f"Ars={len(young[young['condition']=='Arsenite'])}")

for label in ["ORF2", "3'UTR"]:
    sub = young[young['region'] == label]
    norm = sub[sub['condition'] == 'Normal']['polya_length']
    stress = sub[sub['condition'] == 'Arsenite']['polya_length']
    if len(norm) >= 3 and len(stress) >= 3:
        delta = stress.median() - norm.median()
        _, p = stats.mannwhitneyu(norm, stress, alternative='two-sided')
        print(f"  {label}: Normal={norm.median():.1f}(n={len(norm)}), "
              f"Ars={stress.median():.1f}(n={len(stress)}), Δ={delta:.1f}, p={p:.2e}")

# Save
results_df = pd.DataFrame(region_results)
results_df.to_csv(OUTDIR / 'ancient_polya_by_consensus_region_stress.tsv', sep='\t', index=False)

print(f"\n\nResults saved to: {OUTDIR}")
print("Done!")
