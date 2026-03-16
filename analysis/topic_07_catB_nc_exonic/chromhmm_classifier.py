#!/usr/bin/env python3
"""
Test ChromHMM chromatin state as GENCODE-independent classifier for L1 autonomy.

Key idea: H3K36me3 (captured by ChromHMM "Transcribed" state) marks actively
transcribed gene bodies. L1 in "Transcribed" regions = within host gene body.
This is epigenetic, not annotation-based → GENCODE-independent.

Tests:
1. Annotate Cat B reads with ChromHMM (same method as PASS in topic_08)
2. Within each ChromHMM state, compare PASS vs Cat B arsenite response
3. Test if ChromHMM "Transcribed" alone can explain Cat B immunity
4. OLS with ChromHMM + overlap_frac
"""

import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import statsmodels.formula.api as smf

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
TOPIC_08 = PROJECT / 'analysis/01_exploration/topic_08_regulatory_chromatin'
OUTDIR = TOPIC_07 / 'unified_classifier'
OUTDIR.mkdir(exist_ok=True)

CHROMHMM_BED = TOPIC_08 / 'E066_15_coreMarks_hg38lift_mnemonics.bed.gz'

# State grouping (same as topic_08)
STATE_GROUP = {
    '1_TssA': 'Promoter', '2_TssAFlnk': 'Promoter',
    '3_TxFlnk': 'Transcribed', '4_Tx': 'Transcribed', '5_TxWk': 'Transcribed',
    '6_EnhG': 'Enhancer', '7_Enh': 'Enhancer',
    '8_ZNF/Rpts': 'ZNF/Repeats',
    '9_Het': 'Heterochromatin',
    '10_TssBiv': 'Bivalent', '11_BivFlnk': 'Bivalent', '12_EnhBiv': 'Bivalent',
    '13_ReprPC': 'Repressed', '14_ReprPCWk': 'Repressed',
    '15_Quies': 'Quiescent',
}

# =========================================================================
# Step 1: Load unified dataset (PASS + Cat B, HeLa only)
# =========================================================================
print("Step 1: Loading unified dataset...")
unified = pd.read_csv(OUTDIR / 'unified_hela_ars_features.tsv', sep='\t')
print(f"  Total reads: {len(unified)}")

# =========================================================================
# Step 2: Annotate with ChromHMM using read midpoint (same as topic_08)
# =========================================================================
print("\nStep 2: Annotating with ChromHMM...")

# Create midpoint BED
tmp_bed = tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False, prefix='mid_')
for _, row in unified.iterrows():
    mid = (row['start'] + row['end']) // 2
    tmp_bed.write(f"{row['chr']}\t{mid}\t{mid+1}\t{row['read_id']}\n")
tmp_bed.close()

tmp_sorted = tmp_bed.name + '.sorted'
subprocess.run(f"sort -k1,1 -k2,2n {tmp_bed.name} > {tmp_sorted}", shell=True, check=True)

# Intersect
result = subprocess.run(
    f"bedtools intersect -a {tmp_sorted} -b {CHROMHMM_BED} -wo",
    shell=True, capture_output=True, text=True)

# Parse
chromhmm_map = {}
for line in result.stdout.strip().split('\n'):
    if not line:
        continue
    fields = line.split('\t')
    rid = fields[3]
    state = fields[7]  # ChromHMM state mnemonic
    chromhmm_map[rid] = state

import os
os.unlink(tmp_bed.name)
os.unlink(tmp_sorted)

unified['chromhmm_state'] = unified['read_id'].map(chromhmm_map)
unified['chromhmm_group'] = unified['chromhmm_state'].map(STATE_GROUP)

matched = unified['chromhmm_state'].notna().sum()
print(f"  ChromHMM matched: {matched}/{len(unified)} ({matched/len(unified)*100:.1f}%)")

# =========================================================================
# Step 3: ChromHMM distribution — PASS vs Cat B
# =========================================================================
print("\n" + "=" * 80)
print("ChromHMM DISTRIBUTION — PASS vs Cat B")
print("=" * 80)

for source in ['PASS', 'CatB']:
    sub = unified[unified['source'] == source]
    total = len(sub)
    print(f"\n  {source} (n={total}):")
    for grp in ['Transcribed', 'Quiescent', 'Enhancer', 'Promoter',
                 'Heterochromatin', 'Repressed', 'Bivalent', 'ZNF/Repeats']:
        n = (sub['chromhmm_group'] == grp).sum()
        pct = n / total * 100
        print(f"    {grp:20s}: {n:5d} ({pct:5.1f}%)")

# =========================================================================
# Step 4: KEY TEST — Within each ChromHMM state, PASS vs Cat B
# =========================================================================
print("\n" + "=" * 80)
print("KEY TEST: Within each ChromHMM state, PASS vs Cat B arsenite response")
print("=" * 80)

def ars_test(data, label=''):
    hela = data[data['cell_line'] == 'HeLa']['polya_length'].dropna()
    ars = data[data['cell_line'] == 'HeLa-Ars']['polya_length'].dropna()
    if len(hela) < 5 or len(ars) < 5:
        return {'d': np.nan, 'p': np.nan, 'n': len(hela) + len(ars),
                'hela_med': np.nan, 'ars_med': np.nan}
    _, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
    return {'d': ars.median() - hela.median(), 'p': p, 'n': len(hela) + len(ars),
            'hela_med': hela.median(), 'ars_med': ars.median()}

print(f"\n{'ChromHMM':15s} | {'PASS Δ':>8s} {'p':>10s} {'n':>6s} | "
      f"{'CatB Δ':>8s} {'p':>10s} {'n':>6s} | {'ΔΔ':>8s}")
print("-" * 85)

for grp in ['Transcribed', 'Quiescent', 'Enhancer', 'Promoter',
             'Heterochromatin', 'Repressed']:
    pass_sub = unified[(unified['source'] == 'PASS') & (unified['chromhmm_group'] == grp)]
    catb_sub = unified[(unified['source'] == 'CatB') & (unified['chromhmm_group'] == grp)]

    rp = ars_test(pass_sub)
    rc = ars_test(catb_sub)

    dd = (rp['d'] - rc['d']) if not (np.isnan(rp['d']) or np.isnan(rc['d'])) else np.nan

    def fmt(r):
        if np.isnan(r['d']):
            return f"{'':>8s} {'n/a':>10s} {r['n']:6d}"
        return f"{r['d']:+8.1f} {r['p']:10.2e} {r['n']:6d}"

    dd_str = f"{dd:+8.1f}" if not np.isnan(dd) else "    n/a"
    print(f"{grp:15s} | {fmt(rp)} | {fmt(rc)} | {dd_str}")

# =========================================================================
# Step 5: ChromHMM as binary classifier — Transcribed vs not
# =========================================================================
print("\n" + "=" * 80)
print("CLASSIFIER: ChromHMM Transcribed vs Non-Transcribed")
print("=" * 80)

unified['is_transcribed'] = (unified['chromhmm_group'] == 'Transcribed').astype(int)

for label, data in [
    ('Transcribed (all)', unified[unified['is_transcribed'] == 1]),
    ('Non-Transcribed (all)', unified[unified['is_transcribed'] == 0]),
    ('Transcribed PASS', unified[(unified['is_transcribed'] == 1) & (unified['source'] == 'PASS')]),
    ('Transcribed CatB', unified[(unified['is_transcribed'] == 1) & (unified['source'] == 'CatB')]),
    ('Non-Transcribed PASS', unified[(unified['is_transcribed'] == 0) & (unified['source'] == 'PASS')]),
    ('Non-Transcribed CatB', unified[(unified['is_transcribed'] == 0) & (unified['source'] == 'CatB')]),
]:
    r = ars_test(data)
    p_str = f"{r['p']:.2e}" if not np.isnan(r['p']) else 'n/a'
    d_str = f"{r['d']:+.1f}" if not np.isnan(r['d']) else 'n/a'
    print(f"  {label:30s}: Δ={d_str:>8s}, p={p_str:>10s}, n={r['n']}")

# =========================================================================
# Step 6: OLS with ChromHMM
# =========================================================================
print("\n" + "=" * 80)
print("OLS MODELS with ChromHMM")
print("=" * 80)

df = unified.dropna(subset=['chromhmm_group', 'overlap_frac', 'polya_length']).copy()
df['is_ars'] = (df['cell_line'] == 'HeLa-Ars').astype(int)
df['is_catb'] = (df['source'] == 'CatB').astype(int)

# M1: GENCODE-based (current)
m1 = smf.ols('polya_length ~ is_ars * is_catb', data=df).fit()

# M2: ChromHMM-based (GENCODE-free)
m2 = smf.ols('polya_length ~ is_ars * is_transcribed', data=df).fit()

# M3: ChromHMM + overlap_frac (GENCODE-free)
m3 = smf.ols('polya_length ~ is_ars * is_transcribed + is_ars * overlap_frac', data=df).fit()

# M4: GENCODE + ChromHMM + overlap_frac (everything)
m4 = smf.ols('polya_length ~ is_ars * is_catb + is_ars * is_transcribed + is_ars * overlap_frac',
             data=df).fit()

# M5: GENCODE + ChromHMM + overlap + read_length
m5 = smf.ols('polya_length ~ is_ars * is_catb + is_ars * is_transcribed + '
             'is_ars * overlap_frac + read_length', data=df).fit()

print(f"\n{'Model':50s} {'R²':>8s} {'AIC':>10s}")
print("-" * 72)
for name, m in [
    ('M1: ars × catb (GENCODE-based)', m1),
    ('M2: ars × transcribed (ChromHMM only)', m2),
    ('M3: ars × transcribed + ars × ov_frac', m3),
    ('M4: ars × catb + ars × tx + ars × ov', m4),
    ('M5: M4 + read_length', m5),
]:
    print(f"  {name:50s} {m.rsquared:8.4f} {m.aic:10.0f}")

# Print key coefficients
for name, m in [
    ('M2 (ChromHMM only, GENCODE-free)', m2),
    ('M3 (ChromHMM + overlap, GENCODE-free)', m3),
    ('M4 (GENCODE + ChromHMM + overlap)', m4),
]:
    print(f"\n  {name}:")
    for var in m.params.index:
        if var == 'Intercept':
            continue
        sig = '***' if m.pvalues[var] < 0.001 else '**' if m.pvalues[var] < 0.01 else '*' if m.pvalues[var] < 0.05 else 'ns'
        print(f"    {var:40s}: {m.params[var]:+8.2f} (p={m.pvalues[var]:.2e}) {sig}")

# =========================================================================
# Step 7: Does ChromHMM mediate the Cat B effect?
# =========================================================================
print("\n" + "=" * 80)
print("MEDIATION: Does ChromHMM explain why Cat B is immune?")
print("=" * 80)

# If Cat B is immune because it's in Transcribed regions,
# then controlling for ChromHMM should eliminate the Cat B effect
print("\n  is_ars:is_catb coefficient comparison:")
print(f"    Without ChromHMM (M1):       {m1.params.get('is_ars:is_catb', 0):+.2f} "
      f"(p={m1.pvalues.get('is_ars:is_catb', 1):.2e})")
if 'is_ars:is_catb' in m4.params:
    print(f"    With ChromHMM (M4):          {m4.params['is_ars:is_catb']:+.2f} "
          f"(p={m4.pvalues['is_ars:is_catb']:.2e})")
    reduction = (1 - abs(m4.params['is_ars:is_catb']) / abs(m1.params['is_ars:is_catb'])) * 100
    print(f"    Reduction in Cat B effect:   {reduction:.1f}%")
    if m4.pvalues['is_ars:is_catb'] > 0.05:
        print("    → Cat B effect FULLY MEDIATED by ChromHMM!")
    else:
        print("    → Cat B effect only PARTIALLY mediated by ChromHMM")

# =========================================================================
# Step 8: 3-group classifier — Transcribed vs Regulatory vs Other
# =========================================================================
print("\n" + "=" * 80)
print("3-GROUP CLASSIFIER: Transcribed vs Regulatory vs Other")
print("=" * 80)

unified['chromhmm_3group'] = 'Other'
unified.loc[unified['chromhmm_group'] == 'Transcribed', 'chromhmm_3group'] = 'Transcribed'
unified.loc[unified['chromhmm_group'].isin(['Enhancer', 'Promoter']), 'chromhmm_3group'] = 'Regulatory'
unified.loc[unified['chromhmm_group'] == 'Heterochromatin', 'chromhmm_3group'] = 'Heterochromatin'

print(f"\n{'Group':20s} | {'All Δ':>8s} {'p':>10s} {'n':>6s} | "
      f"{'PASS Δ':>8s} {'CatB Δ':>8s}")
print("-" * 75)

for grp in ['Regulatory', 'Transcribed', 'Other', 'Heterochromatin']:
    all_sub = unified[unified['chromhmm_3group'] == grp]
    pass_sub = all_sub[all_sub['source'] == 'PASS']
    catb_sub = all_sub[all_sub['source'] == 'CatB']

    ra = ars_test(all_sub)
    rp = ars_test(pass_sub)
    rc = ars_test(catb_sub)

    p_str = f"{ra['p']:.2e}" if not np.isnan(ra['p']) else 'n/a'
    d_all = f"{ra['d']:+.1f}" if not np.isnan(ra['d']) else 'n/a'
    d_pass = f"{rp['d']:+.1f}" if not np.isnan(rp['d']) else 'n/a'
    d_catb = f"{rc['d']:+.1f}" if not np.isnan(rc['d']) else 'n/a'
    print(f"{grp:20s} | {d_all:>8s} {p_str:>10s} {ra['n']:6d} | {d_pass:>8s} {d_catb:>8s}")

# =========================================================================
# Step 9: Summary comparison table
# =========================================================================
print("\n" + "=" * 80)
print("SUMMARY: Classifier comparison")
print("=" * 80)

classifiers = [
    ('Current: PASS vs CatB (GENCODE)',
     unified['source'] == 'PASS', unified['source'] == 'CatB'),
    ('overlap_frac > 0.7 (struct)',
     unified['overlap_frac'] > 0.7, unified['overlap_frac'] <= 0.7),
    ('ChromHMM non-Transcribed (epigen)',
     unified['chromhmm_group'] != 'Transcribed',
     unified['chromhmm_group'] == 'Transcribed'),
    ('ChromHMM non-Tx + ov>0.3 (combined)',
     (unified['chromhmm_group'] != 'Transcribed') & (unified['overlap_frac'] > 0.3),
     ~((unified['chromhmm_group'] != 'Transcribed') & (unified['overlap_frac'] > 0.3))),
]

print(f"\n{'Classifier':45s} | {'Grp1 Δ':>8s} {'n':>6s} | {'Grp2 Δ':>8s} {'n':>6s} | {'|ΔΔ|':>6s}")
print("-" * 85)

for name, mask1, mask2 in classifiers:
    r1 = ars_test(unified[mask1])
    r2 = ars_test(unified[mask2])
    dd = abs(r1['d'] - r2['d']) if not (np.isnan(r1['d']) or np.isnan(r2['d'])) else np.nan
    d1 = f"{r1['d']:+.1f}" if not np.isnan(r1['d']) else 'n/a'
    d2 = f"{r2['d']:+.1f}" if not np.isnan(r2['d']) else 'n/a'
    dd_str = f"{dd:.1f}" if not np.isnan(dd) else 'n/a'
    print(f"  {name:45s} | {d1:>8s} {r1['n']:6d} | {d2:>8s} {r2['n']:6d} | {dd_str:>6s}")

# Save annotated dataset
unified.to_csv(OUTDIR / 'unified_hela_ars_chromhmm.tsv', sep='\t', index=False)
print(f"\nSaved to: {OUTDIR}/unified_hela_ars_chromhmm.tsv")
print("Done!")
