#!/usr/bin/env python3
"""Read length / 5' truncation analysis: HeLa vs HeLa-Ars.

If arsenite causes 5' degradation in addition to 3' deadenylation,
we expect shorter reads in Ars. Compare PASS L1, Cat B, and Control.
Also check dist_to_3prime (DRS 3' bias) for truncation evidence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
OUTDIR = TOPIC_07 / 'read_length_truncation'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

GROUPS = {
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
}

# =============================================================================
# 1. Load PASS L1 data
# =============================================================================
print("Loading PASS L1 data...")
pass_list = []
for cl, grps in GROUPS.items():
    for grp in grps:
        f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['group'] = grp
            df['cell_line'] = cl
            df['locus_id'] = df['transcript_id']
            df['subfamily'] = df['gene_id']
            df['is_young'] = df['subfamily'].isin(YOUNG)
            df['age'] = np.where(df['is_young'], 'young', 'ancient')
            df = df.rename(columns={'polya_length': 'polya'})
            pass_list.append(df)

pass_df = pd.concat(pass_list, ignore_index=True)
print(f"  PASS L1: {len(pass_df)} reads")

# =============================================================================
# 2. Load Cat B data
# =============================================================================
print("Loading Cat B data...")
catB_list = []
for cl, grps in GROUPS.items():
    for grp in grps:
        f = TOPIC_07 / f'catB_reads_{grp}.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['cell_line'] = cl
            catB_list.append(df)

catB_df = pd.concat(catB_list, ignore_index=True)
catB_df['is_young'] = catB_df['subfamily'].isin(YOUNG)
print(f"  Cat B: {len(catB_df)} reads")

# =============================================================================
# 3. Load Control data
# =============================================================================
print("Loading Control data...")
ctrl_list = []
for cl, grps in GROUPS.items():
    for grp in grps:
        f = RESULTS / grp / 'i_control' / f'{grp}_control_summary.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['group'] = grp
            df['cell_line'] = cl
            ctrl_list.append(df)

ctrl_df = pd.concat(ctrl_list, ignore_index=True)
print(f"  Control: {len(ctrl_df)} reads")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 1: READ LENGTH — HeLa vs HeLa-Ars")
print("=" * 70)

print("\n--- PASS L1 ---")
for age_label in ['all', 'young', 'ancient']:
    if age_label == 'all':
        h = pass_df[pass_df['cell_line'] == 'HeLa']['read_length']
        a = pass_df[pass_df['cell_line'] == 'HeLa-Ars']['read_length']
    else:
        h = pass_df[(pass_df['cell_line'] == 'HeLa') & (pass_df['age'] == age_label)]['read_length']
        a = pass_df[(pass_df['cell_line'] == 'HeLa-Ars') & (pass_df['age'] == age_label)]['read_length']
    if len(h) > 5 and len(a) > 5:
        _, p = stats.mannwhitneyu(h, a, alternative='two-sided')
        print(f"  {age_label:8s}: HeLa={h.median():.0f}(n={len(h)}), "
              f"Ars={a.median():.0f}(n={len(a)}), "
              f"Δ={a.median()-h.median():+.0f}, p={p:.2e}")

print("\n--- Cat B ---")
h = catB_df[catB_df['cell_line'] == 'HeLa']['read_span']
a = catB_df[catB_df['cell_line'] == 'HeLa-Ars']['read_span']
if len(h) > 5 and len(a) > 5:
    _, p = stats.mannwhitneyu(h, a, alternative='two-sided')
    print(f"  all     : HeLa={h.median():.0f}(n={len(h)}), "
          f"Ars={a.median():.0f}(n={len(a)}), "
          f"Δ={a.median()-h.median():+.0f}, p={p:.2e}")

print("\n--- Control ---")
if 'read_length' in ctrl_df.columns:
    h = ctrl_df[ctrl_df['cell_line'] == 'HeLa']['read_length']
    a = ctrl_df[ctrl_df['cell_line'] == 'HeLa-Ars']['read_length']
    if len(h) > 5 and len(a) > 5:
        _, p = stats.mannwhitneyu(h, a, alternative='two-sided')
        print(f"  all     : HeLa={h.median():.0f}(n={len(h)}), "
              f"Ars={a.median():.0f}(n={len(a)}), "
              f"Δ={a.median()-h.median():+.0f}, p={p:.2e}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: 5' TRUNCATION (dist_to_3prime)")
print("=" * 70)

# dist_to_3prime: distance from read 5' end to L1 3' end
# If reads are more truncated in Ars, dist_to_3prime should be shorter (reads start closer to 3')
print("\n--- PASS L1 dist_to_3prime ---")
if 'dist_to_3prime' in pass_df.columns:
    for age_label in ['all', 'young', 'ancient']:
        if age_label == 'all':
            h = pass_df[pass_df['cell_line'] == 'HeLa']['dist_to_3prime'].dropna()
            a = pass_df[pass_df['cell_line'] == 'HeLa-Ars']['dist_to_3prime'].dropna()
        else:
            h = pass_df[(pass_df['cell_line'] == 'HeLa') & (pass_df['age'] == age_label)]['dist_to_3prime'].dropna()
            a = pass_df[(pass_df['cell_line'] == 'HeLa-Ars') & (pass_df['age'] == age_label)]['dist_to_3prime'].dropna()
        if len(h) > 5 and len(a) > 5:
            _, p = stats.mannwhitneyu(h, a, alternative='two-sided')
            print(f"  {age_label:8s}: HeLa={h.median():.0f}(n={len(h)}), "
                  f"Ars={a.median():.0f}(n={len(a)}), "
                  f"Δ={a.median()-h.median():+.0f}, p={p:.2e}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: READ LENGTH DISTRIBUTION PERCENTILES")
print("=" * 70)

print("\n--- PASS L1 Ancient Read Length Percentiles ---")
h = pass_df[(pass_df['cell_line'] == 'HeLa') & (pass_df['age'] == 'ancient')]['read_length']
a = pass_df[(pass_df['cell_line'] == 'HeLa-Ars') & (pass_df['age'] == 'ancient')]['read_length']

rows = []
for pct in [10, 25, 50, 75, 90]:
    rows.append({
        'percentile': f'P{pct}',
        'HeLa': h.quantile(pct/100),
        'HeLa-Ars': a.quantile(pct/100),
        'delta': a.quantile(pct/100) - h.quantile(pct/100),
    })
pct_df = pd.DataFrame(rows)
print(pct_df.to_string(index=False))

# KS test
ks_stat, ks_p = stats.ks_2samp(h, a)
print(f"\n  KS test: D={ks_stat:.4f}, p={ks_p:.2e}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: SHORT READS ENRICHMENT UNDER STRESS")
print("=" * 70)

# Are very short reads (<500bp) enriched in Ars? This would indicate degradation
for threshold in [200, 300, 500, 1000]:
    h_short = (pass_df[(pass_df['cell_line'] == 'HeLa') & (pass_df['age'] == 'ancient')]['read_length'] < threshold).mean()
    a_short = (pass_df[(pass_df['cell_line'] == 'HeLa-Ars') & (pass_df['age'] == 'ancient')]['read_length'] < threshold).mean()
    print(f"  <{threshold}bp: HeLa={h_short*100:.1f}%, Ars={a_short*100:.1f}%, ratio={a_short/h_short:.2f}x" if h_short > 0 else f"  <{threshold}bp: HeLa=0%, Ars={a_short*100:.1f}%")

# Same for Cat B
print("\n--- Cat B Short Reads ---")
for threshold in [200, 300, 500, 1000]:
    h_short = (catB_df[catB_df['cell_line'] == 'HeLa']['read_span'] < threshold).mean()
    a_short = (catB_df[catB_df['cell_line'] == 'HeLa-Ars']['read_span'] < threshold).mean()
    if h_short > 0:
        print(f"  <{threshold}bp: HeLa={h_short*100:.1f}%, Ars={a_short*100:.1f}%, ratio={a_short/h_short:.2f}x")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: POLY(A) vs READ LENGTH CORRELATION")
print("=" * 70)

# Is the poly(A) shortening correlated with read length shortening?
for label, cl in [('HeLa', 'HeLa'), ('HeLa-Ars', 'HeLa-Ars')]:
    sub = pass_df[(pass_df['cell_line'] == cl) & (pass_df['age'] == 'ancient') & (pass_df['polya'].notna())]
    r, p = stats.spearmanr(sub['read_length'], sub['polya'])
    print(f"  {label}: read_length ~ poly(A): r={r:.3f}, p={p:.2e}, n={len(sub)}")

# Within matched read length bins
print("\n--- Poly(A) within read length bins (Ancient) ---")
h = pass_df[(pass_df['cell_line'] == 'HeLa') & (pass_df['age'] == 'ancient') & (pass_df['polya'].notna())]
a = pass_df[(pass_df['cell_line'] == 'HeLa-Ars') & (pass_df['age'] == 'ancient') & (pass_df['polya'].notna())]

bins = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000)]
for lo, hi in bins:
    hb = h[(h['read_length'] >= lo) & (h['read_length'] < hi)]['polya']
    ab = a[(a['read_length'] >= lo) & (a['read_length'] < hi)]['polya']
    if len(hb) > 5 and len(ab) > 5:
        _, p = stats.mannwhitneyu(hb, ab, alternative='two-sided')
        print(f"  {lo}-{hi}bp: HeLa={hb.median():.1f}(n={len(hb)}), "
              f"Ars={ab.median():.1f}(n={len(ab)}), Δ={ab.median()-hb.median():+.1f}, p={p:.2e}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: SUMMARY TABLE")
print("=" * 70)

summary_rows = []
for label, df_h, df_a, rl_col in [
    ('PASS L1 all', pass_df[pass_df['cell_line']=='HeLa'], pass_df[pass_df['cell_line']=='HeLa-Ars'], 'read_length'),
    ('PASS ancient', pass_df[(pass_df['cell_line']=='HeLa')&(pass_df['age']=='ancient')], pass_df[(pass_df['cell_line']=='HeLa-Ars')&(pass_df['age']=='ancient')], 'read_length'),
    ('PASS young', pass_df[(pass_df['cell_line']=='HeLa')&(pass_df['age']=='young')], pass_df[(pass_df['cell_line']=='HeLa-Ars')&(pass_df['age']=='young')], 'read_length'),
    ('Cat B', catB_df[catB_df['cell_line']=='HeLa'], catB_df[catB_df['cell_line']=='HeLa-Ars'], 'read_span'),
]:
    if len(df_h) > 5 and len(df_a) > 5:
        summary_rows.append({
            'category': label,
            'hela_median_rl': df_h[rl_col].median(),
            'ars_median_rl': df_a[rl_col].median(),
            'rl_delta': df_a[rl_col].median() - df_h[rl_col].median(),
        })

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
summary.to_csv(OUTDIR / 'read_length_summary.tsv', sep='\t', index=False)

print(f"\nAll results saved to: {OUTDIR}")
print("Done!")
