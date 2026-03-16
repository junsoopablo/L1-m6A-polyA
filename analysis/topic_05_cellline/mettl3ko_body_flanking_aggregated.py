#!/usr/bin/env python3
"""
Aggregated body vs flanking m6A analysis for METTL3 KO.

Per-read medians are 0 due to sparsity at thr=204.
This script pools reads to compute aggregated m6A densities,
replicate-level statistics, bootstrap CIs, and binary fractions.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
INPUT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/mettl3ko_body_flanking/body_flanking_per_read.tsv"
OUTPUT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/mettl3ko_body_flanking/aggregated_summary.tsv"

# ── Load data ──
df = pd.read_csv(INPUT, sep='\t')
print(f"Loaded {len(df)} reads")
print(f"Conditions: {df['condition'].value_counts().to_dict()}")
print(f"Samples: {df['sample'].value_counts().to_dict()}")

# Handle edge cases: negative flank_bases (from overlap > 1.0)
df = df[df['body_bases'] > 0].copy()
df['flank_bases_safe'] = df['flank_bases'].clip(lower=0)

print(f"After filtering body_bases>0: {len(df)} reads")
print()

# ══════════════════════════════════════
print("=" * 50)
print("AGGREGATED ANALYSIS: Body vs Flanking m6A")
print("=" * 50)
print()

# ── [1] Pooled m6A density (all reads) ──
print("[1] Pooled m6A density (all reads)")
print("-" * 50)

results_rows = []

for cond in ['WT', 'KO']:
    sub = df[df['condition'] == cond]
    n_reads = len(sub)

    total_body_m6a = sub['body_m6a_high'].sum()
    total_body_kb = sub['body_bases'].sum() / 1000.0
    total_flank_m6a = sub['flank_m6a_high'].sum()
    total_flank_kb = sub['flank_bases_safe'].sum() / 1000.0
    total_m6a = sub['total_m6a_high'].sum()
    total_kb = sub['read_length'].sum() / 1000.0

    body_rate = total_body_m6a / total_body_kb if total_body_kb > 0 else 0
    flank_rate = total_flank_m6a / total_flank_kb if total_flank_kb > 0 else 0
    total_rate = total_m6a / total_kb if total_kb > 0 else 0

    print(f"  {cond} (n={n_reads} reads):")
    print(f"    Body:  {total_body_m6a} sites / {total_body_kb:.1f} kb = {body_rate:.3f} m6A/kb")
    print(f"    Flank: {total_flank_m6a} sites / {total_flank_kb:.1f} kb = {flank_rate:.3f} m6A/kb")
    print(f"    Total: {total_m6a} sites / {total_kb:.1f} kb = {total_rate:.3f} m6A/kb")

    results_rows.append({
        'analysis': 'pooled_all', 'condition': cond, 'compartment': 'body',
        'n_reads': n_reads, 'total_sites': int(total_body_m6a),
        'total_kb': round(total_body_kb, 2), 'm6a_per_kb': round(body_rate, 4)
    })
    results_rows.append({
        'analysis': 'pooled_all', 'condition': cond, 'compartment': 'flank',
        'n_reads': n_reads, 'total_sites': int(total_flank_m6a),
        'total_kb': round(total_flank_kb, 2), 'm6a_per_kb': round(flank_rate, 4)
    })
    results_rows.append({
        'analysis': 'pooled_all', 'condition': cond, 'compartment': 'total',
        'n_reads': n_reads, 'total_sites': int(total_m6a),
        'total_kb': round(total_kb, 2), 'm6a_per_kb': round(total_rate, 4)
    })

# Compute fold changes
wt_body = [r for r in results_rows if r['analysis'] == 'pooled_all' and r['condition'] == 'WT' and r['compartment'] == 'body'][0]['m6a_per_kb']
ko_body = [r for r in results_rows if r['analysis'] == 'pooled_all' and r['condition'] == 'KO' and r['compartment'] == 'body'][0]['m6a_per_kb']
wt_flank = [r for r in results_rows if r['analysis'] == 'pooled_all' and r['condition'] == 'WT' and r['compartment'] == 'flank'][0]['m6a_per_kb']
ko_flank = [r for r in results_rows if r['analysis'] == 'pooled_all' and r['condition'] == 'KO' and r['compartment'] == 'flank'][0]['m6a_per_kb']

print(f"\n  KO/WT fold change:")
print(f"    Body:  {ko_body/wt_body:.3f}x" if wt_body > 0 else "    Body:  N/A")
print(f"    Flank: {ko_flank/wt_flank:.3f}x" if wt_flank > 0 else "    Flank: N/A")
print()

# ── [2] Per-replicate pooled rates ──
print("[2] Per-replicate pooled rates")
print("-" * 50)

rep_data = {'body': {'WT': [], 'KO': []}, 'flank': {'WT': [], 'KO': []}, 'total': {'WT': [], 'KO': []}}

for sample in sorted(df['sample'].unique()):
    sub = df[df['sample'] == sample]
    cond = sub['condition'].iloc[0]
    n = len(sub)

    body_rate = sub['body_m6a_high'].sum() / (sub['body_bases'].sum() / 1000.0) if sub['body_bases'].sum() > 0 else 0
    flank_rate = sub['flank_m6a_high'].sum() / (sub['flank_bases_safe'].sum() / 1000.0) if sub['flank_bases_safe'].sum() > 0 else 0
    total_rate = sub['total_m6a_high'].sum() / (sub['read_length'].sum() / 1000.0) if sub['read_length'].sum() > 0 else 0

    rep_data['body'][cond].append(body_rate)
    rep_data['flank'][cond].append(flank_rate)
    rep_data['total'][cond].append(total_rate)

    print(f"  {sample} (n={n}): body={body_rate:.3f}, flank={flank_rate:.3f}, total={total_rate:.3f} m6A/kb")

    results_rows.append({
        'analysis': 'per_replicate', 'condition': cond, 'compartment': 'body',
        'sample': sample, 'n_reads': n, 'm6a_per_kb': round(body_rate, 4)
    })
    results_rows.append({
        'analysis': 'per_replicate', 'condition': cond, 'compartment': 'flank',
        'sample': sample, 'n_reads': n, 'm6a_per_kb': round(flank_rate, 4)
    })

print()
for comp in ['body', 'flank', 'total']:
    wt_vals = rep_data[comp]['WT']
    ko_vals = rep_data[comp]['KO']

    if len(wt_vals) >= 2 and len(ko_vals) >= 2:
        t_stat, t_p = stats.ttest_ind(wt_vals, ko_vals)
        u_stat, u_p = stats.mannwhitneyu(wt_vals, ko_vals, alternative='two-sided')
        fc = np.mean(ko_vals) / np.mean(wt_vals) if np.mean(wt_vals) > 0 else np.nan
        print(f"  {comp.upper()}: WT mean={np.mean(wt_vals):.3f} (SD={np.std(wt_vals, ddof=1):.3f}), "
              f"KO mean={np.mean(ko_vals):.3f} (SD={np.std(ko_vals, ddof=1):.3f})")
        print(f"    FC(KO/WT)={fc:.3f}, t-test P={t_p:.4f}, MWU P={u_p:.4f}")
    else:
        print(f"  {comp.upper()}: Not enough replicates for test")

print()

# ── [3] Bootstrap 95% CI ──
print("[3] Bootstrap 95% CI for pooled m6A/kb (WT-KO difference)")
print("-" * 50)

np.random.seed(42)
N_BOOT = 1000

for comp_name, m6a_col, bases_col in [
    ('body', 'body_m6a_high', 'body_bases'),
    ('flank', 'flank_m6a_high', 'flank_bases_safe'),
    ('total', 'total_m6a_high', 'read_length')
]:
    wt_sub = df[df['condition'] == 'WT']
    ko_sub = df[df['condition'] == 'KO']

    boot_diffs = []
    boot_wt = []
    boot_ko = []

    for _ in range(N_BOOT):
        # Sample with replacement
        wt_samp = wt_sub.sample(n=len(wt_sub), replace=True)
        ko_samp = ko_sub.sample(n=len(ko_sub), replace=True)

        wt_rate = wt_samp[m6a_col].sum() / (wt_samp[bases_col].sum() / 1000.0) if wt_samp[bases_col].sum() > 0 else 0
        ko_rate = ko_samp[m6a_col].sum() / (ko_samp[bases_col].sum() / 1000.0) if ko_samp[bases_col].sum() > 0 else 0

        boot_wt.append(wt_rate)
        boot_ko.append(ko_rate)
        boot_diffs.append(ko_rate - wt_rate)

    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    wt_ci = np.percentile(boot_wt, [2.5, 97.5])
    ko_ci = np.percentile(boot_ko, [2.5, 97.5])
    mean_diff = np.mean(boot_diffs)

    print(f"  {comp_name.upper()}:")
    print(f"    WT pooled m6A/kb: {np.mean(boot_wt):.3f} [95% CI: {wt_ci[0]:.3f} - {wt_ci[1]:.3f}]")
    print(f"    KO pooled m6A/kb: {np.mean(boot_ko):.3f} [95% CI: {ko_ci[0]:.3f} - {ko_ci[1]:.3f}]")
    print(f"    Difference (KO-WT): {mean_diff:.4f} [95% CI: {ci_lo:.4f} - {ci_hi:.4f}]")
    sig = "YES (CI excludes 0)" if (ci_lo > 0 or ci_hi < 0) else "NO (CI includes 0)"
    print(f"    Significant: {sig}")

    results_rows.append({
        'analysis': 'bootstrap_CI', 'compartment': comp_name,
        'wt_mean': round(np.mean(boot_wt), 4), 'ko_mean': round(np.mean(boot_ko), 4),
        'diff_mean': round(mean_diff, 4), 'ci_lo': round(ci_lo, 4), 'ci_hi': round(ci_hi, 4),
        'significant': sig
    })

print()

# ── [4] Binary: fraction with >=1 m6A site ──
print("[4] Binary: fraction of reads with >= 1 m6A site")
print("-" * 50)

for comp_name, col in [('body', 'body_m6a_high'), ('flank', 'flank_m6a_high'), ('total', 'total_m6a_high')]:
    print(f"  {comp_name.upper()}:")

    contingency = []
    for cond in ['WT', 'KO']:
        sub = df[df['condition'] == cond]
        has_site = (sub[col] >= 1).sum()
        no_site = (sub[col] < 1).sum()
        frac = has_site / len(sub) if len(sub) > 0 else 0
        print(f"    {cond}: {has_site}/{len(sub)} = {frac:.4f} ({frac*100:.1f}%)")
        contingency.append([has_site, no_site])

    contingency = np.array(contingency)
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

    wt_frac = contingency[0, 0] / contingency[0].sum()
    ko_frac = contingency[1, 0] / contingency[1].sum()
    or_val = (contingency[1, 0] * contingency[0, 1]) / (contingency[0, 0] * contingency[1, 1]) if (contingency[0, 0] * contingency[1, 1]) > 0 else np.nan

    print(f"    Chi2={chi2:.3f}, P={chi_p:.4e}, OR={or_val:.3f}")

    results_rows.append({
        'analysis': 'binary_fraction', 'compartment': comp_name,
        'wt_frac': round(wt_frac, 4), 'ko_frac': round(ko_frac, 4),
        'chi2': round(chi2, 3), 'p_value': f"{chi_p:.4e}", 'odds_ratio': round(or_val, 3)
    })

print()

# ── [5] By overlap fraction bins ──
print("[5] By overlap fraction bins")
print("-" * 50)

bins = [('low (<0.3)', 0, 0.3), ('medium (0.3-0.7)', 0.3, 0.7), ('high (>0.7)', 0.7, 1.01)]

for bin_name, lo, hi in bins:
    print(f"\n  Overlap bin: {bin_name}")
    bin_df = df[(df['overlap_fraction'] >= lo) & (df['overlap_fraction'] < hi)]

    if len(bin_df) == 0:
        print("    No reads in this bin")
        continue

    for cond in ['WT', 'KO']:
        sub = bin_df[bin_df['condition'] == cond]
        if len(sub) == 0:
            print(f"    {cond}: No reads")
            continue

        body_rate = sub['body_m6a_high'].sum() / (sub['body_bases'].sum() / 1000.0) if sub['body_bases'].sum() > 0 else 0
        flank_rate = sub['flank_m6a_high'].sum() / (sub['flank_bases_safe'].sum() / 1000.0) if sub['flank_bases_safe'].sum() > 0 else 0
        total_rate = sub['total_m6a_high'].sum() / (sub['read_length'].sum() / 1000.0) if sub['read_length'].sum() > 0 else 0

        print(f"    {cond} (n={len(sub)}): body={body_rate:.3f}, flank={flank_rate:.3f}, total={total_rate:.3f} m6A/kb")

        results_rows.append({
            'analysis': f'overlap_bin_{bin_name}', 'condition': cond, 'compartment': 'body',
            'n_reads': len(sub), 'm6a_per_kb': round(body_rate, 4)
        })
        results_rows.append({
            'analysis': f'overlap_bin_{bin_name}', 'condition': cond, 'compartment': 'flank',
            'n_reads': len(sub), 'm6a_per_kb': round(flank_rate, 4)
        })

    # Quick test within each bin
    for comp_name, m6a_col, bases_col in [('body', 'body_m6a_high', 'body_bases'), ('flank', 'flank_m6a_high', 'flank_bases_safe')]:
        wt_sub = bin_df[bin_df['condition'] == 'WT']
        ko_sub = bin_df[bin_df['condition'] == 'KO']

        if len(wt_sub) > 0 and len(ko_sub) > 0:
            wt_rate = wt_sub[m6a_col].sum() / (wt_sub[bases_col].sum() / 1000.0) if wt_sub[bases_col].sum() > 0 else 0
            ko_rate = ko_sub[m6a_col].sum() / (ko_sub[bases_col].sum() / 1000.0) if ko_sub[bases_col].sum() > 0 else 0
            fc = ko_rate / wt_rate if wt_rate > 0 else np.nan
            print(f"    {comp_name} KO/WT FC = {fc:.3f}")

print()

# ── [6] Summary interpretation ──
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print()

# Retrieve key numbers
wt_body_pooled = [r for r in results_rows if r.get('analysis') == 'pooled_all' and r.get('condition') == 'WT' and r.get('compartment') == 'body'][0]
ko_body_pooled = [r for r in results_rows if r.get('analysis') == 'pooled_all' and r.get('condition') == 'KO' and r.get('compartment') == 'body'][0]
wt_flank_pooled = [r for r in results_rows if r.get('analysis') == 'pooled_all' and r.get('condition') == 'WT' and r.get('compartment') == 'flank'][0]
ko_flank_pooled = [r for r in results_rows if r.get('analysis') == 'pooled_all' and r.get('condition') == 'KO' and r.get('compartment') == 'flank'][0]

body_fc = ko_body_pooled['m6a_per_kb'] / wt_body_pooled['m6a_per_kb'] if wt_body_pooled['m6a_per_kb'] > 0 else np.nan
flank_fc = ko_flank_pooled['m6a_per_kb'] / wt_flank_pooled['m6a_per_kb'] if wt_flank_pooled['m6a_per_kb'] > 0 else np.nan

print(f"Body m6A/kb:  WT={wt_body_pooled['m6a_per_kb']:.3f}, KO={ko_body_pooled['m6a_per_kb']:.3f}, FC={body_fc:.3f}")
print(f"Flank m6A/kb: WT={wt_flank_pooled['m6a_per_kb']:.3f}, KO={ko_flank_pooled['m6a_per_kb']:.3f}, FC={flank_fc:.3f}")
print()
if body_fc < flank_fc:
    print("=> Body compartment shows STRONGER KO effect than flanking")
elif body_fc > flank_fc:
    print("=> Flanking compartment shows STRONGER KO effect than body")
else:
    print("=> Body and flanking show EQUAL KO effect")

print()

# ── Save results ──
results_df = pd.DataFrame(results_rows)
results_df.to_csv(OUTPUT, sep='\t', index=False)
print(f"Results saved to: {OUTPUT}")
print("Done.")
