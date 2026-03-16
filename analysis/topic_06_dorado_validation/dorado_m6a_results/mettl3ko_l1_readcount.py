#!/usr/bin/env python3
"""
METTL3 KO Survivorship Bias Analysis: L1 Read Count Changes
============================================================
Hypothesis: If m6A-depleted L1 transcripts are rapidly degraded in METTL3 KO,
            then L1 read count should decrease (especially after library-size normalization).

Data: PRJEB40872 (HEK293T, RNA002 DRS) — 3 WT + 3 KO replicates
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE = "/vault/external-datasets/2026/PRJEB40872_HEK293T_METTL3KO_xPore/l1_filter_guppy"
SAMPLES = {
    'WT': ['WT_rep1', 'WT_rep2', 'WT_rep3'],
    'KO': ['KO_rep1', 'KO_rep2', 'KO_rep3']
}
YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Helper functions
# ============================================================
def load_reads(sample):
    """Load per-read L1 TSV."""
    path = os.path.join(BASE, f"{sample}_L1_reads.tsv")
    return pd.read_csv(path, sep='\t')

def load_summary(sample):
    """Load per-locus L1 summary TSV."""
    path = os.path.join(BASE, f"{sample}_L1_summary.tsv")
    return pd.read_csv(path, sep='\t')

def get_total_reads(sample):
    """Get total mapped reads from QC summary."""
    path = os.path.join(BASE, f"{sample}_qc_summary.txt")
    with open(path) as f:
        line = f.readline().strip()
        return int(line.split('\t')[1])

def ci_95(vals):
    """95% CI for mean."""
    n = len(vals)
    m = np.mean(vals)
    se = stats.sem(vals)
    h = se * stats.t.ppf(0.975, n - 1)
    return m, m - h, m + h

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# 1. Library-size normalized L1 read count
# ============================================================
print_section("1. Library-Size Normalized L1 Read Count (RPM)")

data_rows = []
for condition, samples in SAMPLES.items():
    for s in samples:
        total = get_total_reads(s)
        reads_df = load_reads(s)
        l1_count = len(reads_df)
        rpm = l1_count / total * 1e6
        frac = l1_count / total
        data_rows.append({
            'sample': s, 'condition': condition,
            'l1_count': l1_count, 'total_reads': total,
            'l1_rpm': rpm, 'l1_fraction': frac
        })
        print(f"  {s}: L1={l1_count:,}, total={total:,}, RPM={rpm:.2f}, fraction={frac:.6f}")

df_samples = pd.DataFrame(data_rows)

wt_rpm = df_samples[df_samples['condition'] == 'WT']['l1_rpm'].values
ko_rpm = df_samples[df_samples['condition'] == 'KO']['l1_rpm'].values

wt_mean, wt_lo, wt_hi = ci_95(wt_rpm)
ko_mean, ko_lo, ko_hi = ci_95(ko_rpm)

fc = ko_mean / wt_mean

# Parametric
t_stat, t_pval = stats.ttest_ind(wt_rpm, ko_rpm)
# Nonparametric (given n=3 per group, this has limited power)
u_stat, u_pval = stats.mannwhitneyu(wt_rpm, ko_rpm, alternative='two-sided')
# Welch's t-test (unequal variance)
tw_stat, tw_pval = stats.ttest_ind(wt_rpm, ko_rpm, equal_var=False)

print(f"\n  WT RPM:  mean={wt_mean:.2f}  95%CI=[{wt_lo:.2f}, {wt_hi:.2f}]")
print(f"  KO RPM:  mean={ko_mean:.2f}  95%CI=[{ko_lo:.2f}, {ko_hi:.2f}]")
print(f"  FC (KO/WT) = {fc:.3f}")
print(f"  Student's t-test:  t={t_stat:.3f}, P={t_pval:.4f}")
print(f"  Welch's t-test:    t={tw_stat:.3f}, P={tw_pval:.4f}")
print(f"  Mann-Whitney U:    U={u_stat:.1f}, P={u_pval:.4f}")

# Also test raw counts with Poisson-style approach
# Pool WT vs KO, adjust for library size
wt_l1_total = df_samples[df_samples['condition'] == 'WT']['l1_count'].sum()
ko_l1_total = df_samples[df_samples['condition'] == 'KO']['l1_count'].sum()
wt_lib_total = df_samples[df_samples['condition'] == 'WT']['total_reads'].sum()
ko_lib_total = df_samples[df_samples['condition'] == 'KO']['total_reads'].sum()

# Fisher's exact: L1 vs non-L1 in WT vs KO (pooled)
table = np.array([
    [wt_l1_total, wt_lib_total - wt_l1_total],
    [ko_l1_total, ko_lib_total - ko_l1_total]
])
odds_ratio, fisher_p = stats.fisher_exact(table)
print(f"\n  Pooled Fisher's exact (L1 vs non-L1 × WT vs KO):")
print(f"    WT: {wt_l1_total} L1 / {wt_lib_total:,} total")
print(f"    KO: {ko_l1_total} L1 / {ko_lib_total:,} total")
print(f"    OR={odds_ratio:.3f}, P={fisher_p:.4f}")

# ============================================================
# 2. Age class analysis (Young vs Ancient)
# ============================================================
print_section("2. Age Class Analysis: Young vs Ancient")

age_data = []
for condition, samples in SAMPLES.items():
    for s in samples:
        reads_df = load_reads(s)
        total = get_total_reads(s)

        reads_df['age'] = reads_df['gene_id'].apply(
            lambda x: 'Young' if x in YOUNG_SUBFAMILIES else 'Ancient'
        )

        for age in ['Young', 'Ancient']:
            n = (reads_df['age'] == age).sum()
            rpm = n / total * 1e6
            age_data.append({
                'sample': s, 'condition': condition,
                'age': age, 'count': n, 'rpm': rpm
            })

df_age = pd.DataFrame(age_data)

for age in ['Young', 'Ancient']:
    print(f"\n  --- {age} L1 ---")
    sub = df_age[df_age['age'] == age]
    for _, row in sub.iterrows():
        print(f"    {row['sample']}: n={row['count']}, RPM={row['rpm']:.2f}")

    wt_vals = sub[sub['condition'] == 'WT']['rpm'].values
    ko_vals = sub[sub['condition'] == 'KO']['rpm'].values

    wt_m, wt_l, wt_h = ci_95(wt_vals)
    ko_m, ko_l, ko_h = ci_95(ko_vals)
    fc_age = ko_m / wt_m if wt_m > 0 else np.nan

    t_s, t_p = stats.ttest_ind(wt_vals, ko_vals)
    tw_s, tw_p = stats.ttest_ind(wt_vals, ko_vals, equal_var=False)
    u_s, u_p = stats.mannwhitneyu(wt_vals, ko_vals, alternative='two-sided')

    print(f"    WT RPM: {wt_m:.2f} [{wt_l:.2f}, {wt_h:.2f}]")
    print(f"    KO RPM: {ko_m:.2f} [{ko_l:.2f}, {ko_h:.2f}]")
    print(f"    FC(KO/WT) = {fc_age:.3f}")
    print(f"    t-test P={t_p:.4f}, Welch P={tw_p:.4f}, MWU P={u_p:.4f}")

    # Young L1 raw counts
    wt_n = sub[sub['condition'] == 'WT']['count'].values
    ko_n = sub[sub['condition'] == 'KO']['count'].values
    print(f"    Raw counts — WT: {wt_n} (mean={np.mean(wt_n):.1f}), KO: {ko_n} (mean={np.mean(ko_n):.1f})")

# ============================================================
# 3. Per-locus analysis
# ============================================================
print_section("3. Per-Locus Analysis")

# Collect per-locus counts across all samples
locus_counts = defaultdict(lambda: defaultdict(int))
locus_subfam = {}

for condition, samples in SAMPLES.items():
    for s in samples:
        summ = load_summary(s)
        for _, row in summ.iterrows():
            locus = row['locus']
            locus_counts[locus][(condition, s)] = row['read_count']
            locus_subfam[locus] = row['gene_id']

# Build per-locus count matrix
all_loci = sorted(locus_counts.keys())
all_samples_list = [s for samples in SAMPLES.values() for s in samples]

count_matrix = pd.DataFrame(0, index=all_loci, columns=all_samples_list)
for locus in all_loci:
    for cond, samples in SAMPLES.items():
        for s in samples:
            count_matrix.loc[locus, s] = locus_counts[locus].get((cond, s), 0)

count_matrix['subfamily'] = [locus_subfam.get(l, 'Unknown') for l in count_matrix.index]

# WT-only and KO-only loci
wt_cols = SAMPLES['WT']
ko_cols = SAMPLES['KO']

wt_any = (count_matrix[wt_cols].sum(axis=1) > 0)
ko_any = (count_matrix[ko_cols].sum(axis=1) > 0)

shared = (wt_any & ko_any).sum()
wt_only = (wt_any & ~ko_any).sum()
ko_only = (~wt_any & ko_any).sum()

print(f"  Total unique loci: {len(all_loci)}")
print(f"  Shared (WT & KO): {shared}")
print(f"  WT-only: {wt_only}")
print(f"  KO-only: {ko_only}")
print(f"  WT-only / KO-only ratio: {wt_only/ko_only:.2f}" if ko_only > 0 else "  KO-only = 0")

# If survivorship bias: WT-only >> KO-only
# Fisher's: WT-only vs KO-only vs shared
# Simple chi-square: are WT-only and KO-only symmetric?
if wt_only > 0 and ko_only > 0:
    chi2_sym = (wt_only - ko_only)**2 / (wt_only + ko_only)
    p_sym = 1 - stats.chi2.cdf(chi2_sym, df=1)
    print(f"  Symmetry chi2 = {chi2_sym:.2f}, P = {p_sym:.4f}")

    # Binomial test: P(WT-only >= obs) given p=0.5
    binom_result = stats.binomtest(wt_only, wt_only + ko_only, 0.5, alternative='two-sided')
    binom_p = binom_result.pvalue
    print(f"  Binomial test (WT-only vs KO-only): P = {binom_p:.4f}")

# Age breakdown of unique loci
print(f"\n  --- Age breakdown of condition-specific loci ---")
wt_only_mask = wt_any & ~ko_any
ko_only_mask = ~wt_any & ko_any

for label, mask in [('WT-only', wt_only_mask), ('KO-only', ko_only_mask)]:
    subfams = count_matrix.loc[mask, 'subfamily']
    young_n = subfams.isin(YOUNG_SUBFAMILIES).sum()
    ancient_n = (~subfams.isin(YOUNG_SUBFAMILIES)).sum()
    print(f"  {label}: Young={young_n}, Ancient={ancient_n}")

# Per-locus read count change for shared loci
print(f"\n  --- Shared loci: read count change ---")
shared_mask = wt_any & ko_any
shared_df = count_matrix.loc[shared_mask].copy()

shared_df['wt_sum'] = shared_df[wt_cols].sum(axis=1)
shared_df['ko_sum'] = shared_df[ko_cols].sum(axis=1)

# Normalize by total library size
wt_lib_sizes = [get_total_reads(s) for s in wt_cols]
ko_lib_sizes = [get_total_reads(s) for s in ko_cols]
wt_size_factor = sum(wt_lib_sizes) / 1e6
ko_size_factor = sum(ko_lib_sizes) / 1e6

shared_df['wt_norm'] = shared_df['wt_sum'] / wt_size_factor
shared_df['ko_norm'] = shared_df['ko_sum'] / ko_size_factor
shared_df['log2fc'] = np.log2((shared_df['ko_norm'] + 0.01) / (shared_df['wt_norm'] + 0.01))

# Summary of log2FC
print(f"  Median log2FC = {shared_df['log2fc'].median():.3f}")
print(f"  Mean log2FC = {shared_df['log2fc'].mean():.3f}")
print(f"  Loci with log2FC > 0 (KO up): {(shared_df['log2fc'] > 0).sum()}")
print(f"  Loci with log2FC < 0 (KO down): {(shared_df['log2fc'] < 0).sum()}")
print(f"  Loci with log2FC = 0: {(shared_df['log2fc'] == 0).sum()}")

# Wilcoxon signed-rank on paired normalized counts
try:
    wil_stat, wil_p = stats.wilcoxon(shared_df['wt_norm'], shared_df['ko_norm'])
    print(f"  Wilcoxon signed-rank: stat={wil_stat:.1f}, P={wil_p:.4f}")
except:
    print(f"  Wilcoxon signed-rank: could not compute (ties?)")

# ============================================================
# 4. Subfamily-level breakdown
# ============================================================
print_section("4. Subfamily-Level Breakdown")

subfam_data = []
for condition, samples_list in SAMPLES.items():
    for s in samples_list:
        reads_df = load_reads(s)
        total = get_total_reads(s)
        for subfam, grp in reads_df.groupby('gene_id'):
            subfam_data.append({
                'sample': s, 'condition': condition,
                'subfamily': subfam, 'count': len(grp),
                'rpm': len(grp) / total * 1e6
            })

df_subfam = pd.DataFrame(subfam_data)

# Aggregate by subfamily and condition
pivot = df_subfam.pivot_table(
    index='subfamily', columns='condition',
    values='rpm', aggfunc='mean'
).fillna(0)

pivot['fc_KO_WT'] = (pivot['KO'] + 0.001) / (pivot['WT'] + 0.001)
pivot['log2fc'] = np.log2(pivot['fc_KO_WT'])

# Total reads per subfamily
count_pivot = df_subfam.pivot_table(
    index='subfamily', columns='condition',
    values='count', aggfunc='sum'
).fillna(0)
pivot['wt_total'] = count_pivot['WT']
pivot['ko_total'] = count_pivot['KO']

pivot['age'] = pivot.index.map(lambda x: 'Young' if x in YOUNG_SUBFAMILIES else 'Ancient')

# Sort by total reads
pivot = pivot.sort_values('wt_total', ascending=False)

print(f"  {'Subfamily':<12} {'WT RPM':>8} {'KO RPM':>8} {'FC':>6} {'log2FC':>7} {'WT_n':>5} {'KO_n':>5} {'Age':<8}")
print(f"  {'-'*65}")
for subfam, row in pivot.head(30).iterrows():
    print(f"  {subfam:<12} {row['WT']:>8.2f} {row['KO']:>8.2f} {row['fc_KO_WT']:>6.2f} {row['log2fc']:>7.3f} {int(row['wt_total']):>5} {int(row['ko_total']):>5} {row['age']:<8}")

# Age gradient correlation
# Assign approximate age to each subfamily (Ma since insertion)
# This is approximate; higher number = more ancient
AGE_ORDER = {
    'L1HS': 1, 'L1PA1': 2, 'L1PA2': 3, 'L1PA3': 4, 'L1PA4': 5,
    'L1PA5': 6, 'L1PA6': 7, 'L1PA7': 8, 'L1PA8': 9, 'L1PA10': 10,
    'L1PA11': 11, 'L1PA12': 12, 'L1PA13': 13, 'L1PA14': 14, 'L1PA15': 15,
    'L1PA16': 16, 'L1PA17': 17,
    'L1PB': 18, 'L1PB1': 19, 'L1PB2': 20, 'L1PB3': 21, 'L1PB4': 22,
    'L1MA1': 25, 'L1MA2': 26, 'L1MA3': 27, 'L1MA4': 28, 'L1MA4A': 29,
    'L1MA5': 30, 'L1MA5A': 31, 'L1MA6': 32, 'L1MA7': 33, 'L1MA8': 34,
    'L1MA9': 35,
    'L1MB1': 40, 'L1MB2': 41, 'L1MB3': 42, 'L1MB4': 43, 'L1MB5': 44,
    'L1MB6': 45, 'L1MB7': 46, 'L1MB8': 47,
    'L1MC': 50, 'L1MC1': 51, 'L1MC2': 52, 'L1MC3': 53, 'L1MC4': 54,
    'L1MC4a': 55, 'L1MC5': 56, 'L1MC5a': 57,
    'L1MD': 60, 'L1MD1': 61, 'L1MD2': 62, 'L1MD3': 63,
    'L1ME1': 70, 'L1ME2': 71, 'L1ME2z': 72, 'L1ME3': 73, 'L1ME3A': 74,
    'L1ME3B': 75, 'L1ME3C': 76, 'L1ME3Cz': 77, 'L1ME3D': 78, 'L1ME3E': 79,
    'L1ME3F': 80, 'L1ME3G': 81, 'L1ME4a': 82, 'L1ME4b': 83, 'L1ME5': 84,
    'L1MEa': 85, 'L1MEb': 86, 'L1MEc': 87, 'L1MEd': 88, 'L1MEf': 89,
    'L1MEg': 90, 'L1MEg1': 91, 'L1MEg2': 92, 'L1MEh': 93, 'L1MEi': 94,
    'L1MEj': 95,
}

pivot['age_rank'] = pivot.index.map(lambda x: AGE_ORDER.get(x, np.nan))
valid = pivot.dropna(subset=['age_rank'])
valid = valid[valid['wt_total'] + valid['ko_total'] >= 3]  # at least 3 total reads

if len(valid) > 5:
    rho, rho_p = stats.spearmanr(valid['age_rank'], valid['log2fc'])
    r, r_p = stats.pearsonr(valid['age_rank'], valid['log2fc'])
    print(f"\n  Age-gradient correlation (subfamilies with >= 3 reads, n={len(valid)}):")
    print(f"    Spearman rho={rho:.3f}, P={rho_p:.4f}")
    print(f"    Pearson  r={r:.3f}, P={r_p:.4f}")
    print(f"    (Positive = older subfamilies more depleted in KO)")
else:
    print(f"\n  Too few subfamilies with age rank and >= 3 reads for correlation")

# ============================================================
# 5. Power analysis
# ============================================================
print_section("5. Statistical Power Considerations")

# Observed effect size (Cohen's d)
pooled_std = np.sqrt(((len(wt_rpm)-1)*np.var(wt_rpm, ddof=1) + (len(ko_rpm)-1)*np.var(ko_rpm, ddof=1)) / (len(wt_rpm) + len(ko_rpm) - 2))
cohens_d = (np.mean(wt_rpm) - np.mean(ko_rpm)) / pooled_std if pooled_std > 0 else 0

print(f"  Observed Cohen's d = {cohens_d:.3f}")
print(f"  With n=3 per group, 80% power requires d >= ~2.5 (two-sided alpha=0.05)")
print(f"  Detectable FC at 80% power: ~{1 - 2.5 * pooled_std / np.mean(wt_rpm):.2f}x to {1 + 2.5 * pooled_std / np.mean(wt_rpm):.2f}x")

# What % decrease could we rule out?
# One-sided 95% CI lower bound
se_diff = np.sqrt(np.var(wt_rpm, ddof=1)/3 + np.var(ko_rpm, ddof=1)/3)
diff = np.mean(ko_rpm) - np.mean(wt_rpm)
ci_lower = diff - stats.t.ppf(0.95, df=4) * se_diff
pct_lower = ci_lower / np.mean(wt_rpm) * 100
print(f"\n  Difference KO-WT = {diff:.2f} RPM")
print(f"  One-sided 95% CI lower bound: {ci_lower:.2f} RPM ({pct_lower:.1f}% of WT mean)")
print(f"  We can rule out KO depletion greater than {-pct_lower:.1f}% at 95% confidence")

# ============================================================
# 6. Bootstrap resampling for FC confidence interval
# ============================================================
print_section("6. Bootstrap FC Confidence Interval (10,000 iterations)")

np.random.seed(42)
n_boot = 10000
boot_fcs = []
for _ in range(n_boot):
    wt_boot = np.random.choice(wt_rpm, size=3, replace=True)
    ko_boot = np.random.choice(ko_rpm, size=3, replace=True)
    boot_fcs.append(np.mean(ko_boot) / np.mean(wt_boot))

boot_fcs = np.array(boot_fcs)
boot_lo, boot_hi = np.percentile(boot_fcs, [2.5, 97.5])
print(f"  Observed FC (KO/WT) = {fc:.3f}")
print(f"  Bootstrap 95% CI: [{boot_lo:.3f}, {boot_hi:.3f}]")
print(f"  Bootstrap P(FC < 0.8) = {(boot_fcs < 0.8).mean():.4f}")
print(f"  Bootstrap P(FC < 0.9) = {(boot_fcs < 0.9).mean():.4f}")
print(f"  Bootstrap P(FC < 1.0) = {(boot_fcs < 1.0).mean():.4f}")

# ============================================================
# 7. Summary Table (TSV output)
# ============================================================
print_section("7. Summary")

# Compile results
summary_rows = []

# Overall
summary_rows.append({
    'comparison': 'Overall L1 RPM',
    'WT_mean': f"{wt_mean:.2f}",
    'WT_95CI': f"[{wt_lo:.2f}, {wt_hi:.2f}]",
    'KO_mean': f"{ko_mean:.2f}",
    'KO_95CI': f"[{ko_lo:.2f}, {ko_hi:.2f}]",
    'FC_KO_WT': f"{fc:.3f}",
    'ttest_P': f"{t_pval:.4f}",
    'welch_P': f"{tw_pval:.4f}",
    'MWU_P': f"{u_pval:.4f}",
    'fisher_P': f"{fisher_p:.4f}",
    'bootstrap_95CI': f"[{boot_lo:.3f}, {boot_hi:.3f}]"
})

# Age classes
for age in ['Young', 'Ancient']:
    sub = df_age[df_age['age'] == age]
    wt_v = sub[sub['condition'] == 'WT']['rpm'].values
    ko_v = sub[sub['condition'] == 'KO']['rpm'].values
    wt_m, wt_l, wt_h = ci_95(wt_v)
    ko_m, ko_l, ko_h = ci_95(ko_v)
    fc_a = ko_m / wt_m if wt_m > 0 else np.nan
    _, tp = stats.ttest_ind(wt_v, ko_v)
    _, twp = stats.ttest_ind(wt_v, ko_v, equal_var=False)
    _, up = stats.mannwhitneyu(wt_v, ko_v, alternative='two-sided')

    summary_rows.append({
        'comparison': f'{age} L1 RPM',
        'WT_mean': f"{wt_m:.2f}",
        'WT_95CI': f"[{wt_l:.2f}, {wt_h:.2f}]",
        'KO_mean': f"{ko_m:.2f}",
        'KO_95CI': f"[{ko_l:.2f}, {ko_h:.2f}]",
        'FC_KO_WT': f"{fc_a:.3f}",
        'ttest_P': f"{tp:.4f}",
        'welch_P': f"{twp:.4f}",
        'MWU_P': f"{up:.4f}",
        'fisher_P': 'NA',
        'bootstrap_95CI': 'NA'
    })

# Locus analysis
summary_rows.append({
    'comparison': 'Shared loci log2FC',
    'WT_mean': f"median={shared_df['log2fc'].median():.3f}",
    'WT_95CI': f"mean={shared_df['log2fc'].mean():.3f}",
    'KO_mean': f"up={int((shared_df['log2fc'] > 0).sum())}",
    'KO_95CI': f"down={int((shared_df['log2fc'] < 0).sum())}",
    'FC_KO_WT': 'NA',
    'ttest_P': 'NA',
    'welch_P': 'NA',
    'MWU_P': 'NA',
    'fisher_P': 'NA',
    'bootstrap_95CI': 'NA'
})

summary_df = pd.DataFrame(summary_rows)
out_path = os.path.join(OUTDIR, "mettl3ko_readcount_summary.tsv")
summary_df.to_csv(out_path, sep='\t', index=False)
print(f"  Summary saved to: {out_path}")

# ============================================================
# 8. Interpretation
# ============================================================
print_section("8. Interpretation")

if fc > 0.9 and t_pval > 0.05:
    print("  RESULT: No significant L1 read count decrease in METTL3 KO.")
    print("  The survivorship bias hypothesis is NOT supported by read count data.")
    print(f"  FC={fc:.3f} with bootstrap 95% CI [{boot_lo:.3f}, {boot_hi:.3f}].")
    if boot_hi < 1.0:
        print("  However, the CI is entirely below 1.0, suggesting a possible mild decrease.")
    elif boot_lo > 1.0:
        print("  Surprisingly, the CI is entirely above 1.0, suggesting a mild increase.")
    else:
        print("  The CI spans 1.0, consistent with no change.")
elif fc < 0.9 and t_pval < 0.05:
    print("  RESULT: Significant L1 read count DECREASE in METTL3 KO.")
    print("  This is CONSISTENT with survivorship bias:")
    print("  m6A-depleted L1 may be degraded, reducing detectable L1 reads.")
else:
    print(f"  RESULT: FC={fc:.3f}, P={t_pval:.4f}. Borderline result.")
    print("  Further investigation with more replicates is recommended.")

# Check age-specific pattern
for age in ['Young', 'Ancient']:
    sub = df_age[df_age['age'] == age]
    wt_v = sub[sub['condition'] == 'WT']['rpm'].values
    ko_v = sub[sub['condition'] == 'KO']['rpm'].values
    fc_a = np.mean(ko_v) / np.mean(wt_v) if np.mean(wt_v) > 0 else np.nan
    _, tp = stats.ttest_ind(wt_v, ko_v)
    if tp < 0.05:
        print(f"\n  {age}: FC={fc_a:.3f}, P={tp:.4f} — SIGNIFICANT")
    else:
        print(f"\n  {age}: FC={fc_a:.3f}, P={tp:.4f} — not significant")

print(f"\n  Locus analysis: {wt_only} WT-only vs {ko_only} KO-only loci")
if wt_only > ko_only * 1.5:
    print("  More WT-only loci suggests some L1 expression lost in KO (consistent with survivorship)")
elif ko_only > wt_only * 1.5:
    print("  More KO-only loci suggests L1 de-repression in KO")
else:
    print("  WT-only / KO-only roughly balanced — no strong locus-level evidence for survivorship")

print("\n" + "="*70)
print("  Analysis complete.")
print("="*70)
