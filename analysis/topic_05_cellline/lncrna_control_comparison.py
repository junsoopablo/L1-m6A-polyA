#!/usr/bin/env python3
"""lncRNA control comparison: exclusive lncRNA vs L1 vs mRNA.

Tests whether L1's long baseline poly(A) (~120nt) and arsenite-induced
shortening are L1-specific or general non-coding transcript properties.
lncRNAs serve as a fairer comparison than intronic/intergenic fragments
because they are autonomous polyadenylated transcripts with their own
promoters and PAS.

Data sources:
  - lncRNA: nanopolish + MAFIA from l_lncrna_ctrl/
  - L1: part3 cache + g_summary/ poly(A)
  - mRNA control: part3 ctrl cache (reference)
  - Noncoding non-L1: noncoding_control_figures/ (reference)

Key comparisons (6):
  1. lncRNA vs L1 vs mRNA baseline poly(A)
  2. lncRNA HeLa vs Ars Δpoly(A)
  3. lncRNA m6A/kb vs L1 m6A/kb
  4. lncRNA m6A-poly(A) correlation
  5. lncRNA subtype breakdown (lincRNA vs antisense vs others)
  6. Read-length matched comparison
"""

import pandas as pd
import numpy as np
import pysam
import ast
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================
PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
RESULTS = PROJECT / 'results_group'
OUTDIR = TOPIC_05 / 'lncrna_control_figures'
OUTDIR.mkdir(exist_ok=True)

PROB_THRESHOLD = 204  # 80% on 0-255 scale

HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3']
ARS_GROUPS = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']
ALL_GROUPS = HELA_GROUPS + ARS_GROUPS

YOUNG_FAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 200,
    'font.family': 'sans-serif',
})

# =============================================================================
# MM/ML parsing (from part3_analysis.py)
# =============================================================================
def parse_mm_ml_tags(mm_tag, ml_tag):
    result = {'m6A': [], 'psi': []}
    if mm_tag is None or ml_tag is None:
        return result
    mod_blocks = mm_tag.rstrip(';').split(';')
    ml_idx = 0
    for block in mod_blocks:
        if not block:
            continue
        parts = block.split(',')
        mod_type = parts[0]
        if '17802' in mod_type or 'U+p' in mod_type or 'T+p' in mod_type:
            mod_key = 'psi'
        elif '21891' in mod_type or 'A+a' in mod_type or 'A+m' in mod_type:
            mod_key = 'm6A'
        else:
            ml_idx += len(parts) - 1
            continue
        current_pos = 0
        for pos_str in parts[1:]:
            if pos_str:
                current_pos += int(pos_str)
                if ml_idx < len(ml_tag):
                    result[mod_key].append((current_pos, ml_tag[ml_idx]))
                ml_idx += 1
    return result


def parse_mafia_bam(bam_path):
    """Parse MAFIA BAM to per-read modification counts."""
    records = []
    bam_path = str(bam_path)
    if not Path(bam_path).exists() or Path(bam_path).stat().st_size == 0:
        return pd.DataFrame(columns=['read_id', 'read_length', 'm6a_sites', 'psi_sites',
                                     'm6a_per_kb', 'psi_per_kb'])
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam:
            mm = read.get_tag("MM") if read.has_tag("MM") else None
            ml = read.get_tag("ML") if read.has_tag("ML") else None
            mods = parse_mm_ml_tags(mm, ml)
            rl = read.query_length or read.infer_query_length() or 0
            if rl < 50:
                continue
            m6a_high = [p for p, prob in mods['m6A'] if prob >= PROB_THRESHOLD]
            psi_high = [p for p, prob in mods['psi'] if prob >= PROB_THRESHOLD]
            rl_kb = rl / 1000.0
            records.append({
                'read_id': read.query_name,
                'read_length': rl,
                'm6a_sites': len(m6a_high),
                'psi_sites': len(psi_high),
                'm6a_per_kb': len(m6a_high) / rl_kb,
                'psi_per_kb': len(psi_high) / rl_kb,
            })
    return pd.DataFrame(records)


# =============================================================================
# Data loading
# =============================================================================
def load_lncrna_data(group):
    """Load lncRNA data: nanopolish poly(A) + MAFIA + classification."""
    lnc_dir = RESULTS / group / 'l_lncrna_ctrl'

    # Classification
    cls_file = lnc_dir / f'{group}_lncrna_classification.tsv'
    if not cls_file.exists():
        print(f"  WARNING: lncRNA classification not found for {group}")
        return None
    cls_df = pd.read_csv(cls_file, sep='\t', dtype={'read_id': str})

    # Nanopolish poly(A)
    polya_file = lnc_dir / f'{group}.lncrna.nanopolish.polya.tsv.gz'
    if not polya_file.exists():
        print(f"  WARNING: lncRNA nanopolish not found for {group}")
        return None
    polya = pd.read_csv(polya_file, sep='\t')
    polya = polya[polya['qc_tag'] == 'PASS'].copy()
    polya = polya.rename(columns={'readname': 'read_id', 'polya_length': 'polya'})
    polya = polya[['read_id', 'polya', 'contig', 'position']].copy()

    # MAFIA
    mafia_bam = lnc_dir / f'mafia/{group}.lncrna.mAFiA.reads.bam'
    mafia_df = parse_mafia_bam(mafia_bam)

    # Merge: classification + poly(A) + MAFIA
    merged = cls_df.merge(polya, on='read_id', how='inner')
    if len(mafia_df) > 0:
        merged = merged.merge(
            mafia_df[['read_id', 'read_length', 'm6a_sites', 'psi_sites',
                       'm6a_per_kb', 'psi_per_kb']],
            on='read_id', how='left',
            suffixes=('_cls', '')
        )
        # Use MAFIA read_length if available, else classification
        if 'read_length_cls' in merged.columns:
            merged['read_length'] = merged['read_length'].fillna(merged['read_length_cls'])
            merged.drop(columns=['read_length_cls'], inplace=True)
    else:
        if 'read_length' not in merged.columns:
            merged['read_length'] = np.nan
        merged['m6a_sites'] = 0
        merged['psi_sites'] = 0
        merged['m6a_per_kb'] = 0.0
        merged['psi_per_kb'] = 0.0

    merged['group'] = group
    merged['condition'] = 'Ars' if 'Ars' in group else 'HeLa'
    merged['replicate'] = group.split('_')[-1]

    return merged


def load_l1_data(group):
    """Load L1 data from summary + part3 cache."""
    summary_file = RESULTS / group / 'g_summary' / f'{group}_L1_summary.tsv'
    if not summary_file.exists():
        return None
    summary = pd.read_csv(summary_file, sep='\t')
    summary = summary[summary['qc_tag'] == 'PASS'].copy()

    summary['l1_location'] = summary['overlapping_genes'].apply(
        lambda x: 'intergenic' if pd.isna(x) or x == '' else 'intronic'
    )
    summary['age'] = summary['gene_id'].apply(
        lambda x: 'young' if x in YOUNG_FAMILIES else 'ancient'
    )

    cache_file = TOPIC_05 / f'part3_l1_per_read_cache/{group}_l1_per_read.tsv'
    if cache_file.exists():
        cache = pd.read_csv(cache_file, sep='\t')
        cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000.0)
        cache['psi_per_kb'] = cache['psi_sites_high'] / (cache['read_length'] / 1000.0)
        cache = cache.rename(columns={'m6a_sites_high': 'm6a_sites', 'psi_sites_high': 'psi_sites'})
        summary = summary.merge(
            cache[['read_id', 'read_length', 'm6a_sites', 'psi_sites', 'm6a_per_kb', 'psi_per_kb']],
            on='read_id', how='left', suffixes=('_summary', '')
        )
    else:
        summary['m6a_sites'] = np.nan
        summary['m6a_per_kb'] = np.nan

    summary['group'] = group
    summary['condition'] = 'Ars' if 'Ars' in group else 'HeLa'
    summary['replicate'] = group.split('_')[-1]
    return summary


def load_ctrl_data(group):
    """Load mRNA control from part3 ctrl cache."""
    cache_file = TOPIC_05 / f'part3_ctrl_per_read_cache/{group}_ctrl_per_read.tsv'
    if not cache_file.exists():
        return None
    cache = pd.read_csv(cache_file, sep='\t')
    cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000.0)
    cache['psi_per_kb'] = cache['psi_sites_high'] / (cache['read_length'] / 1000.0)
    cache = cache.rename(columns={'m6a_sites_high': 'm6a_sites', 'psi_sites_high': 'psi_sites'})
    cache['group'] = group
    cache['condition'] = 'Ars' if 'Ars' in group else 'HeLa'
    cache['replicate'] = group.split('_')[-1]

    polya_file = RESULTS / group / 'i_control' / f'{group}.control.nanopolish.polya.tsv.gz'
    if polya_file.exists():
        polya = pd.read_csv(polya_file, sep='\t')
        polya = polya[polya['qc_tag'] == 'PASS']
        polya = polya.rename(columns={'readname': 'read_id', 'polya_length': 'polya'})
        cache = cache.merge(polya[['read_id', 'polya']], on='read_id', how='left')

    return cache


# =============================================================================
# Statistical helpers
# =============================================================================
def bootstrap_median_ci(arr, n_boot=5000, ci=0.95, seed=42):
    """Bootstrap 95% CI for median."""
    rng = np.random.RandomState(seed)
    arr = np.asarray(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return np.nan, np.nan
    medians = [np.median(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(medians, (1 - ci) / 2 * 100)
    hi = np.percentile(medians, (1 + ci) / 2 * 100)
    return lo, hi


def cohen_d(x, y):
    """Cohen's d effect size."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_sd


def replicate_level_test(df, condition_col, value_col, groups_a, groups_b):
    """Per-replicate medians → paired or unpaired test."""
    meds_a = [df[df['group'] == g][value_col].median() for g in groups_a
              if g in df['group'].values]
    meds_b = [df[df['group'] == g][value_col].median() for g in groups_b
              if g in df['group'].values]
    meds_a = [m for m in meds_a if not np.isnan(m)]
    meds_b = [m for m in meds_b if not np.isnan(m)]
    if len(meds_a) >= 2 and len(meds_b) >= 2:
        stat, pval = stats.mannwhitneyu(meds_a, meds_b, alternative='two-sided')
        return meds_a, meds_b, pval
    return meds_a, meds_b, np.nan


def read_length_match(target_df, reference_df, target_rl_col='read_length',
                      ref_rl_col='read_length', n_bins=50, seed=42):
    """Subsample target_df to match reference_df's read length distribution.
    Uses histogram-based matching.
    """
    rng = np.random.RandomState(seed)
    ref_rl = reference_df[ref_rl_col].dropna().values
    tgt_rl = target_df[target_rl_col].dropna().values

    if len(ref_rl) == 0 or len(tgt_rl) == 0:
        return target_df

    # Define bins from combined range
    lo = min(ref_rl.min(), tgt_rl.min())
    hi = max(ref_rl.max(), tgt_rl.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    ref_counts, _ = np.histogram(ref_rl, bins=bins)
    ref_frac = ref_counts / ref_counts.sum()

    # Assign bins to target reads
    target_df = target_df.copy()
    target_df['_rl_bin'] = pd.cut(target_df[target_rl_col], bins=bins, labels=False, include_lowest=True)
    target_df = target_df.dropna(subset=['_rl_bin'])

    # Target n per bin
    total_target = min(len(target_df), len(reference_df))
    target_per_bin = (ref_frac * total_target).astype(int)

    sampled_indices = []
    for b_idx in range(n_bins):
        bin_df = target_df[target_df['_rl_bin'] == b_idx]
        n_want = target_per_bin[b_idx]
        if len(bin_df) == 0 or n_want == 0:
            continue
        n_take = min(n_want, len(bin_df))
        sampled_indices.extend(rng.choice(bin_df.index, size=n_take, replace=False))

    result = target_df.loc[sampled_indices].drop(columns=['_rl_bin'])
    return result


# =============================================================================
# Analysis
# =============================================================================
def main():
    print("=" * 70)
    print("lncRNA control comparison: lncRNA vs L1 vs mRNA")
    print("=" * 70)

    # --- Load all data ---
    lnc_frames, l1_frames, ctrl_frames = [], [], []

    for group in ALL_GROUPS:
        print(f"\nLoading {group}...")

        lnc = load_lncrna_data(group)
        if lnc is not None:
            lnc_frames.append(lnc)
            n_sub = lnc['lncrna_subtype'].value_counts().to_dict() if 'lncrna_subtype' in lnc.columns else {}
            print(f"  lncRNA: {len(lnc)} reads (subtypes: {n_sub})")

        l1 = load_l1_data(group)
        if l1 is not None:
            l1_frames.append(l1)
            print(f"  L1: {len(l1)} reads")

        ctrl = load_ctrl_data(group)
        if ctrl is not None:
            ctrl_frames.append(ctrl)
            print(f"  mRNA ctrl: {len(ctrl)} reads")

    if not lnc_frames:
        print("\nERROR: No lncRNA data loaded. Run pipeline first.")
        return

    lnc_all = pd.concat(lnc_frames, ignore_index=True)
    l1_all = pd.concat(l1_frames, ignore_index=True)
    ctrl_all = pd.concat(ctrl_frames, ignore_index=True) if ctrl_frames else pd.DataFrame()

    print(f"\n{'='*70}")
    print("Data summary")
    print(f"{'='*70}")
    print(f"lncRNA:      {len(lnc_all):,} reads (with poly(A))")
    if 'lncrna_subtype' in lnc_all.columns:
        for st, cnt in lnc_all['lncrna_subtype'].value_counts().items():
            print(f"  {st}: {cnt:,}")
    print(f"L1 (PASS):   {len(l1_all):,} reads")
    if len(ctrl_all) > 0:
        print(f"mRNA ctrl:   {len(ctrl_all):,} reads")

    # =========================================================================
    # Comparison 1: Baseline Poly(A) — lncRNA vs L1 vs mRNA
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 1: Baseline poly(A) length (HeLa only)")
    print(f"{'='*70}")

    lnc_hela = lnc_all[lnc_all['condition'] == 'HeLa']
    l1_hela = l1_all[l1_all['condition'] == 'HeLa']
    ctrl_hela = ctrl_all[ctrl_all['condition'] == 'HeLa'] if len(ctrl_all) > 0 else pd.DataFrame()

    categories_baseline = []
    if len(lnc_hela) > 0:
        categories_baseline.append(('lncRNA', lnc_hela['polya']))
    if len(l1_hela) > 0:
        categories_baseline.append(('L1', l1_hela['polya_length']))
        categories_baseline.append(('L1 (young)', l1_hela[l1_hela['age'] == 'young']['polya_length']))
        categories_baseline.append(('L1 (ancient)', l1_hela[l1_hela['age'] == 'ancient']['polya_length']))
    if len(ctrl_hela) > 0 and 'polya' in ctrl_hela.columns:
        categories_baseline.append(('mRNA', ctrl_hela['polya'].dropna()))

    print(f"\n{'Category':<20} {'n':>8} {'median':>10} {'mean':>10} {'95% CI':>20}")
    print("-" * 72)
    for name, vals in categories_baseline:
        vals = vals.dropna()
        if len(vals) > 0:
            med = vals.median()
            lo, hi = bootstrap_median_ci(vals)
            print(f"{name:<20} {len(vals):>8} {med:>10.1f} {vals.mean():>10.1f} [{lo:.1f}, {hi:.1f}]")

    # Statistical tests
    if len(lnc_hela) > 0 and len(l1_hela) > 0:
        stat, pval = stats.mannwhitneyu(
            lnc_hela['polya'].dropna(), l1_hela['polya_length'].dropna(), alternative='two-sided')
        d = cohen_d(lnc_hela['polya'].dropna().values, l1_hela['polya_length'].dropna().values)
        print(f"\n  lncRNA vs L1: MWU P={pval:.2e}, Cohen's d={d:.3f}")

    if len(lnc_hela) > 0 and len(ctrl_hela) > 0 and 'polya' in ctrl_hela.columns:
        stat, pval = stats.mannwhitneyu(
            lnc_hela['polya'].dropna(), ctrl_hela['polya'].dropna(), alternative='two-sided')
        d = cohen_d(lnc_hela['polya'].dropna().values, ctrl_hela['polya'].dropna().values)
        print(f"  lncRNA vs mRNA: MWU P={pval:.2e}, Cohen's d={d:.3f}")

    # Replicate-level
    print("\n  Per-replicate medians (HeLa):")
    for group in HELA_GROUPS:
        lnc_g = lnc_all[(lnc_all['group'] == group) & (lnc_all['condition'] == 'HeLa')]
        l1_g = l1_all[(l1_all['group'] == group) & (l1_all['condition'] == 'HeLa')]
        lnc_med = lnc_g['polya'].median() if len(lnc_g) > 0 else np.nan
        l1_med = l1_g['polya_length'].median() if len(l1_g) > 0 else np.nan
        print(f"    {group}: lncRNA={lnc_med:.1f}, L1={l1_med:.1f}")

    # =========================================================================
    # Comparison 2: Arsenite Δpoly(A)
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 2: Arsenite-induced poly(A) change (Δ = Ars - HeLa)")
    print(f"{'='*70}")

    delta_results = {}

    comparisons = [
        ('lncRNA', lnc_all, 'polya'),
        ('L1', l1_all, 'polya_length'),
        ('L1 (young)', l1_all[l1_all['age'] == 'young'], 'polya_length'),
        ('L1 (ancient)', l1_all[l1_all['age'] == 'ancient'], 'polya_length'),
    ]
    if len(ctrl_all) > 0 and 'polya' in ctrl_all.columns:
        comparisons.append(('mRNA', ctrl_all, 'polya'))

    print(f"\n{'Category':<20} {'HeLa med':>10} {'Ars med':>10} {'Δ':>8} {'MWU P':>12} {'d':>8}")
    print("-" * 72)

    for name, df, polya_col in comparisons:
        hela_vals = df[df['condition'] == 'HeLa'][polya_col].dropna()
        ars_vals = df[df['condition'] == 'Ars'][polya_col].dropna()
        if len(hela_vals) > 0 and len(ars_vals) > 0:
            med_h = hela_vals.median()
            med_a = ars_vals.median()
            delta = med_a - med_h
            stat, pval = stats.mannwhitneyu(hela_vals, ars_vals, alternative='two-sided')
            d = cohen_d(ars_vals.values, hela_vals.values)
            print(f"{name:<20} {med_h:>10.1f} {med_a:>10.1f} {delta:>+8.1f} {pval:>12.2e} {d:>+8.3f}")
            delta_results[name] = {
                'hela': med_h, 'ars': med_a, 'delta': delta, 'p': pval, 'd': d,
                'n_hela': len(hela_vals), 'n_ars': len(ars_vals),
            }

    # Replicate-level test for lncRNA
    if len(lnc_all) > 0:
        meds_h, meds_a, rep_p = replicate_level_test(
            lnc_all, 'condition', 'polya', HELA_GROUPS, ARS_GROUPS)
        print(f"\n  lncRNA replicate-level: HeLa medians={[f'{m:.1f}' for m in meds_h]}, "
              f"Ars medians={[f'{m:.1f}' for m in meds_a]}, MWU P={rep_p:.3f}")

    # =========================================================================
    # Comparison 3: m6A density
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 3: m6A/kb by category")
    print(f"{'='*70}")

    m6a_cats = [
        ('lncRNA', lnc_all[lnc_all['m6a_per_kb'].notna()]['m6a_per_kb']),
        ('L1', l1_all[l1_all['m6a_per_kb'].notna()]['m6a_per_kb']),
    ]
    if len(ctrl_all) > 0 and 'm6a_per_kb' in ctrl_all.columns:
        m6a_cats.append(('mRNA', ctrl_all[ctrl_all['m6a_per_kb'].notna()]['m6a_per_kb']))

    print(f"\n{'Category':<20} {'n':>8} {'median':>10} {'mean':>10} {'95% CI':>20}")
    print("-" * 62)
    for name, vals in m6a_cats:
        if len(vals) > 0:
            lo, hi = bootstrap_median_ci(vals)
            print(f"{name:<20} {len(vals):>8} {vals.median():>10.3f} {vals.mean():>10.3f} [{lo:.3f}, {hi:.3f}]")

    # L1 vs lncRNA
    lnc_m6a = lnc_all[lnc_all['m6a_per_kb'].notna()]['m6a_per_kb']
    l1_m6a = l1_all[l1_all['m6a_per_kb'].notna()]['m6a_per_kb']
    if len(lnc_m6a) > 0 and len(l1_m6a) > 0:
        stat, pval = stats.mannwhitneyu(l1_m6a, lnc_m6a, alternative='two-sided')
        fc = l1_m6a.median() / lnc_m6a.median() if lnc_m6a.median() > 0 else np.inf
        print(f"\n  L1 vs lncRNA m6A/kb: {l1_m6a.median():.3f} vs {lnc_m6a.median():.3f} "
              f"= {fc:.2f}x, MWU P={pval:.2e}")

    # =========================================================================
    # Comparison 4: m6A-poly(A) correlation
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 4: m6A-poly(A) correlation")
    print(f"{'='*70}")

    corr_sources = [
        ('lncRNA', lnc_all, 'm6a_per_kb', 'polya'),
        ('L1', l1_all, 'm6a_per_kb', 'polya_length'),
    ]
    if len(ctrl_all) > 0 and 'polya' in ctrl_all.columns:
        corr_sources.append(('mRNA', ctrl_all, 'm6a_per_kb', 'polya'))

    for src_name, df, m6a_col, polya_col in corr_sources:
        for cond in ['HeLa', 'Ars']:
            sub = df[(df['condition'] == cond) & df[m6a_col].notna() & df[polya_col].notna()]
            if len(sub) < 10:
                continue
            rho, pval = stats.spearmanr(sub[m6a_col], sub[polya_col])
            print(f"  {src_name} ({cond}): n={len(sub):,}, rho={rho:.3f}, P={pval:.2e}")

    # =========================================================================
    # Comparison 5: lncRNA subtype breakdown
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 5: lncRNA subtype breakdown")
    print(f"{'='*70}")

    if 'lncrna_subtype' in lnc_all.columns:
        subtypes = lnc_all['lncrna_subtype'].value_counts()
        top_subtypes = subtypes[subtypes >= 50].index.tolist()

        print(f"\n{'Subtype':<25} {'n':>6} {'HeLa med':>10} {'Ars med':>10} {'Δ':>8} {'P':>12}")
        print("-" * 75)

        subtype_results = {}
        for st in top_subtypes:
            st_df = lnc_all[lnc_all['lncrna_subtype'] == st]
            hela_vals = st_df[st_df['condition'] == 'HeLa']['polya'].dropna()
            ars_vals = st_df[st_df['condition'] == 'Ars']['polya'].dropna()
            if len(hela_vals) > 5 and len(ars_vals) > 5:
                delta = ars_vals.median() - hela_vals.median()
                stat, pval = stats.mannwhitneyu(hela_vals, ars_vals, alternative='two-sided')
                print(f"{st:<25} {len(st_df):>6} {hela_vals.median():>10.1f} "
                      f"{ars_vals.median():>10.1f} {delta:>+8.1f} {pval:>12.2e}")
                subtype_results[st] = {
                    'n': len(st_df), 'hela': hela_vals.median(),
                    'ars': ars_vals.median(), 'delta': delta, 'p': pval,
                }

        # m6A by subtype
        print(f"\n  m6A/kb by subtype:")
        for st in top_subtypes:
            st_m6a = lnc_all[(lnc_all['lncrna_subtype'] == st) & lnc_all['m6a_per_kb'].notna()]['m6a_per_kb']
            if len(st_m6a) > 0:
                print(f"    {st}: n={len(st_m6a)}, median={st_m6a.median():.3f}")

    # =========================================================================
    # Comparison 6: Read-length matched comparison
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 6: Read-length matched lncRNA vs L1")
    print(f"{'='*70}")

    # Match lncRNA read lengths to L1 distribution
    l1_with_rl = l1_all[l1_all['read_length'].notna() & l1_all['polya_length'].notna()].copy()
    lnc_with_rl = lnc_all[lnc_all['read_length'].notna() & lnc_all['polya'].notna()].copy()

    if len(l1_with_rl) > 0 and len(lnc_with_rl) > 0:
        print(f"\n  Before matching:")
        print(f"    L1 read length:    median={l1_with_rl['read_length'].median():.0f}, "
              f"mean={l1_with_rl['read_length'].mean():.0f}")
        print(f"    lncRNA read length: median={lnc_with_rl['read_length'].median():.0f}, "
              f"mean={lnc_with_rl['read_length'].mean():.0f}")

        lnc_matched = read_length_match(lnc_with_rl, l1_with_rl)
        print(f"\n  After matching: {len(lnc_matched):,} lncRNA reads (from {len(lnc_with_rl):,})")
        print(f"    Matched read length: median={lnc_matched['read_length'].median():.0f}, "
              f"mean={lnc_matched['read_length'].mean():.0f}")

        # Repeat key comparisons on matched data
        for cond in ['HeLa', 'Ars']:
            lnc_cond = lnc_matched[lnc_matched['condition'] == cond]['polya'].dropna()
            l1_cond = l1_with_rl[l1_with_rl['condition'] == cond]['polya_length'].dropna()
            if len(lnc_cond) > 10 and len(l1_cond) > 10:
                stat, pval = stats.mannwhitneyu(lnc_cond, l1_cond, alternative='two-sided')
                print(f"    {cond}: lncRNA(matched) median={lnc_cond.median():.1f} vs "
                      f"L1 median={l1_cond.median():.1f}, MWU P={pval:.2e}")

        # Matched m6A comparison
        lnc_m6a_matched = lnc_matched[lnc_matched['m6a_per_kb'].notna()]['m6a_per_kb']
        if len(lnc_m6a_matched) > 0:
            stat, pval = stats.mannwhitneyu(l1_m6a, lnc_m6a_matched, alternative='two-sided')
            fc = l1_m6a.median() / lnc_m6a_matched.median() if lnc_m6a_matched.median() > 0 else np.inf
            print(f"    m6A/kb (matched): L1 {l1_m6a.median():.3f} vs lncRNA {lnc_m6a_matched.median():.3f} "
                  f"= {fc:.2f}x, P={pval:.2e}")

    # =========================================================================
    # Figures
    # =========================================================================
    print(f"\n{'='*70}")
    print("Generating figures...")
    print(f"{'='*70}")

    # --- Figure 1: Violin — lncRNA vs L1 vs mRNA poly(A) (HeLa / Ars) ---
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_data = []
    for _, row in lnc_all.iterrows():
        if pd.notna(row.get('polya')):
            plot_data.append({'Source': 'lncRNA', 'Condition': row['condition'],
                              'Poly(A)': row['polya']})
    for _, row in l1_all.iterrows():
        if pd.notna(row.get('polya_length')):
            plot_data.append({'Source': 'L1', 'Condition': row['condition'],
                              'Poly(A)': row['polya_length']})
    if len(ctrl_all) > 0 and 'polya' in ctrl_all.columns:
        for _, row in ctrl_all.iterrows():
            if pd.notna(row.get('polya')):
                plot_data.append({'Source': 'mRNA', 'Condition': row['condition'],
                                  'Poly(A)': row['polya']})

    pdf = pd.DataFrame(plot_data)
    if len(pdf) > 0:
        order = ['lncRNA', 'L1', 'mRNA'] if 'mRNA' in pdf['Source'].unique() else ['lncRNA', 'L1']
        sns.violinplot(data=pdf, x='Source', y='Poly(A)', hue='Condition',
                       split=True, inner='quartile', ax=ax, order=order,
                       palette={'HeLa': '#4DBEEE', 'Ars': '#D95319'}, cut=0)
        ax.set_ylabel('Poly(A) length (nt)')
        ax.set_xlabel('')
        ax.set_title('Poly(A) length: lncRNA vs L1 vs mRNA')

        # Annotate deltas
        for i, src in enumerate(order):
            if src in delta_results:
                r = delta_results[src]
                sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
                ax.text(i, ax.get_ylim()[1] * 0.93,
                        f"Δ={r['delta']:+.1f}\n{sig}",
                        ha='center', va='top', fontsize=7)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig1_lncrna_polya_violin.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'fig1_lncrna_polya_violin.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("  Saved: fig1_lncrna_polya_violin.pdf")

    # --- Figure 2: Bar — Δpoly(A) comparison across all categories ---
    fig, ax = plt.subplots(figsize=(7, 4))

    # Also load noncoding results for comparison
    nc_results_file = TOPIC_05 / 'noncoding_control_figures/noncoding_comparison_results.tsv'
    nc_results = {}
    if nc_results_file.exists():
        nc_df = pd.read_csv(nc_results_file, sep='\t')
        for _, row in nc_df.iterrows():
            nc_results[row['category']] = {
                'delta': row['delta_nt'], 'p': row['MWU_pvalue'],
            }

    bar_items = []
    color_map = {
        'lncRNA': '#9467bd',
        'L1': '#d62728', 'L1 (ancient)': '#ff7f7f', 'L1 (young)': '#8b0000',
        'mRNA': '#2ca02c',
        'non-L1 intronic': '#1f77b4', 'non-L1 intergenic': '#aec7e8',
    }

    for name in ['lncRNA', 'L1', 'L1 (ancient)', 'L1 (young)', 'mRNA']:
        if name in delta_results:
            r = delta_results[name]
            bar_items.append((name, r['delta'], r['p'], color_map.get(name, '#999')))

    for name in ['non-L1 intronic', 'non-L1 intergenic']:
        if name in nc_results:
            r = nc_results[name]
            bar_items.append((name, r['delta'], r['p'], color_map.get(name, '#999')))

    if bar_items:
        names, deltas, pvals, colors = zip(*bar_items)
        x = np.arange(len(names))
        bars = ax.bar(x, deltas, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('Δ Poly(A) (Ars − HeLa, nt)')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title('Arsenite-induced poly(A) change by transcript type')

        for bar, p in zip(bars, pvals):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            y = bar.get_height()
            offset = -2 if y < 0 else 2
            ax.text(bar.get_x() + bar.get_width() / 2, y + offset, sig,
                    ha='center', va='bottom' if y >= 0 else 'top', fontsize=7, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig2_delta_polya_bar.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'fig2_delta_polya_bar.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("  Saved: fig2_delta_polya_bar.pdf")

    # --- Figure 3: Box — m6A/kb comparison ---
    fig, ax = plt.subplots(figsize=(6, 5))

    m6a_plot_data = []
    for name, vals in m6a_cats:
        for v in vals:
            m6a_plot_data.append({'Category': name, 'm6A/kb': v})

    m6a_pdf = pd.DataFrame(m6a_plot_data)
    if len(m6a_pdf) > 0:
        order = [name for name, _ in m6a_cats]
        colors = [color_map.get(n, '#999') for n in order]
        sns.boxplot(data=m6a_pdf, x='Category', y='m6A/kb', order=order,
                    showfliers=False, ax=ax, palette=colors)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_title('m6A density: lncRNA vs L1 vs mRNA')
        ax.set_ylabel('m6A sites / kb')
        ax.set_xlabel('')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig3_m6a_comparison.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'fig3_m6a_comparison.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("  Saved: fig3_m6a_comparison.pdf")

    # --- Figure 4: Scatter — m6A vs poly(A) correlation ---
    fig, axes = plt.subplots(1, 3 if len(ctrl_all) > 0 else 2, figsize=(12 if len(ctrl_all) > 0 else 9, 4.5))
    if not hasattr(axes, '__len__'):
        axes = [axes]

    scatter_sources = [
        ('lncRNA (Ars)', lnc_all[lnc_all['condition'] == 'Ars'], 'm6a_per_kb', 'polya', '#9467bd'),
        ('L1 (Ars)', l1_all[l1_all['condition'] == 'Ars'], 'm6a_per_kb', 'polya_length', '#d62728'),
    ]
    if len(ctrl_all) > 0 and 'polya' in ctrl_all.columns:
        scatter_sources.append(
            ('mRNA (Ars)', ctrl_all[ctrl_all['condition'] == 'Ars'], 'm6a_per_kb', 'polya', '#2ca02c'))

    for idx, (src_name, df, m6a_col, polya_col, color) in enumerate(scatter_sources):
        ax = axes[idx]
        sub = df[df[m6a_col].notna() & df[polya_col].notna()].copy()
        if len(sub) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            continue

        ax.scatter(sub[m6a_col], sub[polya_col], alpha=0.15, s=5, c=color)
        rho, pval = stats.spearmanr(sub[m6a_col], sub[polya_col])
        ax.set_xlabel('m6A / kb')
        ax.set_ylabel('Poly(A) length (nt)' if idx == 0 else '')
        ax.set_title(f'{src_name}\n(n={len(sub):,}, ρ={rho:.3f}, P={pval:.1e})')
        ax.set_xlim(left=-0.5)
        ax.set_ylim(bottom=0)

    fig.suptitle('m6A–Poly(A) correlation under arsenite stress', fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig4_m6a_polya_scatter.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'fig4_m6a_polya_scatter.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("  Saved: fig4_m6a_polya_scatter.pdf")

    # --- Figure 5: lncRNA subtype breakdown — poly(A) violin ---
    if 'lncrna_subtype' in lnc_all.columns:
        top_subtypes = lnc_all['lncrna_subtype'].value_counts()
        top_subtypes = top_subtypes[top_subtypes >= 50].index.tolist()

        if top_subtypes:
            fig, ax = plt.subplots(figsize=(max(6, len(top_subtypes) * 1.5), 5))
            sub_df = lnc_all[lnc_all['lncrna_subtype'].isin(top_subtypes)].copy()

            sns.violinplot(data=sub_df, x='lncrna_subtype', y='polya',
                           hue='condition', split=True, inner='quartile',
                           order=top_subtypes, ax=ax,
                           palette={'HeLa': '#4DBEEE', 'Ars': '#D95319'}, cut=0)
            ax.set_xlabel('lncRNA subtype')
            ax.set_ylabel('Poly(A) length (nt)')
            ax.set_title('Poly(A) by lncRNA subtype (HeLa vs Ars)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

            fig.tight_layout()
            fig.savefig(OUTDIR / 'fig5_lncrna_subtype_polya.pdf', bbox_inches='tight')
            fig.savefig(OUTDIR / 'fig5_lncrna_subtype_polya.png', bbox_inches='tight', dpi=200)
            plt.close(fig)
            print("  Saved: fig5_lncrna_subtype_polya.pdf")

    # =========================================================================
    # Save results table
    # =========================================================================
    results_rows = []
    for name, r in delta_results.items():
        results_rows.append({
            'category': name, 'n_HeLa': r['n_hela'], 'n_Ars': r['n_ars'],
            'median_HeLa': r['hela'], 'median_Ars': r['ars'],
            'delta_nt': r['delta'], 'MWU_pvalue': r['p'], 'cohens_d': r['d'],
        })

    results_file = OUTDIR / 'lncrna_comparison_results.tsv'
    if results_rows:
        pd.DataFrame(results_rows).to_csv(results_file, sep='\t', index=False)
        print(f"\n  Results table: {results_file}")

    # =========================================================================
    # Interpretation
    # =========================================================================
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    lnc_r = delta_results.get('lncRNA', {})
    l1_r = delta_results.get('L1', {})

    if lnc_r and l1_r:
        print(f"\n  Baseline poly(A):")
        lnc_base = delta_results.get('lncRNA', {}).get('hela', np.nan)
        l1_base = delta_results.get('L1', {}).get('hela', np.nan)
        if not np.isnan(lnc_base) and not np.isnan(l1_base):
            if abs(lnc_base - l1_base) < 15:
                print(f"    lncRNA ({lnc_base:.0f}nt) ≈ L1 ({l1_base:.0f}nt)")
                print(f"    → Long poly(A) is a general non-coding transcript property")
            elif lnc_base < l1_base:
                print(f"    lncRNA ({lnc_base:.0f}nt) < L1 ({l1_base:.0f}nt)")
                print(f"    → L1 has uniquely long poly(A), not just a non-coding property")
            else:
                print(f"    lncRNA ({lnc_base:.0f}nt) > L1 ({l1_base:.0f}nt)")
                print(f"    → lncRNA has even longer poly(A)")

        print(f"\n  Arsenite shortening:")
        lnc_delta = lnc_r.get('delta', 0)
        l1_delta = l1_r.get('delta', 0)
        lnc_p = lnc_r.get('p', 1)

        if abs(lnc_delta) > 10 and lnc_p < 0.05:
            print(f"    lncRNA Δ={lnc_delta:+.1f}nt (P={lnc_p:.2e})")
            print(f"    → lncRNA also shows significant poly(A) shortening")
            if abs(lnc_delta) > abs(l1_delta) * 0.5:
                print(f"    → Arsenite effect may be a general non-coding property")
            else:
                print(f"    → But much weaker than L1 (Δ={l1_delta:+.1f}nt)")
        elif abs(lnc_delta) < 10 or lnc_p >= 0.05:
            print(f"    lncRNA Δ={lnc_delta:+.1f}nt (P={lnc_p:.2e})")
            print(f"    → lncRNA does NOT show significant shortening")
            print(f"    → Arsenite poly(A) shortening is L1-SPECIFIC")

    print("\nDone.")


if __name__ == '__main__':
    main()
