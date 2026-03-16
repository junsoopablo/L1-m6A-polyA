#!/usr/bin/env python3
"""Non-coding control comparison: intronic/intergenic non-L1 vs L1.

Tests whether arsenite-induced poly(A) shortening is L1-specific or a
general property of intronic/intergenic non-coding RNA.

Data sources:
  - Noncoding non-L1: nanopolish + MAFIA from k_noncoding_ctrl/
  - L1: part3 cache + g_summary/ poly(A)
  - Existing mRNA control: part3 ctrl cache (reference only)

Key comparisons:
  1. Intronic L1 vs Intronic non-L1, HeLa vs Ars (poly(A) shortening)
  2. Intergenic L1 vs Intergenic non-L1, HeLa vs Ars
  3. m6A/kb across categories
  4. m6A-poly(A) correlation in non-L1 noncoding reads
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
OUTDIR = TOPIC_05 / 'noncoding_control_figures'
OUTDIR.mkdir(exist_ok=True)

PROB_THRESHOLD = 204  # 80% on 0-255 scale

HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3']
ARS_GROUPS = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']
ALL_GROUPS = HELA_GROUPS + ARS_GROUPS

YOUNG_FAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 200})

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


def safe_parse_list(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or val in ('', '[]', 'nan'):
        return []
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


# =============================================================================
# Data loading
# =============================================================================
def load_noncoding_data(group):
    """Load noncoding non-L1 data: nanopolish poly(A) + MAFIA + classification."""
    nc_dir = RESULTS / group / 'k_noncoding_ctrl'

    # Read classification
    cls_file = nc_dir / f'{group}_read_classification.tsv'
    if not cls_file.exists():
        print(f"  WARNING: classification not found for {group}")
        return None
    cls_df = pd.read_csv(cls_file, sep='\t')

    # Nanopolish poly(A)
    polya_file = nc_dir / f'{group}.noncoding.nanopolish.polya.tsv.gz'
    if not polya_file.exists():
        print(f"  WARNING: nanopolish not found for {group}")
        return None
    polya = pd.read_csv(polya_file, sep='\t')
    polya = polya[polya['qc_tag'] == 'PASS'].copy()
    polya = polya.rename(columns={'readname': 'read_id', 'polya_length': 'polya'})
    polya = polya[['read_id', 'polya', 'contig', 'position']].copy()

    # MAFIA
    mafia_bam = nc_dir / f'mafia/{group}.noncoding.mAFiA.reads.bam'
    mafia_df = parse_mafia_bam(mafia_bam)

    # Merge: classification + poly(A) + MAFIA
    merged = cls_df.merge(polya, on='read_id', how='inner')
    if len(mafia_df) > 0:
        merged = merged.merge(mafia_df[['read_id', 'read_length', 'm6a_sites', 'psi_sites',
                                         'm6a_per_kb', 'psi_per_kb']],
                              on='read_id', how='left')
    else:
        merged['read_length'] = np.nan
        merged['m6a_sites'] = 0
        merged['psi_sites'] = 0
        merged['m6a_per_kb'] = 0.0
        merged['psi_per_kb'] = 0.0

    merged['group'] = group
    merged['condition'] = 'Ars' if 'Ars' in group else 'HeLa'

    return merged


def load_l1_data(group):
    """Load L1 data from summary + part3 cache."""
    # L1 summary (has poly(A), overlapping_genes for intronic/intergenic classification)
    summary_file = RESULTS / group / 'g_summary' / f'{group}_L1_summary.tsv'
    if not summary_file.exists():
        print(f"  WARNING: L1 summary not found for {group}")
        return None
    summary = pd.read_csv(summary_file, sep='\t')
    summary = summary[summary['qc_tag'] == 'PASS'].copy()

    # Classify L1 as intronic or intergenic
    summary['l1_location'] = summary['overlapping_genes'].apply(
        lambda x: 'intergenic' if pd.isna(x) or x == '' else 'intronic'
    )

    # Identify young L1
    summary['age'] = summary['gene_id'].apply(
        lambda x: 'young' if x in YOUNG_FAMILIES else 'ancient'
    )

    # Part3 cache (modification data)
    cache_file = TOPIC_05 / f'part3_l1_per_read_cache/{group}_l1_per_read.tsv'
    if cache_file.exists():
        cache = pd.read_csv(cache_file, sep='\t')
        # Compute m6a_per_kb from cache
        cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000.0)
        cache['psi_per_kb'] = cache['psi_sites_high'] / (cache['read_length'] / 1000.0)
        cache = cache.rename(columns={
            'm6a_sites_high': 'm6a_sites',
            'psi_sites_high': 'psi_sites',
        })
        summary = summary.merge(
            cache[['read_id', 'read_length', 'm6a_sites', 'psi_sites', 'm6a_per_kb', 'psi_per_kb']],
            on='read_id', how='left', suffixes=('_summary', '')
        )
    else:
        summary['m6a_sites'] = np.nan
        summary['m6a_per_kb'] = np.nan
        summary['psi_sites'] = np.nan
        summary['psi_per_kb'] = np.nan

    summary['group'] = group
    summary['condition'] = 'Ars' if 'Ars' in group else 'HeLa'

    return summary


def load_ctrl_data(group):
    """Load existing mRNA control from part3 ctrl cache."""
    cache_file = TOPIC_05 / f'part3_ctrl_per_read_cache/{group}_ctrl_per_read.tsv'
    if not cache_file.exists():
        return None
    cache = pd.read_csv(cache_file, sep='\t')
    cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000.0)
    cache['psi_per_kb'] = cache['psi_sites_high'] / (cache['read_length'] / 1000.0)
    cache = cache.rename(columns={'m6a_sites_high': 'm6a_sites', 'psi_sites_high': 'psi_sites'})
    cache['group'] = group
    cache['condition'] = 'Ars' if 'Ars' in group else 'HeLa'
    cache['category'] = 'mRNA_ctrl'

    # Poly(A) from control nanopolish
    polya_file = RESULTS / group / 'i_control' / f'{group}.control.nanopolish.polya.tsv.gz'
    if polya_file.exists():
        polya = pd.read_csv(polya_file, sep='\t')
        polya = polya[polya['qc_tag'] == 'PASS']
        polya = polya.rename(columns={'readname': 'read_id', 'polya_length': 'polya'})
        cache = cache.merge(polya[['read_id', 'polya']], on='read_id', how='left')

    return cache


# =============================================================================
# Analysis
# =============================================================================
def main():
    print("=" * 70)
    print("Non-coding control comparison: Intronic/Intergenic non-L1 vs L1")
    print("=" * 70)

    # --- Load all data ---
    nc_frames = []
    l1_frames = []
    ctrl_frames = []

    for group in ALL_GROUPS:
        print(f"\nLoading {group}...")

        nc = load_noncoding_data(group)
        if nc is not None:
            nc_frames.append(nc)
            print(f"  Noncoding: {len(nc)} reads ({nc['category'].value_counts().to_dict()})")

        l1 = load_l1_data(group)
        if l1 is not None:
            l1_frames.append(l1)
            print(f"  L1: {len(l1)} reads ({l1['l1_location'].value_counts().to_dict()})")

        ctrl = load_ctrl_data(group)
        if ctrl is not None:
            ctrl_frames.append(ctrl)
            print(f"  mRNA ctrl: {len(ctrl)} reads")

    if not nc_frames:
        print("\nERROR: No noncoding data loaded. Run pipeline first.")
        return

    nc_all = pd.concat(nc_frames, ignore_index=True)
    l1_all = pd.concat(l1_frames, ignore_index=True)
    ctrl_all = pd.concat(ctrl_frames, ignore_index=True) if ctrl_frames else pd.DataFrame()

    print(f"\n{'='*70}")
    print("Data summary")
    print(f"{'='*70}")
    print(f"Noncoding non-L1: {len(nc_all):,} reads")
    print(f"  Intronic:   {(nc_all['category']=='intronic').sum():,}")
    print(f"  Intergenic: {(nc_all['category']=='intergenic').sum():,}")
    print(f"L1 (PASS):        {len(l1_all):,} reads")
    print(f"  Intronic:   {(l1_all['l1_location']=='intronic').sum():,}")
    print(f"  Intergenic: {(l1_all['l1_location']=='intergenic').sum():,}")
    if len(ctrl_all) > 0:
        print(f"mRNA control:     {len(ctrl_all):,} reads")

    # =========================================================================
    # Comparison 1: Poly(A) shortening by category × condition
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 1: Poly(A) length (HeLa vs HeLa-Ars)")
    print(f"{'='*70}")

    # Build unified table for poly(A) comparison
    # Noncoding non-L1
    nc_polya = nc_all[['read_id', 'category', 'condition', 'polya', 'group']].copy()
    nc_polya['source'] = 'non-L1'
    nc_polya.rename(columns={'category': 'location'}, inplace=True)

    # L1
    l1_polya = l1_all[['read_id', 'l1_location', 'condition', 'polya_length', 'group', 'age']].copy()
    l1_polya.rename(columns={'l1_location': 'location', 'polya_length': 'polya'}, inplace=True)
    l1_polya['source'] = 'L1'
    l1_polya = l1_polya.dropna(subset=['polya'])

    # Compute group-level medians and deltas
    print(f"\n{'Category':<25} {'HeLa med':>10} {'Ars med':>10} {'Delta':>10} {'P-value':>12}")
    print("-" * 70)

    categories = [
        ('L1 intronic', l1_polya[(l1_polya['source']=='L1') & (l1_polya['location']=='intronic')]),
        ('L1 intergenic', l1_polya[(l1_polya['source']=='L1') & (l1_polya['location']=='intergenic')]),
        ('non-L1 intronic', nc_polya[nc_polya['location']=='intronic']),
        ('non-L1 intergenic', nc_polya[nc_polya['location']=='intergenic']),
    ]

    polya_results = {}
    for name, df in categories:
        hela_vals = df[df['condition'] == 'HeLa']['polya']
        ars_vals = df[df['condition'] == 'Ars']['polya']
        if len(hela_vals) > 0 and len(ars_vals) > 0:
            med_h = hela_vals.median()
            med_a = ars_vals.median()
            delta = med_a - med_h
            stat, pval = stats.mannwhitneyu(hela_vals, ars_vals, alternative='two-sided')
            print(f"{name:<25} {med_h:>10.1f} {med_a:>10.1f} {delta:>+10.1f} {pval:>12.2e}")
            polya_results[name] = {'hela': med_h, 'ars': med_a, 'delta': delta, 'p': pval,
                                   'n_hela': len(hela_vals), 'n_ars': len(ars_vals)}

    # =========================================================================
    # Comparison 2: m6A density
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 2: m6A/kb by category")
    print(f"{'='*70}")

    nc_m6a = nc_all[nc_all['m6a_per_kb'].notna()].copy()
    l1_m6a = l1_all[l1_all['m6a_per_kb'].notna()].copy()

    print(f"\n{'Category':<30} {'n':>8} {'median m6A/kb':>14} {'mean m6A/kb':>12}")
    print("-" * 70)

    m6a_categories = [
        ('L1 intronic', l1_m6a[l1_m6a['l1_location'] == 'intronic']['m6a_per_kb']),
        ('L1 intergenic', l1_m6a[l1_m6a['l1_location'] == 'intergenic']['m6a_per_kb']),
        ('non-L1 intronic', nc_m6a[nc_m6a['category'] == 'intronic']['m6a_per_kb']),
        ('non-L1 intergenic', nc_m6a[nc_m6a['category'] == 'intergenic']['m6a_per_kb']),
    ]

    if len(ctrl_all) > 0 and 'polya' in ctrl_all.columns:
        ctrl_m6a = ctrl_all[ctrl_all['m6a_per_kb'].notna()]['m6a_per_kb']
        m6a_categories.append(('mRNA ctrl', ctrl_m6a))

    for name, vals in m6a_categories:
        if len(vals) > 0:
            print(f"{name:<30} {len(vals):>8} {vals.median():>14.3f} {vals.mean():>12.3f}")

    # Statistical tests: L1 vs non-L1 within same location
    print("\nStatistical tests (L1 vs non-L1 within same genomic context):")
    for loc in ['intronic', 'intergenic']:
        l1_vals = l1_m6a[l1_m6a['l1_location'] == loc]['m6a_per_kb']
        nc_vals = nc_m6a[nc_m6a['category'] == loc]['m6a_per_kb']
        if len(l1_vals) > 0 and len(nc_vals) > 0:
            stat, pval = stats.mannwhitneyu(l1_vals, nc_vals, alternative='two-sided')
            fc = l1_vals.median() / nc_vals.median() if nc_vals.median() > 0 else np.inf
            print(f"  {loc}: L1 {l1_vals.median():.3f} vs non-L1 {nc_vals.median():.3f} "
                  f"= {fc:.2f}x, MWU P={pval:.2e}")

    # =========================================================================
    # Comparison 3: m6A-poly(A) correlation in non-L1 noncoding
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 3: m6A-poly(A) correlation")
    print(f"{'='*70}")

    for source_name, df in [('non-L1 noncoding', nc_all), ('L1', l1_all)]:
        if source_name == 'L1':
            m6a_col = 'm6a_per_kb'
            polya_col = 'polya_length'
            cond_col = 'condition'
        else:
            m6a_col = 'm6a_per_kb'
            polya_col = 'polya'
            cond_col = 'condition'

        for cond in ['HeLa', 'Ars']:
            sub = df[(df[cond_col] == cond) & df[m6a_col].notna() & df[polya_col].notna()]
            if len(sub) < 10:
                continue
            rho, pval = stats.spearmanr(sub[m6a_col], sub[polya_col])
            print(f"  {source_name} ({cond}): n={len(sub):,}, "
                  f"Spearman rho={rho:.3f}, P={pval:.2e}")

    # =========================================================================
    # Comparison 4: Group-level analysis (per-replicate consistency)
    # =========================================================================
    print(f"\n{'='*70}")
    print("Comparison 4: Per-replicate poly(A) medians")
    print(f"{'='*70}")

    for loc in ['intronic', 'intergenic']:
        print(f"\n  --- {loc} ---")
        # L1
        for group in ALL_GROUPS:
            sub = l1_polya[(l1_polya['group'] == group) & (l1_polya['location'] == loc)]
            if len(sub) > 0:
                print(f"  L1 {group:<15}: n={len(sub):>5}, median={sub['polya'].median():.1f}")
        # Non-L1
        for group in ALL_GROUPS:
            sub = nc_polya[(nc_polya['group'] == group) & (nc_polya['location'] == loc)]
            if len(sub) > 0:
                print(f"  NC {group:<15}: n={len(sub):>5}, median={sub['polya'].median():.1f}")

    # =========================================================================
    # Figures
    # =========================================================================
    print(f"\n{'='*70}")
    print("Generating figures...")
    print(f"{'='*70}")

    # --- Figure 1: Poly(A) comparison (4 categories × 2 conditions) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for idx, loc in enumerate(['intronic', 'intergenic']):
        ax = axes[idx]

        plot_data = []
        # L1
        for cond in ['HeLa', 'Ars']:
            sub = l1_polya[(l1_polya['location'] == loc) & (l1_polya['condition'] == cond)]
            for _, row in sub.iterrows():
                plot_data.append({'Source': 'L1', 'Condition': cond, 'Poly(A)': row['polya']})
        # Non-L1
        for cond in ['HeLa', 'Ars']:
            sub = nc_polya[(nc_polya['location'] == loc) & (nc_polya['condition'] == cond)]
            for _, row in sub.iterrows():
                plot_data.append({'Source': 'non-L1', 'Condition': cond, 'Poly(A)': row['polya']})

        pdf = pd.DataFrame(plot_data)
        if len(pdf) == 0:
            continue

        sns.violinplot(data=pdf, x='Source', y='Poly(A)', hue='Condition',
                       split=True, inner='quartile', ax=ax,
                       palette={'HeLa': '#4DBEEE', 'Ars': '#D95319'},
                       cut=0)
        ax.set_title(f'{loc.capitalize()}')
        ax.set_ylabel('Poly(A) length (nt)' if idx == 0 else '')
        ax.set_xlabel('')

        # Annotate delta
        for src_idx, src in enumerate(['L1', 'non-L1']):
            key = f'{src} {loc}'
            if key in polya_results:
                r = polya_results[key]
                sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
                ax.text(src_idx, ax.get_ylim()[1] * 0.95,
                        f"Δ={r['delta']:+.1f}\n{sig}",
                        ha='center', va='top', fontsize=7)

    fig.suptitle('Poly(A) length: L1 vs non-L1 noncoding (HeLa vs Arsenite)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig_noncoding_polya_comparison.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'fig_noncoding_polya_comparison.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("  Saved: fig_noncoding_polya_comparison.pdf")

    # --- Figure 2: m6A/kb comparison ---
    fig, ax = plt.subplots(figsize=(7, 5))

    m6a_plot_data = []
    for name, vals_series in m6a_categories:
        for v in vals_series:
            m6a_plot_data.append({'Category': name, 'm6A/kb': v})

    m6a_pdf = pd.DataFrame(m6a_plot_data)
    if len(m6a_pdf) > 0:
        order = [name for name, _ in m6a_categories]
        sns.boxplot(data=m6a_pdf, x='Category', y='m6A/kb', order=order,
                    showfliers=False, ax=ax,
                    palette=['#d62728', '#ff9896', '#1f77b4', '#aec7e8', '#2ca02c'][:len(order)])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_title('m6A density by genomic context')
        ax.set_ylabel('m6A sites / kb')
        ax.set_xlabel('')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig_noncoding_m6a_comparison.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'fig_noncoding_m6a_comparison.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("  Saved: fig_noncoding_m6a_comparison.pdf")

    # --- Figure 3: m6A-poly(A) scatter (non-L1 noncoding, stressed) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for idx, (src_name, df, m6a_col, polya_col) in enumerate([
        ('non-L1 noncoding (Ars)', nc_all[nc_all['condition'] == 'Ars'], 'm6a_per_kb', 'polya'),
        ('L1 (Ars)', l1_all[l1_all['condition'] == 'Ars'], 'm6a_per_kb', 'polya_length'),
    ]):
        ax = axes[idx]
        sub = df[df[m6a_col].notna() & df[polya_col].notna()].copy()
        if len(sub) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            continue

        ax.scatter(sub[m6a_col], sub[polya_col], alpha=0.15, s=5, c='#333333')

        rho, pval = stats.spearmanr(sub[m6a_col], sub[polya_col])
        ax.set_xlabel('m6A / kb')
        ax.set_ylabel('Poly(A) length (nt)')
        ax.set_title(f'{src_name}\n(n={len(sub):,}, ρ={rho:.3f}, P={pval:.1e})')
        ax.set_xlim(left=-0.5)
        ax.set_ylim(bottom=0)

    fig.suptitle('m6A–Poly(A) correlation under arsenite stress', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig_noncoding_m6a_polya_scatter.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'fig_noncoding_m6a_polya_scatter.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("  Saved: fig_noncoding_m6a_polya_scatter.pdf")

    # --- Figure 4: Delta poly(A) summary bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4))

    bar_names = []
    bar_deltas = []
    bar_colors = []
    bar_sigs = []

    color_map = {
        'L1 intronic': '#d62728', 'L1 intergenic': '#ff9896',
        'non-L1 intronic': '#1f77b4', 'non-L1 intergenic': '#aec7e8',
    }

    for name in ['L1 intronic', 'L1 intergenic', 'non-L1 intronic', 'non-L1 intergenic']:
        if name in polya_results:
            r = polya_results[name]
            bar_names.append(name)
            bar_deltas.append(r['delta'])
            bar_colors.append(color_map.get(name, '#999999'))
            sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
            bar_sigs.append(sig)

    if bar_names:
        x = np.arange(len(bar_names))
        bars = ax.bar(x, bar_deltas, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_names, rotation=30, ha='right')
        ax.set_ylabel('Δ Poly(A) (Ars − HeLa, nt)')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title('Arsenite-induced poly(A) change by genomic context')

        for i, (bar, sig) in enumerate(zip(bars, bar_sigs)):
            y = bar.get_height()
            offset = -2 if y < 0 else 2
            ax.text(bar.get_x() + bar.get_width() / 2, y + offset, sig,
                    ha='center', va='bottom' if y >= 0 else 'top', fontsize=8, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig_noncoding_delta_polya_bar.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'fig_noncoding_delta_polya_bar.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("  Saved: fig_noncoding_delta_polya_bar.pdf")

    # =========================================================================
    # Save results table
    # =========================================================================
    results_file = OUTDIR / 'noncoding_comparison_results.tsv'
    rows = []
    for name, r in polya_results.items():
        rows.append({
            'category': name,
            'n_HeLa': r['n_hela'],
            'n_Ars': r['n_ars'],
            'median_HeLa': r['hela'],
            'median_Ars': r['ars'],
            'delta_nt': r['delta'],
            'MWU_pvalue': r['p'],
        })
    if rows:
        pd.DataFrame(rows).to_csv(results_file, sep='\t', index=False)
        print(f"\n  Results table: {results_file}")

    # =========================================================================
    # Interpretation
    # =========================================================================
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    l1_intr = polya_results.get('L1 intronic', {})
    nc_intr = polya_results.get('non-L1 intronic', {})
    l1_inter = polya_results.get('L1 intergenic', {})
    nc_inter = polya_results.get('non-L1 intergenic', {})

    if l1_intr and nc_intr:
        if abs(nc_intr.get('delta', 0)) > 10 and nc_intr.get('p', 1) < 0.05:
            print("  Intronic non-L1 also shows significant poly(A) shortening →")
            print("  Arsenite effect may be a general non-coding RNA property, not L1-specific")
        elif abs(nc_intr.get('delta', 0)) < 10 or nc_intr.get('p', 1) >= 0.05:
            print("  Intronic non-L1 does NOT show significant poly(A) shortening →")
            print("  Arsenite effect is L1-SPECIFIC")

    if l1_inter and nc_inter:
        if abs(nc_inter.get('delta', 0)) > 10 and nc_inter.get('p', 1) < 0.05:
            print("  Intergenic non-L1 also shows significant poly(A) shortening")
        else:
            print("  Intergenic non-L1 does NOT show significant shortening → L1-specific")

    print("\nDone.")


if __name__ == '__main__':
    main()
