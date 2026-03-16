#!/usr/bin/env python3
"""Phase 4: Analyze Cat B (nc_exonic) L1 reads.

Compare nc_exonic (Cat B) vs intronic/intergenic (PASS) L1 reads across:
  1. Psi/kb: nc_exonic vs intronic vs intergenic
  2. m6A/kb: nc_exonic vs intronic vs intergenic
  3. Poly(A): 3-group medians + arsenite effect (HeLa vs HeLa-Ars)
  4. Young vs Ancient within nc_exonic
  5. Host gene annotation (which non-coding genes overlap Cat B reads)
  6. nc_exonic L1 psi/kb vs Control psi/kb

Prerequisites: Phases 1-3 completed (FAST5, nanopolish, MAFIA outputs exist)

Usage:
    conda run -n research python analyze_catB.py
"""

import os
import ast
import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from scipy import stats
from collections import Counter
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
RESULTS = PROJECT / 'results_group'
RESULTS_SAMPLE = PROJECT / 'results'
TOPIC_DIR = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
OUTDIR = TOPIC_DIR / 'output'
OUTDIR.mkdir(exist_ok=True)

PROB_THRESHOLD = 128  # 50% on 0-255 scale
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

CELL_LINES = {
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'Hct116': ['Hct116_3', 'Hct116_4'],
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2': ['HepG2_5', 'HepG2_6'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'MCF7-EV': ['MCF7-EV_1'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

CL_ORDER = ['A549', 'H9', 'Hct116', 'HeLa', 'HeLa-Ars', 'HepG2',
            'HEYA8', 'K562', 'MCF7', 'MCF7-EV', 'SHSY5Y']

CL_COLORS = {
    'A549': '#1f77b4', 'H9': '#ff7f0e', 'Hct116': '#2ca02c', 'HeLa': '#d62728',
    'HeLa-Ars': '#9467bd', 'HepG2': '#8c564b', 'HEYA8': '#e377c2', 'K562': '#7f7f7f',
    'MCF7': '#bcbd22', 'MCF7-EV': '#17becf', 'SHSY5Y': '#aec7e8',
}

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 200})


# =============================================================================
# Helper functions (same as part3_analysis.py)
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
        if '17802' in mod_type or 'A+a' in mod_type or 'A+m' in mod_type:
            mod_key = 'm6A'
        elif '21891' in mod_type or 'U+p' in mod_type or 'T+p' in mod_type:
            mod_key = 'psi'
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


def parse_bam_per_read(bam_path):
    records = []
    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam:
                mm = read.get_tag("MM") if read.has_tag("MM") else None
                ml = read.get_tag("ML") if read.has_tag("ML") else None
                mods = parse_mm_ml_tags(mm, ml)
                rl = read.query_length or read.infer_query_length() or 0
                if rl < 50:
                    continue
                m6a_high = [(p, prob) for p, prob in mods['m6A'] if prob >= PROB_THRESHOLD]
                psi_high = [(p, prob) for p, prob in mods['psi'] if prob >= PROB_THRESHOLD]
                records.append({
                    'read_id': read.query_name,
                    'read_length': rl,
                    'm6a_sites_high': len(m6a_high),
                    'psi_sites_high': len(psi_high),
                })
    except Exception as e:
        print(f"  WARNING: Failed to parse {bam_path}: {e}")
    return pd.DataFrame(records)


# =============================================================================
# 1. Load Cat B read metadata
# =============================================================================
print("=" * 60)
print("Loading Cat B read metadata...")

catB_all = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        detail_file = TOPIC_DIR / f'catB_reads_{g}.tsv'
        if not detail_file.exists():
            continue
        df = pd.read_csv(detail_file, sep='\t')
        df['group'] = g
        df['cell_line'] = cl
        catB_all.append(df)

if not catB_all:
    print("ERROR: No Cat B detail files found. Run prepare_catB_readids.py first.")
    raise SystemExit(1)

catB_meta = pd.concat(catB_all, ignore_index=True)
print(f"  {len(catB_meta):,} Cat B reads across {catB_meta['group'].nunique()} groups")

# =============================================================================
# 2. Load Cat B MAFIA data (Phase 3 output)
# =============================================================================
print("\nLoading Cat B MAFIA per-read data...")
catB_cache = TOPIC_DIR / '_catB_mafia_cache'
catB_cache.mkdir(exist_ok=True)

catB_mafia = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        cache_path = catB_cache / f'{g}_catB_per_read.tsv'
        if cache_path.exists():
            pr = pd.read_csv(cache_path, sep='\t')
        else:
            bam_path = RESULTS / g / 'j_catB' / 'mafia' / f'{g}.catB.mAFiA.reads.bam'
            if not bam_path.exists() or bam_path.stat().st_size == 0:
                print(f"  SKIP: {g} (no MAFIA BAM)")
                continue
            print(f"  Parsing {g} Cat B MAFIA BAM...")
            pr = parse_bam_per_read(bam_path)
            if len(pr) > 0:
                pr.to_csv(cache_path, sep='\t', index=False)

        if len(pr) == 0:
            continue

        # Merge with Cat B metadata
        meta = catB_meta[catB_meta['group'] == g][['read_id', 'subfamily', 'locus_id', 'age']].copy()
        merged = pr.merge(meta, on='read_id', how='inner')
        merged['group'] = g
        merged['cell_line'] = cl
        catB_mafia.append(merged)

if catB_mafia:
    catB_mod = pd.concat(catB_mafia, ignore_index=True)
    catB_mod['m6a_per_kb'] = catB_mod['m6a_sites_high'] / (catB_mod['read_length'] / 1000)
    catB_mod['psi_per_kb'] = catB_mod['psi_sites_high'] / (catB_mod['read_length'] / 1000)
    print(f"  {len(catB_mod):,} Cat B reads with MAFIA data")
else:
    catB_mod = pd.DataFrame()
    print("  WARNING: No Cat B MAFIA data found")

# =============================================================================
# 3. Load Cat B Poly(A) data (Phase 2 output)
# =============================================================================
print("\nLoading Cat B poly(A) data...")

catB_polya = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        polya_path = RESULTS / g / 'j_catB' / f'{g}.catB.nanopolish.polya.tsv.gz'
        if not polya_path.exists() or polya_path.stat().st_size == 0:
            continue
        try:
            pa = pd.read_csv(polya_path, sep='\t', compression='gzip')
        except Exception:
            continue
        if 'readname' in pa.columns:
            pa = pa.rename(columns={'readname': 'read_id'})
        if 'polya_length' not in pa.columns:
            continue
        pa = pa[pa['qc_tag'] == 'PASS'].copy()
        if len(pa) == 0:
            continue

        # Merge with Cat B metadata
        meta = catB_meta[catB_meta['group'] == g][['read_id', 'subfamily', 'locus_id', 'age']].copy()
        merged = pa[['read_id', 'polya_length']].merge(meta, on='read_id', how='inner')
        merged['group'] = g
        merged['cell_line'] = cl
        catB_polya.append(merged)

if catB_polya:
    catB_pa = pd.concat(catB_polya, ignore_index=True)
    print(f"  {len(catB_pa):,} Cat B reads with poly(A) data")
else:
    catB_pa = pd.DataFrame()
    print("  WARNING: No Cat B poly(A) data found")

# =============================================================================
# 4. Load existing L1 PASS data (intronic/intergenic) for comparison
# =============================================================================
print("\nLoading existing PASS L1 data for comparison...")

# Reuse part3 caches if available
l1_cache_dir = TOPIC_05 / 'part3_l1_per_read_cache'
l1_pass = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        # L1 summary for metadata
        sum_path = RESULTS / g / 'g_summary' / f'{g}_L1_summary.tsv'
        if not sum_path.exists():
            continue
        sm = pd.read_csv(sum_path, sep='\t')
        sm = sm[sm['qc_tag'] == 'PASS'].copy()
        if len(sm) == 0:
            continue

        # MAFIA per-read data
        cache_path = l1_cache_dir / f'{g}_l1_per_read.tsv'
        if cache_path.exists():
            pr = pd.read_csv(cache_path, sep='\t')
        else:
            bam_path = RESULTS / g / 'h_mafia' / f'{g}.mAFiA.reads.bam'
            if not bam_path.exists():
                continue
            print(f"  Parsing {g} L1 BAM...")
            pr = parse_bam_per_read(bam_path)
            if len(pr) > 0:
                pr.to_csv(cache_path, sep='\t', index=False)

        if len(pr) == 0:
            continue

        merged = pr.merge(
            sm[['read_id', 'gene_id', 'transcript_id', 'read_length',
                'TE_group', 'polya_length']],
            on='read_id', how='inner', suffixes=('_bam', '')
        )
        merged['group'] = g
        merged['cell_line'] = cl
        merged['l1_age'] = merged['gene_id'].apply(
            lambda x: 'young' if x in YOUNG else 'ancient'
        )
        # Classify as intronic or intergenic from TE_group
        merged['te_context'] = merged['TE_group'].fillna('unknown')
        l1_pass.append(merged)

if l1_pass:
    l1 = pd.concat(l1_pass, ignore_index=True)
    if 'read_length_bam' in l1.columns:
        l1['read_length'] = l1['read_length'].fillna(l1['read_length_bam'])
        l1.drop(columns=['read_length_bam'], inplace=True, errors='ignore')
    l1['m6a_per_kb'] = l1['m6a_sites_high'] / (l1['read_length'] / 1000)
    l1['psi_per_kb'] = l1['psi_sites_high'] / (l1['read_length'] / 1000)
    print(f"  {len(l1):,} PASS L1 reads")
else:
    l1 = pd.DataFrame()
    print("  WARNING: No PASS L1 data found")

# =============================================================================
# 5. Load Control data (from part3 caches)
# =============================================================================
print("\nLoading Control per-read data...")
ctrl_cache_dir = TOPIC_05 / 'part3_ctrl_per_read_cache'
ctrl_reads = []
for cl, groups in CELL_LINES.items():
    if cl == 'MCF7-EV':
        continue
    for g in groups:
        cache_path = ctrl_cache_dir / f'{g}_ctrl_per_read.tsv'
        if not cache_path.exists():
            continue
        df = pd.read_csv(cache_path, sep='\t')
        if len(df) == 0:
            continue
        df['group'] = g
        df['cell_line'] = cl
        ctrl_reads.append(df)

if ctrl_reads:
    ctrl = pd.concat(ctrl_reads, ignore_index=True)
    ctrl['m6a_per_kb'] = ctrl['m6a_sites_high'] / (ctrl['read_length'] / 1000)
    ctrl['psi_per_kb'] = ctrl['psi_sites_high'] / (ctrl['read_length'] / 1000)
    print(f"  {len(ctrl):,} Control reads")
else:
    ctrl = pd.DataFrame()
    print("  WARNING: No Control data found")

# =============================================================================
# Analysis Section 1: Modification density comparison (3-group)
# =============================================================================
print("\n" + "=" * 60)
print("Section 1: Modification density comparison")
print("  nc_exonic (Cat B) vs intronic vs intergenic")

if len(catB_mod) > 0 and len(l1) > 0:
    # Tag each dataset
    catB_mod_tagged = catB_mod.copy()
    catB_mod_tagged['context'] = 'nc_exonic'

    l1_intronic = l1[l1['te_context'] == 'intronic'].copy()
    l1_intronic['context'] = 'intronic'

    l1_intergenic = l1[l1['te_context'] == 'intergenic'].copy()
    l1_intergenic['context'] = 'intergenic'

    cols = ['read_id', 'read_length', 'm6a_sites_high', 'psi_sites_high',
            'm6a_per_kb', 'psi_per_kb', 'group', 'cell_line', 'context']
    for df_tag in [catB_mod_tagged, l1_intronic, l1_intergenic]:
        for c in cols:
            if c not in df_tag.columns:
                df_tag[c] = np.nan

    combined = pd.concat([
        catB_mod_tagged[cols], l1_intronic[cols], l1_intergenic[cols]
    ], ignore_index=True)

    # Overall statistics
    print("\n  Overall median modification density:")
    for ctx in ['nc_exonic', 'intronic', 'intergenic']:
        sub = combined[combined['context'] == ctx]
        print(f"    {ctx}: psi/kb={sub['psi_per_kb'].median():.2f}, "
              f"m6A/kb={sub['m6a_per_kb'].median():.2f}, n={len(sub):,}")

    # KW test across 3 groups
    groups_psi = [combined[combined['context'] == c]['psi_per_kb'].dropna()
                  for c in ['nc_exonic', 'intronic', 'intergenic']]
    groups_m6a = [combined[combined['context'] == c]['m6a_per_kb'].dropna()
                  for c in ['nc_exonic', 'intronic', 'intergenic']]

    if all(len(g) > 0 for g in groups_psi):
        kw_psi = stats.kruskal(*groups_psi)
        print(f"\n  KW test psi/kb: H={kw_psi.statistic:.1f}, p={kw_psi.pvalue:.2e}")
    if all(len(g) > 0 for g in groups_m6a):
        kw_m6a = stats.kruskal(*groups_m6a)
        print(f"  KW test m6A/kb: H={kw_m6a.statistic:.1f}, p={kw_m6a.pvalue:.2e}")

    # Pairwise MWU: nc_exonic vs intronic, nc_exonic vs intergenic
    for ref_ctx in ['intronic', 'intergenic']:
        nc = combined[combined['context'] == 'nc_exonic']
        ref = combined[combined['context'] == ref_ctx]
        for metric in ['psi_per_kb', 'm6a_per_kb']:
            a, b = nc[metric].dropna(), ref[metric].dropna()
            if len(a) > 10 and len(b) > 10:
                u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                print(f"  nc_exonic vs {ref_ctx} {metric}: "
                      f"median {a.median():.2f} vs {b.median():.2f}, "
                      f"MWU p={p:.2e}")

    # Per-cell-line breakdown
    print("\n  Per-cell-line nc_exonic psi/kb:")
    for cl in CL_ORDER:
        sub = catB_mod_tagged[catB_mod_tagged['cell_line'] == cl]
        if len(sub) >= 10:
            print(f"    {cl}: median={sub['psi_per_kb'].median():.2f}, "
                  f"mean={sub['psi_per_kb'].mean():.2f}, n={len(sub)}")

    # ── Figure 1: Boxplot comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (metric, label) in enumerate([('psi_per_kb', 'Psi sites/kb'),
                                          ('m6a_per_kb', 'm6A sites/kb')]):
        ax = axes[i]
        data = []
        labels = []
        for ctx in ['nc_exonic', 'intronic', 'intergenic']:
            vals = combined[combined['context'] == ctx][metric].dropna()
            data.append(vals)
            labels.append(f'{ctx}\n(n={len(vals):,})')
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.set_title(f'{label}: nc_exonic vs intronic vs intergenic')
    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig1_modification_3group.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Save TSV
    summary_rows = []
    for ctx in ['nc_exonic', 'intronic', 'intergenic']:
        sub = combined[combined['context'] == ctx]
        summary_rows.append({
            'context': ctx,
            'n_reads': len(sub),
            'psi_per_kb_median': sub['psi_per_kb'].median(),
            'psi_per_kb_mean': sub['psi_per_kb'].mean(),
            'm6a_per_kb_median': sub['m6a_per_kb'].median(),
            'm6a_per_kb_mean': sub['m6a_per_kb'].mean(),
        })
    pd.DataFrame(summary_rows).to_csv(
        OUTDIR / 'table1_modification_3group.tsv', sep='\t', index=False)

# =============================================================================
# Analysis Section 2: Poly(A) comparison
# =============================================================================
print("\n" + "=" * 60)
print("Section 2: Poly(A) tail comparison")

if len(catB_pa) > 0 and len(l1) > 0 and 'polya_length' in l1.columns:
    catB_pa_tagged = catB_pa.copy()
    catB_pa_tagged['context'] = 'nc_exonic'

    l1_intronic_pa = l1[l1['te_context'] == 'intronic'][
        ['read_id', 'polya_length', 'group', 'cell_line']].copy()
    l1_intronic_pa['context'] = 'intronic'

    l1_intergenic_pa = l1[l1['te_context'] == 'intergenic'][
        ['read_id', 'polya_length', 'group', 'cell_line']].copy()
    l1_intergenic_pa['context'] = 'intergenic'

    pa_cols = ['read_id', 'polya_length', 'group', 'cell_line', 'context']
    pa_combined = pd.concat([
        catB_pa_tagged[pa_cols],
        l1_intronic_pa[pa_cols],
        l1_intergenic_pa[pa_cols]
    ], ignore_index=True).dropna(subset=['polya_length'])

    print("\n  Overall poly(A) median:")
    for ctx in ['nc_exonic', 'intronic', 'intergenic']:
        sub = pa_combined[pa_combined['context'] == ctx]
        print(f"    {ctx}: median={sub['polya_length'].median():.1f}nt, "
              f"mean={sub['polya_length'].mean():.1f}nt, n={len(sub):,}")

    # KW test
    pa_groups = [pa_combined[pa_combined['context'] == c]['polya_length'].dropna()
                 for c in ['nc_exonic', 'intronic', 'intergenic']]
    if all(len(g) > 0 for g in pa_groups):
        kw = stats.kruskal(*pa_groups)
        print(f"  KW test: H={kw.statistic:.1f}, p={kw.pvalue:.2e}")

    # Arsenite effect: HeLa vs HeLa-Ars within nc_exonic
    hela_nc = catB_pa_tagged[catB_pa_tagged['cell_line'] == 'HeLa']['polya_length'].dropna()
    ars_nc = catB_pa_tagged[catB_pa_tagged['cell_line'] == 'HeLa-Ars']['polya_length'].dropna()
    if len(hela_nc) > 10 and len(ars_nc) > 10:
        u, p = stats.mannwhitneyu(hela_nc, ars_nc, alternative='two-sided')
        delta = ars_nc.median() - hela_nc.median()
        print(f"\n  Arsenite effect (nc_exonic): HeLa median={hela_nc.median():.1f}nt, "
              f"HeLa-Ars={ars_nc.median():.1f}nt, Δ={delta:+.1f}nt, p={p:.2e}")

    # ── Figure 2: Poly(A) comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: 3-group comparison
    ax = axes[0]
    data = [pa_combined[pa_combined['context'] == c]['polya_length'].dropna()
            for c in ['nc_exonic', 'intronic', 'intergenic']]
    labels = [f'{c}\n(n={len(d):,})' for c, d in zip(
        ['nc_exonic', 'intronic', 'intergenic'], data)]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Poly(A) length (nt)')
    ax.set_title('Poly(A): nc_exonic vs intronic vs intergenic')

    # Panel B: Arsenite effect
    ax = axes[1]
    if len(hela_nc) > 0 and len(ars_nc) > 0:
        bp = ax.boxplot([hela_nc, ars_nc],
                        labels=[f'HeLa\n(n={len(hela_nc)})',
                                f'HeLa-Ars\n(n={len(ars_nc)})'],
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#d62728')
        bp['boxes'][1].set_facecolor('#9467bd')
        for b in bp['boxes']:
            b.set_alpha(0.6)
    ax.set_ylabel('Poly(A) length (nt)')
    ax.set_title('Arsenite effect (nc_exonic L1)')

    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig2_polya_3group.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Save TSV
    pa_summary = []
    for ctx in ['nc_exonic', 'intronic', 'intergenic']:
        sub = pa_combined[pa_combined['context'] == ctx]
        pa_summary.append({
            'context': ctx,
            'n_reads': len(sub),
            'polya_median': sub['polya_length'].median(),
            'polya_mean': sub['polya_length'].mean(),
        })
    pd.DataFrame(pa_summary).to_csv(
        OUTDIR / 'table2_polya_3group.tsv', sep='\t', index=False)

# =============================================================================
# Analysis Section 3: Young vs Ancient within nc_exonic
# =============================================================================
print("\n" + "=" * 60)
print("Section 3: Young vs Ancient within nc_exonic")

if len(catB_mod) > 0:
    catB_mod['l1_age'] = catB_mod['age']  # Already has 'age' column from metadata

    young = catB_mod[catB_mod['l1_age'] == 'young']
    ancient = catB_mod[catB_mod['l1_age'] == 'ancient']

    print(f"  Young: n={len(young):,}, Ancient: n={len(ancient):,}")

    if len(young) >= 10 and len(ancient) >= 10:
        for metric in ['psi_per_kb', 'm6a_per_kb']:
            y_vals = young[metric].dropna()
            a_vals = ancient[metric].dropna()
            u, p = stats.mannwhitneyu(y_vals, a_vals, alternative='two-sided')
            print(f"  {metric}: young median={y_vals.median():.2f}, "
                  f"ancient median={a_vals.median():.2f}, MWU p={p:.2e}")

    # ── Figure 3: Young vs Ancient ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (metric, label) in enumerate([('psi_per_kb', 'Psi sites/kb'),
                                          ('m6a_per_kb', 'm6A sites/kb')]):
        ax = axes[i]
        data_y = young[metric].dropna()
        data_a = ancient[metric].dropna()
        bp = ax.boxplot([data_y, data_a],
                        labels=[f'Young\n(n={len(data_y):,})',
                                f'Ancient\n(n={len(data_a):,})'],
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][1].set_facecolor('#3498db')
        for b in bp['boxes']:
            b.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.set_title(f'nc_exonic: {label} by L1 age')
    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig3_young_vs_ancient.png', dpi=200, bbox_inches='tight')
    plt.close()

# =============================================================================
# Analysis Section 4: Host gene annotation
# =============================================================================
print("\n" + "=" * 60)
print("Section 4: Host gene annotation (non-coding genes)")

# Load the GTF to map Cat B loci to overlapping nc genes
# For now, summarize from catB_reads_detail which has locus_id info
print(f"  Total Cat B reads: {len(catB_meta):,}")
print(f"  Unique loci: {catB_meta['locus_id'].nunique()}")
print(f"  Unique subfamilies: {catB_meta['subfamily'].nunique()}")

# Subfamily distribution
sf_counts = catB_meta['subfamily'].value_counts().head(20)
print("\n  Top 20 subfamilies (Cat B):")
for sf, n in sf_counts.items():
    print(f"    {sf}: {n} reads ({100*n/len(catB_meta):.1f}%)")

# Age distribution
age_counts = catB_meta['age'].value_counts()
print(f"\n  Age distribution: {dict(age_counts)}")

# Save host gene table
catB_meta.groupby(['subfamily', 'age']).size().reset_index(name='n_reads').to_csv(
    OUTDIR / 'table4_subfamily_distribution.tsv', sep='\t', index=False)

# =============================================================================
# Analysis Section 5: nc_exonic vs Control
# =============================================================================
print("\n" + "=" * 60)
print("Section 5: nc_exonic L1 vs non-L1 Control")

if len(catB_mod) > 0 and len(ctrl) > 0:
    for metric in ['psi_per_kb', 'm6a_per_kb']:
        nc_vals = catB_mod[metric].dropna()
        ct_vals = ctrl[metric].dropna()
        if len(nc_vals) > 10 and len(ct_vals) > 10:
            u, p = stats.mannwhitneyu(nc_vals, ct_vals, alternative='two-sided')
            print(f"  {metric}: nc_exonic={nc_vals.median():.2f}, "
                  f"Control={ct_vals.median():.2f}, MWU p={p:.2e}")

    # ── Figure 4: nc_exonic L1 vs Control ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (metric, label) in enumerate([('psi_per_kb', 'Psi sites/kb'),
                                          ('m6a_per_kb', 'm6A sites/kb')]):
        ax = axes[i]
        nc_vals = catB_mod[metric].dropna()
        ct_vals = ctrl[metric].dropna()
        bp = ax.boxplot([nc_vals, ct_vals],
                        labels=[f'nc_exonic L1\n(n={len(nc_vals):,})',
                                f'Control\n(n={len(ct_vals):,})'],
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#ff7f0e')
        bp['boxes'][1].set_facecolor('#95a5a6')
        for b in bp['boxes']:
            b.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.set_title(f'{label}: nc_exonic L1 vs Control')
    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig4_nc_exonic_vs_control.png', dpi=200, bbox_inches='tight')
    plt.close()

# =============================================================================
# Analysis Section 6: Per-cell-line summary
# =============================================================================
print("\n" + "=" * 60)
print("Section 6: Per-cell-line summary")

if len(catB_mod) > 0:
    cl_summary = []
    for cl in CL_ORDER:
        sub = catB_mod[catB_mod['cell_line'] == cl]
        if len(sub) == 0:
            continue
        row = {
            'cell_line': cl,
            'n_reads': len(sub),
            'n_young': (sub['age'] == 'young').sum(),
            'n_ancient': (sub['age'] == 'ancient').sum(),
            'psi_per_kb_median': sub['psi_per_kb'].median(),
            'psi_per_kb_mean': sub['psi_per_kb'].mean(),
            'm6a_per_kb_median': sub['m6a_per_kb'].median(),
            'm6a_per_kb_mean': sub['m6a_per_kb'].mean(),
            'read_length_median': sub['read_length'].median(),
        }

        # Add poly(A) if available
        if len(catB_pa) > 0:
            pa_sub = catB_pa[catB_pa['cell_line'] == cl]
            row['polya_median'] = pa_sub['polya_length'].median() if len(pa_sub) > 0 else np.nan
            row['polya_n'] = len(pa_sub)

        cl_summary.append(row)

    cl_df = pd.DataFrame(cl_summary)
    cl_df.to_csv(OUTDIR / 'table6_per_cellline_summary.tsv', sep='\t', index=False)
    print(cl_df.to_string(index=False))

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Cat B reads (metadata): {len(catB_meta):,}")
print(f"  Cat B reads (MAFIA):    {len(catB_mod):,}")
print(f"  Cat B reads (poly(A)):  {len(catB_pa):,}")
print(f"  PASS L1 reads:          {len(l1):,}")
print(f"  Control reads:          {len(ctrl):,}")
print(f"\n  Output: {OUTDIR}")
print(f"  Figures: fig1-fig4")
print(f"  Tables: table1, table2, table4, table6")
