#!/usr/bin/env python3
"""
Part 4 Analysis: L1 Stress Response (revised).
4 figures + 4 TSVs in pdf_figures_part4/.

Story flow:
  1. L1-specific poly(A) shortening — the main finding
  2. Post-transcriptional mechanism — cross-CL validation
  3. Young L1 immunity — biological specificity
  4. m6A protection — modification-function connection
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPICDIR = PROJECT / 'analysis/01_exploration'
OUTDIR = TOPICDIR / 'topic_05_cellline/pdf_figures_part4'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# =========================================================================
# Load data — rebuild from Part3 cache + L1 summary (NOT state_classification.tsv)
# state_classification.tsv m6A is outdated (N+ only, topic_03 basis)
# Part3 cache has corrected m6A (N+/N-, chemodCode fixed)
# =========================================================================
print("Loading data from Part3 cache + L1 summary...")

CACHE_L1 = TOPICDIR / 'topic_05_cellline/part3_l1_per_read_cache'
HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

# Load Part3 cache (corrected m6A)
cache_dfs = []
for g in HELA_GROUPS:
    p = CACHE_L1 / f'{g}_l1_per_read.tsv'
    if p.exists():
        df = pd.read_csv(p, sep='\t')
        df['group'] = g
        cache_dfs.append(df)
cache_all = pd.concat(cache_dfs, ignore_index=True)
cache_all['m6a_per_kb'] = cache_all['m6a_sites_high'] / (cache_all['read_length'] / 1000)
cache_all['psi_per_kb'] = cache_all['psi_sites_high'] / (cache_all['read_length'] / 1000)
cache_all['has_m6a'] = cache_all['m6a_sites_high'] > 0

# Load L1 summaries (poly(A), age, genomic context)
summ_dfs = []
for g in HELA_GROUPS:
    p = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
    if p.exists():
        df = pd.read_csv(p, sep='\t')
        df = df[df['qc_tag'] == 'PASS']
        df['group'] = g
        summ_dfs.append(df[['read_id', 'polya_length', 'gene_id', 'TE_group', 'group']])
summ_all = pd.concat(summ_dfs, ignore_index=True)
summ_all['l1_age'] = summ_all['gene_id'].apply(
    lambda x: 'young' if x.split('_dup')[0] in YOUNG else 'ancient')

# Merge
l1_state = cache_all.merge(summ_all, on=['read_id', 'group'], how='inner')
l1_state = l1_state[l1_state['polya_length'] > 0].copy()
l1_state['condition'] = l1_state['group'].apply(lambda x: 'HeLa-Ars' if 'Ars' in x else 'HeLa')

# Load control (poly(A) only, no m6A needed for Figures 1-2)
ctrl_state = pd.read_csv(TOPICDIR / 'topic_04_state/ctrl_state_classification.tsv', sep='\t')
ctrl_state['condition'] = ctrl_state['group'].apply(lambda x: 'HeLa-Ars' if 'Ars' in x else 'HeLa')

print(f"  L1: {len(l1_state):,} reads (from Part3 cache + L1 summary)")
print(f"  Ctrl: {len(ctrl_state):,} reads")
print(f"  L1 m6a_per_kb median: {l1_state['m6a_per_kb'].median():.2f}")
print(f"  L1 m6a_sites_high median: {l1_state['m6a_sites_high'].median():.0f}")

# =========================================================================
# Figure 1: L1-Specific Poly(A) Shortening (3 panels)
# =========================================================================
print("\nFigure 1: L1-specific poly(A) shortening...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# --- 1A: Violin — L1 vs Control × HeLa vs HeLa-Ars ---
ax = axes[0]
groups_v = [
    ('L1\nHeLa', l1_state[l1_state['condition'] == 'HeLa']['polya_length'], '#C44E52'),
    ('L1\nHeLa-Ars', l1_state[l1_state['condition'] == 'HeLa-Ars']['polya_length'], '#e08080'),
    ('Ctrl\nHeLa', ctrl_state[ctrl_state['condition'] == 'HeLa']['polya_length'], '#4C72B0'),
    ('Ctrl\nHeLa-Ars', ctrl_state[ctrl_state['condition'] == 'HeLa-Ars']['polya_length'], '#8fadd0'),
]

data_v = [g[1].clip(0, 400).values for g in groups_v]
parts = ax.violinplot(data_v, positions=range(len(groups_v)),
                      showmedians=True, showextrema=False)
for i, body in enumerate(parts['bodies']):
    body.set_facecolor(groups_v[i][2])
    body.set_alpha(0.7)
parts['cmedians'].set_color('black')

for i, (label, vals, _) in enumerate(groups_v):
    med = np.median(vals)
    ax.text(i, med + 10, f'{med:.0f}', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(groups_v)))
ax.set_xticklabels([g[0] for g in groups_v], fontsize=10)
ax.set_ylabel('Poly(A) Length (nt)', fontsize=11)
ax.set_title('A. Poly(A) Distribution', fontsize=12, fontweight='bold')
ax.set_ylim(0, 350)

mw_l1 = stats.mannwhitneyu(
    l1_state[l1_state['condition'] == 'HeLa']['polya_length'],
    l1_state[l1_state['condition'] == 'HeLa-Ars']['polya_length'],
    alternative='two-sided')
mw_ctrl = stats.mannwhitneyu(
    ctrl_state[ctrl_state['condition'] == 'HeLa']['polya_length'],
    ctrl_state[ctrl_state['condition'] == 'HeLa-Ars']['polya_length'],
    alternative='two-sided')
ax.text(0.02, 0.98, f'L1: MW p={mw_l1.pvalue:.1e}\nCtrl: MW p={mw_ctrl.pvalue:.2f} (ns)',
        transform=ax.transAxes, va='top', fontsize=8, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# --- 1B: CDF comparison ---
ax = axes[1]
for label, vals, color, ls in [
    ('L1 HeLa', l1_state[l1_state['condition']=='HeLa']['polya_length'], '#C44E52', '-'),
    ('L1 HeLa-Ars', l1_state[l1_state['condition']=='HeLa-Ars']['polya_length'], '#C44E52', '--'),
    ('Ctrl HeLa', ctrl_state[ctrl_state['condition']=='HeLa']['polya_length'], '#4C72B0', '-'),
    ('Ctrl HeLa-Ars', ctrl_state[ctrl_state['condition']=='HeLa-Ars']['polya_length'], '#4C72B0', '--'),
]:
    sorted_vals = np.sort(vals.clip(0, 400))
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=1.5, label=label)

ax.set_xlabel('Poly(A) Length (nt)', fontsize=11)
ax.set_ylabel('Cumulative Fraction', fontsize=11)
ax.set_title('B. Cumulative Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(0, 350)

# --- 1C: Delta summary bar ---
ax = axes[2]
l1_hela_med = np.median(l1_state[l1_state['condition'] == 'HeLa']['polya_length'])
l1_ars_med = np.median(l1_state[l1_state['condition'] == 'HeLa-Ars']['polya_length'])
ctrl_hela_med = np.median(ctrl_state[ctrl_state['condition'] == 'HeLa']['polya_length'])
ctrl_ars_med = np.median(ctrl_state[ctrl_state['condition'] == 'HeLa-Ars']['polya_length'])

delta_l1 = l1_ars_med - l1_hela_med
delta_ctrl = ctrl_ars_med - ctrl_hela_med

bars = ax.bar(['L1', 'Control'], [delta_l1, delta_ctrl],
              color=['#C44E52', '#4C72B0'], edgecolor='black', linewidth=0.5, width=0.5)
for bar, val in zip(bars, [delta_l1, delta_ctrl]):
    ax.text(bar.get_x() + bar.get_width()/2,
            val - 2 if val < 0 else val + 1,
            f'{val:+.1f} nt', ha='center',
            va='top' if val < 0 else 'bottom',
            fontsize=11, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Median Poly(A) Change (nt)', fontsize=11)
ax.set_title('C. Arsenite Effect', fontsize=12, fontweight='bold')
ax.set_ylim(-45, 15)

# KS test for L1 distribution shift
ks_l1 = stats.ks_2samp(
    l1_state[l1_state['condition'] == 'HeLa']['polya_length'],
    l1_state[l1_state['condition'] == 'HeLa-Ars']['polya_length'])
ax.text(0.5, 0.02, f'L1: KS D={ks_l1.statistic:.3f}, p={ks_l1.pvalue:.1e}',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=8, style='italic')

plt.tight_layout()
fig.savefig(OUTDIR / 'fig1_polya_shortening.png', dpi=200, bbox_inches='tight')
plt.close()

# Save TSV
tsv1 = []
for src, df in [('L1', l1_state), ('Control', ctrl_state)]:
    for cond in ['HeLa', 'HeLa-Ars']:
        sub = df[df['condition'] == cond]
        tsv1.append({
            'source': src, 'condition': cond, 'n': len(sub),
            'median_polya': np.median(sub['polya_length']),
            'mean_polya': np.mean(sub['polya_length']),
            'q25': np.percentile(sub['polya_length'], 25),
            'q75': np.percentile(sub['polya_length'], 75),
        })
pd.DataFrame(tsv1).to_csv(OUTDIR / 'part4_polya_shortening.tsv', sep='\t', index=False)
print("  Figure 1 done.")

# =========================================================================
# Figure 2: Cross-Cell-Line Validation (2 panels)
# =========================================================================
print("\nFigure 2: Cross-CL validation...")

OTHER_GROUPS = {
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'Hct116': ['Hct116_3', 'Hct116_4'],
    'HepG2': ['HepG2_5', 'HepG2_6'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

def load_l1(groups):
    dfs = []
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        df['group'] = g
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined['l1_age'] = combined['gene_id'].apply(
        lambda x: 'young' if x in YOUNG else 'ancient')
    return combined

hela_df = load_l1(['HeLa_1', 'HeLa_2', 'HeLa_3'])
ars_df = load_l1(['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'])

hela_anc = hela_df[hela_df['l1_age'] == 'ancient']
ars_anc = ars_df[ars_df['l1_age'] == 'ancient']

hela_loci = set(hela_anc['transcript_id'].unique())
ars_loci_set = set(ars_anc['transcript_id'].unique())
ars_only = ars_loci_set - hela_loci

print(f"  Ars-only ancient loci: {len(ars_only)}")

ars_only_polya = ars_anc[ars_anc['transcript_id'].isin(ars_only)]['polya_length']
ars_only_median = np.median(ars_only_polya)

cross_cl_results = []
per_locus_data = []
other_polya_all = []

for cl_name, groups in OTHER_GROUPS.items():
    cl_df = load_l1(groups)
    if len(cl_df) == 0:
        continue
    cl_anc = cl_df[cl_df['l1_age'] == 'ancient']
    overlap = cl_anc[cl_anc['transcript_id'].isin(ars_only)]
    if len(overlap) == 0:
        continue
    cross_cl_results.append({
        'cell_line': cl_name,
        'n_loci': overlap['transcript_id'].nunique(),
        'n_reads': len(overlap),
        'median_polya': np.median(overlap['polya_length']),
    })
    other_polya_all.extend(overlap['polya_length'].tolist())

    cl_locus_polya = overlap.groupby('transcript_id')['polya_length'].median()
    ars_locus_polya = ars_anc[ars_anc['transcript_id'].isin(cl_locus_polya.index)].groupby('transcript_id')['polya_length'].median()
    common_loci = cl_locus_polya.index.intersection(ars_locus_polya.index)
    for loc in common_loci:
        per_locus_data.append({
            'cell_line': cl_name, 'locus': loc,
            'other_cl_polya': cl_locus_polya[loc],
            'hela_ars_polya': ars_locus_polya[loc]
        })

cross_cl_df = pd.DataFrame(cross_cl_results)
per_locus_df = pd.DataFrame(per_locus_data)
other_polya_med = np.median(other_polya_all) if other_polya_all else 0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 2A: Bar per CL ---
ax = axes[0]
all_cl = pd.concat([
    cross_cl_df[['cell_line', 'median_polya']],
    pd.DataFrame([{'cell_line': 'HeLa-Ars', 'median_polya': ars_only_median}])
], ignore_index=True).sort_values('median_polya', ascending=False)

bar_colors = ['#d62728' if cl == 'HeLa-Ars' else '#4C72B0' for cl in all_cl['cell_line']]
bars = ax.bar(range(len(all_cl)), all_cl['median_polya'].values, color=bar_colors,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(all_cl)))
ax.set_xticklabels(all_cl['cell_line'].values, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Median Poly(A) at Ars-only Loci (nt)', fontsize=10)
ax.set_title('A. Cross-CL Poly(A) at Ars-only Loci', fontsize=12, fontweight='bold')

for i, (_, row) in enumerate(all_cl.iterrows()):
    ax.text(i, row['median_polya'] + 3, f'{row["median_polya"]:.0f}', ha='center', fontsize=8)

if other_polya_all:
    mw_cross = stats.mannwhitneyu(ars_only_polya, other_polya_all, alternative='less')
    ax.text(0.02, 0.98,
            f'Ars-only loci:\nHeLa-Ars: {ars_only_median:.1f}nt\nOther CLs: {other_polya_med:.1f}nt\nMW p={mw_cross.pvalue:.1e}',
            transform=ax.transAxes, va='top', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# --- 2B: Paired scatter ---
ax = axes[1]
if len(per_locus_df) > 0:
    ax.scatter(per_locus_df['other_cl_polya'], per_locus_df['hela_ars_polya'],
               alpha=0.2, s=8, color='#4C72B0')
    lim = max(per_locus_df['other_cl_polya'].max(), per_locus_df['hela_ars_polya'].max())
    ax.plot([0, min(lim, 400)], [0, min(lim, 400)], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, min(lim, 400))
    ax.set_ylim(0, min(lim, 400))

    below = (per_locus_df['hela_ars_polya'] < per_locus_df['other_cl_polya']).mean()
    ax.text(0.98, 0.02, f'{below:.0%} below diagonal\n(HeLa-Ars shorter)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Poly(A) in Other Cell Lines (nt)', fontsize=11)
ax.set_ylabel('Poly(A) in HeLa-Ars (nt)', fontsize=11)
ax.set_title('B. Per-Locus Comparison', fontsize=12, fontweight='bold')

plt.tight_layout()
fig.savefig(OUTDIR / 'fig2_cross_cl_validation.png', dpi=200, bbox_inches='tight')
plt.close()

cross_cl_df.to_csv(OUTDIR / 'part4_cross_cl_validation.tsv', sep='\t', index=False)
print("  Figure 2 done.")

# =========================================================================
# Figure 3: Young vs Ancient Immunity (3 panels)
# =========================================================================
print("\nFigure 3: Young vs Ancient immunity...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# --- 3A: Box plot poly(A) by age × condition ---
ax = axes[0]
box_groups = [
    ('Ancient\nHeLa', l1_state[(l1_state['l1_age']=='ancient')&(l1_state['condition']=='HeLa')]['polya_length']),
    ('Ancient\nHeLa-Ars', l1_state[(l1_state['l1_age']=='ancient')&(l1_state['condition']=='HeLa-Ars')]['polya_length']),
    ('Young\nHeLa', l1_state[(l1_state['l1_age']=='young')&(l1_state['condition']=='HeLa')]['polya_length']),
    ('Young\nHeLa-Ars', l1_state[(l1_state['l1_age']=='young')&(l1_state['condition']=='HeLa-Ars')]['polya_length']),
]
box_colors = ['#C44E52', '#e08080', '#4C72B0', '#8fadd0']

bp = ax.boxplot([g[1].clip(0, 400).values for g in box_groups],
                positions=range(len(box_groups)), widths=0.6,
                patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median_line in bp['medians']:
    median_line.set_color('black')
    median_line.set_linewidth(2)

for i, (label, vals) in enumerate(box_groups):
    med = np.median(vals)
    ax.text(i, med + 10, f'{med:.0f}', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(box_groups)))
ax.set_xticklabels([g[0] for g in box_groups], fontsize=9)
ax.set_ylabel('Poly(A) Length (nt)', fontsize=11)
ax.set_title('A. Poly(A) by Age x Condition', fontsize=12, fontweight='bold')
ax.set_ylim(0, 350)

mw_anc = stats.mannwhitneyu(box_groups[0][1], box_groups[1][1], alternative='two-sided')
mw_yng = stats.mannwhitneyu(box_groups[2][1], box_groups[3][1], alternative='two-sided')
ax.text(0.02, 0.98, f'Ancient: p={mw_anc.pvalue:.1e}\nYoung: p={mw_yng.pvalue:.2f}',
        transform=ax.transAxes, va='top', fontsize=8, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# --- 3B: CDF overlay — Ancient HeLa vs HeLa-Ars ---
ax = axes[1]
for label, sub_df, color, ls in [
    ('Ancient HeLa', l1_state[(l1_state['l1_age']=='ancient')&(l1_state['condition']=='HeLa')], '#C44E52', '-'),
    ('Ancient HeLa-Ars', l1_state[(l1_state['l1_age']=='ancient')&(l1_state['condition']=='HeLa-Ars')], '#C44E52', '--'),
    ('Young HeLa', l1_state[(l1_state['l1_age']=='young')&(l1_state['condition']=='HeLa')], '#4C72B0', '-'),
    ('Young HeLa-Ars', l1_state[(l1_state['l1_age']=='young')&(l1_state['condition']=='HeLa-Ars')], '#4C72B0', '--'),
]:
    vals = np.sort(sub_df['polya_length'].clip(0, 400).values)
    cdf = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, cdf, color=color, linestyle=ls, linewidth=1.5, label=label)

ax.set_xlabel('Poly(A) Length (nt)', fontsize=11)
ax.set_ylabel('Cumulative Fraction', fontsize=11)
ax.set_title('B. CDF by Age', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.set_xlim(0, 350)

# --- 3C: m6A-matched immunity test ---
ax = axes[2]

# Young L1 has higher m6A — does controlling for m6A eliminate immunity?
# Compute m6A overlap range (10-90th percentile)
young_m6a = l1_state[l1_state['l1_age']=='young']['m6a_per_kb']
ancient_m6a = l1_state[l1_state['l1_age']=='ancient']['m6a_per_kb']
m6a_lo = max(young_m6a.quantile(0.1), ancient_m6a.quantile(0.1))
m6a_hi = min(young_m6a.quantile(0.9), ancient_m6a.quantile(0.9))

matched = l1_state[(l1_state['m6a_per_kb'] >= m6a_lo) & (l1_state['m6a_per_kb'] <= m6a_hi)]

# Compute deltas: all vs m6A-matched
anc_all_hela = l1_state[(l1_state['l1_age']=='ancient')&(l1_state['condition']=='HeLa')]['polya_length']
anc_all_ars = l1_state[(l1_state['l1_age']=='ancient')&(l1_state['condition']=='HeLa-Ars')]['polya_length']
yng_all_hela = l1_state[(l1_state['l1_age']=='young')&(l1_state['condition']=='HeLa')]['polya_length']
yng_all_ars = l1_state[(l1_state['l1_age']=='young')&(l1_state['condition']=='HeLa-Ars')]['polya_length']

anc_m_hela = matched[(matched['l1_age']=='ancient')&(matched['condition']=='HeLa')]['polya_length']
anc_m_ars = matched[(matched['l1_age']=='ancient')&(matched['condition']=='HeLa-Ars')]['polya_length']
yng_m_hela = matched[(matched['l1_age']=='young')&(matched['condition']=='HeLa')]['polya_length']
yng_m_ars = matched[(matched['l1_age']=='young')&(matched['condition']=='HeLa-Ars')]['polya_length']

delta_anc_all = np.median(anc_all_ars) - np.median(anc_all_hela)
delta_yng_all = np.median(yng_all_ars) - np.median(yng_all_hela)
delta_anc_m = np.median(anc_m_ars) - np.median(anc_m_hela)
delta_yng_m = np.median(yng_m_ars) - np.median(yng_m_hela)

# MW tests for matched
mw_anc_m = stats.mannwhitneyu(anc_m_hela, anc_m_ars, alternative='two-sided')
mw_yng_m = stats.mannwhitneyu(yng_m_hela, yng_m_ars, alternative='two-sided')

categories_3c = ['Ancient\n(all)', 'Ancient\n(m6A-matched)', 'Young\n(all)', 'Young\n(m6A-matched)']
deltas_3c = [delta_anc_all, delta_anc_m, delta_yng_all, delta_yng_m]
bar_colors_3c = ['#C44E52', '#e08080', '#4C72B0', '#8fadd0']

bars = ax.bar(range(4), deltas_3c, color=bar_colors_3c, edgecolor='black', linewidth=0.5, width=0.6)
for bar, val in zip(bars, deltas_3c):
    ax.text(bar.get_x() + bar.get_width()/2,
            val - 2 if val < 0 else val + 1,
            f'{val:+.1f}', ha='center',
            va='top' if val < 0 else 'bottom',
            fontsize=9, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(categories_3c, fontsize=8)
ax.set_ylabel('Median Poly(A) Change (nt)', fontsize=11)
ax.set_title('C. m6A-Matched Immunity Test', fontsize=12, fontweight='bold')
ax.set_ylim(-50, 15)

ax.text(0.02, 0.98,
        f'm6A range: [{m6a_lo:.1f}-{m6a_hi:.1f}]/kb\n'
        f'Matched Ancient: p={mw_anc_m.pvalue:.1e}\n'
        f'Matched Young: p={mw_yng_m.pvalue:.2f} (ns)',
        transform=ax.transAxes, va='top', fontsize=7, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig.savefig(OUTDIR / 'fig3_age_immunity.png', dpi=200, bbox_inches='tight')
plt.close()

# Save TSV
tsv3 = []
for age in ['ancient', 'young']:
    for cond in ['HeLa', 'HeLa-Ars']:
        sub = l1_state[(l1_state['l1_age'] == age) & (l1_state['condition'] == cond)]
        tsv3.append({
            'source': 'L1', 'l1_age': age, 'condition': cond, 'm6a_match': 'all',
            'n': len(sub),
            'median_polya': np.median(sub['polya_length']),
            'mean_polya': np.mean(sub['polya_length']),
        })
        sub_m = matched[(matched['l1_age'] == age) & (matched['condition'] == cond)]
        tsv3.append({
            'source': 'L1', 'l1_age': age, 'condition': cond, 'm6a_match': f'matched [{m6a_lo:.1f}-{m6a_hi:.1f}]',
            'n': len(sub_m),
            'median_polya': np.median(sub_m['polya_length']),
            'mean_polya': np.mean(sub_m['polya_length']),
        })
for cond in ['HeLa', 'HeLa-Ars']:
    sub = ctrl_state[ctrl_state['condition'] == cond]
    tsv3.append({
        'source': 'Control', 'l1_age': 'control', 'condition': cond, 'm6a_match': 'all',
        'n': len(sub),
        'median_polya': np.median(sub['polya_length']),
        'mean_polya': np.mean(sub['polya_length']),
    })
pd.DataFrame(tsv3).to_csv(OUTDIR / 'part4_age_immunity.tsv', sep='\t', index=False)
print("  Figure 3 done.")

# =========================================================================
# Figure 4: m6A Protection (3 panels)
# =========================================================================
print("\nFigure 4: m6A protection...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# Prepare regression data — continuous m6A/kb with age & context covariates
# TE_group already in l1_state from the merge above
reg_df = l1_state[['condition', 'polya_length', 'm6a_per_kb', 'read_length',
                    'l1_age', 'TE_group']].dropna(subset=['polya_length', 'm6a_per_kb']).copy()
reg_df['ars'] = (reg_df['condition'] == 'HeLa-Ars').astype(float)
reg_df['rdLen_z'] = (reg_df['read_length'] - reg_df['read_length'].mean()) / reg_df['read_length'].std()
reg_df['is_young'] = (reg_df['l1_age'] == 'young').astype(float)
reg_df['is_intergenic'] = reg_df['TE_group'].apply(
    lambda x: 1.0 if str(x).lower() == 'intergenic' else 0.0)
reg_df['ars_x_m6a'] = reg_df['ars'] * reg_df['m6a_per_kb']
reg_df['ars_x_young'] = reg_df['ars'] * reg_df['is_young']

# OLS: polyA ~ ars + m6a/kb + rdLen_z + is_young + is_intergenic + ars*m6a/kb + ars*young
X = reg_df[['ars', 'm6a_per_kb', 'rdLen_z', 'is_young', 'is_intergenic',
            'ars_x_m6a', 'ars_x_young']].values
X = np.column_stack([np.ones(len(X)), X])
y = reg_df['polya_length'].values
beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ beta
resid = y - y_hat
n, p = X.shape
mse = np.sum(resid**2) / (n - p)
cov_beta = mse * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov_beta))
t_stat = beta / se
p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-p))
coef_names = ['intercept', 'arsenite', 'm6A/kb', 'read_length_z', 'young',
              'intergenic', 'ars x m6A/kb', 'ars x young']

# --- 4A: Coefficient plot (show all non-intercept) ---
ax = axes[0]
plot_idx = list(range(1, len(coef_names)))  # all except intercept
plot_names = [coef_names[i] for i in plot_idx]
plot_beta = [beta[i] for i in plot_idx]
plot_se = [se[i] for i in plot_idx]
plot_pv = [p_vals[i] for i in plot_idx]

y_pos = np.arange(len(plot_idx))
colors_coef = ['#d6604d' if b > 0 else '#4393c3' for b in plot_beta]
ax.barh(y_pos, plot_beta, xerr=[1.96*s for s in plot_se], color=colors_coef,
        edgecolor='black', linewidth=0.5, capsize=3, height=0.55)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_names, fontsize=9)
ax.set_xlabel('Coefficient (nt)', fontsize=11)
ax.set_title('A. OLS Coefficients', fontsize=12, fontweight='bold')
for i, (b, pv) in enumerate(zip(plot_beta, plot_pv)):
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns'
    x_offset = max(b + 1.96*plot_se[i], 0) + 1
    ax.text(x_offset, i, f'{b:.1f} ({sig})', va='center', fontsize=7)

# --- 4B: Poly(A) residual by m6A quartile ---
ax = axes[1]
X_rdlen = np.column_stack([np.ones(len(reg_df)), reg_df['rdLen_z'].values])
beta_rdlen = np.linalg.lstsq(X_rdlen, reg_df['polya_length'].values, rcond=None)[0]
reg_df['polya_resid'] = reg_df['polya_length'] - X_rdlen @ beta_rdlen

reg_df['m6a_rank'] = reg_df['m6a_per_kb'].rank(method='first')
reg_df['m6a_q'] = pd.qcut(reg_df['m6a_rank'], 4, labels=['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)'])

q_means = reg_df.groupby(['m6a_q', 'condition'])['polya_resid'].mean().unstack()
x_q = np.arange(len(q_means))
width_q = 0.35
if 'HeLa' in q_means.columns:
    ax.bar(x_q - width_q/2, q_means['HeLa'], width_q, color='#C44E52', label='HeLa', alpha=0.8)
if 'HeLa-Ars' in q_means.columns:
    ax.bar(x_q + width_q/2, q_means['HeLa-Ars'], width_q, color='#4C72B0', label='HeLa-Ars', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x_q)
ax.set_xticklabels(q_means.index, fontsize=9)
ax.set_xlabel('m6A Density Quartile', fontsize=11)
ax.set_ylabel('Poly(A) Residual (nt)', fontsize=11)
ax.set_title('B. m6A Dose-Response', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# --- 4C: Per-CL correlation bar (stress-specificity) ---
ax = axes[2]

CACHE_DIR = TOPICDIR / 'topic_05_cellline/part3_l1_per_read_cache'
CL_GROUPS = {
    'A549': ['A549_4','A549_5','A549_6'],
    'H9': ['H9_2','H9_3','H9_4'],
    'Hct116': ['Hct116_3','Hct116_4'],
    'HeLa': ['HeLa_1','HeLa_2','HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1','HeLa-Ars_2','HeLa-Ars_3'],
    'HepG2': ['HepG2_5','HepG2_6'],
    'HEYA8': ['HEYA8_1','HEYA8_2','HEYA8_3'],
    'K562': ['K562_4','K562_5','K562_6'],
    'MCF7': ['MCF7_2','MCF7_3','MCF7_4'],
    'SHSY5Y': ['SHSY5Y_1','SHSY5Y_2','SHSY5Y_3'],
}

cl_corr_results = []
for cl, grps in CL_GROUPS.items():
    dfs_cl = []
    for g in grps:
        cache_path = CACHE_DIR / f'{g}_l1_per_read.tsv'
        sum_path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not cache_path.exists() or not sum_path.exists():
            continue
        mod = pd.read_csv(cache_path, sep='\t')[['read_id','read_length','m6a_sites_high']]
        mod['m6a_per_kb'] = mod['m6a_sites_high'] / mod['read_length'] * 1000
        summ = pd.read_csv(sum_path, sep='\t')
        summ = summ[summ['qc_tag']=='PASS'][['read_id','polya_length']]
        merged = mod.merge(summ, on='read_id', how='inner')
        dfs_cl.append(merged)
    if not dfs_cl:
        continue
    df_cl = pd.concat(dfs_cl, ignore_index=True)
    rdz = (df_cl['read_length'] - df_cl['read_length'].mean()) / df_cl['read_length'].std()
    X_rd = np.column_stack([np.ones(len(df_cl)), rdz.values])
    b_rd = np.linalg.lstsq(X_rd, df_cl['polya_length'].values, rcond=None)[0]
    df_cl['polya_resid'] = df_cl['polya_length'] - X_rd @ b_rd
    r_cl, p_cl = stats.pearsonr(df_cl['m6a_per_kb'], df_cl['polya_resid'])
    cl_corr_results.append({'cell_line': cl, 'r_resid': r_cl, 'p_resid': p_cl, 'n': len(df_cl)})

cl_corr_df = pd.DataFrame(cl_corr_results).sort_values('r_resid', ascending=True)

bar_colors_cl = ['#d62728' if cl == 'HeLa-Ars' else '#4C72B0' for cl in cl_corr_df['cell_line']]
y_cl = np.arange(len(cl_corr_df))
ax.barh(y_cl, cl_corr_df['r_resid'].values, color=bar_colors_cl,
        edgecolor='black', linewidth=0.5, height=0.7)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_yticks(y_cl)
ax.set_yticklabels(cl_corr_df['cell_line'].values, fontsize=9)
ax.set_xlabel('Pearson r (m6A/kb vs poly(A) residual)', fontsize=10)
ax.set_title('C. m6A-Poly(A) Correlation', fontsize=12, fontweight='bold')

for i, (_, row) in enumerate(cl_corr_df.iterrows()):
    sig = '***' if row['p_resid'] < 0.001 else '**' if row['p_resid'] < 0.01 else '*' if row['p_resid'] < 0.05 else ''
    x_pos = row['r_resid'] + (0.005 if row['r_resid'] >= 0 else -0.005)
    ax.text(x_pos, i, f'{row["r_resid"]:+.3f}{sig}', va='center', fontsize=8,
            ha='left' if row['r_resid'] >= 0 else 'right')

plt.tight_layout()
fig.savefig(OUTDIR / 'fig4_m6a_protection.png', dpi=200, bbox_inches='tight')
plt.close()

# Save TSV — combine OLS + per-CL
tsv4_ols = pd.DataFrame({
    'variable': coef_names, 'coefficient': beta, 'se': se,
    't_stat': t_stat, 'p_value': p_vals
})
tsv4_ols.to_csv(OUTDIR / 'part4_m6a_protection.tsv', sep='\t', index=False)
cl_corr_df.to_csv(OUTDIR / 'part4_m6a_crosscl_corr.tsv', sep='\t', index=False)
print("  Figure 4 done.")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "="*60)
print("All 4 figures and 4 TSVs saved to:")
print(f"  {OUTDIR}")
print("="*60)

# Pearson r for Ars condition
r_ars, p_ars = stats.pearsonr(
    reg_df[reg_df['condition']=='HeLa-Ars']['m6a_per_kb'],
    reg_df[reg_df['condition']=='HeLa-Ars']['polya_resid'])

print(f"\nKey statistics:")
print(f"  L1 poly(A): HeLa {l1_hela_med:.1f} -> HeLa-Ars {l1_ars_med:.1f} (delta={delta_l1:.1f})")
print(f"  Ctrl poly(A): HeLa {ctrl_hela_med:.1f} -> HeLa-Ars {ctrl_ars_med:.1f} (delta={delta_ctrl:.1f})")
print(f"  L1 MW p={mw_l1.pvalue:.2e}, Ctrl MW p={mw_ctrl.pvalue:.3f}")
print(f"  Ancient: MW p={mw_anc.pvalue:.2e}")
print(f"  Young: MW p={mw_yng.pvalue:.3f}")
# ars x m6A/kb is index 6 in the new model
print(f"  OLS ars*m6A/kb interaction: coef={beta[6]:.2f}, p={p_vals[6]:.1e}")
print(f"  OLS ars*young: coef={beta[7]:.1f}, p={p_vals[7]:.3f}")
print(f"  HeLa-Ars m6A-polyA: r={r_ars:.3f}, p={p_ars:.1e}")
print(f"  Ars-only loci: HeLa-Ars={ars_only_median:.1f}nt, Others={other_polya_med:.1f}nt")
print("\nDone!")
