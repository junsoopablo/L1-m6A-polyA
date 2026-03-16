#!/usr/bin/env python3
"""
Fig 4: Sequence features of young L1 stress immunity.

(a) Young vs Ancient: immunity feature comparison
(b) Each feature independently confers immunity in ancient L1
(c) Composite immunity score → poly(A) gradient under stress
(d) DRACH/CpG motif architecture by domain
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

OUT_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures'
DATA_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_sequence_features'

plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2, 'ytick.major.size': 2,
})

COL_YOUNG = '#1F77B4'
COL_ANCIENT = '#D62728'
COL_IMMUNE = '#2CA02C'
COL_VULN = '#D62728'
COL_DRACH = '#1F77B4'
COL_CPG = '#FF7F0E'
SCORE_COLORS = ['#D62728', '#FF7F0E', '#FFD700', '#90EE90', '#2CA02C']


def add_panel_label(ax, label, x=-0.15, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', ha='left')


# ═══════════════════════════════════════════════════
# Panel A: Young vs Ancient feature comparison
# ═══════════════════════════════════════════════════
def panel_a():
    fig, ax = plt.subplots(figsize=(3.0, 2.2))

    features = ['m6A/kb', 'Full-\nlength', 'Has EN\ndomain', 'Reaches\n3\'UTR', 'Consensus\nspan (kb)']
    young_vals = [4.6, 45.7, 49.6, 97.1, 3.61]
    ancient_vals = [2.5, 2.3, 12.4, 61.5, 0.41]

    # Normalize to Young = 1.0 for comparison
    young_norm = [1.0] * 5
    ancient_norm = [a / y if y > 0 else 0 for a, y in zip(ancient_vals, young_vals)]

    x = np.arange(len(features))
    width = 0.35

    bars1 = ax.bar(x - width/2, young_norm, width, color=COL_YOUNG,
                    edgecolor='black', linewidth=0.3, label='Young L1')
    bars2 = ax.bar(x + width/2, ancient_norm, width, color=COL_ANCIENT,
                    edgecolor='black', linewidth=0.3, label='Ancient L1')

    # Add ratio labels
    ratios = [y/a if a > 0 else np.inf for y, a in zip(young_vals, ancient_vals)]
    for i, r in enumerate(ratios):
        ax.text(i, max(young_norm[i], ancient_norm[i]) + 0.08,
                f'{r:.1f}x', ha='center', fontsize=5.5, fontweight='bold', color='#333333')

    # Add actual values below bars
    for i in range(len(features)):
        ax.text(i - width/2, -0.08, f'{young_vals[i]:.1f}' if i != 1 else f'{young_vals[i]:.0f}%',
                ha='center', fontsize=4.5, color=COL_YOUNG, rotation=0)
        ax.text(i + width/2, -0.08, f'{ancient_vals[i]:.1f}' if i != 1 else f'{ancient_vals[i]:.0f}%',
                ha='center', fontsize=4.5, color=COL_ANCIENT, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=5.5)
    ax.set_ylabel('Relative to Young L1')
    ax.set_ylim(-0.15, 1.4)
    ax.axhline(1.0, color='gray', linewidth=0.3, linestyle='--')
    ax.legend(fontsize=5.5, loc='upper right', frameon=False)

    add_panel_label(ax, 'a')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig4a_immunity_features.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved fig4a")


# ═══════════════════════════════════════════════════
# Panel B: Each feature confers immunity
# ═══════════════════════════════════════════════════
def panel_b():
    fig, ax = plt.subplots(figsize=(3.0, 2.2))

    # Data from analysis
    features = ['Full-\nlength', 'Has EN\ndomain', 'High\nm6A', 'Long\nspan', 'Reaches\n3\'UTR']
    delta_with = [-30.0, -6.1, -12.6, 5.6, -24.4]
    delta_without = [-33.9, -35.4, -38.7, -34.7, -46.2]
    p_with = [0.884, 0.079, 0.619, 0.966, 4.33e-7]
    sig_with = ['ns', 'ns', 'ns', 'ns', '***']

    x = np.arange(len(features))
    width = 0.35

    bars1 = ax.bar(x - width/2, delta_with, width, color=COL_IMMUNE,
                    edgecolor='black', linewidth=0.3, label='With feature')
    bars2 = ax.bar(x + width/2, delta_without, width, color=COL_VULN,
                    edgecolor='black', linewidth=0.3, label='Without feature')

    # Significance markers for "with feature" bars
    for i in range(len(features)):
        y_pos = delta_with[i] - 4 if delta_with[i] < 0 else delta_with[i] + 2
        label = sig_with[i]
        color = '#666666' if label == 'ns' else 'black'
        fontweight = 'normal' if label == 'ns' else 'bold'
        ax.text(i - width/2, y_pos, label, ha='center', va='top',
                fontsize=5.5, color=color, fontweight=fontweight)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=5.5)
    ax.set_ylabel(r'$\Delta$ poly(A) (nt)')
    ax.set_ylim(-55, 15)
    ax.legend(fontsize=5.5, loc='upper right', frameon=False)
    ax.set_title('Ancient L1 under stress', fontsize=7, color='#555555')

    add_panel_label(ax, 'b')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig4b_feature_immunity.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved fig4b")


# ═══════════════════════════════════════════════════
# Panel C: Composite immunity score → poly(A)
# ═══════════════════════════════════════════════════
def panel_c():
    """Composite immunity score violin plot: stressed vs unstressed."""
    CACHE_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
    YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

    # Load data
    rmsk = pd.read_csv(f'{DATA_DIR}/hg38_L1_rmsk_consensus.tsv', sep='\t')
    def normalize(row):
        if row['strand'] == '+':
            return pd.Series({'cons_start': row['repStart'] + 1, 'cons_end': row['repEnd']})
        else:
            return pd.Series({'cons_start': row['repLeft'] + 1, 'cons_end': row['repEnd']})
    rmsk[['cons_start', 'cons_end']] = rmsk.apply(normalize, axis=1)
    rmsk['te_key'] = rmsk['genoName'] + ':' + rmsk['genoStart'].astype(str) + '-' + rmsk['genoEnd'].astype(str)
    rmsk_slim = rmsk[['te_key', 'cons_start', 'cons_end']].drop_duplicates('te_key')

    dfs = []
    for grp in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
        summ = pd.read_csv(f'/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group/{grp}/g_summary/{grp}_L1_summary.tsv', sep='\t')
        summ = summ[summ['qc_tag'] == 'PASS']
        cache = pd.read_csv(f'{CACHE_DIR}/{grp}_l1_per_read.tsv', sep='\t')
        cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)
        merged = summ.merge(cache[['read_id', 'm6a_sites_high', 'm6a_per_kb']], on='read_id', how='inner')
        merged['group'] = grp
        merged['is_stress'] = 1 if 'Ars' in grp else 0
        dfs.append(merged)
    df = pd.concat(dfs, ignore_index=True)
    df['age'] = df['gene_id'].apply(lambda x: 'Young' if x in YOUNG else 'Ancient')
    df['te_key'] = df['te_chr'] + ':' + df['te_start'].astype(str) + '-' + df['te_end'].astype(str)
    df = df.merge(rmsk_slim, on='te_key', how='left')
    df = df[df['cons_start'].notna()].copy()
    df['cons_start'] = df['cons_start'].astype(int)
    df['cons_end'] = df['cons_end'].astype(int)

    # Compute features
    young_m6a_median = df[df['age'] == 'Young']['m6a_per_kb'].median()

    def get_n_domains(cs, ce):
        domains = {'5UTR': (1, 908), 'ORF1': (909, 1990), 'ORF2': (1991, 5817), '3UTR': (5818, 6064)}
        return sum(1 for _, (ds, de) in domains.items() if max(0, min(ce, de) - max(cs, ds)) > 50)

    def en_cov(cs, ce):
        return max(0, min(ce, 2708) - max(cs, 1991)) / (2708 - 1991 + 1)

    anc = df[df['age'] == 'Ancient'].copy()
    anc['is_fl'] = anc.apply(lambda r: get_n_domains(r['cons_start'], r['cons_end']) == 4, axis=1)
    anc['has_en'] = anc.apply(lambda r: en_cov(r['cons_start'], r['cons_end']) > 0.1, axis=1)
    anc['high_m6a'] = anc['m6a_per_kb'] >= young_m6a_median
    anc['reaches_3utr'] = anc['cons_end'] >= 5818

    # 3-component score: full-length, EN domain, high m6A
    # 3'UTR coverage excluded (does not independently protect)
    anc['immunity_score'] = (
        anc['is_fl'].astype(int) +
        anc['has_en'].astype(int) +
        anc['high_m6a'].astype(int)
    )

    fig, ax = plt.subplots(figsize=(3.0, 2.2))

    # Stressed only
    stressed = anc[anc['is_stress'] == 1].copy()

    score_labels = ['0', '1', '2', '3']
    score_vals = [0, 1, 2, 3]
    positions = range(len(score_labels))

    data_by_score = [stressed[stressed['immunity_score'] == s]['polya_length'].values for s in score_vals]

    # Violin
    parts = ax.violinplot(data_by_score, positions=positions, showmedians=False,
                           showextrema=False, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(SCORE_COLORS[i])
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
        pc.set_alpha(0.8)

    # Medians
    for i in range(len(score_vals)):
        d = data_by_score[i]
        if len(d) > 0:
            med = np.median(d)
            q25, q75 = np.percentile(d, [25, 75])
            ax.plot([i], [med], 'o', color='white', markersize=4, zorder=5,
                    markeredgecolor='black', markeredgewidth=0.5)
            ax.vlines(i, q25, q75, color='black', linewidth=1.5, zorder=4)
            # Label
            ax.text(i, max(d) + 15, f'{med:.0f}', ha='center', fontsize=6,
                    fontweight='bold', color=SCORE_COLORS[i])
            ax.text(i, -25, f'n={len(d)}', ha='center', fontsize=5, color='#666666')

    # Young L1 reference line
    young_st = df[(df['age'] == 'Young') & (df['is_stress'] == 1)]['polya_length']
    ax.axhline(young_st.median(), color=COL_YOUNG, linewidth=1, linestyle='--', alpha=0.7)
    ax.text(3.5, young_st.median() + 5, f'Young L1\n({young_st.median():.0f} nt)',
            fontsize=5, color=COL_YOUNG, va='bottom', ha='center')

    # Spearman annotation
    rho, p = stats.spearmanr(stressed['immunity_score'], stressed['polya_length'])
    ax.text(0.98, 0.02, f'Spearman $\\rho$ = {rho:.3f}\n$P$ = {p:.1e}',
            transform=ax.transAxes, fontsize=5.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    ax.set_xticks(positions)
    ax.set_xticklabels(score_labels, fontsize=6)
    ax.set_xlabel('Immunity score\n(# young-like features)')
    ax.set_ylabel('Poly(A) tail length (nt)')
    ax.set_title('Ancient L1 (stressed)', fontsize=7, color='#555555')
    ax.set_ylim(-40, max(max(d) for d in data_by_score if len(d) > 0) + 40)

    add_panel_label(ax, 'c')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig4c_immunity_score.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved fig4c")


# ═══════════════════════════════════════════════════
# Panel D: DRACH/CpG motif landscape
# ═══════════════════════════════════════════════════
def panel_d():
    with open(f'{DATA_DIR}/L1HS_consensus.fa') as f:
        lines = f.readlines()
        consensus = ''.join(l.strip() for l in lines if not l.startswith('>'))

    domains = {
        "5'UTR": (0, 908),
        'ORF1': (908, 1990),
        'ORF2\nEN': (1990, 2708),
        'ORF2\nRT': (2708, 4149),
        'ORF2\nC-rich': (4149, 5817),
        "3'UTR": (5817, len(consensus)),
    }

    drach_pattern = r'[AGT][AG]AC[ACT]'
    domain_names = list(domains.keys())
    drach_density = []
    cpg_density = []

    for dname, (ds, de) in domains.items():
        seq = consensus[ds:de]
        dlen_kb = len(seq) / 1000
        drach_density.append(len(re.findall(drach_pattern, seq)) / dlen_kb)
        cpg_density.append(seq.count('CG') / dlen_kb)

    fig, ax = plt.subplots(figsize=(2.5, 2.2))

    x_pos = np.arange(len(domain_names))
    width = 0.35

    ax.bar(x_pos - width/2, drach_density, width, color=COL_DRACH,
           edgecolor='black', linewidth=0.3, label='DRACH (m6A)')
    ax.bar(x_pos + width/2, cpg_density, width, color=COL_CPG,
           edgecolor='black', linewidth=0.3, label='CpG (ZAP)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(domain_names, fontsize=5.5, rotation=30, ha='right')
    ax.set_ylabel('Motif density (/kb)')
    ax.legend(fontsize=5.5, loc='upper right', frameon=False)

    # Highlight EN: high DRACH, low CpG
    ax.annotate('High m6A\nLow ZAP', xy=(2, drach_density[2]),
                xytext=(3.3, drach_density[2]+2),
                fontsize=5, color='#333333', ha='center', fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    add_panel_label(ax, 'd')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig4d_motif_landscape.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved fig4d")


# ── Generate all panels ──
print("Generating Fig 4 panels (immunity framing)...")
panel_a()
panel_b()
panel_c()
panel_d()
print("All panels done!")
