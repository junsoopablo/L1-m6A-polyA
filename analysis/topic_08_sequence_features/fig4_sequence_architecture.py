#!/usr/bin/env python3
"""
Generate Fig 4 panels: L1 Sequence Architecture and Stress Vulnerability.

(a) Mutation sensitivity profile along L1 consensus
(b) Stress poly(A) delta by structural domain
(c) DRACH and CpG motif density by domain
(d) m6A quartile poly(A) in EN-containing stressed reads
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re

OUT_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures'
DATA_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_sequence_features'

# Style
plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2, 'ytick.major.size': 2,
})

# Colors
COL_SENS = '#D62728'
COL_RESI = '#2CA02C'
COL_DRACH = '#1F77B4'
COL_CPG = '#FF7F0E'
COL_SIG = '#D62728'
COL_M6A_Q = ['#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']

# Domain definitions (0-based for python)
DOMAINS = {
    "5'UTR": (0, 908, '#E8E8E8'),
    'ORF1': (908, 1990, '#AEC7E8'),
    'ORF2\nEN': (1990, 2708, '#FF9896'),
    'ORF2\nRT': (2708, 4149, '#FFBB78'),
    'ORF2\nC-rich': (4149, 5817, '#C5B0D5'),
    "3'UTR": (5817, 6059, '#C7C7C7'),
}


def add_panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', ha='left')


# ═══════════════════════════════════════════════════
# Panel A: Mutation sensitivity profile
# ═══════════════════════════════════════════════════
def panel_a():
    fig, ax = plt.subplots(figsize=(5.5, 1.8))

    pos_df = pd.read_csv(f'{DATA_DIR}/l1_perposition_mutation_rates.tsv', sep='\t')
    win_df = pd.read_csv(f'{DATA_DIR}/l1_mutation_sensitivity_windows.tsv', sep='\t')

    # Domain backgrounds
    for dname, (ds, de, col) in DOMAINS.items():
        ax.axvspan(ds, de, alpha=0.15, color=col, zorder=0)
        mid = (ds + de) / 2
        ax.text(mid, 0.073, dname, ha='center', va='top', fontsize=5.5,
                fontstyle='italic', color='#555555')

    # Shade low-coverage region
    ax.axvspan(0, 1990, alpha=0.03, color='gray', zorder=0)
    ax.text(950, 0.235, 'low coverage\n(DRS 3\' bias)', ha='center', fontsize=5,
            color='#999999', fontstyle='italic')

    # Plot smoothed mutation rates
    valid = pos_df['mut_smooth_sensitive'].notna() & pos_df['mut_smooth_resistant'].notna()
    ax.plot(pos_df.loc[valid, 'position'], pos_df.loc[valid, 'mut_smooth_resistant'],
            color=COL_RESI, linewidth=0.8, alpha=0.7, label='Resistant (long poly(A))')
    ax.plot(pos_df.loc[valid, 'position'], pos_df.loc[valid, 'mut_smooth_sensitive'],
            color=COL_SENS, linewidth=0.8, alpha=0.7, label='Sensitive (short poly(A))')

    # Highlight significant windows
    sig_wins = win_df[win_df['sig'] == True]
    for _, row in sig_wins.iterrows():
        ax.axvspan(row['cons_start'], row['cons_end'], alpha=0.12, color=COL_SIG, zorder=1)

    # Mutation rate difference as filled area
    ax2 = ax.twinx()
    diff = pos_df['diff_smooth']
    ax2.fill_between(pos_df['position'], 0, diff,
                      where=diff > 0, color=COL_SENS, alpha=0.25)
    ax2.fill_between(pos_df['position'], 0, diff,
                      where=diff < 0, color=COL_RESI, alpha=0.25)
    ax2.axhline(0, color='black', linewidth=0.3, linestyle='-')
    ax2.set_ylabel('Mutation rate\ndifference', fontsize=6)
    ax2.set_ylim(-0.06, 0.06)
    ax2.tick_params(axis='y', labelsize=5.5)

    ax.set_xlabel('L1HS consensus position (bp)')
    ax.set_ylabel('Mutation rate\n(50-bp smoothed)', fontsize=6)
    ax.set_xlim(0, 6059)
    ax.set_ylim(0, 0.25)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.legend(fontsize=5, loc='upper left', frameon=False)

    n_sig = len(sig_wins)
    ax.annotate(f'{n_sig} significant windows\n(Bonferroni P < 0.05)',
                xy=(3000, 0.22), fontsize=5.5, color=COL_SIG, ha='center',
                fontstyle='italic')

    add_panel_label(ax, 'a', x=-0.07, y=1.08)

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig4a_mutation_sensitivity.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved fig4a")


# ═══════════════════════════════════════════════════
# Panel B: Stress delta by structural domain
# ═══════════════════════════════════════════════════
def panel_b():
    groups = [
        ('Full-length\n(all domains)', -9, 0.25, 114, 'ns'),
        ('Has EN\ndomain', -6.1, 0.079, 324, 'ns'),
        ('No EN\ndomain', -35.4, 2.7e-18, 2248, '***'),
        ('ORF2-only\nfragments', -53, 1.5e-9, 172, '***'),
    ]

    fig, ax = plt.subplots(figsize=(2.5, 2.0))

    x = range(len(groups))
    deltas = [g[1] for g in groups]
    colors = [COL_RESI if abs(g[1]) < 15 else COL_SENS for g in groups]

    ax.bar(x, deltas, color=colors, edgecolor='black', linewidth=0.5, width=0.65)

    for i, (label, delta, p, n, sig) in enumerate(groups):
        y_offset = -3 if delta < 0 else 3
        if sig != 'ns':
            ax.text(i, delta + y_offset, sig, ha='center', va='top' if delta < 0 else 'bottom',
                    fontsize=7, fontweight='bold')
        else:
            ax.text(i, delta + y_offset, 'ns', ha='center', va='top' if delta < 0 else 'bottom',
                    fontsize=6, color='gray')
        ax.text(i, 5, f'n={n}', ha='center', va='bottom', fontsize=5, color='#666666')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.5)
    ax.set_ylabel(r'$\Delta$ poly(A) (nt)' + '\n(stress \u2212 unstress)')
    ax.set_ylim(-65, 15)

    add_panel_label(ax, 'b')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig4b_domain_stress_delta.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved fig4b")


# ═══════════════════════════════════════════════════
# Panel C: DRACH and CpG density by domain
# ═══════════════════════════════════════════════════
def panel_c():
    with open(f'{DATA_DIR}/L1HS_consensus.fa') as f:
        lines = f.readlines()
        consensus = ''.join(l.strip() for l in lines if not l.startswith('>'))

    domains_simple = {
        "5'UTR": (0, 908),
        'ORF1': (908, 1990),
        'ORF2\nEN': (1990, 2708),
        'ORF2\nRT': (2708, 4149),
        'ORF2\nC-rich': (4149, 5817),
        "3'UTR": (5817, len(consensus)),
    }

    drach_pattern = r'[AGT][AG]AC[ACT]'

    domain_names = list(domains_simple.keys())
    drach_density = []
    cpg_density = []

    for dname, (ds, de) in domains_simple.items():
        seq = consensus[ds:de]
        dlen_kb = len(seq) / 1000
        n_drach = len(re.findall(drach_pattern, seq))
        n_cpg = seq.count('CG')
        drach_density.append(n_drach / dlen_kb)
        cpg_density.append(n_cpg / dlen_kb)

    fig, ax = plt.subplots(figsize=(2.5, 2.0))

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

    # Annotate EN domain DRACH peak
    ax.annotate('1.38\u00d7\naverage', xy=(2, drach_density[2]),
                xytext=(3.2, drach_density[2]+3),
                fontsize=5, color=COL_DRACH, ha='center',
                arrowprops=dict(arrowstyle='->', color=COL_DRACH, lw=0.5))

    add_panel_label(ax, 'c')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig4c_motif_density.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved fig4c")


# ═══════════════════════════════════════════════════
# Panel D: m6A quartile poly(A) in EN-containing stressed reads
# ═══════════════════════════════════════════════════
def panel_d():
    CACHE_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
    YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

    # Load rmsk
    rmsk = pd.read_csv(f'{DATA_DIR}/hg38_L1_rmsk_consensus.tsv', sep='\t')
    def normalize(row):
        if row['strand'] == '+':
            return pd.Series({'cons_start': row['repStart'] + 1, 'cons_end': row['repEnd']})
        else:
            return pd.Series({'cons_start': row['repLeft'] + 1, 'cons_end': row['repEnd']})
    rmsk[['cons_start', 'cons_end']] = rmsk.apply(normalize, axis=1)
    rmsk['te_key'] = rmsk['genoName'] + ':' + rmsk['genoStart'].astype(str) + '-' + rmsk['genoEnd'].astype(str)
    rmsk_slim = rmsk[['te_key', 'cons_start', 'cons_end']].drop_duplicates('te_key')

    # Load reads
    dfs = []
    for grp in ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
        summ = pd.read_csv(f'/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group/{grp}/g_summary/{grp}_L1_summary.tsv', sep='\t')
        summ = summ[summ['qc_tag'] == 'PASS']
        cache = pd.read_csv(f'{CACHE_DIR}/{grp}_l1_per_read.tsv', sep='\t')
        cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)
        merged = summ.merge(cache[['read_id', 'm6a_sites_high', 'm6a_per_kb']], on='read_id', how='inner')
        dfs.append(merged)
    df = pd.concat(dfs, ignore_index=True)
    df['age'] = df['gene_id'].apply(lambda x: 'Young' if x in YOUNG else 'Ancient')
    df['te_key'] = df['te_chr'] + ':' + df['te_start'].astype(str) + '-' + df['te_end'].astype(str)
    df = df.merge(rmsk_slim, on='te_key', how='left')
    df = df[df['cons_start'].notna()].copy()
    df['cons_start'] = df['cons_start'].astype(int)
    df['cons_end'] = df['cons_end'].astype(int)

    # EN coverage
    def en_cov(cs, ce):
        return max(0, min(ce, 2708) - max(cs, 1991)) / (2708 - 1991 + 1)
    df['en_coverage'] = df.apply(lambda r: en_cov(r['cons_start'], r['cons_end']), axis=1)

    # Ancient stressed with EN domain
    anc_st_en = df[(df['age'] == 'Ancient') & (df['en_coverage'] > 0.1)].copy()
    print(f"  EN-containing ancient stressed reads: {len(anc_st_en)}")

    # m6A quartiles
    anc_st_en['m6a_q'] = pd.qcut(anc_st_en['m6a_per_kb'], 4,
                                   labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

    fig, ax = plt.subplots(figsize=(2.5, 2.0))

    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
    positions = range(4)

    # Collect data and mean m6A/kb per quartile
    data_by_q = []
    m6a_means = []
    for q in quartiles:
        sub = anc_st_en[anc_st_en['m6a_q'] == q]
        data_by_q.append(sub['polya_length'].values)
        m6a_means.append(sub['m6a_per_kb'].mean())

    # Violin plot
    parts = ax.violinplot(data_by_q, positions=positions, showmedians=False,
                           showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COL_M6A_Q[i])
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
        pc.set_alpha(0.8)

    # Medians and IQR
    for i, q in enumerate(quartiles):
        d = data_by_q[i]
        med = np.median(d)
        q25 = np.percentile(d, 25)
        q75 = np.percentile(d, 75)
        ax.plot([i], [med], 'o', color='white', markersize=4, zorder=5,
                markeredgecolor='black', markeredgewidth=0.5)
        ax.vlines(i, q25, q75, color='black', linewidth=1.5, zorder=4)

    # Median labels above violins
    for i, q in enumerate(quartiles):
        med = np.median(data_by_q[i])
        ax.text(i, max(data_by_q[i]) + 15, f'{med:.0f}', ha='center', fontsize=5.5,
                color=COL_M6A_Q[i] if i > 0 else '#888888')

    # Significance bracket
    y_top = max(max(d) for d in data_by_q) + 40
    ax.plot([0, 0, 3, 3], [y_top-10, y_top, y_top, y_top-10], 'k-', linewidth=0.5)
    ax.text(1.5, y_top + 5, r'$P = 5.1 \times 10^{-6}$', ha='center', fontsize=5)

    # x-axis: quartile + mean m6A/kb
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{q}\n({m6a_means[i]:.1f})' for i, q in enumerate(quartiles)],
                        fontsize=5.5)
    ax.set_xlabel('m6A/kb quartile (mean m6A/kb)')
    ax.set_ylabel('Poly(A) tail length (nt)')
    ax.set_title('EN-containing ancient L1\n(stressed)', fontsize=7)
    ax.set_ylim(-5, y_top + 30)

    add_panel_label(ax, 'd')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig4d_en_m6a_quartile.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved fig4d")


# ── Generate all panels ──
print("Generating Fig 4 panels...")
panel_a()
panel_b()
panel_c()
panel_d()
print("All panels done!")
