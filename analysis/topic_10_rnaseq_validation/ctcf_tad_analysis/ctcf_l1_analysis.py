#!/usr/bin/env python3
"""
CTCF binding site x ancient L1 poly(A)/m6A analysis under stress.

Tests whether ancient L1 at CTCF binding sites show different
poly(A)/m6A behavior under arsenite stress compared to non-CTCF L1.

Uses HeLa-S3 CTCF ChIP-seq peaks from ENCODE (ENCFF796WRU).
ChromHMM annotation from E117 (HeLa-S3).

bedtools intersect pre-computed externally.
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -- Paths --
BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
OUTDIR = f'{BASE}/topic_10_rnaseq_validation/ctcf_tad_analysis'
CHROMHMM_TSV = f'{BASE}/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv'

# -- 1. Load L1 data --
print("Loading L1 data...")
df_all = pd.read_csv(CHROMHMM_TSV, sep='\t')
hela_norm = df_all[(df_all['cellline'] == 'HeLa') & (df_all['l1_age'] == 'ancient')].copy()
hela_norm['condition'] = 'normal'
hela_ars = df_all[(df_all['cellline'] == 'HeLa-Ars') & (df_all['l1_age'] == 'ancient')].copy()
hela_ars['condition'] = 'stress'
df = pd.concat([hela_norm, hela_ars], ignore_index=True)
print(f"  HeLa ancient L1: normal={len(hela_norm)}, stress={len(hela_ars)}, total={len(df)}")

# -- 2. Load pre-computed bedtools results --
ctcf_hits = pd.read_csv(f'{OUTDIR}/l1_ctcf_intersect.tsv', sep='\t',
                        header=None, names=['chr', 'start', 'end', 'read_id'])
ctcf_read_ids = set(ctcf_hits['read_id'])

prox_hits = pd.read_csv(f'{OUTDIR}/l1_ctcf_proximity.tsv', sep='\t',
                        header=None, names=['chr', 'start', 'end', 'read_id'])
prox_read_ids = set(prox_hits['read_id'])

print(f"  CTCF-overlapping: {len(ctcf_read_ids)}")
print(f"  CTCF-proximal (<2kb): {len(prox_read_ids)}")

# -- 3. Annotate --
df['ctcf_overlap'] = df['read_id'].isin(ctcf_read_ids)
df['ctcf_proximal'] = df['read_id'].isin(prox_read_ids)
df['ctcf_category'] = 'distal'
df.loc[df['ctcf_proximal'], 'ctcf_category'] = 'proximal'
df.loc[df['ctcf_overlap'], 'ctcf_category'] = 'overlap'

print("\nCTCF category distribution:")
print(df.groupby(['condition', 'ctcf_category']).size().unstack(fill_value=0))

# -- 4. Statistical comparisons --
DECAY_THRESHOLD = 30
results = []

for cat_col, cat_name in [('ctcf_overlap', 'CTCF-overlap'), ('ctcf_proximal', 'CTCF-proximal')]:
    for cond in ['normal', 'stress']:
        grp = df[df['condition'] == cond]
        g_yes = grp[grp[cat_col] == True]
        g_no = grp[grp[cat_col] == False]
        if len(g_yes) < 5:
            continue

        pa_stat, pa_p = stats.mannwhitneyu(g_yes['polya_length'], g_no['polya_length'], alternative='two-sided')
        m6a_stat, m6a_p = stats.mannwhitneyu(g_yes['m6a_per_kb'], g_no['m6a_per_kb'], alternative='two-sided')
        decay_yes = (g_yes['polya_length'] < DECAY_THRESHOLD).mean() * 100
        decay_no = (g_no['polya_length'] < DECAY_THRESHOLD).mean() * 100
        a = int((g_yes['polya_length'] < DECAY_THRESHOLD).sum())
        b = int((g_yes['polya_length'] >= DECAY_THRESHOLD).sum())
        c = int((g_no['polya_length'] < DECAY_THRESHOLD).sum())
        d = int((g_no['polya_length'] >= DECAY_THRESHOLD).sum())
        fisher_or, fisher_p = stats.fisher_exact([[a, b], [c, d]])

        results.append({
            'category': cat_name,
            'condition': cond,
            'n_ctcf': len(g_yes),
            'n_nonctcf': len(g_no),
            'polya_ctcf_median': g_yes['polya_length'].median(),
            'polya_nonctcf_median': g_no['polya_length'].median(),
            'polya_delta': g_yes['polya_length'].median() - g_no['polya_length'].median(),
            'polya_p': pa_p,
            'm6a_ctcf_median': g_yes['m6a_per_kb'].median(),
            'm6a_nonctcf_median': g_no['m6a_per_kb'].median(),
            'm6a_ratio': g_yes['m6a_per_kb'].median() / max(g_no['m6a_per_kb'].median(), 0.001),
            'm6a_p': m6a_p,
            'decay_pct_ctcf': decay_yes,
            'decay_pct_nonctcf': decay_no,
            'decay_fisher_or': fisher_or,
            'decay_fisher_p': fisher_p,
        })

res_df = pd.DataFrame(results)
print("\n=== COMPARISON TABLE ===")
for _, row in res_df.iterrows():
    print(f"\n{row['category']} | {row['condition']}:")
    print(f"  n: {row['n_ctcf']} vs {row['n_nonctcf']}")
    print(f"  poly(A) median: {row['polya_ctcf_median']:.1f} vs {row['polya_nonctcf_median']:.1f} (delta={row['polya_delta']:+.1f}, P={row['polya_p']:.2e})")
    print(f"  m6A/kb median: {row['m6a_ctcf_median']:.3f} vs {row['m6a_nonctcf_median']:.3f} (ratio={row['m6a_ratio']:.2f}x, P={row['m6a_p']:.2e})")
    print(f"  decay zone: {row['decay_pct_ctcf']:.1f}% vs {row['decay_pct_nonctcf']:.1f}% (OR={row['decay_fisher_or']:.2f}, P={row['decay_fisher_p']:.2e})")

# -- 5. Stress delta analysis --
print("\n=== STRESS DELTA ANALYSIS ===")
delta_results = []
for label, mask_col in [('CTCF-overlap', 'ctcf_overlap'), ('CTCF-proximal', 'ctcf_proximal')]:
    for val, name in [(True, label), (False, f'non-{label}')]:
        norm = df[(df['condition'] == 'normal') & (df[mask_col] == val)]['polya_length']
        ars = df[(df['condition'] == 'stress') & (df[mask_col] == val)]['polya_length']
        if len(norm) < 5 or len(ars) < 5:
            print(f"  {name}: insufficient data (norm={len(norm)}, ars={len(ars)})")
            continue
        delta = ars.median() - norm.median()
        stat, p = stats.mannwhitneyu(norm, ars, alternative='two-sided')
        print(f"  {name}: normal={norm.median():.1f}, stress={ars.median():.1f}, "
              f"delta={delta:+.1f}nt, P={p:.2e} (n_norm={len(norm)}, n_ars={len(ars)})")
        delta_results.append({'group': name, 'median_normal': norm.median(),
                              'median_stress': ars.median(), 'delta': delta, 'p': p,
                              'n_normal': len(norm), 'n_stress': len(ars)})

# By 3-way category
print("\n  By 3-way category:")
for cat in ['overlap', 'proximal', 'distal']:
    norm = df[(df['condition'] == 'normal') & (df['ctcf_category'] == cat)]['polya_length']
    ars = df[(df['condition'] == 'stress') & (df['ctcf_category'] == cat)]['polya_length']
    if len(norm) < 5 and len(ars) < 5:
        continue
    if len(norm) < 5 or len(ars) < 5:
        print(f"    {cat}: norm={len(norm)}, ars={len(ars)} (skipped)")
        continue
    delta = ars.median() - norm.median()
    stat, p = stats.mannwhitneyu(norm, ars, alternative='two-sided')
    print(f"    {cat}: normal={norm.median():.1f}, stress={ars.median():.1f}, "
          f"delta={delta:+.1f}nt, P={p:.2e} (n={len(norm)}+{len(ars)})")

# -- 6. ChromHMM distribution --
print("\n=== CHROMHMM STATE DISTRIBUTION ===")
for val, name in [(True, 'CTCF-overlap'), (False, 'non-CTCF')]:
    grp = df[df['ctcf_overlap'] == val]
    if len(grp) == 0:
        continue
    dist = grp['chromhmm_group'].value_counts(normalize=True) * 100
    print(f"\n{name} (n={len(grp)}):")
    for state, pct in dist.items():
        print(f"  {state}: {pct:.1f}%")

# -- 7. Interaction test --
print("\n=== INTERACTION TEST: CTCF x STRESS on poly(A) ===")
try:
    import statsmodels.api as sm
    for label, mask_col in [('CTCF-overlap', 'ctcf_overlap'), ('CTCF-proximal', 'ctcf_proximal')]:
        sub = df[['polya_length', 'condition', mask_col]].dropna()
        sub['stress_binary'] = (sub['condition'] == 'stress').astype(int)
        sub['ctcf_binary'] = sub[mask_col].astype(int)
        sub['interaction'] = sub['stress_binary'] * sub['ctcf_binary']
        X = sm.add_constant(sub[['stress_binary', 'ctcf_binary', 'interaction']])
        y = sub['polya_length']
        model = sm.OLS(y, X).fit()
        print(f"\n{label}:")
        print(f"  stress beta={model.params['stress_binary']:.2f}, P={model.pvalues['stress_binary']:.2e}")
        print(f"  ctcf beta={model.params['ctcf_binary']:.2f}, P={model.pvalues['ctcf_binary']:.2e}")
        print(f"  interaction beta={model.params['interaction']:.2f}, P={model.pvalues['interaction']:.2e}")
        print(f"  R^2={model.rsquared:.4f}")
except ImportError:
    print("  statsmodels not available")

# -- 8. m6A-poly(A) correlation by CTCF status (stress only) --
print("\n=== m6A-poly(A) CORRELATION (stress only) ===")
stress = df[df['condition'] == 'stress']
for val, name in [(True, 'CTCF-overlap'), (False, 'non-CTCF')]:
    sub = stress[stress['ctcf_overlap'] == val]
    if len(sub) < 10:
        print(f"  {name}: n={len(sub)}, insufficient")
        continue
    rho, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
    print(f"  {name}: n={len(sub)}, rho={rho:.3f}, P={p:.2e}")

# For proximal
for val, name in [(True, 'CTCF-proximal'), (False, 'non-proximal')]:
    sub = stress[stress['ctcf_proximal'] == val]
    if len(sub) < 10:
        continue
    rho, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
    print(f"  {name}: n={len(sub)}, rho={rho:.3f}, P={p:.2e}")

# -- 9. Save --
res_df.to_csv(f'{OUTDIR}/ctcf_l1_comparison.tsv', sep='\t', index=False)
df.to_csv(f'{OUTDIR}/hela_ancient_l1_ctcf_annotated.tsv', sep='\t', index=False)

# -- 10. Figure --
print("\nGenerating figure...")
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

ctcf_colors = {'overlap': '#1565C0', 'proximal': '#42A5F5', 'distal': '#BDBDBD'}
cond_hatches = {'normal': '', 'stress': '///'}

# (a) Poly(A) by CTCF category x condition
ax1 = fig.add_subplot(gs[0, 0])
data_list, labels, cols = [], [], []
for cond in ['normal', 'stress']:
    for cat in ['overlap', 'proximal', 'distal']:
        sub = df[(df['condition'] == cond) & (df['ctcf_category'] == cat)]
        if len(sub) == 0:
            data_list.append([0])
        else:
            data_list.append(sub['polya_length'].values)
        labels.append(f"{cat[:4]}\n{'N' if cond=='normal' else 'S'}")
        cols.append(ctcf_colors[cat])

bp = ax1.boxplot(data_list, widths=0.6, patch_artist=True,
                 showfliers=False, medianprops={'color': 'black', 'linewidth': 1.5})
for i, (patch, c) in enumerate(zip(bp['boxes'], cols)):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
    if i >= 3:  # stress
        patch.set_hatch('///')
ax1.set_xticklabels(labels, fontsize=7)
ax1.set_ylabel('Poly(A) length (nt)')
ax1.set_title('(a) Poly(A) by CTCF category', fontsize=10)

# (b) m6A/kb
ax2 = fig.add_subplot(gs[0, 1])
data_list2, labels2, cols2 = [], [], []
for cond in ['normal', 'stress']:
    for cat in ['overlap', 'proximal', 'distal']:
        sub = df[(df['condition'] == cond) & (df['ctcf_category'] == cat)]
        if len(sub) == 0:
            data_list2.append([0])
        else:
            data_list2.append(sub['m6a_per_kb'].values)
        labels2.append(f"{cat[:4]}\n{'N' if cond=='normal' else 'S'}")
        cols2.append(ctcf_colors[cat])

bp2 = ax2.boxplot(data_list2, widths=0.6, patch_artist=True,
                  showfliers=False, medianprops={'color': 'black', 'linewidth': 1.5})
for i, (patch, c) in enumerate(zip(bp2['boxes'], cols2)):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
    if i >= 3:
        patch.set_hatch('///')
ax2.set_xticklabels(labels2, fontsize=7)
ax2.set_ylabel('m6A/kb')
ax2.set_title('(b) m6A/kb by CTCF category', fontsize=10)

# (c) Decay zone
ax3 = fig.add_subplot(gs[0, 2])
cats_3way = ['overlap', 'proximal', 'distal']
decay_normal = []
decay_stress = []
for cat in cats_3way:
    sn = df[(df['condition'] == 'normal') & (df['ctcf_category'] == cat)]
    ss = df[(df['condition'] == 'stress') & (df['ctcf_category'] == cat)]
    decay_normal.append((sn['polya_length'] < DECAY_THRESHOLD).mean() * 100 if len(sn) > 0 else 0)
    decay_stress.append((ss['polya_length'] < DECAY_THRESHOLD).mean() * 100 if len(ss) > 0 else 0)

x = np.arange(3)
w = 0.35
bars1 = ax3.bar(x - w/2, decay_normal, w, label='Normal', color='#4CAF50', alpha=0.7)
bars2 = ax3.bar(x + w/2, decay_stress, w, label='Stress', color='#F44336', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(['Overlap', 'Proximal', 'Distal'], fontsize=8)
ax3.set_ylabel('Decay zone (<30nt) %')
ax3.set_title('(c) Decay zone proportion', fontsize=10)
ax3.legend(fontsize=8)
for bar_set, vals in [(bars1, decay_normal), (bars2, decay_stress)]:
    for bar, v in zip(bar_set, vals):
        if v > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=7)

# (d) ChromHMM stacked bars
ax4 = fig.add_subplot(gs[1, 0])
chromhmm_states = ['Promoter', 'Enhancer', 'Transcribed', 'Heterochromatin', 'Repressed', 'Quiescent']
state_colors_map = {'Promoter': '#FF0000', 'Enhancer': '#FFC107', 'Transcribed': '#4CAF50',
                    'Heterochromatin': '#8E24AA', 'Repressed': '#607D8B', 'Quiescent': '#BDBDBD'}
for idx, (val, name) in enumerate([(True, 'CTCF-\noverlap'), (False, 'non-\nCTCF')]):
    grp = df[df['ctcf_overlap'] == val]
    dist = grp['chromhmm_group'].value_counts(normalize=True) * 100
    bottom = 0
    for state in chromhmm_states:
        pct = dist.get(state, 0)
        ax4.bar(idx, pct, bottom=bottom, color=state_colors_map.get(state, '#999'),
                label=state if idx == 0 else '', edgecolor='white', linewidth=0.5)
        bottom += pct
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['CTCF-\noverlap', 'non-\nCTCF'], fontsize=9)
ax4.set_ylabel('% of reads')
ax4.set_title('(d) ChromHMM distribution', fontsize=10)
ax4.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')

# (e) Poly(A) delta bar by 3-way category
ax5 = fig.add_subplot(gs[1, 1])
deltas, delta_labels, delta_ns = [], [], []
for cat in ['overlap', 'proximal', 'distal']:
    norm_vals = df[(df['condition'] == 'normal') & (df['ctcf_category'] == cat)]['polya_length']
    ars_vals = df[(df['condition'] == 'stress') & (df['ctcf_category'] == cat)]['polya_length']
    if len(norm_vals) >= 3 and len(ars_vals) >= 3:
        d = ars_vals.median() - norm_vals.median()
        deltas.append(d)
        delta_labels.append(cat.capitalize())
        delta_ns.append(f'n={len(norm_vals)}+{len(ars_vals)}')

bars = ax5.bar(range(len(deltas)), deltas,
               color=[ctcf_colors.get(l.lower(), '#999') for l in delta_labels], alpha=0.8)
ax5.set_xticks(range(len(deltas)))
ax5.set_xticklabels([f'{l}\n{n}' for l, n in zip(delta_labels, delta_ns)], fontsize=8)
ax5.set_ylabel('Poly(A) delta (stress-normal, nt)')
ax5.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax5.set_title('(e) Stress-induced poly(A) change', fontsize=10)
for bar, d in zip(bars, deltas):
    yoff = 2 if d >= 0 else -4
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yoff,
             f'{d:+.1f}', ha='center', va='bottom' if d >= 0 else 'top',
             fontsize=10, fontweight='bold')

# (f) m6A-poly(A) scatter (stress)
ax6 = fig.add_subplot(gs[1, 2])
for val, name, color, marker in [(True, 'CTCF-overlap', '#1565C0', 'o'),
                                  (False, 'non-CTCF', '#BDBDBD', '.')]:
    sub = stress[stress['ctcf_overlap'] == val]
    if len(sub) < 5:
        continue
    rho, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
    ax6.scatter(sub['m6a_per_kb'], sub['polya_length'], alpha=0.4, s=15 if val else 5,
                color=color, marker=marker,
                label=f'{name} (n={len(sub)}, rho={rho:.3f})', zorder=3 if val else 1)
ax6.set_xlabel('m6A/kb')
ax6.set_ylabel('Poly(A) length (nt)')
ax6.set_title('(f) m6A-poly(A) (stress)', fontsize=10)
ax6.legend(fontsize=7)

plt.savefig(f'{OUTDIR}/ctcf_l1_analysis.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"Figure saved: {OUTDIR}/ctcf_l1_analysis.pdf")

# -- 11. Summary --
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
n_ctcf = df['ctcf_overlap'].sum()
n_prox = df['ctcf_proximal'].sum()
n_total = len(df)
print(f"Total HeLa ancient L1 reads: {n_total}")
print(f"CTCF-overlapping: {n_ctcf} ({n_ctcf/n_total*100:.1f}%)")
print(f"CTCF-proximal (<2kb): {n_prox} ({n_prox/n_total*100:.1f}%)")
print(f"Distal: {n_total - n_prox} ({(n_total-n_prox)/n_total*100:.1f}%)")
