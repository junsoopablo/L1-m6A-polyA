#!/usr/bin/env python3
"""
Generate figures for LAD and Replication Timing analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
OUT_DIR = f'{BASE_DIR}/topic_10_rnaseq_validation/lad_replication_analysis'

# Load data
l1 = pd.read_csv(f'{BASE_DIR}/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv', sep='\t')
anno = pd.read_csv(f'{OUT_DIR}/l1_lad_repli_annotations.tsv', sep='\t')
l1 = l1.merge(anno, on='read_id', how='left')

# HeLa normal + stress
hela_n = l1[l1['cellline'] == 'HeLa'].copy()
hela_s = l1[l1['cellline'] == 'HeLa-Ars'].copy()

plt.rcParams.update({'font.size': 9, 'font.family': 'Arial'})

# === Figure 1: LAD analysis (4 panels) ===
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# Panel A: ChromHMM distribution in LAD vs non-LAD
ax1 = fig.add_subplot(gs[0, 0])
states = ['Quiescent', 'Transcribed', 'Enhancer', 'Promoter', 'Heterochromatin']
lad_dist = hela_n[hela_n['in_lad']==True]['chromhmm_group'].value_counts(normalize=True) * 100
nonlad_dist = hela_n[hela_n['in_lad']==False]['chromhmm_group'].value_counts(normalize=True) * 100
x = np.arange(len(states))
w = 0.35
bars1 = ax1.bar(x - w/2, [lad_dist.get(s, 0) for s in states], w, label='LAD', color='#d62728', alpha=0.8)
bars2 = ax1.bar(x + w/2, [nonlad_dist.get(s, 0) for s in states], w, label='non-LAD', color='#1f77b4', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(states, rotation=30, ha='right')
ax1.set_ylabel('Percentage (%)')
ax1.set_title('a) ChromHMM States: LAD vs non-LAD\n(HeLa, P=1.7e-30)')
ax1.legend()
ax1.set_ylim(0, 80)

# Panel B: Poly(A) violin LAD vs non-LAD (normal + stress)
ax2 = fig.add_subplot(gs[0, 1])
data_groups = [
    hela_n[hela_n['in_lad']==True]['polya_length'],
    hela_n[hela_n['in_lad']==False]['polya_length'],
    hela_s[hela_s['in_lad']==True]['polya_length'],
    hela_s[hela_s['in_lad']==False]['polya_length'],
]
labels_g = ['LAD\nNormal', 'non-LAD\nNormal', 'LAD\nStress', 'non-LAD\nStress']
colors_g = ['#d62728', '#1f77b4', '#ff7f7f', '#7fb4ff']

parts = ax2.violinplot([d.values for d in data_groups], positions=range(4), showmedians=True, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_g[i])
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')

for i, d in enumerate(data_groups):
    ax2.text(i, 5, f'n={len(d)}\nmed={d.median():.0f}', ha='center', va='bottom', fontsize=7)

ax2.set_xticks(range(4))
ax2.set_xticklabels(labels_g, fontsize=8)
ax2.set_ylabel('Poly(A) length (nt)')
ax2.set_title('b) Poly(A) Length: LAD vs non-LAD')
ax2.set_ylim(0, 350)

# Panel C: m6A/kb by LAD status
ax3 = fig.add_subplot(gs[1, 0])
# Cross-cell-line
cl_data = []
for cl in sorted(l1[l1['condition']=='normal']['cellline'].unique()):
    cl_sub = l1[(l1['cellline']==cl) & (l1['condition']=='normal')]
    lad_m = cl_sub[cl_sub['in_lad']==True]['m6a_per_kb'].median()
    nonlad_m = cl_sub[cl_sub['in_lad']==False]['m6a_per_kb'].median()
    n_lad = (cl_sub['in_lad']==True).sum()
    if n_lad >= 10:
        cl_data.append({'cl': cl, 'LAD': lad_m, 'non-LAD': nonlad_m})

cl_df = pd.DataFrame(cl_data)
x = np.arange(len(cl_df))
w = 0.35
ax3.bar(x - w/2, cl_df['LAD'], w, label='LAD', color='#d62728', alpha=0.8)
ax3.bar(x + w/2, cl_df['non-LAD'], w, label='non-LAD', color='#1f77b4', alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(cl_df['cl'], rotation=45, ha='right', fontsize=7)
ax3.set_ylabel('m6A/kb')
ax3.set_title('c) m6A/kb: LAD vs non-LAD (per cell line)')
ax3.legend(fontsize=8)

# Panel D: Stress delta by LAD × age
ax4 = fig.add_subplot(gs[1, 1])
categories = []
deltas = []
for lad, lad_label in [(True, 'LAD'), (False, 'non-LAD')]:
    for age in ['young', 'ancient']:
        norm = hela_n[(hela_n['in_lad']==lad) & (hela_n['l1_age']==age)]['polya_length']
        strs = hela_s[(hela_s['in_lad']==lad) & (hela_s['l1_age']==age)]['polya_length']
        if len(norm) >= 10 and len(strs) >= 10:
            delta = strs.median() - norm.median()
            _, p = stats.mannwhitneyu(strs, norm, alternative='two-sided')
            categories.append(f'{lad_label}\n{age.capitalize()}')
            deltas.append(delta)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ax4.text(len(categories)-1, delta + (2 if delta < 0 else -2), sig, ha='center', fontsize=8)

colors_d = ['#ff7f7f', '#d62728', '#7fb4ff', '#1f77b4']
bars = ax4.bar(range(len(categories)), deltas, color=colors_d[:len(categories)], alpha=0.8)
ax4.set_xticks(range(len(categories)))
ax4.set_xticklabels(categories, fontsize=8)
ax4.set_ylabel('Poly(A) Delta (nt)')
ax4.set_title('d) Stress Poly(A) Change: LAD × Age')
ax4.axhline(0, color='black', linewidth=0.5, linestyle='--')

plt.savefig(f'{OUT_DIR}/fig_lad_analysis.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig_lad_analysis.pdf")

# === Figure 2: Replication Timing analysis (4 panels) ===
fig2 = plt.figure(figsize=(12, 10))
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.3)

# Panel A: Poly(A) by replication timing (HeLa normal + stress)
ax1 = fig2.add_subplot(gs2[0, 0])
for i, (timing, color) in enumerate([('early', '#2ca02c'), ('mid', '#ff7f0e'), ('late', '#9467bd')]):
    norm = hela_n[hela_n['repli_timing']==timing]['polya_length']
    strs = hela_s[hela_s['repli_timing']==timing]['polya_length']
    if len(norm) >= 5:
        bp = ax1.boxplot([norm.values], positions=[i*2], widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black'))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
    if len(strs) >= 5:
        bp = ax1.boxplot([strs.values], positions=[i*2+0.7], widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black'))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.9)

ax1.set_xticks([0.35, 2.35, 4.35])
ax1.set_xticklabels(['Early', 'Mid', 'Late'])
ax1.set_ylabel('Poly(A) length (nt)')
ax1.set_title('a) Poly(A) by Replication Timing\n(light=Normal, dark=Stress)')

# Panel B: m6A/kb by replication timing
ax2 = fig2.add_subplot(gs2[0, 1])
for i, (timing, color) in enumerate([('early', '#2ca02c'), ('mid', '#ff7f0e'), ('late', '#9467bd')]):
    sub = hela_n[hela_n['repli_timing']==timing]['m6a_per_kb']
    if len(sub) >= 5:
        bp = ax2.boxplot([sub.values], positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black'))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        ax2.text(i, sub.median() + 0.2, f'n={len(sub)}\n{sub.median():.2f}', ha='center', fontsize=7)

ax2.set_xticks(range(3))
ax2.set_xticklabels(['Early', 'Mid', 'Late'])
ax2.set_ylabel('m6A/kb')
ax2.set_title('b) m6A/kb by Replication Timing (HeLa normal)')

# Panel C: LAD × Replication Timing interaction for m6A
ax3 = fig2.add_subplot(gs2[1, 0])
groups = []
m6a_vals = []
labels_int = []
colors_int = []
for lad, lad_label, lad_colors in [(True, 'LAD', ['#ff9999', '#ff4444']),
                                     (False, 'non-LAD', ['#9999ff', '#4444ff'])]:
    for timing, tc in [('early', 0), ('late', 1)]:
        sub = hela_n[(hela_n['in_lad']==lad) & (hela_n['repli_timing']==timing)]
        if len(sub) >= 5:
            groups.append(sub['m6a_per_kb'].values)
            labels_int.append(f'{lad_label}\n{timing}')
            colors_int.append(lad_colors[tc])
            m6a_vals.append(sub['m6a_per_kb'].median())

parts = ax3.violinplot(groups, positions=range(len(groups)), showmedians=True, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_int[i])
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')

for i in range(len(groups)):
    ax3.text(i, m6a_vals[i] + 0.3, f'{m6a_vals[i]:.2f}\nn={len(groups[i])}', ha='center', fontsize=7)

ax3.set_xticks(range(len(labels_int)))
ax3.set_xticklabels(labels_int, fontsize=8)
ax3.set_ylabel('m6A/kb')
ax3.set_title('c) m6A/kb: LAD × Replication Timing')

# Panel D: Cross-cell-line m6A LAD/non-LAD ratio
ax4 = fig2.add_subplot(gs2[1, 1])
ratios = []
cls = []
for cl in sorted(l1[l1['condition']=='normal']['cellline'].unique()):
    cl_sub = l1[(l1['cellline']==cl) & (l1['condition']=='normal')]
    lad_m = cl_sub[cl_sub['in_lad']==True]['m6a_per_kb'].median()
    nonlad_m = cl_sub[cl_sub['in_lad']==False]['m6a_per_kb'].median()
    n_lad = (cl_sub['in_lad']==True).sum()
    if n_lad >= 10 and nonlad_m > 0:
        ratios.append(lad_m / nonlad_m)
        cls.append(cl)

colors_ratio = ['#d62728' if r > 1 else '#1f77b4' for r in ratios]
ax4.bar(range(len(cls)), ratios, color=colors_ratio, alpha=0.8)
ax4.axhline(1.0, color='black', linewidth=0.5, linestyle='--')
ax4.set_xticks(range(len(cls)))
ax4.set_xticklabels(cls, rotation=45, ha='right', fontsize=7)
ax4.set_ylabel('m6A LAD/non-LAD ratio')
ax4.set_title('d) m6A LAD/non-LAD Ratio (per cell line)')

plt.savefig(f'{OUT_DIR}/fig_replication_timing.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig_replication_timing.pdf")

print("\nDone!")
