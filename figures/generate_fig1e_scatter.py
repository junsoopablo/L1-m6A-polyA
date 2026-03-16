#!/usr/bin/env python3
"""
Fig 1e: METTL3 KO — KO/WT fold-change bar + per-replicate dots.

Both L1 and high-m6A genes shown with 3 per-replicate FC dots each.
Bar = overall median(KO)/median(WT). Dots = per-replicate pair FC.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pysam
from scipy import stats
from collections import defaultdict
from fig_style import *

setup_style()

VAULT = '/vault/external-datasets/2026/PRJEB40872_HEK293T_METTL3KO_xPore'
MAFIA_DIR = f'{VAULT}/mafia_guppy'
TX_DIR = f'{VAULT}/mafia_transcriptome'
GENE_BED = f'{TX_DIR}/pc_genes_filtered.bed'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

SAMPLES = {
    'WT_rep1': 'WT', 'WT_rep2': 'WT', 'WT_rep3': 'WT',
    'KO_rep1': 'KO', 'KO_rep2': 'KO', 'KO_rep3': 'KO',
}
REPS = ['rep1', 'rep2', 'rep3']
THR = 128

###############################################################################
# Load gene BED for ctrl
###############################################################################
gene_intervals = defaultdict(list)
with open(GENE_BED) as f:
    for line in f:
        parts = line.strip().split('\t')
        chrom, start, end, gene = parts[0], int(parts[1]), int(parts[2]), parts[3]
        gene_intervals[chrom].append((start, end, gene))
for chrom in gene_intervals:
    gene_intervals[chrom].sort()

def assign_gene(chrom, read_start, read_end):
    if chrom not in gene_intervals:
        return None
    best_gene = None
    best_overlap = 0
    for gstart, gend, gname in gene_intervals[chrom]:
        if gstart > read_end:
            break
        if gend < read_start:
            continue
        overlap = min(read_end, gend) - max(read_start, gstart)
        if overlap > best_overlap:
            best_overlap = overlap
            best_gene = gname
    return best_gene

###############################################################################
# Parse BAMs
###############################################################################
def parse_bam(bam_path, assign_genes=False):
    if not os.path.exists(bam_path):
        return []
    bam = pysam.AlignmentFile(bam_path, 'rb')
    rows = []
    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        rlen = read.query_alignment_length
        if rlen is None or rlen < 100:
            continue
        mm_tag = ml_tag = None
        for t in ['MM', 'Mm']:
            if read.has_tag(t):
                mm_tag = read.get_tag(t); break
        for t in ['ML', 'Ml']:
            if read.has_tag(t):
                ml_tag = read.get_tag(t); break
        n_m6a = 0
        if mm_tag and ml_tag:
            ml_list = list(ml_tag)
            entries = mm_tag.rstrip(';').split(';')
            idx = 0
            for entry in entries:
                parts = entry.strip().split(',')
                base_mod = parts[0]
                skips = [int(x) for x in parts[1:]] if len(parts) > 1 else []
                n_sites = len(skips)
                if '21891' in base_mod:
                    for i in range(n_sites):
                        if idx + i < len(ml_list) and ml_list[idx + i] >= THR:
                            n_m6a += 1
                idx += n_sites
        row = {'read_id': read.query_name, 'read_length': rlen,
               'm6a_sites': n_m6a, 'm6a_per_kb': n_m6a / (rlen / 1000)}
        if assign_genes:
            row['gene_name'] = assign_gene(
                read.reference_name, read.reference_start, read.reference_end)
        rows.append(row)
    bam.close()
    return rows

# --- L1 reads ---
print("Parsing L1 MAFIA BAMs...")
l1_all = []
for sample, condition in SAMPLES.items():
    rows = parse_bam(f'{MAFIA_DIR}/{sample}/mAFiA.reads.bam')
    for r in rows:
        r['sample'] = sample
        r['condition'] = condition
    l1_all.extend(rows)
    print(f"  {sample}: {len(rows)} reads")
l1_df = pd.DataFrame(l1_all)

# --- Transcriptome MAFIA ctrl ---
print("\nParsing transcriptome MAFIA BAMs...")
tx_all = []
for sample, condition in SAMPLES.items():
    bam_path = f'{TX_DIR}/{sample}_protein_coding/mAFiA.reads.bam'
    rows = parse_bam(bam_path, assign_genes=True)
    for r in rows:
        r['sample'] = sample
        r['condition'] = condition
    tx_all.extend(rows)
tx_df = pd.DataFrame(tx_all).dropna(subset=['gene_name'])

# High-m6A genes
gene_cond = tx_df.groupby(['gene_name', 'condition']).agg(
    mean_m6a=('m6a_per_kb', 'mean'), n_reads=('read_id', 'count')
).reset_index()
gene_pivot = gene_cond.pivot(index='gene_name', columns='condition', values='mean_m6a').dropna()
gene_n = gene_cond.pivot(index='gene_name', columns='condition', values='n_reads').dropna()
mask = (gene_n['WT'] >= 3) & (gene_n['KO'] >= 3)
gene_all = gene_pivot[mask]
high_m6a_genes = set(gene_all[gene_all['WT'] >= gene_all['WT'].quantile(0.67)].index)
ctrl_df = tx_df[tx_df['gene_name'].isin(high_m6a_genes)].copy()
print(f"High-m6A genes: {len(high_m6a_genes)}, ctrl reads: {len(ctrl_df)}")

###############################################################################
# Per-replicate medians + fold-changes
###############################################################################
l1_wt_reps, l1_ko_reps = [], []
ctrl_wt_reps, ctrl_ko_reps = [], []
for rep in REPS:
    wt_s, ko_s = f'WT_{rep}', f'KO_{rep}'
    l1_wt_reps.append(l1_df[l1_df['sample'] == wt_s]['m6a_per_kb'].median())
    l1_ko_reps.append(l1_df[l1_df['sample'] == ko_s]['m6a_per_kb'].median())
    ctrl_wt_reps.append(ctrl_df[ctrl_df['sample'] == wt_s]['m6a_per_kb'].median())
    ctrl_ko_reps.append(ctrl_df[ctrl_df['sample'] == ko_s]['m6a_per_kb'].median())

l1_wt_reps = np.array(l1_wt_reps)
l1_ko_reps = np.array(l1_ko_reps)
ctrl_wt_reps = np.array(ctrl_wt_reps)
ctrl_ko_reps = np.array(ctrl_ko_reps)

# Overall FC + P (read-level)
l1_wt_all = l1_df[l1_df['condition'] == 'WT']['m6a_per_kb']
l1_ko_all = l1_df[l1_df['condition'] == 'KO']['m6a_per_kb']
_, l1_p = stats.mannwhitneyu(l1_wt_all, l1_ko_all, alternative='two-sided')
l1_fc = l1_ko_all.median() / l1_wt_all.median()

ctrl_wt_all = ctrl_df[ctrl_df['condition'] == 'WT']['m6a_per_kb']
ctrl_ko_all = ctrl_df[ctrl_df['condition'] == 'KO']['m6a_per_kb']
_, ctrl_p = stats.mannwhitneyu(ctrl_wt_all, ctrl_ko_all, alternative='two-sided')
ctrl_fc = ctrl_ko_all.median() / ctrl_wt_all.median()

l1_rep_fc = l1_ko_reps / l1_wt_reps
ctrl_rep_fc = ctrl_ko_reps / ctrl_wt_reps
print(f"\nL1:   WT_med={l1_wt_all.median():.2f}, KO_med={l1_ko_all.median():.2f}, FC={l1_fc:.3f}x, per-rep FC={l1_rep_fc.round(3)}, P={l1_p:.2e}")
print(f"Ctrl: WT_med={ctrl_wt_all.median():.2f}, KO_med={ctrl_ko_all.median():.2f}, FC={ctrl_fc:.3f}x, per-rep FC={ctrl_rep_fc.round(3)}, P={ctrl_p:.2e}")

###############################################################################
# Plot — FC bar + replicate FC dots (optC styling)
###############################################################################
fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.75))
panel_label(ax, 'd')

# L1 replicate FC dots + SEM (no bar)
l1_fc_sem = l1_rep_fc.std(ddof=1) / np.sqrt(3)
ax.errorbar(0, l1_rep_fc.mean(), yerr=l1_fc_sem,
            fmt='none', ecolor='black', elinewidth=0.7, capsize=3, capthick=0.7, zorder=4)
ax.scatter([0]*3, l1_rep_fc, s=30, c=C_L1, edgecolors='white',
           linewidths=0.5, zorder=5)
# Mean tick
ax.hlines(l1_rep_fc.mean(), -0.15, 0.15, color='black', lw=1.2, zorder=6)

# Ctrl replicate FC dots + SEM (no bar)
ctrl_fc_sem = ctrl_rep_fc.std(ddof=1) / np.sqrt(3)
ax.errorbar(1, ctrl_rep_fc.mean(), yerr=ctrl_fc_sem,
            fmt='none', ecolor='black', elinewidth=0.7, capsize=3, capthick=0.7, zorder=4)
ax.scatter([1]*3, ctrl_rep_fc, s=30, c=C_CTRL, edgecolors='white',
           linewidths=0.5, zorder=5)
# Mean tick
ax.hlines(ctrl_rep_fc.mean(), 0.85, 1.15, color='black', lw=1.2, zorder=6)

# --- FC + significance annotation ---
ax.text(0, min(l1_rep_fc) - 0.015,
        f'{l1_fc:.2f}x {significance_text(l1_p)}',
        ha='center', va='top', fontsize=FS_ANNOT,
        fontweight='bold', color=C_L1)

ax.text(1, min(ctrl_rep_fc) - 0.015,
        f'{ctrl_fc:.2f}x {significance_text(ctrl_p)}',
        ha='center', va='top', fontsize=FS_ANNOT,
        fontweight='bold', color=C_CTRL)

# --- Axes: strong line at y=1.0, labels at bottom ---
ax.set_xticks([0, 1])
ax.set_xticklabels(['L1', 'High-m6A\ngenes'], fontsize=FS_ANNOT)
ax.set_ylabel('KO / WT m6A/kb')
ax.set_xlim(-0.5, 1.55)
ax.set_ylim(0.76, 1.08)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
# Strong reference line at 1.0 (acts as visual x-axis)
ax.axhline(1.0, color=C_TEXT, lw=0.8, zorder=6)
# Hide bottom spine (real x-axis at 0.76 is not meaningful)
ax.spines['bottom'].set_visible(False)
ax.tick_params(axis='x', length=0)  # hide x ticks

save_figure(fig, f'{OUTDIR}/fig1d')
print(f"\nfig1d (METTL3 KO) saved.")
