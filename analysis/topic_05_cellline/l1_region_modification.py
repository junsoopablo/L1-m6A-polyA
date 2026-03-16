#!/usr/bin/env python3
"""Within-read L1 vs flanking modification comparison.

For L1 reads, modification sites fall in two categories:
  - L1-internal: within RepeatMasker L1 TE annotation
  - Flanking: outside L1 TE, in host gene sequence

Same read, same library, same DRS signal → only difference is sequence context.
If L1-internal sites have higher modRatio, L1 sequence promotes modification.

Uses MAFIA sites.bed (already computed, genomic coordinates) + L1 TE BED.

Usage: conda run -n research python l1_region_modification.py
"""
import os
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
L1_TE_BED = BASE / 'reference/L1_TE_L1_family.bed'
OUT_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/l1_region_mod'

CELL_LINES = {
    'A549': ['A549_4','A549_5','A549_6'],
    'H9': ['H9_2','H9_3','H9_4'],
    'Hct116': ['Hct116_3','Hct116_4'],
    'HeLa': ['HeLa_1','HeLa_2','HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1','HeLa-Ars_2','HeLa-Ars_3'],
    'HepG2': ['HepG2_5','HepG2_6'],
    'HEYA8': ['HEYA8_1','HEYA8_2','HEYA8_3'],
    'K562': ['K562_4','K562_5','K562_6'],
    'MCF7': ['MCF7_2','MCF7_3','MCF7_4'],
    'MCF7-EV': ['MCF7-EV_1'],
    'SHSY5Y': ['SHSY5Y_1','SHSY5Y_2','SHSY5Y_3'],
}

GROUP_TO_CL = {}
ALL_GROUPS = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        GROUP_TO_CL[g] = cl
        ALL_GROUPS.append(g)


def classify_sites(group):
    """Classify modification sites as L1-internal vs flanking."""
    sites_bed = BASE / f'results_group/{group}/h_mafia/{group}.mAFiA.sites.bed'
    if not sites_bed.exists():
        return None

    # Load sites.bed: chrom, start, end, name(mod_type), score, strand, ref5mer, coverage, modRatio, confidence
    sites = pd.read_csv(sites_bed, sep='\t',
                        names=['chrom','start','end','mod_type','score','strand',
                               'ref5mer','coverage','modRatio','confidence'])

    # bedtools intersect with L1 TE BED → mark sites in L1
    tmp_sites = tempfile.NamedTemporaryFile(
        mode='w', suffix='.bed', delete=False, prefix=f'sites_{group}_')
    sites[['chrom','start','end','mod_type','score','strand',
           'ref5mer','coverage','modRatio','confidence']].to_csv(
        tmp_sites, sep='\t', header=False, index=False)
    tmp_sites.close()

    # -c: report L1 overlap count, -C: count overlapping features
    result = subprocess.run(
        f"bedtools intersect -a {tmp_sites.name} -b {L1_TE_BED} -c",
        shell=True, capture_output=True, text=True)

    os.unlink(tmp_sites.name)

    # Parse result: original columns + overlap_count
    rows = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        fields = line.split('\t')
        rows.append(fields)

    df = pd.DataFrame(rows, columns=[
        'chrom','start','end','mod_type','score','strand',
        'ref5mer','coverage','modRatio','confidence','l1_overlap_count'])

    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['coverage'] = pd.to_numeric(df['coverage'], errors='coerce')
    df['modRatio'] = pd.to_numeric(df['modRatio'], errors='coerce')
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df['l1_overlap_count'] = df['l1_overlap_count'].astype(int)
    df['in_L1'] = df['l1_overlap_count'] > 0
    df['group'] = group
    df['cell_line'] = GROUP_TO_CL.get(group, group)

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Classifying modification sites as L1-internal vs flanking...")
    all_sites = []

    for group in ALL_GROUPS:
        df = classify_sites(group)
        if df is None:
            print(f"  {group}: sites.bed not found, skipping")
            continue

        n_total = len(df)
        n_l1 = df['in_L1'].sum()
        n_flank = n_total - n_l1
        print(f"  {group}: {n_total} sites, L1={n_l1} ({n_l1/n_total*100:.1f}%), "
              f"flank={n_flank} ({n_flank/n_total*100:.1f}%)")
        all_sites.append(df)

    sites = pd.concat(all_sites, ignore_index=True)
    sites.to_csv(OUT_DIR / 'all_sites_classified.tsv.gz', sep='\t', index=False,
                 compression='gzip')

    # ── Analysis 1: Overall L1-internal vs flanking modRatio ──
    print("\n" + "="*70)
    print("ANALYSIS 1: L1-internal vs Flanking modRatio")
    print("="*70)

    for mod in ['m6A', 'psi']:
        sub = sites[sites['mod_type'] == mod]
        l1 = sub[sub['in_L1']]['modRatio']
        flank = sub[~sub['in_L1']]['modRatio']

        if len(l1) == 0 or len(flank) == 0:
            continue

        stat, pval = stats.mannwhitneyu(l1, flank, alternative='two-sided')
        print(f"\n  {mod}:")
        print(f"    L1-internal: n={len(l1):,}, median={l1.median():.1f}%, mean={l1.mean():.1f}%")
        print(f"    Flanking:    n={len(flank):,}, median={flank.median():.1f}%, mean={flank.mean():.1f}%")
        print(f"    MW U-test: p={pval:.2e}")

    # ── Analysis 2: Per-motif comparison (same motif, L1 vs flanking) ──
    print("\n" + "="*70)
    print("ANALYSIS 2: Per-motif L1 vs Flanking (controls for motif composition)")
    print("="*70)

    motif_results = []
    for mod in ['m6A', 'psi']:
        sub = sites[sites['mod_type'] == mod]
        motifs = sub['ref5mer'].value_counts()
        # Only motifs with >= 50 sites in both L1 and flanking
        for motif in motifs.index:
            l1_m = sub[(sub['ref5mer'] == motif) & sub['in_L1']]['modRatio']
            flank_m = sub[(sub['ref5mer'] == motif) & ~sub['in_L1']]['modRatio']
            if len(l1_m) < 50 or len(flank_m) < 50:
                continue
            stat, pval = stats.mannwhitneyu(l1_m, flank_m, alternative='two-sided')
            motif_results.append({
                'mod_type': mod, 'motif': motif,
                'n_l1': len(l1_m), 'n_flank': len(flank_m),
                'mean_l1': l1_m.mean(), 'mean_flank': flank_m.mean(),
                'median_l1': l1_m.median(), 'median_flank': flank_m.median(),
                'delta': l1_m.mean() - flank_m.mean(),
                'pval': pval,
            })

    motif_df = pd.DataFrame(motif_results)
    if not motif_df.empty:
        motif_df = motif_df.sort_values(['mod_type', 'delta'], ascending=[True, False])
        motif_df.to_csv(OUT_DIR / 'per_motif_l1_vs_flanking.tsv', sep='\t', index=False)

        for mod in ['m6A', 'psi']:
            mdf = motif_df[motif_df['mod_type'] == mod]
            if mdf.empty:
                continue
            print(f"\n  {mod} motifs (n_l1 >= 50 & n_flank >= 50):")
            print(f"  {'Motif':<8} {'L1 mean':>8} {'Flank mean':>10} {'Delta':>8} {'p-value':>10}")
            for _, row in mdf.iterrows():
                sig = '***' if row['pval'] < 0.001 else '**' if row['pval'] < 0.01 else '*' if row['pval'] < 0.05 else 'ns'
                print(f"  {row['motif']:<8} {row['mean_l1']:>7.1f}% {row['mean_flank']:>9.1f}% "
                      f"{row['delta']:>+7.1f}% {row['pval']:>9.2e} {sig}")

    # ── Analysis 3: Per-cell-line consistency ──
    print("\n" + "="*70)
    print("ANALYSIS 3: Per-cell-line L1 vs Flanking consistency")
    print("="*70)

    cl_results = []
    for cl in sorted(CELL_LINES.keys()):
        for mod in ['m6A', 'psi']:
            sub = sites[(sites['cell_line'] == cl) & (sites['mod_type'] == mod)]
            l1 = sub[sub['in_L1']]['modRatio']
            flank = sub[~sub['in_L1']]['modRatio']
            if len(l1) < 10 or len(flank) < 10:
                continue
            stat, pval = stats.mannwhitneyu(l1, flank, alternative='two-sided')
            cl_results.append({
                'cell_line': cl, 'mod_type': mod,
                'n_l1': len(l1), 'n_flank': len(flank),
                'mean_l1': l1.mean(), 'mean_flank': flank.mean(),
                'delta': l1.mean() - flank.mean(),
                'pval': pval,
            })

    cl_df = pd.DataFrame(cl_results)
    cl_df.to_csv(OUT_DIR / 'per_cellline_l1_vs_flanking.tsv', sep='\t', index=False)

    for mod in ['m6A', 'psi']:
        cdf = cl_df[cl_df['mod_type'] == mod].sort_values('delta', ascending=False)
        if cdf.empty:
            continue
        print(f"\n  {mod}:")
        print(f"  {'CL':<12} {'L1 mean':>8} {'Flank mean':>10} {'Delta':>8} {'p-value':>10}")
        n_sig = 0
        for _, row in cdf.iterrows():
            sig = '***' if row['pval'] < 0.001 else '**' if row['pval'] < 0.01 else '*' if row['pval'] < 0.05 else 'ns'
            if row['pval'] < 0.05:
                n_sig += 1
            print(f"  {row['cell_line']:<12} {row['mean_l1']:>7.1f}% {row['mean_flank']:>9.1f}% "
                  f"{row['delta']:>+7.1f}% {row['pval']:>9.2e} {sig}")
        print(f"  Consistent: {n_sig}/{len(cdf)} cell lines significant (p<0.05)")

    # ── Analysis 4: Motif frequency in L1 vs flanking ──
    print("\n" + "="*70)
    print("ANALYSIS 4: Motif frequency distribution")
    print("="*70)

    for mod in ['m6A', 'psi']:
        sub = sites[sites['mod_type'] == mod]
        l1_total = sub['in_L1'].sum()
        flank_total = (~sub['in_L1']).sum()

        motif_freq = sub.groupby(['ref5mer', 'in_L1']).size().unstack(fill_value=0)
        if True in motif_freq.columns and False in motif_freq.columns:
            motif_freq['l1_pct'] = motif_freq[True] / l1_total * 100
            motif_freq['flank_pct'] = motif_freq[False] / flank_total * 100
            motif_freq['enrichment'] = motif_freq['l1_pct'] / motif_freq['flank_pct'].replace(0, np.nan)
            motif_freq = motif_freq.sort_values('enrichment', ascending=False)

            print(f"\n  {mod} motif enrichment (L1 vs flanking):")
            print(f"  {'Motif':<8} {'L1 %':>6} {'Flank %':>8} {'Enrichment':>10}")
            for motif, row in motif_freq.head(10).iterrows():
                print(f"  {motif:<8} {row['l1_pct']:>5.1f}% {row['flank_pct']:>7.1f}% "
                      f"{row['enrichment']:>9.2f}x")

    # ── Figures ──
    print("\nGenerating figures...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fig A: Overall modRatio distribution
    ax = axes[0, 0]
    for mod, color_l1, color_fl in [('psi', '#e74c3c', '#3498db'), ('m6A', '#e67e22', '#2ecc71')]:
        sub = sites[sites['mod_type'] == mod]
        l1 = sub[sub['in_L1']]['modRatio']
        flank = sub[~sub['in_L1']]['modRatio']
        bins = np.arange(0, 105, 5)
        ax.hist(l1, bins=bins, alpha=0.4, density=True, label=f'{mod} L1-internal', color=color_l1)
        ax.hist(flank, bins=bins, alpha=0.4, density=True, label=f'{mod} flanking', color=color_fl,
                linestyle='--')
    ax.set_xlabel('modRatio (%)')
    ax.set_ylabel('Density')
    ax.set_title('A. Modification Rate: L1-internal vs Flanking')
    ax.legend(fontsize=8)

    # Fig B: Per-cell-line delta (L1 - flanking)
    ax = axes[0, 1]
    if not cl_df.empty:
        for i, mod in enumerate(['psi', 'm6A']):
            cdf = cl_df[cl_df['mod_type'] == mod].sort_values('cell_line')
            if cdf.empty:
                continue
            x = np.arange(len(cdf))
            offset = (i - 0.5) * 0.35
            colors = ['#e74c3c' if mod == 'psi' else '#e67e22'] * len(cdf)
            bars = ax.bar(x + offset, cdf['delta'], 0.35, label=mod, color=colors[0], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(cdf['cell_line'], rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Delta modRatio (L1 - flanking, %)')
    ax.set_title('B. L1 vs Flanking Delta per Cell Line')
    ax.legend()

    # Fig C: Per-motif delta for psi
    ax = axes[1, 0]
    if not motif_df.empty:
        psi_motifs = motif_df[motif_df['mod_type'] == 'psi'].sort_values('delta', ascending=False)
        if not psi_motifs.empty:
            x = np.arange(len(psi_motifs))
            colors = ['#e74c3c' if d > 0 else '#3498db' for d in psi_motifs['delta']]
            ax.bar(x, psi_motifs['delta'], color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(psi_motifs['motif'], rotation=45, ha='right', fontsize=8)
            ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Delta modRatio (L1 - flanking, %)')
    ax.set_title('C. Psi Per-Motif: L1 vs Flanking')

    # Fig D: Per-motif delta for m6A
    ax = axes[1, 1]
    if not motif_df.empty:
        m6a_motifs = motif_df[motif_df['mod_type'] == 'm6A'].sort_values('delta', ascending=False)
        if not m6a_motifs.empty:
            x = np.arange(len(m6a_motifs))
            colors = ['#e67e22' if d > 0 else '#2ecc71' for d in m6a_motifs['delta']]
            ax.bar(x, m6a_motifs['delta'], color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(m6a_motifs['motif'], rotation=45, ha='right', fontsize=8)
            ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Delta modRatio (L1 - flanking, %)')
    ax.set_title('D. m6A Per-Motif: L1 vs Flanking')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'l1_vs_flanking_modification.png', dpi=150)
    plt.close()
    print(f"Figure saved: {OUT_DIR / 'l1_vs_flanking_modification.png'}")

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
