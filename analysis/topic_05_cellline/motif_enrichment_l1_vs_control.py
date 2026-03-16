#!/usr/bin/env python3
"""
L1 vs Control motif enrichment analysis.
Updated: All 16 psi classifier motifs + Control comparison.

Key questions:
1. Which motifs have highest modification rates in L1 vs Control?
2. Are motif preferences L1-specific or universal?
3. DRACH position decomposition: L1 vs Control
4. Full psi motif landscape (16/16 corrected)
5. Per-cell-line motif consistency

Run with: conda run -n research python3 motif_enrichment_l1_vs_control.py
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats

PROJECT_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
RESULTS_GROUP_DIR = PROJECT_DIR / "results_group"
OUTPUT_DIR = PROJECT_DIR / "analysis/01_exploration/topic_05_cellline"
REF_FASTA = PROJECT_DIR / "reference" / "Human.fasta"

GROUPS = [
    "A549_4", "A549_5", "A549_6",
    "H9_2", "H9_3", "H9_4",
    "Hct116_3", "Hct116_4",
    "HeLa_1", "HeLa_2", "HeLa_3",
    "HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3",
    "HepG2_5", "HepG2_6",
    "HEYA8_1", "HEYA8_2", "HEYA8_3",
    "K562_4", "K562_5", "K562_6",
    "MCF7_2", "MCF7_3", "MCF7_4",
    "MCF7-EV_1",
    "SHSY5Y_1", "SHSY5Y_2", "SHSY5Y_3",
]
# Control groups: MCF7-EV uses MCF7's control
CTRL_GROUPS = [g for g in GROUPS if g != "MCF7-EV_1"]

M6A_MOTIFS = sorted([
    'AAACA', 'AAACC', 'AAACT', 'AGACA', 'AGACC', 'AGACT',
    'GAACA', 'GAACC', 'GAACT', 'GGACA', 'GGACC', 'GGACT',
    'TAACA', 'TAACC', 'TAACT', 'TGACA', 'TGACC', 'TGACT',
])

PSI_MOTIFS = sorted([
    'AGTGG', 'ATTTG', 'CATAA', 'CATCC', 'CCTCC', 'CTTTA',
    'GATGC', 'GGTCC', 'GGTGG', 'GTTCA', 'GTTCC', 'GTTCG',
    'GTTCT', 'TATAA', 'TGTAG', 'TGTGG',
])

# ============================================================
# PART 1: Load all pileup data
# ============================================================
print("=" * 70)
print("PART 1: Loading pileup data (L1 + Control)")
print("=" * 70)

def load_pileup(groups, source_type):
    """Load pileup from L1 ('l1') or Control ('ctrl') MAFIA sites.bed"""
    dfs = []
    for group in groups:
        if source_type == 'l1':
            path = RESULTS_GROUP_DIR / group / "h_mafia" / f"{group}.mAFiA.sites.bed"
        else:
            path = RESULTS_GROUP_DIR / group / "i_control" / "mafia" / f"{group}.control.mAFiA.sites.bed"
        if not path.exists():
            print(f"  WARNING: missing {path}")
            continue
        df = pd.read_csv(path, sep='\t')
        df['group'] = group
        df['cell_line'] = group.rsplit('_', 1)[0]
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined['source'] = 'L1' if source_type == 'l1' else 'Control'
    return combined

l1_pileup = load_pileup(GROUPS, 'l1')
ctrl_pileup = load_pileup(CTRL_GROUPS, 'ctrl')

print(f"L1 pileup: {len(l1_pileup):,} entries from {l1_pileup['group'].nunique()} groups")
print(f"Control pileup: {len(ctrl_pileup):,} entries from {ctrl_pileup['group'].nunique()} groups")

# Filter to covered sites
l1 = l1_pileup[l1_pileup['coverage'] > 0].copy()
ctrl = ctrl_pileup[ctrl_pileup['coverage'] > 0].copy()
print(f"\nCovered: L1 {len(l1):,}, Control {len(ctrl):,}")

for src, df in [('L1', l1), ('Control', ctrl)]:
    for mod in ['m6A', 'psi']:
        sub = df[df['name'] == mod]
        motifs = sorted(sub['ref5mer'].unique())
        print(f"  {src} {mod}: {len(sub):,} sites, {len(motifs)} motifs: {motifs}")

# ============================================================
# PART 2: Per-motif modification rates
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Per-motif modification rates")
print("=" * 70)

def compute_motif_stats(df, mod_type, motif_list):
    """Compute per-motif stats from pileup data."""
    sub = df[df['name'] == mod_type].copy()
    sub['is_modified'] = sub['modRatio'] > 50

    total_sites = len(sub)
    total_mod = sub['is_modified'].sum()
    overall_rate = 100 * total_mod / total_sites if total_sites > 0 else 0

    results = []
    for motif in motif_list:
        m = sub[sub['ref5mer'] == motif]
        n = len(m)
        if n == 0:
            results.append({
                'motif': motif, 'n_sites': 0, 'n_modified': 0,
                'rate': np.nan, 'weighted_rate': np.nan, 'enrichment': np.nan,
                'mean_coverage': np.nan,
            })
            continue
        n_mod = m['is_modified'].sum()
        rate = 100 * n_mod / n
        w_rate = np.average(m['modRatio'], weights=m['coverage']) if m['coverage'].sum() > 0 else 0
        enrichment = rate / overall_rate if overall_rate > 0 else np.nan
        results.append({
            'motif': motif, 'n_sites': n, 'n_modified': n_mod,
            'rate': rate, 'weighted_rate': w_rate, 'enrichment': enrichment,
            'mean_coverage': m['coverage'].mean(),
        })
    return pd.DataFrame(results), overall_rate

# Compute for all 4 combinations
results = {}
overall_rates = {}
for src_label, src_df in [('L1', l1), ('Control', ctrl)]:
    for mod, motifs in [('m6A', M6A_MOTIFS), ('psi', PSI_MOTIFS)]:
        key = (src_label, mod)
        df_res, overall = compute_motif_stats(src_df, mod, motifs)
        df_res['source'] = src_label
        df_res['mod_type'] = mod
        results[key] = df_res
        overall_rates[key] = overall

        print(f"\n{src_label} {mod}: overall rate={overall:.1f}%")
        for _, r in df_res.iterrows():
            if r['n_sites'] > 0:
                print(f"  {r['motif']}: sites={r['n_sites']:>8,}  mod={r['n_modified']:>7,}  "
                      f"rate={r['rate']:5.1f}%  weighted={r['weighted_rate']:5.1f}%  "
                      f"enrich={r['enrichment']:.2f}x  cov={r['mean_coverage']:.1f}")

# Combine all results
all_results = pd.concat(results.values(), ignore_index=True)
all_results.to_csv(OUTPUT_DIR / 'motif_enrichment_l1_vs_ctrl.tsv', sep='\t', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'motif_enrichment_l1_vs_ctrl.tsv'}")

# ============================================================
# PART 3: L1 vs Control statistical comparison per motif
# ============================================================
print("\n" + "=" * 70)
print("PART 3: L1 vs Control comparison (Fisher's exact per motif)")
print("=" * 70)

comparison_results = []
for mod, motifs in [('m6A', M6A_MOTIFS), ('psi', PSI_MOTIFS)]:
    l1_res = results[('L1', mod)]
    ctrl_res = results[('Control', mod)]

    print(f"\n{mod}:")
    print(f"  {'Motif':>7}  {'L1_rate':>8}  {'Ctrl_rate':>9}  {'Delta':>7}  {'OR':>7}  {'p-value':>10}  {'sig':>5}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*5}")

    for motif in motifs:
        l1_row = l1_res[l1_res['motif'] == motif].iloc[0]
        ctrl_row = ctrl_res[ctrl_res['motif'] == motif].iloc[0]

        if l1_row['n_sites'] == 0 or ctrl_row['n_sites'] == 0:
            comparison_results.append({
                'mod_type': mod, 'motif': motif,
                'l1_rate': np.nan, 'ctrl_rate': np.nan,
                'delta': np.nan, 'OR': np.nan, 'p_value': np.nan,
            })
            continue

        # 2x2: modified/not x L1/Control
        a = int(l1_row['n_modified'])
        b = int(l1_row['n_sites'] - l1_row['n_modified'])
        c = int(ctrl_row['n_modified'])
        d = int(ctrl_row['n_sites'] - ctrl_row['n_modified'])
        OR, p = stats.fisher_exact([[a, b], [c, d]])

        delta = l1_row['rate'] - ctrl_row['rate']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        comparison_results.append({
            'mod_type': mod, 'motif': motif,
            'l1_rate': l1_row['rate'], 'ctrl_rate': ctrl_row['rate'],
            'l1_weighted': l1_row['weighted_rate'], 'ctrl_weighted': ctrl_row['weighted_rate'],
            'delta': delta, 'OR': OR, 'p_value': p,
        })

        print(f"  {motif:>7}  {l1_row['rate']:7.1f}%  {ctrl_row['rate']:8.1f}%  "
              f"{delta:+6.1f}  {OR:7.2f}  {p:10.2e}  {sig}")

comp_df = pd.DataFrame(comparison_results)
comp_df.to_csv(OUTPUT_DIR / 'motif_l1_vs_ctrl_comparison.tsv', sep='\t', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'motif_l1_vs_ctrl_comparison.tsv'}")

# Summary
for mod in ['m6A', 'psi']:
    sub = comp_df[comp_df['mod_type'] == mod].dropna()
    n_higher = (sub['delta'] > 0).sum()
    n_sig = (sub['p_value'] < 0.05).sum()
    mean_delta = sub['delta'].mean()
    print(f"\n{mod}: {n_higher}/{len(sub)} motifs higher in L1, {n_sig} significant (p<0.05)")
    print(f"  Mean delta: {mean_delta:+.1f} pp")

# ============================================================
# PART 4: DRACH decomposition (m6A)
# ============================================================
print("\n" + "=" * 70)
print("PART 4: DRACH position decomposition (m6A)")
print("=" * 70)

for src_label in ['L1', 'Control']:
    m6a_res = results[(src_label, 'm6A')].copy()
    m6a_assessed = m6a_res[m6a_res['n_sites'] > 0].copy()

    print(f"\n--- {src_label} ---")
    for pos, label, bases in [(0, 'D (pos1)', 'AGT'), (1, 'R (pos2)', 'AG'), (4, 'H (pos5)', 'ACT')]:
        m6a_assessed[f'pos_{pos}'] = m6a_assessed['motif'].str[pos]
        grp = m6a_assessed.groupby(f'pos_{pos}').agg(
            n_sites=('n_sites', 'sum'),
            n_modified=('n_modified', 'sum'),
        )
        grp['rate'] = 100 * grp['n_modified'] / grp['n_sites']
        print(f"  {label} ({'/'.join(bases)}):")
        for base, row in grp.iterrows():
            rna = {'A': 'A', 'G': 'G', 'T': 'U', 'C': 'C'}[base]
            print(f"    {base}(RNA:{rna}): sites={row['n_sites']:>9,.0f}  "
                  f"mod={row['n_modified']:>8,.0f}  rate={row['rate']:.1f}%")

# ============================================================
# PART 5: Reference motif frequency scanning
# ============================================================
print("\n" + "=" * 70)
print("PART 5: Reference motif frequency in L1 and Control regions")
print("=" * 70)

def scan_regions_for_motifs(region_files, motif_set):
    """Scan reference sequence in regions for 5-mer motifs."""
    all_regions = set()
    for rfile in region_files:
        if not rfile.exists():
            continue
        with open(rfile) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                    strand = parts[3] if len(parts) >= 4 else '+'
                    all_regions.add((chrom, start, end, strand))

    ref_fasta = pysam.FastaFile(str(REF_FASTA))
    counts = Counter()
    total_bp = 0
    rc_map = str.maketrans('ACGT', 'TGCA')

    for chrom, start, end, strand in sorted(all_regions):
        try:
            seq = ref_fasta.fetch(chrom, start, end).upper()
        except (ValueError, KeyError):
            continue
        total_bp += len(seq)
        for j in range(len(seq) - 4):
            kmer = seq[j:j+5]
            if kmer in motif_set:
                counts[kmer] += 1
        seq_rc = seq[::-1].translate(rc_map)
        for j in range(len(seq_rc) - 4):
            kmer = seq_rc[j:j+5]
            if kmer in motif_set:
                counts[kmer] += 1
    ref_fasta.close()
    return counts, total_bp, len(all_regions)

all_motif_set = set(M6A_MOTIFS) | set(PSI_MOTIFS)

# L1 regions
l1_region_files = [RESULTS_GROUP_DIR / g / "h_mafia" / f"{g}_regions.bed" for g in GROUPS]
print("Scanning L1 regions...")
l1_ref_counts, l1_bp, l1_n_regions = scan_regions_for_motifs(l1_region_files, all_motif_set)
print(f"  {l1_n_regions:,} regions, {l1_bp:,} bp (both strands)")

# Control regions
ctrl_region_files = [RESULTS_GROUP_DIR / g / "i_control" / "mafia" / f"{g}.control_regions.bed" for g in CTRL_GROUPS]
print("Scanning Control regions...")
ctrl_ref_counts, ctrl_bp, ctrl_n_regions = scan_regions_for_motifs(ctrl_region_files, all_motif_set)
print(f"  {ctrl_n_regions:,} regions, {ctrl_bp:,} bp (both strands)")

# Print comparison
print(f"\nReference motif frequency comparison:")
print(f"  {'Motif':>7}  {'Type':>4}  {'L1_count':>10}  {'L1_freq':>8}  {'Ctrl_count':>11}  {'Ctrl_freq':>9}  {'L1/Ctrl':>8}")
for mod, motifs in [('m6A', M6A_MOTIFS), ('psi', PSI_MOTIFS)]:
    for motif in motifs:
        lc = l1_ref_counts.get(motif, 0)
        cc = ctrl_ref_counts.get(motif, 0)
        lf = lc / l1_bp * 1000 if l1_bp > 0 else 0
        cf = cc / ctrl_bp * 1000 if ctrl_bp > 0 else 0
        ratio = lf / cf if cf > 0 else np.nan
        print(f"  {motif:>7}  {mod:>4}  {lc:>10,}  {lf:>7.3f}‰  {cc:>11,}  {cf:>8.3f}‰  {ratio:>7.2f}x")

# ============================================================
# PART 6: Per-cell-line motif consistency
# ============================================================
print("\n" + "=" * 70)
print("PART 6: Per-cell-line motif consistency")
print("=" * 70)

cl_motif_results = []
for src_label, src_df in [('L1', l1), ('Control', ctrl)]:
    for mod in ['m6A', 'psi']:
        sub = src_df[src_df['name'] == mod].copy()
        sub['is_modified'] = sub['modRatio'] > 50
        for cl in sorted(sub['cell_line'].unique()):
            cl_sub = sub[sub['cell_line'] == cl]
            if len(cl_sub) < 100:
                continue
            for motif in cl_sub['ref5mer'].unique():
                m = cl_sub[cl_sub['ref5mer'] == motif]
                if len(m) < 5:
                    continue
                n_mod = m['is_modified'].sum()
                rate = 100 * n_mod / len(m)
                cl_motif_results.append({
                    'source': src_label, 'mod_type': mod,
                    'cell_line': cl, 'motif': motif,
                    'n_sites': len(m), 'n_modified': n_mod, 'rate': rate,
                })

cl_motif_df = pd.DataFrame(cl_motif_results)

# Summary: coefficient of variation across cell lines per motif
print(f"\nPer-motif CV across cell lines:")
for src in ['L1', 'Control']:
    for mod in ['m6A', 'psi']:
        sub = cl_motif_df[(cl_motif_df['source'] == src) & (cl_motif_df['mod_type'] == mod)]
        pivot = sub.pivot_table(index='cell_line', columns='motif', values='rate')
        if pivot.empty:
            continue
        cv = pivot.std() / pivot.mean()
        print(f"  {src} {mod}: mean CV={cv.mean():.3f}, range=[{cv.min():.3f}, {cv.max():.3f}]")
        # Top 3 most variable
        top3 = cv.nlargest(3)
        for motif, c in top3.items():
            print(f"    Most variable: {motif} CV={c:.3f}")

# ============================================================
# PART 7: Figures
# ============================================================
print("\n" + "=" * 70)
print("PART 7: Generating figures")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(26, 16))

# Color palette
L1_COLOR = '#d32f2f'
CTRL_COLOR = '#1565c0'

# --- Fig A: m6A per-motif rate, L1 vs Control ---
ax = axes[0, 0]
l1_m6a = results[('L1', 'm6A')].sort_values('rate', ascending=True)
ctrl_m6a = results[('Control', 'm6A')].set_index('motif')
motif_order = l1_m6a['motif'].values
y = np.arange(len(motif_order))
bar_h = 0.35

for i, motif in enumerate(motif_order):
    l1_rate = l1_m6a[l1_m6a['motif'] == motif]['rate'].values[0]
    ctrl_rate = ctrl_m6a.loc[motif, 'rate'] if motif in ctrl_m6a.index else 0
    ax.barh(i + bar_h/2, l1_rate, bar_h, color=L1_COLOR, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.barh(i - bar_h/2, ctrl_rate, bar_h, color=CTRL_COLOR, alpha=0.85,
            edgecolor='white', linewidth=0.5)

ax.set_yticks(y)
ax.set_yticklabels(motif_order, fontsize=8, fontfamily='monospace')
ax.set_xlabel('Modification rate (% sites with modRatio>50)')
ax.set_title('A. m6A per-DRACH motif rate', fontweight='bold')
ax.legend(['L1', 'Control'], loc='lower right', fontsize=9)

# --- Fig B: Psi per-motif rate, L1 vs Control ---
ax = axes[0, 1]
l1_psi_res = results[('L1', 'psi')].sort_values('rate', ascending=True)
ctrl_psi_res = results[('Control', 'psi')].set_index('motif')
motif_order_psi = l1_psi_res['motif'].values
y = np.arange(len(motif_order_psi))

for i, motif in enumerate(motif_order_psi):
    l1_rate = l1_psi_res[l1_psi_res['motif'] == motif]['rate'].values[0]
    ctrl_rate = ctrl_psi_res.loc[motif, 'rate'] if motif in ctrl_psi_res.index and not np.isnan(ctrl_psi_res.loc[motif, 'rate']) else 0
    ax.barh(i + bar_h/2, l1_rate, bar_h, color=L1_COLOR, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.barh(i - bar_h/2, ctrl_rate, bar_h, color=CTRL_COLOR, alpha=0.85,
            edgecolor='white', linewidth=0.5)

ax.set_yticks(y)
ax.set_yticklabels(motif_order_psi, fontsize=8, fontfamily='monospace')
ax.set_xlabel('Modification rate (% sites with modRatio>50)')
ax.set_title('B. Psi per-motif rate (all 16)', fontweight='bold')
ax.legend(['L1', 'Control'], loc='lower right', fontsize=9)

# --- Fig C: DRACH heatmap - L1 ---
ax = axes[0, 2]
l1_m6a_all = results[('L1', 'm6A')]
l1_m6a_all = l1_m6a_all[l1_m6a_all['n_sites'] > 0]
drach_l1 = {}
for _, row in l1_m6a_all.iterrows():
    d = row['motif'][0]
    rh = row['motif'][1] + row['motif'][4]
    drach_l1[(d, rh)] = row['rate']

d_bases = ['A', 'G', 'T']
rh_combos = ['AA', 'AC', 'AT', 'GA', 'GC', 'GT']
matrix_l1 = np.zeros((len(d_bases), len(rh_combos)))
for i, d in enumerate(d_bases):
    for j, rh in enumerate(rh_combos):
        matrix_l1[i, j] = drach_l1.get((d, rh), 0)

vmin = min(matrix_l1.min(), 5)
vmax = max(matrix_l1.max(), 50)
im = ax.imshow(matrix_l1, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(rh_combos)))
ax.set_xticklabels([f'R={rh[0]}\nH={rh[1]}' for rh in rh_combos], fontsize=8)
ax.set_yticks(range(len(d_bases)))
ax.set_yticklabels([f'D={d}' for d in d_bases], fontsize=9)
ax.set_title('C. m6A DRACH rate (%) — L1', fontweight='bold')
for i in range(len(d_bases)):
    for j in range(len(rh_combos)):
        motif = f"{d_bases[i]}{rh_combos[j][0]}AC{rh_combos[j][1]}"
        val = matrix_l1[i, j]
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8,
                color='white' if val > 35 else 'black')
plt.colorbar(im, ax=ax, shrink=0.8, label='Rate (%)')

# --- Fig D: DRACH heatmap - Control ---
ax = axes[1, 0]
ctrl_m6a_all = results[('Control', 'm6A')]
ctrl_m6a_all = ctrl_m6a_all[ctrl_m6a_all['n_sites'] > 0]
drach_ctrl = {}
for _, row in ctrl_m6a_all.iterrows():
    d = row['motif'][0]
    rh = row['motif'][1] + row['motif'][4]
    drach_ctrl[(d, rh)] = row['rate']

matrix_ctrl = np.zeros((len(d_bases), len(rh_combos)))
for i, d in enumerate(d_bases):
    for j, rh in enumerate(rh_combos):
        matrix_ctrl[i, j] = drach_ctrl.get((d, rh), 0)

im2 = ax.imshow(matrix_ctrl, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(rh_combos)))
ax.set_xticklabels([f'R={rh[0]}\nH={rh[1]}' for rh in rh_combos], fontsize=8)
ax.set_yticks(range(len(d_bases)))
ax.set_yticklabels([f'D={d}' for d in d_bases], fontsize=9)
ax.set_title('D. m6A DRACH rate (%) — Control', fontweight='bold')
for i in range(len(d_bases)):
    for j in range(len(rh_combos)):
        val = matrix_ctrl[i, j]
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8,
                color='white' if val > 35 else 'black')
plt.colorbar(im2, ax=ax, shrink=0.8, label='Rate (%)')

# --- Fig E: L1 vs Control rate scatter ---
ax = axes[1, 1]
for mod, marker, motifs in [('m6A', 'o', M6A_MOTIFS), ('psi', 's', PSI_MOTIFS)]:
    l1_r = results[('L1', mod)].set_index('motif')
    ctrl_r = results[('Control', mod)].set_index('motif')
    for motif in motifs:
        if motif not in l1_r.index or motif not in ctrl_r.index:
            continue
        lv = l1_r.loc[motif, 'rate']
        cv = ctrl_r.loc[motif, 'rate']
        if np.isnan(lv) or np.isnan(cv):
            continue
        color = L1_COLOR if mod == 'm6A' else '#1565c0'
        ax.scatter(cv, lv, marker=marker, s=60, alpha=0.8, c=color, edgecolor='white', linewidth=0.5)
        ax.annotate(motif, (cv, lv), fontsize=5.5, fontfamily='monospace',
                    ha='center', va='bottom', alpha=0.8)

# Diagonal
maxval = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, maxval], [0, maxval], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('Control modification rate (%)')
ax.set_ylabel('L1 modification rate (%)')
ax.set_title('E. L1 vs Control per-motif rate', fontweight='bold')
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=L1_COLOR, markersize=8, label='m6A'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='#1565c0', markersize=8, label='psi')]
ax.legend(handles=legend_elements, fontsize=9)

# Correlation
m6a_comp = comp_df[comp_df['mod_type'] == 'm6A'].dropna()
psi_comp = comp_df[comp_df['mod_type'] == 'psi'].dropna()
if len(m6a_comp) > 2:
    r_m, p_m = stats.pearsonr(m6a_comp['ctrl_rate'], m6a_comp['l1_rate'])
    ax.text(0.05, 0.95, f'm6A: r={r_m:.3f}, p={p_m:.2e}', transform=ax.transAxes,
            fontsize=8, va='top', color=L1_COLOR)
if len(psi_comp) > 2:
    r_p, p_p = stats.pearsonr(psi_comp['ctrl_rate'], psi_comp['l1_rate'])
    ax.text(0.05, 0.88, f'psi: r={r_p:.3f}, p={p_p:.2e}', transform=ax.transAxes,
            fontsize=8, va='top', color='#1565c0')

# --- Fig F: Per-CL motif rate heatmap (psi, L1) ---
ax = axes[1, 2]
psi_cl = cl_motif_df[(cl_motif_df['source'] == 'L1') & (cl_motif_df['mod_type'] == 'psi')]
pivot = psi_cl.pivot_table(index='cell_line', columns='motif', values='rate')

if not pivot.empty:
    # Order cell lines and motifs
    cl_order = pivot.index.tolist()
    motif_order_hm = pivot.columns.tolist()

    matrix_hm = pivot.reindex(index=cl_order, columns=motif_order_hm).values
    im3 = ax.imshow(matrix_hm, cmap='YlOrRd', aspect='auto', vmin=0,
                    vmax=np.nanpercentile(matrix_hm, 95))
    ax.set_xticks(range(len(motif_order_hm)))
    ax.set_xticklabels(motif_order_hm, fontsize=6.5, fontfamily='monospace', rotation=45, ha='right')
    ax.set_yticks(range(len(cl_order)))
    ax.set_yticklabels(cl_order, fontsize=8)
    ax.set_title('F. Psi motif rate by cell line (L1)', fontweight='bold')
    plt.colorbar(im3, ax=ax, shrink=0.8, label='Rate (%)')

plt.tight_layout()
fig_path = OUTPUT_DIR / 'motif_enrichment_l1_vs_ctrl.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure saved: {fig_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for mod in ['m6A', 'psi']:
    l1_r = results[('L1', mod)]
    ctrl_r = results[('Control', mod)]
    l1_assessed = l1_r[l1_r['n_sites'] > 0]
    ctrl_assessed = ctrl_r[ctrl_r['n_sites'] > 0]

    print(f"\n{mod}:")
    print(f"  L1:      {len(l1_assessed)} motifs, rate range [{l1_assessed['rate'].min():.1f}%, {l1_assessed['rate'].max():.1f}%]")
    print(f"  Control: {len(ctrl_assessed)} motifs, rate range [{ctrl_assessed['rate'].min():.1f}%, {ctrl_assessed['rate'].max():.1f}%]")

    # Fisher's combined: overall L1 vs Control
    sub_l1 = l1[l1['name'] == mod]
    sub_ctrl = ctrl[ctrl['name'] == mod]
    sub_l1_mod = (sub_l1['modRatio'] > 50).sum()
    sub_ctrl_mod = (sub_ctrl['modRatio'] > 50).sum()
    OR, p = stats.fisher_exact([
        [sub_l1_mod, len(sub_l1) - sub_l1_mod],
        [sub_ctrl_mod, len(sub_ctrl) - sub_ctrl_mod]
    ])
    print(f"  Overall L1 vs Control: L1={100*sub_l1_mod/len(sub_l1):.1f}%, "
          f"Ctrl={100*sub_ctrl_mod/len(sub_ctrl):.1f}%, OR={OR:.2f}, p={p:.2e}")

    # Top/bottom motif per source
    for src, df_r in [('L1', l1_assessed), ('Control', ctrl_assessed)]:
        top = df_r.nlargest(3, 'rate')
        print(f"  {src} top 3: ", end='')
        for _, r in top.iterrows():
            print(f"{r['motif']}({r['rate']:.1f}%) ", end='')
        print()

# Top L1-enriched motifs (biggest L1 vs Ctrl delta)
print(f"\nMost L1-enriched motifs (biggest L1 > Control delta):")
for mod in ['m6A', 'psi']:
    sub = comp_df[(comp_df['mod_type'] == mod) & (comp_df['p_value'] < 0.05)].sort_values('delta', ascending=False)
    print(f"  {mod}:")
    for _, r in sub.head(5).iterrows():
        print(f"    {r['motif']}: L1={r['l1_rate']:.1f}%, Ctrl={r['ctrl_rate']:.1f}%, "
              f"Δ={r['delta']:+.1f}pp, OR={r['OR']:.2f}, p={r['p_value']:.2e}")

print("\nDone!")
