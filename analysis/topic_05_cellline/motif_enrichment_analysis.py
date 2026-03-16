#!/usr/bin/env python3
"""
L1 motif enrichment analysis for m6A and pseudouridine.

Questions:
1. Which MAFIA target motifs are most frequent in L1 regions?
2. Which motifs have higher-than-expected modification rates?
3. Which classifier motifs are NOT scanned (pipeline gap)?

Run with: conda run -n research python3 motif_enrichment_analysis.py
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from collections import Counter, defaultdict
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

# Actual MAFIA classifier motifs (from model directory)
M6A_CLASSIFIER_MOTIFS = sorted([
    'AAACA', 'AAACC', 'AAACT', 'AGACA', 'AGACC', 'AGACT',
    'GAACA', 'GAACC', 'GAACT', 'GGACA', 'GGACC', 'GGACT',
    'TAACA', 'TAACC', 'TAACT', 'TGACA', 'TGACC', 'TGACT',
])

PSI_CLASSIFIER_MOTIFS = sorted([
    'AGTGG', 'ATTTG', 'CATAA', 'CATCC', 'CCTCC', 'CTTTA',
    'GATGC', 'GGTCC', 'GGTGG', 'GTTCA', 'GTTCC', 'GTTCG',
    'GTTCT', 'TATAA', 'TGTAG', 'TGTGG',
])

ALL_CLASSIFIER_MOTIFS = {'m6A': set(M6A_CLASSIFIER_MOTIFS), 'psi': set(PSI_CLASSIFIER_MOTIFS)}

# ============================================================
# PART 1: Scan L1 regions for ALL classifier motifs
# ============================================================
print("=" * 60)
print("PART 1: Scanning L1 regions for ALL classifier motifs")
print("=" * 60)

# Collect unique L1 regions across groups
all_regions = set()
for group in GROUPS:
    regions_path = RESULTS_GROUP_DIR / group / "h_mafia" / f"{group}_regions.bed"
    if not regions_path.exists():
        continue
    with open(regions_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                strand = parts[3] if len(parts) >= 4 else '+'
                all_regions.add((chrom, start, end, strand))

print(f"Unique L1 regions: {len(all_regions):,}")

# Scan reference for motifs
ref_fasta = pysam.FastaFile(str(REF_FASTA))

m6a_ref_counts = Counter()
psi_ref_counts = Counter()
total_bp_scanned = 0

m6a_motif_set = set(M6A_CLASSIFIER_MOTIFS)
psi_motif_set = set(PSI_CLASSIFIER_MOTIFS)

for i, (chrom, start, end, strand) in enumerate(sorted(all_regions)):
    try:
        seq = ref_fasta.fetch(chrom, start, end).upper()
    except (ValueError, KeyError):
        continue
    total_bp_scanned += len(seq)

    # Scan forward strand
    for j in range(len(seq) - 4):
        kmer = seq[j:j+5]
        if kmer in m6a_motif_set:
            m6a_ref_counts[kmer] += 1
        if kmer in psi_motif_set:
            psi_ref_counts[kmer] += 1

    # Also scan reverse complement for - strand regions
    rc_map = str.maketrans('ACGT', 'TGCA')
    seq_rc = seq[::-1].translate(rc_map)
    for j in range(len(seq_rc) - 4):
        kmer = seq_rc[j:j+5]
        if kmer in m6a_motif_set:
            m6a_ref_counts[kmer] += 1
        if kmer in psi_motif_set:
            psi_ref_counts[kmer] += 1

    if (i + 1) % 5000 == 0:
        print(f"  Scanned {i+1:,}/{len(all_regions):,} regions...")

ref_fasta.close()

print(f"\nTotal bp scanned: {total_bp_scanned:,} (both strands)")
print(f"m6A total motif instances: {sum(m6a_ref_counts.values()):,}")
print(f"psi total motif instances: {sum(psi_ref_counts.values()):,}")

print(f"\nm6A motif frequency in L1 reference ({sum(m6a_ref_counts.values()):,} total):")
for motif in M6A_CLASSIFIER_MOTIFS:
    c = m6a_ref_counts[motif]
    total = sum(m6a_ref_counts.values())
    print(f"  {motif}: {c:>8,} ({100*c/total:.1f}%)")

print(f"\npsi motif frequency in L1 reference ({sum(psi_ref_counts.values()):,} total):")
for motif in PSI_CLASSIFIER_MOTIFS:
    c = psi_ref_counts[motif]
    total = sum(psi_ref_counts.values())
    print(f"  {motif}: {c:>8,} ({100*c/total:.1f}%)")

# ============================================================
# PART 2: Load pileup data and compute per-motif modification rates
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Per-motif modification rates from pileup")
print("=" * 60)

all_pileup = []
for group in GROUPS:
    pileup_path = RESULTS_GROUP_DIR / group / "h_mafia" / f"{group}.mAFiA.sites.bed"
    if not pileup_path.exists():
        continue
    df = pd.read_csv(pileup_path, sep='\t')
    df['group'] = group
    cell_line = group.rsplit('_', 1)[0]
    df['cell_line'] = cell_line
    all_pileup.append(df)

pileup_df = pd.concat(all_pileup, ignore_index=True)
print(f"Total pileup entries: {len(pileup_df):,}")

# Filter to covered sites
covered = pileup_df[pileup_df['coverage'] > 0].copy()
print(f"Covered sites: {len(covered):,}")

# Identify assessed vs unassessed motifs
assessed_m6a = sorted(covered[covered['name'] == 'm6A']['ref5mer'].unique())
assessed_psi = sorted(covered[covered['name'] == 'psi']['ref5mer'].unique())
print(f"\nAssessed m6A motifs ({len(assessed_m6a)}): {assessed_m6a}")
print(f"Assessed psi motifs ({len(assessed_psi)}): {assessed_psi}")

unassessed_psi = sorted(set(PSI_CLASSIFIER_MOTIFS) - set(assessed_psi))
print(f"\nUNASSESSED psi classifier motifs ({len(unassessed_psi)}): {unassessed_psi}")
print("  (These have classifiers but site generation didn't scan for them)")

# Show how common unassessed motifs are in L1
if unassessed_psi:
    print("\n  Unassessed psi motif frequency in L1 reference:")
    for motif in unassessed_psi:
        c = psi_ref_counts[motif]
        total = sum(psi_ref_counts.values())
        print(f"    {motif}: {c:>8,} ({100*c/total:.1f}%)")

# Per-motif modification analysis
results_all = []
for mod_type in ['m6A', 'psi']:
    sub = covered[covered['name'] == mod_type].copy()
    sub['is_modified'] = sub['modRatio'] > 50  # >50% of reads modified at site

    total_sites = len(sub)
    total_modified = sub['is_modified'].sum()
    overall_rate = 100 * total_modified / total_sites if total_sites > 0 else 0
    overall_weighted = np.average(sub['modRatio'], weights=sub['coverage']) if sub['coverage'].sum() > 0 else 0

    ref_counts = m6a_ref_counts if mod_type == 'm6A' else psi_ref_counts
    ref_total = sum(ref_counts.values())

    print(f"\n{mod_type}: {total_sites:,} covered sites, {total_modified:,} modified (rate={overall_rate:.1f}%), "
          f"weighted_modRatio={overall_weighted:.1f}%")

    motifs = M6A_CLASSIFIER_MOTIFS if mod_type == 'm6A' else PSI_CLASSIFIER_MOTIFS
    for motif in motifs:
        m = sub[sub['ref5mer'] == motif]
        n_sites = len(m)
        if n_sites == 0:
            # Motif exists in classifier but not in pileup → unassessed
            results_all.append({
                'mod_type': mod_type, 'motif': motif,
                'n_sites_assessed': 0, 'n_modified': 0,
                'rate': np.nan, 'weighted_rate': np.nan,
                'enrichment': np.nan,
                'ref_count': ref_counts.get(motif, 0),
                'ref_frac': ref_counts.get(motif, 0) / ref_total if ref_total > 0 else 0,
                'assessed': False,
            })
            continue

        n_modified = m['is_modified'].sum()
        rate = 100 * n_modified / n_sites
        weighted_rate = np.average(m['modRatio'], weights=m['coverage']) if m['coverage'].sum() > 0 else 0
        enrichment = rate / overall_rate if overall_rate > 0 else 0

        results_all.append({
            'mod_type': mod_type, 'motif': motif,
            'n_sites_assessed': n_sites, 'n_modified': n_modified,
            'rate': rate, 'weighted_rate': weighted_rate,
            'enrichment': enrichment,
            'ref_count': ref_counts.get(motif, 0),
            'ref_frac': ref_counts.get(motif, 0) / ref_total if ref_total > 0 else 0,
            'assessed': True,
        })

        print(f"  {motif}: assessed={n_sites:>7,}  mod={n_modified:>6,}  "
              f"rate={rate:5.1f}%  weighted={weighted_rate:5.1f}%  "
              f"enrich={enrichment:.2f}x  "
              f"ref_freq={100*ref_counts.get(motif,0)/ref_total:.1f}%")

results_df = pd.DataFrame(results_all)
results_df.to_csv(OUTPUT_DIR / 'motif_enrichment.tsv', sep='\t', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'motif_enrichment.tsv'}")

# ============================================================
# PART 3: Statistical tests
# ============================================================
print("\n" + "=" * 60)
print("PART 3: Statistical tests")
print("=" * 60)

for mod_type in ['m6A', 'psi']:
    sub_results = results_df[(results_df['mod_type'] == mod_type) & results_df['assessed']]
    if len(sub_results) < 2:
        print(f"\n{mod_type}: too few assessed motifs for test")
        continue

    # Chi-square: are modification rates different across motifs?
    obs = sub_results['n_modified'].values
    total = sub_results['n_sites_assessed'].values
    not_mod = total - obs

    contingency = np.array([obs, not_mod])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n{mod_type} chi-square (are rates different across motifs?):")
    print(f"  chi2={chi2:.1f}, p={p:.2e}, dof={dof}")

    # Also test: is motif frequency among modified sites different from reference frequency?
    # Expected from reference frequency
    ref_counts = m6a_ref_counts if mod_type == 'm6A' else psi_ref_counts
    assessed_motifs = sub_results['motif'].values
    ref_freq = np.array([ref_counts[m] for m in assessed_motifs], dtype=float)
    ref_freq = ref_freq / ref_freq.sum()  # normalize
    observed_modified = sub_results['n_modified'].values
    expected_modified = ref_freq * observed_modified.sum()

    chi2_freq, p_freq = stats.chisquare(observed_modified, expected_modified)
    print(f"\n{mod_type} chi-square (observed modified vs expected from L1 motif frequency):")
    print(f"  chi2={chi2_freq:.1f}, p={p_freq:.2e}")

    # Top/bottom enriched
    top = sub_results.nlargest(3, 'rate')
    bot = sub_results.nsmallest(3, 'rate')
    print(f"\n  Top 3 enriched: ", end='')
    for _, r in top.iterrows():
        print(f"{r['motif']}({r['rate']:.1f}%, {r['enrichment']:.2f}x) ", end='')
    print(f"\n  Bottom 3: ", end='')
    for _, r in bot.iterrows():
        print(f"{r['motif']}({r['rate']:.1f}%, {r['enrichment']:.2f}x) ", end='')
    print()

# ============================================================
# PART 4: DRACH decomposition (m6A)
# ============================================================
print("\n" + "=" * 60)
print("PART 4: DRACH position decomposition (m6A)")
print("=" * 60)

m6a_assessed = results_df[(results_df['mod_type'] == 'm6A') & results_df['assessed']].copy()

for pos, label, bases in [(0, 'D (pos1)', 'AGT'), (1, 'R (pos2)', 'AG'), (4, 'H (pos5)', 'ACT')]:
    m6a_assessed[f'pos_{pos}'] = m6a_assessed['motif'].str[pos]
    grp = m6a_assessed.groupby(f'pos_{pos}').agg(
        n_sites=('n_sites_assessed', 'sum'),
        n_modified=('n_modified', 'sum'),
        ref_count=('ref_count', 'sum'),
    )
    grp['rate'] = 100 * grp['n_modified'] / grp['n_sites']
    print(f"\n{label} ({'/'.join(bases)}):")
    for base, row in grp.iterrows():
        dna_base = base
        rna_base = {'A': 'A', 'G': 'G', 'T': 'U', 'C': 'C'}[base]
        print(f"  {dna_base} (RNA:{rna_base}): sites={row['n_sites']:>8,.0f}  "
              f"modified={row['n_modified']:>6,.0f}  rate={row['rate']:.1f}%  "
              f"ref_count={row['ref_count']:>8,.0f}")

# ============================================================
# PART 5: Cell-line level motif analysis
# ============================================================
print("\n" + "=" * 60)
print("PART 5: Per-cell-line motif variation")
print("=" * 60)

for mod_type in ['m6A', 'psi']:
    sub = covered[covered['name'] == mod_type].copy()
    sub['is_modified'] = sub['modRatio'] > 50

    print(f"\n{mod_type} per-cell-line top motif rates:")
    for cl in sorted(sub['cell_line'].unique()):
        cl_sub = sub[sub['cell_line'] == cl]
        total = len(cl_sub)
        if total < 100:
            continue
        modified = cl_sub['is_modified'].sum()
        overall = 100 * modified / total

        # Top motif
        motif_rates = {}
        for motif in cl_sub['ref5mer'].unique():
            m = cl_sub[cl_sub['ref5mer'] == motif]
            if len(m) >= 10:
                motif_rates[motif] = 100 * m['is_modified'].sum() / len(m)

        if motif_rates:
            top_motif = max(motif_rates, key=motif_rates.get)
            bot_motif = min(motif_rates, key=motif_rates.get)
            print(f"  {cl:>10}: overall={overall:5.1f}%  "
                  f"top={top_motif}({motif_rates[top_motif]:.1f}%)  "
                  f"bot={bot_motif}({motif_rates[bot_motif]:.1f}%)")

# ============================================================
# PART 6: Figures
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(24, 14))

# --- Plot 1: m6A motif modification rates ---
ax = axes[0, 0]
m6a_res = results_df[(results_df['mod_type'] == 'm6A') & results_df['assessed']].sort_values('rate', ascending=True)
colors = ['#d32f2f' if e > 1.15 else '#1565c0' if e < 0.85 else '#616161'
          for e in m6a_res['enrichment']]
bars = ax.barh(range(len(m6a_res)), m6a_res['rate'], color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(m6a_res)))
ax.set_yticklabels(m6a_res['motif'], fontsize=9, fontfamily='monospace')
ax.set_xlabel('Modification rate (%)')
ax.set_title('m6A: per-DRACH-motif modification rate')
mean_rate = m6a_res['rate'].mean()
ax.axvline(mean_rate, color='gray', ls='--', alpha=0.7, label=f'mean={mean_rate:.1f}%')
ax.legend(fontsize=8)

# --- Plot 2: psi motif modification rates (assessed only) ---
ax = axes[0, 1]
psi_res = results_df[(results_df['mod_type'] == 'psi') & results_df['assessed']].sort_values('rate', ascending=True)
colors = ['#d32f2f' if e > 1.15 else '#1565c0' if e < 0.85 else '#616161'
          for e in psi_res['enrichment']]
bars = ax.barh(range(len(psi_res)), psi_res['rate'], color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(psi_res)))
ax.set_yticklabels(psi_res['motif'], fontsize=9, fontfamily='monospace')
ax.set_xlabel('Modification rate (%)')
ax.set_title('Psi: per-motif modification rate (assessed only)')
mean_rate = psi_res['rate'].mean()
ax.axvline(mean_rate, color='gray', ls='--', alpha=0.7, label=f'mean={mean_rate:.1f}%')
ax.legend(fontsize=8)

# --- Plot 3: Reference motif frequency (all classifier motifs) ---
ax = axes[0, 2]
# Combine m6A and psi reference counts
all_motifs = []
for motif in M6A_CLASSIFIER_MOTIFS:
    all_motifs.append(('m6A', motif, m6a_ref_counts[motif]))
for motif in PSI_CLASSIFIER_MOTIFS:
    all_motifs.append(('psi', motif, psi_ref_counts[motif]))

# Show psi motifs colored by assessed/unassessed
psi_data = [(m, c) for t, m, c in all_motifs if t == 'psi']
psi_data.sort(key=lambda x: x[1])
psi_motifs_sorted = [m for m, c in psi_data]
psi_counts_sorted = [c for m, c in psi_data]
psi_colors = ['#1565c0' if m in set(assessed_psi) else '#ffab91' for m in psi_motifs_sorted]

ax.barh(range(len(psi_data)), psi_counts_sorted, color=psi_colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(psi_data)))
ax.set_yticklabels(psi_motifs_sorted, fontsize=8, fontfamily='monospace')
ax.set_xlabel('Count in L1 reference regions')
ax.set_title('Psi motif frequency in L1\n(blue=assessed, orange=unassessed)')

# --- Plot 4: m6A frequency vs rate ---
ax = axes[1, 0]
m6a_all = results_df[(results_df['mod_type'] == 'm6A') & results_df['assessed']]
ax.scatter(m6a_all['ref_count'], m6a_all['rate'], s=80, alpha=0.8, c='#d32f2f', edgecolor='white')
for _, row in m6a_all.iterrows():
    ax.annotate(row['motif'], (row['ref_count'], row['rate']),
                fontsize=7, fontfamily='monospace', ha='center', va='bottom')
ax.set_xlabel('Motif count in L1 reference')
ax.set_ylabel('Modification rate (%)')
ax.set_title('m6A: L1 frequency vs modification rate')
r, p = stats.spearmanr(m6a_all['ref_count'], m6a_all['rate'])
ax.text(0.05, 0.95, f'Spearman r={r:.3f}, p={p:.3f}', transform=ax.transAxes, fontsize=9, va='top')

# --- Plot 5: psi frequency vs rate ---
ax = axes[1, 1]
psi_assessed = results_df[(results_df['mod_type'] == 'psi') & results_df['assessed']]
ax.scatter(psi_assessed['ref_count'], psi_assessed['rate'], s=80, alpha=0.8, c='#1565c0', edgecolor='white')
for _, row in psi_assessed.iterrows():
    ax.annotate(row['motif'], (row['ref_count'], row['rate']),
                fontsize=8, fontfamily='monospace', ha='center', va='bottom')
ax.set_xlabel('Motif count in L1 reference')
ax.set_ylabel('Modification rate (%)')
ax.set_title('Psi: L1 frequency vs modification rate')
if len(psi_assessed) > 2:
    r, p = stats.spearmanr(psi_assessed['ref_count'], psi_assessed['rate'])
    ax.text(0.05, 0.95, f'Spearman r={r:.3f}, p={p:.3f}', transform=ax.transAxes, fontsize=9, va='top')

# --- Plot 6: Enrichment heatmap ---
ax = axes[1, 2]
# DRACH heatmap: D (rows) x R*H (columns)
drach_matrix = {}
for _, row in m6a_all.iterrows():
    d = row['motif'][0]
    rh = row['motif'][1] + row['motif'][4]  # R + H
    drach_matrix[(d, rh)] = row['enrichment']

d_bases = ['A', 'G', 'T']
rh_combos = ['AA', 'AC', 'AT', 'GA', 'GC', 'GT']
matrix = np.zeros((len(d_bases), len(rh_combos)))
for i, d in enumerate(d_bases):
    for j, rh in enumerate(rh_combos):
        matrix[i, j] = drach_matrix.get((d, rh), 0)

im = ax.imshow(matrix, cmap='RdBu_r', vmin=0.7, vmax=1.3, aspect='auto')
ax.set_xticks(range(len(rh_combos)))
ax.set_xticklabels([f'R={rh[0]}\nH={rh[1]}' for rh in rh_combos], fontsize=8)
ax.set_yticks(range(len(d_bases)))
ax.set_yticklabels([f'D={d}' for d in d_bases], fontsize=9)
ax.set_title('m6A DRACH enrichment\n(red=enriched, blue=depleted)')

for i in range(len(d_bases)):
    for j in range(len(rh_combos)):
        motif = f"{d_bases[i]}{rh_combos[j][0]}AC{rh_combos[j][1]}"
        ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(matrix[i,j] - 1) > 0.15 else 'black')

plt.colorbar(im, ax=ax, shrink=0.8, label='Enrichment')

plt.tight_layout()
fig_path = OUTPUT_DIR / 'motif_enrichment.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fig_path}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

m6a_assessed = results_df[(results_df['mod_type'] == 'm6A') & results_df['assessed']]
psi_assessed_df = results_df[(results_df['mod_type'] == 'psi') & results_df['assessed']]

print(f"\nm6A: {len(m6a_assessed)} motifs assessed")
print(f"  Rate range: {m6a_assessed['rate'].min():.1f}% - {m6a_assessed['rate'].max():.1f}%")
print(f"  Enrichment range: {m6a_assessed['enrichment'].min():.2f}x - {m6a_assessed['enrichment'].max():.2f}x")

print(f"\nPsi: {len(psi_assessed_df)} motifs assessed (of {len(PSI_CLASSIFIER_MOTIFS)} classifiers)")
if len(psi_assessed_df) > 0:
    print(f"  Rate range: {psi_assessed_df['rate'].min():.1f}% - {psi_assessed_df['rate'].max():.1f}%")
    print(f"  Enrichment range: {psi_assessed_df['enrichment'].min():.2f}x - {psi_assessed_df['enrichment'].max():.2f}x")

print(f"\nUnassessed psi motifs: {len(unassessed_psi)}")
for motif in unassessed_psi:
    c = psi_ref_counts[motif]
    total = sum(psi_ref_counts.values())
    print(f"  {motif}: {c:,} in L1 ({100*c/total:.1f}%)")

print("\nDone!")
