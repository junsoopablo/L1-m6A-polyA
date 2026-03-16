#!/usr/bin/env python3
"""
Two analyses:
  (1) m6A/psi sites/kb per cell line (quantitative, not binary)
  (2) psi/m6A site positional distribution along L1 body
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
MAFIA_DIR = PROJECT / 'analysis/01_exploration/topic_03_m6a_psi'
OUT_DIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
MIN_READS = 200

CELL_LINES = {
    'A549':     ['A549_4', 'A549_5', 'A549_6'],
    'H9':       ['H9_2', 'H9_3', 'H9_4'],
    'Hct116':   ['Hct116_3', 'Hct116_4'],
    'HeLa':     ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2':    ['HepG2_5', 'HepG2_6'],
    'HEYA8':    ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562':     ['K562_4', 'K562_5', 'K562_6'],
    'MCF7':     ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'MCF7-EV':  ['MCF7-EV_1'],
    'SHSY5Y':   ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

# =========================================================================
# Load all per-read MAFIA data + merge with L1 summary
# =========================================================================
print("Loading and merging per-read MAFIA data with L1 summaries...")

all_reads = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        # Load MAFIA per-read
        mafia_path = MAFIA_DIR / f'{g}_mafia_per_read.tsv'
        if not mafia_path.exists():
            print(f"  MISSING: {mafia_path.name}")
            continue
        mdf = pd.read_csv(mafia_path, sep='\t')

        # Load L1 summary
        l1_path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        l1 = pd.read_csv(l1_path, sep='\t')
        l1 = l1[l1['qc_tag'] == 'PASS']

        if len(l1) < MIN_READS:
            print(f"  SKIP {g}: {len(l1)} reads < {MIN_READS}")
            continue

        # Merge on read_id
        merged = mdf.merge(l1[['read_id', 'gene_id', 'transcript_id']],
                           on='read_id', how='inner')
        merged['cell_line'] = cl
        merged['group'] = g
        merged['l1_age'] = merged['gene_id'].apply(
            lambda x: 'young' if x in YOUNG else 'ancient')

        # Compute sites/kb
        merged['m6a_per_kb'] = merged['m6a_sites_high'] / (merged['read_length'] / 1000)
        merged['psi_per_kb'] = merged['psi_sites_high'] / (merged['read_length'] / 1000)

        all_reads.append(merged)
        print(f"  {g}: {len(merged):,} reads merged")

df = pd.concat(all_reads, ignore_index=True)
print(f"\nTotal: {len(df):,} reads")

# =========================================================================
# PART 1: sites/kb per cell line
# =========================================================================
print(f"\n{'='*100}")
print("PART 1: m6A/psi Sites per kb by Cell Line")
print(f"{'='*100}")

# All L1
print(f"\n--- ALL L1 ---")
print(f"{'Cell Line':<12} {'N':>6} {'m6A/kb med':>10} {'m6A/kb mean':>11} "
      f"{'psi/kb med':>10} {'psi/kb mean':>11} {'rdLen med':>9}")
print("-" * 80)

summary_rows = []
for cl in CELL_LINES:
    d = df[df['cell_line'] == cl]
    if len(d) < 10:
        continue
    print(f"{cl:<12} {len(d):>6,} {d['m6a_per_kb'].median():>10.2f} "
          f"{d['m6a_per_kb'].mean():>11.2f} "
          f"{d['psi_per_kb'].median():>10.2f} {d['psi_per_kb'].mean():>11.2f} "
          f"{d['read_length'].median():>9.0f}")
    summary_rows.append({
        'cell_line': cl, 'l1_age': 'all', 'n': len(d),
        'm6a_per_kb_median': d['m6a_per_kb'].median(),
        'm6a_per_kb_mean': d['m6a_per_kb'].mean(),
        'psi_per_kb_median': d['psi_per_kb'].median(),
        'psi_per_kb_mean': d['psi_per_kb'].mean(),
        'rdlen_median': d['read_length'].median(),
    })

# Ancient L1
for age in ['ancient', 'young']:
    print(f"\n--- {age.upper()} L1 ---")
    print(f"{'Cell Line':<12} {'N':>6} {'m6A/kb med':>10} {'m6A/kb mean':>11} "
          f"{'psi/kb med':>10} {'psi/kb mean':>11}")
    print("-" * 65)
    for cl in CELL_LINES:
        d = df[(df['cell_line'] == cl) & (df['l1_age'] == age)]
        if len(d) < 10:
            continue
        print(f"{cl:<12} {len(d):>6,} {d['m6a_per_kb'].median():>10.2f} "
              f"{d['m6a_per_kb'].mean():>11.2f} "
              f"{d['psi_per_kb'].median():>10.2f} {d['psi_per_kb'].mean():>11.2f}")
        summary_rows.append({
            'cell_line': cl, 'l1_age': age, 'n': len(d),
            'm6a_per_kb_median': d['m6a_per_kb'].median(),
            'm6a_per_kb_mean': d['m6a_per_kb'].mean(),
            'psi_per_kb_median': d['psi_per_kb'].median(),
            'psi_per_kb_mean': d['psi_per_kb'].mean(),
            'rdlen_median': d['read_length'].median(),
        })

# KW test across base cell lines
print(f"\n--- Kruskal-Wallis (base CL, ancient) ---")
base_cls = [cl for cl in CELL_LINES if cl not in ('HeLa-Ars', 'MCF7-EV')]
groups_m6a = []
groups_psi = []
for cl in base_cls:
    d = df[(df['cell_line'] == cl) & (df['l1_age'] == 'ancient')]
    if len(d) >= 10:
        groups_m6a.append(d['m6a_per_kb'].values)
        groups_psi.append(d['psi_per_kb'].values)

kw_m6a = stats.kruskal(*groups_m6a)
kw_psi = stats.kruskal(*groups_psi)
sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
print(f"  m6A/kb: H={kw_m6a.statistic:.1f}, p={kw_m6a.pvalue:.2e} ({sig(kw_m6a.pvalue)})")
print(f"  psi/kb: H={kw_psi.statistic:.1f}, p={kw_psi.pvalue:.2e} ({sig(kw_psi.pvalue)})")

# Cross-CL correlation (sites/kb)
print(f"\n--- Cross-CL Correlation (ancient, median sites/kb) ---")
cl_m6a = []
cl_psi = []
cl_names = []
for cl in base_cls:
    d = df[(df['cell_line'] == cl) & (df['l1_age'] == 'ancient')]
    if len(d) >= 10:
        cl_m6a.append(d['m6a_per_kb'].median())
        cl_psi.append(d['psi_per_kb'].median())
        cl_names.append(cl)

r, p = stats.spearmanr(cl_m6a, cl_psi)
print(f"  m6A/kb vs psi/kb: Spearman r={r:.3f}, p={p:.4f}")
for i, cl in enumerate(cl_names):
    print(f"    {cl:<12} m6A/kb={cl_m6a[i]:.2f}  psi/kb={cl_psi[i]:.2f}")

# =========================================================================
# PART 2: Positional distribution of psi/m6A sites along L1 body
# =========================================================================
print(f"\n{'='*100}")
print("PART 2: Modification Site Positional Distribution Along L1 Body")
print(f"{'='*100}")

# Positions are read-relative (0 = start of read, read_length = end)
# DRS reads from 3' end, so position 0 = 3' end, read_length = toward 5'
# We normalize to fractional position: pos / read_length (0=3' end, 1=5' end)

def safe_parse_list(val):
    """Parse string representation of list."""
    if pd.isna(val) or val == '[]':
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        return []

print("\nParsing modification positions...")
all_psi_positions = []
all_m6a_positions = []

for _, row in df.iterrows():
    rl = row['read_length']
    if rl < 100:
        continue

    psi_pos = safe_parse_list(row.get('psi_positions', []))
    m6a_pos = safe_parse_list(row.get('m6a_positions', []))
    psi_probs = safe_parse_list(row.get('psi_probs', []))
    m6a_probs = safe_parse_list(row.get('m6a_probs', []))

    # High-confidence psi sites
    for i, pos in enumerate(psi_pos):
        if i < len(psi_probs) and psi_probs[i] >= 128:
            frac = pos / rl  # 0=3' end, 1=5' end
            if 0 <= frac <= 1:
                all_psi_positions.append({
                    'frac_pos': frac, 'cell_line': row['cell_line'],
                    'l1_age': row['l1_age'], 'mod': 'psi',
                    'read_length': rl,
                })

    # High-confidence m6A sites
    for i, pos in enumerate(m6a_pos):
        if i < len(m6a_probs) and m6a_probs[i] >= 128:
            frac = pos / rl
            if 0 <= frac <= 1:
                all_m6a_positions.append({
                    'frac_pos': frac, 'cell_line': row['cell_line'],
                    'l1_age': row['l1_age'], 'mod': 'm6A',
                    'read_length': rl,
                })

psi_pos_df = pd.DataFrame(all_psi_positions)
m6a_pos_df = pd.DataFrame(all_m6a_positions)
pos_df = pd.concat([psi_pos_df, m6a_pos_df], ignore_index=True)

print(f"  Total psi sites: {len(psi_pos_df):,}")
print(f"  Total m6A sites: {len(m6a_pos_df):,}")

# Summary stats
for mod, mdf in [('psi', psi_pos_df), ('m6A', m6a_pos_df)]:
    print(f"\n  {mod} positional stats (0=3' end, 1=5' end):")
    print(f"    Mean position: {mdf['frac_pos'].mean():.3f}")
    print(f"    Median position: {mdf['frac_pos'].median():.3f}")
    print(f"    % in 3' quarter (0-0.25): {(mdf['frac_pos'] <= 0.25).mean()*100:.1f}%")
    print(f"    % in middle (0.25-0.75): {((mdf['frac_pos'] > 0.25) & (mdf['frac_pos'] <= 0.75)).mean()*100:.1f}%")
    print(f"    % in 5' quarter (0.75-1.0): {(mdf['frac_pos'] > 0.75).mean()*100:.1f}%")

# Per cell line
print(f"\n  Positional mean by cell line (psi):")
for cl in CELL_LINES:
    d = psi_pos_df[psi_pos_df['cell_line'] == cl]
    if len(d) >= 10:
        print(f"    {cl:<12} n={len(d):>5}  mean_pos={d['frac_pos'].mean():.3f}  "
              f"3'quarter={((d['frac_pos']<=0.25).mean()*100):.1f}%")

# Ancient vs Young
print(f"\n  Ancient vs Young positional comparison:")
for mod, mdf in [('psi', psi_pos_df), ('m6A', m6a_pos_df)]:
    anc = mdf[mdf['l1_age'] == 'ancient']['frac_pos']
    yng = mdf[mdf['l1_age'] == 'young']['frac_pos']
    if len(yng) >= 10:
        _, p = stats.mannwhitneyu(anc, yng, alternative='two-sided')
        print(f"    {mod}: ancient mean={anc.mean():.3f} (n={len(anc):,}), "
              f"young mean={yng.mean():.3f} (n={len(yng):,}), MW p={p:.2e}")

# =========================================================================
# PLOTS
# =========================================================================

# --- Plot 1: sites/kb boxplot by cell line ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

cl_order = sorted(
    [cl for cl in CELL_LINES if cl not in ('HeLa-Ars', 'MCF7-EV')],
    key=lambda cl: df[(df['cell_line'] == cl) & (df['l1_age'] == 'ancient')]['psi_per_kb'].median()
)
# Add treatment variants at end
cl_order_full = cl_order + ['HeLa-Ars', 'MCF7-EV']

for ax, mod, mod_col in [(ax1, 'psi', 'psi_per_kb'), (ax2, 'm6A', 'm6a_per_kb')]:
    data_boxes = []
    labels = []
    for cl in cl_order_full:
        d = df[(df['cell_line'] == cl) & (df['l1_age'] == 'ancient')]
        if len(d) >= 10:
            data_boxes.append(d[mod_col].values)
            labels.append(cl)

    bp = ax.boxplot(data_boxes, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    for i, patch in enumerate(bp['boxes']):
        if labels[i] == 'HeLa-Ars':
            patch.set_facecolor('#ff6666')
        elif labels[i] == 'MCF7-EV':
            patch.set_facecolor('#ff66ff')
        else:
            patch.set_facecolor('#66b3ff')
        patch.set_alpha(0.7)

    ax.set_ylabel(f'{mod} sites/kb')
    ax.set_title(f'{mod} Density (Ancient L1)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / 'sites_per_kb_boxplot.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR / 'sites_per_kb_boxplot.png'}")

# --- Plot 2: Positional distribution ---
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

bins = np.linspace(0, 1, 21)  # 20 bins

# (a) Overall psi vs m6A distribution
ax = axes[0, 0]
ax.hist(psi_pos_df['frac_pos'], bins=bins, alpha=0.6, density=True, label='psi', color='#d62728')
ax.hist(m6a_pos_df['frac_pos'], bins=bins, alpha=0.6, density=True, label='m6A', color='#1f77b4')
ax.set_xlabel('Fractional position (0=3\' end, 1=5\' end)')
ax.set_ylabel('Density')
ax.set_title('All Cell Lines Pooled')
ax.legend()
ax.axvline(0.5, color='grey', linestyle='--', alpha=0.3)

# (b) psi by cell line
ax = axes[0, 1]
for cl in ['HeLa', 'MCF7', 'K562', 'HepG2', 'H9']:
    d = psi_pos_df[psi_pos_df['cell_line'] == cl]
    if len(d) >= 50:
        ax.hist(d['frac_pos'], bins=bins, alpha=0.4, density=True, label=cl, histtype='step', linewidth=2)
ax.set_xlabel('Fractional position (0=3\' end, 1=5\' end)')
ax.set_ylabel('Density')
ax.set_title('psi: Per Cell Line')
ax.legend(fontsize=8)

# (c) Ancient vs Young psi
ax = axes[1, 0]
anc_psi = psi_pos_df[psi_pos_df['l1_age'] == 'ancient']
yng_psi = psi_pos_df[psi_pos_df['l1_age'] == 'young']
ax.hist(anc_psi['frac_pos'], bins=bins, alpha=0.6, density=True, label=f'Ancient (n={len(anc_psi):,})', color='#8c564b')
if len(yng_psi) >= 20:
    ax.hist(yng_psi['frac_pos'], bins=bins, alpha=0.6, density=True, label=f'Young (n={len(yng_psi):,})', color='#2ca02c')
ax.set_xlabel('Fractional position (0=3\' end, 1=5\' end)')
ax.set_ylabel('Density')
ax.set_title('psi: Ancient vs Young L1')
ax.legend()

# (d) Cumulative distribution: psi vs m6A vs uniform
ax = axes[1, 1]
psi_sorted = np.sort(psi_pos_df['frac_pos'].values)
m6a_sorted = np.sort(m6a_pos_df['frac_pos'].values)
ax.plot(psi_sorted, np.linspace(0, 1, len(psi_sorted)), label='psi', color='#d62728')
ax.plot(m6a_sorted, np.linspace(0, 1, len(m6a_sorted)), label='m6A', color='#1f77b4')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Uniform')
ax.set_xlabel('Fractional position (0=3\' end, 1=5\' end)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('CDF: psi vs m6A vs Uniform')
ax.legend()

plt.tight_layout()
fig2.savefig(OUT_DIR / 'modification_position_distribution.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'modification_position_distribution.png'}")

# --- Plot 3: sites/kb heatmap ---
fig3, ax3 = plt.subplots(figsize=(8, 6))
hm_data = []
for cl in cl_order_full:
    d = df[(df['cell_line'] == cl) & (df['l1_age'] == 'ancient')]
    if len(d) >= 10:
        hm_data.append({
            'cell_line': cl,
            'm6a_per_kb': d['m6a_per_kb'].median(),
            'psi_per_kb': d['psi_per_kb'].median(),
        })
hm_df = pd.DataFrame(hm_data).set_index('cell_line')
hm_z = (hm_df - hm_df.mean()) / hm_df.std()

im = ax3.imshow(hm_z.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax3.set_yticks(range(len(hm_z)))
ax3.set_yticklabels(hm_z.index, fontsize=10)
ax3.set_xticks(range(len(hm_z.columns)))
ax3.set_xticklabels(['m6A/kb', 'psi/kb'], fontsize=10)
plt.colorbar(im, ax=ax3, label='Z-score', shrink=0.8)
ax3.set_title('Modification Density by Cell Line (Ancient L1, Z-scored)')

for i, cl in enumerate(hm_z.index):
    if cl == 'HeLa-Ars':
        ax3.get_yticklabels()[i].set_color('red')
        ax3.get_yticklabels()[i].set_fontweight('bold')
    elif cl == 'MCF7-EV':
        ax3.get_yticklabels()[i].set_color('magenta')
        ax3.get_yticklabels()[i].set_fontweight('bold')

plt.tight_layout()
fig3.savefig(OUT_DIR / 'sites_per_kb_heatmap.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'sites_per_kb_heatmap.png'}")

# Save summary
pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'sites_per_kb_summary.tsv', sep='\t', index=False)
print(f"Saved: {OUT_DIR / 'sites_per_kb_summary.tsv'}")

# KS test vs uniform for positional distribution
print(f"\n--- KS test vs Uniform distribution ---")
for mod, mdf in [('psi', psi_pos_df), ('m6A', m6a_pos_df)]:
    ks_stat, ks_p = stats.kstest(mdf['frac_pos'], 'uniform')
    print(f"  {mod}: KS stat={ks_stat:.4f}, p={ks_p:.2e} ({sig(ks_p)})")

print("\nDone!")
