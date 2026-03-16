#!/usr/bin/env python3
"""
Per-read psi DENSITY (sites/kb) comparison: L1 body vs flanking.

For each L1 read that extends beyond the TE boundary:
  - Split psi events into L1-body vs flanking based on genomic overlap
  - Compute psi/kb for each region
  - Paired within-read comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import ast

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPICDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'

# Groups
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

# =========================================================================
# Load and merge data
# =========================================================================
print("Loading data...")

all_reads = []
for group in ALL_GROUPS:
    # Per-read cache (psi positions)
    cache_file = TOPICDIR / f'part3_l1_per_read_cache/{group}_l1_per_read.tsv'
    if not cache_file.exists():
        continue
    cache = pd.read_csv(cache_file, sep='\t')
    
    # L1 summary (genomic coordinates + TE coordinates)
    summary_file = PROJECT / f'results_group/{group}/g_summary/{group}_L1_summary.tsv'
    if not summary_file.exists():
        continue
    summary = pd.read_csv(summary_file, sep='\t')
    
    # Merge on read_id
    merged = cache.merge(
        summary[['read_id','chr','start','end','te_start','te_end','read_strand','TE_group']],
        on='read_id', how='inner'
    )
    merged['group'] = group
    merged['cell_line'] = GROUP_TO_CL[group]
    all_reads.append(merged)

df = pd.concat(all_reads, ignore_index=True)
print(f"Total merged reads: {len(df)}")

# =========================================================================
# Compute L1 body vs flanking psi density per read
# =========================================================================
print("Computing per-read L1 body vs flanking psi density...")

results = []
for idx, row in df.iterrows():
    read_start = row['start']
    read_end = row['end']
    te_start = row['te_start']
    te_end = row['te_end']
    strand = row['read_strand']
    
    # L1 body overlap in genomic coords
    body_start = max(read_start, te_start)
    body_end = min(read_end, te_end)
    body_bp = max(0, body_end - body_start)
    
    # Flanking = read extent outside L1 TE
    flank_bp = (read_end - read_start) - body_bp
    
    # Need both regions with minimum 100bp to be meaningful
    if body_bp < 100 or flank_bp < 100:
        continue
    
    # Parse psi positions (read-relative)
    psi_pos_str = row['psi_positions']
    if pd.isna(psi_pos_str) or psi_pos_str == '[]':
        psi_positions = []
    else:
        psi_positions = ast.literal_eval(str(psi_pos_str))
    
    # Also parse m6a
    m6a_pos_str = row['m6a_positions']
    if pd.isna(m6a_pos_str) or m6a_pos_str == '[]':
        m6a_positions = []
    else:
        m6a_positions = ast.literal_eval(str(m6a_pos_str))
    
    # Convert to genomic and classify
    psi_body = 0
    psi_flank = 0
    for p in psi_positions:
        if strand == '+':
            gpos = read_start + p
        else:
            gpos = read_end - p
        
        if te_start <= gpos <= te_end:
            psi_body += 1
        else:
            psi_flank += 1
    
    m6a_body = 0
    m6a_flank = 0
    for p in m6a_positions:
        if strand == '+':
            gpos = read_start + p
        else:
            gpos = read_end - p
        
        if te_start <= gpos <= te_end:
            m6a_body += 1
        else:
            m6a_flank += 1
    
    results.append({
        'read_id': row['read_id'],
        'cell_line': row['cell_line'],
        'TE_group': row['TE_group'],
        'body_bp': body_bp,
        'flank_bp': flank_bp,
        'psi_body': psi_body,
        'psi_flank': psi_flank,
        'psi_body_per_kb': psi_body / (body_bp / 1000),
        'psi_flank_per_kb': psi_flank / (flank_bp / 1000),
        'm6a_body': m6a_body,
        'm6a_flank': m6a_flank,
        'm6a_body_per_kb': m6a_body / (body_bp / 1000),
        'm6a_flank_per_kb': m6a_flank / (flank_bp / 1000),
    })

res = pd.DataFrame(results)
print(f"Reads with both L1 body ≥100bp AND flanking ≥100bp: {len(res)}")

# =========================================================================
# Results
# =========================================================================
print("\n" + "="*70)
print("PSI DENSITY: L1 BODY vs FLANKING (per-read, sites/kb)")
print("="*70)

# Overall
for mod, body_col, flank_col in [('psi', 'psi_body_per_kb', 'psi_flank_per_kb'),
                                   ('m6A', 'm6a_body_per_kb', 'm6a_flank_per_kb')]:
    body_vals = res[body_col]
    flank_vals = res[flank_col]
    
    # Pooled: total sites / total kb
    if mod == 'psi':
        total_body_sites = res['psi_body'].sum()
        total_flank_sites = res['psi_flank'].sum()
    else:
        total_body_sites = res['m6a_body'].sum()
        total_flank_sites = res['m6a_flank'].sum()
    
    total_body_kb = res['body_bp'].sum() / 1000
    total_flank_kb = res['flank_bp'].sum() / 1000
    
    pooled_body = total_body_sites / total_body_kb
    pooled_flank = total_flank_sites / total_flank_kb
    
    # Paired Wilcoxon (within-read comparison)
    stat, pval = stats.wilcoxon(body_vals - flank_vals, alternative='two-sided')
    
    print(f"\n  {mod}:")
    print(f"    Pooled: L1 body = {pooled_body:.3f}/kb, flanking = {pooled_flank:.3f}/kb")
    print(f"    Ratio:  {pooled_body/pooled_flank:.3f}x")
    print(f"    Per-read median: body = {body_vals.median():.3f}/kb, flank = {flank_vals.median():.3f}/kb")
    print(f"    Per-read mean:   body = {body_vals.mean():.3f}/kb, flank = {flank_vals.mean():.3f}/kb")
    print(f"    Paired Wilcoxon: p = {pval:.2e}")
    print(f"    Region sizes: body median = {res['body_bp'].median():.0f}bp, flank median = {res['flank_bp'].median():.0f}bp")

# =========================================================================
# Per-cell-line
# =========================================================================
print("\n" + "="*70)
print("Per-cell-line PSI density comparison")
print("="*70)

print(f"\n{'CL':12s} {'n':>6s} {'Body/kb':>8s} {'Flank/kb':>9s} {'Ratio':>6s} {'p':>10s}")
for cl in sorted(CELL_LINES.keys()):
    sub = res[res['cell_line'] == cl]
    if len(sub) < 20:
        continue
    
    pooled_body = sub['psi_body'].sum() / (sub['body_bp'].sum() / 1000)
    pooled_flank = sub['psi_flank'].sum() / (sub['flank_bp'].sum() / 1000)
    ratio = pooled_body / pooled_flank if pooled_flank > 0 else float('inf')
    
    try:
        _, pval = stats.wilcoxon(sub['psi_body_per_kb'] - sub['psi_flank_per_kb'])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    except:
        pval = 1.0
        sig = ''
    
    print(f"  {cl:12s} {len(sub):5d} {pooled_body:7.3f} {pooled_flank:8.3f} {ratio:5.3f} {pval:10.2e} {sig}")

# =========================================================================
# Young vs Ancient L1
# =========================================================================
print("\n" + "="*70)
print("Young vs Ancient L1: body psi density")
print("="*70)

YOUNG = ['L1HS', 'L1PA2', 'L1PA3']  # L1PA1 often missing
res['is_young'] = res['TE_group'].isin(YOUNG)

for label, mask in [('Young L1', res['is_young']), ('Ancient L1', ~res['is_young'])]:
    sub = res[mask]
    if len(sub) < 20:
        continue
    
    pooled_body = sub['psi_body'].sum() / (sub['body_bp'].sum() / 1000)
    pooled_flank = sub['psi_flank'].sum() / (sub['flank_bp'].sum() / 1000)
    ratio = pooled_body / pooled_flank if pooled_flank > 0 else float('inf')
    
    try:
        _, pval = stats.wilcoxon(sub['psi_body_per_kb'] - sub['psi_flank_per_kb'])
    except:
        pval = 1.0
    
    print(f"\n  {label} (n={len(sub)})")
    print(f"    Body:  {pooled_body:.3f} psi/kb")
    print(f"    Flank: {pooled_flank:.3f} psi/kb")
    print(f"    Ratio: {ratio:.3f}x")
    print(f"    Paired Wilcoxon: p = {pval:.2e}")
    print(f"    Body bp median: {sub['body_bp'].median():.0f}, Flank bp median: {sub['flank_bp'].median():.0f}")

# =========================================================================
# Control for region length: matched comparison
# =========================================================================
print("\n" + "="*70)
print("Length-matched comparison (body_bp ≈ flank_bp)")
print("="*70)

# Reads where body and flanking are similar length (ratio 0.5-2.0)
ratio_col = res['body_bp'] / res['flank_bp']
matched = res[(ratio_col >= 0.5) & (ratio_col <= 2.0)]
print(f"Reads with body/flank ratio 0.5-2.0: {len(matched)}")

if len(matched) > 20:
    pooled_body = matched['psi_body'].sum() / (matched['body_bp'].sum() / 1000)
    pooled_flank = matched['psi_flank'].sum() / (matched['flank_bp'].sum() / 1000)
    _, pval = stats.wilcoxon(matched['psi_body_per_kb'] - matched['psi_flank_per_kb'])
    
    print(f"  Body:  {pooled_body:.3f} psi/kb")
    print(f"  Flank: {pooled_flank:.3f} psi/kb")
    print(f"  Ratio: {pooled_body/pooled_flank:.3f}x")
    print(f"  p = {pval:.2e}")

# =========================================================================
# Summary
# =========================================================================
total_body = res['psi_body'].sum() / (res['body_bp'].sum() / 1000)
total_flank = res['psi_flank'].sum() / (res['flank_bp'].sum() / 1000)

print(f"\n{'='*70}")
print("SUMMARY")
print("='*70")
print(f"""
Question: Does L1 body have higher psi DENSITY than flanking?

Answer: L1 body psi = {total_body:.3f}/kb, flanking = {total_flank:.3f}/kb
  Ratio = {total_body/total_flank:.3f}x
  → L1 body {'HAS HIGHER' if total_body > total_flank else 'HAS LOWER' if total_body < total_flank else 'HAS EQUAL'} psi density than flanking

This means: even when comparing the SAME read, the L1 sequence portion
{'accumulates more' if total_body > total_flank else 'accumulates fewer'} psi modifications per kb than the non-L1 flanking portion.
""")

# Save
res.to_csv(TOPICDIR / 'psi_motif_selectivity/psi_density_body_vs_flanking.tsv',
           sep='\t', index=False)
print("Done!")
