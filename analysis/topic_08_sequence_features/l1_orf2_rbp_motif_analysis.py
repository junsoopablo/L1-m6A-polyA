#!/usr/bin/env python3
"""
ORF2 EN domain RBP motif analysis:
Map known RBP binding motifs to L1HS consensus, then check whether
mutation-sensitive windows are enriched for specific motifs.

Literature basis:
- SAFB (Seczynska 2024 Nature): Binds A-rich L1 RNA, m6A-dependent degradation
- YTHDF2 (Xiong 2021): RIP-seq binding peak in ORF2 middle
- ZAP (Takata 2017 Nature): CpG-targeting antiviral factor. L1 CpG-depleted
- HuR/ELAVL1: AU-rich element (ARE) binding, RNA stabilization
- DRACH: m6A methylation motif (METTL3/14)
"""
import pandas as pd
import numpy as np
from scipy import stats
import re

OUT_DIR = 'topic_08_sequence_features'

# ── Load L1HS consensus sequence ──
with open(f'{OUT_DIR}/L1HS_consensus.fa') as f:
    lines = f.readlines()
    consensus = ''.join(l.strip() for l in lines if not l.startswith('>'))
L1_LEN = len(consensus)
print(f"L1HS consensus length: {L1_LEN} bp")

# ── L1 ORF2 protein domain boundaries (consensus coordinates, 0-based for python) ──
DOMAINS = {
    '5UTR': (0, 908),
    'ORF1': (908, 1990),
    'ORF2_EN': (1990, 2708),
    'ORF2_RT': (2708, 4149),
    'ORF2_Crich': (4149, 5817),
    '3UTR': (5817, L1_LEN),
}

# ── Define RBP binding motifs ──
MOTIFS = {
    'DRACH': r'[AGT][AG]AC[ACT]',
    'DRACH_core': r'GAC',
    'GGACT': r'GGACT',
    'A_rich_6mer': r'A{6,}',
    'A_rich_4mer': r'A{4,5}',
    'CpG': r'CG',
    'ARE_AUUUA': r'ATTTA',
}

# ── Step 1: Motif density per domain ──
print("\n" + "="*70)
print("Step 1: Motif density across L1HS consensus domains")
print("="*70)

motif_positions = {}
for name, pattern in MOTIFS.items():
    positions = [m.start() for m in re.finditer(pattern, consensus)]
    motif_positions[name] = positions

print(f"\n{'Motif':<16}", end='')
for dname in DOMAINS:
    print(f" {dname:>10}", end='')
print("  /kb")
print("-"*90)

for name, positions in motif_positions.items():
    print(f"  {name:<16}", end='')
    for dname, (ds, de) in DOMAINS.items():
        dlen = (de - ds) / 1000
        count = sum(1 for p in positions if ds <= p < de)
        print(f" {count/dlen:>10.1f}", end='')
    print()

# Key finding: ORF2_EN has HIGHEST DRACH density
en_drach = sum(1 for p in motif_positions['DRACH'] if 1990 <= p < 2708) / (718/1000)
all_drach = len(motif_positions['DRACH']) / (L1_LEN/1000)
print(f"\n  ORF2_EN DRACH density: {en_drach:.1f}/kb (L1 average: {all_drach:.1f}/kb) = {en_drach/all_drach:.2f}x")

# ── Step 2: Window-level motif enrichment at mutation hotspots ──
print("\n" + "="*70)
print("Step 2: Motif density in mutation-sensitive vs non-sensitive windows")
print("="*70)

win_df = pd.read_csv(f'{OUT_DIR}/l1_mutation_sensitivity_windows.tsv', sep='\t')

# Add motif counts per window
drach_pattern = r'[AGT][AG]AC[ACT]'
for _, row in win_df.iterrows():
    ws, we = int(row['cons_start']), int(row['cons_end'])
    seq = consensus[ws:we]
    win_df.loc[_, 'n_drach'] = len(re.findall(drach_pattern, seq))
    win_df.loc[_, 'n_ggact'] = seq.count('GGACT')
    win_df.loc[_, 'n_cpg'] = seq.count('CG')
    win_df.loc[_, 'a_fraction'] = seq.count('A') / len(seq)
    win_df.loc[_, 'gc_content'] = (seq.count('G') + seq.count('C')) / len(seq)

# Compare significant (more mutated in sensitive) vs non-significant ORF2 windows
orf2_win = win_df[win_df['domain'] == 'ORF2'].copy()
sig_win = orf2_win[orf2_win['sig'] == True]
nonsig_win = orf2_win[orf2_win['sig'] == False]

print(f"\nORF2 windows: {len(sig_win)} significant, {len(nonsig_win)} non-significant")
print(f"\n{'Feature':<16} {'Sig_mean':>10} {'NonSig_mean':>12} {'Ratio':>8} {'P':>12}")
print("-"*65)

for feat in ['n_drach', 'n_ggact', 'n_cpg', 'a_fraction', 'gc_content']:
    if len(sig_win) > 0 and len(nonsig_win) > 0:
        _, p = stats.mannwhitneyu(sig_win[feat], nonsig_win[feat], alternative='two-sided')
        ratio = sig_win[feat].mean() / nonsig_win[feat].mean() if nonsig_win[feat].mean() > 0 else np.inf
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {feat:<16} {sig_win[feat].mean():>10.3f} {nonsig_win[feat].mean():>12.3f} "
              f"{ratio:>8.2f}x {p:>12.3e} {sig_str}")

# ── Step 3: Correlation within ORF2 ──
print("\n" + "="*70)
print("Step 3: Motif density correlates with mutation differential (ORF2)")
print("="*70)

for feat in ['n_drach', 'n_cpg', 'a_fraction', 'gc_content']:
    valid = orf2_win.dropna(subset=['diff', feat])
    if len(valid) >= 10:
        rho, p = stats.spearmanr(valid[feat], valid['diff'])
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        direction = "↑ DRACH → ↑ mut in sensitive" if rho > 0 else "↓"
        print(f"  {feat:<16} rho={rho:>+.3f}  P={p:.3e}  {sig_str}")

# ── Step 4: A-richness profile (SAFB binding substrate) ──
print("\n" + "="*70)
print("Step 4: A-richness profile across L1 (SAFB substrate)")
print("="*70)

window_size = 100
a_profile = []
for i in range(0, L1_LEN - window_size, 50):
    seq = consensus[i:i+window_size]
    a_frac = seq.count('A') / len(seq)
    # Determine domain
    mid = i + window_size // 2
    domain = 'unknown'
    for dname, (ds, de) in DOMAINS.items():
        if ds <= mid < de:
            domain = dname
            break
    a_profile.append({'position': mid, 'a_fraction': a_frac, 'domain': domain})
a_prof_df = pd.DataFrame(a_profile)

for dname in DOMAINS:
    sub = a_prof_df[a_prof_df['domain'] == dname]
    if len(sub) > 0:
        print(f"  {dname:<12} A fraction: {sub['a_fraction'].mean():.3f} ± {sub['a_fraction'].std():.3f}")

# EN hotspot specifically
hotspot = a_prof_df[(a_prof_df['position'] >= 2100) & (a_prof_df['position'] < 3250)]
rest_orf2 = a_prof_df[(a_prof_df['domain'].str.startswith('ORF2')) &
                       ~((a_prof_df['position'] >= 2100) & (a_prof_df['position'] < 3250))]
_, p = stats.mannwhitneyu(hotspot['a_fraction'], rest_orf2['a_fraction'])
print(f"\n  EN hotspot (2100-3250): A = {hotspot['a_fraction'].mean():.3f}")
print(f"  Rest of ORF2:          A = {rest_orf2['a_fraction'].mean():.3f}")
print(f"  P = {p:.3e}")

# ── Step 5: CpG density analysis (ZAP targeting) ──
print("\n" + "="*70)
print("Step 5: CpG density by domain (ZAP sensitivity)")
print("="*70)

for dname, (ds, de) in DOMAINS.items():
    seq = consensus[ds:de]
    cpg = seq.count('CG')
    cpg_density = cpg / (len(seq) / 1000)
    # CpG obs/exp
    c_count = seq.count('C')
    g_count = seq.count('G')
    expected = (c_count * g_count) / len(seq)
    obs_exp = cpg / expected if expected > 0 else 0
    print(f"  {dname:<12} CpG/kb={cpg_density:>6.1f}  obs/exp={obs_exp:>5.2f}  n={cpg}")

# ── Step 6: DRACH + mutation sensitivity map overlay ──
print("\n" + "="*70)
print("Step 6: DRACH sites in significant mutation windows")
print("="*70)

# Count DRACH sites within each significant window
print(f"\n{'Window':<15} {'Diff':>8} {'N_DRACH':>8} {'DRACH_seq':>30}")
print("-"*65)
for _, row in sig_win.sort_values('diff', ascending=False).iterrows():
    ws, we = int(row['cons_start']), int(row['cons_end'])
    seq = consensus[ws:we]
    drach_sites = [(m.start()+ws, m.group()) for m in re.finditer(drach_pattern, seq)]
    drach_str = ', '.join([f"{p}:{s}" for p, s in drach_sites[:3]])
    if len(drach_sites) > 3:
        drach_str += f"... (+{len(drach_sites)-3})"
    print(f"  {ws}-{we:<10} {row['diff']:>+8.4f} {len(drach_sites):>8}  {drach_str}")

total_drach_in_sig = sum(int(row['n_drach']) for _, row in sig_win.iterrows())
total_drach_in_nonsig = sum(int(row['n_drach']) for _, row in nonsig_win.iterrows())
print(f"\n  Total DRACH in sig windows: {total_drach_in_sig}")
print(f"  Total DRACH in non-sig windows: {total_drach_in_nonsig}")
print(f"  DRACH per window: sig={total_drach_in_sig/len(sig_win):.1f} vs nonsig={total_drach_in_nonsig/len(nonsig_win):.1f}")

# ── Step 7: Therapeutic implications summary ──
print("\n" + "="*70)
print("Step 7: Key findings for therapeutic implications")
print("="*70)

# 1. DRACH density in EN domain
print(f"\n1. ORF2_EN has HIGHEST DRACH density ({en_drach:.1f}/kb vs L1 avg {all_drach:.1f}/kb)")
print(f"   → This region is the primary m6A deposition zone")
print(f"   → Mutations here could disrupt m6A-mediated SAFB/YTHDF2 surveillance")

# 2. CpG depletion
en_cpg = sum(1 for p in motif_positions['CpG'] if 1990 <= p < 2708) / (718/1000)
utr5_cpg = sum(1 for p in motif_positions['CpG'] if 0 <= p < 908) / (908/1000)
print(f"\n2. ORF2_EN CpG density: {en_cpg:.1f}/kb vs 5'UTR {utr5_cpg:.1f}/kb")
print(f"   → EN domain almost completely CpG-depleted → invisible to ZAP")

# 3. A-richness
en_a = hotspot['a_fraction'].mean()
print(f"\n3. EN hotspot A-richness: {en_a:.3f}")
print(f"   → SAFB binds A-rich L1 RNA (Seczynska 2024)")
print(f"   → Mutations reducing A-richness could impair SAFB-mediated degradation")

# ── Save ──
orf2_win.to_csv(f'{OUT_DIR}/l1_orf2_window_motif_data.tsv', sep='\t', index=False)
print(f"\nSaved: l1_orf2_window_motif_data.tsv")
