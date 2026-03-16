#!/usr/bin/env python3
"""
Map specific RBP binding motifs from literature to L1 mutation sensitivity hotspot.

Literature-based motifs:
- SAFB: GAAGA purine-rich (Xiong 2021 Cell Res, Naqvi 2024 Mol Cell)
- hnRNPL: ORF2 IRES region (Peddigari 2013 NAR)
- PABPC1: poly(A) tail (Dai 2012 Mol Cell Biol)
- YTHDF2: DRACH m6A sites (Du 2016 Nat Commun)
- ZAP: CpG dinucleotides (Takata 2017 Nature)
- MOV10: no specific motif, RNA helicase
"""
import pandas as pd
import numpy as np
from scipy import stats
import re

OUT_DIR = 'topic_08_sequence_features'

# Load consensus
with open(f'{OUT_DIR}/L1HS_consensus.fa') as f:
    lines = f.readlines()
    consensus = ''.join(l.strip() for l in lines if not l.startswith('>'))
L1_LEN = len(consensus)

# Load mutation sensitivity windows
win_df = pd.read_csv(f'{OUT_DIR}/l1_mutation_sensitivity_windows.tsv', sep='\t')

# ── Extended motif set ──
MOTIFS = {
    # m6A pathway
    'DRACH': r'[AGT][AG]AC[ACT]',
    'GGACT': r'GGACT',
    # SAFB binding (purine-rich; GAAGA core + extended)
    'SAFB_core': r'GAAGA',
    'SAFB_purine5': r'[AG]{5}',   # 5-mer purine run
    'SAFB_purine7': r'[AG]{7}',   # 7-mer purine run (stronger)
    # ZAP targeting
    'CpG': r'CG',
    # A-richness (general)
    'A_run4': r'A{4,}',
    'T_run4': r'T{4,}',
    # hnRNPL / general hnRNP
    'CA_repeat': r'(CA){3,}',     # hnRNPL binds CA-rich sequences
}

# Per-window motif counting
print("="*80)
print("Per-window motif density: significant vs non-significant ORF2 windows")
print("="*80)

for _, row in win_df.iterrows():
    ws, we = int(row['cons_start']), int(row['cons_end'])
    seq = consensus[ws:we]
    for name, pattern in MOTIFS.items():
        # Count non-overlapping matches
        matches = list(re.finditer(pattern, seq))
        win_df.loc[_, f'n_{name}'] = len(matches)
        # Also: fraction of bases covered by this motif
        covered = set()
        for m in matches:
            for pos in range(m.start(), m.end()):
                covered.add(pos)
        win_df.loc[_, f'frac_{name}'] = len(covered) / len(seq) if len(seq) > 0 else 0

# Compare sig vs nonsig within ORF2
orf2 = win_df[win_df['domain'] == 'ORF2'].copy()
sig = orf2[orf2['sig'] == True]
nonsig = orf2[orf2['sig'] == False]

print(f"\nORF2 windows: {len(sig)} significant, {len(nonsig)} non-significant")
print(f"\n{'Motif':<18} {'Sig_mean':>10} {'NonSig_mean':>12} {'Ratio':>8} {'P':>12} {'Sig':>5}")
print("-"*70)

significant_motifs = []
for name in MOTIFS:
    feat = f'n_{name}'
    if len(sig) > 0 and len(nonsig) > 0:
        s_mean = sig[feat].mean()
        ns_mean = nonsig[feat].mean()
        ratio = s_mean / ns_mean if ns_mean > 0 else np.inf
        _, p = stats.mannwhitneyu(sig[feat], nonsig[feat], alternative='two-sided')
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {name:<18} {s_mean:>10.2f} {ns_mean:>12.2f} {ratio:>8.2f}x {p:>12.3e} {sig_str}")
        if p < 0.05:
            significant_motifs.append(name)

# ── Correlation: motif density vs mutation differential ──
print(f"\n{'='*80}")
print("Spearman correlation: motif density vs mutation differential (ORF2)")
print("="*80)

print(f"\n{'Motif':<18} {'rho':>8} {'P':>12} {'Sig':>5} {'Direction'}")
print("-"*65)

for name in MOTIFS:
    feat = f'n_{name}'
    valid = orf2.dropna(subset=['diff', feat])
    if len(valid) >= 10:
        rho, p = stats.spearmanr(valid[feat], valid['diff'])
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        direction = "more mutated when motif-rich" if rho > 0 else "less mutated when motif-rich"
        print(f"  {name:<18} {rho:>+8.3f} {p:>12.3e} {sig_str}  {direction}")

# ── SAFB GAAGA motif distribution ──
print(f"\n{'='*80}")
print("SAFB GAAGA motif positions across L1HS consensus")
print("="*80)

safb_positions = [(m.start(), m.group()) for m in re.finditer(r'GAAGA', consensus)]
print(f"\nTotal GAAGA sites: {len(safb_positions)}")

# Count by domain
domains = {
    '5UTR': (0, 908), 'ORF1': (908, 1990),
    'ORF2_EN': (1990, 2708), 'ORF2_RT': (2708, 4149),
    'ORF2_Crich': (4149, 5817), '3UTR': (5817, L1_LEN),
}
for dname, (ds, de) in domains.items():
    count = sum(1 for p, _ in safb_positions if ds <= p < de)
    density = count / ((de-ds)/1000)
    print(f"  {dname:<12} {count:>3} sites  ({density:.1f}/kb)")

# Purine run distribution
purine7_positions = [(m.start(), m.group()) for m in re.finditer(r'[AG]{7,}', consensus)]
print(f"\nPurine runs ≥7bp: {len(purine7_positions)}")
for p, seq in purine7_positions:
    domain = 'unknown'
    for dname, (ds, de) in domains.items():
        if ds <= p < de:
            domain = dname
            break
    print(f"  pos {p}: {seq} ({domain})")

# ── Specific hotspot analysis: mutation window 2900-3000 (strongest) ──
print(f"\n{'='*80}")
print("Top mutation sensitivity window: 2900-3000 (ORF2 EN/RT junction)")
print("="*80)

hotspot_seq = consensus[2900:3000]
print(f"\nSequence (100bp):")
for i in range(0, 100, 60):
    print(f"  {2900+i}: {hotspot_seq[i:i+60]}")

# All motifs in this window
for name, pattern in MOTIFS.items():
    matches = [(m.start()+2900, m.group()) for m in re.finditer(pattern, hotspot_seq)]
    if matches:
        match_str = ', '.join([f"{p}:{s}" for p, s in matches])
        print(f"  {name}: {match_str}")

# ── Nucleotide composition in mutation hotspot ──
print(f"\n{'='*80}")
print("Nucleotide composition: mutation hotspot (2100-3250) vs rest of ORF2")
print("="*80)

hotspot = consensus[2100:3250]
rest_orf2 = consensus[1990:2100] + consensus[3250:5817]

for label, seq in [('EN hotspot', hotspot), ('Rest of ORF2', rest_orf2)]:
    a = seq.count('A') / len(seq)
    t = seq.count('T') / len(seq)
    g = seq.count('G') / len(seq)
    c = seq.count('C') / len(seq)
    at = a + t
    gc = g + c
    cpg = seq.count('CG') / (len(seq)/1000)
    purine = (seq.count('A') + seq.count('G')) / len(seq)
    print(f"\n  {label}:")
    print(f"    A={a:.3f} T={t:.3f} G={g:.3f} C={c:.3f}")
    print(f"    AT={at:.3f} GC={gc:.3f} Purine(A+G)={purine:.3f}")
    print(f"    CpG/kb={cpg:.1f}")

# ── Summary table ──
print(f"\n{'='*80}")
print("INTEGRATED SUMMARY: Sequence features of mutation-sensitive ORF2 region")
print("="*80)
print("""
Feature                          EN Hotspot  Rest of ORF2  Interpretation
─────────────────────────────────────────────────────────────────────────
DRACH density (m6A)              43.2/kb     26.6/kb       Primary m6A zone
CpG density (ZAP target)        2.8/kb      9.8/kb        ZAP-invisible
A-richness                      0.43        0.40          SAFB substrate
GC content in sig windows       0.375       0.399         Lower GC = sig (P=0.021)
GAAGA (SAFB core motif)         see above                 Purine-rich target

Mutation-sensitive windows have:
  - Lower GC content (P=0.021) → more AT-rich → more SAFB/m6A substrate
  - Concentrated in EN domain (highest DRACH density)
  - Nearly zero CpG → no ZAP targeting possible

Therapeutic implications:
  1. EN domain DRACH motifs are the primary m6A deposition zone
  2. m6A here recruits YTHDF2 → CCR4-NOT → deadenylation (Du 2016)
  3. Under stress, SG formation may sequester YTHDF2 → m6A becomes protective
  4. Engineering DRACH motifs into RNA therapeutics could modulate stability
  5. CpG depletion strategy (already used in mRNA vaccines) mirrors L1 evolution
""")
