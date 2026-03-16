#!/usr/bin/env python3
"""
Extended ChromHMM × L1 analysis.

1. Read length bin × state: sample size check + m6A/kb with CI
2. Per-site m6A rate by chromatin state (BAM direct, CIGAR-aware)
3. Cross-cell-line validation of chromatin state effects
4. Enhancer L1 host gene GO enrichment
5. Spearman partial correlation: m6A/kb ~ chromatin state, controlling read length
6. Heterochromatin immunity: is it real or just short reads?
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUT_DIR = BASE / 'analysis/01_exploration/topic_08_regulatory_chromatin'

###############################################################################
# Load data
###############################################################################
print("=== Loading ===")
df = pd.read_csv(OUT_DIR / 'l1_chromhmm_annotated.tsv', sep='\t')

# Get read_length from summary
RESULTS = BASE / 'results_group'
rl_rows = []
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    s = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length'])
    rl_rows.append(s)
rl = pd.concat(rl_rows, ignore_index=True).drop_duplicates('read_id')
df = df.merge(rl, on='read_id', how='left')

ancient = df[df['l1_age'] == 'ancient'].copy()
anc_normal = ancient[ancient['condition'] == 'normal'].copy()
print(f"  Ancient: {len(ancient):,}, Normal: {len(anc_normal):,}")

###############################################################################
# 1. DETAILED read length bin × state: sample size + m6A/kb with bootstrap CI
###############################################################################
print("\n" + "="*70)
print("1. READ LENGTH BIN × CHROMATIN STATE: SAMPLE SIZE + m6A/kb")
print("="*70)

rl_bins = [(0, 300, '0-300'), (300, 500, '300-500'), (500, 750, '500-750'),
           (750, 1000, '750-1K'), (1000, 1500, '1K-1.5K'),
           (1500, 2000, '1.5K-2K'), (2000, 5000, '2K-5K')]

states = ['Promoter', 'Enhancer', 'Transcribed', 'Quiescent',
          'Repressed', 'Heterochromatin']

# Print header
print(f"\n{'RL bin':>10s}", end='')
for st in states:
    print(f" | {st:>22s}", end='')
print()
print("-" * 160)

for lo, hi, label in rl_bins:
    rl_sub = anc_normal[(anc_normal['read_length'] >= lo) &
                         (anc_normal['read_length'] < hi)]
    print(f"{label:>10s}", end='')
    for st in states:
        gs = rl_sub[rl_sub['chromhmm_group'] == st]
        n = len(gs)
        if n >= 5:
            med = gs['m6a_per_kb'].median()
            print(f" | n={n:4d} med={med:4.1f}", end='')
        else:
            print(f" |      n={n:2d}    -  ", end='')
    print()

# Overall totals
print(f"{'TOTAL':>10s}", end='')
for st in states:
    gs = anc_normal[anc_normal['chromhmm_group'] == st]
    n = len(gs)
    med = gs['m6a_per_kb'].median()
    print(f" | n={n:4d} med={med:4.1f}", end='')
print()

###############################################################################
# 1b. Within-bin pairwise comparisons (only bins with n≥30 per group)
###############################################################################
print("\n--- 1b. Within-bin Pairwise: Transcribed vs Quiescent ---")
for lo, hi, label in rl_bins:
    rl_sub = anc_normal[(anc_normal['read_length'] >= lo) &
                         (anc_normal['read_length'] < hi)]
    tx = rl_sub[rl_sub['chromhmm_group'] == 'Transcribed']['m6a_per_kb']
    qu = rl_sub[rl_sub['chromhmm_group'] == 'Quiescent']['m6a_per_kb']
    if len(tx) >= 30 and len(qu) >= 30:
        ratio = tx.median() / qu.median() if qu.median() > 0 else np.nan
        _, p = stats.mannwhitneyu(tx, qu, alternative='two-sided')
        print(f"  {label:>10s}: Tx med={tx.median():.2f} vs Qu med={qu.median():.2f}, "
              f"ratio={ratio:.3f}, p={p:.2e} (n={len(tx)},{len(qu)})")

print("\n--- 1c. Within-bin Pairwise: Enhancer vs Quiescent ---")
for lo, hi, label in rl_bins:
    rl_sub = anc_normal[(anc_normal['read_length'] >= lo) &
                         (anc_normal['read_length'] < hi)]
    en = rl_sub[rl_sub['chromhmm_group'] == 'Enhancer']['m6a_per_kb']
    qu = rl_sub[rl_sub['chromhmm_group'] == 'Quiescent']['m6a_per_kb']
    if len(en) >= 20 and len(qu) >= 30:
        ratio = en.median() / qu.median() if qu.median() > 0 else np.nan
        _, p = stats.mannwhitneyu(en, qu, alternative='two-sided')
        print(f"  {label:>10s}: Enh med={en.median():.2f} vs Qu med={qu.median():.2f}, "
              f"ratio={ratio:.3f}, p={p:.2e} (n={len(en)},{len(qu)})")

###############################################################################
# 2. Partial correlation: m6A/kb ~ chromatin, controlling read length
###############################################################################
print("\n" + "="*70)
print("2. PARTIAL CORRELATION: m6A/kb ~ Chromatin State, Controlling RL")
print("="*70)

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # Encode chromatin states as dummies
    anc_n = anc_normal.copy()
    anc_n['is_enhancer'] = (anc_n['chromhmm_group'] == 'Enhancer').astype(int)
    anc_n['is_promoter'] = (anc_n['chromhmm_group'] == 'Promoter').astype(int)
    anc_n['is_transcribed'] = (anc_n['chromhmm_group'] == 'Transcribed').astype(int)
    anc_n['is_heterochromatin'] = (anc_n['chromhmm_group'] == 'Heterochromatin').astype(int)
    anc_n['is_repressed'] = (anc_n['chromhmm_group'] == 'Repressed').astype(int)
    # Reference = Quiescent

    model = ols('m6a_per_kb ~ read_length + is_enhancer + is_promoter + '
                'is_transcribed + is_heterochromatin + is_repressed',
                data=anc_n).fit()
    print(f"\n  N={len(anc_n):,}, R²={model.rsquared:.4f}")
    print(f"\n  Coefficients (ref=Quiescent):")
    for param in model.params.index:
        coef = model.params[param]
        p = model.pvalues[param]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"    {param:30s}: {coef:+8.3f} (p={p:.2e}) {sig}")

except ImportError:
    print("  statsmodels not available")

###############################################################################
# 3. Heterochromatin immunity deep-dive
###############################################################################
print("\n" + "="*70)
print("3. HETEROCHROMATIN IMMUNITY: CONTROLS")
print("="*70)

hela = ancient[ancient['cellline'] == 'HeLa'].copy()
hela_ars = ancient[ancient['cellline'] == 'HeLa-Ars'].copy()

het_hela = hela[hela['chromhmm_group'] == 'Heterochromatin']
het_ars = hela_ars[hela_ars['chromhmm_group'] == 'Heterochromatin']

print(f"\n  Het HeLa: n={len(het_hela)}, median RL={het_hela['read_length'].median():.0f}bp")
print(f"  Het Ars:  n={len(het_ars)}, median RL={het_ars['read_length'].median():.0f}bp")
print(f"  Het HeLa poly(A): {het_hela['polya_length'].median():.0f}nt")
print(f"  Het Ars  poly(A): {het_ars['polya_length'].median():.0f}nt")

# Compare with Quiescent (control) matched on RL
for grp_name, grp_label in [('Heterochromatin', 'Het'), ('Quiescent', 'Qu'),
                              ('Enhancer', 'Enh'), ('Transcribed', 'Tx')]:
    h = hela[hela['chromhmm_group'] == grp_name]
    ha = hela_ars[hela_ars['chromhmm_group'] == grp_name]
    if len(h) >= 5 and len(ha) >= 5:
        # RL distribution comparison
        _, rl_p = stats.mannwhitneyu(h['read_length'], ha['read_length'])
        print(f"\n  {grp_label}: HeLa RL={h['read_length'].median():.0f}, "
              f"Ars RL={ha['read_length'].median():.0f}, MW p={rl_p:.2e}")
        print(f"  {grp_label}: HeLa m6A/kb={h['m6a_per_kb'].median():.2f}, "
              f"Ars m6A/kb={ha['m6a_per_kb'].median():.2f}")

# Heterochromatin: intronic vs intergenic
het_all = ancient[ancient['chromhmm_group'] == 'Heterochromatin']
print(f"\n  Heterochromatin genomic context:")
for ctx in ['intronic', 'intergenic']:
    sub = het_all[het_all['genomic_context'] == ctx]
    print(f"    {ctx}: n={len(sub)} ({100*len(sub)/len(het_all):.0f}%)")

# Heterochromatin: L1 subfamily distribution
het_sf = het_all['gene_id'].str.split('_dup').str[0].value_counts()
print(f"\n  Heterochromatin top subfamilies:")
for sf, n in het_sf.head(10).items():
    print(f"    {sf}: {n}")

###############################################################################
# 4. Cross-cell-line validation: does chromatin state effect hold?
###############################################################################
print("\n" + "="*70)
print("4. CROSS-CELL-LINE: CHROMATIN STATE × POLY(A)")
print("="*70)
print("\n  Note: ChromHMM is HeLa-S3 specific. Other CLs use same annotation")
print("  as proxy (imperfect but informative for conserved states)")

normal_ancient = ancient[ancient['condition'] == 'normal']
for cl in sorted(normal_ancient['cellline'].unique()):
    cl_sub = normal_ancient[normal_ancient['cellline'] == cl]
    if len(cl_sub) < 100:
        continue
    reg = cl_sub[cl_sub['chromhmm_group'].isin(['Enhancer', 'Promoter'])]
    nonreg = cl_sub[~cl_sub['chromhmm_group'].isin(['Enhancer', 'Promoter'])]
    het = cl_sub[cl_sub['chromhmm_group'] == 'Heterochromatin']

    if len(reg) >= 10 and len(nonreg) >= 10:
        reg_pa = reg['polya_length'].median()
        nonreg_pa = nonreg['polya_length'].median()
        het_pa = het['polya_length'].median() if len(het) >= 5 else np.nan
        reg_m6a = reg['m6a_per_kb'].median()
        nonreg_m6a = nonreg['m6a_per_kb'].median()
        het_str = f"{het_pa:.0f}" if not np.isnan(het_pa) else "  -"
        print(f"  {cl:15s}: Reg poly(A)={reg_pa:.0f} (n={len(reg):3d}), "
              f"NonReg={nonreg_pa:.0f} (n={len(nonreg):4d}), "
              f"Het={het_str:>4s} (n={len(het):3d}), "
              f"Reg m6A/kb={reg_m6a:.1f}, NonReg m6A/kb={nonreg_m6a:.1f}")

###############################################################################
# 5. Enhancer vs Promoter: separate effects
###############################################################################
print("\n" + "="*70)
print("5. ENHANCER vs PROMOTER: SEPARATE ARSENITE EFFECTS")
print("="*70)

for grp in ['Enhancer', 'Promoter']:
    h = hela[hela['chromhmm_group'] == grp]
    ha = hela_ars[hela_ars['chromhmm_group'] == grp]
    if len(h) >= 5 and len(ha) >= 5:
        delta = ha['polya_length'].median() - h['polya_length'].median()
        _, p = stats.mannwhitneyu(h['polya_length'], ha['polya_length'])
        print(f"  {grp:15s}: HeLa={h['polya_length'].median():.0f}nt (n={len(h)}), "
              f"Ars={ha['polya_length'].median():.0f}nt (n={len(ha)}), "
              f"Δ={delta:+.1f}nt, p={p:.2e}")
        # m6A comparison
        print(f"  {'':15s}  m6A/kb: HeLa={h['m6a_per_kb'].median():.2f}, "
              f"Ars={ha['m6a_per_kb'].median():.2f}")
        # RL comparison
        print(f"  {'':15s}  RL: HeLa={h['read_length'].median():.0f}bp, "
              f"Ars={ha['read_length'].median():.0f}bp")

###############################################################################
# 6. Promoter L1: why is baseline poly(A) shorter?
###############################################################################
print("\n" + "="*70)
print("6. PROMOTER L1: WHY IS BASELINE POLY(A) SHORTER?")
print("="*70)

prom = anc_normal[anc_normal['chromhmm_group'] == 'Promoter']
nonprom = anc_normal[anc_normal['chromhmm_group'] != 'Promoter']

print(f"\n  Promoter:     RL={prom['read_length'].median():.0f}bp, "
      f"poly(A)={prom['polya_length'].median():.0f}nt, "
      f"m6A/kb={prom['m6a_per_kb'].median():.2f} (n={len(prom):,})")
print(f"  Non-Promoter: RL={nonprom['read_length'].median():.0f}bp, "
      f"poly(A)={nonprom['polya_length'].median():.0f}nt, "
      f"m6A/kb={nonprom['m6a_per_kb'].median():.2f} (n={len(nonprom):,})")

# Is shorter poly(A) at promoters driven by shorter reads?
# Match read length 500-1500bp
prom_rl = prom[(prom['read_length'] >= 500) & (prom['read_length'] <= 1500)]
nonprom_rl = nonprom[(nonprom['read_length'] >= 500) & (nonprom['read_length'] <= 1500)]
print(f"\n  RL-matched (500-1500bp):")
print(f"    Promoter: poly(A)={prom_rl['polya_length'].median():.0f}nt (n={len(prom_rl)})")
print(f"    Non-Prom: poly(A)={nonprom_rl['polya_length'].median():.0f}nt (n={len(nonprom_rl)})")
_, p = stats.mannwhitneyu(prom_rl['polya_length'], nonprom_rl['polya_length'])
print(f"    MW p={p:.2e}")

# Promoter: intergenic dominance
print(f"\n  Promoter genomic context: "
      f"intergenic={len(prom[prom['genomic_context']=='intergenic'])} "
      f"({100*len(prom[prom['genomic_context']=='intergenic'])/len(prom):.0f}%), "
      f"intronic={len(prom[prom['genomic_context']=='intronic'])} "
      f"({100*len(prom[prom['genomic_context']=='intronic'])/len(prom):.0f}%)")

# Dominant subfamily at promoter
prom['subfamily'] = prom['gene_id'].str.split('_dup').str[0]
print(f"\n  Promoter top subfamilies:")
for sf, n in prom['subfamily'].value_counts().head(5).items():
    nonprom_sf = nonprom[nonprom['gene_id'].str.split('_dup').str[0] == sf]
    prom_sf = prom[prom['subfamily'] == sf]
    print(f"    {sf}: {n} reads, poly(A)={prom_sf['polya_length'].median():.0f}nt "
          f"(vs non-prom same SF: {nonprom_sf['polya_length'].median():.0f}nt, "
          f"n={len(nonprom_sf)})")

###############################################################################
# 7. Summary table: save per-state per-bin statistics
###############################################################################
print("\n=== Saving detailed stats ===")
rows = []
for lo, hi, label in rl_bins + [(0, 100000, 'ALL')]:
    for st in states + ['ALL']:
        sub = anc_normal.copy()
        if label != 'ALL':
            sub = sub[(sub['read_length'] >= lo) & (sub['read_length'] < hi)]
        if st != 'ALL':
            sub = sub[sub['chromhmm_group'] == st]
        if len(sub) >= 5:
            rows.append({
                'rl_bin': label, 'chromhmm_group': st,
                'n': len(sub),
                'median_m6a_per_kb': sub['m6a_per_kb'].median(),
                'mean_m6a_per_kb': sub['m6a_per_kb'].mean(),
                'median_polya': sub['polya_length'].median(),
                'median_rl': sub['read_length'].median(),
            })

pd.DataFrame(rows).to_csv(OUT_DIR / 'chromhmm_rl_bin_stats.tsv', sep='\t', index=False)
print("  Saved: chromhmm_rl_bin_stats.tsv")

print("\n=== DONE ===")
