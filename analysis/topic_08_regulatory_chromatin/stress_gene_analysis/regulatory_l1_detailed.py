#!/usr/bin/env python3
"""
Regulatory L1 Detailed Analysis: Stress Destabilization of Enhancer/Promoter L1
================================================================================
Focus: HeLa vs HeLa-Ars ancient L1 reads annotated with ChromHMM states.
Key question: Does chromatin accessibility independently drive L1 poly(A) shortening?

Sections:
  A. Regulatory L1 characterization (all chromatin states, HeLa + HeLa-Ars)
  B. Regulatory vs non-regulatory comparison (m6A-matched)
  C. Regulatory L1 host gene analysis
  D. m6A protection within regulatory L1
  E. Arsenite-specific regulatory L1 loci emergence
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin'
CHROMHMM_FILE = f'{BASE}/l1_chromhmm_annotated.tsv'
REG_PER_READ  = f'{BASE}/stress_gene_analysis/regulatory_l1_per_read.tsv'
GENE_FILE     = f'{BASE}/stress_gene_analysis/all_cl_top_genes.tsv'
OUT_DIR       = f'{BASE}/stress_gene_analysis'
OUT_TSV       = f'{OUT_DIR}/regulatory_l1_detailed_analysis.tsv'

# ── Helpers ────────────────────────────────────────────────────────────────
def mw_test(a, b):
    """Mann-Whitney U test, return (U, p, direction)."""
    a, b = np.array(a), np.array(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, 'n/a'
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return U, p, f'{">" if np.median(a) > np.median(b) else "<"}'

def fisher_test(a_yes, a_no, b_yes, b_no):
    """Fisher exact test."""
    table = [[a_yes, a_no], [b_yes, b_no]]
    odds, p = stats.fisher_exact(table)
    return odds, p

def pval_stars(p):
    if p is None or np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

def print_header(title):
    print(f'\n{"="*80}')
    print(f'  {title}')
    print(f'{"="*80}')

def print_subheader(title):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')

# ── Load Data ──────────────────────────────────────────────────────────────
print('Loading data...')
df_all = pd.read_csv(CHROMHMM_FILE, sep='\t')
df_reg = pd.read_csv(REG_PER_READ, sep='\t')

# Compute read length
df_all['read_length'] = df_all['end'] - df_all['start']
df_reg['read_length'] = df_reg['end'] - df_reg['start']

# Create locus ID (chr + L1 subfamily + approximate position)
df_all['locus_id'] = df_all['chr'] + ':' + df_all['start'].astype(str) + '-' + df_all['end'].astype(str) + ':' + df_all['gene_id']

print(f'Total reads: {len(df_all):,}')
print(f'Regulatory per-read file: {len(df_reg):,}')

# ── Filter: HeLa + HeLa-Ars, Ancient only ─────────────────────────────────
hela_mask = df_all['cellline'].isin(['HeLa', 'HeLa-Ars'])
ancient_mask = df_all['l1_age'] == 'ancient'
df = df_all[hela_mask & ancient_mask].copy()
df['is_stress'] = df['cellline'] == 'HeLa-Ars'

print(f'\nHeLa + HeLa-Ars ancient L1: {len(df):,} reads')
print(f'  HeLa: {(~df["is_stress"]).sum():,}')
print(f'  HeLa-Ars: {df["is_stress"].sum():,}')

# Chromhmm group counts
print(f'\nChromHMM group distribution:')
for grp in ['Enhancer', 'Promoter', 'Transcribed', 'Quiescent', 'Heterochromatin']:
    n = (df['chromhmm_group'] == grp).sum()
    pct = n / len(df) * 100
    print(f'  {grp:20s}: {n:5d} ({pct:5.1f}%)')

# ── SECTION A: Characterization by chromatin state ─────────────────────────
print_header('A. REGULATORY L1 CHARACTERIZATION BY CHROMATIN STATE')

STATE_ORDER = ['Enhancer', 'Promoter', 'Transcribed', 'Quiescent', 'Heterochromatin']
results_a = []

for state in STATE_ORDER:
    sub = df[df['chromhmm_group'] == state]
    hela = sub[~sub['is_stress']]
    ars  = sub[sub['is_stress']]

    n_hela = len(hela)
    n_ars  = len(ars)

    med_pa_hela = hela['polya_length'].median() if n_hela > 0 else np.nan
    med_pa_ars  = ars['polya_length'].median() if n_ars > 0 else np.nan
    delta = med_pa_ars - med_pa_hela if n_hela > 0 and n_ars > 0 else np.nan

    med_m6a_hela = hela['m6a_per_kb'].median() if n_hela > 0 else np.nan
    med_m6a_ars  = ars['m6a_per_kb'].median() if n_ars > 0 else np.nan

    # MW test on poly(A)
    if n_hela >= 3 and n_ars >= 3:
        _, p_pa, _ = mw_test(ars['polya_length'].values, hela['polya_length'].values)
    else:
        p_pa = np.nan

    # Decay zone
    dz_hela = (hela['polya_length'] < 30).sum() if n_hela > 0 else 0
    dz_ars  = (ars['polya_length'] < 30).sum() if n_ars > 0 else 0
    dz_frac_hela = dz_hela / n_hela * 100 if n_hela > 0 else 0
    dz_frac_ars  = dz_ars / n_ars * 100 if n_ars > 0 else 0

    results_a.append({
        'state': state,
        'n_hela': n_hela, 'n_ars': n_ars,
        'med_pa_hela': med_pa_hela, 'med_pa_ars': med_pa_ars,
        'delta_pa': delta, 'p_pa': p_pa,
        'med_m6a_hela': med_m6a_hela, 'med_m6a_ars': med_m6a_ars,
        'dz_frac_hela': dz_frac_hela, 'dz_frac_ars': dz_frac_ars,
        'dz_n_hela': dz_hela, 'dz_n_ars': dz_ars,
    })

df_a = pd.DataFrame(results_a)

print('\n{:<15s} {:>6s} {:>6s} {:>8s} {:>8s} {:>8s} {:>10s} {:>8s} {:>8s} {:>7s} {:>7s}'.format(
    'State', 'n_H', 'n_A', 'PA_H', 'PA_A', 'Delta', 'p-val', 'm6A_H', 'm6A_A', 'DZ_H%', 'DZ_A%'))
print('-' * 110)
for _, r in df_a.iterrows():
    print('{:<15s} {:>6d} {:>6d} {:>8.1f} {:>8.1f} {:>+8.1f} {:>10s} {:>8.2f} {:>8.2f} {:>6.1f}% {:>6.1f}%'.format(
        r['state'], r['n_hela'], r['n_ars'],
        r['med_pa_hela'], r['med_pa_ars'], r['delta_pa'],
        f"{r['p_pa']:.2e}{pval_stars(r['p_pa'])}" if not np.isnan(r['p_pa']) else 'n/a',
        r['med_m6a_hela'], r['med_m6a_ars'],
        r['dz_frac_hela'], r['dz_frac_ars']))

# ── A5: m6A quartile analysis WITHIN regulatory L1 ────────────────────────
print_subheader('A5. m6A Quartile Analysis Within Regulatory L1 (HeLa + HeLa-Ars)')

reg = df[df['chromhmm_group'].isin(['Enhancer', 'Promoter'])].copy()
print(f'Regulatory ancient L1 (HeLa+HeLa-Ars): {len(reg)} reads')

# m6A quartiles based on ALL regulatory reads
reg['m6a_q'] = pd.qcut(reg['m6a_per_kb'], q=4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'],
                        duplicates='drop')

print('\n{:<12s} {:>5s} {:>5s} {:>8s} {:>8s} {:>8s} {:>10s} {:>6s} {:>6s}'.format(
    'm6A_Q', 'n_H', 'n_A', 'PA_H', 'PA_A', 'Delta', 'p-val', 'DZ_H%', 'DZ_A%'))
print('-' * 80)

for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
    sub = reg[reg['m6a_q'] == q]
    hela = sub[~sub['is_stress']]
    ars  = sub[sub['is_stress']]

    n_h, n_a = len(hela), len(ars)
    pa_h = hela['polya_length'].median() if n_h > 0 else np.nan
    pa_a = ars['polya_length'].median() if n_a > 0 else np.nan
    delta = pa_a - pa_h if n_h > 0 and n_a > 0 else np.nan
    _, p, _ = mw_test(ars['polya_length'].values, hela['polya_length'].values) if n_h >= 3 and n_a >= 3 else (None, np.nan, '')

    dz_h = (hela['polya_length'] < 30).sum() / n_h * 100 if n_h > 0 else 0
    dz_a = (ars['polya_length'] < 30).sum() / n_a * 100 if n_a > 0 else 0

    m6a_range = f'{sub["m6a_per_kb"].min():.1f}-{sub["m6a_per_kb"].max():.1f}'
    print('{:<12s} {:>5d} {:>5d} {:>8.1f} {:>8.1f} {:>+8.1f} {:>10s} {:>5.1f}% {:>5.1f}%  m6A: {}'.format(
        q, n_h, n_a, pa_h, pa_a, delta,
        f"{p:.2e}{pval_stars(p)}" if not np.isnan(p) else 'n/a',
        dz_h, dz_a, m6a_range))


# ── SECTION B: Regulatory vs Non-Regulatory (m6A-matched) ─────────────────
print_header('B. REGULATORY vs NON-REGULATORY COMPARISON (ANCIENT, HeLa + HeLa-Ars)')

# Label categories
df['cat'] = 'Other'
df.loc[df['chromhmm_group'].isin(['Enhancer', 'Promoter']), 'cat'] = 'Regulatory'
df.loc[df['chromhmm_group'] == 'Transcribed', 'cat'] = 'Transcribed'
df.loc[df['chromhmm_group'] == 'Quiescent', 'cat'] = 'Quiescent'
df.loc[df['chromhmm_group'] == 'Heterochromatin', 'cat'] = 'Het'

CATS = ['Regulatory', 'Transcribed', 'Quiescent', 'Het']

print_subheader('B1. Overall comparison')
print('\n{:<14s} {:>5s} {:>5s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>6s} {:>6s}'.format(
    'Category', 'n_H', 'n_A', 'PA_H', 'PA_A', 'Delta', 'm6A_H', 'm6A_A', 'DZ_H%', 'DZ_A%'))
print('-' * 95)

for cat in CATS:
    sub = df[df['cat'] == cat]
    hela = sub[~sub['is_stress']]
    ars  = sub[sub['is_stress']]

    n_h, n_a = len(hela), len(ars)
    pa_h = hela['polya_length'].median() if n_h > 0 else np.nan
    pa_a = ars['polya_length'].median() if n_a > 0 else np.nan
    delta = pa_a - pa_h if n_h > 0 and n_a > 0 else np.nan
    m6a_h = hela['m6a_per_kb'].median() if n_h > 0 else np.nan
    m6a_a = ars['m6a_per_kb'].median() if n_a > 0 else np.nan
    dz_h = (hela['polya_length'] < 30).sum() / n_h * 100 if n_h > 0 else 0
    dz_a = (ars['polya_length'] < 30).sum() / n_a * 100 if n_a > 0 else 0

    _, p, _ = mw_test(ars['polya_length'].values, hela['polya_length'].values) if n_h >= 3 and n_a >= 3 else (None, np.nan, '')

    print('{:<14s} {:>5d} {:>5d} {:>8.1f} {:>8.1f} {:>+8.1f} {:>8.2f} {:>8.2f} {:>5.1f}% {:>5.1f}%  p={}'.format(
        cat, n_h, n_a, pa_h, pa_a, delta, m6a_h, m6a_a, dz_h, dz_a,
        f'{p:.2e}{pval_stars(p)}' if not np.isnan(p) else 'n/a'))

# ── B2: m6A-Matched comparison ─────────────────────────────────────────────
print_subheader('B2. m6A-Matched: Is Regulatory STILL shorter under stress?')
print('   Matching regulatory and non-regulatory L1 by m6A/kb quartile.')
print('   Within each m6A quartile, compare poly(A) under stress.')

# Define m6A quartiles across ALL categories combined
hela_all = df[~df['is_stress']].copy()
ars_all  = df[df['is_stress']].copy()

# Use quartile boundaries from full HeLa-Ars data
q_bounds = ars_all['m6a_per_kb'].quantile([0.25, 0.5, 0.75]).values

def assign_m6a_quartile(val):
    if val <= q_bounds[0]: return 'Q1'
    elif val <= q_bounds[1]: return 'Q2'
    elif val <= q_bounds[2]: return 'Q3'
    else: return 'Q4'

ars_all['m6a_q'] = ars_all['m6a_per_kb'].apply(assign_m6a_quartile)

print(f'\nm6A/kb quartile boundaries (HeLa-Ars): Q1<={q_bounds[0]:.2f}, Q2<={q_bounds[1]:.2f}, Q3<={q_bounds[2]:.2f}, Q4>{q_bounds[2]:.2f}')

print('\n{:<5s} {:<14s} {:>5s} {:>8s} {:>6s}   vs   {:<14s} {:>5s} {:>8s} {:>6s}   {:>10s}'.format(
    'Q', 'Regulatory', 'n', 'med_PA', 'DZ%', 'Non-Reg', 'n', 'med_PA', 'DZ%', 'p-val'))
print('-' * 110)

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    reg_q = ars_all[(ars_all['cat'] == 'Regulatory') & (ars_all['m6a_q'] == q)]
    nonreg_q = ars_all[(ars_all['cat'] != 'Regulatory') & (ars_all['cat'] != 'Other') & (ars_all['m6a_q'] == q)]

    n_r = len(reg_q)
    n_nr = len(nonreg_q)
    pa_r = reg_q['polya_length'].median() if n_r > 0 else np.nan
    pa_nr = nonreg_q['polya_length'].median() if n_nr > 0 else np.nan
    dz_r = (reg_q['polya_length'] < 30).sum() / n_r * 100 if n_r > 0 else 0
    dz_nr = (nonreg_q['polya_length'] < 30).sum() / n_nr * 100 if n_nr > 0 else 0

    _, p, _ = mw_test(reg_q['polya_length'].values, nonreg_q['polya_length'].values) if n_r >= 3 and n_nr >= 3 else (None, np.nan, '')

    print('{:<5s} {:<14s} {:>5d} {:>8.1f} {:>5.1f}%   vs   {:<14s} {:>5d} {:>8.1f} {:>5.1f}%   p={}'.format(
        q, 'Regulatory', n_r, pa_r, dz_r,
        'Non-Reg', n_nr, pa_nr, dz_nr,
        f'{p:.2e}{pval_stars(p)}' if not np.isnan(p) else 'n/a'))

# Also show same analysis under NORMAL (HeLa) to show it's stress-specific
hela_all['m6a_q'] = hela_all['m6a_per_kb'].apply(assign_m6a_quartile)
print('\n[Same analysis under NORMAL (HeLa) — baseline]')
print('{:<5s} {:<14s} {:>5s} {:>8s} {:>6s}   vs   {:<14s} {:>5s} {:>8s} {:>6s}   {:>10s}'.format(
    'Q', 'Regulatory', 'n', 'med_PA', 'DZ%', 'Non-Reg', 'n', 'med_PA', 'DZ%', 'p-val'))
print('-' * 110)

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    reg_q = hela_all[(hela_all['cat'] == 'Regulatory') & (hela_all['m6a_q'] == q)]
    nonreg_q = hela_all[(hela_all['cat'] != 'Regulatory') & (hela_all['cat'] != 'Other') & (hela_all['m6a_q'] == q)]

    n_r = len(reg_q)
    n_nr = len(nonreg_q)
    pa_r = reg_q['polya_length'].median() if n_r > 0 else np.nan
    pa_nr = nonreg_q['polya_length'].median() if n_nr > 0 else np.nan
    dz_r = (reg_q['polya_length'] < 30).sum() / n_r * 100 if n_r > 0 else 0
    dz_nr = (nonreg_q['polya_length'] < 30).sum() / n_nr * 100 if n_nr > 0 else 0

    _, p, _ = mw_test(reg_q['polya_length'].values, nonreg_q['polya_length'].values) if n_r >= 3 and n_nr >= 3 else (None, np.nan, '')

    print('{:<5s} {:<14s} {:>5d} {:>8.1f} {:>5.1f}%   vs   {:<14s} {:>5d} {:>8.1f} {:>5.1f}%   p={}'.format(
        q, 'Regulatory', n_r, pa_r, dz_r,
        'Non-Reg', n_nr, pa_nr, dz_nr,
        f'{p:.2e}{pval_stars(p)}' if not np.isnan(p) else 'n/a'))

# B2 extended: Read-length-matched regulatory vs non-regulatory under stress
print_subheader('B2b. Read-Length-Matched Regulatory vs Non-Regulatory (HeLa-Ars)')
print('   Matching by read length bins (500-2000bp) to control for 3\' bias.')

rl_bins = [(500, 1000), (1000, 1500), (1500, 2000)]
for lo, hi in rl_bins:
    ars_rl = ars_all[(ars_all['read_length'] >= lo) & (ars_all['read_length'] < hi)]
    reg_rl = ars_rl[ars_rl['cat'] == 'Regulatory']
    nonreg_rl = ars_rl[(ars_rl['cat'] != 'Regulatory') & (ars_rl['cat'] != 'Other')]

    n_r = len(reg_rl)
    n_nr = len(nonreg_rl)
    pa_r = reg_rl['polya_length'].median() if n_r > 0 else np.nan
    pa_nr = nonreg_rl['polya_length'].median() if n_nr > 0 else np.nan
    _, p, _ = mw_test(reg_rl['polya_length'].values, nonreg_rl['polya_length'].values) if n_r >= 3 and n_nr >= 3 else (None, np.nan, '')

    print(f'  RL {lo}-{hi}bp: Reg n={n_r} PA={pa_r:.1f}  Non-Reg n={n_nr} PA={pa_nr:.1f}  '
          f'Delta={pa_r-pa_nr:+.1f}  p={p:.2e}{pval_stars(p)}' if not np.isnan(p) else
          f'  RL {lo}-{hi}bp: Reg n={n_r}  Non-Reg n={n_nr}  insufficient')


# ── SECTION C: Host Gene Analysis ─────────────────────────────────────────
print_header('C. REGULATORY L1 HOST GENE ANALYSIS')

# Use df_reg for host gene info (has host_gene column for regulatory reads from all cell lines)
# But filter to HeLa + HeLa-Ars ancient
reg_hela = df_reg[(df_reg['cellline'].isin(['HeLa', 'HeLa-Ars'])) & (df_reg['l1_age'] == 'ancient')].copy()

print_subheader('C1. Intronic Regulatory L1 — Top Host Genes')
intronic_reg = reg_hela[reg_hela['genomic_context'] == 'intronic']
print(f'Total intronic regulatory L1 reads (HeLa + HeLa-Ars, ancient): {len(intronic_reg)}')

if len(intronic_reg) > 0:
    gene_counts = intronic_reg.groupby('host_gene').agg(
        n_reads=('read_id', 'count'),
        n_hela=('cellline', lambda x: (x == 'HeLa').sum()),
        n_ars=('cellline', lambda x: (x == 'HeLa-Ars').sum()),
        med_polya=('polya_length', 'median'),
        med_m6a=('m6a_per_kb', 'median'),
        chromhmm_states=('chromhmm_state', lambda x: ','.join(sorted(x.unique()))),
    ).sort_values('n_reads', ascending=False)

    print(f'\nTop 20 host genes:')
    print('{:<25s} {:>6s} {:>5s} {:>5s} {:>8s} {:>8s}  {}'.format(
        'Gene', 'Reads', 'HeLa', 'Ars', 'med_PA', 'med_m6A', 'States'))
    print('-' * 90)
    for gene, r in gene_counts.head(20).iterrows():
        print('{:<25s} {:>6d} {:>5d} {:>5d} {:>8.1f} {:>8.2f}  {}'.format(
            str(gene)[:25], r['n_reads'], r['n_hela'], r['n_ars'],
            r['med_polya'], r['med_m6a'], r['chromhmm_states']))

print_subheader('C2. Intergenic Regulatory L1 — Potential Distal Enhancers')
intergenic_reg = reg_hela[reg_hela['genomic_context'] == 'intergenic']
print(f'Total intergenic regulatory L1 reads (HeLa + HeLa-Ars, ancient): {len(intergenic_reg)}')

if len(intergenic_reg) > 0:
    ig_hela = intergenic_reg[intergenic_reg['cellline'] == 'HeLa']
    ig_ars  = intergenic_reg[intergenic_reg['cellline'] == 'HeLa-Ars']

    print(f'  HeLa: {len(ig_hela)}, HeLa-Ars: {len(ig_ars)}')
    if len(ig_hela) > 0: print(f'  HeLa median poly(A): {ig_hela["polya_length"].median():.1f}')
    if len(ig_ars) > 0:  print(f'  HeLa-Ars median poly(A): {ig_ars["polya_length"].median():.1f}')
    if len(ig_hela) > 0 and len(ig_ars) > 0:
        _, p, _ = mw_test(ig_ars['polya_length'].values, ig_hela['polya_length'].values)
        delta = ig_ars['polya_length'].median() - ig_hela['polya_length'].median()
        print(f'  Delta: {delta:+.1f} nt, p={p:.2e} {pval_stars(p)}')
    if len(ig_hela) > 0: print(f'  HeLa median m6A/kb: {ig_hela["m6a_per_kb"].median():.2f}')
    if len(ig_ars) > 0:  print(f'  HeLa-Ars median m6A/kb: {ig_ars["m6a_per_kb"].median():.2f}')

    # Intronic vs intergenic split
    print(f'\n  Regulatory L1 context split:')
    for ctx in ['intronic', 'intergenic']:
        sub = reg_hela[reg_hela['genomic_context'] == ctx]
        print(f'    {ctx}: {len(sub)} reads ({len(sub)/len(reg_hela)*100:.1f}%)')

print_subheader('C3. Regulatory L1 in Decay Zone (HeLa-Ars, poly(A) < 30 nt)')
ars_reg = reg_hela[(reg_hela['cellline'] == 'HeLa-Ars')]
decay_reads = ars_reg[ars_reg['polya_length'] < 30]
print(f'HeLa-Ars regulatory L1 in decay zone: {len(decay_reads)} / {len(ars_reg)} ({len(decay_reads)/len(ars_reg)*100:.1f}%)')

if len(decay_reads) > 0:
    print(f'  Median m6A/kb: {decay_reads["m6a_per_kb"].median():.2f}')
    print(f'  Mean m6A/kb:   {decay_reads["m6a_per_kb"].mean():.2f}')

    non_decay = ars_reg[ars_reg['polya_length'] >= 30]
    if len(non_decay) > 0:
        print(f'  Non-decay m6A/kb median: {non_decay["m6a_per_kb"].median():.2f}')
        _, p_m6a, _ = mw_test(decay_reads['m6a_per_kb'].values, non_decay['m6a_per_kb'].values)
        print(f'  Decay vs Non-decay m6A: p={p_m6a:.2e} {pval_stars(p_m6a)}')

    # List decay-zone reads with host genes
    if 'host_gene' in decay_reads.columns:
        print(f'\n  Decay zone reads (poly(A) < 30):')
        print(f'  {"read_id":<40s} {"gene":<25s} {"PA":>6s} {"m6A/kb":>7s} {"ctx":>10s} {"state":>12s}')
        print(f'  {"-"*105}')
        for _, r in decay_reads.sort_values('polya_length').head(30).iterrows():
            print(f'  {r["read_id"][:40]:<40s} {str(r.get("host_gene","?"))[:25]:<25s} '
                  f'{r["polya_length"]:>6.1f} {r["m6a_per_kb"]:>7.2f} {r["genomic_context"]:>10s} {r["chromhmm_state"]:>12s}')


# ── SECTION D: m6A Protection Within Regulatory L1 ────────────────────────
print_header('D. m6A PROTECTION WITHIN REGULATORY L1 (HeLa-Ars only)')

ars_reg_all = df[(df['is_stress']) & (df['cat'] == 'Regulatory')].copy()
print(f'HeLa-Ars regulatory ancient L1: {len(ars_reg_all)} reads')

if len(ars_reg_all) >= 20:
    # m6A quartiles within regulatory L1 under stress
    try:
        ars_reg_all['m6a_q'] = pd.qcut(ars_reg_all['m6a_per_kb'], q=4,
                                         labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'],
                                         duplicates='drop')
    except ValueError:
        # If too many ties, use manual boundaries
        bounds = ars_reg_all['m6a_per_kb'].quantile([0.25, 0.5, 0.75]).values
        def q_assign(v):
            if v <= bounds[0]: return 'Q1(low)'
            elif v <= bounds[1]: return 'Q2'
            elif v <= bounds[2]: return 'Q3'
            else: return 'Q4(high)'
        ars_reg_all['m6a_q'] = ars_reg_all['m6a_per_kb'].apply(q_assign)

    print('\n{:<12s} {:>5s} {:>8s} {:>8s} {:>8s} {:>6s} {:>10s}'.format(
        'm6A_Q', 'n', 'med_PA', 'mean_PA', 'm6A_rng', 'DZ%', 'mean_m6A'))
    print('-' * 70)

    q_labels = ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']
    q_data = {}
    for q in q_labels:
        sub = ars_reg_all[ars_reg_all['m6a_q'] == q]
        n = len(sub)
        if n == 0: continue
        med_pa = sub['polya_length'].median()
        mean_pa = sub['polya_length'].mean()
        mean_m6a = sub['m6a_per_kb'].mean()
        dz = (sub['polya_length'] < 30).sum() / n * 100
        m6a_rng = f'{sub["m6a_per_kb"].min():.1f}-{sub["m6a_per_kb"].max():.1f}'
        q_data[q] = sub['polya_length'].values

        print('{:<12s} {:>5d} {:>8.1f} {:>8.1f} {:>8s} {:>5.1f}% {:>10.2f}'.format(
            q, n, med_pa, mean_pa, m6a_rng, dz, mean_m6a))

    # Q1 vs Q4 test
    if 'Q1(low)' in q_data and 'Q4(high)' in q_data:
        _, p_q14, _ = mw_test(q_data['Q4(high)'], q_data['Q1(low)'])
        delta_q = np.median(q_data['Q4(high)']) - np.median(q_data['Q1(low)'])
        print(f'\n  Q4 vs Q1 poly(A) delta: {delta_q:+.1f} nt, p={p_q14:.2e} {pval_stars(p_q14)}')
        print(f'  Interpretation: {"High m6A protects poly(A) even in regulatory L1" if delta_q > 0 and p_q14 < 0.05 else "Effect not significant at this sample size"}')

    # Also compare: HeLa (normal) regulatory by m6A quartile for reference
    print('\n  [Reference: HeLa (normal) regulatory L1 by m6A quartile]')
    hela_reg = df[(~df['is_stress']) & (df['cat'] == 'Regulatory')].copy()
    if len(hela_reg) >= 20:
        try:
            hela_reg['m6a_q'] = pd.qcut(hela_reg['m6a_per_kb'], q=4,
                                          labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'],
                                          duplicates='drop')
        except ValueError:
            bounds_h = hela_reg['m6a_per_kb'].quantile([0.25, 0.5, 0.75]).values
            def q_assign_h(v):
                if v <= bounds_h[0]: return 'Q1(low)'
                elif v <= bounds_h[1]: return 'Q2'
                elif v <= bounds_h[2]: return 'Q3'
                else: return 'Q4(high)'
            hela_reg['m6a_q'] = hela_reg['m6a_per_kb'].apply(q_assign_h)

        print('  {:<12s} {:>5s} {:>8s} {:>6s}'.format('m6A_Q', 'n', 'med_PA', 'DZ%'))
        for q in q_labels:
            sub = hela_reg[hela_reg['m6a_q'] == q]
            n = len(sub)
            if n == 0: continue
            print('  {:<12s} {:>5d} {:>8.1f} {:>5.1f}%'.format(
                q, n, sub['polya_length'].median(),
                (sub['polya_length'] < 30).sum() / n * 100))
else:
    print('  Insufficient reads for quartile analysis.')


# ── D2: Direct test — m6A-matched regulatory vs non-regulatory under stress
print_subheader('D2. m6A-Matched: Regulatory vs Non-Regulatory Poly(A) Under Stress')
print('   Key test: within same m6A quartile, is regulatory L1 STILL shorter?')

ars_only = df[df['is_stress']].copy()
# Use global m6A quartile boundaries
ars_only['m6a_q'] = ars_only['m6a_per_kb'].apply(assign_m6a_quartile)

print('\n{:<5s} {:>14s} {:>5s} {:>8s} {:>7s}  {:>14s} {:>5s} {:>8s} {:>7s}  {:>8s} {:>10s}'.format(
    'Q', 'Regulatory', 'n', 'med_PA', 'DZ%', 'Txd+Qui+Het', 'n', 'med_PA', 'DZ%', 'Delta', 'p-val'))
print('-' * 120)

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    r = ars_only[(ars_only['cat'] == 'Regulatory') & (ars_only['m6a_q'] == q)]
    nr = ars_only[(ars_only['cat'].isin(['Transcribed', 'Quiescent', 'Het'])) & (ars_only['m6a_q'] == q)]

    n_r = len(r)
    n_nr = len(nr)
    pa_r = r['polya_length'].median() if n_r > 0 else np.nan
    pa_nr = nr['polya_length'].median() if n_nr > 0 else np.nan
    dz_r = (r['polya_length'] < 30).sum() / n_r * 100 if n_r > 0 else 0
    dz_nr = (nr['polya_length'] < 30).sum() / n_nr * 100 if n_nr > 0 else 0
    delta = pa_r - pa_nr if n_r > 0 and n_nr > 0 else np.nan
    _, p, _ = mw_test(r['polya_length'].values, nr['polya_length'].values) if n_r >= 3 and n_nr >= 3 else (None, np.nan, '')

    print('{:<5s} {:>14s} {:>5d} {:>8.1f} {:>6.1f}% {:>14s} {:>5d} {:>8.1f} {:>6.1f}%  {:>+8.1f} p={}'.format(
        q, '', n_r, pa_r, dz_r, '', n_nr, pa_nr, dz_nr, delta,
        f'{p:.2e}{pval_stars(p)}' if not np.isnan(p) else 'n/a'))


# ── SECTION E: Arsenite-Specific Regulatory L1 Loci ───────────────────────
print_header('E. ARSENITE-SPECIFIC REGULATORY L1 LOCI')

# Create locus IDs for regulatory L1 in HeLa and HeLa-Ars
reg_df = df[df['cat'] == 'Regulatory'].copy()

# Create a coarser locus ID: round start to nearest 100bp + gene_id
# This accounts for slight alignment differences between reads at the same locus
reg_df['coarse_locus'] = reg_df['chr'] + ':' + (reg_df['start'] // 100 * 100).astype(str) + ':' + reg_df['gene_id']

hela_reg = reg_df[~reg_df['is_stress']]
ars_reg  = reg_df[reg_df['is_stress']]

hela_loci = set(hela_reg['coarse_locus'].unique())
ars_loci  = set(ars_reg['coarse_locus'].unique())
shared_loci = hela_loci & ars_loci
hela_only_loci = hela_loci - ars_loci
ars_only_loci  = ars_loci - hela_loci

print(f'Regulatory L1 loci (coarse ID):')
print(f'  HeLa total:     {len(hela_loci)}')
print(f'  HeLa-Ars total: {len(ars_loci)}')
print(f'  Shared:          {len(shared_loci)}')
print(f'  HeLa-only:       {len(hela_only_loci)}')
print(f'  Ars-only:        {len(ars_only_loci)}')
print(f'  Union:            {len(hela_loci | ars_loci)}')

# Compare poly(A) for shared vs Ars-only loci
shared_reads = ars_reg[ars_reg['coarse_locus'].isin(shared_loci)]
ars_only_reads = ars_reg[ars_reg['coarse_locus'].isin(ars_only_loci)]
hela_only_reads = hela_reg[hela_reg['coarse_locus'].isin(hela_only_loci)]

print(f'\nPoly(A) comparison:')
if len(shared_reads) > 0:
    print(f'  Shared loci (Ars reads):  n={len(shared_reads)}, median PA={shared_reads["polya_length"].median():.1f}')
if len(ars_only_reads) > 0:
    print(f'  Ars-only loci:            n={len(ars_only_reads)}, median PA={ars_only_reads["polya_length"].median():.1f}')
if len(hela_only_reads) > 0:
    print(f'  HeLa-only loci (HeLa):    n={len(hela_only_reads)}, median PA={hela_only_reads["polya_length"].median():.1f}')

if len(shared_reads) >= 3 and len(ars_only_reads) >= 3:
    _, p, _ = mw_test(ars_only_reads['polya_length'].values, shared_reads['polya_length'].values)
    delta = ars_only_reads['polya_length'].median() - shared_reads['polya_length'].median()
    print(f'  Ars-only vs Shared delta: {delta:+.1f} nt, p={p:.2e} {pval_stars(p)}')

# Shared loci: compare HeLa vs HeLa-Ars poly(A)
shared_hela = hela_reg[hela_reg['coarse_locus'].isin(shared_loci)]
if len(shared_hela) >= 3 and len(shared_reads) >= 3:
    _, p, _ = mw_test(shared_reads['polya_length'].values, shared_hela['polya_length'].values)
    delta_shared = shared_reads['polya_length'].median() - shared_hela['polya_length'].median()
    print(f'  Shared loci HeLa→Ars delta: {delta_shared:+.1f} nt, p={p:.2e} {pval_stars(p)}')

# m6A comparison
print(f'\nm6A/kb comparison:')
if len(shared_reads) > 0:
    print(f'  Shared loci (Ars reads): median m6A/kb={shared_reads["m6a_per_kb"].median():.2f}')
if len(ars_only_reads) > 0:
    print(f'  Ars-only loci:           median m6A/kb={ars_only_reads["m6a_per_kb"].median():.2f}')

# Decay zone in Ars-only
print(f'\nDecay zone (poly(A) < 30) in Ars regulatory loci:')
for name, sub in [('Shared', shared_reads), ('Ars-only', ars_only_reads)]:
    if len(sub) > 0:
        dz = (sub['polya_length'] < 30).sum()
        print(f'  {name}: {dz}/{len(sub)} ({dz/len(sub)*100:.1f}%)')

# E2: Enhancer vs Promoter split
print_subheader('E2. Enhancer vs Promoter Split (HeLa + HeLa-Ars, Ancient)')

for state in ['Enhancer', 'Promoter']:
    sub = df[df['chromhmm_group'] == state]
    hela = sub[~sub['is_stress']]
    ars  = sub[sub['is_stress']]

    print(f'\n  {state}:')
    print(f'    HeLa:     n={len(hela)}, median PA={hela["polya_length"].median():.1f}, '
          f'median m6A/kb={hela["m6a_per_kb"].median():.2f}, '
          f'DZ={((hela["polya_length"]<30).sum()/len(hela)*100):.1f}%' if len(hela) > 0 else f'    HeLa: n=0')
    print(f'    HeLa-Ars: n={len(ars)}, median PA={ars["polya_length"].median():.1f}, '
          f'median m6A/kb={ars["m6a_per_kb"].median():.2f}, '
          f'DZ={((ars["polya_length"]<30).sum()/len(ars)*100):.1f}%' if len(ars) > 0 else f'    HeLa-Ars: n=0')
    if len(hela) >= 3 and len(ars) >= 3:
        _, p, _ = mw_test(ars['polya_length'].values, hela['polya_length'].values)
        delta = ars['polya_length'].median() - hela['polya_length'].median()
        print(f'    Delta: {delta:+.1f} nt, p={p:.2e} {pval_stars(p)}')


# ── SUMMARY TABLE ──────────────────────────────────────────────────────────
print_header('SUMMARY: KEY FINDINGS')

print("""
1. CHROMATIN STATE HIERARCHY:
   Regulatory (Enh+Prom) > Transcribed > Quiescent >> Heterochromatin
   in arsenite poly(A) shortening severity.

2. m6A PROTECTION WITHIN REGULATORY L1:
   Even within regulatory L1, high m6A reads retain longer poly(A) under stress.

3. CHROMATIN AS INDEPENDENT VULNERABILITY:
   m6A-matched regulatory L1 is STILL shorter than non-regulatory under stress
   = chromatin accessibility is an independent vulnerability axis.

4. ARSENITE-SPECIFIC REGULATORY LOCI:
   New regulatory L1 loci appearing only under arsenite may represent
   stress-activated enhancer L1 or reads that become detectable
   due to altered RNA processing.
""")


# ── Save comprehensive results table ───────────────────────────────────────
print_header('SAVING RESULTS')

# Compile all key statistics into a single TSV
out_rows = []

# Section A results
for _, r in df_a.iterrows():
    out_rows.append({
        'section': 'A_characterization',
        'category': r['state'],
        'metric': 'poly(A)',
        'n_hela': r['n_hela'], 'n_ars': r['n_ars'],
        'value_hela': r['med_pa_hela'], 'value_ars': r['med_pa_ars'],
        'delta': r['delta_pa'], 'p_value': r['p_pa'],
        'note': f"DZ_HeLa={r['dz_frac_hela']:.1f}%, DZ_Ars={r['dz_frac_ars']:.1f}%"
    })
    out_rows.append({
        'section': 'A_characterization',
        'category': r['state'],
        'metric': 'm6A/kb',
        'n_hela': r['n_hela'], 'n_ars': r['n_ars'],
        'value_hela': r['med_m6a_hela'], 'value_ars': r['med_m6a_ars'],
        'delta': r['med_m6a_ars'] - r['med_m6a_hela'] if not np.isnan(r['med_m6a_hela']) and not np.isnan(r['med_m6a_ars']) else np.nan,
        'p_value': np.nan,
        'note': ''
    })

# Section B regulatory vs non-regulatory
for cat in CATS:
    sub = df[df['cat'] == cat]
    hela = sub[~sub['is_stress']]
    ars  = sub[sub['is_stress']]
    if len(hela) > 0 and len(ars) > 0:
        _, p, _ = mw_test(ars['polya_length'].values, hela['polya_length'].values)
        out_rows.append({
            'section': 'B_comparison',
            'category': cat,
            'metric': 'poly(A)_delta',
            'n_hela': len(hela), 'n_ars': len(ars),
            'value_hela': hela['polya_length'].median(),
            'value_ars': ars['polya_length'].median(),
            'delta': ars['polya_length'].median() - hela['polya_length'].median(),
            'p_value': p,
            'note': f"m6A/kb: HeLa={hela['m6a_per_kb'].median():.2f}, Ars={ars['m6a_per_kb'].median():.2f}"
        })

# Section D: m6A quartile within regulatory
if 'q_data' in dir() and len(q_data) > 0:
    for q in q_labels:
        sub_q = ars_reg_all[ars_reg_all['m6a_q'] == q] if 'm6a_q' in ars_reg_all.columns else pd.DataFrame()
        if len(sub_q) > 0:
            out_rows.append({
                'section': 'D_m6a_protection_regulatory',
                'category': q,
                'metric': 'poly(A)_stress_regulatory',
                'n_hela': np.nan, 'n_ars': len(sub_q),
                'value_hela': np.nan,
                'value_ars': sub_q['polya_length'].median(),
                'delta': np.nan,
                'p_value': np.nan,
                'note': f"m6A/kb={sub_q['m6a_per_kb'].mean():.2f}, DZ={(sub_q['polya_length']<30).sum()/len(sub_q)*100:.1f}%"
            })

# Section E loci
out_rows.append({
    'section': 'E_loci',
    'category': 'shared',
    'metric': 'n_loci',
    'n_hela': np.nan, 'n_ars': len(shared_reads),
    'value_hela': np.nan,
    'value_ars': shared_reads['polya_length'].median() if len(shared_reads) > 0 else np.nan,
    'delta': np.nan, 'p_value': np.nan,
    'note': f'{len(shared_loci)} loci'
})
out_rows.append({
    'section': 'E_loci',
    'category': 'ars_only',
    'metric': 'n_loci',
    'n_hela': np.nan, 'n_ars': len(ars_only_reads),
    'value_hela': np.nan,
    'value_ars': ars_only_reads['polya_length'].median() if len(ars_only_reads) > 0 else np.nan,
    'delta': np.nan, 'p_value': np.nan,
    'note': f'{len(ars_only_loci)} loci'
})

out_df = pd.DataFrame(out_rows)
out_df.to_csv(OUT_TSV, sep='\t', index=False)
print(f'Saved: {OUT_TSV}')
print(f'Total rows: {len(out_df)}')
print('\nDone.')
