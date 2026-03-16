#!/usr/bin/env python3
"""
Decay zone enrichment analysis:
Are L1 reads with critically short poly(A) (<30nt) under arsenite
enriched in specific functional contexts?
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
CHROMHMM = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv"
DESEQ2 = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/tetranscripts_output/HeLa_SA_vs_UN_gene_TE_analysis.txt"
SUMMARY_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group")
OUTDIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/decay_zone_enrichment")
OUTDIR.mkdir(parents=True, exist_ok=True)

DECAY_THRESHOLD = 30  # PABP dissociation threshold

# ── 1. Load ChromHMM annotated data ──
print("=" * 70)
print("DECAY ZONE ENRICHMENT ANALYSIS")
print("Critically short poly(A) < 30nt under arsenite stress")
print("=" * 70)

df = pd.read_csv(CHROMHMM, sep='\t')
print(f"\nTotal reads: {len(df):,}")

# Filter: HeLa-Ars, ancient L1
ars = df[(df['cellline'] == 'HeLa-Ars') & (df['l1_age'] == 'ancient')].copy()
print(f"HeLa-Ars ancient L1 reads: {len(ars):,}")

# Also get HeLa baseline for comparison
baseline = df[(df['cellline'] == 'HeLa') & (df['l1_age'] == 'ancient')].copy()
print(f"HeLa baseline ancient L1 reads: {len(baseline):,}")

# Split into critically short vs retained
ars['decay_zone'] = ars['polya_length'] < DECAY_THRESHOLD
n_decay = ars['decay_zone'].sum()
n_retained = (~ars['decay_zone']).sum()
pct_decay = n_decay / len(ars) * 100
print(f"\nArsenite: critically_short (<{DECAY_THRESHOLD}nt): {n_decay:,} ({pct_decay:.1f}%)")
print(f"Arsenite: retained (>={DECAY_THRESHOLD}nt): {n_retained:,} ({100-pct_decay:.1f}%)")

baseline['decay_zone'] = baseline['polya_length'] < DECAY_THRESHOLD
n_decay_bl = baseline['decay_zone'].sum()
pct_decay_bl = n_decay_bl / len(baseline) * 100
print(f"Baseline: critically_short (<{DECAY_THRESHOLD}nt): {n_decay_bl:,} ({pct_decay_bl:.1f}%)")

results = []

# ── 2a. ChromHMM group distribution ──
print("\n" + "=" * 70)
print("2a. ChromHMM GROUP distribution: critically_short vs retained")
print("=" * 70)

ct = pd.crosstab(ars['decay_zone'], ars['chromhmm_group'])
ct.index = ['retained', 'critically_short']
print("\nCounts:")
print(ct.to_string())

# Proportions
props = ct.div(ct.sum(axis=1), axis=0) * 100
print("\nProportions (%):")
print(props.round(2).to_string())

# Chi-square test
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-square: {chi2:.2f}, df={dof}, P={p_chi2:.2e}")

# Per-group Fisher's exact (decay_zone vs not, in group vs not)
print("\nPer-group Fisher's exact tests:")
for grp in sorted(ct.columns):
    a = ct.loc['critically_short', grp] if grp in ct.columns else 0
    b = ct.loc['critically_short'].sum() - a
    c = ct.loc['retained', grp] if grp in ct.columns else 0
    d = ct.loc['retained'].sum() - c
    table = [[a, b], [c, d]]
    odds, p_fisher = stats.fisher_exact(table)
    pct_short = a / (a + b) * 100 if (a + b) > 0 else 0
    pct_ret = c / (c + d) * 100 if (c + d) > 0 else 0
    sig = "***" if p_fisher < 0.001 else "**" if p_fisher < 0.01 else "*" if p_fisher < 0.05 else "ns"
    print(f"  {grp:15s}: short={pct_short:5.1f}% ret={pct_ret:5.1f}% OR={odds:.2f} P={p_fisher:.2e} {sig}")
    results.append({
        'test': 'chromhmm_group',
        'category': grp,
        'n_short': a,
        'pct_short': pct_short,
        'n_retained': c,
        'pct_retained': pct_ret,
        'odds_ratio': odds,
        'p_value': p_fisher
    })

# ── 2a'. ChromHMM state (detailed) ──
print("\n" + "=" * 70)
print("2a'. ChromHMM STATE (15-state) distribution")
print("=" * 70)

ct_state = pd.crosstab(ars['decay_zone'], ars['chromhmm_state'])
ct_state.index = ['retained', 'critically_short']
props_state = ct_state.div(ct_state.sum(axis=1), axis=0) * 100

print("\nPer-state Fisher's exact (top enrichments):")
state_results = []
for state in sorted(ct_state.columns):
    a = ct_state.loc['critically_short', state]
    b = ct_state.loc['critically_short'].sum() - a
    c = ct_state.loc['retained', state]
    d = ct_state.loc['retained'].sum() - c
    odds, p_fisher = stats.fisher_exact([[a, b], [c, d]])
    pct_s = a / (a + b) * 100
    pct_r = c / (c + d) * 100
    state_results.append((state, a, pct_s, c, pct_r, odds, p_fisher))

state_results.sort(key=lambda x: x[5], reverse=True)
for state, a, pct_s, c, pct_r, odds, p in state_results:
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {state:20s}: short={pct_s:5.1f}% ({a:4d}) ret={pct_r:5.1f}% ({c:4d}) OR={odds:.2f} P={p:.2e} {sig}")

# ── 2b. Genomic context (intronic vs intergenic) ──
print("\n" + "=" * 70)
print("2b. Genomic context: intronic vs intergenic")
print("=" * 70)

ct_gc = pd.crosstab(ars['decay_zone'], ars['genomic_context'])
ct_gc.index = ['retained', 'critically_short']
print("\nCounts:")
print(ct_gc.to_string())

props_gc = ct_gc.div(ct_gc.sum(axis=1), axis=0) * 100
print("\nProportions (%):")
print(props_gc.round(2).to_string())

# Fisher's for intronic enrichment
if 'intronic' in ct_gc.columns and 'intergenic' in ct_gc.columns:
    a = ct_gc.loc['critically_short', 'intronic']
    b = ct_gc.loc['critically_short', 'intergenic']
    c = ct_gc.loc['retained', 'intronic']
    d = ct_gc.loc['retained', 'intergenic']
    odds, p = stats.fisher_exact([[a, b], [c, d]])
    print(f"\nIntronic vs Intergenic: OR={odds:.3f}, P={p:.2e}")
    results.append({
        'test': 'genomic_context',
        'category': 'intronic_vs_intergenic',
        'n_short': a,
        'pct_short': a/(a+b)*100,
        'n_retained': c,
        'pct_retained': c/(c+d)*100,
        'odds_ratio': odds,
        'p_value': p
    })

# ── 2c. Host gene DESeq2 mapping ──
print("\n" + "=" * 70)
print("2c. Host gene expression (DESeq2 log2FC) for intronic L1 reads")
print("=" * 70)

# Load L1 summaries for HeLa-Ars to get overlapping_genes
summary_dfs = []
for grp in ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    fpath = SUMMARY_DIR / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
    if fpath.exists():
        tmp = pd.read_csv(fpath, sep='\t', usecols=['read_id', 'overlapping_genes', 'gene_id'])
        summary_dfs.append(tmp)
        print(f"  Loaded {grp}: {len(tmp):,} reads")

summary = pd.concat(summary_dfs, ignore_index=True)
summary = summary.rename(columns={'gene_id': 'l1_subfamily', 'overlapping_genes': 'host_gene'})

# Merge with arsenite data
ars_merged = ars.merge(summary[['read_id', 'host_gene']], on='read_id', how='left')
print(f"  Merged: {ars_merged['host_gene'].notna().sum():,} reads with host gene annotation")

# Load DESeq2
deseq = pd.read_csv(DESEQ2, sep='\t', index_col=0)
# Strip version numbers from gene IDs
deseq.index = deseq.index.str.replace(r'\.\d+$', '', regex=True)
deseq = deseq.dropna(subset=['log2FoldChange', 'padj'])
print(f"  DESeq2 genes with results: {len(deseq):,}")

# For intronic reads, map host gene symbols to ENSEMBL IDs via GTF
# First check if host_gene is symbol or ENSEMBL
print(f"\n  Sample host genes: {ars_merged[ars_merged['host_gene'].notna()]['host_gene'].head(10).tolist()}")

# Host genes are symbols. Need gene name -> ENSEMBL mapping from GTF
# Build quick mapping from DESeq2 file + GTF
import re

print("  Building gene symbol -> ENSEMBL ID mapping from GTF...")
gene_map = {}
gtf_path = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.gtf"
with open(gtf_path) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9 or fields[2] != 'gene':
            continue
        attrs = fields[8]
        gid_match = re.search(r'gene_id "([^"]+)"', attrs)
        gname_match = re.search(r'gene_name "([^"]+)"', attrs)
        if gid_match and gname_match:
            gid = gid_match.group(1).split('.')[0]  # strip version
            gname = gname_match.group(1)
            gene_map[gname] = gid

print(f"  Gene symbol -> ENSEMBL mapping: {len(gene_map):,} genes")

# Map host genes to ENSEMBL and then to log2FC
intronic_ars = ars_merged[(ars_merged['genomic_context'] == 'intronic') & (ars_merged['host_gene'].notna())].copy()
intronic_ars['ensembl_id'] = intronic_ars['host_gene'].map(gene_map)
intronic_ars = intronic_ars[intronic_ars['ensembl_id'].notna()]
intronic_ars['log2fc'] = intronic_ars['ensembl_id'].map(deseq['log2FoldChange'])
intronic_ars['padj'] = intronic_ars['ensembl_id'].map(deseq['padj'])
intronic_ars = intronic_ars[intronic_ars['log2fc'].notna()]

print(f"\n  Intronic reads with DESeq2 match: {len(intronic_ars):,}")

# Compare log2FC between decay zone and retained
decay_fc = intronic_ars[intronic_ars['decay_zone']]['log2fc']
retained_fc = intronic_ars[~intronic_ars['decay_zone']]['log2fc']

print(f"\n  Critically short (n={len(decay_fc):,}): mean log2FC = {decay_fc.mean():.3f} (median {decay_fc.median():.3f})")
print(f"  Retained (n={len(retained_fc):,}): mean log2FC = {retained_fc.mean():.3f} (median {retained_fc.median():.3f})")

mwu_stat, mwu_p = stats.mannwhitneyu(decay_fc, retained_fc, alternative='two-sided')
print(f"  MWU P = {mwu_p:.2e}")

results.append({
    'test': 'host_gene_log2fc',
    'category': 'decay_vs_retained',
    'n_short': len(decay_fc),
    'pct_short': decay_fc.mean(),
    'n_retained': len(retained_fc),
    'pct_retained': retained_fc.mean(),
    'odds_ratio': np.nan,
    'p_value': mwu_p
})

# Additional: fraction of sig up/down genes
for label, sub in [('critically_short', intronic_ars[intronic_ars['decay_zone']]),
                   ('retained', intronic_ars[~intronic_ars['decay_zone']])]:
    sig_up = ((sub['padj'] < 0.05) & (sub['log2fc'] > 0)).sum()
    sig_down = ((sub['padj'] < 0.05) & (sub['log2fc'] < 0)).sum()
    ns = ((sub['padj'] >= 0.05) | sub['padj'].isna()).sum()
    total = len(sub)
    print(f"\n  {label} (n={total}): sig_up={sig_up} ({sig_up/total*100:.1f}%), "
          f"sig_down={sig_down} ({sig_down/total*100:.1f}%), ns={ns} ({ns/total*100:.1f}%)")

# ── 2d. m6A/kb distribution (confirmation) ──
print("\n" + "=" * 70)
print("2d. m6A/kb distribution: critically_short vs retained")
print("=" * 70)

decay_m6a = ars[ars['decay_zone']]['m6a_per_kb']
retained_m6a = ars[~ars['decay_zone']]['m6a_per_kb']

print(f"  Critically short: mean m6A/kb = {decay_m6a.mean():.3f} (median {decay_m6a.median():.3f})")
print(f"  Retained: mean m6A/kb = {retained_m6a.mean():.3f} (median {retained_m6a.median():.3f})")
print(f"  Ratio: {retained_m6a.mean() / decay_m6a.mean():.2f}x higher in retained")

mwu_m6a, p_m6a = stats.mannwhitneyu(decay_m6a, retained_m6a, alternative='two-sided')
print(f"  MWU P = {p_m6a:.2e}")

results.append({
    'test': 'm6a_per_kb',
    'category': 'decay_vs_retained',
    'n_short': len(decay_m6a),
    'pct_short': decay_m6a.mean(),
    'n_retained': len(retained_m6a),
    'pct_retained': retained_m6a.mean(),
    'odds_ratio': retained_m6a.mean() / max(decay_m6a.mean(), 0.001),
    'p_value': p_m6a
})

# ── 3. Regulatory L1: fraction entering decay zone ──
print("\n" + "=" * 70)
print("3. Regulatory vs non-regulatory: decay zone entry rate")
print("=" * 70)

# Regulatory = Enhancer + Promoter
ars['is_regulatory'] = ars['chromhmm_group'].isin(['Enhancer', 'Promoter'])

reg = ars[ars['is_regulatory']]
nonreg = ars[~ars['is_regulatory']]

reg_decay = reg['decay_zone'].sum()
reg_total = len(reg)
nonreg_decay = nonreg['decay_zone'].sum()
nonreg_total = len(nonreg)

print(f"  Regulatory (n={reg_total:,}): {reg_decay} in decay zone ({reg_decay/reg_total*100:.1f}%)")
print(f"  Non-regulatory (n={nonreg_total:,}): {nonreg_decay} in decay zone ({nonreg_decay/nonreg_total*100:.1f}%)")

odds, p = stats.fisher_exact([[reg_decay, reg_total - reg_decay],
                               [nonreg_decay, nonreg_total - nonreg_decay]])
print(f"  OR = {odds:.3f}, P = {p:.2e}")

results.append({
    'test': 'regulatory_decay_zone',
    'category': 'regulatory_vs_nonregulatory',
    'n_short': reg_decay,
    'pct_short': reg_decay/reg_total*100,
    'n_retained': nonreg_decay,
    'pct_retained': nonreg_decay/nonreg_total*100,
    'odds_ratio': odds,
    'p_value': p
})

# Also breakdown by specific chromhmm group
print("\n  Per chromhmm_group decay zone entry rate:")
for grp in sorted(ars['chromhmm_group'].unique()):
    sub = ars[ars['chromhmm_group'] == grp]
    n_d = sub['decay_zone'].sum()
    n_t = len(sub)
    print(f"    {grp:15s}: {n_d:4d}/{n_t:5d} = {n_d/n_t*100:.1f}%")

# ── 4. Baseline comparison ──
print("\n" + "=" * 70)
print("4. Baseline (unstressed HeLa) comparison")
print("=" * 70)

baseline['is_regulatory'] = baseline['chromhmm_group'].isin(['Enhancer', 'Promoter'])

for label, sub in [('Baseline', baseline), ('Arsenite', ars)]:
    reg_sub = sub[sub['is_regulatory']]
    nonreg_sub = sub[~sub['is_regulatory']]
    reg_d = reg_sub['decay_zone'].sum()
    nonreg_d = nonreg_sub['decay_zone'].sum()
    print(f"\n  {label}:")
    print(f"    Overall decay zone: {sub['decay_zone'].sum()}/{len(sub)} = {sub['decay_zone'].mean()*100:.1f}%")
    print(f"    Regulatory: {reg_d}/{len(reg_sub)} = {reg_d/len(reg_sub)*100:.1f}%")
    print(f"    Non-regulatory: {nonreg_d}/{len(nonreg_sub)} = {nonreg_d/len(nonreg_sub)*100:.1f}%")

# Stress × regulatory interaction test (2x2x2 → logistic)
from statsmodels.formula.api import logit as sm_logit

combined = pd.concat([ars.assign(stressed=1), baseline.assign(stressed=0)], ignore_index=True)
combined['reg_int'] = combined['is_regulatory'].astype(int)

try:
    model = sm_logit('decay_zone ~ stressed * reg_int', data=combined).fit(disp=0)
    print("\n  Logistic regression: decay_zone ~ stressed * regulatory")
    print(model.summary2().tables[1].to_string())
except Exception as e:
    print(f"\n  Logistic regression failed: {e}")

# ── 5. Poly(A) length distributions by chromhmm ──
print("\n" + "=" * 70)
print("5. Poly(A) distribution by chromhmm_group (arsenite)")
print("=" * 70)

for grp in sorted(ars['chromhmm_group'].unique()):
    sub = ars[ars['chromhmm_group'] == grp]
    print(f"  {grp:15s}: n={len(sub):5d}  median={sub['polya_length'].median():.1f}  "
          f"mean={sub['polya_length'].mean():.1f}  <30nt={sub['decay_zone'].mean()*100:.1f}%")

# ── 6. Unique host genes comparison ──
print("\n" + "=" * 70)
print("6. Host gene categories for intronic decay-zone reads")
print("=" * 70)

# Top host genes in decay zone
decay_genes = intronic_ars[intronic_ars['decay_zone']].groupby('host_gene').agg(
    n_reads=('read_id', 'count'),
    mean_polya=('polya_length', 'mean'),
    log2fc=('log2fc', 'first'),
    padj_deseq=('padj', 'first')
).sort_values('n_reads', ascending=False)

print(f"\n  Unique host genes with decay-zone L1: {len(decay_genes)}")
print(f"  Top 20 host genes:")
print(decay_genes.head(20).to_string())

# ── Save results ──
results_df = pd.DataFrame(results)
results_df.to_csv(OUTDIR / 'decay_zone_enrichment_results.tsv', sep='\t', index=False)
decay_genes.to_csv(OUTDIR / 'decay_zone_host_genes.tsv', sep='\t')

# Save full annotated data for decay zone reads
ars_out = ars[ars['decay_zone']][['read_id', 'chr', 'start', 'end', 'gene_id', 'polya_length',
                                    'm6a_per_kb', 'genomic_context', 'chromhmm_state', 'chromhmm_group']].copy()
ars_out.to_csv(OUTDIR / 'decay_zone_reads.tsv', sep='\t', index=False)

# Also save the intronic reads with host gene + DESeq2 info
intronic_out = intronic_ars[['read_id', 'chr', 'start', 'end', 'gene_id', 'polya_length',
                              'm6a_per_kb', 'chromhmm_group', 'host_gene', 'ensembl_id',
                              'log2fc', 'padj', 'decay_zone']].copy()
intronic_out.to_csv(OUTDIR / 'intronic_reads_with_deseq2.tsv', sep='\t', index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nOutput saved to: {OUTDIR}")
print("Files: decay_zone_enrichment_results.tsv, decay_zone_host_genes.tsv,")
print("       decay_zone_reads.tsv, intronic_reads_with_deseq2.tsv")
