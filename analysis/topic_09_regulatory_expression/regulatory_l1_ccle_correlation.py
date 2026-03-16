#!/usr/bin/env python3
"""
Regulatory L1 expression × host gene expression (CCLE) correlation.

Question: When a cell line expresses L1 at a regulatory (enhancer/promoter) position,
is the host gene also more highly expressed in that cell line?

Approach:
1. Binary: L1 present/absent at regulatory locus × host gene TPM
2. Within-gene paired: For genes with L1 in some CLs, compare TPM in L1+ vs L1- CLs
3. Enhancer vs Promoter L1
4. Quantitative: L1 read count × host gene TPM correlation
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, sys, json

BASEDIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
CHROMHMM_DIR = f'{BASEDIR}/analysis/01_exploration/topic_08_regulatory_chromatin'
CCLE_DIR = '/vault/external-datasets/2026/CCLE_DepMap'
OUTDIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTDIR, exist_ok=True)

# ── Cell line mapping: our names → CCLE ModelIDs ──
CL_TO_ACH = {
    'A549':   'ACH-000681',
    'HEYA8':  'ACH-000542',
    'Hct116': 'ACH-000971',
    'HeLa':   'ACH-001086',
    'HepG2':  'ACH-000739',
    'K562':   'ACH-000551',
    'MCF7':   'ACH-000019',
    'SHSY5Y': 'ACH-001188',
    # H9 (ESC) not in CCLE expression data
}

# ════════════════════════════════════════
# 1. Load CCLE expression data
# ════════════════════════════════════════
print("=== 1. Loading CCLE expression data ===")

ccle_file = f'{CCLE_DIR}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv'
if not os.path.exists(ccle_file) or os.path.getsize(ccle_file) < 1000:
    print(f"ERROR: CCLE file not found: {ccle_file}")
    sys.exit(1)

# Load — ModelID is in column 2 (0-indexed), gene data starts at column 6
print(f"  Loading {os.path.basename(ccle_file)}...")
ccle_raw = pd.read_csv(ccle_file)
print(f"  Raw shape: {ccle_raw.shape}")

# Filter to default entries only and set ModelID as index
ccle_raw = ccle_raw[ccle_raw['IsDefaultEntryForModel'] == 'Yes'].copy()
ccle_raw = ccle_raw.set_index('ModelID')

# Get gene columns (skip metadata columns)
meta_cols = {'SequencingID', 'IsDefaultEntryForModel', 'ModelConditionID', 'IsDefaultEntryForMC'}
gene_cols = [c for c in ccle_raw.columns if c not in meta_cols]
ccle = ccle_raw[gene_cols].copy()
print(f"  Expression matrix: {ccle.shape[0]} cell lines × {ccle.shape[1]} genes")

# ── Parse gene names: "GENE (EntrezID)" → HUGO symbol ──
gene_map = {}
for col in ccle.columns:
    if '(' in col:
        hugo = col.split('(')[0].strip()
        gene_map[hugo] = col
    else:
        gene_map[col] = col
print(f"  Mapped {len(gene_map)} CCLE genes")

# ── Verify our cell lines are in CCLE ──
found_cls = {}
for our_name, ach_id in CL_TO_ACH.items():
    if ach_id in ccle.index:
        found_cls[our_name] = ach_id
        print(f"  ✓ {our_name} → {ach_id}")
    else:
        print(f"  ✗ {our_name} ({ach_id}) not in expression data")
print(f"  Matched: {len(found_cls)}/{len(CL_TO_ACH)}")

# ════════════════════════════════════════
# 2. Load regulatory L1 data
# ════════════════════════════════════════
print("\n=== 2. Loading regulatory L1 data ===")

reg_reads = pd.read_csv(f'{CHROMHMM_DIR}/stress_gene_analysis/regulatory_l1_per_read.tsv', sep='\t')
reg_reads = reg_reads[reg_reads['l1_age'] == 'ancient'].copy()
reg_reads = reg_reads[reg_reads['condition'] == 'normal'].copy()
reg_reads = reg_reads[~reg_reads['cellline'].isin(['HeLa-Ars', 'MCF7-EV', 'H9'])].copy()
print(f"  Ancient regulatory reads (normal, 8 base CLs): {len(reg_reads)}")
print(f"  Cell lines: {sorted(reg_reads['cellline'].unique())}")

# Build gene × CL read count matrix
reg_genes = reg_reads.groupby(['host_gene', 'cellline']).agg(
    n_reads=('read_id', 'count'),
    median_polya=('polya_length', 'median'),
    median_m6a=('m6a_per_kb', 'median'),
    n_enhancer=('chromhmm_group', lambda x: (x == 'Enhancer').sum()),
    n_promoter=('chromhmm_group', lambda x: (x == 'Promoter').sum()),
).reset_index()

print(f"  Gene × CL combinations: {len(reg_genes)}")
print(f"  Unique host genes: {reg_genes['host_gene'].nunique()}")

# ════════════════════════════════════════
# 3. Build analysis matrix
# ════════════════════════════════════════
print("\n=== 3. Building L1 presence × host gene TPM matrix ===")

results = []
genes_matched = 0
genes_not_found = 0

for host_gene in reg_genes['host_gene'].unique():
    gene_parts = [g.strip() for g in host_gene.split(';')]

    # Find first matching gene in CCLE
    ccle_col = None
    matched_gene = None
    for gp in gene_parts:
        if gp in gene_map:
            ccle_col = gene_map[gp]
            matched_gene = gp
            break
    if ccle_col is None:
        genes_not_found += 1
        continue
    genes_matched += 1

    # Get which CLs have L1 at this gene
    gene_cls = set(reg_genes[reg_genes['host_gene'] == host_gene]['cellline'].values)

    # Determine if mostly enhancer or promoter
    gene_ctx = reg_genes[reg_genes['host_gene'] == host_gene]
    total_enh = gene_ctx['n_enhancer'].sum()
    total_prom = gene_ctx['n_promoter'].sum()
    primary_ctx = 'Enhancer' if total_enh >= total_prom else 'Promoter'

    for our_cl, ach_id in found_cls.items():
        tpm_logp1 = ccle.loc[ach_id, ccle_col]
        l1_present = our_cl in gene_cls

        l1_reads = 0
        l1_m6a = np.nan
        if l1_present:
            cl_data = gene_ctx[gene_ctx['cellline'] == our_cl]
            l1_reads = cl_data['n_reads'].sum()
            l1_m6a = cl_data['median_m6a'].median()

        results.append({
            'host_gene': host_gene,
            'matched_gene': matched_gene,
            'cellline': our_cl,
            'tpm_logp1': tpm_logp1,
            'l1_present': l1_present,
            'l1_reads': l1_reads,
            'l1_m6a': l1_m6a,
            'primary_context': primary_ctx,
            'n_cls_with_l1': len(gene_cls & set(found_cls.keys())),
        })

df = pd.DataFrame(results)
print(f"  Genes matched to CCLE: {genes_matched}")
print(f"  Genes not found: {genes_not_found} (mostly lncRNAs/predicted genes)")
print(f"  Total gene × CL pairs: {len(df)}")
print(f"  L1 present: {df['l1_present'].sum()}, absent: {(~df['l1_present']).sum()}")

# ════════════════════════════════════════
# 4A. Binary: L1 present vs absent → TPM
# ════════════════════════════════════════
print("\n" + "="*60)
print("=== 4A. Binary: L1 present vs absent → host gene TPM ===")
print("="*60)

present = df[df['l1_present']]['tpm_logp1']
absent = df[~df['l1_present']]['tpm_logp1']

print(f"  L1 present: n={len(present)}, median log(TPM+1) = {present.median():.3f}")
print(f"  L1 absent:  n={len(absent)},  median log(TPM+1) = {absent.median():.3f}")
print(f"  Difference: {present.median() - absent.median():+.3f}")

# Convert back to TPM for interpretability
tpm_present = np.exp(present * np.log(2)) - 1  # log2(TPM+1) → TPM
tpm_absent = np.exp(absent * np.log(2)) - 1
print(f"  L1 present: median TPM = {tpm_present.median():.1f}")
print(f"  L1 absent:  median TPM = {tpm_absent.median():.1f}")

mw_stat, mw_p = stats.mannwhitneyu(present, absent, alternative='two-sided')
n1, n2 = len(present), len(absent)
r_rb = 1 - 2 * mw_stat / (n1 * n2)
print(f"  Mann-Whitney U: P = {mw_p:.2e}")
print(f"  Rank-biserial r = {r_rb:.3f}")

# ════════════════════════════════════════
# 4B. Within-gene paired comparison
# ════════════════════════════════════════
print("\n" + "="*60)
print("=== 4B. Within-gene: paired L1+ vs L1- cell lines ===")
print("="*60)

multi_genes = df.groupby('host_gene').agg(
    n_with_l1=('l1_present', 'sum'),
    n_without_l1=('l1_present', lambda x: (~x).sum()),
).reset_index()
multi_genes = multi_genes[(multi_genes['n_with_l1'] >= 1) & (multi_genes['n_without_l1'] >= 1)]
print(f"  Genes with both L1+ and L1- cell lines: {len(multi_genes)}")

paired_deltas = []
for _, row in multi_genes.iterrows():
    gene = row['host_gene']
    gdata = df[df['host_gene'] == gene]
    tpm_with = gdata[gdata['l1_present']]['tpm_logp1'].median()
    tpm_without = gdata[~gdata['l1_present']]['tpm_logp1'].median()
    delta = tpm_with - tpm_without
    matched = gdata.iloc[0]['matched_gene']
    ctx = gdata.iloc[0]['primary_context']
    paired_deltas.append({
        'host_gene': gene,
        'matched_gene': matched,
        'primary_context': ctx,
        'tpm_l1_present': tpm_with,
        'tpm_l1_absent': tpm_without,
        'delta_tpm': delta,
        'n_with': int(row['n_with_l1']),
        'n_without': int(row['n_without_l1']),
    })

paired_df = pd.DataFrame(paired_deltas)
n_pos = (paired_df['delta_tpm'] > 0).sum()
n_neg = (paired_df['delta_tpm'] < 0).sum()
n_zero = (paired_df['delta_tpm'] == 0).sum()
print(f"  Median Δ log(TPM+1) = {paired_df['delta_tpm'].median():.3f}")
print(f"  Mean Δ log(TPM+1) = {paired_df['delta_tpm'].mean():.3f}")
print(f"  Positive (L1+ > L1-): {n_pos}/{len(paired_df)} ({100*n_pos/len(paired_df):.1f}%)")
print(f"  Negative (L1+ < L1-): {n_neg}/{len(paired_df)}")

if len(paired_df) >= 10:
    wsr_stat, wsr_p = stats.wilcoxon(paired_df['delta_tpm'].dropna())
    print(f"  Wilcoxon signed-rank: P = {wsr_p:.2e}")

    # Sign test (simpler, more robust)
    from scipy.stats import binomtest
    bt = binomtest(n_pos, n_pos + n_neg, 0.5)
    print(f"  Sign test: P = {bt.pvalue:.2e}")

# ════════════════════════════════════════
# 4C. Enhancer vs Promoter L1
# ════════════════════════════════════════
print("\n" + "="*60)
print("=== 4C. Enhancer vs Promoter L1 ===")
print("="*60)

for ctx in ['Enhancer', 'Promoter']:
    ctx_df = df[df['primary_context'] == ctx]
    ctx_present = ctx_df[ctx_df['l1_present']]['tpm_logp1']
    ctx_absent = ctx_df[~ctx_df['l1_present']]['tpm_logp1']
    if len(ctx_present) > 5 and len(ctx_absent) > 5:
        mw, p = stats.mannwhitneyu(ctx_present, ctx_absent, alternative='two-sided')
        r = 1 - 2 * mw / (len(ctx_present) * len(ctx_absent))
        print(f"  {ctx:10s}: L1+ median={ctx_present.median():.3f}, L1- median={ctx_absent.median():.3f}, "
              f"Δ={ctx_present.median()-ctx_absent.median():+.3f}, P={p:.2e}, r={r:.3f}")

    # Paired within-gene for this context
    ctx_paired = paired_df[paired_df['primary_context'] == ctx]
    if len(ctx_paired) >= 5:
        ctx_n_pos = (ctx_paired['delta_tpm'] > 0).sum()
        print(f"             Paired: {ctx_n_pos}/{len(ctx_paired)} positive ({100*ctx_n_pos/len(ctx_paired):.1f}%), "
              f"median Δ={ctx_paired['delta_tpm'].median():+.3f}")

# ════════════════════════════════════════
# 4D. Quantitative: L1 read count × TPM
# ════════════════════════════════════════
print("\n" + "="*60)
print("=== 4D. Quantitative: L1 read count × host gene TPM ===")
print("="*60)

l1_present_df = df[df['l1_present']].copy()
if len(l1_present_df) > 10:
    rho, p = stats.spearmanr(l1_present_df['l1_reads'], l1_present_df['tpm_logp1'])
    print(f"  Spearman(L1 reads, TPM): ρ = {rho:.3f}, P = {p:.2e}, n = {len(l1_present_df)}")

    # Also: log reads vs TPM
    l1_present_df['log_reads'] = np.log2(l1_present_df['l1_reads'] + 1)
    rho2, p2 = stats.spearmanr(l1_present_df['log_reads'], l1_present_df['tpm_logp1'])
    print(f"  Spearman(log2 L1 reads, TPM): ρ = {rho2:.3f}, P = {p2:.2e}")

# ════════════════════════════════════════
# 4E. Top genes: strongest associations
# ════════════════════════════════════════
print("\n" + "="*60)
print("=== 4E. Top genes with strongest L1-TPM association ===")
print("="*60)

paired_df = paired_df.sort_values('delta_tpm', ascending=False)

print("\n  Top 15 L1+ > L1- (positive regulation candidates):")
print(f"  {'Gene':25s} {'Δ':>7s} {'L1+':>7s} {'L1-':>7s} {'n+':>3s} {'n-':>3s} {'Ctx':>8s}")
for _, row in paired_df.head(15).iterrows():
    print(f"  {row['matched_gene']:25s} {row['delta_tpm']:+7.2f} "
          f"{row['tpm_l1_present']:7.2f} {row['tpm_l1_absent']:7.2f} "
          f"{row['n_with']:3.0f} {row['n_without']:3.0f} {row['primary_context']:>8s}")

print(f"\n  Top 15 L1+ < L1- (negative regulation / silencing candidates):")
for _, row in paired_df.tail(15).iterrows():
    print(f"  {row['matched_gene']:25s} {row['delta_tpm']:+7.2f} "
          f"{row['tpm_l1_present']:7.2f} {row['tpm_l1_absent']:7.2f} "
          f"{row['n_with']:3.0f} {row['n_without']:3.0f} {row['primary_context']:>8s}")

# ════════════════════════════════════════
# 5. Save results
# ════════════════════════════════════════
print("\n=== 5. Saving results ===")

df.to_csv(f'{OUTDIR}/l1_ccle_gene_cl_matrix.tsv', sep='\t', index=False)
paired_df.to_csv(f'{OUTDIR}/l1_ccle_paired_comparison.tsv', sep='\t', index=False)

summary = {
    'n_genes_matched': genes_matched,
    'n_genes_not_found': genes_not_found,
    'n_cell_lines': len(found_cls),
    'n_l1_present_pairs': int(df['l1_present'].sum()),
    'n_l1_absent_pairs': int((~df['l1_present']).sum()),
    'median_tpm_l1_present': float(present.median()),
    'median_tpm_l1_absent': float(absent.median()),
    'binary_mw_pvalue': float(mw_p),
    'binary_rank_biserial_r': float(r_rb),
    'paired_n_genes': len(paired_df),
    'paired_median_delta': float(paired_df['delta_tpm'].median()),
    'paired_pct_positive': float(n_pos / max(n_pos + n_neg, 1)),
}
if len(paired_df) >= 10:
    summary['wilcoxon_p'] = float(wsr_p)
    summary['sign_test_p'] = float(bt.pvalue)

with open(f'{OUTDIR}/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  Saved: l1_ccle_gene_cl_matrix.tsv ({len(df)} rows)")
print(f"  Saved: l1_ccle_paired_comparison.tsv ({len(paired_df)} genes)")
print(f"  Saved: analysis_summary.json")
print("\n=== Done ===")
