#!/usr/bin/env python3
"""
Part 4 Analysis 3: L1 host gene functional analysis — ALL cell lines
- Gene length-controlled enrichment (key bias correction)
- Cross-CL consistency stratification
- Psi-stratified host gene function
- Per-CL tissue-specific enrichment
"""

import pandas as pd
import numpy as np
import os
import gseapy
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{BASE}/results_group"
OUTDIR = f"{BASE}/analysis/01_exploration/topic_05_cellline/part4_host_gene_enrichment"
os.makedirs(OUTDIR, exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

CELL_LINES = {
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'Hct116': ['Hct116_3', 'Hct116_4'],
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2': ['HepG2_5', 'HepG2_6'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'MCF7-EV': ['MCF7-EV_1'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

gene_sets_to_test = [
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'KEGG_2021_Human',
    'MSigDB_Hallmark_2020',
]

def is_standard_gene(g):
    if g.startswith(('RP11-', 'RP4-', 'RP5-', 'RP3-', 'RP1-', 'AC0', 'AL0', 'AL1',
                     'AL3', 'AL4', 'AL5', 'AL6', 'AL7', 'AL8', 'AL9',
                     'AP0', 'CTC-', 'CTB-', 'CTD-', 'KB-', 'LA16')):
        return False
    return True

def run_enrichr(gene_list, label, gene_sets_list):
    all_results = []
    for gs in gene_sets_list:
        try:
            enr = gseapy.enrichr(gene_list=gene_list, gene_sets=gs,
                                  organism='human', outdir=None, no_plot=True)
            res = enr.results.copy()
            res['database'] = gs
            res['gene_list'] = label
            all_results.append(res)
        except Exception as e:
            print(f"  WARNING: {gs} failed for {label}: {e}")
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# ─── Load all data ────────────────────────────────────────────────────────────
print("=" * 70)
print("Part 4 Analysis 3: All-CL Host Gene Functional Analysis")
print("=" * 70)

all_dfs = []
for cl, reps in CELL_LINES.items():
    for rep in reps:
        fpath = f"{RESULTS}/{rep}/g_summary/{rep}_L1_summary.tsv"
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, sep='\t')
            df['cell_line'] = cl
            all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)
df_all['l1_age'] = df_all['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')

df_intr = df_all[df_all['TE_group'] == 'intronic'].copy()
print(f"Total intronic L1 reads: {len(df_intr):,} from {df_intr['cell_line'].nunique()} CLs")

# Extract genes
def extract_genes(genes_str):
    if pd.isna(genes_str) or genes_str == '':
        return []
    return [g.strip() for g in str(genes_str).split(';') if g.strip()]

df_intr['host_gene_list'] = df_intr['overlapping_genes'].apply(extract_genes)
df_exp = df_intr.explode('host_gene_list')
df_exp = df_exp[df_exp['host_gene_list'].notna()]
df_exp.rename(columns={'host_gene_list': 'host_gene'}, inplace=True)

# ─── Load gene lengths ───────────────────────────────────────────────────────
print("\nLoading gene lengths from GTF...")
GTF_PATH = f"{BASE}/reference/Human.gtf"
gene_lengths = {}
with open(GTF_PATH) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9 or fields[2] != 'gene':
            continue
        attrs = fields[8]
        gn = None
        for attr in attrs.split(';'):
            attr = attr.strip()
            if attr.startswith('gene_name'):
                gn = attr.split('"')[1] if '"' in attr else attr.split()[-1]
                break
        if gn:
            length = int(fields[4]) - int(fields[3]) + 1
            if gn not in gene_lengths or length > gene_lengths[gn]:
                gene_lengths[gn] = length
print(f"Gene lengths loaded: {len(gene_lengths)} genes")

# ─── Gene summary across all CLs ─────────────────────────────────────────────
gene_summary = df_exp.groupby('host_gene').agg(
    n_reads=('read_id', 'count'),
    n_cell_lines=('cell_line', 'nunique'),
    n_young=('l1_age', lambda x: (x == 'young').sum()),
    n_ancient=('l1_age', lambda x: (x == 'ancient').sum()),
    psi_frac=('psi', 'mean'),
    polya_median=('polya_length', 'median'),
).reset_index()
gene_summary['gene_length'] = gene_summary['host_gene'].map(gene_lengths)
gene_summary = gene_summary.sort_values('n_reads', ascending=False)

all_host_genes = sorted(gene_summary['host_gene'].unique())
print(f"\nTotal unique host genes: {len(all_host_genes)}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: GENE LENGTH-CONTROLLED ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION A: Gene Length-Controlled Enrichment")
print("=" * 70)

# Strategy: Compare L1 host genes to length-matched non-host genes
# 1. Get all genes with length info
# 2. For each L1 host gene, find non-host genes with similar length
# 3. Run enrichment on L1 host genes with length-matched background

# Build length bins
host_with_len = gene_summary[gene_summary['gene_length'].notna()].copy()
all_genes_with_len = {g: l for g, l in gene_lengths.items()}

# Non-host genes
non_host_genes = {g: l for g, l in all_genes_with_len.items()
                  if g not in set(all_host_genes)}
print(f"Host genes with length: {len(host_with_len)}")
print(f"Non-host genes with length: {len(non_host_genes)}")

# Length-matched background: for each host gene, sample 3 non-host genes from same length decile
host_lens = host_with_len['gene_length'].values
non_host_df = pd.DataFrame({'gene': list(non_host_genes.keys()),
                             'length': list(non_host_genes.values())})

# Assign length deciles based on host gene distribution
decile_edges = np.percentile(host_lens, np.arange(0, 110, 10))
non_host_df['decile'] = np.digitize(non_host_df['length'], decile_edges)
host_with_len['decile'] = np.digitize(host_with_len['gene_length'].values, decile_edges)

# Build length-matched background
np.random.seed(42)
bg_genes = []
for dec in range(1, 12):
    pool = non_host_df[non_host_df['decile'] == dec]['gene'].tolist()
    n_host_in_dec = (host_with_len['decile'] == dec).sum()
    n_sample = min(n_host_in_dec * 3, len(pool))
    if pool and n_sample > 0:
        sampled = np.random.choice(pool, size=n_sample, replace=False)
        bg_genes.extend(sampled)

bg_genes_filt = [g for g in bg_genes if is_standard_gene(g)]
print(f"Length-matched background genes: {len(bg_genes_filt)}")

# Compare: L1 host vs length-matched background using Fisher's exact test per GO term
# Run enrichment on L1 host genes
host_filt = [g for g in all_host_genes if is_standard_gene(g)]

print(f"\nRunning enrichment: L1 host genes (n={len(host_filt)})...")
results_host = run_enrichr(host_filt, 'L1_host', gene_sets_to_test)

print(f"Running enrichment: Length-matched background (n={len(bg_genes_filt)})...")
results_bg = run_enrichr(bg_genes_filt, 'length_matched_bg', gene_sets_to_test)

# Compare: which terms enriched in host but NOT in background?
if len(results_host) > 0 and len(results_bg) > 0:
    host_sig = results_host[results_host['Adjusted P-value'] < 0.05][['Term', 'database', 'P-value', 'Adjusted P-value', 'Overlap', 'Genes']].copy()
    host_sig.rename(columns={'P-value': 'host_pval', 'Adjusted P-value': 'host_adjp',
                             'Overlap': 'host_overlap', 'Genes': 'host_genes'}, inplace=True)

    bg_terms = results_bg[['Term', 'database', 'P-value', 'Adjusted P-value']].copy()
    bg_terms.rename(columns={'P-value': 'bg_pval', 'Adjusted P-value': 'bg_adjp'}, inplace=True)

    merged = host_sig.merge(bg_terms, on=['Term', 'database'], how='left')
    merged['L1_specific'] = merged['bg_adjp'] > 0.05  # significant in host but not in background

    print(f"\nTerms significant in L1 host: {len(host_sig)}")
    print(f"Terms ALSO significant in length-matched BG: {(~merged['L1_specific']).sum()}")
    print(f"Terms L1-SPECIFIC (not in BG): {merged['L1_specific'].sum()}")

    if merged['L1_specific'].any():
        print(f"\n*** L1-SPECIFIC enrichment (beyond gene length): ***")
        for _, row in merged[merged['L1_specific']].iterrows():
            print(f"  [{row['database']}] {row['Term'][:65]}")
            print(f"    Host: {row['host_overlap']}, adj.p={row['host_adjp']:.2e} | BG adj.p={row['bg_adjp']:.2e}")

    print(f"\nTerms shared with background (gene length-driven):")
    for _, row in merged[~merged['L1_specific']].iterrows():
        print(f"  [{row['database']}] {row['Term'][:65]}")
        print(f"    Host adj.p={row['host_adjp']:.2e} | BG adj.p={row['bg_adjp']:.2e}")

    merged.to_csv(f"{OUTDIR}/length_controlled_enrichment.tsv", sep='\t', index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: CROSS-CL CONSISTENCY STRATIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION B: Cross-CL Consistency — Ubiquitous vs CL-Specific Host Genes")
print("=" * 70)

# Genes detected in >=5 CLs vs 1-2 CLs
base_cls = [cl for cl in CELL_LINES if cl not in ('HeLa-Ars', 'MCF7-EV')]

cl_gene_sets = {}
for cl in base_cls:
    mask = df_exp['cell_line'] == cl
    cl_gene_sets[cl] = set(df_exp[mask]['host_gene'].unique())

# Count CL presence per gene
gene_cl_count = Counter()
for cl, genes in cl_gene_sets.items():
    for g in genes:
        gene_cl_count[g] += 1

gene_summary['n_base_cl'] = gene_summary['host_gene'].map(
    lambda g: gene_cl_count.get(g, 0)
)

# Stratify
ubiq_genes = gene_summary[gene_summary['n_base_cl'] >= 5]['host_gene'].tolist()
mid_genes = gene_summary[(gene_summary['n_base_cl'] >= 3) & (gene_summary['n_base_cl'] < 5)]['host_gene'].tolist()
rare_genes = gene_summary[gene_summary['n_base_cl'] <= 2]['host_gene'].tolist()

print(f"Ubiquitous (>=5 CLs): {len(ubiq_genes)} genes")
print(f"Mid (3-4 CLs): {len(mid_genes)} genes")
print(f"Rare (1-2 CLs): {len(rare_genes)} genes")

# Gene length by stratum
for label, genes in [('Ubiquitous', ubiq_genes), ('Mid', mid_genes), ('Rare', rare_genes)]:
    lens = [gene_lengths[g] for g in genes if g in gene_lengths]
    if lens:
        print(f"  {label}: median length={np.median(lens)/1000:.1f}kb (n={len(lens)})")

# Enrichment for ubiquitous genes
ubiq_filt = [g for g in ubiq_genes if is_standard_gene(g)]
if len(ubiq_filt) >= 10:
    print(f"\nRunning enrichment: Ubiquitous host genes (n={len(ubiq_filt)})...")
    results_ubiq = run_enrichr(ubiq_filt, 'ubiquitous', gene_sets_to_test)
    sig_u = results_ubiq[results_ubiq['Adjusted P-value'] < 0.05]
    print(f"Significant terms: {len(sig_u)}")
    for _, row in results_ubiq.nsmallest(15, 'P-value').iterrows():
        sig_mark = "***" if row['Adjusted P-value'] < 0.05 else ""
        print(f"  [{row['database']}] {row['Term'][:60]} | {row['Overlap']} | p={row['P-value']:.2e} {sig_mark}")
    results_ubiq.to_csv(f"{OUTDIR}/enrichr_ubiquitous_all.tsv", sep='\t', index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: PSI-STRATIFIED HOST GENE ENRICHMENT (ALL CLs)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION C: Psi-Stratified Host Gene Enrichment (All CLs)")
print("=" * 70)

# Genes where L1 has high psi (>50% reads have psi) vs low psi
high_psi_genes = gene_summary[gene_summary['psi_frac'] > 0.5]['host_gene'].tolist()
low_psi_genes = gene_summary[gene_summary['psi_frac'] <= 0.5]['host_gene'].tolist()

# Require >=3 reads for reliable psi estimate
gene_sum_3 = gene_summary[gene_summary['n_reads'] >= 3]
high_psi_3 = gene_sum_3[gene_sum_3['psi_frac'] > 0.5]['host_gene'].tolist()
low_psi_3 = gene_sum_3[gene_sum_3['psi_frac'] <= 0.5]['host_gene'].tolist()

print(f"High-psi host genes (>=3 reads, >50% psi): {len(high_psi_3)}")
print(f"Low-psi host genes (>=3 reads, <=50% psi): {len(low_psi_3)}")

# Length comparison
h_lens = [gene_lengths[g] for g in high_psi_3 if g in gene_lengths]
l_lens = [gene_lengths[g] for g in low_psi_3 if g in gene_lengths]
if h_lens and l_lens:
    print(f"  High-psi median length: {np.median(h_lens)/1000:.1f}kb")
    print(f"  Low-psi median length: {np.median(l_lens)/1000:.1f}kb")
    u, p = stats.mannwhitneyu(h_lens, l_lens)
    print(f"  Length difference: p={p:.2e}")

# Enrichment
high_filt = [g for g in high_psi_3 if is_standard_gene(g)]
low_filt = [g for g in low_psi_3 if is_standard_gene(g)]

if len(high_filt) >= 10:
    print(f"\nRunning enrichment: High-psi host genes (n={len(high_filt)})...")
    results_high = run_enrichr(high_filt, 'high_psi', gene_sets_to_test)
    sig_h = results_high[results_high['Adjusted P-value'] < 0.05]
    print(f"Significant terms: {len(sig_h)}")
    for _, row in results_high.nsmallest(10, 'P-value').iterrows():
        sig_mark = "***" if row['Adjusted P-value'] < 0.05 else ""
        print(f"  [{row['database']}] {row['Term'][:60]} | {row['Overlap']} | p={row['P-value']:.2e} {sig_mark}")
    results_high.to_csv(f"{OUTDIR}/enrichr_high_psi.tsv", sep='\t', index=False)

if len(low_filt) >= 10:
    print(f"\nRunning enrichment: Low-psi host genes (n={len(low_filt)})...")
    results_low = run_enrichr(low_filt, 'low_psi', gene_sets_to_test)
    sig_l = results_low[results_low['Adjusted P-value'] < 0.05]
    print(f"Significant terms: {len(sig_l)}")
    for _, row in results_low.nsmallest(10, 'P-value').iterrows():
        sig_mark = "***" if row['Adjusted P-value'] < 0.05 else ""
        print(f"  [{row['database']}] {row['Term'][:60]} | {row['Overlap']} | p={row['P-value']:.2e} {sig_mark}")
    results_low.to_csv(f"{OUTDIR}/enrichr_low_psi.tsv", sep='\t', index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: PER-CL TISSUE-SPECIFIC ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION D: Per-CL Top Host Gene Functions")
print("=" * 70)

# For each CL, find CL-specific host genes and their top enrichments
for cl in base_cls:
    cl_genes = sorted(cl_gene_sets[cl])
    # CL-specific: in this CL but <2 others
    cl_specific = [g for g in cl_genes if gene_cl_count[g] <= 2]
    cl_specific_filt = [g for g in cl_specific if is_standard_gene(g)]

    print(f"\n{cl}: {len(cl_genes)} host genes, {len(cl_specific_filt)} CL-specific (<=2 CLs)")

    if len(cl_specific_filt) >= 20:
        res_cl = run_enrichr(cl_specific_filt, f'{cl}_specific', ['GO_Biological_Process_2023', 'MSigDB_Hallmark_2020'])
        sig_cl = res_cl[res_cl['Adjusted P-value'] < 0.05]
        if len(sig_cl) > 0:
            print(f"  *** Significant terms: {len(sig_cl)}")
            for _, row in sig_cl.nsmallest(5, 'Adjusted P-value').iterrows():
                print(f"    {row['Term'][:60]} | {row['Overlap']} | adj.p={row['Adjusted P-value']:.2e}")
        else:
            for _, row in res_cl.nsmallest(3, 'P-value').iterrows():
                print(f"    {row['Term'][:60]} | {row['Overlap']} | p={row['P-value']:.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E: COMPREHENSIVE GENE-LEVEL TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION E: Comprehensive Host Gene Table")
print("=" * 70)

# Save comprehensive gene table
gene_summary['gene_length_kb'] = gene_summary['gene_length'] / 1000
gene_summary['n_base_cl'] = gene_summary['host_gene'].map(lambda g: gene_cl_count.get(g, 0))
gene_summary['psi_category'] = np.where(gene_summary['psi_frac'] > 0.5, 'high', 'low')

# Flag known functional categories
DDR_SET = {'BRCA1', 'BRCA2', 'ATR', 'ATM', 'ATRX', 'RAD51', 'RAD50', 'FANCC', 'FANCD2',
           'FANCI', 'ZRANB3', 'SMARCAL1', 'RB1', 'NBN', 'BLM', 'WRN', 'XRCC5', 'XRCC6',
           'CHEK1', 'CHEK2', 'PARP1', 'TP53', 'MRE11'}
IMMUNE_SET = {'IFNAR1', 'IFNAR2', 'IFNGR1', 'IFNGR2', 'IRF1', 'IRF3', 'IRF7',
              'STAT1', 'STAT2', 'JAK1', 'JAK2', 'TLR3', 'TLR7', 'TLR8',
              'SAMHD1', 'TRIM5', 'MOV10', 'ADAR', 'APOBEC3A', 'APOBEC3B'}
STRESS_SET = {'NFE2L2', 'KEAP1', 'HMOX1', 'NQO1', 'SOD1', 'SOD2', 'GPX1',
              'HSP90AA1', 'HSP90AB1', 'HSPA1A', 'HSPA5', 'HSF1',
              'EIF2AK1', 'EIF2AK2', 'EIF2AK3', 'EIF2AK4', 'ATF4', 'ATF6', 'XBP1',
              'DDIT3', 'GADD45A', 'GADD45B'}
NEURONAL_SET = {'GPHN', 'UBE3A', 'CTNND2', 'DLG2', 'GABRG3', 'GRIA4', 'NRXN1', 'NRXN3',
                'CNTNAP2', 'RBFOX1', 'PARK2', 'LSAMP', 'CSMD1', 'PTPRD', 'DCC'}

gene_summary['functional_category'] = gene_summary['host_gene'].apply(
    lambda g: 'DDR' if g in DDR_SET else
              'Immune' if g in IMMUNE_SET else
              'Stress' if g in STRESS_SET else
              'Neuronal' if g in NEURONAL_SET else
              ''
)

cols_out = ['host_gene', 'n_reads', 'n_cell_lines', 'n_base_cl', 'n_young', 'n_ancient',
            'psi_frac', 'psi_category', 'polya_median', 'gene_length_kb', 'functional_category']
gene_summary[cols_out].to_csv(f"{OUTDIR}/comprehensive_host_gene_table.tsv", sep='\t', index=False)

# Summary stats
print(f"\nFunctional category breakdown among L1 host genes:")
for cat in ['DDR', 'Immune', 'Stress', 'Neuronal']:
    sub = gene_summary[gene_summary['functional_category'] == cat]
    total_in_set = len(DDR_SET if cat == 'DDR' else IMMUNE_SET if cat == 'Immune'
                       else STRESS_SET if cat == 'Stress' else NEURONAL_SET)
    print(f"  {cat}: {len(sub)}/{total_in_set} found | "
          f"mean reads={sub['n_reads'].mean():.1f} | "
          f"mean CLs={sub['n_cell_lines'].mean():.1f} | "
          f"mean psi={sub['psi_frac'].mean():.2f} | "
          f"median length={sub['gene_length_kb'].median():.0f}kb")

print(f"\nAll results saved to: {OUTDIR}")
print("Done!")
