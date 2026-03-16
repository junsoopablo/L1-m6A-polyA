#!/usr/bin/env python3
"""
Part 4 Analysis 1: GO enrichment of L1-harboring host genes
- Extract intronic L1 host genes from all 11 cell lines
- Run Enrichr (GO_BP, GO_MF, KEGG, Reactome)
- Stratify by young vs ancient, by cell line
- Check stress response / DDR / immune gene enrichment
- Gene length bias check
"""

import pandas as pd
import numpy as np
import os
import glob
import gseapy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{BASE}/results_group"
OUTDIR = f"{BASE}/analysis/01_exploration/topic_05_cellline/part4_host_gene_enrichment"
os.makedirs(OUTDIR, exist_ok=True)

# 11 cell lines used in paper (MIN_READS>=200, >=2 reps)
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

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# ─── Load all L1 summary data ────────────────────────────────────────────────
print("=" * 70)
print("Part 4 Analysis 1: GO Enrichment of L1-Harboring Host Genes")
print("=" * 70)

all_dfs = []
for cl, reps in CELL_LINES.items():
    for rep in reps:
        fpath = f"{RESULTS}/{rep}/g_summary/{rep}_L1_summary.tsv"
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, sep='\t')
            df['cell_line'] = cl
            all_dfs.append(df)
        else:
            print(f"  WARNING: {fpath} not found")

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal L1 reads loaded: {len(df_all):,}")
print(f"Cell lines: {df_all['cell_line'].nunique()}")

# ─── Extract host genes ──────────────────────────────────────────────────────
# Filter to intronic only
df_intronic = df_all[df_all['TE_group'] == 'intronic'].copy()
print(f"\nIntronic L1 reads: {len(df_intronic):,} ({len(df_intronic)/len(df_all)*100:.1f}%)")

# Classify young vs ancient
df_intronic['l1_age'] = df_intronic['gene_id'].apply(
    lambda x: 'young' if x in YOUNG_SUBFAMILIES else 'ancient'
)

# Extract host genes (split semicolon-separated)
def extract_genes(genes_str):
    if pd.isna(genes_str) or genes_str == '':
        return []
    return [g.strip() for g in str(genes_str).split(';') if g.strip()]

df_intronic['host_gene_list'] = df_intronic['overlapping_genes'].apply(extract_genes)

# Explode to one row per gene
df_exploded = df_intronic.explode('host_gene_list')
df_exploded = df_exploded[df_exploded['host_gene_list'].notna()]
df_exploded.rename(columns={'host_gene_list': 'host_gene'}, inplace=True)

print(f"Unique host genes (all): {df_exploded['host_gene'].nunique()}")
print(f"Unique host genes (young L1): {df_exploded[df_exploded['l1_age']=='young']['host_gene'].nunique()}")
print(f"Unique host genes (ancient L1): {df_exploded[df_exploded['l1_age']=='ancient']['host_gene'].nunique()}")

# ─── Gene summary table ──────────────────────────────────────────────────────
gene_summary = df_exploded.groupby('host_gene').agg(
    n_reads=('read_id', 'count'),
    n_cell_lines=('cell_line', 'nunique'),
    n_young=('l1_age', lambda x: (x == 'young').sum()),
    n_ancient=('l1_age', lambda x: (x == 'ancient').sum()),
    psi_mean=('psi', 'mean'),
    polya_median=('polya_length', 'median'),
).reset_index()
gene_summary = gene_summary.sort_values('n_reads', ascending=False)
gene_summary.to_csv(f"{OUTDIR}/host_gene_summary.tsv", sep='\t', index=False)
print(f"\nTop 20 host genes:")
print(gene_summary.head(20).to_string(index=False))

# ─── Prepare gene lists for enrichment ────────────────────────────────────────
# All intronic L1 host genes
all_host_genes = sorted(df_exploded['host_gene'].unique())

# Filter: genes detected in >=2 cell lines (more robust)
genes_multi_cl = gene_summary[gene_summary['n_cell_lines'] >= 2]['host_gene'].tolist()

# Young L1 host genes
young_host_genes = sorted(df_exploded[df_exploded['l1_age'] == 'young']['host_gene'].unique())

# Ancient L1 host genes
ancient_host_genes = sorted(df_exploded[df_exploded['l1_age'] == 'ancient']['host_gene'].unique())

print(f"\n--- Gene lists for enrichment ---")
print(f"All host genes: {len(all_host_genes)}")
print(f"Multi-CL (>=2 CLs): {len(genes_multi_cl)}")
print(f"Young L1 host genes: {len(young_host_genes)}")
print(f"Ancient L1 host genes: {len(ancient_host_genes)}")

# ─── Filter out non-standard gene names ──────────────────────────────────────
# Remove RP11-*, AC*, AL*, LINC* etc. that Enrichr can't map
def is_standard_gene(g):
    if g.startswith(('RP11-', 'RP4-', 'RP5-', 'RP3-', 'RP1-', 'AC0', 'AL0', 'AL1',
                     'AL3', 'AL4', 'AL5', 'AL6', 'AL7', 'AL8', 'AL9',
                     'AP0', 'CTC-', 'CTB-', 'CTD-', 'KB-', 'LA16')):
        return False
    return True

all_host_genes_filt = [g for g in all_host_genes if is_standard_gene(g)]
genes_multi_cl_filt = [g for g in genes_multi_cl if is_standard_gene(g)]
young_host_genes_filt = [g for g in young_host_genes if is_standard_gene(g)]
ancient_host_genes_filt = [g for g in ancient_host_genes if is_standard_gene(g)]

print(f"\nAfter filtering non-standard names:")
print(f"All: {len(all_host_genes_filt)}, Multi-CL: {len(genes_multi_cl_filt)}")
print(f"Young: {len(young_host_genes_filt)}, Ancient: {len(ancient_host_genes_filt)}")

# ─── Run Enrichr ──────────────────────────────────────────────────────────────
gene_sets_to_test = [
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'KEGG_2021_Human',
    'Reactome_2022',
    'MSigDB_Hallmark_2020',
]

def run_enrichr(gene_list, label, gene_sets_list):
    """Run Enrichr and return combined results"""
    all_results = []
    for gs in gene_sets_list:
        try:
            enr = gseapy.enrichr(
                gene_list=gene_list,
                gene_sets=gs,
                organism='human',
                outdir=None,
                no_plot=True,
            )
            res = enr.results.copy()
            res['database'] = gs
            res['gene_list'] = label
            all_results.append(res)
        except Exception as e:
            print(f"  WARNING: {gs} failed for {label}: {e}")
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

print("\n" + "=" * 70)
print("Running Enrichr analysis...")
print("=" * 70)

# Run for all gene lists
results_all = run_enrichr(all_host_genes_filt, 'all_L1_host', gene_sets_to_test)
print(f"All host genes: {len(results_all)} terms tested")

results_multi = run_enrichr(genes_multi_cl_filt, 'multi_CL_host', gene_sets_to_test)
print(f"Multi-CL host genes: {len(results_multi)} terms tested")

results_young = run_enrichr(young_host_genes_filt, 'young_L1_host', gene_sets_to_test)
print(f"Young L1 host genes: {len(results_young)} terms tested")

results_ancient = run_enrichr(ancient_host_genes_filt, 'ancient_L1_host', gene_sets_to_test)
print(f"Ancient L1 host genes: {len(results_ancient)} terms tested")

# Combine all
results_combined = pd.concat([results_all, results_multi, results_young, results_ancient], ignore_index=True)
results_combined.to_csv(f"{OUTDIR}/enrichr_all_results.tsv", sep='\t', index=False)

# ─── Show top results ─────────────────────────────────────────────────────────
def show_top_results(results, label, n=15):
    sig = results[(results['Adjusted P-value'] < 0.05) & (results['gene_list'] == label)]
    if len(sig) == 0:
        print(f"\n--- {label}: No significant terms (adj.p < 0.05) ---")
        # Show top by raw p-value instead
        top = results[results['gene_list'] == label].nsmallest(n, 'P-value')
        if len(top) > 0:
            print(f"Top {n} by raw P-value:")
            for _, row in top.iterrows():
                print(f"  [{row['database']}] {row['Term'][:70]} | overlap={row['Overlap']} | p={row['P-value']:.2e} | adj.p={row['Adjusted P-value']:.2e}")
        return

    print(f"\n--- {label}: {len(sig)} significant terms (adj.p < 0.05) ---")
    top = sig.nsmallest(n, 'Adjusted P-value')
    for _, row in top.iterrows():
        print(f"  [{row['database']}] {row['Term'][:70]} | overlap={row['Overlap']} | adj.p={row['Adjusted P-value']:.2e}")

show_top_results(results_combined, 'all_L1_host')
show_top_results(results_combined, 'multi_CL_host')
show_top_results(results_combined, 'young_L1_host')
show_top_results(results_combined, 'ancient_L1_host')

# ─── Stress-specific analysis ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Stress Response Gene Analysis")
print("=" * 70)

# Define stress-related keyword patterns
STRESS_KEYWORDS = {
    'oxidative_stress': ['oxidative', 'reactive oxygen', 'ROS', 'NRF2', 'NFE2L2', 'antioxid',
                          'glutathione', 'peroxid', 'thioredoxin'],
    'heat_shock': ['heat shock', 'HSP', 'HSF', 'chaperone', 'unfolded protein', 'ER stress'],
    'DNA_damage': ['DNA repair', 'DNA damage', 'double-strand break', 'base excision',
                    'nucleotide excision', 'mismatch repair', 'homologous recombination',
                    'ATM', 'ATR', 'BRCA', 'FANC', 'RAD'],
    'immune_inflammatory': ['interferon', 'interleukin', 'inflammatory', 'immune', 'NF-kB',
                            'TNF', 'cytokine', 'chemokine', 'toll-like', 'cGAS', 'STING'],
    'apoptosis': ['apoptosis', 'programmed cell death', 'caspase', 'BCL'],
    'stress_granule': ['stress granule', 'P-body', 'RNA granule', 'translation'],
}

# Check overlap with known stress response gene sets
# First, check Enrichr results for stress-related terms
stress_terms = []
for _, row in results_combined[results_combined['gene_list'] == 'all_L1_host'].iterrows():
    term_lower = row['Term'].lower()
    for cat, keywords in STRESS_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in term_lower:
                stress_terms.append({
                    'category': cat,
                    'term': row['Term'],
                    'database': row['database'],
                    'overlap': row['Overlap'],
                    'p_value': row['P-value'],
                    'adj_p_value': row['Adjusted P-value'],
                    'genes': row['Genes'],
                })
                break

stress_df = pd.DataFrame(stress_terms).drop_duplicates(subset=['term'])
stress_df = stress_df.sort_values('p_value')
stress_df.to_csv(f"{OUTDIR}/stress_related_terms.tsv", sep='\t', index=False)

print(f"\nStress-related terms found: {len(stress_df)}")
for cat in STRESS_KEYWORDS:
    cat_df = stress_df[stress_df['category'] == cat]
    n_sig = (cat_df['adj_p_value'] < 0.05).sum()
    print(f"\n  {cat}: {len(cat_df)} terms ({n_sig} significant)")
    for _, row in cat_df.head(5).iterrows():
        sig = "***" if row['adj_p_value'] < 0.05 else ""
        print(f"    {row['term'][:65]} | {row['overlap']} | p={row['p_value']:.2e} {sig}")

# ─── Specific gene checks ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Known Stress/DDR/Immune Genes in Our L1 Host Gene List")
print("=" * 70)

# Key genes from literature (Xiong 2021 MIL-hosting genes)
XIONG_DDR_GENES = ['ZRANB3', 'SMARCAL1', 'ATR', 'ATRX', 'RB1', 'FANCC', 'FANCD2', 'FANCI']
XIONG_NEURONAL = ['GPHN', 'UBE3A', 'CTNND2', 'DLG2', 'GABRG3', 'GRIA4']

# Other important gene sets
IMMUNE_GENES = ['IFNAR1', 'IFNAR2', 'IFNGR1', 'IFNGR2', 'IRF1', 'IRF3', 'IRF7',
                'STAT1', 'STAT2', 'JAK1', 'JAK2', 'TLR3', 'TLR7', 'TLR8',
                'CGAS', 'STING1', 'MAVS', 'MDA5', 'RIG-I', 'SAMHD1',
                'TRIM5', 'MOV10', 'ADAR', 'APOBEC3A', 'APOBEC3B', 'APOBEC3G']
DDR_GENES = ['ATM', 'ATR', 'BRCA1', 'BRCA2', 'ATRX', 'RAD51', 'RAD50',
             'TP53', 'CHEK1', 'CHEK2', 'PARP1', 'XRCC1', 'XRCC5', 'XRCC6',
             'FANCC', 'FANCD2', 'FANCI', 'NBN', 'MRE11', 'BLM', 'WRN']
STRESS_GENES = ['NFE2L2', 'KEAP1', 'HMOX1', 'NQO1', 'SOD1', 'SOD2', 'GPX1',
                'HSP90AA1', 'HSP90AB1', 'HSPA1A', 'HSPA5', 'HSF1',
                'EIF2AK1', 'EIF2AK2', 'EIF2AK3', 'EIF2AK4', 'ATF4', 'ATF6', 'XBP1',
                'DDIT3', 'GADD45A', 'GADD45B']

all_genes_set = set(all_host_genes)

for name, gene_set in [('Xiong DDR', XIONG_DDR_GENES), ('Xiong Neuronal', XIONG_NEURONAL),
                         ('Immune/IFN', IMMUNE_GENES), ('DDR', DDR_GENES),
                         ('Stress response', STRESS_GENES)]:
    found = [g for g in gene_set if g in all_genes_set]
    not_found = [g for g in gene_set if g not in all_genes_set]
    print(f"\n{name}: {len(found)}/{len(gene_set)} found in our data")
    if found:
        # Get read counts for found genes
        for g in found:
            info = gene_summary[gene_summary['host_gene'] == g]
            if len(info) > 0:
                row = info.iloc[0]
                print(f"  + {g}: {int(row['n_reads'])} reads, {int(row['n_cell_lines'])} CLs, "
                      f"psi={row['psi_mean']:.1f}, polyA={row['polya_median']:.1f}nt")
            else:
                print(f"  + {g}: in exploded but not in summary")

# ─── Gene length analysis ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Gene Length Distribution of L1 Host Genes")
print("=" * 70)

# Load gene lengths from GTF
GTF_PATH = f"{BASE}/reference/Human.gtf"
print(f"Loading gene lengths from {GTF_PATH}...")

gene_lengths = {}
try:
    with open(GTF_PATH) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            attrs = fields[8]
            # Parse gene_name
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
except Exception as e:
    print(f"GTF loading error: {e}")

# Compare L1 host genes vs all genes
if gene_lengths:
    host_lens = [gene_lengths[g] for g in all_host_genes if g in gene_lengths]
    all_lens = list(gene_lengths.values())

    from scipy import stats

    print(f"\nL1 host genes: n={len(host_lens)}, median={np.median(host_lens)/1000:.1f}kb")
    print(f"All genes: n={len(all_lens)}, median={np.median(all_lens)/1000:.1f}kb")
    u, p = stats.mannwhitneyu(host_lens, all_lens, alternative='greater')
    print(f"Mann-Whitney U (host > all): p={p:.2e}")

    # Percentile breakdown
    for pct in [25, 50, 75, 90]:
        h = np.percentile(host_lens, pct) / 1000
        a = np.percentile(all_lens, pct) / 1000
        print(f"  {pct}th pct: host={h:.1f}kb, all={a:.1f}kb, ratio={h/a:.2f}x")

    # How many host genes are >100kb (Xiong 2021 threshold)
    n_long = sum(1 for l in host_lens if l > 100000)
    pct_long = n_long / len(host_lens) * 100
    n_long_all = sum(1 for l in all_lens if l > 100000)
    pct_long_all = n_long_all / len(all_lens) * 100
    print(f"\n>100kb: host {n_long}/{len(host_lens)} ({pct_long:.1f}%) vs all {n_long_all}/{len(all_lens)} ({pct_long_all:.1f}%)")

    # Save gene length info
    len_df = pd.DataFrame({
        'host_gene': all_host_genes,
        'gene_length': [gene_lengths.get(g, np.nan) for g in all_host_genes],
        'in_our_data': True,
    })
    len_df.to_csv(f"{OUTDIR}/host_gene_lengths.tsv", sep='\t', index=False)

# ─── Per-cell-line host gene overlap ──────────────────────────────────────────
print("\n" + "=" * 70)
print("Per-Cell-Line Host Gene Sharing")
print("=" * 70)

cl_genes = {}
for cl in CELL_LINES:
    mask = (df_exploded['cell_line'] == cl)
    cl_genes[cl] = set(df_exploded[mask]['host_gene'].unique())
    print(f"  {cl}: {len(cl_genes[cl])} host genes")

# Ubiquitous genes (detected in all base cell lines, excluding HeLa-Ars, MCF7-EV)
base_cls = [cl for cl in CELL_LINES if cl not in ('HeLa-Ars', 'MCF7-EV')]
ubiq_genes = set.intersection(*[cl_genes[cl] for cl in base_cls])
print(f"\nUbiquitous host genes (in all {len(base_cls)} base CLs): {len(ubiq_genes)}")
if ubiq_genes:
    ubiq_list = sorted(ubiq_genes)
    print(f"  Examples: {', '.join(ubiq_list[:20])}")

    # Enrichr on ubiquitous genes
    ubiq_filt = [g for g in ubiq_list if is_standard_gene(g)]
    if len(ubiq_filt) >= 5:
        results_ubiq = run_enrichr(ubiq_filt, 'ubiquitous_host', gene_sets_to_test)
        show_top_results(results_ubiq, 'ubiquitous_host')
        results_ubiq.to_csv(f"{OUTDIR}/enrichr_ubiquitous.tsv", sep='\t', index=False)

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Count significant terms per category per gene list
for label in ['all_L1_host', 'multi_CL_host', 'young_L1_host', 'ancient_L1_host']:
    sub = results_combined[(results_combined['gene_list'] == label) & (results_combined['Adjusted P-value'] < 0.05)]
    print(f"\n{label}: {len(sub)} significant terms")
    for db in gene_sets_to_test:
        n = len(sub[sub['database'] == db])
        if n > 0:
            print(f"  {db}: {n}")

print("\nDone! Results saved to:", OUTDIR)
