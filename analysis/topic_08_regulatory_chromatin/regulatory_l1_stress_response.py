#!/usr/bin/env python3
"""
Regulatory L1 stress-response analysis.
Question: Are stress-resistant ancient L1s at functionally important genomic positions?
Reframing: Regulatory L1 = dynamically regulated stress-responsive modules.

Analyses:
A. Per-gene poly(A) delta classification (shortening vs lengthening vs stable)
B. Functional annotation of host genes (GO enrichment via Enrichr REST API)
C. m6A as predictor of direction
D. Per-read level: within regulatory L1, what predicts poly(A) retention?
E. Cross-CL: are regulatory L1 patterns consistent across cell lines?
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, json, urllib.request, urllib.parse

BASEDIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASEDIR, 'regulatory_stress_response')
os.makedirs(OUTDIR, exist_ok=True)

# ── Load data ──
print("Loading regulatory L1 per-read data...")
df = pd.read_csv(f'{BASEDIR}/stress_gene_analysis/regulatory_l1_per_read.tsv', sep='\t')
print(f"  Total regulatory L1 reads: {len(df)}")
print(f"  Conditions: {df['condition'].value_counts().to_dict()}")
print(f"  Cell lines: {df['cellline'].nunique()}")

# Focus on ancient only (young too few at regulatory positions)
df = df[df['l1_age'] == 'ancient'].copy()
print(f"  Ancient regulatory reads: {len(df)}")

# ═══════════════════════════════════════
# A. Per-gene HeLa vs HeLa-Ars delta
# ═══════════════════════════════════════
print("\n=== A. Per-gene poly(A) delta (HeLa vs HeLa-Ars) ===")

hela = df[df['condition'] == 'normal'].copy()
ars = df[df['condition'] == 'stress'].copy()
hela_genes = hela.groupby('host_gene').agg(
    n_hela=('polya_length', 'count'),
    med_hela=('polya_length', 'median'),
    m6a_hela=('m6a_per_kb', 'median'),
).reset_index()
ars_genes = ars.groupby('host_gene').agg(
    n_ars=('polya_length', 'count'),
    med_ars=('polya_length', 'median'),
    m6a_ars=('m6a_per_kb', 'median'),
).reset_index()

gene_delta = hela_genes.merge(ars_genes, on='host_gene', how='inner')
gene_delta['delta'] = gene_delta['med_ars'] - gene_delta['med_hela']
gene_delta['m6a_avg'] = (gene_delta['m6a_hela'] + gene_delta['m6a_ars']) / 2
gene_delta['n_total'] = gene_delta['n_hela'] + gene_delta['n_ars']
gene_delta = gene_delta.sort_values('delta')

print(f"\nGenes with reads in BOTH conditions: {len(gene_delta)}")
print(f"  n_total range: {gene_delta['n_total'].min()}-{gene_delta['n_total'].max()}")

# Classify
gene_delta['response'] = 'stable'
gene_delta.loc[gene_delta['delta'] < -20, 'response'] = 'shortened'
gene_delta.loc[gene_delta['delta'] > 20, 'response'] = 'lengthened'

resp_counts = gene_delta['response'].value_counts()
print(f"\n  Response classification (threshold ±20nt):")
for r, c in resp_counts.items():
    print(f"    {r}: {c} genes")

# Save
gene_delta.to_csv(f'{OUTDIR}/gene_polya_delta.tsv', sep='\t', index=False)
print(f"\n  Top 10 shortened:")
for _, row in gene_delta.head(10).iterrows():
    print(f"    {row['host_gene']:30s} Δ={row['delta']:+8.1f}  m6A/kb={row['m6a_avg']:.1f}  n={row['n_total']}")
print(f"\n  Top 10 lengthened:")
for _, row in gene_delta.tail(10).iterrows():
    print(f"    {row['host_gene']:30s} Δ={row['delta']:+8.1f}  m6A/kb={row['m6a_avg']:.1f}  n={row['n_total']}")

# ═══════════════════════════════════════
# B. Functional annotation via Enrichr
# ═══════════════════════════════════════
print("\n=== B. GO enrichment: shortened vs lengthened genes ===")

def enrichr_submit(gene_list, description):
    """Submit gene list to Enrichr REST API."""
    url = 'https://maayanlab.cloud/Enrichr/addList'
    payload = {'list': (None, '\n'.join(gene_list)),
               'description': (None, description)}
    # Use urllib
    genes_str = '\n'.join(gene_list)
    data = urllib.parse.urlencode({'list': genes_str, 'description': description}).encode()
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        return result.get('userListId')
    except Exception as e:
        print(f"  Enrichr submit failed: {e}")
        return None

def enrichr_results(list_id, library='GO_Biological_Process_2023'):
    """Get enrichment results from Enrichr."""
    url = f'https://maayanlab.cloud/Enrichr/enrich?userListId={list_id}&backgroundType={library}'
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        return result.get(library, [])
    except Exception as e:
        print(f"  Enrichr query failed: {e}")
        return []

# Clean gene names (remove lncRNAs, split multi-gene entries)
def clean_genes(gene_series):
    genes = set()
    for g in gene_series:
        if pd.isna(g): continue
        for part in str(g).split(';'):
            part = part.strip()
            if part and not part.startswith('RP11-') and not part.startswith('RP4-') \
               and not part.startswith('CTD-') and not part.startswith('XXbac-') \
               and not part.startswith('LINC') and not part.startswith('AC0'):
                genes.add(part)
    return sorted(genes)

shortened_genes = clean_genes(gene_delta[gene_delta['response'] == 'shortened']['host_gene'])
lengthened_genes = clean_genes(gene_delta[gene_delta['response'] == 'lengthened']['host_gene'])
stable_genes = clean_genes(gene_delta[gene_delta['response'] == 'stable']['host_gene'])

print(f"  Shortened protein-coding genes: {len(shortened_genes)}")
print(f"  Lengthened protein-coding genes: {len(lengthened_genes)}")
print(f"  Stable protein-coding genes: {len(stable_genes)}")

# Submit to Enrichr
for group_name, gene_list in [('shortened', shortened_genes),
                               ('lengthened', lengthened_genes)]:
    if len(gene_list) < 3:
        print(f"  {group_name}: too few genes ({len(gene_list)}), skipping")
        continue

    print(f"\n  --- {group_name.upper()} genes ({len(gene_list)}) ---")
    print(f"  Genes: {', '.join(gene_list[:20])}{'...' if len(gene_list)>20 else ''}")

    list_id = enrichr_submit(gene_list, f'regulatory_L1_{group_name}')
    if list_id is None:
        continue

    for lib in ['GO_Biological_Process_2023', 'KEGG_2021_Human', 'WikiPathway_2023_Human']:
        results = enrichr_results(list_id, lib)
        if not results:
            print(f"    {lib}: no results")
            continue
        # Top 5 by p-value
        results.sort(key=lambda x: x[2])  # sort by p-value (index 2)
        print(f"    {lib}:")
        out_rows = []
        for r in results[:5]:
            term = r[1]
            pval = r[2]
            adj_p = r[6]
            genes_hit = r[5]
            print(f"      {term[:60]:60s}  p={pval:.2e}  adj_p={adj_p:.2e}  genes={genes_hit}")
            out_rows.append({'term': term, 'p': pval, 'adj_p': adj_p, 'genes': ';'.join(genes_hit)})

        pd.DataFrame(out_rows).to_csv(
            f'{OUTDIR}/enrichr_{group_name}_{lib.split("_")[0]}.tsv', sep='\t', index=False)

# ═══════════════════════════════════════
# C. m6A as predictor of direction
# ═══════════════════════════════════════
print("\n=== C. m6A vs poly(A) delta direction ===")

# All genes with both conditions
gd = gene_delta[gene_delta['n_total'] >= 2].copy()
print(f"  Genes with n>=2: {len(gd)}")

# Correlation: m6A_avg vs delta
r, p = stats.spearmanr(gd['m6a_avg'], gd['delta'])
print(f"  m6A/kb vs delta Spearman r={r:.3f}, p={p:.3e}")

# Compare m6A between groups
for group in ['shortened', 'stable', 'lengthened']:
    sub = gd[gd['response'] == group]
    if len(sub) > 0:
        print(f"  {group:12s}: n={len(sub):3d}, median m6A/kb={sub['m6a_avg'].median():.2f}, "
              f"median delta={sub['delta'].median():.1f}")

# Mann-Whitney shortened vs lengthened m6A
short_m6a = gd[gd['response'] == 'shortened']['m6a_avg']
long_m6a = gd[gd['response'] == 'lengthened']['m6a_avg']
if len(short_m6a) > 2 and len(long_m6a) > 2:
    u, p = stats.mannwhitneyu(short_m6a, long_m6a, alternative='two-sided')
    print(f"\n  Shortened vs Lengthened m6A/kb: "
          f"{short_m6a.median():.2f} vs {long_m6a.median():.2f}, MW p={p:.3e}")

# ═══════════════════════════════════════
# D. Per-read analysis within regulatory L1
# ═══════════════════════════════════════
print("\n=== D. Per-read: what predicts poly(A) retention under stress? ===")

# Within HeLa-Ars regulatory ancient L1
ars_reg = df[(df['condition'] == 'stress') & (df['l1_age'] == 'ancient')].copy()
print(f"  HeLa-Ars regulatory ancient reads: {len(ars_reg)}")

if len(ars_reg) > 20:
    # Classify reads by poly(A)
    med_polya = ars_reg['polya_length'].median()
    ars_reg['retained'] = ars_reg['polya_length'] > med_polya

    retained = ars_reg[ars_reg['retained']]
    degraded = ars_reg[~ars_reg['retained']]

    print(f"  Retained (poly(A) > {med_polya:.0f}nt): n={len(retained)}, "
          f"median m6A/kb={retained['m6a_per_kb'].median():.2f}")
    print(f"  Degraded (poly(A) <= {med_polya:.0f}nt): n={len(degraded)}, "
          f"median m6A/kb={degraded['m6a_per_kb'].median():.2f}")

    u, p = stats.mannwhitneyu(retained['m6a_per_kb'], degraded['m6a_per_kb'],
                               alternative='greater')
    print(f"  m6A retained > degraded: MW p={p:.3e}")

    # Chromatin subtype
    for state in ['Enhancer', 'Promoter']:
        sub = ars_reg[ars_reg['chromhmm_group'] == state]
        if len(sub) > 5:
            r, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
            print(f"\n  {state} (n={len(sub)}): m6A-poly(A) r={r:.3f}, p={p:.3e}")
            print(f"    median poly(A)={sub['polya_length'].median():.1f}, "
                  f"median m6A/kb={sub['m6a_per_kb'].median():.2f}")
            # Decay zone
            dz = (sub['polya_length'] < 30).sum() / len(sub) * 100
            print(f"    Decay zone (<30nt): {dz:.1f}%")

# ═══════════════════════════════════════
# E. Cross-CL: regulatory L1 poly(A) consistency
# ═══════════════════════════════════════
print("\n=== E. Cross-CL: Are regulatory L1 patterns consistent? ===")

# Per cell line, compare regulatory vs non-regulatory L1
all_reads = pd.read_csv(f'{BASEDIR}/l1_chromhmm_annotated.tsv', sep='\t')
all_ancient = all_reads[all_reads['l1_age'] == 'ancient'].copy()
all_ancient['is_regulatory'] = all_ancient['chromhmm_group'].isin(['Enhancer', 'Promoter'])

cl_stats = []
for cl in sorted(all_ancient['cellline'].unique()):
    sub = all_ancient[all_ancient['cellline'] == cl]
    reg = sub[sub['is_regulatory']]
    non_reg = sub[~sub['is_regulatory']]

    if len(reg) >= 5 and len(non_reg) >= 20:
        u, p = stats.mannwhitneyu(reg['polya_length'], non_reg['polya_length'],
                                   alternative='two-sided')
        delta = reg['polya_length'].median() - non_reg['polya_length'].median()
        cl_stats.append({
            'cellline': cl,
            'n_reg': len(reg),
            'n_nonreg': len(non_reg),
            'med_reg': reg['polya_length'].median(),
            'med_nonreg': non_reg['polya_length'].median(),
            'delta': delta,
            'p': p,
            'm6a_reg': reg['m6a_per_kb'].median(),
            'm6a_nonreg': non_reg['m6a_per_kb'].median(),
        })

cl_df = pd.DataFrame(cl_stats).sort_values('delta')
print(f"\n  Regulatory vs Non-regulatory poly(A) by cell line:")
print(f"  {'CL':15s} {'n_reg':>6s} {'med_reg':>8s} {'med_nonreg':>10s} {'Δ':>8s} {'p':>10s} {'m6A_reg':>8s} {'m6A_nonreg':>10s}")
for _, row in cl_df.iterrows():
    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else 'ns'
    print(f"  {row['cellline']:15s} {row['n_reg']:6d} {row['med_reg']:8.1f} {row['med_nonreg']:10.1f} "
          f"{row['delta']:+8.1f} {row['p']:10.3e} {row['m6a_reg']:8.2f} {row['m6a_nonreg']:10.2f}  {sig}")

cl_df.to_csv(f'{OUTDIR}/cross_cl_regulatory_vs_nonreg.tsv', sep='\t', index=False)

# Count how many CLs show reg < nonreg
n_shorter = (cl_df['delta'] < 0).sum()
n_total = len(cl_df)
print(f"\n  Regulatory shorter in {n_shorter}/{n_total} cell lines")

# ═══════════════════════════════════════
# F. Gene-level: bidirectional response summary
# ═══════════════════════════════════════
print("\n=== F. Bidirectional response summary ===")

# For genes with enough reads in both conditions
gd_sig = gene_delta[(gene_delta['n_hela'] >= 2) & (gene_delta['n_ars'] >= 2)].copy()
print(f"  Genes with n>=2 per condition: {len(gd_sig)}")

resp = gd_sig['response'].value_counts()
for r, c in resp.items():
    pct = c / len(gd_sig) * 100
    print(f"    {r}: {c} ({pct:.0f}%)")

# Are lengthened genes more likely to be at enhancers vs promoters?
reg_gene_info = df.groupby('host_gene').agg(
    n_enhancer=('chromhmm_group', lambda x: (x == 'Enhancer').sum()),
    n_promoter=('chromhmm_group', lambda x: (x == 'Promoter').sum()),
    n_intronic=('genomic_context', lambda x: (x == 'intronic').sum()),
    n_total=('read_id', 'count'),
).reset_index()

gd_info = gd_sig.merge(reg_gene_info, on='host_gene', how='left', suffixes=('', '_ctx'))
gd_info['pct_enhancer'] = gd_info['n_enhancer'] / gd_info['n_total_ctx'] * 100
gd_info['pct_intronic'] = gd_info['n_intronic'] / gd_info['n_total_ctx'] * 100

for resp_group in ['shortened', 'lengthened', 'stable']:
    sub = gd_info[gd_info['response'] == resp_group]
    if len(sub) > 0:
        print(f"\n  {resp_group} (n={len(sub)}):")
        print(f"    median m6A/kb: {sub['m6a_avg'].median():.2f}")
        print(f"    median n_total: {sub['n_total'].median():.0f}")
        print(f"    % enhancer: {sub['pct_enhancer'].median():.0f}%")
        print(f"    % intronic: {sub['pct_intronic'].median():.0f}%")

# Save comprehensive summary
gd_info.to_csv(f'{OUTDIR}/gene_response_annotated.tsv', sep='\t', index=False)

print(f"\n=== All results saved to {OUTDIR}/ ===")
