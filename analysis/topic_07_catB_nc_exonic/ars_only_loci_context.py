#!/usr/bin/env python3
"""Analyze genomic context of Ars-only vs Shared vs HeLa-only L1 loci.

1. Host gene annotation & genomic context
2. GO enrichment of host genes (Enrichr via gseapy)
3. Functional category comparison
"""

import pandas as pd
import numpy as np
import gseapy as gp
from pathlib import Path
from scipy import stats
from collections import Counter

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
OUTDIR = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic/ars_loci_context'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# =============================================================================
# 1. Load HeLa / HeLa-Ars L1 data with host gene info
# =============================================================================
print("Loading L1 summary data...")

groups = {
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
}

all_data = []
for cl, grps in groups.items():
    for grp in grps:
        f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['group'] = grp
            df['cell_line'] = cl
            df['locus_id'] = df['transcript_id']
            df['subfamily'] = df['gene_id']
            df['is_young'] = df['subfamily'].isin(YOUNG)
            df['age'] = np.where(df['is_young'], 'young', 'ancient')
            df = df.rename(columns={'polya_length': 'polya'})
            all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
data_polya = data[data['polya'].notna()].copy()

hela = data_polya[data_polya['cell_line'] == 'HeLa']
ars = data_polya[data_polya['cell_line'] == 'HeLa-Ars']

hela_loci = set(hela['locus_id'].unique())
ars_loci = set(ars['locus_id'].unique())
shared_loci = hela_loci & ars_loci
hela_only_loci = hela_loci - ars_loci
ars_only_loci = ars_loci - hela_loci

print(f"  HeLa-only: {len(hela_only_loci)} loci")
print(f"  Shared: {len(shared_loci)} loci")
print(f"  Ars-only: {len(ars_only_loci)} loci")

# =============================================================================
# 2. Build per-locus annotation table
# =============================================================================
print("\nBuilding per-locus annotation...")

def get_locus_info(locus_id, df):
    """Get consensus info for a locus from all reads."""
    sub = df[df['locus_id'] == locus_id]
    if len(sub) == 0:
        return {}
    row = sub.iloc[0]
    # Most common overlapping gene
    genes = sub['overlapping_genes'].dropna()
    genes = genes[genes != '']
    gene = genes.mode().iloc[0] if len(genes) > 0 else ''
    te_group = sub['TE_group'].mode().iloc[0] if 'TE_group' in sub.columns else ''
    return {
        'locus_id': locus_id,
        'chr': row['chr'] if 'chr' in row else row.get('te_chr', ''),
        'te_start': row.get('te_start', 0),
        'te_end': row.get('te_end', 0),
        'subfamily': row['subfamily'],
        'age': row['age'],
        'TE_group': te_group,
        'host_gene': gene,
        'n_reads': len(sub),
        'polya_median': sub['polya'].median(),
    }

# Combine info from both conditions
locus_info = {}
for locus in hela_loci | ars_loci:
    # Prefer HeLa info, supplement with Ars
    info_h = get_locus_info(locus, hela)
    info_a = get_locus_info(locus, ars)
    info = info_h if info_h else info_a
    if not info:
        continue

    # Add category
    if locus in shared_loci:
        info['category'] = 'shared'
    elif locus in hela_only_loci:
        info['category'] = 'hela_only'
    else:
        info['category'] = 'ars_only'

    # Add Ars poly(A) if available
    if locus in ars_loci:
        ars_sub = ars[ars['locus_id'] == locus]
        info['ars_polya_median'] = ars_sub['polya'].median()
        info['ars_n_reads'] = len(ars_sub)
    else:
        info['ars_polya_median'] = np.nan
        info['ars_n_reads'] = 0

    if locus in hela_loci:
        hela_sub = hela[hela['locus_id'] == locus]
        info['hela_polya_median'] = hela_sub['polya'].median()
        info['hela_n_reads'] = len(hela_sub)
    else:
        info['hela_polya_median'] = np.nan
        info['hela_n_reads'] = 0

    # Get host gene from either condition
    if not info.get('host_gene'):
        info_other = info_a if info_h else info_h
        if info_other:
            info['host_gene'] = info_other.get('host_gene', '')

    locus_info[locus] = info

locus_table = pd.DataFrame(locus_info.values())
locus_table.to_csv(OUTDIR / 'locus_annotation.tsv', sep='\t', index=False)

# =============================================================================
# 3. Genomic context comparison
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: GENOMIC CONTEXT (TE_group)")
print("=" * 70)

for cat in ['hela_only', 'shared', 'ars_only']:
    sub = locus_table[locus_table['category'] == cat]
    print(f"\n  {cat} ({len(sub)} loci):")
    te_counts = sub['TE_group'].value_counts()
    for tg, cnt in te_counts.items():
        print(f"    {tg:15s}: {cnt:4d} ({cnt/len(sub)*100:.1f}%)")

# Chi-square test: intronic vs intergenic across categories
print("\n  Chi-square (intronic vs intergenic):")
ct = pd.crosstab(locus_table['category'], locus_table['TE_group'])
if 'intronic' in ct.columns and 'intergenic' in ct.columns:
    ct_sub = ct[['intronic', 'intergenic']].loc[['hela_only', 'shared', 'ars_only']]
    print(ct_sub)
    chi2, p, _, _ = stats.chi2_contingency(ct_sub)
    print(f"  chi2={chi2:.2f}, p={p:.2e}")

# =============================================================================
# 4. Age composition
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: AGE COMPOSITION")
print("=" * 70)

for cat in ['hela_only', 'shared', 'ars_only']:
    sub = locus_table[locus_table['category'] == cat]
    young_pct = (sub['age'] == 'young').mean() * 100
    print(f"  {cat:12s}: {young_pct:.1f}% young, {100-young_pct:.1f}% ancient")

# =============================================================================
# 5. Host gene analysis
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: HOST GENE ANALYSIS")
print("=" * 70)

# Get intronic loci with host genes
for cat in ['hela_only', 'shared', 'ars_only']:
    sub = locus_table[(locus_table['category'] == cat) &
                       (locus_table['TE_group'] == 'intronic') &
                       (locus_table['host_gene'] != '') &
                       (locus_table['host_gene'].notna())]
    genes = sub['host_gene'].unique()
    print(f"\n  {cat} intronic host genes: {len(genes)}")
    # Top genes by read count
    gene_reads = sub.groupby('host_gene')['n_reads'].sum().sort_values(ascending=False)
    for gene, cnt in gene_reads.head(15).items():
        n_loci = len(sub[sub['host_gene'] == gene])
        print(f"    {gene:20s}: {cnt:3d} reads, {n_loci} loci")

# =============================================================================
# 6. GO Enrichment (Ars-only vs background)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: GO ENRICHMENT")
print("=" * 70)

# Ars-only intronic host genes
ars_only_genes = locus_table[
    (locus_table['category'] == 'ars_only') &
    (locus_table['TE_group'] == 'intronic') &
    (locus_table['host_gene'] != '') &
    (locus_table['host_gene'].notna())
]['host_gene'].unique().tolist()

# Background: all intronic host genes
all_intronic_genes = locus_table[
    (locus_table['TE_group'] == 'intronic') &
    (locus_table['host_gene'] != '') &
    (locus_table['host_gene'].notna())
]['host_gene'].unique().tolist()

print(f"\n  Ars-only intronic genes: {len(ars_only_genes)}")
print(f"  All intronic genes (background): {len(all_intronic_genes)}")

# Save gene lists
with open(OUTDIR / 'ars_only_host_genes.txt', 'w') as f:
    f.write('\n'.join(ars_only_genes))
with open(OUTDIR / 'all_host_genes.txt', 'w') as f:
    f.write('\n'.join(all_intronic_genes))

# Shared intronic host genes (for comparison)
shared_genes = locus_table[
    (locus_table['category'] == 'shared') &
    (locus_table['TE_group'] == 'intronic') &
    (locus_table['host_gene'] != '') &
    (locus_table['host_gene'].notna())
]['host_gene'].unique().tolist()
with open(OUTDIR / 'shared_host_genes.txt', 'w') as f:
    f.write('\n'.join(shared_genes))

hela_only_genes = locus_table[
    (locus_table['category'] == 'hela_only') &
    (locus_table['TE_group'] == 'intronic') &
    (locus_table['host_gene'] != '') &
    (locus_table['host_gene'].notna())
]['host_gene'].unique().tolist()
with open(OUTDIR / 'hela_only_host_genes.txt', 'w') as f:
    f.write('\n'.join(hela_only_genes))

# Run Enrichr for each category
gene_sets = ['GO_Biological_Process_2023', 'GO_Molecular_Function_2023',
             'KEGG_2021_Human', 'WikiPathway_2023_Human']

for cat_label, gene_list in [('ars_only', ars_only_genes),
                              ('shared', shared_genes),
                              ('hela_only', hela_only_genes)]:
    if len(gene_list) < 5:
        print(f"\n  {cat_label}: too few genes ({len(gene_list)}), skipping Enrichr")
        continue

    print(f"\n  Running Enrichr for {cat_label} ({len(gene_list)} genes)...")
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism='human',
            outdir=str(OUTDIR / f'enrichr_{cat_label}'),
            no_plot=True,
        )
        res = enr.results
        res = res[res['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')

        if len(res) > 0:
            print(f"    Significant terms (adj.p < 0.05): {len(res)}")
            for _, row in res.head(20).iterrows():
                print(f"    [{row['Gene_set'][:15]:15s}] {row['Term'][:60]:60s} "
                      f"p={row['Adjusted P-value']:.2e} ({row['Overlap']})")
            res.to_csv(OUTDIR / f'enrichr_{cat_label}_significant.tsv', sep='\t', index=False)
        else:
            print(f"    No significant terms found")
    except Exception as e:
        print(f"    Enrichr failed: {e}")

# =============================================================================
# 7. Gene overlap between categories
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: GENE OVERLAP BETWEEN CATEGORIES")
print("=" * 70)

ars_set = set(ars_only_genes)
shared_set = set(shared_genes)
hela_set = set(hela_only_genes)

print(f"\n  Ars-only genes: {len(ars_set)}")
print(f"  Shared genes: {len(shared_set)}")
print(f"  HeLa-only genes: {len(hela_set)}")
print(f"  Ars-only ∩ Shared: {len(ars_set & shared_set)}")
print(f"  Ars-only ∩ HeLa-only: {len(ars_set & hela_set)}")
print(f"  Ars-only unique: {len(ars_set - shared_set - hela_set)}")

# Ars-only exclusive genes
ars_exclusive = ars_set - shared_set - hela_set
print(f"\n  Ars-only exclusive genes ({len(ars_exclusive)}):")
for gene in sorted(ars_exclusive)[:30]:
    n_loci = len(locus_table[(locus_table['host_gene'] == gene) &
                              (locus_table['category'] == 'ars_only')])
    print(f"    {gene:25s}: {n_loci} L1 loci")

# =============================================================================
# 8. Poly(A) comparison within shared genes
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: SHARED GENES — LOCI IN MULTIPLE CATEGORIES")
print("=" * 70)

# Genes that have L1 loci in both ars_only and shared categories
genes_with_both = ars_set & shared_set
print(f"\n  Genes with L1 in both ars_only and shared: {len(genes_with_both)}")

if len(genes_with_both) > 0:
    rows = []
    for gene in genes_with_both:
        ars_sub = locus_table[(locus_table['host_gene'] == gene) &
                               (locus_table['category'] == 'ars_only')]
        sh_sub = locus_table[(locus_table['host_gene'] == gene) &
                              (locus_table['category'] == 'shared')]
        rows.append({
            'gene': gene,
            'ars_only_loci': len(ars_sub),
            'shared_loci': len(sh_sub),
            'ars_only_polya': ars_sub['ars_polya_median'].median(),
            'shared_hela_polya': sh_sub['hela_polya_median'].median(),
            'shared_ars_polya': sh_sub['ars_polya_median'].median(),
        })

    gene_comp = pd.DataFrame(rows).sort_values('ars_only_loci', ascending=False)
    print(f"\n  Top genes:")
    for _, row in gene_comp.head(15).iterrows():
        print(f"    {row['gene']:25s}  ars_only={row['ars_only_loci']}loci/{row['ars_only_polya']:.0f}nt  "
              f"shared_hela={row['shared_loci']}loci/{row['shared_hela_polya']:.0f}nt  "
              f"shared_ars={row['shared_ars_polya']:.0f}nt")

# =============================================================================
# 9. Chromosome distribution
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: CHROMOSOME DISTRIBUTION")
print("=" * 70)

for cat in ['hela_only', 'shared', 'ars_only']:
    sub = locus_table[locus_table['category'] == cat]
    chr_counts = sub['chr'].value_counts()
    top_chr = chr_counts.head(5)
    print(f"\n  {cat} top chromosomes:")
    for ch, cnt in top_chr.items():
        print(f"    {ch}: {cnt} ({cnt/len(sub)*100:.1f}%)")

print(f"\nAll results saved to: {OUTDIR}")
print("Done!")
