#!/usr/bin/env python3
"""
GO/Pathway enrichment analysis of regulatory L1 host genes.

Uses g:Profiler REST API for enrichment analysis of ancient L1 elements
located in regulatory chromatin (enhancer/promoter, ChromHMM E117).

Queries: GO:BP, GO:MF, GO:CC, KEGG, Reactome, WikiPathways
"""

import os
import json
import requests
import pandas as pd
import numpy as np

# ── Paths ──
REG_GENES = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin/stress_gene_analysis/regulatory_l1_genes.tsv"
ANNOTATED = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin/regulatory_stress_response/gene_response_annotated.tsv"
DESEQ2 = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/tetranscripts_output/HeLa_SA_vs_UN_gene_TE_analysis.txt"
OUTDIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/regulatory_l1_go_enrichment"

os.makedirs(OUTDIR, exist_ok=True)

# ── 1. Load genes ──
print("=== Loading gene lists ===")
reg_df = pd.read_csv(REG_GENES, sep='\t')
ann_df = pd.read_csv(ANNOTATED, sep='\t')

# Extract unique gene symbols, splitting semicolon-separated entries
# Filter out non-standard gene names (RP11-, AC-, CTD-, etc.)
raw_genes = set()
for g in reg_df['gene'].dropna().unique():
    for part in g.split(';'):
        part = part.strip()
        if part:
            raw_genes.add(part)

# Filter: keep only likely protein-coding gene symbols
# Remove pseudogenes, lncRNAs with accession-like names
def is_standard_gene(name):
    """Filter out non-standard gene symbols that g:Profiler won't recognize."""
    prefixes = ('RP11-', 'RP4-', 'RP13-', 'AC0', 'AC1', 'AP0', 'CTD-', 'CTC-',
                'XXbac-', 'KB-', 'LINC0', 'LINC1', 'LINC2', 'LA16c-',
                'KRT18P', 'APOC1P', 'VTRNA', 'CNN3-DT')
    for p in prefixes:
        if name.startswith(p):
            return False
    if name.startswith('C') and 'orf' in name and len(name) < 12:
        return True  # e.g. C11orf52
    return True

all_genes = sorted([g for g in raw_genes if is_standard_gene(g)])
print(f"Total raw gene symbols: {len(raw_genes)}")
print(f"After filtering non-standard: {len(all_genes)}")

# Also get annotated subset (HeLa + HeLa-Ars, with poly(A) response)
ann_genes_raw = set()
for g in ann_df['host_gene'].dropna().unique():
    for part in g.split(';'):
        part = part.strip()
        if part and is_standard_gene(part):
            ann_genes_raw.add(part)
ann_genes = sorted(ann_genes_raw)
print(f"Annotated response genes: {len(ann_genes)}")

# ── 2. Load background universe from DESeq2 ──
print("\n=== Loading background gene universe ===")
try:
    deseq = pd.read_csv(DESEQ2, sep='\t')
    # Gene names are row index or first column
    if 'gene' in deseq.columns:
        bg_genes_raw = deseq['gene'].dropna().unique()
    elif deseq.index.name:
        bg_genes_raw = deseq.index.dropna().unique()
    else:
        # First column might be gene names
        bg_genes_raw = deseq.iloc[:, 0].dropna().unique()
    bg_genes = sorted(set(str(g) for g in bg_genes_raw if not str(g).startswith(('__', 'ENSG'))))
    print(f"Background universe: {len(bg_genes)} genes")
except Exception as e:
    print(f"Warning: Could not load DESeq2 background: {e}")
    bg_genes = None

# ── 3. g:Profiler REST API enrichment ──
def run_gprofiler(gene_list, background=None, tag="all_regulatory"):
    """Run g:Profiler enrichment via REST API."""
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

    payload = {
        "organism": "hsapiens",
        "query": gene_list,
        "sources": ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "WP"],
        "user_threshold": 0.05,
        "significance_threshold_method": "g_SCS",  # g:Profiler's own correction
        "no_evidences": False,
        "combined": False,
        "measure_underrepresentation": False,
        "no_iea": False,  # include electronic annotations
        "domain_scope": "annotated" if background is None else "custom",
        "numeric_ns": "ENTREZGENE_ACC",
        "all_results": False,
    }

    if background is not None:
        payload["background"] = background

    print(f"\nQuerying g:Profiler with {len(gene_list)} genes...")
    if background:
        print(f"  Custom background: {len(background)} genes")

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if 'result' not in data:
            print(f"  Warning: No 'result' key in response. Keys: {list(data.keys())}")
            # Save raw response for debugging
            with open(os.path.join(OUTDIR, f"gprofiler_raw_{tag}.json"), 'w') as f:
                json.dump(data, f, indent=2)
            return pd.DataFrame()

        results = data['result']
        if not results:
            print("  No significant results found.")
            return pd.DataFrame()

        rows = []
        for r in results:
            rows.append({
                'source': r.get('source', ''),
                'term_id': r.get('native', ''),
                'term_name': r.get('name', ''),
                'p_value': r.get('p_value', 1.0),
                'term_size': r.get('term_size', 0),
                'query_size': r.get('query_size', 0),
                'intersection_size': r.get('intersection_size', 0),
                'precision': r.get('precision', 0),
                'recall': r.get('recall', 0),
                'effective_domain_size': r.get('effective_domain_size', 0),
                'intersections': ';'.join(
                    [x if isinstance(x, str) else ','.join(x) if isinstance(x, list) else str(x)
                     for x in r.get('intersections', [])]
                ) if r.get('intersections') else '',
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('p_value')
        print(f"  Found {len(df)} significant terms")
        return df

    except requests.exceptions.RequestException as e:
        print(f"  API error: {e}")
        return pd.DataFrame()

# ── 4. Run enrichment analyses ──
print("\n" + "="*60)
print("=== Analysis 1: All regulatory L1 host genes (no custom bg) ===")
print("="*60)
res_all = run_gprofiler(all_genes, background=None, tag="all_regulatory")

print("\n" + "="*60)
print("=== Analysis 2: All regulatory L1 host genes (DESeq2 bg) ===")
print("="*60)
if bg_genes:
    res_bg = run_gprofiler(all_genes, background=bg_genes, tag="all_regulatory_custombg")
else:
    res_bg = pd.DataFrame()

# ── 5. Save results ──
def save_and_report(df, name, outdir):
    """Save results and print summary."""
    if df.empty:
        print(f"\n  [{name}] No results to save.")
        return

    outpath = os.path.join(outdir, f"{name}.tsv")
    df.to_csv(outpath, sep='\t', index=False)
    print(f"\n  [{name}] Saved {len(df)} terms to {outpath}")

    # Summary by source
    print(f"\n  --- {name} Summary ---")
    for src in ['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC', 'WP']:
        sub = df[df['source'] == src]
        if len(sub) > 0:
            print(f"\n  {src}: {len(sub)} significant terms")
            top = sub.head(10)
            for _, row in top.iterrows():
                print(f"    {row['term_id']:20s} {row['term_name'][:60]:60s} "
                      f"p={row['p_value']:.2e}  ({row['intersection_size']}/{row['query_size']})")

save_and_report(res_all, "enrichment_all_regulatory", OUTDIR)
save_and_report(res_bg, "enrichment_all_regulatory_custombg", OUTDIR)

# ── 6. Categorize enriched terms ──
def categorize_terms(df):
    """Flag stress/chromatin/signaling related terms."""
    if df.empty:
        return df

    stress_kw = ['stress', 'apoptos', 'damage', 'repair', 'heat shock', 'unfold',
                 'hypoxia', 'oxidat', 'inflamm', 'immune', 'NF-kB', 'cytokine',
                 'interferon', 'defense', 'p53', 'autophagy', 'senescen']
    chromatin_kw = ['chromat', 'histone', 'epigenet', 'methylat', 'acetylat',
                    'nucleosome', 'heterochrom', 'silenc', 'transcription factor',
                    'enhancer', 'promoter']
    signaling_kw = ['signal', 'kinase', 'phosphat', 'MAPK', 'Wnt', 'Notch',
                    'receptor', 'pathway', 'PI3K', 'mTOR', 'JAK', 'STAT',
                    'calcium', 'cAMP', 'RAS', 'tyrosine']
    development_kw = ['develop', 'morphogen', 'differentiat', 'embryo', 'organ',
                      'pattern', 'cell fate', 'stem cell', 'lineage']

    def check_kw(name, keywords):
        name_lower = name.lower()
        return any(kw.lower() in name_lower for kw in keywords)

    df = df.copy()
    df['is_stress'] = df['term_name'].apply(lambda x: check_kw(x, stress_kw))
    df['is_chromatin'] = df['term_name'].apply(lambda x: check_kw(x, chromatin_kw))
    df['is_signaling'] = df['term_name'].apply(lambda x: check_kw(x, signaling_kw))
    df['is_development'] = df['term_name'].apply(lambda x: check_kw(x, development_kw))

    return df

# Apply categorization to main result
if not res_all.empty:
    res_cat = categorize_terms(res_all)
    res_cat.to_csv(os.path.join(OUTDIR, "enrichment_categorized.tsv"), sep='\t', index=False)

    print("\n" + "="*60)
    print("=== Category Highlights ===")
    print("="*60)

    for cat, label in [('is_stress', 'STRESS/IMMUNE'), ('is_chromatin', 'CHROMATIN/EPIGENETIC'),
                       ('is_signaling', 'SIGNALING'), ('is_development', 'DEVELOPMENT')]:
        sub = res_cat[res_cat[cat]]
        if len(sub) > 0:
            print(f"\n  {label} ({len(sub)} terms):")
            for _, row in sub.head(15).iterrows():
                print(f"    {row['source']:6s} {row['term_id']:15s} {row['term_name'][:65]:65s} p={row['p_value']:.2e}  n={row['intersection_size']}")

# ── 7. Enhancer-only vs Promoter-only subsets ──
print("\n" + "="*60)
print("=== Analysis 3: Enhancer-enriched genes ===")
print("="*60)
enh_genes_raw = set()
prom_genes_raw = set()
for _, row in reg_df.iterrows():
    gene = row['gene']
    parts = [p.strip() for p in gene.split(';') if p.strip() and is_standard_gene(p.strip())]
    if row.get('n_enhancer', 0) > row.get('n_promoter', 0):
        enh_genes_raw.update(parts)
    elif row.get('n_promoter', 0) > row.get('n_enhancer', 0):
        prom_genes_raw.update(parts)

enh_genes = sorted(enh_genes_raw)
prom_genes = sorted(prom_genes_raw)
print(f"Enhancer-dominant genes: {len(enh_genes)}")
print(f"Promoter-dominant genes: {len(prom_genes)}")

if len(enh_genes) >= 20:
    res_enh = run_gprofiler(enh_genes, tag="enhancer")
    save_and_report(res_enh, "enrichment_enhancer", OUTDIR)

if len(prom_genes) >= 20:
    res_prom = run_gprofiler(prom_genes, tag="promoter")
    save_and_report(res_prom, "enrichment_promoter", OUTDIR)

# ── 8. Gene list file for reproducibility ──
with open(os.path.join(OUTDIR, "gene_list_all_regulatory.txt"), 'w') as f:
    f.write('\n'.join(all_genes))
with open(os.path.join(OUTDIR, "gene_list_enhancer.txt"), 'w') as f:
    f.write('\n'.join(enh_genes))
with open(os.path.join(OUTDIR, "gene_list_promoter.txt"), 'w') as f:
    f.write('\n'.join(prom_genes))

print(f"\n\nGene lists saved to {OUTDIR}/gene_list_*.txt")
print("Done!")
