#!/usr/bin/env python3
"""
Refined super-enhancer analysis:
1. Use stricter SE definition: top 5% of merged enhancer regions by size (~500 SEs)
2. Also compare regular enhancer L1 vs SE L1 vs non-enhancer L1
3. Better gene analysis with GATA2 highlight
"""

import pandas as pd
import numpy as np
from scipy import stats
import subprocess
import os

OUT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/super_enhancer_analysis"
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration"
L1_TSV = f"{BASE}/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv"
BEDTOOLS = "/blaze/apps/envs/bedtools/2.31.0/bin/bedtools"

# ── Load merged enhancers ──
merged = pd.read_csv(f"{OUT}/e117_enhancers_merged_12.5kb.bed", sep="\t", header=None,
                     names=["chr","start","end"])
merged["size"] = merged["end"] - merged["start"]
merged = merged.sort_values("size", ascending=False).reset_index(drop=True)

# ── Define SE as top 5% by size (ROSE-like inflection approach) ──
# Typical: ~200-500 SEs per cell type
percentile_95 = merged["size"].quantile(0.95)
print(f"95th percentile of merged enhancer size: {percentile_95/1000:.1f} kb")

for pct in [0.90, 0.95, 0.97, 0.99]:
    cutoff = merged["size"].quantile(pct)
    n = (merged["size"] >= cutoff).sum()
    print(f"  Top {(1-pct)*100:.0f}% (>= {cutoff/1000:.1f}kb): {n} regions")

# Use top 3% → ~900 SEs (typical range for HeLa)
# Actually, standard ROSE identifies ~300-700, so top 2-5%
SE_CUTOFF_PCT = 0.97  # top 3%
se_cutoff = merged["size"].quantile(SE_CUTOFF_PCT)
super_enhancers = merged[merged["size"] >= se_cutoff].copy()
regular_enhancers = merged[(merged["size"] < se_cutoff)].copy()

print(f"\n=== REFINED SE Definition: top {(1-SE_CUTOFF_PCT)*100:.0f}% ===")
print(f"SE cutoff: >= {se_cutoff/1000:.1f} kb")
print(f"Super-enhancers: {len(super_enhancers)}")
print(f"Regular enhancers: {len(regular_enhancers)}")
print(f"SE size range: {super_enhancers['size'].min()/1000:.1f} - {super_enhancers['size'].max()/1000:.1f} kb")
print(f"SE total coverage: {super_enhancers['size'].sum()/1e6:.1f} Mb")

# Save refined SE bed
se_bed = f"{OUT}/hela_super_enhancers_refined.bed"
super_enhancers[["chr","start","end"]].to_csv(se_bed, sep="\t", header=False, index=False)

re_bed = f"{OUT}/hela_regular_enhancers.bed"
regular_enhancers[["chr","start","end"]].to_csv(re_bed, sep="\t", header=False, index=False)

# ── Load L1 data ──
df = pd.read_csv(L1_TSV, sep="\t")

# Process HeLa and HeLa-Ars
for cellline_label, cl_name in [("HeLa", "HeLa"), ("HeLa-Ars", "HeLa-Ars")]:
    subset = df[df["cellline"] == cl_name].copy()
    if len(subset) == 0:
        continue

    # Write BED
    bed_file = f"{OUT}/{cl_name}_l1.bed"
    subset[["chr","start","end","read_id"]].to_csv(bed_file, sep="\t", header=False, index=False)

    # Intersect with SE
    se_int = f"{OUT}/{cl_name}_se_intersect.tsv"
    subprocess.run(f"{BEDTOOLS} intersect -a {bed_file} -b {se_bed} -wa -f 0.1 | cut -f4 | sort -u > {se_int}",
                   shell=True, check=True)
    se_reads = set(open(se_int).read().strip().split("\n")) - {""}

    # Intersect with regular enhancers
    re_int = f"{OUT}/{cl_name}_re_intersect.tsv"
    subprocess.run(f"{BEDTOOLS} intersect -a {bed_file} -b {re_bed} -wa -f 0.1 | cut -f4 | sort -u > {re_int}",
                   shell=True, check=True)
    re_reads = set(open(re_int).read().strip().split("\n")) - {""}

    # Assign categories
    subset["region"] = "non-enhancer"
    subset.loc[subset["read_id"].isin(re_reads), "region"] = "regular_enhancer"
    subset.loc[subset["read_id"].isin(se_reads), "region"] = "super_enhancer"

    df.loc[subset.index, "region"] = subset["region"]

# ── Comparison: Ancient L1 only ──
print("\n" + "=" * 60)
print("Ancient L1: SE vs Regular Enhancer vs Non-Enhancer")
print("=" * 60)

results = []
for cl_name in ["HeLa", "HeLa-Ars"]:
    subset = df[(df["cellline"] == cl_name) & (df["l1_age"] == "ancient") & (df["region"].notna())].copy()
    if len(subset) == 0:
        continue

    print(f"\n--- {cl_name} (ancient L1) ---")
    for region in ["super_enhancer", "regular_enhancer", "non-enhancer"]:
        r = subset[subset["region"] == region]
        pa = r["polya_length"].dropna()
        m6a = r["m6a_per_kb"].dropna()
        decay = (pa < 30).mean() * 100 if len(pa) > 0 else np.nan

        print(f"  {region:20s}: n={len(r):5d}, poly(A)={pa.median():.1f}, m6A/kb={m6a.median():.2f}, decay={decay:.1f}%")

        results.append({
            "cellline": cl_name, "region": region, "n": len(r),
            "polya_median": pa.median(), "m6a_median": m6a.median(), "decay_pct": decay
        })

    # Statistical tests: SE vs non-enhancer
    se = subset[subset["region"] == "super_enhancer"]
    ne = subset[subset["region"] == "non-enhancer"]
    if len(se) >= 5 and len(ne) >= 5:
        pa_p = stats.mannwhitneyu(se["polya_length"].dropna(), ne["polya_length"].dropna()).pvalue
        m6a_p = stats.mannwhitneyu(se["m6a_per_kb"].dropna(), ne["m6a_per_kb"].dropna()).pvalue
        print(f"  SE vs non-enh: poly(A) P={pa_p:.2e}, m6A P={m6a_p:.2e}")

    # SE vs regular enhancer
    re = subset[subset["region"] == "regular_enhancer"]
    if len(se) >= 5 and len(re) >= 5:
        pa_p2 = stats.mannwhitneyu(se["polya_length"].dropna(), re["polya_length"].dropna()).pvalue
        m6a_p2 = stats.mannwhitneyu(se["m6a_per_kb"].dropna(), re["m6a_per_kb"].dropna()).pvalue
        print(f"  SE vs reg-enh: poly(A) P={pa_p2:.2e}, m6A P={m6a_p2:.2e}")

# ── Delta poly(A) ──
print("\n--- Poly(A) shortening (delta = Ars - normal) ---")
for region in ["super_enhancer", "regular_enhancer", "non-enhancer"]:
    normal = df[(df["cellline"]=="HeLa") & (df["l1_age"]=="ancient") & (df["region"]==region)]["polya_length"].median()
    stress = df[(df["cellline"]=="HeLa-Ars") & (df["l1_age"]=="ancient") & (df["region"]==region)]["polya_length"].median()
    if not np.isnan(normal) and not np.isnan(stress):
        delta = stress - normal
        print(f"  {region:20s}: normal={normal:.1f}, stress={stress:.1f}, delta={delta:.1f} nt")

# Save results
pd.DataFrame(results).to_csv(f"{OUT}/refined_se_comparison.tsv", sep="\t", index=False)

# ── Gene analysis for refined SE ──
print("\n" + "=" * 60)
print("Genes near refined SE L1 loci")
print("=" * 60)

hela_ancient_se = df[(df["cellline"]=="HeLa") & (df["l1_age"]=="ancient") & (df["region"]=="super_enhancer")]
if len(hela_ancient_se) > 0:
    # Create BED of unique loci
    loci = hela_ancient_se[["chr","start","end"]].drop_duplicates().sort_values(["chr","start"])
    loci_bed = f"{OUT}/refined_se_l1_loci.bed"
    loci.to_csv(loci_bed, sep="\t", header=False, index=False)

    nearest_out = f"{OUT}/refined_se_l1_nearest_genes.tsv"
    subprocess.run(f"{BEDTOOLS} closest -a {loci_bed} -b {OUT}/gencode_genes.bed -d -t first > {nearest_out}",
                   shell=True, check=True)

    nearest = pd.read_csv(nearest_out, sep="\t", header=None,
                          names=["l1_chr","l1_start","l1_end","gene_chr","gene_start","gene_end",
                                 "gene_id","score","gene_name","distance"])

    print(f"Unique refined SE L1 loci: {len(nearest)}")
    print(f"Unique nearest genes: {nearest['gene_name'].nunique()}")
    print(f"Within gene (dist=0): {(nearest['distance']==0).sum()}")

    print("\nTop genes:")
    for gene, count in nearest["gene_name"].value_counts().head(15).items():
        dist = nearest[nearest["gene_name"]==gene]["distance"].median()
        print(f"  {gene}: {count} L1 loci (dist={dist:.0f})")

    # Notable genes
    notable = {"GATA2","MYC","SOX2","FOSL1","JUN","FOS","TP63","KLF5","CEBPB",
               "CCND1","CDK6","RB1","TP53","BRCA1","EZH2","METTL3","METTL14",
               "HRH1","PARN","GSDMD","CDC20","PKN2","DLG2"}
    found = set(nearest["gene_name"].unique()) & notable
    if found:
        print(f"\nNotable genes found: {found}")
        for g in found:
            sub = nearest[nearest["gene_name"]==g]
            print(f"  {g}: {len(sub)} L1 loci, dist={sub['distance'].min()}-{sub['distance'].max()}")

    # Save
    nearest.to_csv(f"{OUT}/refined_se_l1_gene_list.tsv", sep="\t", index=False)

print("\nDone.")
