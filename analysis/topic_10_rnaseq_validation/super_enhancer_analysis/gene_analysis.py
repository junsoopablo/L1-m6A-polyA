#!/usr/bin/env python3
"""
Analyze genes near super-enhancer L1 elements.
Check if they are cell-identity / housekeeping / cancer-related genes.
"""

import pandas as pd
import numpy as np
from collections import Counter

OUT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/super_enhancer_analysis"

# Load nearest genes
nearest = pd.read_csv(f"{OUT}/se_l1_nearest_genes.tsv", sep="\t", header=None,
                       names=["l1_chr","l1_start","l1_end","gene_chr","gene_start","gene_end",
                              "gene_id","score","gene_name","distance"])

print(f"Total SE L1 loci: {len(nearest)}")
print(f"Unique nearest genes: {nearest['gene_name'].nunique()}")
print(f"Distance = 0 (within gene): {(nearest['distance']==0).sum()} ({(nearest['distance']==0).mean()*100:.1f}%)")
print(f"Distance <= 50kb: {(nearest['distance']<=50000).sum()} ({(nearest['distance']<=50000).mean()*100:.1f}%)")

print("\n--- Top 20 genes hosting SE L1 ---")
gene_counts = nearest["gene_name"].value_counts().head(20)
for gene, count in gene_counts.items():
    dists = nearest[nearest["gene_name"]==gene]["distance"]
    print(f"  {gene}: {count} L1 loci (median dist: {dists.median():.0f} bp)")

# Known cell-identity / super-enhancer target genes in HeLa (cervical cancer)
# MYC, TP63, SOX2, KLF5, FOSL1, ID1, CEBPB are known SE targets in HeLa-S3
known_se_targets = {"MYC","SOX2","KLF5","FOSL1","ID1","CEBPB","TP63","FGFR1","RB1",
                    "JUN","FOS","MYB","ETS1","RUNX1","TERT","MDM2","CDK6","CCND1","BCL2"}

found_targets = set(nearest["gene_name"].unique()) & known_se_targets
print(f"\nKnown SE/cell-identity genes found: {found_targets if found_targets else 'none'}")

# Broader cancer/cell-identity gene categories
tf_genes = {"MYC","SOX2","KLF5","FOSL1","CEBPB","TP63","JUN","FOS","ETS1","RUNX1",
            "MYB","FOXO1","STAT3","HIF1A","NF1","RB1","TP53","BRCA1","EGFR"}
signaling = {"FGFR1","FGFR2","ERBB2","EGFR","KRAS","PIK3CA","AKT1","MTOR","RAF1",
             "BRAF","CDK4","CDK6","CCND1","CCNE1","MDM2","TERT","VEGFA","MET"}
epigenetic = {"METTL3","METTL14","WTAP","ALKBH5","FTO","YTHDF1","YTHDF2","YTHDC1",
              "SETD2","KMT2A","DNMT1","DNMT3A","TET1","TET2","EZH2","SUZ12","KDM1A"}

all_genes = set(nearest["gene_name"].unique())
for label, gset in [("TF/cancer", tf_genes), ("Signaling", signaling), ("Epigenetic/m6A", epigenetic)]:
    overlap = all_genes & gset
    if overlap:
        print(f"  {label}: {overlap}")

# Save full gene list
gene_summary = nearest.groupby("gene_name").agg(
    n_l1=("l1_chr", "size"),
    median_dist=("distance", "median"),
    min_dist=("distance", "min"),
    chromosomes=("l1_chr", lambda x: ",".join(x.unique()))
).sort_values("n_l1", ascending=False)

gene_summary.to_csv(f"{OUT}/se_l1_gene_summary.tsv", sep="\t")
print(f"\nSaved gene summary: {len(gene_summary)} genes")

# Also check: are SE L1 more likely intronic?
print("\n--- Genomic context of SE L1 (from distance) ---")
print(f"  Intronic (dist=0): {(nearest['distance']==0).sum()}")
print(f"  Intergenic (dist>0): {(nearest['distance']>0).sum()}")
