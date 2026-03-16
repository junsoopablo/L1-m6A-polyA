#!/usr/bin/env python3
"""
Super-enhancer x L1 analysis for HeLa.
1. Construct super-enhancers from E117 ChromHMM enhancer states (ROSE-like: merge within 12.5kb, top by size)
2. Intersect with ancient L1 reads
3. Compare poly(A), m6A, delta, decay zone
4. Identify nearby genes
"""

import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
from scipy import stats
from pathlib import Path

# ── Paths ──
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration"
CHROMHMM_BED = f"{BASE}/topic_08_regulatory_chromatin/E117_15_coreMarks_hg38lift_mnemonics.bed.gz"
L1_TSV = f"{BASE}/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv"
OUT_DIR = f"{BASE}/topic_10_rnaseq_validation/super_enhancer_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Construct super-enhancers from ChromHMM ──
print("=" * 60)
print("Step 1: Constructing super-enhancers from E117 ChromHMM")
print("=" * 60)

# Extract enhancer states (7_Enh and 6_EnhG)
enh_bed = os.path.join(OUT_DIR, "e117_enhancers.bed")
subprocess.run(
    f"zcat {CHROMHMM_BED} | grep -E '(7_Enh|6_EnhG)$' | sort -k1,1 -k2,2n > {enh_bed}",
    shell=True, check=True
)
n_enh = int(subprocess.check_output(f"wc -l < {enh_bed}", shell=True).strip())
print(f"  Raw enhancer segments: {n_enh}")

# Merge enhancers within 12.5kb (ROSE algorithm standard distance)
merged_bed = os.path.join(OUT_DIR, "e117_enhancers_merged_12.5kb.bed")
subprocess.run(
    f"/blaze/apps/envs/bedtools/2.31.0/bin/bedtools merge -i {enh_bed} -d 12500 > {merged_bed}",
    shell=True, check=True, executable="/bin/bash"
)

# Read merged regions, calculate sizes
merged = pd.read_csv(merged_bed, sep="\t", header=None, names=["chr", "start", "end"])
merged["size"] = merged["end"] - merged["start"]
print(f"  Merged enhancer regions: {len(merged)}")
print(f"  Size range: {merged['size'].min()}-{merged['size'].max()} bp")
print(f"  Median size: {merged['size'].median():.0f} bp")

# Define super-enhancers: top by size (ROSE uses inflection point; we use >12.5kb as proxy)
# Typical ROSE: sort by H3K27ac signal, find inflection. Without signal, use size cutoff.
# HeLa typically has ~200-700 super-enhancers
# Let's try multiple thresholds and see what gives ~300-500 SEs
for cutoff in [10000, 12500, 15000, 20000, 25000]:
    n = (merged["size"] >= cutoff).sum()
    print(f"  Size >= {cutoff/1000:.1f}kb: {n} regions")

# Use 12.5kb cutoff (standard ROSE stitching distance as minimum SE size)
SE_SIZE_CUTOFF = 12500
super_enhancers = merged[merged["size"] >= SE_SIZE_CUTOFF].copy()
print(f"\n  ** Super-enhancers (>={SE_SIZE_CUTOFF/1000:.1f}kb): {len(super_enhancers)} **")
print(f"  Total SE coverage: {super_enhancers['size'].sum()/1e6:.1f} Mb")
print(f"  Mean SE size: {super_enhancers['size'].mean()/1000:.1f} kb")

# Save SE bed
se_bed = os.path.join(OUT_DIR, "hela_super_enhancers.bed")
super_enhancers[["chr", "start", "end"]].to_csv(se_bed, sep="\t", header=False, index=False)

# ── Step 2: Load L1 data and intersect ──
print("\n" + "=" * 60)
print("Step 2: Intersecting L1 reads with super-enhancers")
print("=" * 60)

df = pd.read_csv(L1_TSV, sep="\t")
hela = df[df["cellline"] == "HeLa"].copy()
print(f"  Total HeLa L1 reads: {len(hela)}")
print(f"  Ancient HeLa: {(hela['l1_age']=='ancient').sum()}")
print(f"  Young HeLa: {(hela['l1_age']=='young').sum()}")

# Create BED from L1 reads
l1_bed = os.path.join(OUT_DIR, "hela_l1_reads.bed")
hela_bed = hela[["chr", "start", "end", "read_id"]].copy()
hela_bed.to_csv(l1_bed, sep="\t", header=False, index=False)

# Intersect
intersect_out = os.path.join(OUT_DIR, "l1_se_intersect.tsv")
subprocess.run(
    f"/blaze/apps/envs/bedtools/2.31.0/bin/bedtools intersect -a {l1_bed} -b {se_bed} -wa -wb -f 0.1 > {intersect_out}",
    shell=True, check=True, executable="/bin/bash"
)

# Get read IDs in super-enhancers
se_reads = set()
with open(intersect_out) as f:
    for line in f:
        fields = line.strip().split("\t")
        se_reads.add(fields[3])

print(f"  L1 reads overlapping super-enhancers: {len(se_reads)}")

# Annotate
hela["in_SE"] = hela["read_id"].isin(se_reads)
hela_ancient = hela[hela["l1_age"] == "ancient"].copy()
hela_young = hela[hela["l1_age"] == "young"].copy()

print(f"  Ancient L1 in SE: {hela_ancient['in_SE'].sum()}")
print(f"  Ancient L1 not in SE: {(~hela_ancient['in_SE']).sum()}")
print(f"  Young L1 in SE: {hela_young['in_SE'].sum()}")

# ── Step 3: Compare poly(A) and m6A ──
print("\n" + "=" * 60)
print("Step 3: Comparing super-enhancer vs non-super-enhancer ancient L1")
print("=" * 60)

# Split by condition
for cond_label, cond_val in [("normal", "normal"), ("stressed", "stressed")]:
    subset = hela_ancient[hela_ancient["condition"] == cond_val]
    if len(subset) == 0:
        # Try alternate condition names
        continue
    se_sub = subset[subset["in_SE"]]
    non_se_sub = subset[~subset["in_SE"]]

    if len(se_sub) < 5:
        print(f"\n  [{cond_label}] Too few SE reads ({len(se_sub)}), skipping")
        continue

    print(f"\n  [{cond_label.upper()}] SE: n={len(se_sub)}, non-SE: n={len(non_se_sub)}")

    # Poly(A)
    pa_se = se_sub["polya_length"].dropna()
    pa_nonse = non_se_sub["polya_length"].dropna()
    mwu_pa = stats.mannwhitneyu(pa_se, pa_nonse, alternative="two-sided")
    print(f"    Poly(A) median: SE={pa_se.median():.1f} vs non-SE={pa_nonse.median():.1f} (MWU P={mwu_pa.pvalue:.2e})")

    # m6A/kb
    m6a_se = se_sub["m6a_per_kb"].dropna()
    m6a_nonse = non_se_sub["m6a_per_kb"].dropna()
    mwu_m6a = stats.mannwhitneyu(m6a_se, m6a_nonse, alternative="two-sided")
    print(f"    m6A/kb median: SE={m6a_se.median():.2f} vs non-SE={m6a_nonse.median():.2f} (MWU P={mwu_m6a.pvalue:.2e})")

# Check condition values
print(f"\n  Available conditions: {hela['condition'].unique()}")

# If conditions are group-based, detect stress from group name
if "stressed" not in hela["condition"].unique():
    # Check if HeLa-Ars is a separate cellline
    hela_all = df[df["cellline"].isin(["HeLa", "HeLa-Ars"])].copy()
    hela_all["in_SE"] = hela_all["read_id"].isin(se_reads)

    # Need to also intersect HeLa-Ars reads
    hela_ars = df[df["cellline"] == "HeLa-Ars"].copy()
    if len(hela_ars) > 0:
        print(f"\n  HeLa-Ars found as separate cellline: {len(hela_ars)} reads")
        ars_bed_file = os.path.join(OUT_DIR, "hela_ars_l1_reads.bed")
        hela_ars[["chr", "start", "end", "read_id"]].to_csv(ars_bed_file, sep="\t", header=False, index=False)

        ars_intersect = os.path.join(OUT_DIR, "l1_ars_se_intersect.tsv")
        subprocess.run(
            f"/blaze/apps/envs/bedtools/2.31.0/bin/bedtools intersect -a {ars_bed_file} -b {se_bed} -wa -wb -f 0.1 > {ars_intersect}",
            shell=True, check=True, executable="/bin/bash"
        )

        ars_se_reads = set()
        with open(ars_intersect) as f:
            for line in f:
                fields = line.strip().split("\t")
                ars_se_reads.add(fields[3])

        hela_ars["in_SE"] = hela_ars["read_id"].isin(ars_se_reads)

        print(f"  HeLa-Ars ancient in SE: {hela_ars[hela_ars['l1_age']=='ancient']['in_SE'].sum()}")

        # Now compare normal vs stressed for SE L1
        conditions = [
            ("HeLa (normal)", hela[hela["l1_age"] == "ancient"]),
            ("HeLa-Ars (stressed)", hela_ars[hela_ars["l1_age"] == "ancient"]),
        ]
    else:
        conditions = [("HeLa", hela[hela["l1_age"] == "ancient"])]

    print("\n" + "=" * 60)
    print("Step 3b: SE vs non-SE comparison by condition")
    print("=" * 60)

    results = []
    for label, data in conditions:
        se_sub = data[data["in_SE"]]
        non_se_sub = data[~data["in_SE"]]

        if len(se_sub) < 5:
            print(f"\n  [{label}] Too few SE reads ({len(se_sub)}), skipping")
            continue

        print(f"\n  [{label}] SE: n={len(se_sub)}, non-SE: n={len(non_se_sub)}")

        # Poly(A)
        pa_se = se_sub["polya_length"].dropna()
        pa_nonse = non_se_sub["polya_length"].dropna()
        mwu_pa = stats.mannwhitneyu(pa_se, pa_nonse, alternative="two-sided")
        print(f"    Poly(A) median: SE={pa_se.median():.1f} vs non-SE={pa_nonse.median():.1f} (MWU P={mwu_pa.pvalue:.2e})")

        # m6A/kb
        m6a_se = se_sub["m6a_per_kb"].dropna()
        m6a_nonse = non_se_sub["m6a_per_kb"].dropna()
        mwu_m6a = stats.mannwhitneyu(m6a_se, m6a_nonse, alternative="two-sided")
        print(f"    m6A/kb median: SE={m6a_se.median():.2f} vs non-SE={m6a_nonse.median():.2f} (MWU P={mwu_m6a.pvalue:.2e})")

        # Decay zone (<30nt)
        decay_se = (pa_se < 30).mean() * 100
        decay_nonse = (pa_nonse < 30).mean() * 100
        print(f"    Decay zone (<30nt): SE={decay_se:.1f}% vs non-SE={decay_nonse:.1f}%")

        results.append({
            "condition": label,
            "n_SE": len(se_sub), "n_nonSE": len(non_se_sub),
            "polya_SE": pa_se.median(), "polya_nonSE": pa_nonse.median(),
            "polya_P": mwu_pa.pvalue,
            "m6a_SE": m6a_se.median(), "m6a_nonSE": m6a_nonse.median(),
            "m6a_P": mwu_m6a.pvalue,
            "decay_SE": decay_se, "decay_nonSE": decay_nonse,
        })

    # Delta poly(A) for SE vs non-SE reads (locus-matched if possible)
    if len(conditions) == 2:
        print("\n  --- Delta poly(A) (Ars - normal) ---")
        normal_se = conditions[0][1][conditions[0][1]["in_SE"]]["polya_length"].median()
        normal_nonse = conditions[0][1][~conditions[0][1]["in_SE"]]["polya_length"].median()
        stress_se = conditions[1][1][conditions[1][1]["in_SE"]]["polya_length"].median()
        stress_nonse = conditions[1][1][~conditions[1][1]["in_SE"]]["polya_length"].median()

        delta_se = stress_se - normal_se
        delta_nonse = stress_nonse - normal_nonse
        print(f"    SE delta: {delta_se:.1f} nt")
        print(f"    non-SE delta: {delta_nonse:.1f} nt")
        print(f"    SE shows {'MORE' if abs(delta_se) > abs(delta_nonse) else 'LESS'} shortening")

    # Save results
    if results:
        pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "se_vs_nonse_comparison.tsv"), sep="\t", index=False)

# ── Step 4: ChromHMM group breakdown within SE ──
print("\n" + "=" * 60)
print("Step 4: ChromHMM states of L1 in super-enhancers")
print("=" * 60)

hela_ancient_se = hela_ancient[hela_ancient["in_SE"]]
if len(hela_ancient_se) > 0:
    print(hela_ancient_se["chromhmm_group"].value_counts().to_string())

# ── Step 5: Nearby genes for SE L1 ──
print("\n" + "=" * 60)
print("Step 5: Genes near super-enhancer L1 elements")
print("=" * 60)

# Get unique SE L1 loci
se_loci = hela_ancient_se[["chr", "start", "end", "read_id"]].drop_duplicates(subset=["chr", "start", "end"])
print(f"  Unique SE L1 loci: {len(se_loci)}")

# Create BED of SE L1 loci and find nearest genes using GENCODE
gencode_gtf = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/gencode.v38.annotation.gtf"
if not os.path.exists(gencode_gtf):
    # Try alternate paths
    import glob
    gtf_files = glob.glob("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/*.gtf*")
    if gtf_files:
        gencode_gtf = gtf_files[0]
        print(f"  Using GTF: {gencode_gtf}")

# Extract gene features from GTF for nearest-gene analysis
gene_bed = os.path.join(OUT_DIR, "gencode_genes.bed")
if os.path.exists(gencode_gtf):
    cmd = f"""zcat {gencode_gtf} 2>/dev/null || cat {gencode_gtf} | awk '$3=="gene"' | grep 'gene_type "protein_coding"' | awk -F'\\t' '{{OFS="\\t"; split($9,a,"\\\""); print $1,$4,$5,a[2],$6,a[4]}}' | sort -k1,1 -k2,2n > {gene_bed}"""
    subprocess.run(cmd, shell=True, executable="/bin/bash")

if os.path.exists(gene_bed) and os.path.getsize(gene_bed) > 0:
    se_loci_bed = os.path.join(OUT_DIR, "se_l1_loci.bed")
    se_loci_out = se_loci[["chr", "start", "end"]].drop_duplicates()
    se_loci_out = se_loci_out.sort_values(["chr", "start"])
    se_loci_out.to_csv(se_loci_bed, sep="\t", header=False, index=False)

    nearest_out = os.path.join(OUT_DIR, "se_l1_nearest_genes.tsv")
    subprocess.run(
        f"/blaze/apps/envs/bedtools/2.31.0/bin/bedtools closest -a {se_loci_bed} -b {gene_bed} -d -t first > {nearest_out}",
        shell=True, check=True, executable="/bin/bash"
    )

    # Read nearest genes
    nearest = pd.read_csv(nearest_out, sep="\t", header=None)
    if len(nearest.columns) >= 9:
        # columns: L1_chr, L1_start, L1_end, gene_chr, gene_start, gene_end, gene_id, score, gene_name, distance
        nearest.columns = ["l1_chr", "l1_start", "l1_end", "gene_chr", "gene_start", "gene_end", "gene_id", "score", "gene_name", "distance"][:len(nearest.columns)]

        print(f"\n  Nearest genes to SE L1 loci:")
        if "gene_name" in nearest.columns:
            gene_counts = nearest["gene_name"].value_counts().head(20)
            for gene, count in gene_counts.items():
                print(f"    {gene}: {count} L1 loci")

            # Save full gene list
            nearest.to_csv(os.path.join(OUT_DIR, "se_l1_nearest_genes_full.tsv"), sep="\t", index=False)

            unique_genes = nearest["gene_name"].unique()
            print(f"\n  Total unique genes near SE L1: {len(unique_genes)}")
        else:
            if "gene_id" in nearest.columns:
                print(f"  Gene IDs found: {nearest['gene_id'].nunique()}")
    else:
        print(f"  Output has {len(nearest.columns)} columns, check format")
else:
    print("  GTF not found or empty, skipping gene analysis")

# ── Step 6: Young L1 in SE (for comparison) ──
print("\n" + "=" * 60)
print("Step 6: Young vs Ancient L1 in super-enhancers")
print("=" * 60)

hela_young_check = hela[hela["l1_age"] == "young"]
young_se = hela_young_check["in_SE"].sum()
young_total = len(hela_young_check)
ancient_se = hela_ancient["in_SE"].sum()
ancient_total = len(hela_ancient)

print(f"  Young: {young_se}/{young_total} in SE ({young_se/young_total*100:.1f}%)")
print(f"  Ancient: {ancient_se}/{ancient_total} in SE ({ancient_se/ancient_total*100:.1f}%)")

# Fisher's exact test
from scipy.stats import fisher_exact
table = [[young_se, young_total - young_se],
         [ancient_se, ancient_total - ancient_se]]
odds, fisher_p = fisher_exact(table)
print(f"  Fisher's exact: OR={odds:.2f}, P={fisher_p:.2e}")

# ── Step 7: Summary ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Super-enhancers defined: {len(super_enhancers)} (merged enhancer regions >= {SE_SIZE_CUTOFF/1000:.1f}kb)")
print(f"Total SE coverage: {super_enhancers['size'].sum()/1e6:.1f} Mb")
print(f"Ancient L1 in SE: {ancient_se}/{ancient_total} ({ancient_se/ancient_total*100:.1f}%)")
print(f"Young L1 in SE: {young_se}/{young_total} ({young_se/young_total*100:.1f}%)")
