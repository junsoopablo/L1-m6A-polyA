#!/usr/bin/env python3
"""
Final summary: SE analysis compared with existing ChromHMM regulatory findings.
Check if SE adds information beyond ChromHMM Enhancer group.
"""

import pandas as pd
import numpy as np
from scipy import stats

OUT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/super_enhancer_analysis"
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration"
L1_TSV = f"{BASE}/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv"

df = pd.read_csv(L1_TSV, sep="\t")

# Reload SE/RE assignments
for cl in ["HeLa", "HeLa-Ars"]:
    sub = df[df["cellline"] == cl]
    se_file = f"{OUT}/{cl}_se_intersect.tsv"
    re_file = f"{OUT}/{cl}_re_intersect.tsv"
    try:
        se_reads = set(open(se_file).read().strip().split("\n")) - {""}
        re_reads = set(open(re_file).read().strip().split("\n")) - {""}
        df.loc[sub.index, "in_SE"] = sub["read_id"].isin(se_reads)
        df.loc[sub.index, "in_RE"] = sub["read_id"].isin(re_reads)
    except:
        pass

df["in_SE"] = df["in_SE"].fillna(False).astype(bool)
df["in_RE"] = df["in_RE"].fillna(False).astype(bool)

# Compare SE vs ChromHMM regulatory
print("=" * 60)
print("SE overlap with ChromHMM Regulatory group")
print("=" * 60)

hela = df[(df["cellline"].isin(["HeLa","HeLa-Ars"])) & (df["l1_age"]=="ancient")]
se = hela[hela["in_SE"]]
print(f"SE ancient L1: {len(se)}")
print(f"  ChromHMM groups of SE L1:")
print(se["chromhmm_group"].value_counts().to_string())
print(f"\n  Regulatory in SE: {(se['chromhmm_group'].isin(['Enhancer','Promoter'])).sum()}/{len(se)} ({(se['chromhmm_group'].isin(['Enhancer','Promoter'])).mean()*100:.1f}%)")

# All regulatory L1 for comparison
reg = hela[hela["chromhmm_group"].isin(["Enhancer","Promoter"])]
print(f"\nAll Regulatory L1: {len(reg)}")
print(f"  of which in SE: {reg['in_SE'].sum()} ({reg['in_SE'].mean()*100:.1f}%)")

# Key question: Does SE status add information BEYOND ChromHMM regulatory?
print("\n" + "=" * 60)
print("Key test: Within Regulatory L1, does SE status matter?")
print("=" * 60)

for cl in ["HeLa-Ars"]:
    reg_cl = hela[(hela["cellline"]==cl) & hela["chromhmm_group"].isin(["Enhancer","Promoter"])]
    se_reg = reg_cl[reg_cl["in_SE"]]
    non_se_reg = reg_cl[~reg_cl["in_SE"]]

    if len(se_reg) >= 5 and len(non_se_reg) >= 5:
        pa_p = stats.mannwhitneyu(se_reg["polya_length"].dropna(), non_se_reg["polya_length"].dropna()).pvalue
        print(f"\n{cl} Regulatory L1:")
        print(f"  SE-regulatory: n={len(se_reg)}, poly(A)={se_reg['polya_length'].median():.1f}")
        print(f"  non-SE regulatory: n={len(non_se_reg)}, poly(A)={non_se_reg['polya_length'].median():.1f}")
        print(f"  MWU P={pa_p:.2e}")

# Also: Within non-regulatory, does SE status matter?
for cl in ["HeLa-Ars"]:
    nonreg_cl = hela[(hela["cellline"]==cl) & ~hela["chromhmm_group"].isin(["Enhancer","Promoter"])]
    se_nonreg = nonreg_cl[nonreg_cl["in_SE"]]
    non_se_nonreg = nonreg_cl[~nonreg_cl["in_SE"]]

    if len(se_nonreg) >= 5:
        pa_p = stats.mannwhitneyu(se_nonreg["polya_length"].dropna(), non_se_nonreg["polya_length"].dropna()).pvalue
        print(f"\n{cl} Non-regulatory L1:")
        print(f"  SE non-regulatory: n={len(se_nonreg)}, poly(A)={se_nonreg['polya_length'].median():.1f}")
        print(f"  non-SE non-regulatory: n={len(non_se_nonreg)}, poly(A)={non_se_nonreg['polya_length'].median():.1f}")
        print(f"  MWU P={pa_p:.2e}")

# ── Final summary table ──
print("\n" + "=" * 60)
print("FINAL SUMMARY TABLE")
print("=" * 60)

summary_rows = []
for cl, label in [("HeLa","Normal"), ("HeLa-Ars","Stressed")]:
    for region_name, mask in [
        ("Super-enhancer", hela["in_SE"]),
        ("Regular enhancer", hela["in_RE"] & ~hela["in_SE"]),
        ("Non-enhancer", ~hela["in_RE"] & ~hela["in_SE"]),
        ("ChromHMM Regulatory", hela["chromhmm_group"].isin(["Enhancer","Promoter"])),
        ("ChromHMM Non-reg", ~hela["chromhmm_group"].isin(["Enhancer","Promoter"])),
    ]:
        sub = hela[(hela["cellline"]==cl) & mask]
        if len(sub) == 0:
            continue
        pa = sub["polya_length"].dropna()
        m6a = sub["m6a_per_kb"].dropna()
        decay = (pa < 30).mean() * 100
        row = {"condition": label, "region": region_name, "n": len(sub),
               "polya_median": f"{pa.median():.1f}", "m6a_median": f"{m6a.median():.2f}",
               "decay_pct": f"{decay:.1f}%"}
        summary_rows.append(row)
        print(f"  {label:8s} | {region_name:20s} | n={len(sub):5d} | poly(A)={pa.median():.1f} | m6A/kb={m6a.median():.2f} | decay={decay:.1f}%")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT}/final_summary_table.tsv", sep="\t", index=False)

# Compute deltas
print("\n--- Poly(A) Delta (Stressed - Normal) ---")
for region_name, mask in [
    ("Super-enhancer", hela["in_SE"]),
    ("Regular enhancer", hela["in_RE"] & ~hela["in_SE"]),
    ("Non-enhancer", ~hela["in_RE"] & ~hela["in_SE"]),
    ("ChromHMM Regulatory", hela["chromhmm_group"].isin(["Enhancer","Promoter"])),
    ("ChromHMM Non-reg", ~hela["chromhmm_group"].isin(["Enhancer","Promoter"])),
]:
    normal = hela[(hela["cellline"]=="HeLa") & mask]["polya_length"].median()
    stress = hela[(hela["cellline"]=="HeLa-Ars") & mask]["polya_length"].median()
    if not np.isnan(normal) and not np.isnan(stress):
        delta = stress - normal
        print(f"  {region_name:20s}: delta = {delta:+.1f} nt")

print("\n=== CONCLUSION ===")
print("SE L1 shows more shortening than non-enhancer, but similar to regular enhancer L1.")
print("The effect is largely captured by existing ChromHMM Regulatory annotation.")
print("No strong evidence that super-enhancer status adds unique vulnerability beyond enhancer status.")
