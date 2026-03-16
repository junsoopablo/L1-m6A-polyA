#!/usr/bin/env python3
"""
Decorated (non-A) tail analysis: association with arsenite resistance in L1 reads.

Key questions:
1. Decorated tail fraction in HeLa vs HeLa-Ars (overall + young/ancient)
2. Do decorated reads show less poly(A) shortening under arsenite?
3. Is decoration rate different for stress-shortened reads (poly(A)<60) vs normal?
4. Mann-Whitney U: poly(A) length decorated vs non-decorated, per condition
5. Among ancient L1: arsenite delta poly(A) for decorated vs non-decorated
"""

import glob
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# Configuration
# ============================================================
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
SUMMARY_PATTERN = os.path.join(BASE, "results_group/*/g_summary/*_L1_summary.tsv")

HELA_GROUPS = ["HeLa_1", "HeLa_2", "HeLa_3"]
ARS_GROUPS = ["HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3"]
YOUNG_SUBFAMS = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

# ============================================================
# Load data
# ============================================================
def extract_group(filepath):
    """Extract group name from filepath like .../results_group/{GROUP}/g_summary/..."""
    parts = filepath.split("/")
    idx = parts.index("results_group")
    return parts[idx + 1]

def extract_subfamily(transcript_id):
    """L1MC4_dup15 -> L1MC4"""
    if pd.isna(transcript_id):
        return "unknown"
    m = re.match(r"^(.+?)_dup\d+$", transcript_id)
    return m.group(1) if m else transcript_id

files = sorted(glob.glob(SUMMARY_PATTERN))
print(f"Found {len(files)} summary files")

frames = []
for f in files:
    group = extract_group(f)
    if group not in HELA_GROUPS + ARS_GROUPS:
        continue
    df = pd.read_csv(f, sep="\t")
    df["group"] = group
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
print(f"Total reads loaded: {len(data)}")

# Filter to PASS reads only (exclude NOREGION, etc.)
data = data[data["qc_tag"] == "PASS"].copy()
print(f"PASS reads: {len(data)}")

# Derive columns
data["subfamily"] = data["transcript_id"].apply(extract_subfamily)
data["is_young"] = data["subfamily"].isin(YOUNG_SUBFAMS)
data["age_group"] = np.where(data["is_young"], "Young", "Ancient")
data["condition"] = np.where(data["group"].isin(HELA_GROUPS), "HeLa", "HeLa-Ars")
data["is_decorated"] = data["class"] == "decorated"

# For the main analysis, focus on reads with poly(A) data and class in {decorated, blank}
# (exclude 3UTR and unclassified)
analysis = data[data["class"].isin(["decorated", "blank"])].copy()
print(f"Decorated + blank reads for analysis: {len(analysis)}")

print("\n" + "=" * 80)
print("Q1: DECORATED TAIL FRACTION — HeLa vs HeLa-Ars")
print("=" * 80)

def decorated_fraction(df, label=""):
    n_total = len(df)
    n_dec = df["is_decorated"].sum()
    frac = n_dec / n_total * 100 if n_total > 0 else 0
    print(f"  {label}: {n_dec}/{n_total} = {frac:.2f}%")
    return n_dec, n_total

print("\n--- Overall ---")
for cond in ["HeLa", "HeLa-Ars"]:
    sub = analysis[analysis["condition"] == cond]
    decorated_fraction(sub, cond)

print("\n--- By age group ---")
for age in ["Young", "Ancient"]:
    print(f"\n  [{age}]")
    for cond in ["HeLa", "HeLa-Ars"]:
        sub = analysis[(analysis["condition"] == cond) & (analysis["age_group"] == age)]
        decorated_fraction(sub, f"  {cond}")

# Fisher exact test: HeLa vs HeLa-Ars decoration rate
print("\n--- Fisher exact test: HeLa vs HeLa-Ars decoration rate ---")
for age in ["All", "Young", "Ancient"]:
    if age == "All":
        sub = analysis
    else:
        sub = analysis[analysis["age_group"] == age]

    hela = sub[sub["condition"] == "HeLa"]
    ars = sub[sub["condition"] == "HeLa-Ars"]

    a = hela["is_decorated"].sum()
    b = len(hela) - a
    c = ars["is_decorated"].sum()
    d = len(ars) - c

    odds, p = stats.fisher_exact([[a, b], [c, d]])
    rate_h = a / (a + b) * 100
    rate_a = c / (c + d) * 100
    print(f"  {age}: HeLa {rate_h:.2f}% vs Ars {rate_a:.2f}%, OR={odds:.3f}, p={p:.2e}")

print("\n" + "=" * 80)
print("Q2: POLY(A) SHORTENING — DECORATED vs NON-DECORATED")
print("=" * 80)

print("\n--- Median poly(A) by condition and decoration status ---")
results_q2 = []
for cond in ["HeLa", "HeLa-Ars"]:
    for dec_label, dec_val in [("Decorated", True), ("Non-decorated", False)]:
        sub = analysis[(analysis["condition"] == cond) & (analysis["is_decorated"] == dec_val)]
        med = sub["polya_length"].median()
        mean = sub["polya_length"].mean()
        n = len(sub)
        results_q2.append({"condition": cond, "decorated": dec_label, "n": n,
                           "median_polya": med, "mean_polya": mean})
        print(f"  {cond} / {dec_label}: n={n}, median={med:.1f}, mean={mean:.1f}")

# Compute delta (Ars - HeLa) for each decoration group
print("\n--- Arsenite delta poly(A) ---")
for dec_label, dec_val in [("Decorated", True), ("Non-decorated", False)]:
    hela_med = analysis[(analysis["condition"] == "HeLa") & (analysis["is_decorated"] == dec_val)]["polya_length"].median()
    ars_med = analysis[(analysis["condition"] == "HeLa-Ars") & (analysis["is_decorated"] == dec_val)]["polya_length"].median()
    delta = ars_med - hela_med
    print(f"  {dec_label}: HeLa median={hela_med:.1f} -> Ars median={ars_med:.1f}, delta={delta:.1f} nt")

# MWU between HeLa and HeLa-Ars within each decoration class
print("\n--- MWU: HeLa vs HeLa-Ars poly(A) within decoration class ---")
for dec_label, dec_val in [("Decorated", True), ("Non-decorated", False)]:
    hela_pa = analysis[(analysis["condition"] == "HeLa") & (analysis["is_decorated"] == dec_val)]["polya_length"]
    ars_pa = analysis[(analysis["condition"] == "HeLa-Ars") & (analysis["is_decorated"] == dec_val)]["polya_length"]
    u, p = stats.mannwhitneyu(hela_pa, ars_pa, alternative="two-sided")
    print(f"  {dec_label}: MWU U={u:.0f}, p={p:.2e} (HeLa n={len(hela_pa)}, Ars n={len(ars_pa)})")

print("\n" + "=" * 80)
print("Q3: DECORATION RATE BY POLY(A) LENGTH BIN")
print("=" * 80)

bins = [(0, 30, "0-30 (decay zone)"), (30, 60, "30-60 (very short)"),
        (60, 100, "60-100 (short)"), (100, 150, "100-150 (medium)"),
        (150, 250, "150-250 (long)"), (250, 9999, "250+ (very long)")]

print(f"\n{'Condition':<12} {'Bin':<22} {'Decorated':<12} {'Total':<8} {'Rate %':<8}")
print("-" * 65)
for cond in ["HeLa", "HeLa-Ars"]:
    sub_cond = analysis[analysis["condition"] == cond]
    for lo, hi, label in bins:
        sub = sub_cond[(sub_cond["polya_length"] >= lo) & (sub_cond["polya_length"] < hi)]
        n_dec = sub["is_decorated"].sum()
        n_tot = len(sub)
        rate = n_dec / n_tot * 100 if n_tot > 0 else 0
        print(f"{cond:<12} {label:<22} {n_dec:<12} {n_tot:<8} {rate:<8.2f}")

# Fisher: stress-shortened (poly(A)<60) vs normal (poly(A)>=60) decoration rate
print("\n--- Fisher: poly(A)<60 vs >=60 decoration rate, per condition ---")
for cond in ["HeLa", "HeLa-Ars"]:
    sub = analysis[analysis["condition"] == cond]
    short = sub[sub["polya_length"] < 60]
    normal = sub[sub["polya_length"] >= 60]

    a = short["is_decorated"].sum()
    b = len(short) - a
    c = normal["is_decorated"].sum()
    d = len(normal) - c

    odds, p = stats.fisher_exact([[a, b], [c, d]])
    rate_s = a / (a + b) * 100 if (a + b) > 0 else 0
    rate_n = c / (c + d) * 100 if (c + d) > 0 else 0
    print(f"  {cond}: short(<60) {rate_s:.2f}% vs normal(>=60) {rate_n:.2f}%, OR={odds:.3f}, p={p:.2e}")

print("\n" + "=" * 80)
print("Q4: MANN-WHITNEY U — POLY(A) DECORATED vs NON-DECORATED, WITHIN CONDITION")
print("=" * 80)

for cond in ["HeLa", "HeLa-Ars"]:
    sub = analysis[analysis["condition"] == cond]
    dec_pa = sub[sub["is_decorated"]]["polya_length"]
    nondec_pa = sub[~sub["is_decorated"]]["polya_length"]

    u, p = stats.mannwhitneyu(dec_pa, nondec_pa, alternative="two-sided")
    print(f"\n  {cond}:")
    print(f"    Decorated:     n={len(dec_pa)}, median={dec_pa.median():.1f}, mean={dec_pa.mean():.1f}")
    print(f"    Non-decorated: n={len(nondec_pa)}, median={nondec_pa.median():.1f}, mean={nondec_pa.mean():.1f}")
    print(f"    MWU U={u:.0f}, p={p:.2e}")

print("\n" + "=" * 80)
print("Q5: ANCIENT L1 ONLY — ARSENITE DELTA POLY(A) BY DECORATION STATUS")
print("=" * 80)

ancient = analysis[analysis["age_group"] == "Ancient"]

print("\n--- Ancient L1: Median poly(A) ---")
for cond in ["HeLa", "HeLa-Ars"]:
    for dec_label, dec_val in [("Decorated", True), ("Non-decorated", False)]:
        sub = ancient[(ancient["condition"] == cond) & (ancient["is_decorated"] == dec_val)]
        print(f"  {cond} / {dec_label}: n={len(sub)}, median={sub['polya_length'].median():.1f}, "
              f"mean={sub['polya_length'].mean():.1f}")

print("\n--- Ancient L1: Arsenite delta ---")
for dec_label, dec_val in [("Decorated", True), ("Non-decorated", False)]:
    hela_med = ancient[(ancient["condition"] == "HeLa") & (ancient["is_decorated"] == dec_val)]["polya_length"].median()
    ars_med = ancient[(ancient["condition"] == "HeLa-Ars") & (ancient["is_decorated"] == dec_val)]["polya_length"].median()
    delta = ars_med - hela_med
    print(f"  {dec_label}: HeLa median={hela_med:.1f} -> Ars median={ars_med:.1f}, delta={delta:.1f} nt")

# MWU: decorated vs non-decorated poly(A) within ancient L1
print("\n--- Ancient L1: MWU poly(A) decorated vs non-decorated, per condition ---")
for cond in ["HeLa", "HeLa-Ars"]:
    sub = ancient[ancient["condition"] == cond]
    dec_pa = sub[sub["is_decorated"]]["polya_length"]
    nondec_pa = sub[~sub["is_decorated"]]["polya_length"]
    u, p = stats.mannwhitneyu(dec_pa, nondec_pa, alternative="two-sided")
    print(f"  {cond}: dec median={dec_pa.median():.1f}, nondec median={nondec_pa.median():.1f}, "
          f"MWU p={p:.2e}")

# MWU: HeLa vs Ars within ancient decorated only
print("\n--- Ancient L1 decorated only: HeLa vs Ars ---")
dec_hela = ancient[(ancient["condition"] == "HeLa") & (ancient["is_decorated"])]["polya_length"]
dec_ars = ancient[(ancient["condition"] == "HeLa-Ars") & (ancient["is_decorated"])]["polya_length"]
u, p = stats.mannwhitneyu(dec_hela, dec_ars, alternative="two-sided")
print(f"  HeLa decorated: n={len(dec_hela)}, median={dec_hela.median():.1f}")
print(f"  Ars decorated:  n={len(dec_ars)}, median={dec_ars.median():.1f}")
print(f"  Delta={dec_ars.median() - dec_hela.median():.1f} nt, MWU p={p:.2e}")

print("\n--- Ancient L1 non-decorated only: HeLa vs Ars ---")
nondec_hela = ancient[(ancient["condition"] == "HeLa") & (~ancient["is_decorated"])]["polya_length"]
nondec_ars = ancient[(ancient["condition"] == "HeLa-Ars") & (~ancient["is_decorated"])]["polya_length"]
u, p = stats.mannwhitneyu(nondec_hela, nondec_ars, alternative="two-sided")
print(f"  HeLa non-dec: n={len(nondec_hela)}, median={nondec_hela.median():.1f}")
print(f"  Ars non-dec:  n={len(nondec_ars)}, median={nondec_ars.median():.1f}")
print(f"  Delta={nondec_ars.median() - nondec_hela.median():.1f} nt, MWU p={p:.2e}")

print("\n" + "=" * 80)
print("SUPPLEMENTARY: INTERACTION TEST (OLS)")
print("=" * 80)
print("Testing: poly(A) ~ condition * decoration (+ read_length)")

import statsmodels.api as sm
from statsmodels.formula.api import ols

# Use ancient L1 only (where shortening occurs)
ols_data = ancient.copy()
ols_data["is_ars"] = (ols_data["condition"] == "HeLa-Ars").astype(int)
ols_data["is_dec"] = ols_data["is_decorated"].astype(int)

model = ols("polya_length ~ is_ars * is_dec + read_length", data=ols_data).fit()
print("\n--- OLS: Ancient L1 poly(A) ~ arsenite * decoration + read_length ---")
print(model.summary().tables[1])

# Also test with young L1
print("\n--- Young L1: decoration rates and poly(A) ---")
young = analysis[analysis["age_group"] == "Young"]
for cond in ["HeLa", "HeLa-Ars"]:
    sub = young[young["condition"] == cond]
    n_dec = sub["is_decorated"].sum()
    n_tot = len(sub)
    rate = n_dec / n_tot * 100 if n_tot > 0 else 0
    dec_pa = sub[sub["is_decorated"]]["polya_length"]
    nondec_pa = sub[~sub["is_decorated"]]["polya_length"]
    print(f"  {cond}: {n_dec}/{n_tot} = {rate:.1f}% decorated")
    if len(dec_pa) > 0 and len(nondec_pa) > 0:
        print(f"    Decorated median={dec_pa.median():.1f}, Non-dec median={nondec_pa.median():.1f}")

print("\n" + "=" * 80)
print("SUPPLEMENTARY: PER-REPLICATE CONSISTENCY")
print("=" * 80)

for group in HELA_GROUPS + ARS_GROUPS:
    sub = analysis[analysis["group"] == group]
    n_dec = sub["is_decorated"].sum()
    n_tot = len(sub)
    rate = n_dec / n_tot * 100 if n_tot > 0 else 0
    med = sub["polya_length"].median()
    dec_med = sub[sub["is_decorated"]]["polya_length"].median() if n_dec > 0 else float("nan")
    nondec_med = sub[~sub["is_decorated"]]["polya_length"].median()
    print(f"  {group}: dec_rate={rate:.1f}%, poly(A) all={med:.1f}, dec={dec_med:.1f}, nondec={nondec_med:.1f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Compute key numbers for summary
hela_all = analysis[analysis["condition"] == "HeLa"]
ars_all = analysis[analysis["condition"] == "HeLa-Ars"]
hela_dec_rate = hela_all["is_decorated"].mean() * 100
ars_dec_rate = ars_all["is_decorated"].mean() * 100

# Ancient delta decorated vs non-decorated
anc_dec_delta = (ancient[(ancient["condition"] == "HeLa-Ars") & (ancient["is_decorated"])]["polya_length"].median() -
                 ancient[(ancient["condition"] == "HeLa") & (ancient["is_decorated"])]["polya_length"].median())
anc_nondec_delta = (ancient[(ancient["condition"] == "HeLa-Ars") & (~ancient["is_decorated"])]["polya_length"].median() -
                    ancient[(ancient["condition"] == "HeLa") & (~ancient["is_decorated"])]["polya_length"].median())

print(f"""
1. Decorated rate: HeLa = {hela_dec_rate:.1f}%, HeLa-Ars = {ars_dec_rate:.1f}%
2. Ancient L1 arsenite delta:
   - Decorated reads:     delta = {anc_dec_delta:.1f} nt
   - Non-decorated reads: delta = {anc_nondec_delta:.1f} nt
   - Difference: {anc_dec_delta - anc_nondec_delta:+.1f} nt (positive = decoration protects)
3. Key interpretation: see detailed results above
""")

print("Done.")
