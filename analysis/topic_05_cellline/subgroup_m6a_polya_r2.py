#!/usr/bin/env python3
"""
Subgroup m6A–poly(A) R² analysis.

Test whether the m6A–poly(A) correlation strengthens when restricting to
biologically relevant L1 subgroups (stressed Ancient L1, high-overlap, etc.).

Input:
  - l1_chromhmm_annotated.tsv  (m6a_per_kb, polya_length, l1_age, chromhmm_group, condition)
  - L1 summary files           (overlap_length, read_length → overlap_frac)
  - Part3 caches               (m6a_sites_high, read_length for reads without ChromHMM)

Output:
  - subgroup_m6a_r2/subgroup_r2_summary.tsv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
CHROMHMM = os.path.join(BASE, "analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv")
OUTDIR = os.path.join(BASE, "analysis/01_exploration/topic_05_cellline/subgroup_m6a_r2")
os.makedirs(OUTDIR, exist_ok=True)

HELA_GROUPS = {
    "HeLa":     ["HeLa_1", "HeLa_2", "HeLa_3"],
    "HeLa-Ars": ["HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3"],
}

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}
REGULATORY_GROUPS = {"Enhancer", "Promoter"}  # chromhmm_group values


# ── 1. Load ChromHMM annotated data (HeLa + HeLa-Ars only) ────────
print("=== Loading ChromHMM annotated data ===")
chromhmm = pd.read_csv(CHROMHMM, sep="\t")
print(f"  Total rows: {len(chromhmm)}")
print(f"  Columns: {list(chromhmm.columns)}")

# Filter to HeLa only
hela_mask = chromhmm["cellline"].isin(["HeLa", "HeLa-Ars"])
df_chrom = chromhmm[hela_mask].copy()
print(f"  HeLa + HeLa-Ars rows: {len(df_chrom)}")
print(f"  Condition counts: {df_chrom['condition'].value_counts().to_dict()}")


# ── 2. Load L1 summaries for overlap_frac ─────────────────────────
print("\n=== Loading L1 summaries for overlap_frac ===")
summary_frames = []
for label, groups in HELA_GROUPS.items():
    for grp in groups:
        fpath = os.path.join(BASE, f"results_group/{grp}/g_summary/{grp}_L1_summary.tsv")
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        tmp = pd.read_csv(fpath, sep="\t", usecols=["read_id", "read_length", "overlap_length", "gene_id", "qc_tag"])
        tmp["overlap_frac"] = tmp["overlap_length"] / tmp["read_length"]
        summary_frames.append(tmp)

summary = pd.concat(summary_frames, ignore_index=True)
# gene_id in L1 summary = subfamily (e.g. L1HS, L1PA2, L1MC4)
print(f"  Total L1 summary reads (HeLa all): {len(summary)}")


# ── 3. Merge overlap_frac into ChromHMM data ──────────────────────
print("\n=== Merging overlap_frac ===")
df = df_chrom.merge(
    summary[["read_id", "overlap_frac", "read_length"]].drop_duplicates("read_id"),
    on="read_id",
    how="left"
)
print(f"  After merge: {len(df)} rows")
print(f"  overlap_frac available: {df['overlap_frac'].notna().sum()}")
print(f"  overlap_frac missing:   {df['overlap_frac'].isna().sum()}")

# For missing overlap_frac, also try Part3 cache to get read_length
# (ChromHMM file already has m6a_per_kb but not read_length for all)
# Fill missing overlap_frac from Part3 cache
if df['overlap_frac'].isna().any():
    print("  Filling missing from Part3 cache...")
    missing_ids = set(df.loc[df['overlap_frac'].isna(), 'read_id'])
    part3_frames = []
    for label, groups in HELA_GROUPS.items():
        for grp in groups:
            fpath = os.path.join(BASE, f"analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache/{grp}_l1_per_read.tsv")
            if not os.path.exists(fpath):
                continue
            tmp = pd.read_csv(fpath, sep="\t", usecols=["read_id", "read_length"])
            part3_frames.append(tmp)
    if part3_frames:
        part3 = pd.concat(part3_frames, ignore_index=True).drop_duplicates("read_id")
        # We can't compute overlap_frac without overlap_length from summary
        # So these reads simply won't have overlap_frac — that's OK for subgroups a-c, f-i


# ── 4. Determine Young/Ancient from gene_id in ChromHMM file ──────
# In the ChromHMM annotated file, gene_id = subfamily
df["is_young"] = df["gene_id"].isin(YOUNG_SUBFAMILIES)
df["age_group"] = np.where(df["is_young"], "Young", "Ancient")

# Condition: "normal" → unstressed, "stress" → stressed
df["is_stressed"] = df["condition"] == "stress"

# Regulatory
df["is_regulatory"] = df["chromhmm_group"].isin(REGULATORY_GROUPS)

print(f"\n  Age distribution: {df['age_group'].value_counts().to_dict()}")
print(f"  Condition: {df['condition'].value_counts().to_dict()}")
print(f"  Regulatory: {df['is_regulatory'].value_counts().to_dict()}")


# ── 5. Define subgroups ───────────────────────────────────────────
def make_subgroup(name, mask, df=df):
    """Return subgroup DataFrame with label."""
    sub = df[mask].copy()
    return name, sub

subgroups = []

# a. Baseline: all L1, HeLa + HeLa-Ars
subgroups.append(make_subgroup(
    "a. All L1 (HeLa+Ars)",
    pd.Series(True, index=df.index)
))

# b. Stressed only
subgroups.append(make_subgroup(
    "b. All L1 (Ars only)",
    df["is_stressed"]
))

# c. Stressed + Ancient
subgroups.append(make_subgroup(
    "c. Stressed + Ancient",
    df["is_stressed"] & (df["age_group"] == "Ancient")
))

# d. Stressed + Ancient + overlap > 0.5
subgroups.append(make_subgroup(
    "d. Stressed + Ancient + OV>0.5",
    df["is_stressed"] & (df["age_group"] == "Ancient") & (df["overlap_frac"] > 0.5)
))

# e. Stressed + Ancient + overlap > 0.7
subgroups.append(make_subgroup(
    "e. Stressed + Ancient + OV>0.7",
    df["is_stressed"] & (df["age_group"] == "Ancient") & (df["overlap_frac"] > 0.7)
))

# f. Stressed + Ancient + Regulatory
subgroups.append(make_subgroup(
    "f. Stressed + Ancient + Regulatory",
    df["is_stressed"] & (df["age_group"] == "Ancient") & df["is_regulatory"]
))

# g. Stressed + Ancient + non-Regulatory
subgroups.append(make_subgroup(
    "g. Stressed + Ancient + non-Reg",
    df["is_stressed"] & (df["age_group"] == "Ancient") & ~df["is_regulatory"]
))

# h. Stressed + Young (expect no correlation)
subgroups.append(make_subgroup(
    "h. Stressed + Young",
    df["is_stressed"] & (df["age_group"] == "Young")
))

# i. Unstressed + Ancient (expect weak)
subgroups.append(make_subgroup(
    "i. Unstressed + Ancient",
    ~df["is_stressed"] & (df["age_group"] == "Ancient")
))


# ── 6. Compute OLS for each subgroup ──────────────────────────────
def compute_stats(name, sub_df):
    """Compute OLS regression and Spearman correlation."""
    # Drop NaN in key columns
    valid = sub_df.dropna(subset=["m6a_per_kb", "polya_length"])
    n = len(valid)

    if n < 10:
        return {
            "subgroup": name, "N": n,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "ols_beta": np.nan, "ols_beta_p": np.nan,
            "ols_R2": np.nan, "ols_R2_adj": np.nan,
            "mean_m6a_per_kb": np.nan, "mean_polya": np.nan,
            "median_polya": np.nan,
        }

    x = valid["m6a_per_kb"].values
    y = valid["polya_length"].values

    # Spearman
    rho, sp_p = spearmanr(x, y)

    # OLS: polya ~ m6a_per_kb
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    beta = model.params[1]
    beta_p = model.pvalues[1]
    r2 = model.rsquared
    r2_adj = model.rsquared_adj

    return {
        "subgroup": name,
        "N": n,
        "spearman_rho": rho,
        "spearman_p": sp_p,
        "ols_beta": beta,
        "ols_beta_p": beta_p,
        "ols_R2": r2,
        "ols_R2_adj": r2_adj,
        "mean_m6a_per_kb": np.mean(x),
        "mean_polya": np.mean(y),
        "median_polya": np.median(y),
    }


def compute_interaction(name, sub_df):
    """
    For combined stressed+unstressed groups:
    polya ~ m6a_per_kb * stress_condition
    Returns partial R² for stress×m6a interaction.
    """
    valid = sub_df.dropna(subset=["m6a_per_kb", "polya_length"])
    n = len(valid)

    if n < 20:
        return np.nan, np.nan, np.nan

    # Need both conditions present
    if valid["is_stressed"].nunique() < 2:
        return np.nan, np.nan, np.nan

    y = valid["polya_length"].values
    m6a = valid["m6a_per_kb"].values
    stress = valid["is_stressed"].astype(float).values
    interaction = m6a * stress

    # Full model: polya ~ m6a + stress + m6a*stress
    X_full = sm.add_constant(np.column_stack([m6a, stress, interaction]))
    model_full = sm.OLS(y, X_full).fit()

    # Reduced model: polya ~ m6a + stress
    X_reduced = sm.add_constant(np.column_stack([m6a, stress]))
    model_reduced = sm.OLS(y, X_reduced).fit()

    # Partial R² for interaction = (R²_full - R²_reduced) / (1 - R²_reduced)
    partial_r2 = (model_full.rsquared - model_reduced.rsquared) / (1 - model_reduced.rsquared)

    # Interaction term p-value (last coefficient in full model)
    interaction_beta = model_full.params[3]
    interaction_p = model_full.pvalues[3]

    return partial_r2, interaction_beta, interaction_p


print("\n" + "=" * 100)
print("SUBGROUP m6A–POLY(A) R² ANALYSIS")
print("=" * 100)

results = []
for name, sub_df in subgroups:
    stats = compute_stats(name, sub_df)

    # For subgroup (a) which has both conditions, compute interaction
    if name.startswith("a."):
        partial_r2, int_beta, int_p = compute_interaction(name, sub_df)
        stats["interaction_partial_R2"] = partial_r2
        stats["interaction_beta"] = int_beta
        stats["interaction_p"] = int_p
    else:
        stats["interaction_partial_R2"] = np.nan
        stats["interaction_beta"] = np.nan
        stats["interaction_p"] = np.nan

    results.append(stats)

results_df = pd.DataFrame(results)

# ── Pretty print ──
print(f"\n{'Subgroup':<38s} {'N':>6s} {'Spearman_rho':>13s} {'Spearman_P':>11s} "
      f"{'OLS_β':>8s} {'OLS_P':>11s} {'OLS_R²':>8s} {'Med_polyA':>10s}")
print("-" * 120)

for _, row in results_df.iterrows():
    sp_p_str = f"{row['spearman_p']:.2e}" if not np.isnan(row['spearman_p']) else "NA"
    beta_p_str = f"{row['ols_beta_p']:.2e}" if not np.isnan(row['ols_beta_p']) else "NA"

    print(f"{row['subgroup']:<38s} {row['N']:>6.0f} {row['spearman_rho']:>13.4f} {sp_p_str:>11s} "
          f"{row['ols_beta']:>8.3f} {beta_p_str:>11s} {row['ols_R2']:>8.4f} {row['median_polya']:>10.1f}")

# Interaction model for baseline
baseline = results_df[results_df["subgroup"].str.startswith("a.")]
if not baseline.empty:
    row = baseline.iloc[0]
    print(f"\n--- Interaction model (subgroup a) ---")
    print(f"  Full model: polya ~ m6a + stress + m6a×stress")
    if not np.isnan(row["interaction_partial_R2"]):
        print(f"  Interaction β:          {row['interaction_beta']:.3f}")
        print(f"  Interaction P:          {row['interaction_p']:.2e}")
        print(f"  Partial R² (m6a×stress): {row['interaction_partial_R2']:.5f} ({row['interaction_partial_R2']*100:.3f}%)")


# ── 7. Additional: Full interaction model for each subgroup that has BOTH conditions ──
# For subgroups with only one condition, interaction is N/A
# Let's also compute interaction for "All Ancient (HeLa+Ars)" and "All Young (HeLa+Ars)"

print("\n\n" + "=" * 100)
print("ADDITIONAL: INTERACTION MODELS (polya ~ m6a * stress) BY AGE GROUP")
print("=" * 100)

extra_subgroups = [
    ("All Ancient (HeLa+Ars)", df[df["age_group"] == "Ancient"]),
    ("All Young (HeLa+Ars)",   df[df["age_group"] == "Young"]),
    ("All L1 OV>0.5",         df[df["overlap_frac"] > 0.5]),
    ("All L1 OV>0.7",         df[df["overlap_frac"] > 0.7]),
    ("Ancient + Regulatory",   df[(df["age_group"] == "Ancient") & df["is_regulatory"]]),
]

for ename, edf in extra_subgroups:
    valid = edf.dropna(subset=["m6a_per_kb", "polya_length"])
    n = len(valid)

    # Simple OLS within stressed subset only
    stressed_sub = valid[valid["is_stressed"]]
    if len(stressed_sub) > 10:
        x = stressed_sub["m6a_per_kb"].values
        y = stressed_sub["polya_length"].values
        rho, sp_p = spearmanr(x, y)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        print(f"\n  {ename} [stressed only]: N={len(stressed_sub)}, "
              f"rho={rho:.4f} (P={sp_p:.2e}), R²={model.rsquared:.4f}, "
              f"β={model.params[1]:.3f} (P={model.pvalues[1]:.2e})")

    # Interaction model
    partial_r2, int_beta, int_p = compute_interaction(ename, edf)
    if not np.isnan(partial_r2):
        print(f"  {ename} [interaction]:    N={n}, "
              f"int_β={int_beta:.3f} (P={int_p:.2e}), "
              f"partial_R²={partial_r2:.5f} ({partial_r2*100:.3f}%)")


# ── 8. Decay zone analysis per subgroup ───────────────────────────
print("\n\n" + "=" * 100)
print("DECAY ZONE (<30nt) FRACTION BY m6A QUARTILE")
print("=" * 100)

for name, sub_df in subgroups:
    valid = sub_df.dropna(subset=["m6a_per_kb", "polya_length"])
    if len(valid) < 40:
        continue

    try:
        valid = valid.copy()
        valid["m6a_q"] = pd.qcut(valid["m6a_per_kb"], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
    except ValueError:
        continue

    valid["decay_zone"] = valid["polya_length"] < 30

    q1 = valid[valid["m6a_q"] == "Q1(low)"]
    q4 = valid[valid["m6a_q"] == "Q4(high)"]

    if len(q1) > 0 and len(q4) > 0:
        dz_q1 = q1["decay_zone"].mean() * 100
        dz_q4 = q4["decay_zone"].mean() * 100
        ratio = dz_q1 / dz_q4 if dz_q4 > 0 else np.inf
        print(f"  {name:<38s} Q1={dz_q1:5.1f}% Q4={dz_q4:5.1f}% ratio={ratio:.2f}x")


# ── 9. Save results ──────────────────────────────────────────────
outpath = os.path.join(OUTDIR, "subgroup_r2_summary.tsv")
results_df.to_csv(outpath, sep="\t", index=False, float_format="%.6f")
print(f"\n\nResults saved to: {outpath}")


# ── 10. Key comparison summary ────────────────────────────────────
print("\n\n" + "=" * 100)
print("KEY COMPARISONS")
print("=" * 100)

baseline_r2 = results_df.loc[results_df["subgroup"].str.startswith("a."), "ols_R2"].values[0]
print(f"\nBaseline R² (all L1, HeLa+Ars): {baseline_r2:.4f}")

for _, row in results_df.iterrows():
    if row["subgroup"].startswith("a."):
        continue
    fold = row["ols_R2"] / baseline_r2 if baseline_r2 > 0 else np.nan
    print(f"  {row['subgroup']:<38s} R²={row['ols_R2']:.4f} ({fold:.1f}x baseline)  "
          f"rho={row['spearman_rho']:.4f}  β={row['ols_beta']:.2f}")

print("\nDone.")
