#!/usr/bin/env python3
"""
PUS7/DKC1 KD ONT DRS → L1 Expression Analysis
PRJNA1220613: BE(2)-C neuroblastoma, shPUS7/shDKC1 vs shGFP control

Analyzes L1 read counts from ONT DRS after PUS7 or DKC1 knockdown.
"""

import os
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

# --- Configuration ---
DATA_DIR = "/vault/external-datasets/2026/PRJNA1220613_PUS7_DKC1_KD_RNA002"
OUT_DIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/pus_enzyme_analysis"

SAMPLES = {
    "shPUS7_rep1": {"condition": "shPUS7", "rep": 1},
    "shPUS7_rep2": {"condition": "shPUS7", "rep": 2},
    "shGFP_rep1":  {"condition": "shGFP",  "rep": 1},
    "shGFP_rep2":  {"condition": "shGFP",  "rep": 2},
    "shDKC1_rep1": {"condition": "shDKC1", "rep": 1},
    "shDKC1_rep2": {"condition": "shDKC1", "rep": 2},
}

YOUNG_L1 = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}
SAMTOOLS = subprocess.check_output(
    "conda run -n bioinfo3 which samtools", shell=True
).decode().strip()


def get_total_mapped(bam_path):
    """Count total mapped reads in BAM."""
    result = subprocess.run(
        [SAMTOOLS, "view", "-c", "-F", "4", bam_path],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def load_l1_reads(sample_name):
    """Load L1 read info: read_id, read_len, subfamily, locus_id."""
    info_file = os.path.join(DATA_DIR, f"{sample_name}_L1_reads_info.tsv")
    if not os.path.exists(info_file) or os.path.getsize(info_file) == 0:
        return pd.DataFrame(columns=["read_id", "read_len", "subfamily", "locus_id"])
    df = pd.read_csv(info_file, sep="\t", header=None,
                     names=["read_id", "read_len", "subfamily", "locus_id"])
    return df


def classify_age(subfamily):
    """Classify L1 subfamily as young or ancient."""
    return "young" if subfamily in YOUNG_L1 else "ancient"


def fisher_exact_test(a, b, c, d):
    """2x2 Fisher exact test. Returns OR and p-value."""
    table = np.array([[a, b], [c, d]])
    oddsratio, pvalue = stats.fisher_exact(table)
    return oddsratio, pvalue


def main():
    print("=" * 70)
    print("PUS7/DKC1 KD ONT DRS → L1 Expression Analysis")
    print("=" * 70)

    # --- 1. Load data ---
    print("\n--- 1. Loading sample data ---")
    sample_data = {}
    for sample_name, meta in SAMPLES.items():
        bam_path = os.path.join(DATA_DIR, f"{sample_name}.sorted.bam")
        total_mapped = get_total_mapped(bam_path)
        l1_df = load_l1_reads(sample_name)
        l1_df["age"] = l1_df["subfamily"].apply(classify_age)
        n_l1 = len(l1_df)
        rpm = n_l1 / total_mapped * 1e6 if total_mapped > 0 else 0

        sample_data[sample_name] = {
            "condition": meta["condition"],
            "rep": meta["rep"],
            "total_mapped": total_mapped,
            "n_l1": n_l1,
            "l1_rpm": rpm,
            "l1_df": l1_df,
            "n_young": (l1_df["age"] == "young").sum(),
            "n_ancient": (l1_df["age"] == "ancient").sum(),
        }
        print(f"  {sample_name}: {total_mapped:,} total, {n_l1:,} L1 "
              f"({rpm:.1f} RPM), {sample_data[sample_name]['n_young']} young, "
              f"{sample_data[sample_name]['n_ancient']} ancient")

    # --- 2. Per-condition summary ---
    print("\n--- 2. Per-condition summary ---")
    conditions = {}
    for cond in ["shGFP", "shPUS7", "shDKC1"]:
        reps = {k: v for k, v in sample_data.items() if v["condition"] == cond}
        total_mapped = sum(v["total_mapped"] for v in reps.values())
        n_l1 = sum(v["n_l1"] for v in reps.values())
        n_young = sum(v["n_young"] for v in reps.values())
        n_ancient = sum(v["n_ancient"] for v in reps.values())
        rpm = n_l1 / total_mapped * 1e6 if total_mapped > 0 else 0
        l1_df = pd.concat([v["l1_df"] for v in reps.values()], ignore_index=True)

        conditions[cond] = {
            "total_mapped": total_mapped,
            "n_l1": n_l1,
            "n_young": n_young,
            "n_ancient": n_ancient,
            "l1_rpm": rpm,
            "l1_df": l1_df,
            "rep_rpms": [v["l1_rpm"] for v in reps.values()],
        }
        print(f"  {cond}: {total_mapped:,} total, {n_l1:,} L1 ({rpm:.1f} RPM), "
              f"young={n_young}, ancient={n_ancient}")

    # --- 3. Replicate consistency ---
    print("\n--- 3. Replicate consistency ---")
    for cond in ["shGFP", "shPUS7", "shDKC1"]:
        reps = {k: v for k, v in sample_data.items() if v["condition"] == cond}
        rep_names = sorted(reps.keys())
        # Subfamily count vectors for correlation
        sf_counts = {}
        for rname in rep_names:
            vc = reps[rname]["l1_df"]["subfamily"].value_counts()
            sf_counts[rname] = vc
        all_sfs = sorted(set().union(*[set(v.index) for v in sf_counts.values()]))
        vec1 = np.array([sf_counts[rep_names[0]].get(sf, 0) for sf in all_sfs])
        vec2 = np.array([sf_counts[rep_names[1]].get(sf, 0) for sf in all_sfs])
        r, p = stats.pearsonr(vec1, vec2) if len(all_sfs) > 1 else (np.nan, np.nan)
        print(f"  {cond}: subfamily count Pearson r = {r:.4f} (p={p:.2e})")

    # --- 4. KD vs Control comparison (Fisher exact) ---
    print("\n--- 4. KD vs Control comparison ---")
    ctrl = conditions["shGFP"]

    comparisons = []
    for kd_name in ["shPUS7", "shDKC1"]:
        kd = conditions[kd_name]
        print(f"\n  === {kd_name} vs shGFP ===")

        # Total L1
        # Fisher: L1 vs non-L1, KD vs Ctrl
        a = kd["n_l1"]  # KD L1
        b = kd["total_mapped"] - kd["n_l1"]  # KD non-L1
        c = ctrl["n_l1"]  # Ctrl L1
        d = ctrl["total_mapped"] - ctrl["n_l1"]  # Ctrl non-L1
        OR, p = fisher_exact_test(a, b, c, d)
        fc = kd["l1_rpm"] / ctrl["l1_rpm"] if ctrl["l1_rpm"] > 0 else np.inf
        print(f"  Total L1: KD={a:,} ({kd['l1_rpm']:.1f} RPM), "
              f"Ctrl={c:,} ({ctrl['l1_rpm']:.1f} RPM)")
        print(f"    FC={fc:.3f}, OR={OR:.3f}, p={p:.2e}")
        comparisons.append({
            "comparison": f"{kd_name}_vs_shGFP",
            "category": "total_L1",
            "kd_count": a, "ctrl_count": c,
            "kd_rpm": kd["l1_rpm"], "ctrl_rpm": ctrl["l1_rpm"],
            "fold_change": fc, "odds_ratio": OR, "pvalue": p
        })

        # Young L1
        a_y = kd["n_young"]
        b_y = kd["total_mapped"] - a_y
        c_y = ctrl["n_young"]
        d_y = ctrl["total_mapped"] - c_y
        OR_y, p_y = fisher_exact_test(a_y, b_y, c_y, d_y)
        rpm_kd_y = a_y / kd["total_mapped"] * 1e6
        rpm_ctrl_y = c_y / ctrl["total_mapped"] * 1e6
        fc_y = rpm_kd_y / rpm_ctrl_y if rpm_ctrl_y > 0 else np.inf
        print(f"  Young L1: KD={a_y} ({rpm_kd_y:.1f} RPM), "
              f"Ctrl={c_y} ({rpm_ctrl_y:.1f} RPM)")
        print(f"    FC={fc_y:.3f}, OR={OR_y:.3f}, p={p_y:.2e}")
        comparisons.append({
            "comparison": f"{kd_name}_vs_shGFP",
            "category": "young_L1",
            "kd_count": a_y, "ctrl_count": c_y,
            "kd_rpm": rpm_kd_y, "ctrl_rpm": rpm_ctrl_y,
            "fold_change": fc_y, "odds_ratio": OR_y, "pvalue": p_y
        })

        # Ancient L1
        a_a = kd["n_ancient"]
        b_a = kd["total_mapped"] - a_a
        c_a = ctrl["n_ancient"]
        d_a = ctrl["total_mapped"] - c_a
        OR_a, p_a = fisher_exact_test(a_a, b_a, c_a, d_a)
        rpm_kd_a = a_a / kd["total_mapped"] * 1e6
        rpm_ctrl_a = c_a / ctrl["total_mapped"] * 1e6
        fc_a = rpm_kd_a / rpm_ctrl_a if rpm_ctrl_a > 0 else np.inf
        print(f"  Ancient L1: KD={a_a} ({rpm_kd_a:.1f} RPM), "
              f"Ctrl={c_a} ({rpm_ctrl_a:.1f} RPM)")
        print(f"    FC={fc_a:.3f}, OR={OR_a:.3f}, p={p_a:.2e}")
        comparisons.append({
            "comparison": f"{kd_name}_vs_shGFP",
            "category": "ancient_L1",
            "kd_count": a_a, "ctrl_count": c_a,
            "kd_rpm": rpm_kd_a, "ctrl_rpm": rpm_ctrl_a,
            "fold_change": fc_a, "odds_ratio": OR_a, "pvalue": p_a
        })

    # --- 5. Per-subfamily comparison ---
    print("\n--- 5. Top subfamily changes ---")
    for kd_name in ["shPUS7", "shDKC1"]:
        kd = conditions[kd_name]
        kd_sf = kd["l1_df"]["subfamily"].value_counts()
        ctrl_sf = ctrl["l1_df"]["subfamily"].value_counts()
        all_sfs = sorted(set(kd_sf.index) | set(ctrl_sf.index))

        sf_results = []
        for sf in all_sfs:
            nk = kd_sf.get(sf, 0)
            nc = ctrl_sf.get(sf, 0)
            rpm_k = nk / kd["total_mapped"] * 1e6
            rpm_c = nc / ctrl["total_mapped"] * 1e6
            fc = rpm_k / rpm_c if rpm_c > 0 else (np.inf if nk > 0 else 1.0)
            age = "young" if sf in YOUNG_L1 else "ancient"
            # Fisher exact
            a = nk; b = kd["total_mapped"] - nk
            c = nc; d = ctrl["total_mapped"] - nc
            OR, p = fisher_exact_test(a, b, c, d) if (nk + nc) >= 5 else (np.nan, np.nan)
            sf_results.append({
                "comparison": f"{kd_name}_vs_shGFP",
                "subfamily": sf, "age": age,
                "kd_count": nk, "ctrl_count": nc,
                "kd_rpm": rpm_k, "ctrl_rpm": rpm_c,
                "fold_change": fc, "odds_ratio": OR, "pvalue": p
            })
            comparisons.append({
                "comparison": f"{kd_name}_vs_shGFP",
                "category": f"subfamily_{sf}",
                "kd_count": nk, "ctrl_count": nc,
                "kd_rpm": rpm_k, "ctrl_rpm": rpm_c,
                "fold_change": fc, "odds_ratio": OR, "pvalue": p
            })

        sf_df = pd.DataFrame(sf_results).sort_values("pvalue")
        print(f"\n  === {kd_name} vs shGFP: Top subfamilies ===")
        sig = sf_df[sf_df["pvalue"] < 0.05]
        if len(sig) > 0:
            for _, row in sig.head(15).iterrows():
                print(f"    {row['subfamily']} ({row['age']}): "
                      f"KD={row['kd_count']}, Ctrl={row['ctrl_count']}, "
                      f"FC={row['fold_change']:.2f}, p={row['pvalue']:.2e}")
        else:
            print("    No significant subfamilies (p<0.05)")

        # Show all young subfamilies regardless of significance
        print(f"\n  Young L1 subfamilies:")
        for sf in sorted(YOUNG_L1):
            row = sf_df[sf_df["subfamily"] == sf]
            if len(row) > 0:
                row = row.iloc[0]
                print(f"    {sf}: KD={row['kd_count']}, Ctrl={row['ctrl_count']}, "
                      f"FC={row['fold_change']:.2f}, "
                      f"p={row['pvalue']:.2e}" if not np.isnan(row['pvalue'])
                      else f"    {sf}: KD={row['kd_count']}, Ctrl={row['ctrl_count']} (n<5)")
            else:
                print(f"    {sf}: 0 reads in both")

    # --- 6. Read length distribution ---
    print("\n--- 6. Read length distribution by condition ---")
    for cond_name, cond in conditions.items():
        l1_df = cond["l1_df"]
        if len(l1_df) > 0:
            med = l1_df["read_len"].median()
            mean = l1_df["read_len"].mean()
            print(f"  {cond_name}: n={len(l1_df)}, median={med:.0f}bp, mean={mean:.0f}bp")
            for age in ["young", "ancient"]:
                sub = l1_df[l1_df["age"] == age]
                if len(sub) > 0:
                    print(f"    {age}: n={len(sub)}, median={sub['read_len'].median():.0f}bp")

    # --- 7. KD verification: check PUS7/DKC1 gene reads ---
    print("\n--- 7. KD verification (PUS7/DKC1 gene expression) ---")
    _verify_kd(sample_data)

    # --- 8. Save results ---
    print("\n--- 8. Saving results ---")
    comp_df = pd.DataFrame(comparisons)
    comp_out = os.path.join(OUT_DIR, "pus_kd_ont_l1_comparisons.tsv")
    comp_df.to_csv(comp_out, sep="\t", index=False)
    print(f"  Saved: {comp_out}")

    # Per-sample summary
    summary_rows = []
    for sample_name, d in sample_data.items():
        summary_rows.append({
            "sample": sample_name,
            "condition": d["condition"],
            "rep": d["rep"],
            "total_mapped": d["total_mapped"],
            "n_l1": d["n_l1"],
            "l1_rpm": d["l1_rpm"],
            "n_young": d["n_young"],
            "n_ancient": d["n_ancient"],
            "young_frac": d["n_young"] / d["n_l1"] if d["n_l1"] > 0 else 0,
            "median_rdlen": d["l1_df"]["read_len"].median() if len(d["l1_df"]) > 0 else 0,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["condition", "rep"])
    summary_out = os.path.join(OUT_DIR, "pus_kd_ont_sample_summary.tsv")
    summary_df.to_csv(summary_out, sep="\t", index=False)
    print(f"  Saved: {summary_out}")

    print("\n=== Analysis complete ===")


def _verify_kd(sample_data):
    """Check PUS7/DKC1 gene expression by counting reads mapping to those genes."""
    # Gene coordinates (hg38, approximate from GENCODE)
    genes = {
        "PUS7":  ("chr7", 105_099_000, 105_160_000),
        "DKC1":  ("chrX", 154_762_000, 154_778_000),
        "ACTB":  ("chr7", 5_527_000, 5_530_000),  # housekeeping control
    }

    results = defaultdict(dict)
    for sample_name, meta in sample_data.items():
        bam_path = os.path.join(DATA_DIR, f"{sample_name}.sorted.bam")
        for gene_name, (chrom, start, end) in genes.items():
            region = f"{chrom}:{start}-{end}"
            result = subprocess.run(
                [SAMTOOLS, "view", "-c", "-F", "4", bam_path, region],
                capture_output=True, text=True
            )
            results[sample_name][gene_name] = int(result.stdout.strip())

    # Print table
    print(f"  {'Sample':<16} {'PUS7':>6} {'DKC1':>6} {'ACTB':>6}")
    for sample_name in sorted(results.keys()):
        r = results[sample_name]
        print(f"  {sample_name:<16} {r['PUS7']:>6} {r['DKC1']:>6} {r['ACTB']:>6}")

    # Per-condition average
    print()
    for cond in ["shGFP", "shPUS7", "shDKC1"]:
        cond_samples = [k for k in results if sample_data[k]["condition"] == cond]
        for gene in ["PUS7", "DKC1", "ACTB"]:
            counts = [results[s][gene] for s in cond_samples]
            total_mapped = sum(sample_data[s]["total_mapped"] for s in cond_samples)
            total_gene = sum(counts)
            rpm = total_gene / total_mapped * 1e6 if total_mapped > 0 else 0
            if gene in ["PUS7", "DKC1"]:
                ctrl_samples = [k for k in results if sample_data[k]["condition"] == "shGFP"]
                ctrl_total = sum(results[s][gene] for s in ctrl_samples)
                ctrl_mapped = sum(sample_data[s]["total_mapped"] for s in ctrl_samples)
                ctrl_rpm = ctrl_total / ctrl_mapped * 1e6 if ctrl_mapped > 0 else 0
                if cond != "shGFP" and ctrl_rpm > 0:
                    fc = rpm / ctrl_rpm
                    print(f"  {cond} {gene}: {total_gene} reads ({rpm:.1f} RPM), "
                          f"FC vs shGFP = {fc:.2f}")
                else:
                    print(f"  {cond} {gene}: {total_gene} reads ({rpm:.1f} RPM)")


if __name__ == "__main__":
    main()
