#!/usr/bin/env python3
"""
Analyze L1 expression in TRUB1/PUS7 KD SH-SY5Y ONT DRS (PRJNA1092333).

Fanari et al., Cell Systems 2025: siTRUB1, siPUS7, siSCR control in SH-SY5Y.
Applies matched L1 filter (same as PRJNA1220613 analysis) then compares
KD vs control L1 expression with Fisher exact test.

Cross-validates with:
  - Our SH-SY5Y DRS data (from project pipeline)
  - PRJNA1220613 BE(2)-C DKC1/PUS7 KD results
"""

import os
import re
import sys
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

# --- Configuration ---
DATA_DIR = "/vault/external-datasets/2026/PRJNA1092333_TRUB1_PUS7_KD_SHSY5Y_RNA002"
PROJECT_DIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
OUT_DIR = os.path.join(PROJECT_DIR, "analysis/01_exploration/topic_05_cellline/pus_enzyme_analysis")

L1_TE_GTF = os.path.join(PROJECT_DIR, "reference/hg38_rmsk_TE.gtf")
EXON_GTF = os.path.join(PROJECT_DIR, "reference/Human.gtf")
EXON_OVERLAP_MIN = 100

SAMTOOLS = subprocess.check_output(
    "which samtools", shell=True
).decode().strip()
BEDTOOLS = "/blaze/junsoopablo/conda/envs/research/bin/bedtools"

# Sample definitions
SAMPLES = {
    "SCR_control":    {"condition": "SCR",       "rep": 1},
    "PUS7KD":         {"condition": "PUS7KD",    "rep": 1},
    "TRUB1KD":        {"condition": "TRUB1KD",   "rep": 1},
    "Untreated_rep1": {"condition": "Untreated", "rep": 1},
    "Untreated_rep2": {"condition": "Untreated", "rep": 2},
    "Untreated_rep3": {"condition": "Untreated", "rep": 3},
}

YOUNG_L1 = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

# Our SH-SY5Y data for cross-study comparison
OUR_SHSY5Y_GROUPS = ["SHSY5Y_R1", "SHSY5Y_R2"]


# ====================================================================
# L1 filter functions (same as apply_matched_filter.py)
# ====================================================================

def alignment_score(fields):
    """Extract alignment score from SAM fields."""
    for tag in fields[11:]:
        if tag.startswith("AS:i:"):
            try:
                return int(tag.split(":")[-1])
            except ValueError:
                return 0
    try:
        return int(fields[4])
    except ValueError:
        return 0


def cigar_blocks(pos, cigar):
    """Compute reference blocks from CIGAR (splitting on N)."""
    blocks = []
    ref_pos = pos - 1
    block_start = ref_pos
    block_len = 0
    for length_str, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        length = int(length_str)
        if op in ("M", "=", "X", "D"):
            ref_pos += length
            block_len += length
        elif op == "N":
            if block_len > 0:
                blocks.append((block_start, block_start + block_len))
            ref_pos += length
            block_start = ref_pos
            block_len = 0
    if block_len > 0:
        blocks.append((block_start, block_start + block_len))
    return blocks


def build_l1_te_bed(gtf_path, out_bed):
    """Parse L1 entries from TE GTF → 6-col BED."""
    with open(gtf_path) as gtf, open(out_bed, "w") as out:
        for line in gtf:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            attrs = fields[8]
            if 'family_id "L1"' not in attrs:
                continue
            gene_id = ""
            transcript_id = ""
            m = re.search(r'gene_id "([^"]+)"', attrs)
            if m:
                gene_id = m.group(1)
            m = re.search(r'transcript_id "([^"]+)"', attrs)
            if m:
                transcript_id = m.group(1)
            chrom = fields[0]
            start = max(int(fields[3]) - 1, 0)
            end = int(fields[4])
            strand = fields[6]
            out.write(f"{chrom}\t{start}\t{end}\t{transcript_id}\t{gene_id}\t{strand}\n")


def build_exon_bed(gtf_path, out_bed):
    """Parse exon entries from gene GTF → 3-col BED."""
    with open(gtf_path) as gtf, open(out_bed, "w") as out:
        for line in gtf:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != "exon":
                continue
            chrom = fields[0]
            start = max(int(fields[3]) - 1, 0)
            end = int(fields[4])
            out.write(f"{chrom}\t{start}\t{end}\n")


def filter_sample_from_bam(sample_name, bam_path, l1_te_bed, exon_bed, tmp_dir):
    """
    Apply matched L1 filter directly from BAM (no prior L1 ID list).

    For this dataset we don't have a pre-computed L1 ID list.
    Instead we:
    1. Parse all reads from BAM
    2. Apply CIGAR/supplementary filters
    3. Keep best AS per read
    4. Write reads BED
    5. Intersect with L1 TE (>= 10% of read aligned length)
    6. Exclude exon overlap >= 100bp
    7. Strand match
    """
    print(f"\n  === Filtering {sample_name} ===")
    print(f"    BAM: {bam_path}")

    # --- Stage 1: Parse BAM, apply CIGAR filter, keep best AS ---
    best_align = {}
    n_total = 0
    n_unmapped = 0
    n_supp = 0
    n_spliced = 0

    proc = subprocess.Popen(
        [SAMTOOLS, "view", bam_path],
        stdout=subprocess.PIPE, text=True
    )
    for line in proc.stdout:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 11:
            continue
        n_total += 1
        qname = fields[0]
        flag = int(fields[1])
        rname = fields[2]
        pos = int(fields[3])
        cigar = fields[5]
        seq = fields[9]

        if flag & 0x4:
            n_unmapped += 1
            continue
        if flag & 0x800:
            n_supp += 1
            continue
        if "N" in cigar:
            n_spliced += 1
            continue
        if rname == "*":
            continue

        blocks = cigar_blocks(pos, cigar)
        if not blocks:
            continue

        start = min(b[0] for b in blocks)
        end = max(b[1] for b in blocks)
        read_len = len(seq) if seq != "*" else 0
        read_strand = "-" if flag & 0x10 else "+"
        score = alignment_score(fields)

        current = best_align.get(qname)
        if current is None or score > current["score"]:
            best_align[qname] = {
                "chrom": rname,
                "start": start,
                "end": end,
                "read_len": read_len,
                "score": score,
                "blocks": blocks,
                "strand": read_strand,
            }

    proc.stdout.close()
    proc.wait()

    print(f"    Total SAM records: {n_total:,}")
    print(f"    Unmapped: {n_unmapped:,}, Supplementary: {n_supp:,}, Spliced: {n_spliced:,}")
    print(f"    After CIGAR/best-AS: {len(best_align):,} unique reads")

    # --- Stage 2: Write reads BED ---
    reads_bed = os.path.join(tmp_dir, f"{sample_name}_reads.bed")
    with open(reads_bed, "w") as bed_out:
        for qname, entry in best_align.items():
            for block_start, block_end in entry["blocks"]:
                bed_out.write(f"{entry['chrom']}\t{block_start}\t{block_end}\t{qname}\n")

    # --- Stage 3: Exon overlap filter ---
    exon_overlaps = os.path.join(tmp_dir, f"{sample_name}_exon_ov.tsv")
    subprocess.run(
        f"{BEDTOOLS} intersect -a {reads_bed} -b {exon_bed} -wo > {exon_overlaps}",
        shell=True, check=True
    )

    exon_overlap = {}
    with open(exon_overlaps) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue
            read_id = cols[3]
            overlap_len = int(cols[7])
            if overlap_len > exon_overlap.get(read_id, 0):
                exon_overlap[read_id] = overlap_len

    exclude_exon = {rid for rid, olen in exon_overlap.items() if olen >= EXON_OVERLAP_MIN}
    print(f"    Exon overlap >= {EXON_OVERLAP_MIN}bp: {len(exclude_exon):,} reads excluded")

    # --- Stage 4: L1 TE overlap with strand match ---
    te_overlaps = os.path.join(tmp_dir, f"{sample_name}_te_ov.tsv")
    subprocess.run(
        f"{BEDTOOLS} intersect -a {reads_bed} -b {l1_te_bed} -wo > {te_overlaps}",
        shell=True, check=True
    )

    te_best = {}
    with open(te_overlaps) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 11:
                continue
            read_id = cols[3]
            te_transcript_id = cols[7]
            te_gene_id = cols[8]
            te_strand = cols[9]
            overlap_len = int(cols[10])

            if read_id not in best_align:
                continue
            if read_id in exclude_exon:
                continue

            read_entry = best_align[read_id]
            # Strand match
            if read_entry["strand"] != te_strand:
                continue

            # Check >= 10% of aligned span
            aligned_span = read_entry["end"] - read_entry["start"]
            if aligned_span <= 0:
                continue

            current = te_best.get(read_id)
            if current is None or overlap_len > current["overlap_len"]:
                te_best[read_id] = {
                    "overlap_len": overlap_len,
                    "gene_id": te_gene_id,
                    "transcript_id": te_transcript_id,
                    "strand": te_strand,
                    "aligned_span": aligned_span,
                }

    # Apply 10% overlap threshold
    te_pass = {}
    for read_id, entry in te_best.items():
        if entry["overlap_len"] >= 0.10 * entry["aligned_span"]:
            te_pass[read_id] = entry

    n_final = len(te_pass)
    print(f"    L1 reads (matched filter): {n_final:,}")

    # Clean up temp files
    for f in [reads_bed, exon_overlaps, te_overlaps]:
        try:
            os.remove(f)
        except OSError:
            pass

    # Build DataFrame
    rows = []
    for read_id, entry in te_pass.items():
        read_entry = best_align[read_id]
        rows.append({
            "read_id": read_id,
            "read_len": read_entry["read_len"],
            "subfamily": entry["gene_id"],
            "locus_id": entry["transcript_id"],
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["read_id", "read_len", "subfamily", "locus_id"]
    )


def get_total_mapped(bam_path):
    """Count total mapped reads (excluding supplementary and secondary)."""
    result = subprocess.run(
        [SAMTOOLS, "view", "-c", "-F", "0x904", bam_path],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def fisher_exact_test(a, b, c, d):
    table = np.array([[a, b], [c, d]])
    return stats.fisher_exact(table)


def load_our_shsy5y_l1():
    """Load our SH-SY5Y L1 data for cross-study comparison."""
    rows = []
    for group in OUR_SHSY5Y_GROUPS:
        summary_file = os.path.join(
            PROJECT_DIR, f"results_group/{group}/g_summary/{group}_L1_summary.tsv"
        )
        if not os.path.exists(summary_file):
            print(f"  Warning: {summary_file} not found")
            continue
        df = pd.read_csv(summary_file, sep="\t")
        n_l1 = len(df)
        n_young = df["subfamily"].isin(YOUNG_L1).sum() if "subfamily" in df.columns else 0

        # Get total mapped
        bam_path = os.path.join(
            PROJECT_DIR, f"results_group/{group}/h_mafia/{group}.mAFiA.reads.bam"
        )
        if os.path.exists(bam_path):
            total = get_total_mapped(bam_path)
        else:
            bam_path = os.path.join(
                PROJECT_DIR, f"results_group/{group}/b_align/{group}.sorted.bam"
            )
            total = get_total_mapped(bam_path) if os.path.exists(bam_path) else 0

        rows.append({
            "group": group,
            "total_mapped": total,
            "n_l1": n_l1,
            "n_young": n_young,
            "n_ancient": n_l1 - n_young,
            "rpm": n_l1 / total * 1e6 if total > 0 else 0,
        })
    return rows


def verify_kd(bam_path, sample_name):
    """Check KD by counting reads mapping to target genes."""
    targets = {
        "PUS7": ("chr7", 105095000, 105130000),
        "TRUB1": ("chr10", 64750000, 64790000),
    }
    counts = {}
    for gene, (chrom, start, end) in targets.items():
        result = subprocess.run(
            [SAMTOOLS, "view", "-c", "-F", "0x904", bam_path,
             f"{chrom}:{start}-{end}"],
            capture_output=True, text=True
        )
        counts[gene] = int(result.stdout.strip())
    return counts


def main():
    print("=" * 70)
    print("PRJNA1092333: TRUB1/PUS7 KD SH-SY5Y ONT DRS — L1 Analysis")
    print("=" * 70)

    # --- Build reference BEDs ---
    tmp_dir = os.path.join(OUT_DIR, "matched_filter_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    l1_te_bed = os.path.join(tmp_dir, "L1_TE_6col.bed")
    exon_bed = os.path.join(tmp_dir, "exons.bed")

    print("\n--- Building reference BEDs ---")
    if not os.path.exists(l1_te_bed):
        print("  Building L1 TE BED...")
        build_l1_te_bed(L1_TE_GTF, l1_te_bed)
        n_l1 = sum(1 for _ in open(l1_te_bed))
        print(f"  L1 TE BED: {n_l1:,} entries")
    else:
        print(f"  L1 TE BED: already exists")

    if not os.path.exists(exon_bed):
        print("  Building exon BED...")
        build_exon_bed(EXON_GTF, exon_bed)
        n_exon = sum(1 for _ in open(exon_bed))
        print(f"  Exon BED: {n_exon:,} entries")
    else:
        print(f"  Exon BED: already exists")

    # --- Check which samples are available ---
    available = {}
    for sample_name, meta in SAMPLES.items():
        bam_path = os.path.join(DATA_DIR, f"{sample_name}.sorted.bam")
        if os.path.exists(bam_path):
            available[sample_name] = meta
        else:
            print(f"  Skipping {sample_name}: BAM not found")

    if not available:
        print("\nERROR: No BAM files found. Run download_align_trub1_pus7.sh first.")
        sys.exit(1)

    # --- Verify KD ---
    print("\n--- KD Verification ---")
    for sample_name in available:
        bam_path = os.path.join(DATA_DIR, f"{sample_name}.sorted.bam")
        counts = verify_kd(bam_path, sample_name)
        total = get_total_mapped(bam_path)
        print(f"  {sample_name}: total={total:,}, PUS7={counts['PUS7']:,}, TRUB1={counts['TRUB1']:,}")

    # --- Filter each sample ---
    print("\n--- Applying matched L1 filter ---")
    sample_data = {}
    for sample_name, meta in sorted(available.items()):
        bam_path = os.path.join(DATA_DIR, f"{sample_name}.sorted.bam")
        total_mapped = get_total_mapped(bam_path)

        # Check for cached filtered data
        cached_file = os.path.join(DATA_DIR, f"{sample_name}_L1_reads_info_matched.tsv")
        if os.path.exists(cached_file) and os.path.getsize(cached_file) > 0:
            print(f"\n  === {sample_name} (cached) ===")
            filtered_df = pd.read_csv(cached_file, sep="\t", header=None,
                                      names=["read_id", "read_len", "subfamily", "locus_id"])
        else:
            filtered_df = filter_sample_from_bam(
                sample_name, bam_path, l1_te_bed, exon_bed, tmp_dir
            )
            # Save filtered info
            filtered_df.to_csv(cached_file, sep="\t", index=False, header=False)

        filtered_df["age"] = filtered_df["subfamily"].apply(
            lambda sf: "young" if sf in YOUNG_L1 else "ancient"
        )

        n_filt = len(filtered_df)
        rpm = n_filt / total_mapped * 1e6 if total_mapped > 0 else 0

        sample_data[sample_name] = {
            "condition": meta["condition"],
            "rep": meta["rep"],
            "total_mapped": total_mapped,
            "n_l1": n_filt,
            "rpm": rpm,
            "l1_df": filtered_df,
            "n_young": (filtered_df["age"] == "young").sum(),
            "n_ancient": (filtered_df["age"] == "ancient").sum(),
        }

        print(f"    {sample_name}: {total_mapped:,} mapped, {n_filt:,} L1 ({rpm:.1f} RPM), "
              f"young={sample_data[sample_name]['n_young']}, "
              f"ancient={sample_data[sample_name]['n_ancient']}")

    # --- Per-condition aggregation ---
    print("\n" + "=" * 70)
    print("Per-Condition Summary")
    print("=" * 70)

    conditions = {}
    for cond in ["SCR", "PUS7KD", "TRUB1KD", "Untreated"]:
        reps = {k: v for k, v in sample_data.items() if v["condition"] == cond}
        if not reps:
            continue
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
        }
        young_pct = n_young / n_l1 * 100 if n_l1 > 0 else 0
        print(f"  {cond}: {total_mapped:,} total, {n_l1:,} L1 ({rpm:.1f} RPM), "
              f"young={n_young} ({young_pct:.1f}%), ancient={n_ancient}")

    # --- KD vs SCR comparison ---
    print("\n" + "=" * 70)
    print("KD vs SCR_control Comparison (Matched Filter)")
    print("=" * 70)

    if "SCR" not in conditions:
        print("  ERROR: SCR control not available")
        sys.exit(1)

    ctrl = conditions["SCR"]
    comparisons = []

    for kd_name in ["PUS7KD", "TRUB1KD"]:
        if kd_name not in conditions:
            continue
        kd = conditions[kd_name]
        print(f"\n  === {kd_name} vs SCR ===")

        for category, n_kd, n_ctrl_cat in [
            ("total_L1", kd["n_l1"], ctrl["n_l1"]),
            ("young_L1", kd["n_young"], ctrl["n_young"]),
            ("ancient_L1", kd["n_ancient"], ctrl["n_ancient"]),
        ]:
            a = n_kd
            b = kd["total_mapped"] - n_kd
            c = n_ctrl_cat
            d = ctrl["total_mapped"] - n_ctrl_cat
            OR, p = fisher_exact_test(a, b, c, d)
            rpm_kd = a / kd["total_mapped"] * 1e6
            rpm_ctrl = c / ctrl["total_mapped"] * 1e6
            fc = rpm_kd / rpm_ctrl if rpm_ctrl > 0 else np.inf

            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"  {category}: KD={a:,} ({rpm_kd:.1f} RPM), "
                  f"Ctrl={c:,} ({rpm_ctrl:.1f} RPM)")
            print(f"    FC={fc:.3f}, OR={OR:.3f}, p={p:.2e} {sig}")

            comparisons.append({
                "comparison": f"{kd_name}_vs_SCR",
                "category": category,
                "kd_count": a, "ctrl_count": c,
                "kd_total": kd["total_mapped"], "ctrl_total": ctrl["total_mapped"],
                "kd_rpm": rpm_kd, "ctrl_rpm": rpm_ctrl,
                "fold_change": fc, "odds_ratio": OR, "pvalue": p,
            })

    # --- Untreated vs SCR comparison (check siSCR effect) ---
    if "Untreated" in conditions:
        print("\n  === Untreated vs SCR (siSCR effect check) ===")
        ut = conditions["Untreated"]
        for category, n_ut, n_ctrl_cat in [
            ("total_L1", ut["n_l1"], ctrl["n_l1"]),
            ("young_L1", ut["n_young"], ctrl["n_young"]),
            ("ancient_L1", ut["n_ancient"], ctrl["n_ancient"]),
        ]:
            a = n_ut
            b = ut["total_mapped"] - n_ut
            c = n_ctrl_cat
            d = ctrl["total_mapped"] - n_ctrl_cat
            OR, p = fisher_exact_test(a, b, c, d)
            rpm_ut = a / ut["total_mapped"] * 1e6
            rpm_ctrl = c / ctrl["total_mapped"] * 1e6
            fc = rpm_ut / rpm_ctrl if rpm_ctrl > 0 else np.inf
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"  {category}: Untreated={a:,} ({rpm_ut:.1f} RPM), "
                  f"SCR={c:,} ({rpm_ctrl:.1f} RPM)")
            print(f"    FC={fc:.3f}, p={p:.2e} {sig}")

            comparisons.append({
                "comparison": "Untreated_vs_SCR",
                "category": category,
                "kd_count": a, "ctrl_count": c,
                "kd_total": ut["total_mapped"], "ctrl_total": ctrl["total_mapped"],
                "kd_rpm": rpm_ut, "ctrl_rpm": rpm_ctrl,
                "fold_change": fc, "odds_ratio": OR, "pvalue": p,
            })

    # --- Per-subfamily analysis ---
    print("\n" + "=" * 70)
    print("Per-Subfamily L1 Changes")
    print("=" * 70)

    for kd_name in ["PUS7KD", "TRUB1KD"]:
        if kd_name not in conditions:
            continue
        kd_df = conditions[kd_name]["l1_df"]
        ctrl_df = ctrl["l1_df"]
        kd_total = conditions[kd_name]["total_mapped"]
        ctrl_total = ctrl["total_mapped"]

        kd_sf = kd_df["subfamily"].value_counts()
        ctrl_sf = ctrl_df["subfamily"].value_counts()

        all_sf = set(kd_sf.index) | set(ctrl_sf.index)
        sf_results = []
        for sf in sorted(all_sf):
            n_kd = kd_sf.get(sf, 0)
            n_ctrl = ctrl_sf.get(sf, 0)
            if n_kd + n_ctrl < 5:
                continue
            rpm_kd_val = n_kd / kd_total * 1e6
            rpm_ctrl_val = n_ctrl / ctrl_total * 1e6
            fc = rpm_kd_val / rpm_ctrl_val if rpm_ctrl_val > 0 else np.inf
            OR, p = fisher_exact_test(n_kd, kd_total - n_kd, n_ctrl, ctrl_total - n_ctrl)
            age = "young" if sf in YOUNG_L1 else "ancient"
            sf_results.append({
                "subfamily": sf, "age": age,
                "kd_count": n_kd, "ctrl_count": n_ctrl,
                "kd_rpm": rpm_kd_val, "ctrl_rpm": rpm_ctrl_val,
                "fold_change": fc, "pvalue": p,
            })

        sf_df = pd.DataFrame(sf_results).sort_values("pvalue")
        print(f"\n  Top 15 subfamilies: {kd_name} vs SCR")
        print(f"  {'Subfamily':<18} {'Age':<8} {'KD':>6} {'Ctrl':>6} {'FC':>7} {'p':>10}")
        for _, row in sf_df.head(15).iterrows():
            sig = "***" if row["pvalue"] < 0.001 else ("**" if row["pvalue"] < 0.01 else (
                "*" if row["pvalue"] < 0.05 else ""))
            print(f"  {row['subfamily']:<18} {row['age']:<8} {row['kd_count']:>6} "
                  f"{row['ctrl_count']:>6} {row['fold_change']:>7.2f} {row['pvalue']:>10.2e} {sig}")

    # --- Cross-study comparison ---
    print("\n" + "=" * 70)
    print("Cross-Study Comparison")
    print("=" * 70)

    # Our SH-SY5Y data
    print("\n  Our SH-SY5Y DRS (project pipeline):")
    our_data = load_our_shsy5y_l1()
    for d in our_data:
        print(f"    {d['group']}: {d['total_mapped']:,} mapped, {d['n_l1']:,} L1 ({d['rpm']:.1f} RPM), "
              f"young={d['n_young']}")

    # PRJNA1220613 BE(2)-C results (from memory)
    print("\n  PRJNA1220613 BE(2)-C (matched filter, from previous analysis):")
    print("    DKC1 KD → L1 1.88x (p=2.0e-173), young 3.87x")
    print("    PUS7 KD → L1 1.22x (p=1.6e-10), young 3.56x")

    # This study
    print("\n  PRJNA1092333 SH-SY5Y (this analysis):")
    for comp in comparisons:
        if comp["comparison"].endswith("_vs_SCR") and comp["category"] == "total_L1":
            print(f"    {comp['comparison']}: FC={comp['fold_change']:.3f}, p={comp['pvalue']:.2e}")

    # --- Read length distribution ---
    print("\n" + "=" * 70)
    print("Read Length Distribution by Condition")
    print("=" * 70)

    for cond_name, cond in conditions.items():
        l1_df = cond["l1_df"]
        if len(l1_df) == 0:
            continue
        young_len = l1_df[l1_df["age"] == "young"]["read_len"]
        ancient_len = l1_df[l1_df["age"] == "ancient"]["read_len"]
        print(f"  {cond_name}: median={l1_df['read_len'].median():.0f}bp, "
              f"young={young_len.median():.0f}bp (n={len(young_len)}), "
              f"ancient={ancient_len.median():.0f}bp (n={len(ancient_len)})")

    # --- Save results ---
    print("\n" + "=" * 70)
    print("Saving results")
    print("=" * 70)

    # Comparisons TSV
    comp_df = pd.DataFrame(comparisons)
    comp_out = os.path.join(OUT_DIR, "trub1_pus7_kd_shsy5y_comparisons.tsv")
    comp_df.to_csv(comp_out, sep="\t", index=False)
    print(f"  Saved: {comp_out}")

    # Per-sample summary TSV
    summary_rows = []
    for sn, d in sorted(sample_data.items()):
        summary_rows.append({
            "sample": sn,
            "condition": d["condition"],
            "rep": d["rep"],
            "total_mapped": d["total_mapped"],
            "n_l1_matched": d["n_l1"],
            "rpm_matched": d["rpm"],
            "n_young": d["n_young"],
            "n_ancient": d["n_ancient"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_out = os.path.join(OUT_DIR, "trub1_pus7_kd_shsy5y_sample_summary.tsv")
    summary_df.to_csv(summary_out, sep="\t", index=False)
    print(f"  Saved: {summary_out}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
