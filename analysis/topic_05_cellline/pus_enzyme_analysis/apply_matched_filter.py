#!/usr/bin/env python3
"""
Apply MAFIA-pipeline-matched L1 filters to PUS7/DKC1 KD ONT DRS data.

The original download_and_align_ont.sh only applied:
  - Exclude unmapped (-F 4)
  - L1 overlap >= 10% of read length

The main MAFIA pipeline (07_L1Base2_match.py) additionally applies:
  1. Exclude supplementary (0x800)
  2. Exclude spliced alignments ('N' in CIGAR)
  3. Best alignment score per read (AS tag)
  4. Exon overlap < 100bp (exclude host gene contamination)
  5. Strand match (read strand == TE strand)

This script applies those same filters to the KD BAM files
and re-runs the L1 expression comparison.
"""

import os
import re
import sys
import subprocess
import tempfile
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

# --- Configuration ---
DATA_DIR = "/vault/external-datasets/2026/PRJNA1220613_PUS7_DKC1_KD_RNA002"
PROJECT_DIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
OUT_DIR = os.path.join(PROJECT_DIR, "analysis/01_exploration/topic_05_cellline/pus_enzyme_analysis")

L1_TE_GTF = os.path.join(PROJECT_DIR, "reference/hg38_rmsk_TE.gtf")
EXON_GTF = os.path.join(PROJECT_DIR, "reference/Human.gtf")
EXON_OVERLAP_MIN = 100

SAMTOOLS = subprocess.check_output(
    "conda run -n bioinfo3 which samtools", shell=True
).decode().strip()
BEDTOOLS = "/blaze/junsoopablo/conda/envs/research/bin/bedtools"

SAMPLES = {
    "shPUS7_rep1": {"condition": "shPUS7", "rep": 1},
    "shPUS7_rep2": {"condition": "shPUS7", "rep": 2},
    "shGFP_rep1":  {"condition": "shGFP",  "rep": 1},
    "shGFP_rep2":  {"condition": "shGFP",  "rep": 2},
    "shDKC1_rep1": {"condition": "shDKC1", "rep": 1},
    "shDKC1_rep2": {"condition": "shDKC1", "rep": 2},
}

YOUNG_L1 = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}


def alignment_score(fields):
    """Extract alignment score from SAM fields (AS:i tag, fallback to MAPQ)."""
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
    ref_pos = pos - 1  # 0-based
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
    """Parse L1 entries from TE GTF → 6-col BED (chr, start, end, transcript_id, gene_id, strand)."""
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


def filter_sample(sample_name, l1_te_bed, exon_bed, tmp_dir):
    """Apply matched filters to one sample. Returns filtered L1 read info DataFrame."""
    bam_path = os.path.join(DATA_DIR, f"{sample_name}.sorted.bam")

    # Load original L1 read IDs (from 10% overlap filter)
    orig_ids_file = os.path.join(DATA_DIR, f"{sample_name}_L1_readIDs.txt")
    l1_id_set = set()
    with open(orig_ids_file) as f:
        for line in f:
            rid = line.strip()
            if rid:
                l1_id_set.add(rid)

    print(f"  {sample_name}: {len(l1_id_set)} reads from original filter")

    # --- Stage 1: Parse BAM, select best alignment per read ---
    # Apply: exclude unmapped (0x4), supplementary (0x800), spliced ('N' in CIGAR)
    best_align = {}
    proc = subprocess.Popen(
        [SAMTOOLS, "view", bam_path],
        stdout=subprocess.PIPE, text=True
    )
    for line in proc.stdout:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 11:
            continue
        qname = fields[0]
        if qname not in l1_id_set:
            continue
        flag = int(fields[1])
        rname = fields[2]
        pos = int(fields[3])
        cigar = fields[5]
        seq = fields[9]

        # Exclude unmapped and supplementary
        if flag & 0x4 or flag & 0x800:
            continue
        # Exclude spliced alignments
        if "N" in cigar:
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

    n_after_cigar = len(best_align)
    n_lost_cigar = len(l1_id_set) - n_after_cigar
    print(f"    After CIGAR/best-AS filter: {n_after_cigar} ({n_lost_cigar} removed)")

    # --- Stage 2: Write reads BED for bedtools intersect ---
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
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            read_id = fields[3]
            overlap_len = int(fields[7])
            if overlap_len > exon_overlap.get(read_id, 0):
                exon_overlap[read_id] = overlap_len

    exclude_exon = {rid for rid, olen in exon_overlap.items() if olen >= EXON_OVERLAP_MIN}
    print(f"    Exon overlap >= {EXON_OVERLAP_MIN}bp: {len(exclude_exon)} reads excluded")

    # --- Stage 4: L1 TE overlap with strand match ---
    te_overlaps = os.path.join(tmp_dir, f"{sample_name}_te_ov.tsv")
    subprocess.run(
        f"{BEDTOOLS} intersect -a {reads_bed} -b {l1_te_bed} -wo > {te_overlaps}",
        shell=True, check=True
    )

    te_best = {}
    with open(te_overlaps) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue
            read_id = fields[3]
            # L1 BED columns: chr(4), start(5), end(6), transcript_id(7), gene_id(8), strand(9)
            te_transcript_id = fields[7]
            te_gene_id = fields[8]
            te_strand = fields[9]
            overlap_len = int(fields[10])

            if read_id not in best_align:
                continue
            if read_id in exclude_exon:
                continue

            # Strand match filter
            read_entry = best_align[read_id]
            if read_entry["strand"] != te_strand:
                continue

            current = te_best.get(read_id)
            if current is None or overlap_len > current["overlap_len"]:
                te_best[read_id] = {
                    "overlap_len": overlap_len,
                    "gene_id": te_gene_id,          # = subfamily (e.g., L1M2)
                    "transcript_id": te_transcript_id,  # = locus_id (e.g., L1M2_dup229)
                    "strand": te_strand,
                }

    n_final = len(te_best)
    n_lost_exon_strand = n_after_cigar - len(exclude_exon) - n_final
    print(f"    After exon+strand filter: {n_final} reads pass")
    print(f"    Total removed: {len(l1_id_set) - n_final} "
          f"({(len(l1_id_set) - n_final) / len(l1_id_set) * 100:.1f}%)")

    # Clean up temp files
    for f in [reads_bed, exon_overlaps, te_overlaps]:
        try:
            os.remove(f)
        except OSError:
            pass

    # Build filtered DataFrame
    rows = []
    for read_id, entry in te_best.items():
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
    """Count total mapped reads in BAM."""
    result = subprocess.run(
        [SAMTOOLS, "view", "-c", "-F", "4", bam_path],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def fisher_exact_test(a, b, c, d):
    table = np.array([[a, b], [c, d]])
    return stats.fisher_exact(table)


def main():
    print("=" * 70)
    print("MAFIA-Matched L1 Filter for PUS7/DKC1 KD ONT DRS")
    print("=" * 70)

    # --- Build reference BEDs ---
    tmp_dir = os.path.join(OUT_DIR, "matched_filter_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    l1_te_bed = os.path.join(tmp_dir, "L1_TE_6col.bed")
    exon_bed = os.path.join(tmp_dir, "exons.bed")

    print("\n--- Building reference BEDs ---")
    if not os.path.exists(l1_te_bed):
        print("  Building L1 TE BED (6-col with strand)...")
        build_l1_te_bed(L1_TE_GTF, l1_te_bed)
        n_l1 = sum(1 for _ in open(l1_te_bed))
        print(f"  L1 TE BED: {n_l1} entries")
    else:
        print(f"  L1 TE BED: already exists")

    if not os.path.exists(exon_bed):
        print("  Building exon BED...")
        build_exon_bed(EXON_GTF, exon_bed)
        n_exon = sum(1 for _ in open(exon_bed))
        print(f"  Exon BED: {n_exon} entries")
    else:
        print(f"  Exon BED: already exists")

    # --- Filter each sample ---
    print("\n--- Filtering samples ---")
    sample_data = {}
    for sample_name, meta in sorted(SAMPLES.items()):
        print(f"\n  === {sample_name} ===")
        bam_path = os.path.join(DATA_DIR, f"{sample_name}.sorted.bam")
        total_mapped = get_total_mapped(bam_path)

        filtered_df = filter_sample(sample_name, l1_te_bed, exon_bed, tmp_dir)
        filtered_df["age"] = filtered_df["subfamily"].apply(
            lambda sf: "young" if sf in YOUNG_L1 else "ancient"
        )

        # Load original for comparison
        orig_info = os.path.join(DATA_DIR, f"{sample_name}_L1_reads_info.tsv")
        orig_df = pd.read_csv(orig_info, sep="\t", header=None,
                              names=["read_id", "read_len", "subfamily", "locus_id"])

        n_orig = len(orig_df)
        n_filt = len(filtered_df)
        rpm_orig = n_orig / total_mapped * 1e6
        rpm_filt = n_filt / total_mapped * 1e6

        sample_data[sample_name] = {
            "condition": meta["condition"],
            "rep": meta["rep"],
            "total_mapped": total_mapped,
            "n_l1_orig": n_orig,
            "n_l1_filt": n_filt,
            "rpm_orig": rpm_orig,
            "rpm_filt": rpm_filt,
            "l1_df": filtered_df,
            "n_young": (filtered_df["age"] == "young").sum(),
            "n_ancient": (filtered_df["age"] == "ancient").sum(),
        }

        print(f"    Original: {n_orig} reads ({rpm_orig:.1f} RPM)")
        print(f"    Filtered: {n_filt} reads ({rpm_filt:.1f} RPM)")
        print(f"    Retention: {n_filt/n_orig*100:.1f}%")

        # Save filtered info
        filt_out = os.path.join(DATA_DIR, f"{sample_name}_L1_reads_info_matched.tsv")
        filtered_df.to_csv(filt_out, sep="\t", index=False, header=False)

    # --- Comparison table ---
    print("\n" + "=" * 70)
    print("COMPARISON: Original vs Matched Filter")
    print("=" * 70)

    print(f"\n{'Sample':<16} {'Orig':>8} {'Filt':>8} {'Ret%':>6} "
          f"{'RPM_orig':>10} {'RPM_filt':>10} {'Young':>6} {'Anc':>6}")
    for sn in sorted(sample_data.keys()):
        d = sample_data[sn]
        ret = d["n_l1_filt"] / d["n_l1_orig"] * 100 if d["n_l1_orig"] > 0 else 0
        print(f"{sn:<16} {d['n_l1_orig']:>8,} {d['n_l1_filt']:>8,} {ret:>5.1f}% "
              f"{d['rpm_orig']:>10.1f} {d['rpm_filt']:>10.1f} "
              f"{d['n_young']:>6} {d['n_ancient']:>6}")

    # --- Per-condition summary & KD comparison ---
    print("\n" + "=" * 70)
    print("KD vs Control Comparison (Matched Filter)")
    print("=" * 70)

    conditions = {}
    for cond in ["shGFP", "shPUS7", "shDKC1"]:
        reps = {k: v for k, v in sample_data.items() if v["condition"] == cond}
        total_mapped = sum(v["total_mapped"] for v in reps.values())
        n_l1 = sum(v["n_l1_filt"] for v in reps.values())
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
        print(f"\n  {cond}: {total_mapped:,} total, {n_l1:,} L1 ({rpm:.1f} RPM), "
              f"young={n_young}, ancient={n_ancient}")

    ctrl = conditions["shGFP"]

    comparisons = []
    for kd_name in ["shPUS7", "shDKC1"]:
        kd = conditions[kd_name]
        print(f"\n  === {kd_name} vs shGFP (matched filter) ===")

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

            print(f"  {category}: KD={a:,} ({rpm_kd:.1f} RPM), "
                  f"Ctrl={c:,} ({rpm_ctrl:.1f} RPM)")
            print(f"    FC={fc:.3f}, OR={OR:.3f}, p={p:.2e}")

            comparisons.append({
                "comparison": f"{kd_name}_vs_shGFP",
                "category": category,
                "filter": "matched",
                "kd_count": a, "ctrl_count": c,
                "kd_rpm": rpm_kd, "ctrl_rpm": rpm_ctrl,
                "fold_change": fc, "odds_ratio": OR, "pvalue": p,
            })

    # --- Compare with original results ---
    print("\n" + "=" * 70)
    print("Original vs Matched Filter FC Comparison")
    print("=" * 70)

    orig_comp_file = os.path.join(OUT_DIR, "pus_kd_ont_l1_comparisons.tsv")
    if os.path.exists(orig_comp_file):
        orig_comp = pd.read_csv(orig_comp_file, sep="\t")
        for _, row in orig_comp.iterrows():
            if row["category"] in ("total_L1", "young_L1", "ancient_L1"):
                matched = [c for c in comparisons
                           if c["comparison"] == row["comparison"]
                           and c["category"] == row["category"]]
                if matched:
                    m = matched[0]
                    print(f"  {row['comparison']} {row['category']}:")
                    print(f"    Original: FC={row['fold_change']:.3f}, p={row['pvalue']:.2e}")
                    print(f"    Matched:  FC={m['fold_change']:.3f}, p={m['pvalue']:.2e}")

    # --- Save results ---
    comp_df = pd.DataFrame(comparisons)
    comp_out = os.path.join(OUT_DIR, "pus_kd_ont_l1_comparisons_matched.tsv")
    comp_df.to_csv(comp_out, sep="\t", index=False)
    print(f"\n  Saved: {comp_out}")

    # Per-sample summary
    summary_rows = []
    for sn, d in sorted(sample_data.items()):
        summary_rows.append({
            "sample": sn,
            "condition": d["condition"],
            "rep": d["rep"],
            "total_mapped": d["total_mapped"],
            "n_l1_orig": d["n_l1_orig"],
            "n_l1_matched": d["n_l1_filt"],
            "retention_pct": d["n_l1_filt"] / d["n_l1_orig"] * 100 if d["n_l1_orig"] > 0 else 0,
            "rpm_orig": d["rpm_orig"],
            "rpm_matched": d["rpm_filt"],
            "n_young": d["n_young"],
            "n_ancient": d["n_ancient"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_out = os.path.join(OUT_DIR, "pus_kd_ont_sample_summary_matched.tsv")
    summary_df.to_csv(summary_out, sep="\t", index=False)
    print(f"  Saved: {summary_out}")

    # Clean up tmp dir (keep BEDs for reuse)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
