#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import tempfile


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


def cigar_ref_len(cigar):
    total = 0
    for length, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        if op in ("M", "D", "N", "=", "X"):
            total += int(length)
    return total


def cigar_blocks(pos, cigar):
    blocks = []
    ref_pos = pos - 1
    block_start = ref_pos
    block_len = 0
    for length, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        length = int(length)
        if op in ("M", "=", "X", "D"):
            ref_pos += length
            block_len += length
        elif op == "N":
            if block_len > 0:
                blocks.append((block_start, block_start + block_len))
            ref_pos += length
            block_start = ref_pos
            block_len = 0
        else:
            continue
    if block_len > 0:
        blocks.append((block_start, block_start + block_len))
    return blocks


def parse_total_reads(qc_path):
    try:
        with open(qc_path, "r") as f:
            line = f.readline().strip()
        parts = line.split("\t")
        if len(parts) >= 2:
            return int(float(parts[1]))
    except Exception:
        return 0
    return 0


def alignment_score(fields):
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


def parse_l1_te_gtf(gtf_path, out_bed):
    if not gtf_path:
        return
    with open(gtf_path, "r") as gtf, open(out_bed, "w") as out:
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
            strand = fields[6] if len(fields) > 6 else "."
            out.write(f"{chrom}\t{start}\t{end}\t{transcript_id}\t{gene_id}\t{strand}\n")


def parse_exon_gtf(gtf_path, out_bed):
    with open(gtf_path, "r") as gtf, open(out_bed, "w") as out:
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


def write_combined_bed(active_bed, inactive_bed, orf2_bed, out_path):
    with open(out_path, "w") as out:
        for bed_path, category in (
            (active_bed, "active"),
            (inactive_bed, "inactive"),
            (orf2_bed, "orf2"),
        ):
            with open(bed_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 6:
                        continue
                    out.write("\t".join(fields[:6] + [category]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bam", required=True)
    parser.add_argument("--l1-ids", required=True)
    parser.add_argument("--l1-gtf", required=True)
    parser.add_argument("--l1-3prime-window", type=int, default=100)
    parser.add_argument("--exon-gtf", required=True)
    parser.add_argument("--exon-overlap-min", type=int, default=100)
    parser.add_argument("--samtools-cmd", required=True)
    parser.add_argument("--bedtools-cmd", required=True)
    parser.add_argument("--qc-summary", required=True)
    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, args.sample, "d_LINE_quantification")
    os.makedirs(out_dir, exist_ok=True)

    l1_bam = os.path.join(out_dir, f"{args.sample}_L1_reads.bam")
    l1_bam_bai = l1_bam + ".bai"

    # Build TE BED for gene_id/transcript_id lookup.
    l1_te_bed = os.path.join(out_dir, "L1_TE_L1_family.bed")
    parse_l1_te_gtf(args.l1_gtf, l1_te_bed)

    exon_bed = os.path.join(out_dir, "exons.bed")
    parse_exon_gtf(args.exon_gtf, exon_bed)

    l1_id_set = set()
    with open(args.l1_ids, "r") as f:
        for line in f:
            rid = line.strip()
            if rid:
                l1_id_set.add(rid)

    # Create read BED from L1 alignments, using best-scoring alignment per read.
    reads_bed = os.path.join(out_dir, "L1_reads.bed")
    read_info = {}
    best_align = {}
    proc = subprocess.Popen(
        f"{args.samtools_cmd} view {args.bam}",
        shell=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
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
        if flag & 0x4 or flag & 0x800:
            continue
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

    with open(reads_bed, "w") as bed_out:
        for qname, entry in best_align.items():
            for block_start, block_end in entry["blocks"]:
                bed_out.write(f"{entry['chrom']}\t{block_start}\t{block_end}\t{qname}\n")
            read_info[qname] = (entry["chrom"], entry["start"], entry["end"], entry["read_len"])

    exon_overlaps_fd, exon_overlaps_path = tempfile.mkstemp(prefix="L1_reads_exon_overlaps_", suffix=".tsv", dir=out_dir)
    os.close(exon_overlaps_fd)
    run(f"{args.bedtools_cmd} intersect -a {reads_bed} -b {exon_bed} -wo > {exon_overlaps_path}")
    exon_overlap = {}
    with open(exon_overlaps_path, "r") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            read_id = fields[3]
            overlap_len = int(fields[7])
            current = exon_overlap.get(read_id, 0)
            if overlap_len > current:
                exon_overlap[read_id] = overlap_len
    exclude_reads = {rid for rid, olen in exon_overlap.items() if olen >= args.exon_overlap_min}

    # Intersect with TE L1 GTF for gene/transcript annotation.
    te_overlaps_fd, te_overlaps_path = tempfile.mkstemp(prefix="L1_reads_L1TE_overlaps_", suffix=".tsv", dir=out_dir)
    os.close(te_overlaps_fd)
    if os.path.getsize(l1_te_bed) == 0:
        open(te_overlaps_path, "w").close()
    else:
        run(f"{args.bedtools_cmd} intersect -a {reads_bed} -b {l1_te_bed} -wo > {te_overlaps_path}")

    te_best = {}
    if os.path.exists(te_overlaps_path):
        with open(te_overlaps_path, "r") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 11:
                    continue
                read_id = fields[3]
                transcript_id = fields[7]
                gene_id = fields[8]
                strand = fields[9]
                overlap_len = int(fields[10])
                if read_id not in best_align:
                    continue
                if read_id in exclude_reads:
                    continue
                read_entry = best_align[read_id]
                if read_entry["strand"] != strand:
                    continue
                if strand == "+":
                    dist_3p = abs(int(fields[6]) - read_entry["end"])
                else:
                    dist_3p = abs(read_entry["start"] - int(fields[5]))
                region_key = (fields[4], fields[5], fields[6], transcript_id, strand)
                current = te_best.get(read_id)
                if current is None or overlap_len > current["overlap_len"]:
                    te_best[read_id] = {
                        "overlap_len": overlap_len,
                        "gene_id": gene_id,
                        "transcript_id": transcript_id,
                        "read_strand": read_entry["strand"],
                        "dist_3p": dist_3p,
                        "region_key": region_key,
                    }

    # Write read-level TSV.
    read_tsv = os.path.join(out_dir, f"{args.sample}_L1_reads.tsv")
    with open(read_tsv, "w") as out:
        out.write(
            "read_id\tchr\tstart\tend\tread_length\toverlap_length\tte_chr\tte_start\tte_end\ttranscript_id\tgene_id\tte_strand\tread_strand\tdist_to_3prime\n"
        )
        for read_id, (chrom, start, end, read_len) in read_info.items():
            if read_id not in te_best:
                continue
            entry = te_best[read_id]
            te_chr, te_start, te_end, _, te_strand = entry["region_key"]
            out.write(
                f"{read_id}\t{chrom}\t{start}\t{end}\t{read_len}\t"
                f"{entry['overlap_len']}\t{te_chr}\t{te_start}\t{te_end}\t"
                f"{entry['transcript_id']}\t{entry['gene_id']}\t{te_strand}\t"
                f"{entry['read_strand']}\t{entry['dist_3p']}\n"
            )

    pass_ids = sorted(te_best.keys())
    pass_ids_path = os.path.join(out_dir, f"{args.sample}_L1_pass_readIDs.txt")
    with open(pass_ids_path, "w") as f:
        for rid in pass_ids:
            f.write(rid + "\n")

    if pass_ids:
        run(f"{args.samtools_cmd} view -b -N {pass_ids_path} {args.bam} -o {l1_bam}")
    else:
        run(f"{args.samtools_cmd} view -H {args.bam} | {args.samtools_cmd} view -b -o {l1_bam} -")
    run(f"{args.samtools_cmd} index {l1_bam}")

    # Transcript-level summary TSV (L1 family TE regions).
    total_reads = parse_total_reads(args.qc_summary)
    summary_tsv = os.path.join(out_dir, f"{args.sample}_L1_summary.tsv")
    transcript_counts = {}
    for entry in te_best.values():
        key = entry["region_key"]
        transcript_counts[key] = transcript_counts.get(key, 0) + 1

    entries = []
    with open(l1_te_bed, "r") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                continue
            chrom, start, end, transcript_id, gene_id, strand = fields[:6]
            key = (chrom, start, end, transcript_id, strand)
            count = transcript_counts.get(key, 0)
            if count == 0:
                continue
            region_len = int(end) - int(start)
            rpk = (count / (region_len / 1000.0)) if region_len else 0.0
            normalized = (count / total_reads) if total_reads else 0
            entries.append(
                (chrom, start, end, transcript_id, gene_id, strand, region_len, count, normalized, rpk)
            )

    sum_rpk = sum(entry[9] for entry in entries)
    with open(summary_tsv, "w") as out:
        out.write("chr\tstart\tend\tlocus\ttranscript_id\tgene_id\tstrand\tregion_length\tread_count\tnormalized_count\tTPM\n")
        for entry in entries:
            chrom, start, end, transcript_id, gene_id, strand, region_len, count, normalized, rpk = entry
            tpm = (rpk / sum_rpk * 1_000_000) if sum_rpk else 0
            locus = f"{chrom}:{start}-{end}"
            out.write(
                f"{chrom}\t{start}\t{end}\t{locus}\t{transcript_id}\t{gene_id}\t{strand}\t{region_len}\t{count}\t{normalized}\t{tpm}\n"
            )

    for path in (te_overlaps_path, exon_overlaps_path):
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
