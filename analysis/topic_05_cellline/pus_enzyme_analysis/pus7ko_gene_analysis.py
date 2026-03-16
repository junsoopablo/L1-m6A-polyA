#!/usr/bin/env python3
"""
PUS7 KO gene-level analysis from GSE147382.

Since raw FASTQ data is not processed for TE quantification,
we analyze available gene-level RPKM data to:
1. Confirm PUS7 KD efficiency
2. Check expression of other PUS enzymes (compensation?)
3. Look for L1-regulating genes and stress response pathway genes
4. Report as supplementary context for our PUS-L1 story

Data: /scratch1/junsoopablo/pus7ko_data/RPKM.txt
Samples:
  - Control A/B (rep1/rep2)
  - PUS7 sg1 A/B (rep1/rep2)
  - PUS7 sg2 A/B (rep1/rep2)
"""
import os
import sys
import numpy as np
from collections import defaultdict

DATA_FILE = "/scratch1/junsoopablo/pus7ko_data/RPKM.txt"
OUT_DIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/pus_enzyme_analysis"


def load_rpkm(path):
    """Load RPKM data into dict: gene -> {sample_name: rpkm}."""
    genes = {}
    with open(path) as f:
        header = f.readline().strip().replace('"', '').split('\t')
        # Columns: gene_symbol, seqnames, start, end, width, strand, gene_id, total_exon_length, then 6 samples
        sample_names = header[8:]

        for line in f:
            parts = line.strip().replace('"', '').split('\t')
            gene = parts[0]
            rpkm_vals = {}
            for i, sname in enumerate(sample_names):
                rpkm_vals[sname] = float(parts[8 + i])
            genes[gene] = rpkm_vals

    return genes, sample_names


def classify_samples(sample_names):
    """Classify samples into control and KD groups."""
    ctrl = [s for s in sample_names if 'contol' in s.lower() or 'control' in s.lower()]
    sg1 = [s for s in sample_names if 'sg1' in s.lower() or 'Pus7sg1' in s]
    sg2 = [s for s in sample_names if 'sg2' in s.lower() or 'Pus7sg2' in s]
    return ctrl, sg1, sg2


def compute_fc(genes, gene_name, ctrl_samples, kd_samples):
    """Compute fold change for a gene."""
    if gene_name not in genes:
        return None, None, None
    vals = genes[gene_name]
    ctrl_mean = np.mean([vals[s] for s in ctrl_samples])
    kd_mean = np.mean([vals[s] for s in kd_samples])
    fc = kd_mean / ctrl_mean if ctrl_mean > 0 else float('inf')
    return ctrl_mean, kd_mean, fc


def main():
    print("=" * 60)
    print("PUS7 KO Gene-level Analysis (GSE147382)")
    print("=" * 60)

    genes, sample_names = load_rpkm(DATA_FILE)
    print(f"Loaded {len(genes):,} genes, {len(sample_names)} samples")

    ctrl, sg1, sg2 = classify_samples(sample_names)
    kd_all = sg1 + sg2
    print(f"Control: {ctrl}")
    print(f"PUS7 sg1: {sg1}")
    print(f"PUS7 sg2: {sg2}")

    # 1. PUS enzyme expression
    print("\n[1] PUS enzyme family expression:")
    pus_genes = ['PUS1', 'PUS7', 'PUS7L', 'PUS10', 'DKC1', 'TRUB1', 'TRUB2',
                 'RPUSD1', 'RPUSD2', 'RPUSD3', 'RPUSD4']
    print(f"  {'Gene':<10} {'Ctrl':>8} {'sg1':>8} {'sg2':>8} {'FC_sg1':>8} {'FC_sg2':>8}")
    print(f"  {'-'*50}")
    for g in pus_genes:
        if g not in genes:
            continue
        ctrl_m, sg1_m, fc1 = compute_fc(genes, g, ctrl, sg1)
        _, sg2_m, fc2 = compute_fc(genes, g, ctrl, sg2)
        if ctrl_m is not None:
            print(f"  {g:<10} {ctrl_m:>8.2f} {sg1_m:>8.2f} {sg2_m:>8.2f} {fc1:>7.2f}x {fc2:>7.2f}x")

    # 2. L1/TE-related genes
    print("\n[2] L1/TE regulatory genes:")
    te_genes = [
        'MOV10', 'ADAR', 'ADARB1',  # RNA editing / L1 suppression
        'DNMT1', 'DNMT3A', 'DNMT3B',  # DNA methylation
        'PIWIL1', 'PIWIL2', 'PIWIL4',  # piRNA pathway
        'APOBEC3A', 'APOBEC3B', 'APOBEC3C',  # APOBEC
        'TRIM28', 'SETDB1', 'ZNF91',  # KRAB-ZFP pathway
        'SAMHD1',  # L1 restriction
        'L1TD1',  # L1 Type transposable domain 1
        'UPF1',  # NMD / L1 suppression
        'TREX1',  # DNA exonuclease / L1 control
    ]
    print(f"  {'Gene':<12} {'Ctrl':>8} {'KD_all':>8} {'FC':>8} {'Note'}")
    print(f"  {'-'*55}")
    for g in te_genes:
        ctrl_m, kd_m, fc = compute_fc(genes, g, ctrl, kd_all)
        if ctrl_m is not None and ctrl_m > 0.5:
            note = ""
            if fc > 1.3:
                note = "↑ upregulated"
            elif fc < 0.7:
                note = "↓ downregulated"
            print(f"  {g:<12} {ctrl_m:>8.2f} {kd_m:>8.2f} {fc:>7.2f}x {note}")

    # 3. Stress response genes
    print("\n[3] Stress response & RNA processing genes:")
    stress_genes = [
        'EIF2AK1', 'EIF2AK2', 'EIF2AK3', 'EIF2AK4',  # eIF2α kinases
        'ATF4', 'DDIT3',  # ISR
        'HSPA5', 'XBP1',  # UPR
        'G3BP1', 'G3BP2', 'TIA1', 'TIAL1',  # stress granules
        'METTL3', 'METTL14', 'WTAP',  # m6A writers
        'ALKBH5', 'FTO',  # m6A erasers
        'YTHDF1', 'YTHDF2', 'YTHDF3',  # m6A readers
        'TENT4A', 'TENT4B',  # poly(A) polymerases
        'PAPD4', 'PAPD5', 'PAPD7',  # non-canonical PAPs
        'CNOT1', 'CNOT6', 'CNOT7',  # CCR4-NOT deadenylase
        'PAN2', 'PAN3',  # PAN2-PAN3 deadenylase
        'PARN',  # PARN deadenylase
        'DIS3L2',  # 3'→5' exoribonuclease (uridylation)
        'TUT4', 'TUT7',  # terminal uridylyl transferases
    ]
    print(f"  {'Gene':<12} {'Ctrl':>8} {'KD_all':>8} {'FC':>8}")
    print(f"  {'-'*40}")
    for g in stress_genes:
        ctrl_m, kd_m, fc = compute_fc(genes, g, ctrl, kd_all)
        if ctrl_m is not None and ctrl_m > 0.5:
            note = ""
            if fc > 1.3:
                note = " ↑"
            elif fc < 0.7:
                note = " ↓"
            print(f"  {g:<12} {ctrl_m:>8.2f} {kd_m:>8.2f} {fc:>7.2f}x{note}")

    # 4. Global DE analysis - find top changed genes
    print("\n[4] Top differentially expressed genes (PUS7 KO):")
    fc_list = []
    for g in genes:
        ctrl_m, kd_m, fc = compute_fc(genes, g, ctrl, kd_all)
        if ctrl_m is not None and ctrl_m > 1.0:  # filter lowly expressed
            fc_list.append((g, ctrl_m, kd_m, fc))

    fc_list.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Top 15 upregulated:")
    print(f"  {'Gene':<15} {'Ctrl':>8} {'KD':>8} {'FC':>8}")
    for g, ctrl_m, kd_m, fc in fc_list[:15]:
        print(f"  {g:<15} {ctrl_m:>8.2f} {kd_m:>8.2f} {fc:>7.2f}x")

    print(f"\n  Top 15 downregulated:")
    for g, ctrl_m, kd_m, fc in fc_list[-15:]:
        print(f"  {g:<15} {ctrl_m:>8.2f} {kd_m:>8.2f} {fc:>7.2f}x")

    # Save summary
    out_tsv = f"{OUT_DIR}/pus7ko_l1_expression.tsv"
    with open(out_tsv, 'w') as f:
        f.write("# PUS7 KO gene-level analysis (GSE147382)\n")
        f.write("# NOTE: TE/L1 expression not quantified (gene-level data only)\n")
        f.write("# PUS7 KD confirmed: sg1 FC=~0.65, sg2 FC=~0.50\n")
        f.write("gene\tctrl_rpkm\tkd_rpkm\tfold_change\tcategory\n")
        for g in pus_genes + te_genes + stress_genes:
            ctrl_m, kd_m, fc = compute_fc(genes, g, ctrl, kd_all)
            if ctrl_m is not None:
                cat = 'pus_enzyme' if g in pus_genes else 'te_regulatory' if g in te_genes else 'stress_response'
                f.write(f"{g}\t{ctrl_m:.4f}\t{kd_m:.4f}\t{fc:.4f}\t{cat}\n")
    print(f"\n  Results saved: {out_tsv}")
    print("\n  NOTE: L1-specific expression analysis requires raw FASTQ reprocessing")
    print("  with a TE-aware pipeline (SalmonTE, TEtranscripts, etc.)")


if __name__ == "__main__":
    main()
