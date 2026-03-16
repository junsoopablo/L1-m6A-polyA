#!/bin/bash
#SBATCH --job-name=mafia_H9_2
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/mafia_H9_2_%j.out
#SBATCH --error=logs/mafia_H9_2_%j.err

set -euo pipefail

# Paths
MAFIA_DIR="/qbio/junsoopablo/00_Programs/psi-co-mAFiA"
MAFIA_VENV="${MAFIA_DIR}/mafia-venv"
PROJECT_DIR="/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
REF_GENOME="${PROJECT_DIR}/reference/Human.fasta"
GROUP="H9_2"
FAST5_DIR="/scratch1/junsoopablo/IsoTENT_002_L1/${GROUP}"
OUT_DIR="${PROJECT_DIR}/results_group/${GROUP}/h_mafia"
L1_BAM="${PROJECT_DIR}/results_group/${GROUP}/d_LINE_quantification/${GROUP}_L1_reads.bam"

# Model paths
BACKBONE="${MAFIA_DIR}/models/RODAN_HEK293_IVT.torch"
CLASSIFIERS="${MAFIA_DIR}/models/psi-co-mAFiA"

# Load modules
source /etc/profile.d/modules.sh
module load minimap2/2.28
module load samtools/1.23
module load bedtools/2.31.0

mkdir -p "${OUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# Activate mAFiA virtual environment
source "${MAFIA_VENV}/bin/activate"

echo "=== Step 1: Basecalling with RODAN ==="
if [ ! -f "${OUT_DIR}/rodan.fasta" ]; then
    echo "Running RODAN basecalling on ${FAST5_DIR}..."
    basecall \
        --fast5_dir "${FAST5_DIR}" \
        --model "${BACKBONE}" \
        --batchsize 4096 \
        --out_dir "${OUT_DIR}"
else
    echo "rodan.fasta already exists, skipping basecalling..."
fi

echo "=== Step 2: Alignment with minimap2 ==="
if [ ! -f "${OUT_DIR}/aligned.bam" ]; then
    echo "Aligning reads to ${REF_GENOME}..."
    minimap2 --secondary=no -ax splice -uf -k14 -t 16 --cs "${REF_GENOME}" \
        "${OUT_DIR}/rodan.fasta" | \
        samtools view -bST "${REF_GENOME}" -q50 - | \
        samtools sort -@ 8 - > "${OUT_DIR}/aligned.bam"
    samtools index "${OUT_DIR}/aligned.bam"
else
    echo "aligned.bam already exists, skipping alignment..."
fi

echo "=== Step 3: Generate sites BED from aligned BAM regions ==="
SITES_BED="${OUT_DIR}/${GROUP}_sites.bed"
if [ ! -f "${SITES_BED}" ]; then
    echo "Extracting regions from aligned BAM..."
    REGIONS_BED="${OUT_DIR}/${GROUP}_regions.bed"
    bedtools bamtobed -i "${OUT_DIR}/aligned.bam" | sort -k1,1 -k2,2n | bedtools merge -i - > "${REGIONS_BED}"
    echo "Generating m6A and psi sites for $(wc -l < ${REGIONS_BED}) regions..."

    conda run -n research python "${PROJECT_DIR}/scripts/generate_mafia_sites_for_regions.py" \
        --ref_file "${REF_GENOME}" \
        --regions_bed "${REGIONS_BED}" \
        --out_file "${SITES_BED}" \
        --flank 100
else
    echo "Sites BED already exists: ${SITES_BED}"
fi

echo "=== Step 4: Read-level modification prediction ==="
if [ ! -f "${OUT_DIR}/mAFiA.reads.bam" ]; then
    echo "Running read-level prediction..."
    process_reads \
        --bam_file "${OUT_DIR}/aligned.bam" \
        --fast5_dir "${FAST5_DIR}" \
        --sites "${SITES_BED}" \
        --ref_file "${REF_GENOME}" \
        --backbone_model_path "${BACKBONE}" \
        --classifier_model_dir "${CLASSIFIERS}" \
        --num_jobs 4 \
        --batchsize 128 \
        --out_dir "${OUT_DIR}"
else
    echo "mAFiA.reads.bam already exists, skipping..."
fi

echo "=== Step 5: Site-level prediction (pileup) ==="
if [ ! -f "${OUT_DIR}/mAFiA.sites.bed" ]; then
    echo "Running site-level pileup..."
    pileup \
        --bam_file "${OUT_DIR}/mAFiA.reads.bam" \
        --sites "${SITES_BED}" \
        --min_coverage 5 \
        --out_dir "${OUT_DIR}" \
        --num_jobs 16
else
    echo "mAFiA.sites.bed already exists, skipping..."
fi

echo "=== Done ==="
echo "Output files in: ${OUT_DIR}"
ls -la "${OUT_DIR}"

echo ""
echo "Key output files:"
echo "  - ${OUT_DIR}/rodan.fasta: RODAN basecalled sequences"
echo "  - ${OUT_DIR}/aligned.bam: Aligned reads"
echo "  - ${OUT_DIR}/mAFiA.reads.bam: Read-level modification predictions (modBAM)"
echo "  - ${OUT_DIR}/mAFiA.sites.bed: Site-level modification ratios"
