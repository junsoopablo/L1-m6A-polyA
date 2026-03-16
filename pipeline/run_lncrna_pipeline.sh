#!/bin/bash
set -euo pipefail

# =============================================================================
# lncRNA control pipeline: guppy → minimap2 → nanopolish → MAFIA
# Replicates the control pipeline for exclusive lncRNA reads.
#
# Usage: ./scripts/run_lncrna_pipeline.sh <GROUP> [--skip-guppy] [--skip-mafia]
#   e.g.: ./scripts/run_lncrna_pipeline.sh HeLa_1
#
# Requires: GPU node (guppy + MAFIA process_reads need CUDA)
# =============================================================================

GROUP="${1:?Usage: $0 <GROUP> [--skip-guppy] [--skip-mafia]}"
SKIP_GUPPY=false
SKIP_MAFIA=false
for arg in "${@:2}"; do
    case "$arg" in
        --skip-guppy) SKIP_GUPPY=true ;;
        --skip-mafia) SKIP_MAFIA=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# =============================================================================
# Paths
# =============================================================================
PROJECT="/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
REF="${PROJECT}/reference/Human.fasta"
JUNC_BED="${PROJECT}/reference/junctions.bed"

# Tool paths
GUPPY_BIN="/qbio/junsoopablo/00_Programs/ont-guppy/bin/guppy_basecaller"
GUPPY_CONFIG="/qbio/junsoopablo/00_Programs/ont-guppy/data/rna_r9.4.1_70bps_hac.cfg"
GUPPY_DEVICE="cuda:0,1"
SAMTOOLS="/blaze/junsoopablo/conda/envs/research/bin/samtools"
MINIMAP2="/blaze/junsoopablo/conda/envs/research/bin/minimap2"

# MAFIA
MAFIA_DIR="/qbio/junsoopablo/00_Programs/psi-co-mAFiA"
MAFIA_VENV="${MAFIA_DIR}/mafia-venv"
MAFIA_BACKBONE="${MAFIA_DIR}/models/RODAN_HEK293_IVT.torch"
MAFIA_CLASSIFIERS="${MAFIA_DIR}/models/psi-co-mAFiA"
MAFIA_NUM_JOBS=4
MAFIA_BATCHSIZE_BC=4096
MAFIA_BATCHSIZE_PR=128
MAFIA_GPU_DEVICES="1"

# Input/output directories — lncRNA-specific paths
FAST5_DIR="/scratch1/junsoopablo/IsoTENT_002_L1_lncrna/${GROUP}"
GUPPY_SAVE="/scratch1/junsoopablo/IsoTENT_002_L1_lncrna_guppy/${GROUP}"
OUT_DIR="${PROJECT}/results_group/${GROUP}/l_lncrna_ctrl"
BASECALL_DIR="${OUT_DIR}/lncrna_basecall"
MAFIA_OUT="${OUT_DIR}/mafia"

GENERATE_SITES_SCRIPT="${PROJECT}/scripts/generate_mafia_sites_for_regions.py"
PYTHON_CMD="conda run -n research python"

THREADS=12
NANOPOLISH_THREADS=20

echo "============================================================"
echo "lncRNA control pipeline: ${GROUP}"
echo "FAST5 input: ${FAST5_DIR}"
echo "Guppy save:  ${GUPPY_SAVE}"
echo "Output:      ${OUT_DIR}"
echo "============================================================"

mkdir -p "${BASECALL_DIR}" "${MAFIA_OUT}"

# Check FAST5 files exist
FAST5_COUNT=$(find "${FAST5_DIR}" -maxdepth 1 -name "*.fast5" 2>/dev/null | wc -l)
echo "FAST5 files: ${FAST5_COUNT}"
if [ "${FAST5_COUNT}" -eq 0 ]; then
    echo "ERROR: No FAST5 files found in ${FAST5_DIR}"
    exit 1
fi

# =============================================================================
# Phase A: Guppy basecall (GPU)
# =============================================================================
FASTQ="${BASECALL_DIR}/${GROUP}.lncrna.fastq.gz"
SUMMARY="${BASECALL_DIR}/${GROUP}.lncrna.sequencing_summary.txt"

if [ "$SKIP_GUPPY" = true ] && [ -s "${FASTQ}" ]; then
    echo "[Phase A] Skipping guppy (--skip-guppy, FASTQ exists)"
else
    echo ""
    echo "[Phase A] Guppy basecalling..."
    mkdir -p "${GUPPY_SAVE}"

    ${GUPPY_BIN} \
        --input_path "${FAST5_DIR}" \
        --save_path "${GUPPY_SAVE}" \
        --config "${GUPPY_CONFIG}" \
        --recursive \
        --fast5_out \
        --post_out \
        --compress_fastq \
        --disable_qscore_filtering \
        --device "${GUPPY_DEVICE}"

    # Collect sequencing summary
    summary_src=$(find "${GUPPY_SAVE}" -name "sequencing_summary*.txt" -type f | head -n 1)
    if [ -n "${summary_src}" ]; then
        cp "${summary_src}" "${SUMMARY}"
    else
        echo "WARNING: No sequencing_summary.txt found"
        : > "${SUMMARY}"
    fi

    # Concatenate all FASTQ files
    : > "${FASTQ}"
    find "${GUPPY_SAVE}" -type f -name "*.fastq.gz" -print0 \
        | xargs -0 -I {} cat {} >> "${FASTQ}"

    echo "  FASTQ: $(zcat -f "${FASTQ}" | awk 'NR%4==1' | wc -l) reads"
fi

if [ ! -s "${FASTQ}" ]; then
    echo "ERROR: Empty FASTQ after basecalling"
    exit 1
fi

# =============================================================================
# Phase B: minimap2 alignment (CPU)
# =============================================================================
BAM="${OUT_DIR}/${GROUP}_lncrna_mapped.bam"

echo ""
echo "[Phase B] minimap2 alignment..."

${MINIMAP2} -ax splice -uf -k14 -t ${THREADS} \
    "${REF}" "${FASTQ}" \
| ${SAMTOOLS} view -b - \
| ${SAMTOOLS} sort -@ ${THREADS} -o "${BAM}" -

${SAMTOOLS} index "${BAM}"

MAPPED=$(${SAMTOOLS} view -c -F 4 "${BAM}")
echo "  Mapped reads: ${MAPPED}"

# =============================================================================
# Phase C: nanopolish index + polya (CPU)
# =============================================================================
POLYA="${OUT_DIR}/${GROUP}.lncrna.nanopolish.polya.tsv.gz"

echo ""
echo "[Phase C] nanopolish polya..."

# Load nanopolish
module purge 2>/dev/null || true
module load nanopolish/0.14.0.20240114 2>/dev/null || true

nanopolish index -d "${GUPPY_SAVE}/workspace" "${FASTQ}"

nanopolish polya \
    --reads "${FASTQ}" \
    --bam "${BAM}" \
    --genome "${REF}" \
    --threads ${NANOPOLISH_THREADS} \
| gzip -c > "${POLYA}"

PASS_READS=$(zcat "${POLYA}" | awk -F'\t' 'NR>1 && $NF=="PASS"' | wc -l)
echo "  nanopolish PASS reads: ${PASS_READS}"

# =============================================================================
# Phase D-E: MAFIA (GPU)
# =============================================================================
if [ "$SKIP_MAFIA" = true ]; then
    echo ""
    echo "[Phase D-E] Skipping MAFIA (--skip-mafia)"
else
    echo ""
    echo "[Phase D] MAFIA: rodan basecall + align + generate sites..."

    # Save full paths to system tools BEFORE venv activation
    module purge 2>/dev/null || true
    module load minimap2/2.28 2>/dev/null || true
    module load samtools/1.23 2>/dev/null || true
    module load bedtools/2.31.0 2>/dev/null || true
    SYS_MINIMAP2=$(which minimap2)
    SYS_SAMTOOLS=$(which samtools 2>/dev/null || echo "${SAMTOOLS}")
    SYS_BEDTOOLS=$(which bedtools 2>/dev/null || echo "/blaze/junsoopablo/conda/envs/research/bin/bedtools")

    export CUDA_VISIBLE_DEVICES="${MAFIA_GPU_DEVICES}"
    source "${MAFIA_VENV}/bin/activate"

    # D1: RODAN basecall (GPU)
    RODAN_FASTA="${MAFIA_OUT}/${GROUP}.lncrna.rodan.fasta"

    basecall \
        --fast5_dir "${FAST5_DIR}" \
        --model "${MAFIA_BACKBONE}" \
        --batchsize "${MAFIA_BATCHSIZE_BC}" \
        --out_dir "${MAFIA_OUT}"

    mv "${MAFIA_OUT}/rodan.fasta" "${RODAN_FASTA}"
    echo "  RODAN reads: $(grep -c '^>' "${RODAN_FASTA}" || echo 0)"

    # D2: Align rodan output (with --junc-bed for MAFIA)
    MAFIA_BAM="${MAFIA_OUT}/${GROUP}.lncrna.aligned.bam"

    ${SYS_MINIMAP2} -ax splice --junc-bed "${JUNC_BED}" -uf --secondary=no -k14 \
        -t ${THREADS} --cs "${REF}" "${RODAN_FASTA}" \
    | ${SYS_SAMTOOLS} view -b - \
    | ${SYS_SAMTOOLS} sort -@ ${THREADS} - > "${MAFIA_BAM}"

    ${SYS_SAMTOOLS} index "${MAFIA_BAM}"

    # D3: Generate modification sites
    REGIONS_BED="${MAFIA_OUT}/${GROUP}.lncrna_regions.bed"
    SITES_BED="${MAFIA_OUT}/${GROUP}.lncrna_sites.bed"

    ${SYS_BEDTOOLS} bamtobed -i "${MAFIA_BAM}" \
    | awk -F'\t' 'BEGIN{OFS="\t"} {print $1, $2, $3, ".", ".", $6}' \
    | sort -k1,1 -k2,2n -k6,6 \
    | ${SYS_BEDTOOLS} merge -i - -s -c 6 -o distinct > "${REGIONS_BED}"

    ${PYTHON_CMD} "${GENERATE_SITES_SCRIPT}" \
        --ref_file "${REF}" \
        --regions_bed "${REGIONS_BED}" \
        --out_file "${SITES_BED}" \
        --flank 100 \
        --strand_specific

    SITES_COUNT=$(wc -l < "${SITES_BED}")
    echo "  Modification sites: ${SITES_COUNT}"

    # E: process_reads (GPU)
    echo ""
    echo "[Phase E] MAFIA: process_reads..."
    READS_BAM="${MAFIA_OUT}/${GROUP}.lncrna.mAFiA.reads.bam"

    if [ "${SITES_COUNT}" -le 1 ]; then
        echo "  No sites to process, creating empty BAM"
        ${SYS_SAMTOOLS} view -H "${MAFIA_BAM}" | ${SYS_SAMTOOLS} view -b - > "${READS_BAM}"
        ${SYS_SAMTOOLS} index "${READS_BAM}"
    else
        set +e
        process_reads \
            --bam_file "${MAFIA_BAM}" \
            --fast5_dir "${FAST5_DIR}" \
            --sites "${SITES_BED}" \
            --ref_file "${REF}" \
            --backbone_model_path "${MAFIA_BACKBONE}" \
            --classifier_model_dir "${MAFIA_CLASSIFIERS}" \
            --num_jobs "${MAFIA_NUM_JOBS}" \
            --batchsize "${MAFIA_BATCHSIZE_PR}" \
            --out_dir "${MAFIA_OUT}"
        pr_status=$?
        set -e

        if [ -f "${MAFIA_OUT}/mAFiA.reads.bam" ] && [ -s "${MAFIA_OUT}/mAFiA.reads.bam" ]; then
            mv "${MAFIA_OUT}/mAFiA.reads.bam" "${READS_BAM}"
            ${SYS_SAMTOOLS} index "${READS_BAM}"
        else
            echo "  WARNING: process_reads produced no output, creating header-only BAM"
            ${SYS_SAMTOOLS} view -H "${MAFIA_BAM}" | ${SYS_SAMTOOLS} view -b - > "${READS_BAM}"
            ${SYS_SAMTOOLS} index "${READS_BAM}"
        fi
    fi

    # Pileup (optional, for per-site stats)
    PILEUP_BED="${MAFIA_OUT}/${GROUP}.lncrna.mAFiA.sites.bed"
    if [ -s "${READS_BAM}" ]; then
        set +e
        pileup \
            --bam_file "${READS_BAM}" \
            --sites "${SITES_BED}" \
            --min_coverage 1 \
            --out_dir "${MAFIA_OUT}" \
            --num_jobs ${THREADS}
        set -e

        if [ -f "${MAFIA_OUT}/mAFiA.sites.bed" ]; then
            mv "${MAFIA_OUT}/mAFiA.sites.bed" "${PILEUP_BED}"
        else
            echo -e "chrom\tchromStart\tchromEnd\tname\tscore\tstrand\tref5mer\tcoverage\tmodRatio\tconfidence" > "${PILEUP_BED}"
        fi
    fi

    deactivate 2>/dev/null || true

    MAFIA_READS=$(${SAMTOOLS} view -c "${READS_BAM}" 2>/dev/null || echo 0)
    echo "  MAFIA reads BAM: ${MAFIA_READS} reads"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Pipeline complete for ${GROUP}"
echo "  nanopolish poly(A): ${POLYA}"
echo "  nanopolish PASS:    ${PASS_READS} reads"
if [ "$SKIP_MAFIA" = false ]; then
    echo "  MAFIA reads BAM:   ${READS_BAM}"
fi
echo "============================================================"
