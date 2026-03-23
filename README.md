# L1-m6A-polyA

Analysis code for: "Poly(A) tail retention of LINE-1 transcripts under arsenite stress covaries with m6A density and structural completeness"

Junsoo Kim, Sojeong Lee, Seungbeom Han

Center for RNA Research, Institute for Basic Science (IBS), Seoul National University

## Overview

This repository contains the analysis pipeline and figure generation code for our study of LINE-1 retrotransposon m6A modifications and poly(A) tail dynamics under arsenite stress using Oxford Nanopore direct RNA sequencing.

## Repository Structure

```
├── pipeline/          # Snakemake pipeline for L1 identification + modification calling
├── config/            # Pipeline configuration (config.yaml)
├── analysis/          # Exploratory analysis scripts (organized by topic)
├── figures/           # Figure generation scripts + composed PDFs
│   ├── fig_style.py           # Shared publication style (colors, fonts, line widths)
│   ├── compose_figures.py     # Assembles individual panels into fig1.pdf, fig2.pdf
│   ├── generate_fig1.py       # Fig 1a (ECDF m6A), 1c (DRACH ratio)
│   ├── generate_fig2.py       # Fig 1b (poly(A) ECDF), 1c (Young/Anc), 1f (ChromHMM)
│   ├── generate_fig2ef.py     # Fig 1d (CHX rescue), 1e (XRN1 KD)
│   ├── generate_fig3.py       # Fig 2a (quartile ECDF), 2b (slope chart)
│   ├── generate_fig3cd.py     # Supplementary heatmap
│   ├── fig1d_rna004_validation.py  # Fig 1d (RNA004 violin — via Supplementary)
│   └── fig1ef_consensus_hotspot.py # Hotspot analysis (Supplementary)
├── manuscript/        # LaTeX source
│   ├── main.tex               # Main manuscript
│   ├── supplementary.tex      # Supplementary (8 figures, 4 tables)
│   ├── references.bib
│   ├── main.pdf               # Compiled PDF
│   └── supplementary.pdf
└── CLAUDE.md          # Project context and analysis log
```

## Main Figures

- **Figure 1** (6 panels): L1 m6A enrichment + arsenite-selective poly(A) shortening + mechanism
  - (a) m6A ECDF, (b) poly(A) ECDF, (c) Young vs Ancient, (d) CHX rescue, (e) XRN1 KD, (f) ChromHMM
- **Figure 2** (2 panels): m6A-poly(A) coupling under stress
  - (a) Q1/Q4 quartile ECDF, (b) Slope chart (normal → stress)

## Data Sources

This study reanalysed publicly available datasets:
- **SG-NEx** (ENA: ERP125816): Baseline DRS for 9 human cell lines (23 libraries)
- **PRJNA842344**: HeLa arsenite stress, CHX, XRN1 KD (de Muro et al., eLife 2024)
- **PRJNA1220613 / PRJNA1092333**: PUS enzyme KD datasets

Raw FAST5/POD5 files must be downloaded separately from SRA (not included in this repository due to size).

## Setup on New Server

### 1. Environment
```bash
conda create -n research python=3.10
conda activate research
pip install pandas numpy scipy matplotlib seaborn statsmodels
conda install -c conda-forge tectonic  # LaTeX compiler
```

### 2. System tools
```bash
# Via module load or conda
minimap2 2.28, samtools 1.23, bedtools 2.31.0
# ONT basecaller (requires GPU)
guppy v6.0.0 (rna_r9.4.1_70bps_hac)
```

### 3. Pipeline execution
```bash
# See config/config.yaml for paths
conda run -n bioinfo3 snakemake --cores 16 --use-conda
```

### 4. Figure regeneration
```bash
cd figures/
python generate_fig1.py      # panels a, c + S-figures
python generate_fig2.py      # panels b, c(violin), f(ChromHMM)
python generate_fig2ef.py    # panels d(CHX), e(XRN1)
python generate_fig3.py      # Fig 2a, 2b + S-figures
python compose_figures.py    # assemble into fig1.pdf, fig2.pdf
cd ../manuscript/
tectonic main.tex            # compile PDF
tectonic supplementary.tex
```

## Key Parameters

- **m6A threshold**: 0.80 (ML >= 204/255)
- **Young L1**: L1HS, L1PA1, L1PA2, L1PA3
- **Two-stage filter**: 10% L1 overlap → CIGAR/exon/strand refinement
- **Basecaller**: Guppy only (dorado has systematic bias with MAFIA)
- **Figure style**: 165mm full width, 80mm half width, 8pt fonts, Okabe-Ito-adjacent palette

## License

MIT
