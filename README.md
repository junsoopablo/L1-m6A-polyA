# L1-m6A-polyA

Analysis code for: "Poly(A) tail retention of LINE-1 transcripts under arsenite stress covaries with m6A density and structural completeness"

Junsoo Kim, Sojeong Lee, Seungbeom Han

Center for RNA Research, Institute for Basic Science (IBS), Seoul National University

## Overview

This repository contains the analysis pipeline and figure generation code for our study of LINE-1 retrotransposon m6A modifications and poly(A) tail dynamics under arsenite stress using Oxford Nanopore direct RNA sequencing.

## Repository Structure

- `pipeline/` — Snakemake pipeline for L1 read identification, filtering, and modification calling
- `config/` — Pipeline configuration files
- `analysis/` — Exploratory analysis scripts organized by topic
- `figures/` — Figure generation scripts for all main and supplementary figures
- `manuscript/` — LaTeX source files

## Data Sources

This study reanalysed publicly available datasets:
- **SG-NEx** (ENA: ERP125816): Baseline DRS for 9 human cell lines
- **PRJNA842344**: HeLa arsenite stress, CHX, XRN1 KD (de Muro et al., eLife 2024)
- **PRJNA1220613**: PUS enzyme KD in BE(2)-C
- **PRJNA1092333**: PUS enzyme KD in SH-SY5Y

## Requirements

- Python 3.9+ with pandas, numpy, scipy, matplotlib
- Snakemake 7+
- minimap2 2.28, samtools 1.23, bedtools 2.31.0
- Guppy v6.0.0 (ONT basecaller)
- nanopolish 0.14.0 (poly(A) estimation)
- MAFIA/m6Anet (m6A modification calling)

## License

MIT
