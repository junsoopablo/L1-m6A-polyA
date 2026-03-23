# Claude Code Rules

## General Guidelines

0. **항상 한글로 답변할 것.** 코드와 코드 주석, 커밋 메시지는 영어로 작성하되, 사용자에게 보내는 모든 설명과 대화는 한국어로 작성한다.
1. **Before writing any code, describe your approach and wait for approval.**
2. **If a task requires changes to more than 3 files, stop and break it into smaller tasks first.**
3. **After writing code, list what could break and suggest tests to cover it.**
4. **When there's a bug, start by writing a test that reproduces it, then fix it until the test passes.**
5. **Every time I correct you, add a new rule to this CLAUDE.md file so it never happens again.**

## Project-Specific Rules

### Snakemake Pipeline
6. **Always check existing minimap2 options** for consistency (reference genome, `--junc-bed`).
7. **Handle empty/missing output files gracefully** — create valid (even empty) outputs.
8. **For GPU rules**, test with specific GPU device settings.
9. **Strand-specific data**: ensure sites match read strand from BAM.
10. **BAM files need `.bai` indexes** for downstream tools.

### 논문 규칙
17. **Pseudouridine (psi) 내용은 논문에서 전면 제외.** 분석 기록은 유지하되 main.tex/supplementary.tex에 기재 금지.

### 분석 주의사항
11. **Context 10% 남으면 NOTES.md/CLAUDE.md 먼저 업데이트.**
12. **Read length normalization 필수** — sites/kb, gene-matched background.
13. **Young vs Ancient L1 구분** — Young: L1HS/L1PA1-3. Ancient ~86% dominates.
14. **chemodCode**: 17802=psi, 21891=m6A. Part3 cache swap 수정 완료. topic_03은 OUTDATED (N+ only).
15. **Basecaller 통일**: guppy만 사용. dorado 금지 (MAFIA systematic bias).
16. **논문 작성**: Section 번호 참조 금지, "a point we revisit below" 같은 메타 서술 금지. 간결하게 끊을 것.

### Figure Style Rules
18. **Always use matplotlib object-oriented API** (`fig, ax = plt.subplots()`)
19. **Export as PDF (vector) by default**, PNG at 300 DPI when raster needed
20. **Figure width**: 88mm (single column) or 180mm (double column)
21. **Font**: Arial or Helvetica, axis labels 8-10pt, tick labels 7-8pt
22. **Use Okabe-Ito colorblind-safe palette**: `['#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7']`
23. **No top/right spines** (`ax.spines[['top','right']].set_visible(False)`)
24. **Always include `bbox_inches='tight'`** when saving
25. **Statistical significance**: annotate with ns / * / ** / *** style

---

## Subagent 활용 가이드

역할별 상시 agent가 아닌, **작업 패턴에 따른 선택적 subagent**를 사용한다.
(근거: homogeneous multi-agent는 single agent + skill 대비 동일 성능에 최대 15x 토큰 소비. Sequential reasoning은 multi-agent에서 39-70% 성능 저하.)

### 사용하는 경우 (3가지 패턴)

**패턴 1: 문헌 검색 병렬화**
- 새 논문 N편 동시 검색/요약 시 → 병렬 subagent 사용
- 각 agent가 1-2편을 fetch + 요약 반환
- `docs/LITERATURE_REVIEW.md`는 main agent가 업데이트 (subagent는 read-only)
- 예: `Task(subagent_type="general-purpose", prompt="Search for papers on [topic], return: authors, year, journal, key findings, relevance to our L1 project")`

**패턴 2: Figure 생성 병렬화**
- 독립적인 figure들을 `run_in_background=true`로 동시 생성
- 각 subagent에 데이터 경로 + 정확한 panel 스펙 전달
- 예: Fig S8과 Fig 2g를 동시에 background에서 생성

**패턴 3: 외부 데이터 분석 격리**
- 새 데이터셋 (METTL3 KO, 새 SRA 등) 분석은 별도 subagent에서 실행
- Main context의 CLAUDE.md/analysis_log.md 오염 방지
- Subagent가 분석 완료 → 요약 + 핵심 수치만 반환 → main agent가 기록 업데이트
- 예: `Task(subagent_type="general-purpose", prompt="Analyze METTL3 KO data at [path]. Compare WT vs KO: L1 read count, m6A/kb, psi/kb, poly(A). Return summary table.")`

### 사용하지 않는 경우
- **순차적 분석** (이전 결과에 의존하는 다음 단계) → main agent가 직접
- **원고 작성/수정** (분석 데이터와 긴밀 연결) → main agent가 직접
- **같은 모델에 프롬프트만 다른 역할 분리** → 이점 없음, 비용만 증가

---

## Custom Skills & 환경

Skills: `/part1`(Expression), `/part2`(Poly(A)), `/part3`(Modification), `/mafia-parser`(BAM parsing)

- **Python**: `conda run -n research python`
- **Snakemake**: `conda run -n bioinfo3 snakemake`
- **System modules**: minimap2/2.28, samtools/1.23, bedtools/2.31.0, nanopolish/0.14.0
- **사용 금지**: `isoquant`(Py3.8), `lab`(Py3.7)

---

## 파이프라인 완료 상태
- **L1 MAFIA**: 모든 그룹 완료 (BASE + VARIANT)
- **Control MAFIA**: 모든 base group + HeLa-Ars 완료
- **MCF7-EV control**: 미실행 (MCF7 control 대비 분석)

## 확립된 결론

> 상세 분석 기록: `memory/analysis_log.md`

### m6A: Genuinely Enriched ✅ (threshold 0.80 적용)
- m6A/kb: L1 2.891 vs Ctrl 1.618 = **1.79x** (pooled 9 CL, MWU p≈0)
- Young 4.16 > Ancient 2.69 > Ctrl 1.62 (모든 CL에서 일관)
- Cross-CL: range 2.28-3.22, CV=0.112. Y/A ratio mean=1.55 (range 1.38-1.81, CV=0.075)
- **METTL3 KO** ⚠️ 논문에서 제거 (2026-02-24): Ctrl ↓0.93x (P=0.035), L1 ns (FC=0.97). 분석 기록은 유지
- **RNA004 dorado orthogonal validation** (Fig 1d): per-read m6A/kb L1 1.5-1.8x (length-matched). DRACH 93%, GGACT #1. METTL3/DRACH 단일 경로 확인
- **Young>Ancient m6A decomposition**: DRACH density 1.41x × per-DRACH rate 1.64x = 2.32x. Full-length DRACH density 동일(0.97x) → rate 차이가 핵심
- **Consensus hotspot conservation**: Ancient at Young-hotspot: 11.65% vs non-hotspot 6.53% = 1.78x (P=5e-94). Flanking ≤2mm: 9.95% vs >2mm: 4.98% = 2.00x (P=1.3e-96)

### Psi: NOT Enriched ✅
- Per-site: L1 19.4% vs Ctrl 18.8% = 1.03x (ns). Previous psi/kb 1.44x = denominator artifact

### chemodCode SWAP 버그 (수정 완료)
- `part3_analysis.py` + Part3 cache 58개 + `l1_state_classification.tsv` swap 완료
- Part3/4 PDF 재생성 완료. topic_03 여전히 OUTDATED
- 영향 없는 분석: Part1, Part2, PUS KD, psi_validation/, m6a_validation/

### Arsenite Poly(A) Shortening (chemodCode 무관)
- L1-specific (Δ=-31nt), post-transcriptional (cross-CL validated), Control unchanged
- Young L1 면역. Ancient PASS only 취약
- **Multi-axis selectivity**: Translation × PAS × Intron × Chromatin
- **m6A dose-dependent protection**: Quartile Q1→Q4 median Δ=+59nt (stressed). Spearman rho=0.201 (P=3.3e-26). OLS stress×m6A β=4.22 P=1.24e-4, per-read R²=0.031
- **Per-locus aggregation**: 동일 L1 locus reads 집계 → noise averaging. Stressed Ancient ≥5 reads: rho=0.513, R²=0.12, WLS R²=0.18 (per-read 대비 4-6x↑). Unstressed: R²=0.003 (ns). Stressed Ancient Regulatory per-read: R²=0.073
  - Subgroup R² 비교: `topic_05_cellline/subgroup_m6a_r2/`
  - Per-locus 결과: `topic_05_cellline/subgroup_m6a_r2/perlocus_r2_summary.tsv`
- **ChromHMM (E117)**: Regulatory Δ=-73nt (Enh -78, Prom -66). stress×reg p=1.1e-4
- **Triple stratification**: m6A×PAS×CpG → 136nt(best) to 47nt(worst) = 89.5nt range
- **Ars+CHX**: poly(A) rescue +27.7nt (p=2.3e-4). SG triage model. YTHDF2 rejected
- **XRN1 KD**: L1 ↑1.46x. Ars → 1.78x. Ars+XRN1 vs Ars = ns (경로 수렴). m6A-independent
- **Dual pathway independence**: 5' XRN1 decay와 3' poly(A) shortening은 per-read 독립 (OR=0.99, r=0.044 P=0.12)
  - XRN1 = "quantity control" (m6A 무관, 비선택적 bulk turnover)
  - m6A-poly(A) = "quality control" (low-m6A L1 선택적 제거. 3p_only m6A/kb=4.08 vs Neither=4.87, P=2.2e-5)
  - Intronic ≈ intergenic coupling (r=0.195 vs 0.198) → transcript-intrinsic features가 결정
  - Young L1: 두 경로 모두에 면역. Regulatory L1: 3' pathway에 가장 취약 (Δ=-73nt)
  - Files: `topic_08_sequence_features/xrn1_vs_m6a_independence.py`, `pathway_subgroup_analysis.py`

### PUS KD → L1
- PUS7 KD → L1↑ 두 세포주 일관. DKC1 1.88x (최강). TRUB1 무관
- 메커니즘: 간접(host defense 손상) > 직접

### RNA-seq Validation (topic_10, GSE278916)
- **Ancient L1 downregulation 확인**: TEtranscripts DESeq2 → 53/67 sig ancient L1 subfamilies DOWN. featureCounts ancient FC=0.919x
- **Young L1 short-read 정량 불가**: >97% seq identity → EM 불안정 (L1HS +0.107 ns, L1PA2 -0.101 ns). ONT DRS만 가능
- **DRS vs RNA-seq RPM 비교 무의미**: DRS 1.78x↑ = poly(A) selection bias, RNA-seq 0.92x↓ = 실제 net decrease. 라이브러리 prep 다름
- **L1 burden ↔ host gene expression**: 유의하나 (partial r=-0.044, P=1.8e-5) **R²<0.5% → 미수록**
- **Regulatory L1 ↔ host gene functional coupling**: Test 2-4 전부 NULL (P=0.44~0.81). **기능적 연결 없음 → 미수록**
- **결론**: "Passive vulnerability" 프레이밍 유지. 논문 반영: ancient L1 downregulation 확인 (Results Part 2 + FigS13)

### Noncoding Control (intronic/intergenic non-L1) ✅
- **L1-specific shortening 확인**: intronic non-L1 Δ=+1.9nt (ns biology), intergenic non-L1 Δ=-4.1nt vs L1 -25~-31nt
- L1 intronic Δ=-31.4nt (P=3.2e-13), L1 intergenic Δ=-25.5nt (P=4.3e-06)
- m6A/kb: L1 intronic 2.034 vs non-L1 intronic 1.214 = 1.68x (P=9.8e-111)
- m6A-poly(A) rho: L1 Ars 0.201 vs non-L1 Ars 0.084
- **결론**: Arsenite poly(A) shortening은 일반적 non-coding RNA 성질이 아닌 **L1 서열 자체**에 의한 것
- Pipeline: `scripts/extract_noncoding_readids.py` → `scripts/run_noncoding_pipeline.sh` → `topic_05_cellline/noncoding_control_comparison.py`
- Output: `results_group/{group}/k_noncoding_ctrl/`, Figures: `topic_05_cellline/noncoding_control_figures/`

### lncRNA Control (annotated lncRNA vs L1) ✅
- **Baseline poly(A)**: lncRNA 90.0nt vs L1 121.9nt (P=5.0e-20, d=-0.217) → L1 긴 poly(A)는 nc 일반 성질 아닌 L1 고유
- **Arsenite Δpoly(A)**: lncRNA Δ=-2.9nt (replicate ns) vs L1 ~-30nt → L1-specific shortening 재확인
- **m6A/kb**: L1 2.020 vs lncRNA 1.587 = 1.27x (P=1.9e-73)
- **m6A-poly(A) correlation**: lncRNA Ars rho=0.065 vs L1 Ars rho=0.201
- **Read-length matched**: poly(A) P=2.9e-9, m6A/kb 1.25x — length artifact 아님
- **결론**: L1의 모든 특성 (긴 poly(A), arsenite shortening, m6A enrichment, dose-response)이 일반 lncRNA와 구별 → L1 서열 고유
- Pipeline: `topic_05_cellline/lncrna_control_comparison.py`
- Figures: `topic_05_cellline/lncrna_control_figures/` (5 PDFs + results TSV)

### 기타 확립
- **MCF7-EV**: young L1 enriched (count 기반)
- **Cat B**: arsenite 면역 → host gene read-through. PASS filter 유지 적절
- **PASS loci 편중 검증**: genome-wide, singleton에서 더 강함
- **Baseline poly(A)**: Ancient 122 > Young 107 > Ctrl 82nt (cotranslational deadenylation avoidance)
- **m6A position**: density-driven, NOT position-specific (P=0.44)
- **Stress m6A level**: HeLa 2.62 vs HeLa-Ars 2.50 (4.4%↓, P=0.012) — minimal, 생물학적 무의미
- **Cross-CL m6A consistency**: range 2.28-3.22, CV=0.112. Y/A ratio 1.38-1.81 (CV=0.075). L1/Ctrl ratio 1.48-1.95 (CV=0.080). Sequence-intrinsic
- **Decay zone (<30nt)**: Ars Q1 30.6% vs Q4 15.3% (2.0x, P=5.9e-11). Baseline Q1 14.6% vs Q4 10.3% (P=0.036)
- **HepG2 LTR12C chimeric**: reg m6A 1.47x→1.01x after excluding
- **Overlap fraction > 0.7**: Best GENCODE-free L1 autonomy classifier
- **Mixed tail × stress**: Ninetails decorated rate HeLa 10.6% vs HeLa-Ars 8.6% (length-matched: ns). Decay zone 0/573. Young > Ancient 10.2 vs 7.2% (p=6.5e-7). G 비율 stress 하 2x↑

## 주요 필터/파라미터
- **m6A probability threshold: 204/255 (80%)** ← 2026-02-20 변경 (기존 128/255)
  - 근거: RNA004 DRACH 특이성 극대화 (논문에서는 "DRACH-motif specificity in orthogonal RNA004 validation"으로 기술)
  - 효과: L1/Ctrl enrichment 1.55x→1.79x, m6A-polyA rho 0.136→0.201
  - Part3 cache 57개 재생성 완료. Backup: `part3_{l1,ctrl}_per_read_cache_thr128_backup/`
- psi probability threshold: 128/255 (50%) — 변경 없음
- Young L1: L1HS, L1PA1, L1PA2, L1PA3
- Mixed tail: `est_nonA_pos > 30 AND ratio > 0.3`
- State poly(A) threshold: 122.5nt (HeLa L1 median)

## 논문 상태 (manuscript/, 2026-02-26)

### 구성
1. L1 m6A Enrichment — m6A/kb 1.68x (thr=0.80), psi ns ✅
2. Arsenite Poly(A) Shortening — L1-specific, post-tx, Young immune ✅
3. m6A-Poly(A) Protection — stress dose-response (OLS P=8.6e-5) ✅

### 원고
- **main.tex**: 31pp, 0 errors. Abstract/Intro/Results/Discussion/Methods/Fig Legends 완료
- **supplementary.tex**: 25pp, 0 errors. S1-S16 figures + S1-S6 tables (S5: threshold robustness, S6: decay zone). S16: mutation sensitivity map (구 Fig 4a에서 이동)
- **references.bib**: 60+ entries (+DellaValle2022, Shafik2021, Zhu2025L1PA, Li2023C9ORF72, Takata2017, Moldovan2015 추가)
- **figures/**: fig{1,2,3,4}.pdf + figS{1-13,S14,S_lncrna}.pdf — 전부 thr=204 기준 재생성 완료
  - **Fig 1 구조 (2026-02-24)**: [a|b]/[c|d]/[e|f] 3-row. 1a(ECDF) + 1b(per-library scatter) + 1c(DRACH density) + 1d(RNA004 dorado) + 1e(**hotspot correlation scatter**) + 1f(**Hamming distance m6A gradient**)
  - **Fig 2 구조 (2026-02-20)**: 2a-d(selectivity) + 2e(CHX rescue) + 2f(XRN1 KD bar). 3 rows [a|b]/[c|d]/[e|f]
  - **Fig 3 구조 (2026-02-24)**: 3a(m6A quartile violin) + 3b(scatter) + 3c(triple heatmap). [a|b]/[c centered]
  - **Fig 4 구조 (2026-03-06)**: [a|b]/[c|d] 2-row. 4a(Young vs Ancient feature comparison) + 4b(each feature confers immunity) + 4c(composite immunity score violin) + 4d(DRACH/CpG motif landscape). **Immunity framing**: "ancient L1 vulnerability" → "young L1 immunity features"
- **Narrative**: Part1(m6A + RNA004 validation) → Part2(arsenite selectivity + CHX + XRN1 + RNA-seq) → Part3(m6A-poly(A) protection) → Part4(immunity features) → Discussion(dual pathway + clinical implications)

### 남은 TODO
- 없음 (모든 TODO 해결 완료)

### GitHub Repositories (2026-03-16)
- **Public**: https://github.com/junsoopablo/L1-m6A-polyA (code + manuscript, preprint 공개용)
- **Private**: https://github.com/junsoopablo/L1-m6A-polyA-full (full project archive, 새 서버 재구축용)
- Public data는 GitHub에 포함 불가 → 새 서버에서 SRA 재다운로드 필요 (SETUP.md 참조)

### 완료 확인
- [x] ~~METTL3 KO validation~~ → **논문에서 제거** (2026-02-24). RNA004 dorado per-read m6A/kb로 대체 (Fig 1d)
- [x] m6A threshold 변경 0.50→0.80 → Part3 cache 57개 재생성, METTL3 KO sweep 근거
- [x] FigS8 (Ars+CHX rescue) → figS8.pdf 존재, legend 완성
- [x] FigS13 (RNA-seq validation) → figS13.pdf 3-panel, ancient L1 focus, legend 완성
- [x] Causal→correlation language revision → 완료
- [x] topic_10 RNA-seq validation → ancient L1 downregulation 확인, null results 미수록
- [x] **thr=204 figure 전면 재생성** (2026-02-20): 중간 TSV 재생성 + fig1-4/figS1-13 전부 완료
  - 주요 변경: L1 m6A/kb 2.572, Ctrl 1.527, ratio 1.684; OLS stress×m6A P=8.59e-05
  - ~~METTL3 KO (Fig 1e)~~ → 삭제, RNA004 validation (Fig 1d)로 대체
  - Fig S5: L1-internal m6A/kb 사용, UTR 영역 부족 명시
  - Fig S6: qcut rank fallback 추가 (duplicate bin edges)
- [x] **Fig 2 restructure + Het 제거** (2026-02-20)
  - Fig 2e: CHX rescue violin (S8에서 승격). Fig 2f: XRN1 KD bar (pathway convergence)
  - PAS panel 제거 (본문에서 충분). Het 제거: fig2d/figS6에서 Heterochromatin (n=28)
- [x] **Supplementary S4(DDR), S11(HepG2 LTR12C) 제거 + 번호 재매핑** (2026-02-20)
  - S15→S13 (물리 파일명 유지, section 제목만 변경)
- [x] **Fig S14 (L1 3' positioning & PAS vulnerability)** (2026-02-23)
  - L1 at 3' end (own PAS): Δ=-21nt. L1 upstream (downstream PAS): Δ=-41nt (2× stronger)
  - Young 62% at 3' end vs Ancient 35% → Young immunity 부분 설명
- [x] **Dual pathway independence 분석 + 논문 반영** (2026-02-20)
  - `xrn1_vs_m6a_independence.py`: OR=0.99, m6A uniform across conditions, per-read r=0.044
  - `pathway_subgroup_analysis.py`: intronic≈intergenic r, regulatory 3' 최취약, young immune
  - Discussion: XRN1 modification-blind (Athapattu2021), quantity vs quality control, 3p_only m6A/kb=4.08
  - references.bib: Athapattu2021 추가 (구 Wulf2021 → 저자 완전 오류 발견, 교체)
- [x] **가상 Peer Review 대응 Revision** (2026-02-20): Phase A-C 완료
  - A1: bib 5건 수정 (LinZ2025/ChenC2021ythdc1 삭제, Lima2017 저자 교체, Wulf2021→Athapattu2021, Loman→Simpson key)
  - A2: OLS β=4.44, P=8.59e-5, R²=0.041 통일. Read count 54,234/38,544 관계 명시
  - B1: METTL3 KO poly(A) L1-specific (Ctrl Δ+0.6nt ns, L1 Δ+13.1nt P=4.3e-4)
  - B2: R²=4.10%, partial R² stress×m6A=0.31%
  - B3: 5 threshold ALL P<2e-4 (Table S5)
  - B4: Decay zone Q1/Q4 all thresholds sig (Table S6)
  - C1-C6: METTL3 해석 강화, selectivity narrative, XRN1 batch, limitations, 인용 8+추가, PAS 반박
  - 상세: `memory/revision_progress.md`
- [x] **원고 proofreading + 숫자 일관성 검증** (2026-02-20)
  - main.tex 9개 이슈 수정: #1 PAS Δ 텍스트-범례 불일치(−47.7→−35), #3 근거없는 locus 재현성 주장 삭제, #4 Bhatt2025 인용 추가, #5 run-on 문장 분리, #6 LaTeX subscript, #7 Abstract 축약, #8 Intro 전환 개선, #9 unused \m 매크로 삭제
  - **S6 범례 E066→E117 업데이트** (figure는 재생성 완료였으나 범례 텍스트만 구값 잔존): S6a Δ=-69→-73, S6b 37-44%→44-48%, S6c 36-57→49-84nt, S6d Q4=102→59nt
  - **main.tex S6d 참조 수정**: Q4=90→59, Q1=32→28, P=0.024→2.9e-4
  - **Decay zone 숫자 Table S6 일치화**: Q4 14.4→15.3%, 2.1→2.0x, P 7.7e-13→5.9e-11
- [x] **Fig S15 (lncRNA control) + 논문 반영 + 용어 통일** (2026-02-23)
  - Results Part 2: lncRNA Δ=-2.9nt, m6A 1.59 vs 2.02, ρ=0.065 vs 0.201
  - Methods: lncRNA pipeline (GENCODE v38, 6 libraries)
  - Supplementary: S15 4-panel figure + legend
  - 용어 통일: "control transcripts/reads" → "non-L1 transcripts" 전면 교체
  - "non-L1 transcripts" 첫 등장에 조성 명시 (93.9% mRNA, 3.2% MT-rRNA, 1.7% lncRNA)
  - "non-L1 non-coding reads" → "non-L1 genomic fragments" (lncRNA와 구분)
- [x] ~~METTL3 KO thr=204 통일 + body/flank 분석~~ (2026-02-23) → **논문에서 제거** (2026-02-24)
  - 분석 자체는 완료: body/flank NS, pooled FC=0.98x, DRACH 차이 미미
  - Files 유지: `topic_05_cellline/mettl3ko_body_flanking/`, `mettl3ko_motif_analysis/`
- [x] **12개 리뷰 항목 전면 반영** (2026-02-23)
  - #1 Fig 3 legend 2.0x/5.9e-11 수정. #3 Psi 부정 결과 추가 (1.03x ns)
  - #4 Part 2 → 3 subsubsections 분할. #5 Part 1→2 전환 문장. #6 Implications 축약
  - #7 Causal language 수정. #8 Guppy v6.0.0/psi-co-mAFiA v0.0.1. #9 ChromHMM Methods
  - #10 Abstract 강화. #11 +13.1nt 반복 축약. #12 3' positioning 간결화
  - Fig 3d legend 136/47/89nt 통일. Fig 3c P=4.3e-4 통일. ≥3 reads 통일
  - Spearman rho 0.201/3.3e-26 통일 (main + supplementary)
  - Discussion Fig.~2f→2e 수정. 컴파일 0 errors (30pp + 23pp)
- [x] **Per-locus aggregation + subgroup R² 분석** (2026-02-24)
  - Subgroup R²: baseline 1.58% → stressed ancient regulatory 7.29% (4.6x)
  - Per-locus (stressed ancient ≥5 reads): rho=0.513, R²=12%, WLS R²=17.6% (11x vs baseline)
  - Decay zone OR=0.85 (m6A/kb +1 → 분해 확률 15%↓)
  - Negative controls: Young ns (P=0.31), Unstressed R²=0.3%
  - 논문 반영: Results Part 3 per-locus 문단, Discussion R² limitation 개선, Methods 추가, Table S7
  - Files: `topic_05_cellline/subgroup_m6a_r2/`
- [x] **Dorado RNA004 all-context m6A 분석** (2026-02-24)
  - dorado sup@v5.2.0 + inosine_m6A_2OmeA@v1 → HeLa RNA004 5.89M reads
  - **METTL16 가설 기각**: canonical motif 20/715K (0.003%), core motif non-L1가 더 높음
  - **L1 DRACH m6A rate 3.85% vs non-L1 1.15% (3.3x)**: L1은 DRACH-dominant
  - Young L1 DRACH rate 10.9% (Ancient 3.9%의 2.8x) → DRACH motif 진화적 보존
  - DRACH/non-DRACH ratio: L1 3.04x vs non-L1 0.63x
  - **Ancient L1 motif/locus 분석**: Ancient non-DRACH = DRACH motif 진화적 퇴화 (METTL16 아님)
  - Position-reproducible m6A: 539/51,144 sites. **93.1% DRACH** (non-DRACH 37개, 0.08%)
  - Stem-loop enrichment at non-DRACH: ~10% vs background ~9% (ns) → METTL16 최종 기각
  - **결론: L1 m6A = METTL3/DRACH 단일 경로**
  - BAM: `/blaze/junsoopablo/dorado_validation/HeLa_1_1_m6A/`
  - Files: `topic_06_dorado_validation/dorado_m6a_results/`, `ancient_l1_motif_locus.py`
- [x] **METTL3 KO 논문에서 제거 + RNA004 dorado per-read m6A/kb validation** (2026-02-24)
  - Survivorship bias 가설 기각: KO에서 L1 read count FC=1.15 (↑, not ↓). Wilcoxon P=0.49
  - Per-read m6A/kb (dorado A+a): L1 6.24 vs non-L1 4.93 (overall 1.30x). Length-matched 2-5kb: **1.79x** (MAFIA 일치)
  - Young 11.2 > Ancient 6.2 > non-L1 4.9. Young/Ancient 1.81x (MAFIA 1.55x와 유사)
  - DRACH motif hierarchy: L1 vs non-L1 rho=0.909 (P=1.7e-7). GGACT #1 양쪽 동일
  - **Reverse-strand MM/ML parsing bug 발견**: `A+a` tag에서 reverse read → skip는 T position 기준. 이전 분석 (#54-55) 일부 영향
  - Fig 1d: RNA004 per-read m6A/kb violin + 93% DRACH donut. Fig 3: 4→3 panels (fig3d→fig3c)
  - main.tex 11곳 편집: METTL3 KO 전면 제거, RNA004 validation 삽입
  - supplementary.tex Table S5 caption 수정
  - compose_figures.py: fig3 [a|b]/[c centered] 레이아웃
  - Files: `topic_06_dorado_validation/fig1d_rna004_validation.py`, `dorado_m6a_per_read_validation.py`
- [x] **Young vs Ancient m6A decomposition + consensus hotspot** (2026-02-24)
  - Decomposition: DRACH density 1.41x × per-DRACH rate 1.64x = 2.32x (length-matched)
  - Hotspot: Ancient at Young-hotspot 11.65% vs non-hotspot 6.53% = 1.78x (P=5e-94)
  - Flanking: Hamming≤2 9.95% vs >2 4.98% = 2.00x (P=1.3e-96)
  - **Fig 1e/f로 승격** (scatter + Hamming bar). S16 삭제. compose_figures.py 3-row
  - Fig 1 legend에 (e)/(f) 추가. 본문 S16a→Fig 1e, S16b→Fig 1f
  - Luo2022 + Shachar2024 인용 추가
  - Fig 1e/f 프레이밍: "Ancient L1 m6A validation" (기존 Young>Ancient decomposition에서 변경)
  - Files: `topic_06_dorado_validation/{young_vs_ancient_m6a_decomposition,consensus_hotspot_m6a_analysis,fig1ef_consensus_hotspot}.py`
- [x] **논문 구조 정리 + 불필요 결과 축소** (2026-02-24)
  - Part 1: decomposition 문단 → "Ancient m6A is genuine" 프레이밍, cross-CL/positional/intergenic 상세 축소
  - Part 2: read-length independence 이동 (mechanistic → L1-specific 뒤), genome-wide extent 1문장화, HepG2 LTR12C 삭제
  - Part 3: m6A position (P=0.44) 3문장→1문장
  - Discussion: reader switch 50% 축소
  - 결과: 31pp → 30pp, 0 errors
- [x] **Discussion Clinical Implications 추가** (2026-02-26)
  - 노화/세포 노화를 primary narrative: L1 RNA→SUV39H1→heterochromatin 침식 (DellaValle2022)
  - METTL3 decline in aging brain → m6A-poly(A) QC 실패 → L1 축적 (Shafik2021 + DeCecco2019)
  - L1PA m6A → EP300/KAP1 → LTR trans-silencing (Zhu2025L1PA, Cell Stem Cell 2025)
  - C9ORF72-ALS/FTD: m6A hypo + SG dysfunction = dual pathway 손상 (Li2023C9ORF72)
  - 기존 cancer 문장 (DeCecco2019 오용) 제거 → senescence로 정확히 재배치
  - references.bib +4 entries. 컴파일 28pp, 0 errors
- [x] **Fig 4 Immunity Framing 전면 개편** (2026-03-06)
  - "Ancient L1 vulnerability" → "Young L1 immunity features" 프레이밍 변경
  - Fig 4: [a|b]/[c|d]. 4a(Young vs Ancient 5-feature bar), 4b(each feature confers immunity), 4c(composite immunity score violin), 4d(DRACH/CpG motif landscape)
  - Young vs Ancient 차이: m6A/kb 1.8x, full-length 19.9x, EN domain 4.0x, consensus span 8.8x, 3'UTR 1.6x
  - Composite score 0→3+: 66→137nt (rho=0.169, P=5.5e-18). Stress-specific
  - 3'UTR coverage 단독으로는 보호 불가 (Δ=-24.4nt, ***)
  - Results/Discussion/Fig legend 전면 재작성. 구 mutation sensitivity map → Fig S16
  - Files: `topic_08_sequence_features/fig4_immunity_features.py`, `young_l1_immunity_features.py`
  - 컴파일: main 31pp 0 errors, supplementary 25pp 0 errors
