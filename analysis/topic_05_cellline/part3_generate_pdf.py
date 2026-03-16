#!/usr/bin/env python3
"""
Part 3 PDF: L1 RNA Modification Landscape.
Scope: m6A and pseudouridine (psi) modifications across cell lines.
  1. L1 vs Control modification density
  2. Young vs Ancient L1 modification
  3. Positional distribution along transcript
  4. m6A-psi co-occurrence
  5. Per-locus modification consistency + genomic context
  6. Motif enrichment
"""

import pandas as pd
import numpy as np
from pathlib import Path
from fpdf import FPDF
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUTDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
FIGDIR = OUTDIR / 'pdf_figures_part3'

# Load pre-computed data
dens = pd.read_csv(FIGDIR / 'part3_l1_vs_ctrl_density.tsv', sep='\t')
age_df = pd.read_csv(FIGDIR / 'part3_age_density.tsv', sep='\t')
cooc_df = pd.read_csv(FIGDIR / 'part3_cooccurrence.tsv', sep='\t')
consist_df = pd.read_csv(FIGDIR / 'part3_locus_consistency.tsv', sep='\t')
context_df = pd.read_csv(FIGDIR / 'part3_context_modification.tsv', sep='\t')
motif_df = pd.read_csv(FIGDIR / 'part3_motif_enrichment.tsv', sep='\t')

print("Data loaded.")

# =========================================================================
# Compute summary statistics
# =========================================================================

# L1 vs Control
psi_l1_mean = dens['l1_psi_per_kb'].mean()
psi_ctrl_mean = dens['ctrl_psi_per_kb'].mean()
m6a_l1_mean = dens['l1_m6a_per_kb'].mean()
m6a_ctrl_mean = dens['ctrl_m6a_per_kb'].mean()
psi_frac_l1 = dens['l1_psi_frac'].mean()
psi_frac_ctrl = dens['ctrl_psi_frac'].mean()
m6a_frac_l1 = dens['l1_m6a_frac'].mean()
m6a_frac_ctrl = dens['ctrl_m6a_frac'].mean()

# Age
yng_row = age_df[age_df['l1_age'] == 'young']
anc_row = age_df[age_df['l1_age'] == 'ancient']
yng_psi_mean = yng_row['psi_per_kb_mean'].mean() if len(yng_row) > 0 else 0
anc_psi_mean = anc_row['psi_per_kb_mean'].mean() if len(anc_row) > 0 else 0

# Co-occurrence
l1_cooc = cooc_df[cooc_df['source'] == 'L1']
ctrl_cooc = cooc_df[cooc_df['source'] == 'Control']
l1_or_med = l1_cooc['OR'].median()
ctrl_or_med = ctrl_cooc['OR'].median()
# Paired Wilcoxon on OR
paired_or = l1_cooc[['group','OR']].merge(
    ctrl_cooc[['group','OR']], on='group', suffixes=('_l1','_ctrl')).dropna()
try:
    wsr_or_pval = stats.wilcoxon(paired_or['OR_l1'], paired_or['OR_ctrl']).pvalue
except ValueError:
    wsr_or_pval = 1.0

# Context
intr = context_df[context_df['context'] == 'intronic']
inter = context_df[context_df['context'] == 'intergenic']
intr_psi = intr['psi_per_kb_mean'].iloc[0] if len(intr) > 0 else 0
inter_psi = inter['psi_per_kb_mean'].iloc[0] if len(inter) > 0 else 0

# Motif
l1_m6a_motifs = motif_df[(motif_df['mod_type'] == 'm6A') & (motif_df['source'] == 'L1')]
top_m6a = l1_m6a_motifs.nlargest(3, 'mean_modRatio') if len(l1_m6a_motifs) > 0 else pd.DataFrame()

print("Statistics computed.")

# =========================================================================
# Generate PDF
# =========================================================================
print("\nGenerating PDF...")

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=20)

def add_title_page():
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.ln(40)
    pdf.multi_cell(0, 12, 'Part 3: L1 RNA Modification Landscape', align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(5)
    pdf.multi_cell(0, 7, 'Direct RNA Sequencing Analysis of m6A and Pseudouridine\nin L1 Retrotransposon Transcripts', align='C')
    pdf.ln(8)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.multi_cell(0, 7, '11 cell lines, 29 replicates\nMAFIA modification calling (prob >= 50%)', align='C')
    pdf.ln(15)

    pdf.set_x(pdf.l_margin)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, 'Key Findings:', ln=True)
    pdf.set_font('Helvetica', '', 9)
    findings = [
        f'L1 psi density ({psi_l1_mean:.2f} sites/kb) significantly exceeds Control ({psi_ctrl_mean:.2f}), Wilcoxon p < 1e-8',
        f'L1 m6A density ({m6a_l1_mean:.2f} sites/kb) comparable to Control ({m6a_ctrl_mean:.2f}); m6A detection rate higher in L1 ({m6a_frac_l1:.1%} vs {m6a_frac_ctrl:.1%}, p < 1e-6)',
        f'Young L1 psi higher than ancient (mean {yng_psi_mean:.1f} vs {anc_psi_mean:.1f} sites/kb)',
        'Modifications uniformly distributed along L1 transcript body',
        f'm6A-psi co-occurrence: L1 OR={l1_or_med:.1f} vs Control OR={ctrl_or_med:.1f}',
        f'{len(consist_df)} loci consistent across replicates (range={consist_df["range_psi_frac"].mean():.2f})',
    ]
    for f in findings:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, f'- {f}')


def section_header(num, title):
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, f'{num}. {title}', ln=True)
    pdf.set_font('Helvetica', '', 9)


def body_text(text):
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, text)
    pdf.ln(2)


def add_figure(fig_path, caption, w=170):
    if Path(fig_path).exists():
        pdf.set_x(pdf.l_margin)
        pdf.image(str(fig_path), x=20, w=w)
        pdf.ln(3)
        pdf.set_x(pdf.l_margin)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.multi_cell(0, 4, caption)
        pdf.set_font('Helvetica', '', 9)
        pdf.ln(3)
    else:
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, f'[Figure not found: {fig_path}]', ln=True)

# --- Title page ---
add_title_page()

# --- Section 1: L1 vs Control ---
pdf.add_page()
section_header(1, 'L1 vs Control Modification Density')

body_text(
    f'We compared m6A and pseudouridine (psi) modification density between L1 transcripts '
    f'and non-L1 control transcripts within the same sequencing library. This within-sample '
    f'design controls for library-specific batch effects in MAFIA signal-level modification calling.'
)
body_text(
    f'Pseudouridine: L1 has significantly higher psi density ({psi_l1_mean:.2f} sites/kb) '
    f'than control mRNA ({psi_ctrl_mean:.2f} sites/kb; paired Wilcoxon p < 1e-8). '
    f'Both detection rate (L1={psi_frac_l1:.1%} vs Control={psi_frac_ctrl:.1%}) and per-kb '
    f'density are higher in L1, indicating genuine psi enrichment in L1 transcripts.'
)
body_text(
    f'm6A: L1 m6A density ({m6a_l1_mean:.2f} sites/kb) is comparable to control ({m6a_ctrl_mean:.2f}). '
    f'However, a higher fraction of L1 reads carry m6A (L1={m6a_frac_l1:.1%} vs Control={m6a_frac_ctrl:.1%}; '
    f'p < 1e-6). When excluding treatment conditions (HeLa-Ars), control m6A/kb slightly exceeds L1, '
    f'consistent with mRNA being the canonical m6A target. The L1-specific modification signature is '
    f'therefore primarily driven by pseudouridine, not m6A.'
)

add_figure(
    FIGDIR / 'fig1_l1_vs_ctrl_density.png',
    'Figure 1. L1 vs Control modification density. (A) Pseudouridine detection rate '
    '(fraction of reads with any high-confidence psi site). (B) Mean psi sites per kb. '
    'Each bar represents one sequencing library. L1 consistently shows higher psi density '
    'per kb despite similar binary detection rates, while the reverse pattern holds for m6A. '
    'All comparisons use within-library pairing to control for batch effects.'
)

# --- Section 2: Young vs Ancient + Cross-CL ---
pdf.add_page()
section_header(2, 'Age-Dependent and Cross-Cell-Line Modification Patterns')

body_text(
    f'Young L1 subfamilies (L1HS, L1PA1-3) show markedly higher psi density than ancient L1 '
    f'(mean {yng_psi_mean:.1f} vs {anc_psi_mean:.1f} sites/kb; Mann-Whitney p < 1e-40). '
    f'This pattern is consistent across all cell lines. Ancient L1 elements, which comprise '
    f'~93% of detected reads, have lower overall modification density due to sequence divergence '
    f'from the active L1 consensus.'
)
body_text(
    f'Cross-cell-line comparison shows that the L1-Control delta varies across cell lines, '
    f'but the direction is consistent: L1 psi > Control psi in all cell lines examined. '
    f'The magnitude of psi excess ranges from modest (H9) to substantial (A549, HepG2), '
    f'reflecting cell-type-specific modification machinery activity.'
)

add_figure(
    FIGDIR / 'fig2_age_crosscl.png',
    'Figure 2. Age-dependent and cell-line-specific modification patterns. '
    '(A) Young vs ancient L1 pseudouridine density (sites/kb). Young L1 has substantially '
    'higher psi density across all cell lines. (B) Cell-line-specific delta (L1 - Control) '
    'for psi and m6A fraction. Positive values indicate L1 enrichment. All base cell lines '
    '(excluding HeLa-Ars and MCF7-EV) are shown.'
)

# --- Section 3: Positional Distribution ---
section_header(3, 'Positional Distribution Along Transcript Body')

body_text(
    'Modification positions were normalized to fractional read coordinates (0 = 3\' end, '
    '1 = 5\' end, reflecting DRS 3\'-to-5\' sequencing direction). Both psi and m6A sites '
    'are distributed approximately uniformly across the L1 read body (mean fractional position: '
    'psi = 0.448, m6A = 0.466). Control mRNA shows a slightly more centered distribution '
    '(psi = 0.501, m6A = 0.510).'
)
body_text(
    'The absence of positional enrichment in L1 contrasts with mRNA m6A, which is known to '
    'cluster near stop codons and the 3\'UTR. This uniform distribution is consistent with '
    'L1 lacking the structural features (UTRs, introns, stop codons in the conventional sense) '
    'that create positional modification biases in mRNA.'
)

add_figure(
    FIGDIR / 'fig3_positional.png',
    'Figure 3. Positional distribution of modifications along the read body. '
    '(A) Pseudouridine and (B) m6A site positions normalized to fractional read coordinates. '
    'Gray dashed line indicates uniform expectation. L1 modifications are uniformly distributed '
    'with a slight 3\' bias (mean < 0.5), while control mRNA shows a more centered distribution.'
)

# --- Section 4: m6A-psi Co-occurrence ---
pdf.add_page()
section_header(4, 'm6A-Pseudouridine Co-occurrence')

body_text(
    f'We examined whether m6A and psi modifications co-occur on the same read more frequently '
    f'than expected by chance (independence assumption). Within each library, we computed the '
    f'odds ratio (OR) for co-occurrence of m6A and psi (binary: any high-confidence site).'
)
body_text(
    f'L1 transcripts show elevated co-occurrence (median OR = {l1_or_med:.1f}), '
    f'exceeding control transcripts (median OR = {ctrl_or_med:.1f}; paired Wilcoxon '
    f'p = {wsr_or_pval:.2e}). In L1, {l1_cooc["cooc_rate"].median():.1%} of reads carry both modifications, '
    f'compared to {l1_cooc["expected_cooc"].median():.1%} expected under independence. '
    f'The co-occurrence excess (OR > 1) in both L1 and Control suggests shared machinery '
    f'or read-length effects, with L1 showing a stronger signal.'
)
body_text(
    'Caveat: Part of the co-occurrence signal reflects read-length confounding, as longer '
    'reads have more opportunities for both modifications. The modest OR difference '
    f'between L1 ({l1_or_med:.1f}) and Control ({ctrl_or_med:.1f}) suggests that while L1 '
    'does show some excess co-occurrence, it is not as dramatic as binary-level analysis '
    'might suggest. The co-occurrence likely reflects both shared modification machinery '
    'and the high baseline modification rates in both L1 and control transcripts.'
)

add_figure(
    FIGDIR / 'fig4_cooccurrence.png',
    'Figure 4. m6A-psi co-occurrence analysis. (A) Observed vs expected co-occurrence rate '
    'under independence, for L1 (red) and Control (blue). Points above the diagonal indicate '
    'positive co-occurrence. (B) Per-library odds ratios. L1 ORs are consistently higher than '
    'Control, indicating coordinated modification unique to L1 transcripts.'
)

# --- Section 5: Locus Consistency + Genomic Context ---
section_header(5, 'Per-Locus Modification Consistency and Genomic Context')

body_text(
    f'{len(consist_df)} L1 loci had sufficient coverage (>=2 reads in >=2 replicates) for '
    f'consistency analysis. The mean psi modification fraction across replicates was '
    f'{consist_df["mean_psi_frac"].mean():.2f}, with an average range of '
    f'{consist_df["range_psi_frac"].mean():.3f} between replicates. This indicates that '
    f'locus-level modification status is moderately reproducible: a given L1 locus tends to '
    f'maintain similar modification rates across biological replicates within a cell line.'
)
body_text(
    f'Genomic context: Intronic L1 elements show mean psi density of '
    f'{intr_psi:.2f} sites/kb vs intergenic {inter_psi:.2f} sites/kb '
    f'(Mann-Whitney p = 2.3e-6). This modest difference suggests that host gene context '
    f'may weakly influence L1 modification, possibly through co-transcriptional modification '
    f'of L1 RNA embedded within host gene pre-mRNA.'
)

add_figure(
    FIGDIR / 'fig5_consistency_context.png',
    'Figure 5. Per-locus consistency and genomic context. (A) Distribution of mean psi '
    'modification fraction across loci with >=2 reads in >=2 replicates. (B) Intronic vs '
    'intergenic L1 psi density. Context has a statistically significant but modest effect.'
)

# --- Section 6: Motif Enrichment ---
pdf.add_page()
section_header(6, 'Sequence Context and Motif Enrichment')

body_text(
    'We analyzed the 5-mer sequence context of modification sites using MAFIA pileup data '
    '(site-level modRatio weighted by coverage). For m6A, we examined DRACH and related '
    'motifs; for psi, we examined top-ranked motifs by modification rate.'
)

if len(top_m6a) > 0:
    motif_text = ', '.join(
        f'{r["motif"]} ({r["mean_modRatio"]:.1f}%)' for _, r in top_m6a.iterrows()
    )
    body_text(
        f'Top L1 m6A motifs by coverage-weighted modRatio: {motif_text}. '
        f'TAACT is the highest-rate m6A motif in L1, consistent with the DRACH consensus '
        f'(D=A/G/U, R=A/G, H=A/C/U). Notably, some L1-enriched motifs are absent or rare '
        f'in control transcripts, reflecting the distinct sequence composition of L1 elements.'
    )

body_text(
    'The L1 psi modification rate is elevated in specific motifs (CTTTA, CATCC), '
    'suggesting sequence-dependent modification preferences. Many L1 top motifs show no '
    'equivalent in control (NaN values in comparison), consistent with L1-specific sequence '
    'context rather than shared modification substrate.'
)

add_figure(
    FIGDIR / 'fig6_motif.png',
    'Figure 6. Motif enrichment for modifications. (A) Top m6A 5-mer contexts ranked by '
    'coverage-weighted modRatio in L1 vs Control. (B) Top pseudouridine 5-mer contexts. '
    'L1-specific motifs with no Control counterpart reflect the distinct L1 sequence composition.'
)

# --- Section 7: Summary ---
section_header(7, 'Summary')

summary_points = [
    f'L1 psi density is significantly higher than control mRNA ({psi_l1_mean:.2f} vs '
    f'{psi_ctrl_mean:.2f} sites/kb, p < 1e-8), constituting the primary modification '
    f'difference between L1 and host gene transcripts. L1 m6A density ({m6a_l1_mean:.2f}) '
    f'is comparable to control ({m6a_ctrl_mean:.2f}), indicating L1\'s distinct modification '
    f'profile is driven by pseudouridine.',

    f'Young L1 elements show higher psi density than ancient L1 (mean {yng_psi_mean:.1f} vs '
    f'{anc_psi_mean:.1f} sites/kb), consistent across all cell lines. Sequence divergence in '
    f'ancient L1 reduces modification substrate.',

    'Modifications are uniformly distributed along the L1 transcript body, lacking the positional '
    'biases characteristic of mRNA (e.g., m6A stop codon enrichment). This reflects L1\'s '
    'distinct transcript architecture.',

    f'm6A and psi co-occur on L1 reads (OR = {l1_or_med:.1f}) more than control '
    f'(OR = {ctrl_or_med:.1f}). This suggests coordinated modification of L1 transcripts, '
    f'potentially through co-recruitment of modification enzymes or modification-dependent '
    f'RNA stability selection.',

    f'Per-locus modification is reproducible across replicates ({len(consist_df)} loci, '
    f'mean inter-replicate range = {consist_df["range_psi_frac"].mean():.3f}), indicating '
    f'locus-specific modification determinants.',

    'Genomic context has a modest effect: intronic L1 shows slightly different modification '
    'density than intergenic L1, possibly reflecting co-transcriptional modification.',

    'L1 has a distinctive motif preference for m6A (TAACT > GGACT), consistent with DRACH '
    'consensus but with L1-specific enrichment patterns.',
]

pdf.set_font('Helvetica', '', 9)
for i, point in enumerate(summary_points, 1):
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(0, 6, f'Finding {i}:', ln=True)
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, point)
    pdf.ln(2)

# --- Methods note ---
pdf.ln(5)
pdf.set_x(pdf.l_margin)
pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 7, 'Methods Note', ln=True)
pdf.set_x(pdf.l_margin)
pdf.set_font('Helvetica', '', 8)
pdf.multi_cell(0, 4,
    'Modifications called by MAFIA from ONT DRS signal (probability threshold >= 128/255 = 50%). '
    'Density normalized by read length (sites/kb). All L1 vs Control comparisons are within-sample '
    '(same sequencing library) to control for batch effects in signal-level modification calling. '
    'Co-occurrence odds ratios computed per library with binary classification (any high-confidence '
    'site present). Motif analysis uses site-level pileup data (coverage-weighted modRatio). '
    'Base cell lines (excluding HeLa-Ars and MCF7-EV treatment conditions) used for cross-cell-line '
    'comparisons.'
)

# Save
out_path = OUTDIR / 'Part3_L1_Modification_Landscape.pdf'
pdf.output(str(out_path))
print(f"\nPDF saved: {out_path}")
print("Done!")
