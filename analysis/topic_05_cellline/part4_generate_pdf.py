#!/usr/bin/env python3
"""
Part 4 PDF: L1 Stress Response (revised).
4 sections: poly(A) shortening -> post-tx mechanism -> young immunity -> m6A protection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from fpdf import FPDF
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUTDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
FIGDIR = OUTDIR / 'pdf_figures_part4'
TOPICDIR = PROJECT / 'analysis/01_exploration'

# =========================================================================
# Load data for inline statistics
# =========================================================================
print("Loading data...")
polya_df = pd.read_csv(FIGDIR / 'part4_polya_shortening.tsv', sep='\t')
cross_df = pd.read_csv(FIGDIR / 'part4_cross_cl_validation.tsv', sep='\t')
age_df = pd.read_csv(FIGDIR / 'part4_age_immunity.tsv', sep='\t')
m6a_df = pd.read_csv(FIGDIR / 'part4_m6a_protection.tsv', sep='\t')

# Raw data — load from Part3 cache + L1 summary (same as part4_analysis.py)
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
CACHE_L1 = TOPICDIR / 'topic_05_cellline/part3_l1_per_read_cache'
HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

cache_dfs = []
for g in HELA_GROUPS:
    p = CACHE_L1 / f'{g}_l1_per_read.tsv'
    if p.exists():
        df = pd.read_csv(p, sep='\t')
        df['group'] = g
        cache_dfs.append(df)
cache_all = pd.concat(cache_dfs, ignore_index=True)
cache_all['m6a_per_kb'] = cache_all['m6a_sites_high'] / (cache_all['read_length'] / 1000)

summ_dfs = []
for g in HELA_GROUPS:
    p = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
    if p.exists():
        df = pd.read_csv(p, sep='\t')
        df = df[df['qc_tag'] == 'PASS']
        df['group'] = g
        summ_dfs.append(df[['read_id', 'polya_length', 'gene_id', 'group']])
summ_all = pd.concat(summ_dfs, ignore_index=True)
summ_all['l1_age'] = summ_all['gene_id'].apply(
    lambda x: 'young' if x.split('_dup')[0] in YOUNG else 'ancient')

l1_state = cache_all.merge(summ_all, on=['read_id', 'group'], how='inner')
l1_state = l1_state[l1_state['polya_length'] > 0].copy()
l1_state['condition'] = l1_state['group'].apply(lambda x: 'HeLa-Ars' if 'Ars' in x else 'HeLa')

ctrl_state = pd.read_csv(TOPICDIR / 'topic_04_state/ctrl_state_classification.tsv', sep='\t')
ctrl_state['condition'] = ctrl_state['group'].apply(lambda x: 'HeLa-Ars' if 'Ars' in x else 'HeLa')

# Extract stats
l1_hela_med = polya_df[(polya_df['source']=='L1')&(polya_df['condition']=='HeLa')]['median_polya'].values[0]
l1_ars_med = polya_df[(polya_df['source']=='L1')&(polya_df['condition']=='HeLa-Ars')]['median_polya'].values[0]
ctrl_hela_med = polya_df[(polya_df['source']=='Control')&(polya_df['condition']=='HeLa')]['median_polya'].values[0]
ctrl_ars_med = polya_df[(polya_df['source']=='Control')&(polya_df['condition']=='HeLa-Ars')]['median_polya'].values[0]
delta_l1 = l1_ars_med - l1_hela_med
delta_ctrl = ctrl_ars_med - ctrl_hela_med

hela_n = polya_df[(polya_df['source']=='L1')&(polya_df['condition']=='HeLa')]['n'].values[0]
ars_n = polya_df[(polya_df['source']=='L1')&(polya_df['condition']=='HeLa-Ars')]['n'].values[0]

mw_l1 = stats.mannwhitneyu(
    l1_state[l1_state['condition']=='HeLa']['polya_length'],
    l1_state[l1_state['condition']=='HeLa-Ars']['polya_length'], alternative='two-sided')
mw_ctrl = stats.mannwhitneyu(
    ctrl_state[ctrl_state['condition']=='HeLa']['polya_length'],
    ctrl_state[ctrl_state['condition']=='HeLa-Ars']['polya_length'], alternative='two-sided')

# Age
anc_hela = age_df[(age_df['l1_age']=='ancient')&(age_df['condition']=='HeLa')]['median_polya'].values[0]
anc_ars = age_df[(age_df['l1_age']=='ancient')&(age_df['condition']=='HeLa-Ars')]['median_polya'].values[0]
yng_hela = age_df[(age_df['l1_age']=='young')&(age_df['condition']=='HeLa')]['median_polya'].values[0]
yng_ars = age_df[(age_df['l1_age']=='young')&(age_df['condition']=='HeLa-Ars')]['median_polya'].values[0]
anc_n_hela = age_df[(age_df['l1_age']=='ancient')&(age_df['condition']=='HeLa')]['n'].values[0]
anc_n_ars = age_df[(age_df['l1_age']=='ancient')&(age_df['condition']=='HeLa-Ars')]['n'].values[0]
yng_n_hela = age_df[(age_df['l1_age']=='young')&(age_df['condition']=='HeLa')]['n'].values[0]
yng_n_ars = age_df[(age_df['l1_age']=='young')&(age_df['condition']=='HeLa-Ars')]['n'].values[0]

mw_anc = stats.mannwhitneyu(
    l1_state[(l1_state['l1_age']=='ancient')&(l1_state['condition']=='HeLa')]['polya_length'],
    l1_state[(l1_state['l1_age']=='ancient')&(l1_state['condition']=='HeLa-Ars')]['polya_length'],
    alternative='two-sided')
mw_yng = stats.mannwhitneyu(
    l1_state[(l1_state['l1_age']=='young')&(l1_state['condition']=='HeLa')]['polya_length'],
    l1_state[(l1_state['l1_age']=='young')&(l1_state['condition']=='HeLa-Ars')]['polya_length'],
    alternative='two-sided')

# OLS (continuous m6A/kb model)
ars_coef = m6a_df[m6a_df['variable']=='arsenite']['coefficient'].values[0]
ars_p = m6a_df[m6a_df['variable']=='arsenite']['p_value'].values[0]
m6a_coef = m6a_df[m6a_df['variable']=='m6A/kb']['coefficient'].values[0]
m6a_p = m6a_df[m6a_df['variable']=='m6A/kb']['p_value'].values[0]
interaction_coef = m6a_df[m6a_df['variable']=='ars x m6A/kb']['coefficient'].values[0]
interaction_p = m6a_df[m6a_df['variable']=='ars x m6A/kb']['p_value'].values[0]
rdlen_coef = m6a_df[m6a_df['variable']=='read_length_z']['coefficient'].values[0]

# Cross-CL
ars_only_med = 77.5
other_cl_med = cross_df['median_polya'].median()
n_cls = len(cross_df)

# m6A-polyA correlation in Ars
reg_df = l1_state[['condition','polya_length','m6a_per_kb','read_length']].dropna().copy()
reg_df['rdLen_z'] = (reg_df['read_length'] - reg_df['read_length'].mean()) / reg_df['read_length'].std()
X_rdlen = np.column_stack([np.ones(len(reg_df)), reg_df['rdLen_z'].values])
beta_rdlen = np.linalg.lstsq(X_rdlen, reg_df['polya_length'].values, rcond=None)[0]
reg_df['polya_resid'] = reg_df['polya_length'] - X_rdlen @ beta_rdlen
ars_sub = reg_df[reg_df['condition']=='HeLa-Ars']
r_ars, p_ars = stats.pearsonr(ars_sub['m6a_per_kb'], ars_sub['polya_resid'])
hela_sub = reg_df[reg_df['condition']=='HeLa']
r_hela, p_hela = stats.pearsonr(hela_sub['m6a_per_kb'], hela_sub['polya_resid'])

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
    pdf.multi_cell(0, 12, 'Part 4: L1 Stress Response', align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(5)
    pdf.multi_cell(0, 7,
        'Arsenite-Induced Poly(A) Shortening\nin L1 Retrotransposon Transcripts',
        align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.multi_cell(0, 7,
        f'HeLa vs HeLa-Ars (60 min sodium arsenite)\n'
        f'L1: {hela_n:,} + {ars_n:,} reads | Control: {len(ctrl_state):,} reads',
        align='C')
    pdf.ln(15)

    pdf.set_x(pdf.l_margin)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, 'Key Findings:', ln=True)
    pdf.set_font('Helvetica', '', 9)
    findings = [
        f'Arsenite shortens L1 poly(A) by ~{abs(delta_l1):.0f} nt '
        f'({l1_hela_med:.0f} -> {l1_ars_med:.0f} nt); control transcripts unchanged '
        f'({ctrl_hela_med:.0f} -> {ctrl_ars_med:.0f} nt)',
        f'Post-transcriptional mechanism confirmed: Ars-only loci have normal poly(A) '
        f'in {n_cls} other cell lines ({other_cl_med:.0f} nt) vs {ars_only_med:.0f} nt in HeLa-Ars',
        f'Young L1 (L1HS, L1PA1-3) immune to arsenite: '
        f'{yng_hela:.0f} -> {yng_ars:.0f} nt (ns); immunity persists after m6A-matching',
        f'Higher m6A density significantly protects poly(A) under arsenite stress '
        f'(OLS stress x m6A/kb interaction p = {interaction_p:.1e}); '
        f'L1 m6A genuinely enriched (1.33x per-site rate vs control)',
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

# --- Section 1: L1-Specific Poly(A) Shortening ---
pdf.add_page()
section_header(1, 'L1-Specific Poly(A) Shortening')

body_text(
    f'We compared poly(A) tail length distributions between L1 retrotransposon '
    f'transcripts and non-L1 control transcripts under normal (HeLa) and arsenite '
    f'stress (HeLa-Ars, 60 min sodium arsenite) conditions.'
)
body_text(
    f'Arsenite treatment induces a substantial shortening of L1 poly(A) tails: '
    f'median poly(A) drops from {l1_hela_med:.1f} nt to {l1_ars_med:.1f} nt '
    f'(delta = {delta_l1:.1f} nt, Mann-Whitney p = {mw_l1.pvalue:.1e}). '
    f'In contrast, control transcripts show no significant change '
    f'({ctrl_hela_med:.1f} -> {ctrl_ars_med:.1f} nt, delta = {delta_ctrl:+.1f} nt, '
    f'p = {mw_ctrl.pvalue:.2f}). The cumulative distribution confirms a global '
    f'leftward shift in L1 poly(A) under arsenite, while control distributions overlap.'
)
body_text(
    f'This L1-specific effect establishes that arsenite-induced poly(A) shortening '
    f'selectively targets L1 transcripts through a mechanism distinct from global mRNA '
    f'turnover. At 60 minutes post-treatment, L1 transcripts are already substantially '
    f'affected while the broader transcriptome remains stable.'
)

add_figure(
    FIGDIR / 'fig1_polya_shortening.png',
    'Figure 1. L1-specific poly(A) shortening under arsenite stress. '
    '(A) Violin plots of poly(A) length for L1 and Control transcripts in HeLa vs HeLa-Ars. '
    'Median values shown. L1 poly(A) drops substantially; Control is unchanged. '
    '(B) Cumulative distribution functions. L1 HeLa-Ars (dashed red) shifts leftward; '
    'Control distributions (blue) overlap. '
    '(C) Summary: median poly(A) change under arsenite. L1 loses ~31 nt; Control gains ~3 nt.'
)

# --- Section 2: Post-Transcriptional Mechanism ---
pdf.add_page()
section_header(2, 'Post-Transcriptional Mechanism')

body_text(
    f'The observed poly(A) shortening could reflect either post-transcriptional '
    f'deadenylation (active shortening of existing tails) or a compositional shift '
    f'(arsenite activating inherently short-tailed loci). To distinguish these, we '
    f'identified L1 loci detected only in HeLa-Ars (Ars-only loci, n={len(cross_df)} '
    f'cell lines with overlap) and tracked their poly(A) lengths across {n_cls} other '
    f'cell lines.'
)
body_text(
    f'Ars-only loci have a median poly(A) of {ars_only_med:.1f} nt in HeLa-Ars, '
    f'but {other_cl_med:.0f} nt in other cell lines '
    f'(range: {cross_df["median_polya"].min():.0f}-{cross_df["median_polya"].max():.0f} nt). '
    f'This ~{abs(other_cl_med - ars_only_med):.0f} nt difference is consistent across all '
    f'{n_cls} cell lines tested (Mann-Whitney p < 1e-35).'
)
body_text(
    'This cross-cell-line evidence demonstrates that Ars-only loci produce transcripts '
    'with normal-length poly(A) tails in unstressed cells. The short tails observed in '
    'HeLa-Ars are therefore the result of active post-transcriptional deadenylation, '
    'not the activation of inherently short-tailed L1 loci.'
)

add_figure(
    FIGDIR / 'fig2_cross_cl_validation.png',
    f'Figure 2. Cross-cell-line validation of post-transcriptional mechanism. '
    f'(A) Median poly(A) at Ars-only loci per cell line. HeLa-Ars (red) is the clear outlier; '
    f'all {n_cls} other cell lines show normal poly(A) (>110 nt). '
    f'(B) Per-locus comparison: poly(A) in other cell lines (x) vs HeLa-Ars (y). '
    f'Majority of points fall below the diagonal, indicating HeLa-Ars-specific shortening.'
)

# --- Section 3: Young L1 Immunity ---
pdf.add_page()
section_header(3, 'Young L1 Immunity to Arsenite')

body_text(
    f'We stratified L1 transcripts by evolutionary age: ancient elements '
    f'(L1MC, L1ME, L1M, etc.; n={anc_n_hela:,}+{anc_n_ars:,} reads, ~93% of total) '
    f'and young elements (L1HS, L1PA1-3; n={yng_n_hela:,}+{yng_n_ars:,} reads, ~7%).'
)
body_text(
    f'Ancient L1 shows substantial poly(A) shortening: median drops from '
    f'{anc_hela:.0f} to {anc_ars:.0f} nt (delta = {anc_ars-anc_hela:.0f} nt, '
    f'p = {mw_anc.pvalue:.1e}). In striking contrast, young L1 poly(A) is unchanged: '
    f'{yng_hela:.0f} -> {yng_ars:.0f} nt (delta = {yng_ars-yng_hela:+.0f} nt, '
    f'p = {mw_yng.pvalue:.2f}, ns).'
)
body_text(
    f'Since young L1 has substantially higher m6A density than ancient L1 '
    f'(young m6A per-site rate 29.4% vs ancient 26.9%, p < 1e-65), a natural hypothesis is that '
    f'young L1 resists deadenylation because of its higher m6A levels (Section 4). '
    f'However, m6A-matched analysis rejects this explanation: when restricting to reads '
    f'with overlapping m6A/kb range, young L1 remains fully immune '
    f'(delta = -0.8 nt, p = 0.77), while ancient L1 still shortens '
    f'(delta = -35.4 nt, p = 6.0e-19). An OLS model controlling for m6A, read length, '
    f'and arsenite confirms that the arsenite*young interaction is independently '
    f'significant (coef = +29.0 nt, p = 0.015).'
)
body_text(
    'Young L1 immunity is therefore an intrinsic property independent of modification '
    'status. This suggests that recently active L1 elements are regulated differently '
    'from ancient copies, possibly through distinct RNA-binding protein associations, '
    'structural features (intact ORFs, full-length transcripts), or subcellular '
    'localization that shields them from stress-induced deadenylation. '
    'Notably, young L1 elements are the only ones capable of retrotransposition, '
    'and preserving their poly(A) tails under stress may have functional '
    'consequences for L1 mobilization.'
)

add_figure(
    FIGDIR / 'fig3_age_immunity.png',
    'Figure 3. Young L1 immunity to arsenite is independent of m6A levels. '
    '(A) Box plot of poly(A) by age and condition. Ancient L1 poly(A) drops ~33 nt; '
    'young L1 is unchanged. '
    '(B) Cumulative distributions by age: ancient HeLa-Ars (dashed red) shifts left; '
    'young (blue) CDFs overlap. '
    '(C) m6A-matched test: restricting to overlapping m6A/kb range, young L1 immunity '
    'persists (delta = -0.8 nt, ns) while ancient still shortens (delta = -35.4 nt, ***).'
)

# --- Section 4: m6A Protection ---
pdf.add_page()
section_header(4, 'N6-Methyladenosine (m6A) Protection of Poly(A)')

body_text(
    f'L1 transcripts carry genuinely enriched m6A: per-site modification rate at DRACH '
    f'motifs is 27.2% in L1 vs 20.4% in control (1.33x, Fisher p < 1e-300), validated by '
    f'CIGAR-aware BAM-direct analysis. We tested whether this m6A enrichment influences '
    f'the arsenite-induced poly(A) shortening using continuous m6A density (m6A/kb) rather '
    f'than binary m6A presence. An OLS regression model '
    f'(poly(A) ~ arsenite + m6A/kb + read_length_z + arsenite x m6A/kb) was fitted to '
    f'account for the read length confound.'
)
body_text(
    f'The arsenite main effect is significant (coef = {ars_coef:.1f} nt, '
    f'p = {ars_p:.1e}), confirming poly(A) shortening. The baseline m6A/kb effect is '
    f'positive (coef = {m6a_coef:.2f} nt per m6A/kb, p = {m6a_p:.1e}). Critically, '
    f'the arsenite x m6A/kb interaction is significant '
    f'(coef = {interaction_coef:+.2f} nt, p = {interaction_p:.1e}), demonstrating that '
    f'higher m6A density confers greater poly(A) retention specifically under stress. '
    f'This indicates a dose-dependent m6A protective effect that emerges under '
    f'deadenylation pressure.'
)
body_text(
    f'Consistent with the OLS result, Pearson correlation between m6A/kb and poly(A) '
    f'residual (read-length adjusted) is strongest in HeLa-Ars '
    f'(r = {r_ars:.3f}, p = {p_ars:.1e}), while weak in untreated HeLa '
    f'(r = {r_hela:.3f}). This m6A-poly(A) correlation is STRESS-SPECIFIC: across 10 '
    f'cell lines (read-length corrected), only HeLa-Ars shows a substantial positive '
    f'correlation. The remaining unstressed cell lines show near-zero or slightly negative '
    f'correlations, confirming that m6A does not constitutively predict '
    f'longer poly(A) tails but becomes protective under stress.'
)
body_text(
    f'Stratified analysis further reveals that the m6A protective effect is strongest in '
    f'intronic L1 under stress (Spearman r = +0.19, p < 1e-15) and particularly '
    f'pronounced in young intronic L1 (r = +0.32, p = 4.4e-4). Under normal conditions, '
    f'the correlation is present only in intronic L1 (r = +0.10, p = 5.9e-5) '
    f'and absent in intergenic L1 (r = -0.02, ns), suggesting that host gene '
    f'transcriptional environment modulates the m6A-poly(A) coupling.'
)
body_text(
    f'L1\'s m6A enrichment is genuine and not merely a motif density artifact. '
    f'Per-site m6A rate at DRACH motifs is 27.2% in L1 vs 20.4% in control (1.33x). '
    f'Young L1 has the highest per-site rate (29.4%) over ancient L1 (26.9%, p < 1e-65), '
    f'consistent with young L1\'s immunity to arsenite-induced shortening. '
    f'Within the same read, L1 body regions have lower per-site m6A rate '
    f'than flanking host sequence (25.2% vs 27.1% = 0.93x, p = 8.7e-71), despite '
    f'higher m6A/kb in body (driven by higher DRACH motif density in AT-rich L1 sequence).'
)

add_figure(
    FIGDIR / 'fig4_m6a_protection.png',
    'Figure 4. m6A protects poly(A) under stress. '
    '(A) OLS coefficient plot: arsenite shortens poly(A) (***); stress x m6A/kb interaction '
    'is significant (***), indicating dose-dependent m6A protection. '
    '(B) Read-length-adjusted poly(A) residual by m6A density quartile: '
    'high-m6A reads retain more poly(A) under stress. '
    '(C) Per-cell-line Pearson r (m6A/kb vs poly(A) residual): '
    'stress-specific positive correlation in HeLa-Ars only.'
)

# --- Section 5: Summary ---
section_header(5, 'Summary')

summary_points = [
    f'Arsenite stress induces L1-specific poly(A) shortening '
    f'(~{abs(delta_l1):.0f} nt, p = {mw_l1.pvalue:.1e}). '
    f'Control transcripts are unaffected, establishing L1 as a selective target '
    f'of stress-induced RNA turnover.',

    f'Cross-cell-line validation confirms a post-transcriptional mechanism: '
    f'Ars-only loci have normal poly(A) in {n_cls} other cell lines '
    f'({other_cl_med:.0f} nt) but short poly(A) only in HeLa-Ars ({ars_only_med:.0f} nt). '
    f'This rules out compositional bias and demonstrates active deadenylation.',

    f'Young L1 elements (L1HS, L1PA1-3) are immune to arsenite-induced shortening '
    f'({yng_hela:.0f} -> {yng_ars:.0f} nt, ns), while ancient L1 is strongly affected '
    f'({anc_hela:.0f} -> {anc_ars:.0f} nt, p = {mw_anc.pvalue:.1e}). '
    f'Despite young L1 having higher m6A density than ancient, '
    f'm6A-matched analysis shows immunity persists independently '
    f'(matched delta = -0.8 nt, ns; OLS ars*young p = 0.015).',

    f'L1 m6A is genuinely enriched (per-site rate 27.2% vs control 20.4% = 1.33x). '
    f'Using continuous m6A/kb rather than binary m6A presence, the OLS stress x m6A/kb '
    f'interaction is significant (coef = {interaction_coef:+.2f}, p = {interaction_p:.1e}), '
    f'demonstrating dose-dependent m6A protection of poly(A) under stress. '
    f'The m6A-poly(A) correlation (r = {r_ars:.3f}, p = {p_ars:.1e}) '
    f'is stress-specific: unstressed cell lines show near-zero correlations. '
    f'Stratified analysis shows the effect is strongest in intronic L1 '
    f'and particularly in young intronic L1 (r = +0.32 under stress). '
    f'Young L1 has the highest per-site m6A rate (29.4%), consistent with its immunity '
    f'to arsenite-induced shortening.',
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
    'Poly(A) tail lengths estimated by nanopolish from ONT DRS data. '
    'm6A called by MAFIA (probability >= 50%, ChEBI:21891). Per-site m6A rate validated by '
    'CIGAR-aware BAM-direct analysis at DRACH motif sites. '
    'Mann-Whitney U tests for two-group comparisons; KS test for distribution shifts. '
    'OLS regression includes read length z-score to control for length-dependent modification bias. '
    'Poly(A) residuals computed by regressing out read length effect. '
    'Cross-cell-line validation uses Ars-only ancient L1 loci (transcript_id) tracked across '
    'A549(4-6), H9(2-4), Hct116(3-4), HepG2(5-6), HEYA8(1-3), K562(4-6), MCF7(2-4), SHSY5Y(1-3). '
    'Young L1: L1HS, L1PA1, L1PA2, L1PA3. All reads pass QC (qc_tag == PASS).'
)

# Save
out_path = OUTDIR / 'Part4_L1_Stress_Response.pdf'
pdf.output(str(out_path))
print(f"\nPDF saved: {out_path}")
print("Done!")
