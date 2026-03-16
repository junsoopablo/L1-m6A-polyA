#!/usr/bin/env python3
"""
Histone mark overlap analysis for L1 loci.
Uses pre-computed bedtools intersect results (overlap_*.tsv).
"""

import pandas as pd
import numpy as np
from scipy import stats

# ── Paths ──
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration"
OUTDIR = f"{BASE}/topic_10_rnaseq_validation/histone_mark_analysis"
CHROMHMM_TSV = f"{BASE}/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv"

MARKS = ["H3K27ac", "H3K4me1", "H3K4me3"]

# ── Load L1 data ──
print("Loading L1 ChromHMM annotated data...")
df_all = pd.read_csv(CHROMHMM_TSV, sep='\t')

# Filter: HeLa + HeLa-Ars, ancient only
df = df_all[
    df_all['cellline'].isin(['HeLa', 'HeLa-Ars']) &
    (df_all['l1_age'] == 'ancient')
].copy()

df['cond'] = df['condition'].map({'normal': 'normal', 'stress': 'stress'})
print(f"  HeLa ancient L1 reads: normal={len(df[df.cond=='normal'])}, stress={len(df[df.cond=='stress'])}")

# ── Parse pre-computed overlap files ──
print("\nParsing bedtools intersect results...")
for mark_name in MARKS:
    overlap_file = f"{OUTDIR}/overlap_{mark_name}.tsv"
    overlaps = {}
    with open(overlap_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            read_id = parts[3]
            # narrowPeak: chr(4) start(5) end(6) name(7) score(8) strand(9) signalValue(10) pValue(11) qValue(12) peak(13)
            signal_value = float(parts[10])
            if read_id not in overlaps or signal_value > overlaps[read_id]:
                overlaps[read_id] = signal_value

    df[f'{mark_name}_overlap'] = df['read_id'].isin(overlaps).astype(int)
    df[f'{mark_name}_signal'] = df['read_id'].map(overlaps).fillna(0)

    n_ov = df[f'{mark_name}_overlap'].sum()
    print(f"  {mark_name}: {n_ov} reads overlap ({n_ov/len(df)*100:.1f}%)")

# ── Per-mark analysis ──
print("\n" + "="*80)
print("PER-MARK ANALYSIS")
print("="*80)

results_rows = []

for mark_name in MARKS:
    print(f"\n{'─'*60}")
    print(f"  {mark_name}")
    print(f"{'─'*60}")

    col = f'{mark_name}_overlap'

    for cond in ['normal', 'stress']:
        sub = df[df.cond == cond]
        overlap = sub[sub[col] == 1]
        no_overlap = sub[sub[col] == 0]

        print(f"\n  [{cond.upper()}] overlap={len(overlap)}, no_overlap={len(no_overlap)}")

        if len(overlap) < 10:
            print(f"    Too few overlapping reads, skipping.")
            continue

        # Poly(A) length
        pa_ov = overlap['polya_length']
        pa_no = no_overlap['polya_length']
        u_stat, p_pa = stats.mannwhitneyu(pa_ov, pa_no, alternative='two-sided')
        print(f"    Poly(A): overlap={pa_ov.median():.1f}nt, no_overlap={pa_no.median():.1f}nt, "
              f"diff={pa_ov.median()-pa_no.median():.1f}nt, P={p_pa:.2e}")

        # m6A/kb
        m6a_ov = overlap['m6a_per_kb']
        m6a_no = no_overlap['m6a_per_kb']
        u_stat, p_m6a = stats.mannwhitneyu(m6a_ov, m6a_no, alternative='two-sided')
        ratio_m6a = m6a_ov.median() / m6a_no.median() if m6a_no.median() > 0 else np.nan
        print(f"    m6A/kb: overlap={m6a_ov.median():.2f}, no_overlap={m6a_no.median():.2f}, "
              f"ratio={ratio_m6a:.2f}x, P={p_m6a:.2e}")

        # Decay zone (<30nt)
        dz_ov = (pa_ov < 30).mean() * 100
        dz_no = (pa_no < 30).mean() * 100
        ct = pd.crosstab(sub[col], (sub['polya_length'] < 30).astype(int))
        if ct.shape == (2, 2):
            odds, p_dz = stats.fisher_exact(ct)
        else:
            odds, p_dz = np.nan, np.nan
        print(f"    Decay zone (<30nt): overlap={dz_ov:.1f}%, no_overlap={dz_no:.1f}%, "
              f"OR={odds:.2f}, P={p_dz:.2e}")

        results_rows.append({
            'mark': mark_name, 'condition': cond,
            'n_overlap': len(overlap), 'n_no_overlap': len(no_overlap),
            'median_polyA_overlap': round(pa_ov.median(), 1),
            'median_polyA_no_overlap': round(pa_no.median(), 1),
            'polyA_diff': round(pa_ov.median() - pa_no.median(), 1),
            'polyA_P': p_pa,
            'median_m6a_overlap': round(m6a_ov.median(), 2),
            'median_m6a_no_overlap': round(m6a_no.median(), 2),
            'm6a_ratio': round(ratio_m6a, 2) if not np.isnan(ratio_m6a) else np.nan,
            'm6a_P': p_m6a,
            'decay_zone_pct_overlap': round(dz_ov, 1),
            'decay_zone_pct_no_overlap': round(dz_no, 1),
            'decay_zone_OR': round(odds, 2) if not np.isnan(odds) else np.nan,
            'decay_zone_P': p_dz,
        })

# ── Poly(A) delta analysis ──
print("\n" + "="*80)
print("POLY(A) DELTA ANALYSIS (stress median - normal median)")
print("="*80)

delta_rows = []
for mark_name in MARKS:
    col = f'{mark_name}_overlap'
    for label, val in [('overlap', 1), ('no_overlap', 0)]:
        sub_n = df[(df.cond == 'normal') & (df[col] == val)]
        sub_s = df[(df.cond == 'stress') & (df[col] == val)]
        if len(sub_n) >= 10 and len(sub_s) >= 10:
            med_n = sub_n['polya_length'].median()
            med_s = sub_s['polya_length'].median()
            delta = med_s - med_n
            u, p = stats.mannwhitneyu(sub_s['polya_length'], sub_n['polya_length'], alternative='two-sided')
            print(f"  {mark_name:10s} {label:12s}: normal={med_n:.1f}nt, stress={med_s:.1f}nt, "
                  f"delta={delta:+.1f}nt, P={p:.2e}")
            delta_rows.append({
                'mark': mark_name, 'group': label,
                'normal_polyA': round(med_n, 1), 'stress_polyA': round(med_s, 1),
                'delta': round(delta, 1), 'P': p,
                'n_normal': len(sub_n), 'n_stress': len(sub_s)
            })

# ── Combinatorial analysis ──
print("\n" + "="*80)
print("COMBINATORIAL HISTONE MARK ANALYSIS")
print("="*80)

def categorize(row):
    k27ac = row['H3K27ac_overlap']
    k4me1 = row['H3K4me1_overlap']
    k4me3 = row['H3K4me3_overlap']

    if k27ac and k4me1 and not k4me3:
        return 'Active_Enhancer'
    elif k27ac and k4me3:
        return 'Active_Promoter'
    elif k4me1 and not k27ac and not k4me3:
        return 'Poised_Enhancer'
    elif k4me3 and not k27ac:
        return 'Poised_Promoter'
    elif k27ac and not k4me1 and not k4me3:
        return 'H3K27ac_only'
    elif not k27ac and not k4me1 and not k4me3:
        return 'No_marks'
    else:
        return 'Other'

df['histone_category'] = df.apply(categorize, axis=1)
print("\nCategory counts (total):")
for cat, cnt in df['histone_category'].value_counts().items():
    print(f"  {cat:20s}: {cnt}")

combo_rows = []
print(f"\n{'Category':<20s} {'Cond':>7s} {'n':>5s} {'polyA':>7s} {'m6A/kb':>7s} {'DZ%':>6s}")
print("-" * 60)
for cat in ['Active_Enhancer', 'Active_Promoter', 'Poised_Enhancer', 'Poised_Promoter',
            'H3K27ac_only', 'No_marks', 'Other']:
    sub = df[df.histone_category == cat]
    if len(sub) < 10:
        continue

    for cond in ['normal', 'stress']:
        cs = sub[sub.cond == cond]
        if len(cs) < 5:
            continue

        med_pa = cs['polya_length'].median()
        med_m6a = cs['m6a_per_kb'].median()
        dz = (cs['polya_length'] < 30).mean() * 100

        combo_rows.append({
            'category': cat, 'condition': cond, 'n': len(cs),
            'median_polyA': round(med_pa, 1), 'median_m6a': round(med_m6a, 2),
            'decay_zone_pct': round(dz, 1)
        })

        print(f"  {cat:<20s} {cond:>7s} {len(cs):>5d} {med_pa:>7.1f} {med_m6a:>7.2f} {dz:>6.1f}")

# Delta for combinatorial
print("\nPoly(A) delta by category:")
for cat in ['Active_Enhancer', 'Active_Promoter', 'Poised_Enhancer', 'Poised_Promoter',
            'H3K27ac_only', 'No_marks']:
    sub_n = df[(df.histone_category == cat) & (df.cond == 'normal')]
    sub_s = df[(df.histone_category == cat) & (df.cond == 'stress')]
    if len(sub_n) >= 5 and len(sub_s) >= 5:
        delta = sub_s['polya_length'].median() - sub_n['polya_length'].median()
        u, p = stats.mannwhitneyu(sub_s['polya_length'], sub_n['polya_length'], alternative='two-sided')
        print(f"  {cat:20s}: delta={delta:+.1f}nt (n_norm={len(sub_n)}, n_stress={len(sub_s)}, P={p:.2e})")

# ── Signal strength correlation ──
print("\n" + "="*80)
print("SIGNAL STRENGTH vs POLY(A) (H3K27ac, H3K4me1)")
print("="*80)

for mark_name in ['H3K27ac', 'H3K4me1']:
    for cond in ['stress', 'normal']:
        sub = df[(df.cond == cond) & (df[f'{mark_name}_overlap'] == 1)].copy()
        if len(sub) >= 20:
            rho, p_rho = stats.spearmanr(sub[f'{mark_name}_signal'], sub['polya_length'])
            print(f"  {mark_name} [{cond}]: n={len(sub)}, signal vs poly(A) rho={rho:.3f}, P={p_rho:.2e}")

            # Quartile analysis of signal
            try:
                sub['signal_q'] = pd.qcut(sub[f'{mark_name}_signal'], 4,
                                          labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
                for q in ['Q1','Q2','Q3','Q4']:
                    qs = sub[sub.signal_q == q]
                    if len(qs) > 0:
                        print(f"    {q}: n={len(qs)}, polyA={qs['polya_length'].median():.1f}nt, "
                              f"decay_zone={((qs['polya_length']<30).mean()*100):.1f}%")
            except ValueError:
                print(f"    (quartile binning failed - too few unique values)")

# ── Any active mark vs no marks ──
print("\n" + "="*80)
print("ANY ACTIVE MARK (H3K27ac or H3K4me3) vs NO MARKS")
print("="*80)

df['any_active'] = ((df['H3K27ac_overlap'] == 1) | (df['H3K4me3_overlap'] == 1)).astype(int)
df['any_mark'] = ((df['H3K27ac_overlap'] == 1) | (df['H3K4me1_overlap'] == 1) |
                   (df['H3K4me3_overlap'] == 1)).astype(int)

for label, col in [('any_active (K27ac|K4me3)', 'any_active'),
                    ('any_mark (K27ac|K4me1|K4me3)', 'any_mark')]:
    print(f"\n  {label}:")
    for cond in ['normal', 'stress']:
        sub = df[df.cond == cond]
        active = sub[sub[col] == 1]
        inactive = sub[sub[col] == 0]

        if len(active) < 10:
            continue

        u, p_pa = stats.mannwhitneyu(active['polya_length'], inactive['polya_length'], alternative='two-sided')
        u, p_m6a = stats.mannwhitneyu(active['m6a_per_kb'], inactive['m6a_per_kb'], alternative='two-sided')
        dz_a = (active['polya_length'] < 30).mean() * 100
        dz_i = (inactive['polya_length'] < 30).mean() * 100

        print(f"    [{cond}] active={len(active)}, inactive={len(inactive)}")
        print(f"      Poly(A): {active['polya_length'].median():.1f} vs {inactive['polya_length'].median():.1f}nt, P={p_pa:.2e}")
        print(f"      m6A/kb:  {active['m6a_per_kb'].median():.2f} vs {inactive['m6a_per_kb'].median():.2f}, P={p_m6a:.2e}")
        print(f"      DZ:      {dz_a:.1f}% vs {dz_i:.1f}%")

# Delta comparison
print("\n  Poly(A) delta (any_mark vs no_mark):")
for label, col in [('any_active', 'any_active'), ('any_mark', 'any_mark')]:
    for val, name in [(1, 'marked'), (0, 'unmarked')]:
        n_sub = df[(df.cond == 'normal') & (df[col] == val)]
        s_sub = df[(df.cond == 'stress') & (df[col] == val)]
        if len(n_sub) >= 10 and len(s_sub) >= 10:
            delta = s_sub['polya_length'].median() - n_sub['polya_length'].median()
            u, p = stats.mannwhitneyu(s_sub['polya_length'], n_sub['polya_length'])
            print(f"    {label} {name}: delta={delta:+.1f}nt (P={p:.2e})")

# ── Cross-validation with ChromHMM ──
print("\n" + "="*80)
print("CROSS-VALIDATION: Histone marks vs ChromHMM categories")
print("="*80)

print(f"\n{'ChromHMM':<14s} {'n':>5s} {'K27ac%':>7s} {'K4me1%':>7s} {'K4me3%':>7s}")
print("-" * 45)
for chromhmm_cat in ['Enhancer', 'Promoter', 'Transcribed', 'Quiescent', 'Repressed']:
    sub = df[df.chromhmm_group == chromhmm_cat]
    if len(sub) < 10:
        continue
    k27ac_frac = sub['H3K27ac_overlap'].mean() * 100
    k4me1_frac = sub['H3K4me1_overlap'].mean() * 100
    k4me3_frac = sub['H3K4me3_overlap'].mean() * 100
    print(f"  {chromhmm_cat:<12s} {len(sub):>5d} {k27ac_frac:>7.1f} {k4me1_frac:>7.1f} {k4me3_frac:>7.1f}")

# ── Regulatory vs non-regulatory with histone marks ──
print("\n" + "="*80)
print("REGULATORY L1 (ChromHMM Enhancer+Promoter) x HISTONE MARKS")
print("="*80)

df['regulatory'] = df['chromhmm_group'].isin(['Enhancer', 'Promoter']).astype(int)

for cond in ['stress']:
    sub = df[df.cond == cond]
    for reg_val, reg_label in [(1, 'Regulatory'), (0, 'Non-regulatory')]:
        rsub = sub[sub.regulatory == reg_val]
        marked = rsub[rsub.any_mark == 1]
        unmarked = rsub[rsub.any_mark == 0]
        print(f"\n  {reg_label} [{cond}]: marked={len(marked)}, unmarked={len(unmarked)}")
        if len(marked) >= 10 and len(unmarked) >= 10:
            print(f"    Poly(A): {marked['polya_length'].median():.1f} vs {unmarked['polya_length'].median():.1f}nt")
            print(f"    m6A/kb:  {marked['m6a_per_kb'].median():.2f} vs {unmarked['m6a_per_kb'].median():.2f}")
            dz_m = (marked['polya_length'] < 30).mean() * 100
            dz_u = (unmarked['polya_length'] < 30).mean() * 100
            print(f"    DZ:      {dz_m:.1f}% vs {dz_u:.1f}%")

# ── Save results ──
results_df = pd.DataFrame(results_rows)
results_df.to_csv(f"{OUTDIR}/per_mark_results.tsv", sep='\t', index=False)

combo_df = pd.DataFrame(combo_rows)
combo_df.to_csv(f"{OUTDIR}/combinatorial_results.tsv", sep='\t', index=False)

delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(f"{OUTDIR}/delta_by_mark.tsv", sep='\t', index=False)

df.to_csv(f"{OUTDIR}/l1_histone_annotated.tsv", sep='\t', index=False)

print(f"\n{'='*80}")
print("FILES SAVED:")
print(f"  {OUTDIR}/per_mark_results.tsv")
print(f"  {OUTDIR}/combinatorial_results.tsv")
print(f"  {OUTDIR}/delta_by_mark.tsv")
print(f"  {OUTDIR}/l1_histone_annotated.tsv")
print("Done.")
