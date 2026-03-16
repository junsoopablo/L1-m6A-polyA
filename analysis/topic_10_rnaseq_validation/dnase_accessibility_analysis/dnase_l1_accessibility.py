#!/usr/bin/env python3
"""
DNase-seq accessibility vs L1 poly(A) shortening vulnerability.
Tests whether L1 loci in open chromatin (DNase peaks) show greater
poly(A) shortening under arsenite stress.

Uses ENCODE HeLa-S3 DNase hotspots (ENCFF856MFN) and L1 ChromHMM-annotated data.
"""

import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
from scipy import stats

OUT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/dnase_accessibility_analysis"
DNASE = os.path.join(OUT, "ENCFF856MFN.bed.gz")
CHROMHMM = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin/cross_cl_chromhmm_annotated.tsv"

# ── 1. Load L1 data ──────────────────────────────────────────────────
print("Loading L1 data...")
df = pd.read_csv(CHROMHMM, sep='\t')
# Filter HeLa (normal) and HeLa-Ars (stress), ancient L1 only
hela_mask = df['cellline'].isin(['HeLa', 'HeLa-Ars'])
ancient_mask = df['l1_age'] == 'ancient'
df_hela = df[hela_mask & ancient_mask].copy()
print(f"  HeLa ancient L1 reads: {len(df_hela)}")
print(f"  Normal: {(df_hela['condition']=='normal').sum()}, Stress: {(df_hela['condition']=='stress').sum()}")

# ── 2. Get unique L1 loci and intersect with DNase ───────────────────
# Create BED of unique L1 loci
loci = df_hela[['chr', 'start', 'end']].drop_duplicates()
loci_bed = os.path.join(OUT, "hela_ancient_l1_loci.bed")
loci[['chr', 'start', 'end']].sort_values(['chr', 'start']).to_csv(
    loci_bed, sep='\t', header=False, index=False
)
print(f"  Unique L1 loci: {len(loci)}")

# Run bedtools intersect
accessible_bed = os.path.join(OUT, "l1_dnase_accessible.bed")
cmd = (
    f"source /etc/profile.d/modules.sh 2>/dev/null; module load bedtools/2.31.0 2>/dev/null; "
    f"bedtools intersect -a {loci_bed} -b {DNASE} -wa -u > {accessible_bed}"
)
subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')

# Load accessible loci
acc = pd.read_csv(accessible_bed, sep='\t', header=None, names=['chr', 'start', 'end'])
acc_set = set(zip(acc['chr'], acc['start'], acc['end']))
print(f"  L1 loci overlapping DNase peaks: {len(acc_set)} ({100*len(acc_set)/len(loci):.1f}%)")

# ── 3. Annotate reads with accessibility ─────────────────────────────
df_hela['dnase_accessible'] = df_hela.apply(
    lambda r: (r['chr'], r['start'], r['end']) in acc_set, axis=1
)

# Also add a "locus_key" for per-locus aggregation
df_hela['locus_key'] = df_hela['chr'] + ':' + df_hela['start'].astype(str) + '-' + df_hela['end'].astype(str)

print(f"\n  Accessible reads: {df_hela['dnase_accessible'].sum()}")
print(f"  Non-accessible reads: {(~df_hela['dnase_accessible']).sum()}")

# ── 4. Analysis ──────────────────────────────────────────────────────
results = []

for cond_label, cond_val in [('Normal', 'normal'), ('Stress', 'stress')]:
    sub = df_hela[df_hela['condition'] == cond_val]
    acc_data = sub[sub['dnase_accessible']]
    nonacc_data = sub[~sub['dnase_accessible']]

    print(f"\n{'='*60}")
    print(f"Condition: {cond_label}")
    print(f"  Accessible: n={len(acc_data)}, Non-accessible: n={len(nonacc_data)}")

    # a. Poly(A) length
    pa_acc = acc_data['polya_length'].values
    pa_nonacc = nonacc_data['polya_length'].values
    u_pa, p_pa = stats.mannwhitneyu(pa_acc, pa_nonacc, alternative='two-sided')
    med_acc = np.median(pa_acc)
    med_nonacc = np.median(pa_nonacc)
    print(f"  Poly(A) median: Accessible={med_acc:.1f}, Non-accessible={med_nonacc:.1f}")
    print(f"    Delta={med_acc - med_nonacc:.1f}nt, MWU P={p_pa:.2e}")

    results.append({
        'metric': 'polya_length',
        'condition': cond_label,
        'accessible_n': len(acc_data),
        'nonacc_n': len(nonacc_data),
        'accessible_median': med_acc,
        'nonacc_median': med_nonacc,
        'delta': med_acc - med_nonacc,
        'mwu_p': p_pa
    })

    # b. m6A/kb
    m6a_acc = acc_data['m6a_per_kb'].values
    m6a_nonacc = nonacc_data['m6a_per_kb'].values
    u_m6a, p_m6a = stats.mannwhitneyu(m6a_acc, m6a_nonacc, alternative='two-sided')
    med_m6a_acc = np.median(m6a_acc)
    med_m6a_nonacc = np.median(m6a_nonacc)
    print(f"  m6A/kb median: Accessible={med_m6a_acc:.2f}, Non-accessible={med_m6a_nonacc:.2f}")
    print(f"    Ratio={med_m6a_acc/med_m6a_nonacc:.2f}x, MWU P={p_m6a:.2e}")

    results.append({
        'metric': 'm6a_per_kb',
        'condition': cond_label,
        'accessible_n': len(acc_data),
        'nonacc_n': len(nonacc_data),
        'accessible_median': med_m6a_acc,
        'nonacc_median': med_m6a_nonacc,
        'delta': med_m6a_acc - med_m6a_nonacc,
        'mwu_p': p_m6a
    })

    # c. Decay zone (<30nt)
    decay_acc = (pa_acc < 30).sum()
    decay_nonacc = (pa_nonacc < 30).sum()
    frac_acc = decay_acc / len(pa_acc) * 100
    frac_nonacc = decay_nonacc / len(pa_nonacc) * 100
    # Fisher's exact test
    table = [[decay_acc, len(pa_acc) - decay_acc],
             [decay_nonacc, len(pa_nonacc) - decay_nonacc]]
    or_val, p_fisher = stats.fisher_exact(table)
    print(f"  Decay zone (<30nt): Accessible={frac_acc:.1f}%, Non-accessible={frac_nonacc:.1f}%")
    print(f"    OR={or_val:.2f}, Fisher P={p_fisher:.2e}")

    results.append({
        'metric': 'decay_zone_pct',
        'condition': cond_label,
        'accessible_n': len(acc_data),
        'nonacc_n': len(nonacc_data),
        'accessible_median': frac_acc,
        'nonacc_median': frac_nonacc,
        'delta': frac_acc - frac_nonacc,
        'mwu_p': p_fisher
    })

# ── 5. Poly(A) delta (stress - normal) per locus ────────────────────
print(f"\n{'='*60}")
print("Per-locus poly(A) delta (stress - normal)")

# Aggregate per locus
locus_normal = df_hela[df_hela['condition']=='normal'].groupby('locus_key').agg(
    polya_normal=('polya_length', 'median'),
    m6a_normal=('m6a_per_kb', 'median'),
    n_normal=('read_id', 'count')
).reset_index()

locus_stress = df_hela[df_hela['condition']=='stress'].groupby('locus_key').agg(
    polya_stress=('polya_length', 'median'),
    m6a_stress=('m6a_per_kb', 'median'),
    n_stress=('read_id', 'count')
).reset_index()

locus_merged = locus_normal.merge(locus_stress, on='locus_key', how='inner')
locus_merged['delta_polya'] = locus_merged['polya_stress'] - locus_merged['polya_normal']

# Get accessibility for each locus
locus_acc = df_hela[['locus_key', 'dnase_accessible']].drop_duplicates('locus_key')
locus_merged = locus_merged.merge(locus_acc, on='locus_key')

print(f"  Loci with both conditions: {len(locus_merged)}")
print(f"    Accessible: {locus_merged['dnase_accessible'].sum()}")
print(f"    Non-accessible: {(~locus_merged['dnase_accessible']).sum()}")

acc_delta = locus_merged[locus_merged['dnase_accessible']]['delta_polya']
nonacc_delta = locus_merged[~locus_merged['dnase_accessible']]['delta_polya']

if len(acc_delta) > 0 and len(nonacc_delta) > 0:
    u, p = stats.mannwhitneyu(acc_delta, nonacc_delta, alternative='two-sided')
    print(f"  Delta poly(A) median: Accessible={acc_delta.median():.1f}nt, Non-accessible={nonacc_delta.median():.1f}nt")
    print(f"    MWU P={p:.2e}")

    results.append({
        'metric': 'locus_delta_polya',
        'condition': 'stress-normal',
        'accessible_n': len(acc_delta),
        'nonacc_n': len(nonacc_delta),
        'accessible_median': acc_delta.median(),
        'nonacc_median': nonacc_delta.median(),
        'delta': acc_delta.median() - nonacc_delta.median(),
        'mwu_p': p
    })

# ── 6. ChromHMM Regulatory overlap with DNase ────────────────────────
print(f"\n{'='*60}")
print("ChromHMM Regulatory vs DNase Accessibility overlap")

if 'chromhmm_group' in df_hela.columns:
    for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Quiescent']:
        sub = df_hela[df_hela['chromhmm_group'] == grp]
        if len(sub) == 0:
            sub = df_hela[df_hela.get('is_regulatory', pd.Series(dtype=bool)) == True] if grp == 'Regulatory' else pd.DataFrame()
        if len(sub) > 0:
            pct = sub['dnase_accessible'].mean() * 100
            print(f"  {grp}: {pct:.1f}% DNase-accessible (n={len(sub)})")
elif 'is_regulatory' in df_hela.columns:
    for reg_val, label in [(True, 'Regulatory'), (False, 'Non-regulatory')]:
        sub = df_hela[df_hela['is_regulatory'] == reg_val]
        pct = sub['dnase_accessible'].mean() * 100
        print(f"  {label}: {pct:.1f}% DNase-accessible (n={len(sub)})")

# ── 7. Interaction: DNase x Stress ───────────────────────────────────
print(f"\n{'='*60}")
print("Interaction: DNase accessibility x Stress condition")

# 2x2: (normal/stress) x (accessible/non-accessible) median poly(A)
for acc_label, acc_val in [('Accessible', True), ('Non-accessible', False)]:
    for cond_label, cond_val in [('Normal', 'normal'), ('Stress', 'stress')]:
        sub = df_hela[(df_hela['dnase_accessible']==acc_val) & (df_hela['condition']==cond_val)]
        print(f"  {acc_label} × {cond_label}: median poly(A)={sub['polya_length'].median():.1f}nt (n={len(sub)})")

# Compute per-read delta by matching locus
# Get read-level: for stress reads, how does accessible vs non-accessible differ?
stress_acc = df_hela[(df_hela['condition']=='stress') & df_hela['dnase_accessible']]['polya_length']
stress_nonacc = df_hela[(df_hela['condition']=='stress') & ~df_hela['dnase_accessible']]['polya_length']
normal_acc = df_hela[(df_hela['condition']=='normal') & df_hela['dnase_accessible']]['polya_length']
normal_nonacc = df_hela[(df_hela['condition']=='normal') & ~df_hela['dnase_accessible']]['polya_length']

delta_acc = stress_acc.median() - normal_acc.median()
delta_nonacc = stress_nonacc.median() - normal_nonacc.median()
print(f"\n  Stress-induced shortening:")
print(f"    Accessible L1:     Delta = {delta_acc:.1f}nt")
print(f"    Non-accessible L1: Delta = {delta_nonacc:.1f}nt")
print(f"    Difference (interaction): {delta_acc - delta_nonacc:.1f}nt")

# ── 8. DNase signal strength correlation ─────────────────────────────
print(f"\n{'='*60}")
print("DNase signal strength analysis")

# Re-run bedtools with -wo to get overlap size and signal
signal_bed = os.path.join(OUT, "l1_dnase_signal.bed")
cmd = (
    f"source /etc/profile.d/modules.sh 2>/dev/null; module load bedtools/2.31.0 2>/dev/null; "
    f"bedtools intersect -a {loci_bed} -b {DNASE} -wo > {signal_bed}"
)
subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')

# Parse signal (col 9 = DNase signal score from narrowPeak, col 13 = overlap bp)
try:
    sig = pd.read_csv(signal_bed, sep='\t', header=None)
    if len(sig) > 0:
        sig.columns = ['l1_chr', 'l1_start', 'l1_end',
                       'dnase_chr', 'dnase_start', 'dnase_end', 'dnase_id', 'dnase_score',
                       'dnase_strand', 'dnase_s1', 'dnase_s2', 'dnase_signal', 'overlap_bp']
        sig['locus_key'] = sig['l1_chr'] + ':' + sig['l1_start'].astype(str) + '-' + sig['l1_end'].astype(str)
        # Take max signal per locus
        locus_signal = sig.groupby('locus_key')['dnase_signal'].max().reset_index()
        locus_signal.columns = ['locus_key', 'max_dnase_signal']

        # Merge with locus delta
        locus_with_signal = locus_merged.merge(locus_signal, on='locus_key', how='left')
        locus_with_signal['max_dnase_signal'] = locus_with_signal['max_dnase_signal'].fillna(0)

        # Correlate DNase signal with delta poly(A) among accessible loci
        acc_sig = locus_with_signal[locus_with_signal['dnase_accessible']]
        if len(acc_sig) > 5:
            rho, p_rho = stats.spearmanr(acc_sig['max_dnase_signal'], acc_sig['delta_polya'])
            print(f"  Among accessible loci:")
            print(f"    DNase signal vs delta poly(A): Spearman rho={rho:.3f}, P={p_rho:.2e} (n={len(acc_sig)})")
except Exception as e:
    print(f"  Signal analysis failed: {e}")

# ── 9. Save results ──────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT, "dnase_accessibility_results.tsv"), sep='\t', index=False)
print(f"\nResults saved to {OUT}/dnase_accessibility_results.tsv")

# ── 10. Generate summary figure ──────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("DNase Accessibility vs L1 Poly(A) Vulnerability\n(HeLa Ancient L1, ENCODE ENCFF856MFN)", fontsize=12)

# Panel A: Poly(A) by accessibility x condition
ax = axes[0, 0]
positions = [1, 2, 4, 5]
data_boxes = [
    normal_acc.values, normal_nonacc.values,
    stress_acc.values, stress_nonacc.values
]
bp = ax.boxplot(data_boxes, positions=positions, widths=0.6, showfliers=False,
                patch_artist=True, medianprops=dict(color='black', linewidth=2))
colors = ['#ff7f7f', '#7fbfff', '#ff7f7f', '#7fbfff']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticks([1.5, 4.5])
ax.set_xticklabels(['Normal', 'Stress'])
ax.set_ylabel('Poly(A) length (nt)')
ax.set_title('A) Poly(A) by Condition')
ax.legend([bp['boxes'][0], bp['boxes'][1]], ['Accessible', 'Non-accessible'],
          fontsize=8, loc='upper right')

# Panel B: m6A/kb by accessibility x condition
ax = axes[0, 1]
m6a_data = []
for cond in ['normal', 'stress']:
    for acc_val in [True, False]:
        sub = df_hela[(df_hela['condition']==cond) & (df_hela['dnase_accessible']==acc_val)]
        m6a_data.append(sub['m6a_per_kb'].values)
bp2 = ax.boxplot(m6a_data, positions=positions, widths=0.6, showfliers=False,
                 patch_artist=True, medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticks([1.5, 4.5])
ax.set_xticklabels(['Normal', 'Stress'])
ax.set_ylabel('m6A/kb')
ax.set_title('B) m6A/kb by Condition')

# Panel C: Per-locus delta poly(A) by accessibility
ax = axes[1, 0]
bp3 = ax.boxplot([acc_delta.values, nonacc_delta.values],
                 labels=['Accessible', 'Non-accessible'],
                 showfliers=False, patch_artist=True,
                 medianprops=dict(color='black', linewidth=2))
bp3['boxes'][0].set_facecolor('#ff7f7f')
bp3['boxes'][1].set_facecolor('#7fbfff')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Delta poly(A) (stress - normal, nt)')
ax.set_title('C) Per-locus shortening')

# Panel D: Decay zone proportion
ax = axes[1, 1]
decay_data = []
for cond in ['normal', 'stress']:
    for acc_val in [True, False]:
        sub = df_hela[(df_hela['condition']==cond) & (df_hela['dnase_accessible']==acc_val)]
        pct = (sub['polya_length'] < 30).mean() * 100
        decay_data.append(pct)
x = np.array([0, 1, 3, 4])
bar_colors = ['#ff7f7f', '#7fbfff', '#ff7f7f', '#7fbfff']
ax.bar(x, decay_data, color=bar_colors, edgecolor='black', width=0.7)
ax.set_xticks([0.5, 3.5])
ax.set_xticklabels(['Normal', 'Stress'])
ax.set_ylabel('% reads in decay zone (<30nt)')
ax.set_title('D) Decay zone entry')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "dnase_accessibility_analysis.pdf"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure saved to {OUT}/dnase_accessibility_analysis.pdf")

print("\n" + "="*60)
print("DONE")
