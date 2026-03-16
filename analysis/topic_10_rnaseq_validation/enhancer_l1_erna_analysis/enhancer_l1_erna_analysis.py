#!/usr/bin/env python
"""
Enhancer L1 eRNA-like property analysis.

Compares L1 RNA at enhancer regions with canonical eRNA characteristics:
- Read length, poly(A) length, m6A/kb distributions
- Strand orientation relative to host gene (eRNAs are often antisense)
- Gradient of poly(A) shortening by enhancer type (6_EnhG vs 7_Enh)
- Comparison across chromatin states: Enhancer vs Transcribed vs Quiescent

Uses all cell lines with ChromHMM annotation (E117 for HeLa).
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
CHROMHMM = BASE / 'analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv'
SUMMARY_DIR = BASE / 'results_group'
GTF = BASE / 'reference/Human.gtf'
OUTDIR = BASE / 'analysis/01_exploration/topic_10_rnaseq_validation/enhancer_l1_erna_analysis'
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load ChromHMM annotated data ──
print("Loading ChromHMM-annotated L1 reads...")
df = pd.read_csv(CHROMHMM, sep='\t')
df['read_length'] = df['end'] - df['start']
print(f"  Total reads: {len(df)}")

# ── 2. Load strand info from L1 summary files ──
print("Loading strand information from L1 summary files...")
strand_dfs = []
for grp_dir in sorted(SUMMARY_DIR.iterdir()):
    summary_file = grp_dir / 'g_summary' / f'{grp_dir.name}_L1_summary.tsv'
    if summary_file.exists():
        tmp = pd.read_csv(summary_file, sep='\t', usecols=['read_id', 'te_strand', 'read_strand'])
        strand_dfs.append(tmp)

strand_info = pd.concat(strand_dfs, ignore_index=True).drop_duplicates(subset='read_id')
print(f"  Strand info for {len(strand_info)} reads")

df = df.merge(strand_info, on='read_id', how='left')
print(f"  Merged: {df['read_strand'].notna().sum()} reads with strand info")

# ── 3. Load gene strand from GTF (for intronic reads) ──
print("Parsing GTF for gene strand info...")
gene_strand = {}
with open(GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene':
            continue
        attrs = parts[8]
        # Extract gene_name
        for attr in attrs.split(';'):
            attr = attr.strip()
            if attr.startswith('gene_name'):
                gname = attr.split('"')[1]
                gene_strand[gname] = parts[6]
                break
print(f"  Gene strand info for {len(gene_strand)} genes")

# ── 4. For intronic reads, determine sense/antisense relative to host gene ──
# overlapping_genes is in L1 summary but not in chromhmm file; use L1 summary
print("Loading overlapping gene info...")
gene_dfs = []
for grp_dir in sorted(SUMMARY_DIR.iterdir()):
    summary_file = grp_dir / 'g_summary' / f'{grp_dir.name}_L1_summary.tsv'
    if summary_file.exists():
        tmp = pd.read_csv(summary_file, sep='\t', usecols=['read_id', 'overlapping_genes'])
        gene_dfs.append(tmp)

gene_info = pd.concat(gene_dfs, ignore_index=True).drop_duplicates(subset='read_id')
df = df.merge(gene_info, on='read_id', how='left')

# Determine sense/antisense for intronic reads
def get_orientation(row):
    if row['genomic_context'] != 'intronic' or pd.isna(row.get('overlapping_genes')) or pd.isna(row.get('read_strand')):
        return 'unknown'
    genes = str(row['overlapping_genes']).split(',')
    # Use first gene
    gname = genes[0].strip()
    gstrand = gene_strand.get(gname, None)
    if gstrand is None:
        return 'unknown'
    if row['read_strand'] == gstrand:
        return 'sense'
    else:
        return 'antisense'

df['orientation'] = df.apply(get_orientation, axis=1)
print(f"  Orientation: {df['orientation'].value_counts().to_dict()}")

# ── 5. Analysis ──

# Helper
def mwu_p(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return stats.mannwhitneyu(a, b, alternative='two-sided').pvalue

# --- 5a. Enhancer vs non-Enhancer L1 (all cell lines, normal condition) ---
print("\n=== Enhancer vs Non-Enhancer L1 (all CL, normal) ===")
normal = df[df['condition'] == 'normal'].copy()
enh = normal[normal['chromhmm_group'] == 'Enhancer']
non_enh = normal[normal['chromhmm_group'] != 'Enhancer']

results = {}

for metric, col in [('Read Length', 'read_length'), ('Poly(A) Length', 'polya_length'), ('m6A/kb', 'm6a_per_kb')]:
    e_vals = enh[col].dropna()
    ne_vals = non_enh[col].dropna()
    p = mwu_p(e_vals, ne_vals)
    print(f"  {metric}: Enhancer median={e_vals.median():.1f} (n={len(e_vals)}), "
          f"Non-Enhancer median={ne_vals.median():.1f} (n={len(ne_vals)}), P={p:.2e}")
    results[metric] = {'enh_median': e_vals.median(), 'non_enh_median': ne_vals.median(), 'p': p}

# --- 5b. Enhancer orientation (intronic enhancer L1) ---
print("\n=== Orientation of Intronic Enhancer L1 ===")
enh_intronic = normal[(normal['chromhmm_group'] == 'Enhancer') & (normal['genomic_context'] == 'intronic')]
print(f"  Intronic Enhancer L1: {len(enh_intronic)} reads")
ort_counts = enh_intronic['orientation'].value_counts()
print(f"  Orientation: {ort_counts.to_dict()}")

# Compare with non-enhancer intronic
non_enh_intronic = normal[(normal['chromhmm_group'] != 'Enhancer') & (normal['genomic_context'] == 'intronic')]
non_enh_ort = non_enh_intronic['orientation'].value_counts()
print(f"  Non-Enhancer Intronic orientation: {non_enh_ort.to_dict()}")

# Antisense fraction
for label, subset in [('Enhancer intronic', enh_intronic), ('Non-Enhancer intronic', non_enh_intronic)]:
    known = subset[subset['orientation'].isin(['sense', 'antisense'])]
    if len(known) > 0:
        anti_frac = (known['orientation'] == 'antisense').mean()
        print(f"  {label}: antisense fraction = {anti_frac:.3f} ({(known['orientation']=='antisense').sum()}/{len(known)})")

# --- 5c. Enhancer vs Transcribed vs Quiescent (all CL, normal) ---
print("\n=== Chromatin State Comparison (all CL, normal) ===")
groups = ['Enhancer', 'Transcribed', 'Quiescent']
for metric, col in [('Read Length', 'read_length'), ('Poly(A) Length', 'polya_length'), ('m6A/kb', 'm6a_per_kb')]:
    print(f"\n  {metric}:")
    for grp in groups:
        vals = normal[normal['chromhmm_group'] == grp][col].dropna()
        print(f"    {grp:15s}: median={vals.median():.1f}, mean={vals.mean():.1f}, n={len(vals)}")
    # Pairwise Enhancer vs others
    enh_vals = normal[normal['chromhmm_group'] == 'Enhancer'][col].dropna()
    for other in ['Transcribed', 'Quiescent']:
        other_vals = normal[normal['chromhmm_group'] == other][col].dropna()
        p = mwu_p(enh_vals, other_vals)
        print(f"    Enhancer vs {other}: P={p:.2e}")

# --- 5d. HeLa stress: enhancer poly(A) delta ---
print("\n=== HeLa Enhancer Poly(A) Stress Response ===")
hela_all = df[df['cellline'].isin(['HeLa', 'HeLa-Ars'])].copy()
hela_all['is_stress'] = hela_all['condition'] == 'stress'

for grp in groups:
    subset = hela_all[hela_all['chromhmm_group'] == grp]
    baseline = subset[subset['condition'] == 'normal']['polya_length'].dropna()
    stressed = subset[subset['condition'] == 'stress']['polya_length'].dropna()
    if len(baseline) > 0 and len(stressed) > 0:
        delta = stressed.median() - baseline.median()
        p = mwu_p(baseline, stressed)
        print(f"  {grp:15s}: baseline={baseline.median():.1f} (n={len(baseline)}), "
              f"stressed={stressed.median():.1f} (n={len(stressed)}), "
              f"Δ={delta:.1f}nt, P={p:.2e}")
    else:
        print(f"  {grp:15s}: insufficient data (baseline={len(baseline)}, stressed={len(stressed)})")

# --- 5e. 6_EnhG vs 7_Enh (genic vs non-genic enhancer) ---
print("\n=== 6_EnhG vs 7_Enh (Genic vs Non-Genic Enhancer) ===")
for state in ['6_EnhG', '7_Enh']:
    subset = normal[normal['chromhmm_state'] == state]
    for metric, col in [('Read Length', 'read_length'), ('Poly(A) Length', 'polya_length'), ('m6A/kb', 'm6a_per_kb')]:
        vals = subset[col].dropna()
        print(f"  {state} {metric}: median={vals.median():.1f}, n={len(vals)}")

# Stress delta by enhancer subtype
print("\n  HeLa stress by enhancer subtype:")
for state in ['6_EnhG', '7_Enh']:
    hela_state = hela_all[hela_all['chromhmm_state'] == state]
    baseline = hela_state[hela_state['condition'] == 'normal']['polya_length'].dropna()
    stressed = hela_state[hela_state['condition'] == 'stress']['polya_length'].dropna()
    if len(baseline) > 0 and len(stressed) > 0:
        delta = stressed.median() - baseline.median()
        p = mwu_p(baseline, stressed)
        print(f"    {state}: baseline={baseline.median():.1f} (n={len(baseline)}), "
              f"stressed={stressed.median():.1f} (n={len(stressed)}), Δ={delta:.1f}nt, P={p:.2e}")
    else:
        print(f"    {state}: insufficient data")

# --- 5f. Expression level proxy (reads per locus) and poly(A) correlation ---
print("\n=== Expression Level (reads/locus) vs Poly(A) ===")
# Use gene_id as locus identifier (L1 subfamily instance)
# For enhancer L1, count reads per locus
enh_all = df[df['chromhmm_group'] == 'Enhancer'].copy()
# Create locus key from chr+approx position (within 1kb)
enh_all['locus_key'] = enh_all['chr'] + ':' + (enh_all['start'] // 1000 * 1000).astype(str)
locus_counts = enh_all.groupby('locus_key').size().reset_index(name='reads_at_locus')
enh_all = enh_all.merge(locus_counts, on='locus_key')

# Correlation: reads_at_locus vs polya_length
r, p = stats.spearmanr(enh_all['reads_at_locus'], enh_all['polya_length'])
print(f"  Spearman rho (reads/locus vs poly(A)): {r:.3f}, P={p:.2e}, n={len(enh_all)}")

# Compare high vs low expression loci
median_expr = enh_all['reads_at_locus'].median()
high = enh_all[enh_all['reads_at_locus'] > median_expr]['polya_length'].dropna()
low = enh_all[enh_all['reads_at_locus'] <= median_expr]['polya_length'].dropna()
p = mwu_p(high, low)
print(f"  High-expr loci poly(A): {high.median():.1f} (n={len(high)})")
print(f"  Low-expr loci poly(A): {low.median():.1f} (n={len(low)}), P={p:.2e}")

# --- 5g. eRNA-like properties summary ---
print("\n=== eRNA-Like Property Summary ===")
print("Canonical eRNA properties:")
print("  1. Short transcripts (<2kb) → check")
print("  2. Short/absent poly(A) tails → check")
print("  3. Unstable (rapid turnover) → indirect via poly(A)")
print("  4. Often antisense to genes → check")
print("  5. Levels correlate with enhancer activity → read count proxy")

enh_rl = normal[normal['chromhmm_group'] == 'Enhancer']['read_length'].dropna()
tx_rl = normal[normal['chromhmm_group'] == 'Transcribed']['read_length'].dropna()
quie_rl = normal[normal['chromhmm_group'] == 'Quiescent']['read_length'].dropna()

print(f"\n  Read length: Enhancer {enh_rl.median():.0f} vs Transcribed {tx_rl.median():.0f} vs Quiescent {quie_rl.median():.0f}")
enh_pa = normal[normal['chromhmm_group'] == 'Enhancer']['polya_length'].dropna()
tx_pa = normal[normal['chromhmm_group'] == 'Transcribed']['polya_length'].dropna()
print(f"  Poly(A): Enhancer {enh_pa.median():.1f} vs Transcribed {tx_pa.median():.1f}")

# ── 6. Figures ──
print("\nGenerating figures...")

fig = plt.figure(figsize=(16, 20))
gs = gridspec.GridSpec(4, 3, hspace=0.35, wspace=0.3)

# Panel A: Read length distribution by chromatin group
ax = fig.add_subplot(gs[0, 0])
data_rl = [normal[normal['chromhmm_group'] == g]['read_length'].dropna() for g in groups]
parts = ax.violinplot(data_rl, positions=range(len(groups)), showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#e74c3c', '#3498db', '#95a5a6'][i])
    pc.set_alpha(0.7)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, rotation=15, fontsize=9)
ax.set_ylabel('Read Length (nt)')
ax.set_title('(a) Read Length by Chromatin State', fontsize=10, fontweight='bold')

# Panel B: Poly(A) distribution by chromatin group
ax = fig.add_subplot(gs[0, 1])
data_pa = [normal[normal['chromhmm_group'] == g]['polya_length'].dropna() for g in groups]
parts = ax.violinplot(data_pa, positions=range(len(groups)), showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#e74c3c', '#3498db', '#95a5a6'][i])
    pc.set_alpha(0.7)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, rotation=15, fontsize=9)
ax.set_ylabel('Poly(A) Length (nt)')
ax.set_title('(b) Poly(A) Length by Chromatin State', fontsize=10, fontweight='bold')

# Panel C: m6A/kb by chromatin group
ax = fig.add_subplot(gs[0, 2])
data_m6a = [normal[normal['chromhmm_group'] == g]['m6a_per_kb'].dropna() for g in groups]
parts = ax.violinplot(data_m6a, positions=range(len(groups)), showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#e74c3c', '#3498db', '#95a5a6'][i])
    pc.set_alpha(0.7)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, rotation=15, fontsize=9)
ax.set_ylabel('m6A/kb')
ax.set_title('(c) m6A/kb by Chromatin State', fontsize=10, fontweight='bold')

# Panel D: Strand orientation (Enhancer intronic vs Non-Enhancer intronic)
ax = fig.add_subplot(gs[1, 0])
categories = ['Enhancer\nIntronic', 'Non-Enhancer\nIntronic']
enh_known = enh_intronic[enh_intronic['orientation'].isin(['sense', 'antisense'])]
ne_known = non_enh_intronic[non_enh_intronic['orientation'].isin(['sense', 'antisense'])]
anti_fracs = []
sense_fracs = []
for known in [enh_known, ne_known]:
    if len(known) > 0:
        anti_fracs.append((known['orientation'] == 'antisense').mean())
        sense_fracs.append((known['orientation'] == 'sense').mean())
    else:
        anti_fracs.append(0)
        sense_fracs.append(0)
x = np.arange(len(categories))
ax.bar(x, sense_fracs, 0.6, label='Sense', color='#3498db', alpha=0.8)
ax.bar(x, anti_fracs, 0.6, bottom=sense_fracs, label='Antisense', color='#e74c3c', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('Fraction')
ax.set_title('(d) L1 Orientation vs Host Gene', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
# Add counts
for i, known in enumerate([enh_known, ne_known]):
    ax.text(i, 1.02, f'n={len(known)}', ha='center', fontsize=8)

# Panel E: HeLa stress poly(A) delta by chromatin state
ax = fig.add_subplot(gs[1, 1])
deltas = []
delta_labels = []
delta_ns = []
for grp in groups:
    subset = hela_all[hela_all['chromhmm_group'] == grp]
    baseline = subset[subset['condition'] == 'normal']['polya_length'].dropna()
    stressed = subset[subset['condition'] == 'stress']['polya_length'].dropna()
    if len(baseline) > 0 and len(stressed) > 0:
        delta = stressed.median() - baseline.median()
        deltas.append(delta)
        delta_labels.append(grp)
        delta_ns.append(f'{len(baseline)}/{len(stressed)}')
    else:
        deltas.append(0)
        delta_labels.append(grp)
        delta_ns.append('insuf.')

colors = ['#e74c3c' if d < 0 else '#3498db' for d in deltas]
bars = ax.bar(range(len(delta_labels)), deltas, color=colors, alpha=0.8)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(delta_labels)))
ax.set_xticklabels(delta_labels, rotation=15, fontsize=9)
ax.set_ylabel('Δ Poly(A) (nt)')
ax.set_title('(e) HeLa Stress Poly(A) Change', fontsize=10, fontweight='bold')
for i, (d, n) in enumerate(zip(deltas, delta_ns)):
    ax.text(i, d + (2 if d >= 0 else -4), f'Δ={d:.0f}\n({n})', ha='center', fontsize=7, va='bottom' if d >= 0 else 'top')

# Panel F: 6_EnhG vs 7_Enh poly(A) comparison
ax = fig.add_subplot(gs[1, 2])
enhg = normal[normal['chromhmm_state'] == '6_EnhG']['polya_length'].dropna()
enh7 = normal[normal['chromhmm_state'] == '7_Enh']['polya_length'].dropna()
parts = ax.violinplot([enhg, enh7], positions=[0, 1], showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#e67e22', '#e74c3c'][i])
    pc.set_alpha(0.7)
ax.set_xticks([0, 1])
ax.set_xticklabels(['6_EnhG\n(Genic)', '7_Enh\n(Intergenic)'], fontsize=9)
ax.set_ylabel('Poly(A) Length (nt)')
p = mwu_p(enhg, enh7)
ax.set_title(f'(f) Enhancer Subtypes Poly(A)\nP={p:.2e}', fontsize=10, fontweight='bold')
ax.text(0, enhg.max() * 0.95, f'n={len(enhg)}', ha='center', fontsize=8)
ax.text(1, enh7.max() * 0.95, f'n={len(enh7)}', ha='center', fontsize=8)

# Panel G: Reads/locus vs poly(A) scatter (enhancer L1)
ax = fig.add_subplot(gs[2, 0])
enh_plot = enh_all.dropna(subset=['polya_length', 'reads_at_locus'])
ax.scatter(enh_plot['reads_at_locus'], enh_plot['polya_length'], alpha=0.3, s=10, color='#e74c3c')
r_val, p_val = stats.spearmanr(enh_plot['reads_at_locus'], enh_plot['polya_length'])
ax.set_xlabel('Reads at Locus')
ax.set_ylabel('Poly(A) Length (nt)')
ax.set_title(f'(g) Expression vs Poly(A) (Enhancer L1)\nρ={r_val:.3f}, P={p_val:.2e}', fontsize=10, fontweight='bold')

# Panel H: Enhancer vs Transcribed ECDF of poly(A)
ax = fig.add_subplot(gs[2, 1])
for grp, color, ls in [('Enhancer', '#e74c3c', '-'), ('Transcribed', '#3498db', '--'), ('Quiescent', '#95a5a6', ':')]:
    vals = normal[normal['chromhmm_group'] == grp]['polya_length'].dropna().sort_values()
    ecdf = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, ecdf, color=color, linestyle=ls, label=f'{grp} (n={len(vals)})', linewidth=1.5)
ax.set_xlabel('Poly(A) Length (nt)')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('(h) Poly(A) ECDF by Chromatin State', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(0, 300)

# Panel I: m6A/kb ECDF
ax = fig.add_subplot(gs[2, 2])
for grp, color, ls in [('Enhancer', '#e74c3c', '-'), ('Transcribed', '#3498db', '--'), ('Quiescent', '#95a5a6', ':')]:
    vals = normal[normal['chromhmm_group'] == grp]['m6a_per_kb'].dropna().sort_values()
    ecdf = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, ecdf, color=color, linestyle=ls, label=f'{grp} (n={len(vals)})', linewidth=1.5)
ax.set_xlabel('m6A/kb')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('(i) m6A/kb ECDF by Chromatin State', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(0, 10)

# Panel J: Enhancer L1 age composition vs other groups
ax = fig.add_subplot(gs[3, 0])
age_comp = normal.groupby(['chromhmm_group', 'l1_age']).size().unstack(fill_value=0)
age_frac = age_comp.div(age_comp.sum(axis=1), axis=0)
if 'ancient' in age_frac.columns and 'young' in age_frac.columns:
    for i, grp in enumerate(groups):
        if grp in age_frac.index:
            ax.bar(i, age_frac.loc[grp, 'ancient'], 0.6, color='#95a5a6', label='Ancient' if i == 0 else '')
            ax.bar(i, age_frac.loc[grp, 'young'], 0.6, bottom=age_frac.loc[grp, 'ancient'], color='#e74c3c', label='Young' if i == 0 else '')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, rotation=15, fontsize=9)
ax.set_ylabel('Fraction')
ax.set_title('(j) L1 Age Composition', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

# Panel K: HeLa stress delta by age and chromatin state
ax = fig.add_subplot(gs[3, 1])
bar_data = []
bar_labels = []
bar_colors = []
for grp in ['Enhancer', 'Transcribed', 'Quiescent']:
    for age in ['ancient', 'young']:
        subset = hela_all[(hela_all['chromhmm_group'] == grp) & (hela_all['l1_age'] == age)]
        baseline = subset[subset['condition'] == 'normal']['polya_length'].dropna()
        stressed = subset[subset['condition'] == 'stress']['polya_length'].dropna()
        if len(baseline) >= 5 and len(stressed) >= 5:
            delta = stressed.median() - baseline.median()
            bar_data.append(delta)
            bar_labels.append(f'{grp[:3]}\n{age[:3]}')
            bar_colors.append('#e74c3c' if age == 'ancient' else '#3498db')
        else:
            bar_data.append(0)
            bar_labels.append(f'{grp[:3]}\n{age[:3]}')
            bar_colors.append('#cccccc')

ax.bar(range(len(bar_data)), bar_data, color=bar_colors, alpha=0.8)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(bar_labels)))
ax.set_xticklabels(bar_labels, fontsize=7)
ax.set_ylabel('Δ Poly(A) (nt)')
ax.set_title('(k) Stress Δ by Chromatin × Age', fontsize=10, fontweight='bold')

# Panel L: Genomic context of enhancer L1
ax = fig.add_subplot(gs[3, 2])
enh_ctx = normal[normal['chromhmm_group'] == 'Enhancer']['genomic_context'].value_counts()
ax.pie(enh_ctx.values, labels=enh_ctx.index, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
ax.set_title('(l) Enhancer L1 Genomic Context', fontsize=10, fontweight='bold')

plt.savefig(OUTDIR / 'enhancer_l1_erna_analysis.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved: {OUTDIR / 'enhancer_l1_erna_analysis.pdf'}")

# ── 7. Save summary table ──
summary_rows = []
for grp in ['Enhancer', 'Transcribed', 'Quiescent', 'Promoter', 'Repressed']:
    subset = normal[normal['chromhmm_group'] == grp]
    if len(subset) == 0:
        continue
    row = {
        'chromhmm_group': grp,
        'n_reads': len(subset),
        'ancient_frac': (subset['l1_age'] == 'ancient').mean(),
        'intronic_frac': (subset['genomic_context'] == 'intronic').mean(),
        'read_length_median': subset['read_length'].median(),
        'polya_median': subset['polya_length'].median(),
        'm6a_per_kb_median': subset['m6a_per_kb'].median(),
    }
    # Antisense fraction (intronic only)
    intr = subset[(subset['genomic_context'] == 'intronic') & (subset['orientation'].isin(['sense', 'antisense']))]
    if len(intr) > 0:
        row['antisense_frac'] = (intr['orientation'] == 'antisense').mean()
        row['n_orientation'] = len(intr)
    else:
        row['antisense_frac'] = np.nan
        row['n_orientation'] = 0
    # HeLa stress delta
    hela_grp = hela_all[hela_all['chromhmm_group'] == grp]
    bl = hela_grp[hela_grp['condition'] == 'normal']['polya_length'].dropna()
    st = hela_grp[hela_grp['condition'] == 'stress']['polya_length'].dropna()
    if len(bl) >= 5 and len(st) >= 5:
        row['hela_polya_delta'] = st.median() - bl.median()
        row['hela_delta_p'] = mwu_p(bl, st)
        row['hela_n_normal'] = len(bl)
        row['hela_n_stress'] = len(st)
    else:
        row['hela_polya_delta'] = np.nan
        row['hela_delta_p'] = np.nan
        row['hela_n_normal'] = len(bl)
        row['hela_n_stress'] = len(st)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTDIR / 'enhancer_l1_summary.tsv', sep='\t', index=False)
print(f"  Saved: {OUTDIR / 'enhancer_l1_summary.tsv'}")

# ── 8. Print final verdict ──
print("\n" + "="*70)
print("VERDICT: Does Enhancer L1 Show eRNA-Like Properties?")
print("="*70)

enh_pa_med = normal[normal['chromhmm_group'] == 'Enhancer']['polya_length'].median()
tx_pa_med = normal[normal['chromhmm_group'] == 'Transcribed']['polya_length'].median()
enh_rl_med = normal[normal['chromhmm_group'] == 'Enhancer']['read_length'].median()
tx_rl_med = normal[normal['chromhmm_group'] == 'Transcribed']['read_length'].median()

print(f"\n1. Short transcripts: Enhancer L1 read length {enh_rl_med:.0f} vs Transcribed {tx_rl_med:.0f}")
print(f"   → {'YES' if enh_rl_med < tx_rl_med else 'NO'}: Enhancer L1 {'shorter' if enh_rl_med < tx_rl_med else 'not shorter'}")

print(f"\n2. Short poly(A): Enhancer {enh_pa_med:.1f} vs Transcribed {tx_pa_med:.1f}")
print(f"   → {'YES' if enh_pa_med < tx_pa_med else 'NO'}: Enhancer L1 {'shorter' if enh_pa_med < tx_pa_med else 'not shorter'} poly(A)")

enh_anti = enh_intronic[enh_intronic['orientation'].isin(['sense', 'antisense'])]
if len(enh_anti) > 0:
    anti_frac = (enh_anti['orientation'] == 'antisense').mean()
    print(f"\n3. Antisense orientation: {anti_frac:.1%} of enhancer intronic L1")
    print(f"   → {'Elevated' if anti_frac > 0.55 else 'Similar to'} vs expected ~50%")

# HeLa enhancer stress delta
hela_enh_bl = hela_all[(hela_all['chromhmm_group'] == 'Enhancer') & (hela_all['condition'] == 'normal')]['polya_length'].dropna()
hela_enh_st = hela_all[(hela_all['chromhmm_group'] == 'Enhancer') & (hela_all['condition'] == 'stress')]['polya_length'].dropna()
if len(hela_enh_bl) > 0 and len(hela_enh_st) > 0:
    delta = hela_enh_st.median() - hela_enh_bl.median()
    print(f"\n4. Stress poly(A) shortening: Enhancer Δ={delta:.1f}nt")
    print(f"   → {'STRONG' if abs(delta) > 30 else 'MODERATE' if abs(delta) > 15 else 'WEAK'} shortening under stress")

print("\n" + "="*70)
print("Done.")
