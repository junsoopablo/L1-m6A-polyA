#!/usr/bin/env python3
"""
Analyze L1 subfamily expression changes under arsenite stress (v2).
Proper normalization: L1 reads / total mapped reads (from STAR Log.final.out).
GSE278916 (Liu et al. Cell 2025): HeLa ribo-depleted RNA-seq, UN vs SA.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC = PROJECT / 'analysis/01_exploration/topic_10_rnaseq_validation'
BAM_DIR = Path('/scratch1/junsoopablo/GSE278916_alignment')
OUTPUT = TOPIC / 'rnaseq_validation_results'
OUTPUT.mkdir(exist_ok=True)

YOUNG_L1 = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

def classify_l1_age(subfamily):
    if subfamily in YOUNG_L1:
        return 'young'
    if subfamily.startswith('L1') or subfamily.startswith('HAL1'):
        return 'ancient'
    return 'other'

# ==============================
# 1. Parse STAR mapping statistics
# ==============================
print("=" * 70)
print("1. STAR mapping statistics")
print("=" * 70)

samples = {
    'HeLa_UN_rep1': 'SRR30897703',
    'HeLa_UN_rep2': 'SRR30897702',
    'HeLa_SA_rep1': 'SRR30897698',
    'HeLa_SA_rep2': 'SRR30897697',
}

star_stats = {}
for sample in samples:
    log_file = BAM_DIR / f'{sample}_Log.final.out'
    with open(log_file) as f:
        stats_dict = {}
        for line in f:
            line = line.strip()
            if '|' in line:
                key, val = line.split('|', 1)
                stats_dict[key.strip()] = val.strip()
        star_stats[sample] = {
            'input_reads': int(stats_dict['Number of input reads']),
            'unique_mapped': int(stats_dict['Uniquely mapped reads number']),
            'unique_pct': float(stats_dict['Uniquely mapped reads %'].rstrip('%')),
            'multi_mapped': int(stats_dict['Number of reads mapped to multiple loci']),
            'multi_pct': float(stats_dict['% of reads mapped to multiple loci'].rstrip('%')),
        }
        star_stats[sample]['total_mapped'] = (
            star_stats[sample]['unique_mapped'] + star_stats[sample]['multi_mapped']
        )

for s, st in star_stats.items():
    cond = 'SA' if 'SA' in s else 'UN'
    print(f"  {s}: {st['input_reads']:,} input, {st['unique_pct']:.1f}% unique, "
          f"{st['multi_pct']:.1f}% multi, {st['total_mapped']:,} total mapped")

# ==============================
# 2. Load featureCounts L1 results
# ==============================
print("\n" + "=" * 70)
print("2. featureCounts L1 quantification")
print("=" * 70)

fc_file = OUTPUT / 'featurecounts_L1.txt'
fc = pd.read_csv(fc_file, sep='\t', comment='#')

# Map BAM column names to sample names
bam_cols = [c for c in fc.columns if 'Aligned' in c]
col_map = {}
for col in bam_cols:
    for name in samples:
        if name in col:
            col_map[col] = name
fc = fc.rename(columns=col_map)
sample_names = list(samples.keys())

# Classify L1 age
fc['age'] = fc['Geneid'].apply(classify_l1_age)
fc = fc[fc['age'].isin(['young', 'ancient'])]

print(f"  L1 subfamilies: {len(fc)} (young={sum(fc.age=='young')}, ancient={sum(fc.age=='ancient')})")

# ==============================
# 3. Per-replicate L1 fraction of total mapped reads
# ==============================
print("\n" + "=" * 70)
print("3. L1 fraction of total mapped reads (per replicate)")
print("=" * 70)

results = []
for sample in sample_names:
    total_mapped = star_stats[sample]['total_mapped']
    condition = 'SA' if 'SA' in sample else 'UN'
    rep = sample.split('_')[-1]

    for age in ['young', 'ancient']:
        sub = fc[fc['age'] == age]
        l1_reads = sub[sample].sum()
        fraction = l1_reads / total_mapped
        rpm = l1_reads / (total_mapped / 1e6)
        results.append({
            'sample': sample, 'condition': condition, 'replicate': rep,
            'age': age, 'l1_reads': l1_reads, 'total_mapped': total_mapped,
            'fraction': fraction, 'rpm': rpm
        })

    # Total L1
    total_l1 = fc[sample].sum()
    results.append({
        'sample': sample, 'condition': condition, 'replicate': rep,
        'age': 'all_L1', 'l1_reads': total_l1, 'total_mapped': total_mapped,
        'fraction': total_l1 / total_mapped, 'rpm': total_l1 / (total_mapped / 1e6)
    })

res_df = pd.DataFrame(results)

for age in ['young', 'ancient', 'all_L1']:
    print(f"\n  {age.upper()}:")
    for _, row in res_df[res_df['age'] == age].iterrows():
        print(f"    {row['sample']}: {row['l1_reads']:,.0f} reads, "
              f"{row['fraction']*100:.4f}% of mapped, {row['rpm']:.1f} RPM")

    un = res_df[(res_df['age'] == age) & (res_df['condition'] == 'UN')]
    sa = res_df[(res_df['age'] == age) & (res_df['condition'] == 'SA')]
    un_mean = un['fraction'].mean()
    sa_mean = sa['fraction'].mean()
    fc_ratio = sa_mean / un_mean if un_mean > 0 else np.nan
    print(f"    UN mean: {un_mean*100:.4f}%, SA mean: {sa_mean*100:.4f}%")
    print(f"    FC (SA/UN): {fc_ratio:.4f}x (log2={np.log2(fc_ratio):.3f})")

# ==============================
# 4. Gene-level normalization (use STAR GeneCounts)
# ==============================
print("\n" + "=" * 70)
print("4. Gene-level normalization (STAR ReadsPerGene)")
print("=" * 70)

gene_totals = {}
for sample in sample_names:
    rpg_file = BAM_DIR / f'{sample}_ReadsPerGene.out.tab'
    rpg = pd.read_csv(rpg_file, sep='\t', header=None,
                      names=['gene', 'unstranded', 'sense', 'antisense'])
    # Skip first 4 rows (N_unmapped, N_multimapping, N_noFeature, N_ambiguous)
    rpg = rpg.iloc[4:]
    gene_totals[sample] = rpg['unstranded'].sum()
    print(f"  {sample}: {gene_totals[sample]:,} gene-assigned reads")

# DESeq2-like size factors (median of ratios)
gene_counts = {}
for sample in sample_names:
    rpg_file = BAM_DIR / f'{sample}_ReadsPerGene.out.tab'
    rpg = pd.read_csv(rpg_file, sep='\t', header=None,
                      names=['gene', 'unstranded', 'sense', 'antisense'])
    rpg = rpg.iloc[4:].set_index('gene')
    gene_counts[sample] = rpg['unstranded']

gene_df = pd.DataFrame(gene_counts)
# Geometric mean per gene (filter zeros)
gene_df_nz = gene_df[(gene_df > 0).all(axis=1)]
geo_mean = gene_df_nz.apply(np.log).mean(axis=1).apply(np.exp)
ratios = gene_df_nz.div(geo_mean, axis=0)
size_factors = ratios.median()
print(f"\n  DESeq2-like size factors:")
for s, sf in size_factors.items():
    print(f"    {s}: {sf:.4f}")

# Normalize L1 by gene size factors
print(f"\n  L1 reads normalized by gene size factors:")
for age in ['young', 'ancient', 'all_L1']:
    un_norm = []
    sa_norm = []
    for _, row in res_df[res_df['age'] == age].iterrows():
        norm_count = row['l1_reads'] / size_factors[row['sample']]
        if row['condition'] == 'UN':
            un_norm.append(norm_count)
        else:
            sa_norm.append(norm_count)
    un_mean = np.mean(un_norm)
    sa_mean = np.mean(sa_norm)
    fc_ratio = sa_mean / un_mean if un_mean > 0 else np.nan
    print(f"  {age.upper()}: UN={un_mean:,.0f}, SA={sa_mean:,.0f}, "
          f"FC={fc_ratio:.4f}x (log2={np.log2(fc_ratio):.3f})")

# ==============================
# 5. Per-subfamily analysis
# ==============================
print("\n" + "=" * 70)
print("5. Per-subfamily fold changes (normalized by gene size factors)")
print("=" * 70)

un_cols = [c for c in sample_names if 'UN' in c]
sa_cols = [c for c in sample_names if 'SA' in c]

# Normalize each sample by size factor
for s in sample_names:
    fc[f'{s}_norm'] = fc[s] / size_factors[s]

fc['UN_norm'] = fc[[f'{s}_norm' for s in un_cols]].mean(axis=1)
fc['SA_norm'] = fc[[f'{s}_norm' for s in sa_cols]].mean(axis=1)
fc['log2FC'] = np.log2((fc['SA_norm'] + 0.5) / (fc['UN_norm'] + 0.5))

# Filter to subfamilies with sufficient reads
fc['total'] = fc[sample_names].sum(axis=1)
fc_sig = fc[fc['total'] >= 50].copy()
print(f"  Subfamilies with ≥50 total reads: {len(fc_sig)}")

for age in ['young', 'ancient']:
    sub = fc_sig[fc_sig['age'] == age].sort_values('log2FC')
    print(f"\n  {age.upper()} L1 (n={len(sub)}):")
    print(f"    Median log2FC: {sub['log2FC'].median():.3f} ({2**sub['log2FC'].median():.3f}x)")
    print(f"    Mean log2FC: {sub['log2FC'].mean():.3f}")
    for _, row in sub.head(3).iterrows():
        print(f"    DOWN: {row['Geneid']:15s} log2FC={row['log2FC']:.3f} (UN={row['UN_norm']:.0f}, SA={row['SA_norm']:.0f})")
    for _, row in sub.tail(3).iterrows():
        print(f"    UP:   {row['Geneid']:15s} log2FC={row['log2FC']:.3f} (UN={row['UN_norm']:.0f}, SA={row['SA_norm']:.0f})")

# ==============================
# 6. Young vs Ancient comparison
# ==============================
print("\n" + "=" * 70)
print("6. Young vs Ancient fold change comparison")
print("=" * 70)

young_lfc = fc_sig[fc_sig['age'] == 'young']['log2FC'].values
ancient_lfc = fc_sig[fc_sig['age'] == 'ancient']['log2FC'].values

if len(young_lfc) > 0 and len(ancient_lfc) > 0:
    u_stat, p_mw = stats.mannwhitneyu(young_lfc, ancient_lfc, alternative='two-sided')
    print(f"  Young median log2FC: {np.median(young_lfc):.3f}")
    print(f"  Ancient median log2FC: {np.median(ancient_lfc):.3f}")
    print(f"  Mann-Whitney U={u_stat:.0f}, P={p_mw:.2e}")

# Total young / total ancient fold change
for age, label in [('young', 'Young'), ('ancient', 'Ancient')]:
    sub = fc[fc['age'] == age]
    un_total = sub['UN_norm'].sum()
    sa_total = sub['SA_norm'].sum()
    total_fc = sa_total / un_total if un_total > 0 else np.nan
    print(f"  {label} total FC: {total_fc:.4f}x (log2={np.log2(total_fc):.3f})")

# ==============================
# 7. Comparison with DRS findings
# ==============================
print("\n" + "=" * 70)
print("7. Comparison with DRS findings")
print("=" * 70)

print("""
  DRS (ONT, this study):
    - L1 overall RPM: HeLa 293.9 → HeLa-Ars 522.8 = 1.78x ↑
    - Young L1 poly(A): Δ = 0 (immune)
    - Ancient L1 poly(A): Δ = -31 nt (shortened → decay)
    - Interpretation: L1 reads INCREASE despite decay → read capture bias
      (DRS poly(A) selection captures degradation intermediates)

  RNA-seq (Illumina, GSE278916, this analysis):
    - Ribo-depleted → captures both poly(A)+ and poly(A)- RNA
    - L1 fraction change under arsenite: see above
    - Young vs Ancient: see above
""")

# ==============================
# 8. Comparison with host genes (stress markers)
# ==============================
print("=" * 70)
print("8. Stress marker gene expression changes")
print("=" * 70)

# Check known stress-responsive genes
stress_genes = ['HSPA1A', 'HSPA1B', 'HSPA6', 'DDIT3', 'ATF3', 'GADD45A',
                'HMOX1', 'SLC7A11', 'NQO1', 'SQSTM1', 'BAG3']
gene_results = []
for sample in sample_names:
    rpg_file = BAM_DIR / f'{sample}_ReadsPerGene.out.tab'
    rpg = pd.read_csv(rpg_file, sep='\t', header=None,
                      names=['gene', 'unstranded', 'sense', 'antisense'])
    rpg = rpg.iloc[4:].set_index('gene')
    gene_results.append(rpg['unstranded'])

gene_all = pd.concat(gene_results, axis=1, keys=sample_names)
gene_norm = gene_all.div(size_factors)

for gene in stress_genes:
    matches = [g for g in gene_norm.index if gene in g]
    if matches:
        g = matches[0]
        un_mean = gene_norm.loc[g, un_cols].mean()
        sa_mean = gene_norm.loc[g, sa_cols].mean()
        if un_mean > 10:
            fc_val = sa_mean / un_mean
            print(f"  {gene:12s}: UN={un_mean:,.0f}, SA={sa_mean:,.0f}, FC={fc_val:.2f}x")

# ==============================
# 9. Generate publication figure
# ==============================
print("\n" + "=" * 70)
print("9. Generating figure")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('RNA-seq validation: L1 expression under arsenite\n'
             '(GSE278916, HeLa, ribo-depleted, Liu et al. Cell 2025)',
             fontsize=12, fontweight='bold', y=0.98)

# (a) L1 fraction of total mapped reads
ax = axes[0, 0]
for i, age in enumerate(['young', 'ancient', 'all_L1']):
    sub = res_df[res_df['age'] == age]
    un = sub[sub['condition'] == 'UN']['fraction'].values * 100
    sa = sub[sub['condition'] == 'SA']['fraction'].values * 100
    x_pos = i * 3
    ax.bar(x_pos, un.mean(), 0.8, color='#2196F3', alpha=0.7, label='UN' if i == 0 else '')
    ax.bar(x_pos + 1, sa.mean(), 0.8, color='#F44336', alpha=0.7, label='SA' if i == 0 else '')
    # Replicate dots
    ax.scatter([x_pos]*len(un), un, color='black', s=25, zorder=5)
    ax.scatter([x_pos+1]*len(sa), sa, color='black', s=25, zorder=5)
    # FC annotation
    fc_val = sa.mean() / un.mean()
    ymax = max(un.max(), sa.max())
    ax.text(x_pos + 0.5, ymax * 1.05, f'{fc_val:.2f}x', ha='center', fontsize=8, fontweight='bold')

ax.set_xticks([0.5, 3.5, 6.5])
ax.set_xticklabels(['Young L1', 'Ancient L1', 'All L1'])
ax.set_ylabel('% of total mapped reads')
ax.legend(fontsize=9)
ax.set_title('L1 fraction of mapped reads', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.08, 1.05, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (b) Per-subfamily log2FC distribution
ax = axes[0, 1]
for age, color in [('young', '#4CAF50'), ('ancient', '#8D6E63')]:
    sub = fc_sig[fc_sig['age'] == age]
    if len(sub) > 0:
        ax.hist(sub['log2FC'], bins=25, alpha=0.6, color=color,
                label=f'{age.capitalize()} (n={len(sub)})', edgecolor='white')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('log₂(SA/UN)', fontsize=10)
ax.set_ylabel('Number of subfamilies', fontsize=10)
ax.set_title('Per-subfamily expression change', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.08, 1.05, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (c) UN vs SA scatter for individual subfamilies
ax = axes[1, 0]
for age, color, marker in [('ancient', '#8D6E63', 'o'), ('young', '#4CAF50', '^')]:
    sub = fc_sig[fc_sig['age'] == age]
    ax.scatter(np.log10(sub['UN_norm'] + 1), np.log10(sub['SA_norm'] + 1),
              c=color, alpha=0.5, s=30 if age == 'ancient' else 80, marker=marker,
              label=age.capitalize(), edgecolors='white', linewidths=0.3)
    # Label young L1
    if age == 'young':
        for _, row in sub.iterrows():
            ax.annotate(row['Geneid'], (np.log10(row['UN_norm']+1), np.log10(row['SA_norm']+1)),
                       fontsize=7, ha='left', va='bottom')

maxval = max(np.log10(fc_sig['UN_norm'].max() + 1), np.log10(fc_sig['SA_norm'].max() + 1))
ax.plot([0, maxval+0.2], [0, maxval+0.2], 'k--', alpha=0.3)
ax.set_xlabel('log₁₀(UN normalized count + 1)', fontsize=10)
ax.set_ylabel('log₁₀(SA normalized count + 1)', fontsize=10)
ax.set_title('Subfamily expression: UN vs SA', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.08, 1.05, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (d) Comparison: RNA-seq FC vs DRS poly(A) delta
ax = axes[1, 1]
categories = ['Young L1\n(RNA-seq)', 'Ancient L1\n(RNA-seq)',
              'Young L1\n(DRS poly(A))', 'Ancient L1\n(DRS poly(A))']
young_rnaseq_fc = fc[fc['age'] == 'young']['SA_norm'].sum() / fc[fc['age'] == 'young']['UN_norm'].sum()
ancient_rnaseq_fc = fc[fc['age'] == 'ancient']['SA_norm'].sum() / fc[fc['age'] == 'ancient']['UN_norm'].sum()

rnaseq_vals = [young_rnaseq_fc, ancient_rnaseq_fc]
# DRS: L1 RPM increased under arsenite (HeLa -> HeLa-Ars)
drs_vals = [1.0, 1.0]  # placeholder for visual

colors_bar = ['#4CAF50', '#8D6E63', '#4CAF50', '#8D6E63']
x = [0, 1]
bars = ax.bar(x, rnaseq_vals, 0.6, color=colors_bar[:2], alpha=0.8, edgecolor='white')
ax.axhline(1.0, color='grey', linestyle='--', alpha=0.5, label='No change')

for i, (val, cat) in enumerate(zip(rnaseq_vals, ['Young', 'Ancient'])):
    ax.text(i, val + 0.01, f'{val:.3f}x', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['Young L1', 'Ancient L1'])
ax.set_ylabel('Fold change (SA/UN)', fontsize=10)
ax.set_title('RNA-seq expression FC under arsenite', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add text box with DRS comparison
textstr = ('DRS comparison:\n'
           f'• Young poly(A): Δ=0 (immune)\n'
           f'• Ancient poly(A): Δ=-31nt (decay)\n'
           f'• RNA-seq shows L1 RNA\n'
           f'  level change under stress')
ax.text(0.98, 0.45, textstr, transform=ax.transAxes, fontsize=7.5,
        va='center', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9, edgecolor='#ccc'))
ax.text(-0.08, 1.05, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = OUTPUT / 'rnaseq_l1_arsenite_v2.pdf'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# Save summary
summary = fc_sig[['Geneid', 'age', 'Length', 'total'] + sample_names +
                 ['UN_norm', 'SA_norm', 'log2FC']].sort_values('log2FC')
summary.to_csv(OUTPUT / 'l1_subfamily_expression_v2.tsv', sep='\t', index=False)
print(f"  Saved: {OUTPUT / 'l1_subfamily_expression_v2.tsv'}")

print("\n===== Analysis v2 complete =====")
