#!/usr/bin/env python3
"""
Analyze L1 subfamily expression changes under arsenite stress
from GSE278916 (Liu et al. Cell 2025) RNA-seq data.

Uses TEtranscripts output (DESeq2-based) + STAR GeneCounts for validation.
Compares with DRS findings: ancient L1 poly(A) shortening (decay) vs young L1 immunity.

Expected: Ancient L1 ↓ under arsenite, Young L1 stable/immune.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC = PROJECT / 'analysis/01_exploration/topic_10_rnaseq_validation'
TET_DIR = TOPIC / 'tetranscripts_output'
BAM_DIR = Path('/scratch1/junsoopablo/GSE278916_alignment')
OUTPUT = TOPIC / 'rnaseq_validation_results'
OUTPUT.mkdir(exist_ok=True)

# L1 classification
YOUNG_L1 = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
ANCIENT_PREFIXES = ['L1MC', 'L1ME', 'L1M', 'L1MA', 'L1MB', 'L1PB', 'L1PA4', 'L1PA5',
                    'L1PA6', 'L1PA7', 'L1PA8', 'L1PA9', 'L1PA10', 'L1PA11', 'L1PA12',
                    'L1PA13', 'L1PA14', 'L1PA15', 'L1PA16', 'L1PA17', 'HAL1', 'L1P']

def classify_l1_age(subfamily):
    """Classify L1 subfamily as young/ancient/other."""
    if subfamily in YOUNG_L1:
        return 'young'
    for prefix in ANCIENT_PREFIXES:
        if subfamily.startswith(prefix):
            return 'ancient'
    if subfamily.startswith('L1'):
        return 'ancient'  # remaining L1 are ancient
    return 'other'

# =============================
# 1. Load TEtranscripts output
# =============================
print("=" * 60)
print("1. TEtranscripts DESeq2 results")
print("=" * 60)

# TEtranscripts produces a DESeq2 table with gene_id, baseMean, log2FoldChange, pvalue, padj
tet_file = TET_DIR / 'HeLa_SA_vs_UN_DEresult.csv'
if not tet_file.exists():
    # Try alternative naming
    candidates = list(TET_DIR.glob('*DEresult*'))
    if candidates:
        tet_file = candidates[0]
    else:
        print(f"  WARNING: No TEtranscripts output found in {TET_DIR}")
        tet_file = None

if tet_file and tet_file.exists():
    de = pd.read_csv(tet_file)
    print(f"  Loaded {len(de)} features from {tet_file.name}")

    # Filter to L1 family (LINE/L1)
    l1_de = de[de.iloc[:, 0].str.startswith('L1')].copy()
    l1_de.columns = ['gene_id'] + list(l1_de.columns[1:])
    l1_de['age'] = l1_de['gene_id'].apply(classify_l1_age)
    l1_de = l1_de[l1_de['age'].isin(['young', 'ancient'])]

    print(f"  L1 subfamilies: {len(l1_de)} (young={sum(l1_de['age']=='young')}, ancient={sum(l1_de['age']=='ancient')})")

    # Summary by age
    for age in ['young', 'ancient']:
        sub = l1_de[l1_de['age'] == age]
        if 'log2FoldChange' in sub.columns:
            sig = sub[sub['padj'] < 0.05] if 'padj' in sub.columns else sub[sub['pvalue'] < 0.05]
            up = sig[sig['log2FoldChange'] > 0]
            down = sig[sig['log2FoldChange'] < 0]
            median_lfc = sub['log2FoldChange'].median()
            print(f"\n  {age.upper()} L1 (n={len(sub)} subfamilies):")
            print(f"    Median log2FC: {median_lfc:.3f} ({2**median_lfc:.3f}x)")
            print(f"    Significant (padj<0.05): {len(sig)} ({len(up)} up, {len(down)} down)")
            if len(sub) > 0:
                print(f"    Range: {sub['log2FoldChange'].min():.3f} to {sub['log2FoldChange'].max():.3f}")
else:
    print("  No TEtranscripts DESeq2 output. Proceeding with STAR GeneCounts.")

# =============================
# 2. Load STAR GeneCounts for independent verification
# =============================
print("\n" + "=" * 60)
print("2. STAR GeneCounts + featureCounts approach")
print("=" * 60)

# Load TE GTF to get L1 subfamily -> family mapping
te_gtf = PROJECT / 'reference/hg38_rmsk_TE.gtf'
print(f"  Loading TE annotation from {te_gtf.name}...")
te_info = []
with open(te_gtf) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        attrs = parts[8]
        gene_id = attrs.split('gene_id "')[1].split('"')[0] if 'gene_id' in attrs else ''
        family_id = attrs.split('family_id "')[1].split('"')[0] if 'family_id' in attrs else ''
        class_id = attrs.split('class_id "')[1].split('"')[0] if 'class_id' in attrs else ''
        if class_id == 'LINE' and family_id == 'L1':
            te_info.append({'subfamily': gene_id, 'family': family_id, 'class': class_id})

te_df = pd.DataFrame(te_info)
l1_subfamilies = te_df['subfamily'].unique()
print(f"  L1 subfamilies in annotation: {len(l1_subfamilies)}")

# Classify each
sf_age = {sf: classify_l1_age(sf) for sf in l1_subfamilies}
n_young = sum(1 for v in sf_age.values() if v == 'young')
n_ancient = sum(1 for v in sf_age.values() if v == 'ancient')
print(f"  Young: {n_young}, Ancient: {n_ancient}")

# Count L1 elements per subfamily in annotation
l1_element_counts = te_df['subfamily'].value_counts()
print(f"\n  Top 10 L1 subfamilies by genomic copies:")
for sf in l1_element_counts.head(10).index:
    age = sf_age.get(sf, 'other')
    print(f"    {sf}: {l1_element_counts[sf]:,} elements ({age})")

# =============================
# 3. Count reads per TE using featureCounts (alternative to TEtranscripts)
# =============================
print("\n" + "=" * 60)
print("3. featureCounts-based TE quantification")
print("=" * 60)

# Use featureCounts SAF format for L1 elements
# This is a backup approach if TEtranscripts didn't work
# We'll generate per-subfamily read counts

saf_file = OUTPUT / 'L1_elements.saf'
if not saf_file.exists():
    print("  Generating L1 SAF annotation...")
    with open(te_gtf) as fin, open(saf_file, 'w') as fout:
        fout.write("GeneID\tChr\tStart\tEnd\tStrand\n")
        for line in fin:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            attrs = parts[8]
            family_id = attrs.split('family_id "')[1].split('"')[0] if 'family_id' in attrs else ''
            class_id = attrs.split('class_id "')[1].split('"')[0] if 'class_id' in attrs else ''
            if class_id == 'LINE' and family_id == 'L1':
                gene_id = attrs.split('gene_id "')[1].split('"')[0]
                chrom = parts[0]
                start = parts[3]
                end = parts[4]
                strand = parts[6]
                fout.write(f"{gene_id}\t{chrom}\t{start}\t{end}\t{strand}\n")
    print(f"  SAF file written: {saf_file}")

# Run featureCounts
fc_output = OUTPUT / 'featurecounts_L1.txt'
samples = {
    'HeLa_UN_rep1': BAM_DIR / 'HeLa_UN_rep1_Aligned.sortedByCoord.out.bam',
    'HeLa_UN_rep2': BAM_DIR / 'HeLa_UN_rep2_Aligned.sortedByCoord.out.bam',
    'HeLa_SA_rep1': BAM_DIR / 'HeLa_SA_rep1_Aligned.sortedByCoord.out.bam',
    'HeLa_SA_rep2': BAM_DIR / 'HeLa_SA_rep2_Aligned.sortedByCoord.out.bam',
}

bam_exists = all(p.exists() for p in samples.values())
if not bam_exists:
    print("  BAM files not yet available. Will run featureCounts later.")
    print("  Missing:", [str(p) for p in samples.values() if not p.exists()])
else:
    import subprocess
    bam_list = ' '.join(str(p) for p in samples.values())
    cmd = (f"module load subread/2.0.6 && featureCounts "
           f"-F SAF -a {saf_file} "
           f"-o {fc_output} "
           f"-T 16 "
           f"--fraction "       # fractional counting for multi-mappers
           f"-M "               # count multi-mapping reads
           f"-p --countReadPairs "  # paired-end
           f"--primary "        # primary alignments only
           f"{bam_list}")
    print(f"  Running featureCounts...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
    else:
        print(f"  featureCounts complete: {fc_output}")

# =============================
# 4. Analyze L1 expression changes
# =============================
print("\n" + "=" * 60)
print("4. L1 expression change analysis")
print("=" * 60)

if fc_output.exists():
    fc = pd.read_csv(fc_output, sep='\t', comment='#')
    # featureCounts columns: Geneid, Chr, Start, End, Strand, Length, then sample BAMs
    bam_cols = [c for c in fc.columns if 'Aligned' in c]
    # Rename to sample names
    col_map = {}
    for col in bam_cols:
        for name, path in samples.items():
            if name in col:
                col_map[col] = name
    fc = fc.rename(columns=col_map)

    sample_names = list(samples.keys())
    available_names = [n for n in sample_names if n in fc.columns]
    if not available_names:
        print("  ERROR: Could not match BAM columns to sample names")
        print(f"  Available columns: {fc.columns.tolist()}")
    else:
        # Classify L1
        fc['age'] = fc['Geneid'].apply(classify_l1_age)
        fc = fc[fc['age'].isin(['young', 'ancient'])]

        # Calculate total mapped reads for RPM normalization
        un_cols = [c for c in available_names if 'UN' in c]
        sa_cols = [c for c in available_names if 'SA' in c]

        # Sum across replicates for each condition
        fc['UN_total'] = fc[un_cols].sum(axis=1)
        fc['SA_total'] = fc[sa_cols].sum(axis=1)

        # Get total library sizes
        un_lib_size = fc[un_cols].sum().sum()
        sa_lib_size = fc[sa_cols].sum().sum()
        print(f"  Library sizes: UN={un_lib_size:,.0f}, SA={sa_lib_size:,.0f}")

        # RPM normalization
        fc['UN_rpm'] = fc['UN_total'] / (un_lib_size / 1e6)
        fc['SA_rpm'] = fc['SA_total'] / (sa_lib_size / 1e6)

        # Filter: at least 10 reads total
        fc_filt = fc[(fc['UN_total'] + fc['SA_total']) >= 10].copy()
        print(f"  L1 subfamilies with ≥10 reads: {len(fc_filt)}")

        # Log2 fold change (SA/UN)
        fc_filt['log2FC'] = np.log2((fc_filt['SA_rpm'] + 0.01) / (fc_filt['UN_rpm'] + 0.01))

        # Summary by age
        for age in ['young', 'ancient']:
            sub = fc_filt[fc_filt['age'] == age]
            if len(sub) == 0:
                continue
            total_un = sub['UN_total'].sum()
            total_sa = sub['SA_total'].sum()
            fc_total = (total_sa / sa_lib_size) / (total_un / un_lib_size)
            median_lfc = sub['log2FC'].median()
            weighted_lfc = np.log2(fc_total) if fc_total > 0 else np.nan

            print(f"\n  {age.upper()} L1:")
            print(f"    Subfamilies: {len(sub)}")
            print(f"    Total reads: UN={total_un:,.0f}, SA={total_sa:,.0f}")
            print(f"    Weighted FC (RPM): {fc_total:.3f}x (log2={weighted_lfc:.3f})")
            print(f"    Median per-subfamily log2FC: {median_lfc:.3f} ({2**median_lfc:.3f}x)")

            # Top changers
            top_up = sub.nlargest(5, 'log2FC')
            top_down = sub.nsmallest(5, 'log2FC')
            print(f"    Top up: {', '.join(f'{r.Geneid}({r.log2FC:.2f})' for _, r in top_up.iterrows())}")
            print(f"    Top down: {', '.join(f'{r.Geneid}({r.log2FC:.2f})' for _, r in top_down.iterrows())}")

        # =============================
        # 5. Statistical test
        # =============================
        print("\n" + "=" * 60)
        print("5. Statistical tests")
        print("=" * 60)

        # Per-replicate analysis for proper statistics
        for age in ['young', 'ancient']:
            sub = fc_filt[fc_filt['age'] == age]
            if len(sub) == 0:
                continue

            # Total L1 reads per replicate, normalized by library size
            for rep_un, rep_sa in zip(un_cols, sa_cols):
                un_total = sub[rep_un].sum()
                sa_total = sub[rep_sa].sum()
                un_lib = fc[rep_un].sum()
                sa_lib = fc[rep_sa].sum()
                un_frac = un_total / un_lib * 1e6 if un_lib > 0 else 0
                sa_frac = sa_total / sa_lib * 1e6 if sa_lib > 0 else 0
                print(f"  {age} {rep_un}: {un_frac:.1f} RPM, {rep_sa}: {sa_frac:.1f} RPM, ratio={sa_frac/un_frac:.3f}x" if un_frac > 0 else f"  {age}: no reads")

        # Young vs Ancient comparison of fold change
        young_lfc = fc_filt[fc_filt['age'] == 'young']['log2FC']
        ancient_lfc = fc_filt[fc_filt['age'] == 'ancient']['log2FC']
        if len(young_lfc) > 5 and len(ancient_lfc) > 5:
            u_stat, p_val = stats.mannwhitneyu(young_lfc, ancient_lfc, alternative='two-sided')
            print(f"\n  Young vs Ancient log2FC: MW U={u_stat:.0f}, P={p_val:.2e}")
            print(f"  Young median: {young_lfc.median():.3f}, Ancient median: {ancient_lfc.median():.3f}")

        # =============================
        # 6. Generate figures
        # =============================
        print("\n" + "=" * 60)
        print("6. Generating figures")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # Panel (a): log2FC distribution by age
        ax = axes[0, 0]
        for age, color, label in [('young', '#4CAF50', 'Young L1'),
                                   ('ancient', '#8D6E63', 'Ancient L1')]:
            sub = fc_filt[fc_filt['age'] == age]
            ax.hist(sub['log2FC'], bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('log₂(SA/UN fold change)', fontsize=10)
        ax.set_ylabel('Number of subfamilies', fontsize=10)
        ax.set_title('L1 subfamily expression change under arsenite', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.08, 1.05, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold')

        # Panel (b): Volcano plot
        ax = axes[0, 1]
        if tet_file and tet_file.exists() and 'padj' in de.columns:
            # Use TEtranscripts DESeq2 results
            l1_plot = l1_de.dropna(subset=['log2FoldChange', 'padj'])
            for age, color, marker in [('ancient', '#8D6E63', 'o'), ('young', '#4CAF50', '^')]:
                sub = l1_plot[l1_plot['age'] == age]
                sig = sub['padj'] < 0.05
                ax.scatter(sub[~sig]['log2FoldChange'], -np.log10(sub[~sig]['padj']),
                          c=color, alpha=0.3, s=20, marker=marker)
                ax.scatter(sub[sig]['log2FoldChange'], -np.log10(sub[sig]['padj']),
                          c=color, alpha=0.8, s=40, marker=marker, label=f'{age.capitalize()} (sig)')
            ax.axhline(-np.log10(0.05), color='grey', linestyle='--', alpha=0.5)
            ax.set_xlabel('log₂(SA/UN)', fontsize=10)
            ax.set_ylabel('-log₁₀(padj)', fontsize=10)
            ax.set_title('L1 volcano (TEtranscripts DESeq2)', fontsize=11, fontweight='bold')
        else:
            # Simple scatter: UN RPM vs SA RPM
            for age, color, marker in [('ancient', '#8D6E63', 'o'), ('young', '#4CAF50', '^')]:
                sub = fc_filt[fc_filt['age'] == age]
                ax.scatter(np.log10(sub['UN_rpm'] + 0.1), np.log10(sub['SA_rpm'] + 0.1),
                          c=color, alpha=0.5, s=20, marker=marker, label=age.capitalize())
            maxval = max(np.log10(fc_filt['UN_rpm'].max() + 0.1), np.log10(fc_filt['SA_rpm'].max() + 0.1))
            ax.plot([0, maxval], [0, maxval], 'k--', alpha=0.3)
            ax.set_xlabel('log₁₀(UN RPM)', fontsize=10)
            ax.set_ylabel('log₁₀(SA RPM)', fontsize=10)
            ax.set_title('L1 expression: UN vs SA', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.08, 1.05, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold')

        # Panel (c): Top L1 subfamilies bar chart
        ax = axes[1, 0]
        # Show top expressed L1 subfamilies
        top_l1 = fc_filt.nlargest(15, 'UN_total')
        top_l1 = top_l1.sort_values('log2FC')
        colors = ['#4CAF50' if a == 'young' else '#8D6E63' for a in top_l1['age']]
        bars = ax.barh(range(len(top_l1)), top_l1['log2FC'], color=colors, alpha=0.8, edgecolor='white')
        ax.set_yticks(range(len(top_l1)))
        ax.set_yticklabels(top_l1['Geneid'], fontsize=8)
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('log₂(SA/UN)', fontsize=10)
        ax.set_title('Top 15 L1 subfamilies by expression', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.08, 1.05, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold')

        # Panel (d): Summary comparison with DRS
        ax = axes[1, 1]
        # DRS findings (from our data)
        drs_data = {
            'Young L1': {'DRS_polya_delta': 0, 'label': 'Poly(A) Δ=0'},
            'Ancient L1': {'DRS_polya_delta': -31, 'label': 'Poly(A) Δ=-31nt'},
        }

        young_fc_total = fc_filt[fc_filt['age'] == 'young']
        ancient_fc_total = fc_filt[fc_filt['age'] == 'ancient']
        young_total_lfc = np.log2((young_fc_total['SA_total'].sum() / sa_lib_size) /
                                   (young_fc_total['UN_total'].sum() / un_lib_size)) if young_fc_total['UN_total'].sum() > 0 else 0
        ancient_total_lfc = np.log2((ancient_fc_total['SA_total'].sum() / sa_lib_size) /
                                     (ancient_fc_total['UN_total'].sum() / un_lib_size)) if ancient_fc_total['UN_total'].sum() > 0 else 0

        categories = ['Young L1', 'Ancient L1']
        rnaseq_fc = [2**young_total_lfc, 2**ancient_total_lfc]
        drs_polya = [0, -31]  # poly(A) delta from DRS

        x = np.arange(len(categories))
        w = 0.35
        bars1 = ax.bar(x - w/2, rnaseq_fc, w, color='#2196F3', alpha=0.8, label='RNA-seq FC (SA/UN)')
        ax.axhline(1.0, color='grey', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel('Fold change (SA/UN)', fontsize=10, color='#2196F3')
        ax.set_title('RNA-seq FC vs DRS poly(A) change', fontsize=11, fontweight='bold')

        # Second y-axis for DRS poly(A)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + w/2, drs_polya, w, color='#FF9800', alpha=0.8, label='DRS poly(A) Δnt')
        ax2.axhline(0, color='grey', linestyle='--', alpha=0.3)
        ax2.set_ylabel('DRS poly(A) Δ (nt)', fontsize=10, color='#FF9800')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.text(-0.08, 1.05, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold')

        plt.tight_layout()
        fig_path = OUTPUT / 'rnaseq_l1_arsenite_validation.pdf'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

        # =============================
        # 7. Save summary table
        # =============================
        summary = fc_filt[['Geneid', 'age', 'Length', 'UN_total', 'SA_total',
                           'UN_rpm', 'SA_rpm', 'log2FC']].sort_values('log2FC')
        summary.to_csv(OUTPUT / 'l1_subfamily_expression_change.tsv', sep='\t', index=False)
        print(f"  Saved: {OUTPUT / 'l1_subfamily_expression_change.tsv'}")

else:
    print(f"  featureCounts output not found: {fc_output}")
    print("  Run alignment and featureCounts first.")

print("\n===== Analysis complete =====")
