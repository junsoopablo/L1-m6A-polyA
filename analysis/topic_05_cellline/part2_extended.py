#!/usr/bin/env python3
"""
Part 2 extension: Additional poly(A) analyses.
1. Intronic vs intergenic L1 poly(A) + host gene analysis
2. Hotspot poly(A) replicate consistency
3. Poly(A) bimodality detection
4. Replicate poly(A) PCA
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUTDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
FIGDIR = OUTDIR / 'pdf_figures_part2'
FIGDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

CELL_LINES = {
    'A549':     ['A549_4', 'A549_5', 'A549_6'],
    'H9':       ['H9_2', 'H9_3', 'H9_4'],
    'Hct116':   ['Hct116_3', 'Hct116_4'],
    'HeLa':     ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2':    ['HepG2_5', 'HepG2_6'],
    'HEYA8':    ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562':     ['K562_4', 'K562_5', 'K562_6'],
    'MCF7':     ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'MCF7-EV':  ['MCF7-EV_1'],
    'SHSY5Y':   ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

CL_COLORS = {
    'A549': '#E24A33', 'H9': '#348ABD', 'Hct116': '#988ED5',
    'HeLa': '#D4A017', 'HeLa-Ars': '#FF6F61', 'HepG2': '#8EBA42',
    'HEYA8': '#E07B91', 'K562': '#77BEDB', 'MCF7': '#C49C94',
    'MCF7-EV': '#8C564B', 'SHSY5Y': '#55A868',
}

# =========================================================================
# Load data
# =========================================================================
print("Loading data...")
all_dfs = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        df['group'] = g
        df['cell_line'] = cl
        df['l1_age'] = df['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
        all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)
data['host_gene'] = data['overlapping_genes'].str.split(';').str[0]
print(f"  Total: {len(data):,} reads")

# Cell line order
cl_polya = data.groupby('cell_line')['polya_length'].median().sort_values(ascending=False)
cl_order = cl_polya.index.tolist()

samples = sorted(data['group'].unique())
sample_meta = {g: cl for cl, groups in CELL_LINES.items() for g in groups}

# =========================================================================
# 1. Intronic vs Intergenic + Host Gene Analysis
# =========================================================================
print("\n=== Analysis 1: Intronic vs Intergenic ===")

intronic = data[data['TE_group'] == 'intronic']
intergenic = data[data['TE_group'] == 'intergenic']

mw_context = stats.mannwhitneyu(intronic['polya_length'].dropna(),
                                 intergenic['polya_length'].dropna())
print(f"  Intronic: n={len(intronic):,}, median polyA={intronic['polya_length'].median():.1f}")
print(f"  Intergenic: n={len(intergenic):,}, median polyA={intergenic['polya_length'].median():.1f}")
print(f"  MW p={mw_context.pvalue:.2e}")

# Host gene stats
host_stats = intronic.groupby('host_gene').agg(
    n_reads=('read_id', 'count'),
    n_loci=('transcript_id', 'nunique'),
    n_cl=('cell_line', 'nunique'),
    polya_median=('polya_length', 'median'),
    polya_iqr=('polya_length', lambda x: x.quantile(0.75) - x.quantile(0.25)),
).sort_values('n_reads', ascending=False)
host_stats.to_csv(OUTDIR / 'part2_host_gene_stats.tsv', sep='\t', float_format='%.1f')
print(f"  Host genes: {len(host_stats):,}")

# Per-CL intronic vs intergenic
context_by_cl = []
for cl in cl_order:
    d = data[data['cell_line'] == cl]
    intr_med = d[d['TE_group'] == 'intronic']['polya_length'].median()
    inter_med = d[d['TE_group'] == 'intergenic']['polya_length'].median()
    context_by_cl.append({'cell_line': cl, 'intronic': intr_med, 'intergenic': inter_med,
                          'delta': intr_med - inter_med})

context_df = pd.DataFrame(context_by_cl)
print(f"  Mean delta (intronic - intergenic): {context_df['delta'].mean():.1f} nt")

# =========================================================================
# 2. Hotspot poly(A) replicate consistency
# =========================================================================
print("\n=== Analysis 2: Hotspot Poly(A) Consistency ===")

# Select loci with reads in >=3 replicates (for meaningful comparison)
loci_rep_count = data.groupby('transcript_id')['group'].nunique()
robust_loci = loci_rep_count[loci_rep_count >= 5].index  # >=5 replicates
print(f"  Loci with >=5 replicates: {len(robust_loci):,}")

# For each robust locus, compute within-CL and between-CL poly(A) CV
locus_consistency = []
for locus in robust_loci:
    d = data[data['transcript_id'] == locus]
    if len(d) < 10:
        continue
    overall_med = d['polya_length'].median()
    overall_cv = d['polya_length'].std() / d['polya_length'].mean() if d['polya_length'].mean() > 0 else np.nan

    # Per-CL medians
    cl_meds = d.groupby('cell_line')['polya_length'].median()
    between_cl_cv = cl_meds.std() / cl_meds.mean() if len(cl_meds) > 1 and cl_meds.mean() > 0 else np.nan

    locus_consistency.append({
        'locus': locus,
        'gene_id': d['gene_id'].iloc[0],
        'n_reads': len(d),
        'n_cl': d['cell_line'].nunique(),
        'n_reps': d['group'].nunique(),
        'polya_median': overall_med,
        'polya_cv': overall_cv,
        'between_cl_cv': between_cl_cv,
    })

consist_df = pd.DataFrame(locus_consistency).sort_values('n_reads', ascending=False)
consist_df.to_csv(OUTDIR / 'part2_hotspot_consistency.tsv', sep='\t', index=False, float_format='%.3f')
print(f"  Analyzed {len(consist_df)} loci")
print(f"  Median between-CL CV: {consist_df['between_cl_cv'].median():.3f}")

# =========================================================================
# 3. Poly(A) Bimodality Detection
# =========================================================================
print("\n=== Analysis 3: Poly(A) Bimodality ===")

# Use Hartigan's dip test (diptest package) if available, else use custom
try:
    from diptest import diptest as dip_test
    has_diptest = True
except ImportError:
    has_diptest = False
    print("  diptest not available, using KDE-based approach")

# Candidate loci: >=30 reads for reliable distribution
candidate_loci = data.groupby('transcript_id').filter(lambda x: len(x) >= 30)
locus_list = candidate_loci['transcript_id'].unique()
print(f"  Candidate loci (>=30 reads): {len(locus_list)}")

bimodality_results = []
for locus in locus_list:
    d = data[data['transcript_id'] == locus]
    vals = d['polya_length'].dropna().values
    n = len(vals)
    if n < 30:
        continue

    if has_diptest:
        dip_stat, dip_p = dip_test(vals)
    else:
        # Silverman bandwidth test approximation: bicoefficient of skewness/kurtosis
        dip_stat = np.nan
        dip_p = np.nan

    # Also compute kurtosis (platykurtic = bimodal hint) and bimodality coefficient
    skw = stats.skew(vals)
    krt = stats.kurtosis(vals)  # excess kurtosis
    # Bimodality coefficient: BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2/((n-2)*(n-3)))
    if n > 3:
        bc = (skw**2 + 1) / (krt + 3 * (n-1)**2 / ((n-2)*(n-3)))
    else:
        bc = np.nan

    bimodality_results.append({
        'locus': locus,
        'gene_id': d['gene_id'].iloc[0],
        'n_reads': n,
        'polya_median': np.median(vals),
        'polya_iqr': np.percentile(vals, 75) - np.percentile(vals, 25),
        'skewness': skw,
        'kurtosis': krt,
        'bimodality_coeff': bc,
        'dip_stat': dip_stat,
        'dip_p': dip_p,
    })

bimod_df = pd.DataFrame(bimodality_results).sort_values('bimodality_coeff', ascending=False)
bimod_df.to_csv(OUTDIR / 'part2_bimodality.tsv', sep='\t', index=False, float_format='%.4f')

# BC > 0.555 suggests bimodality (Pfister et al. 2013)
n_bimodal = (bimod_df['bimodality_coeff'] > 0.555).sum()
print(f"  BC > 0.555 (bimodal hint): {n_bimodal}/{len(bimod_df)} loci")
if has_diptest:
    n_dip_sig = (bimod_df['dip_p'] < 0.05).sum()
    print(f"  Dip test p<0.05: {n_dip_sig}/{len(bimod_df)} loci")

# =========================================================================
# 4. Replicate poly(A) PCA
# =========================================================================
print("\n=== Analysis 4: Replicate Poly(A) PCA ===")

# Feature: binned poly(A) distribution per sample
bins = np.arange(0, 401, 20)  # 20 bins, 0-400
bin_labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]

polya_features = {}
for s in samples:
    d = data[data['group'] == s]['polya_length'].dropna().clip(upper=400)
    hist, _ = np.histogram(d, bins=bins, density=True)
    polya_features[s] = hist

feat_df = pd.DataFrame(polya_features, index=bin_labels).T
feat_df.index.name = 'sample'

# PCA
X = feat_df.values
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=min(len(samples), 10))
pcs = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_ * 100
print(f"  PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%, PC3: {var_exp[2]:.1f}%")

pca_polya_df = pd.DataFrame({
    'sample': samples,
    'cell_line': [sample_meta[s] for s in samples],
    'PC1': pcs[:, 0], 'PC2': pcs[:, 1], 'PC3': pcs[:, 2],
})

# UMAP
try:
    import umap
    reducer = umap.UMAP(n_neighbors=min(8, len(samples)-1), min_dist=0.3, random_state=42)
    umap_emb = reducer.fit_transform(X_scaled)
    method_name = 'UMAP'
except ImportError:
    from sklearn.manifold import TSNE
    umap_emb = TSNE(n_components=2, perplexity=min(8, len(samples)//2),
                     random_state=42).fit_transform(X_scaled)
    method_name = 't-SNE'

# Sample-sample distance (Jensen-Shannon divergence on distributions)
from scipy.spatial.distance import jensenshannon
n_s = len(samples)
js_dist = np.zeros((n_s, n_s))
for i in range(n_s):
    for j in range(n_s):
        p = feat_df.iloc[i].values + 1e-10
        q = feat_df.iloc[j].values + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        js_dist[i, j] = jensenshannon(p, q)

js_sim = 1 - js_dist / js_dist.max()

# Within vs between CL
within_js, between_js = [], []
for i in range(n_s):
    for j in range(i+1, n_s):
        if sample_meta[samples[i]] == sample_meta[samples[j]]:
            within_js.append(js_sim[i, j])
        else:
            between_js.append(js_sim[i, j])

js_ratio = np.mean(within_js) / np.mean(between_js)
print(f"  JS similarity within-CL: {np.mean(within_js):.3f}")
print(f"  JS similarity between-CL: {np.mean(between_js):.3f}")
print(f"  Ratio: {js_ratio:.2f}x")

# =========================================================================
# FIGURES
# =========================================================================
print("\nGenerating figures...")

# --- Figure 4: Intronic vs Intergenic + Host Gene (3 panels) ---
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5.5))
fig4.subplots_adjust(wspace=0.35)

# 4A: Intronic vs intergenic poly(A) per CL
ax = axes4[0]
x = np.arange(len(cl_order))
w = 0.35
intr_meds = [context_df[context_df['cell_line']==cl]['intronic'].values[0] for cl in cl_order]
inter_meds = [context_df[context_df['cell_line']==cl]['intergenic'].values[0] for cl in cl_order]
ax.bar(x - w/2, intr_meds, w, label='Intronic', color='#4C72B0', alpha=0.8)
ax.bar(x + w/2, inter_meds, w, label='Intergenic', color='#DD8452', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Median poly(A) (nt)')
ax.legend(fontsize=8)
ax.text(0.03, 0.97, f'Overall: {intronic["polya_length"].median():.0f} vs {intergenic["polya_length"].median():.0f} nt\n'
        f'MW p={mw_context.pvalue:.2e}\ndelta={context_df["delta"].mean():+.1f} nt',
        transform=ax.transAxes, fontsize=7.5, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Intronic vs intergenic L1 poly(A)', fontsize=9, loc='center')

# 4B: Top 15 host genes
ax = axes4[1]
top15_genes = host_stats.head(15)
y_pos = range(len(top15_genes))
colors_gene = ['#4C72B0' if r['n_cl'] >= 5 else '#C44E52' for _, r in top15_genes.iterrows()]
ax.barh(y_pos, top15_genes['n_reads'], color=colors_gene, alpha=0.8, edgecolor='gray', lw=0.3)
ax.set_yticks(y_pos)
labels_gene = [f"{gene} ({int(row['n_loci'])}L, {int(row['n_cl'])}CL)" for gene, row in top15_genes.iterrows()]
ax.set_yticklabels(labels_gene, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('L1 reads in gene')
for i, (gene, row) in enumerate(top15_genes.iterrows()):
    ax.text(row['n_reads'] + 5, i, f'pA={row["polya_median"]:.0f}', fontsize=6.5, va='center')
ax.legend(handles=[mpatches.Patch(color='#4C72B0', label='Ubiquitous (>=5 CL)'),
                   mpatches.Patch(color='#C44E52', label='CL-specific (<5 CL)')],
          fontsize=7, loc='lower right')
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Top host genes (intronic L1)', fontsize=9, loc='center')

# 4C: Host gene L1 count vs poly(A) median
ax = axes4[2]
plot_genes = host_stats[host_stats['n_reads'] >= 10].copy()
ax.scatter(plot_genes['n_reads'], plot_genes['polya_median'],
           s=15, alpha=0.5, c='#4C72B0', edgecolors='none')
r_gene, p_gene = stats.spearmanr(plot_genes['n_reads'], plot_genes['polya_median'])
ax.set_xlabel('L1 reads in host gene')
ax.set_ylabel('Median poly(A) of L1 in gene (nt)')
ax.set_xscale('log')
ax.text(0.03, 0.97, f'Spearman r={r_gene:.3f}\np={p_gene:.2e}\nn={len(plot_genes)} genes',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
# Annotate extreme genes
for gene, row in plot_genes.nlargest(3, 'n_reads').iterrows():
    ax.annotate(gene, (row['n_reads'], row['polya_median']),
                fontsize=6, alpha=0.7, xytext=(5, 3), textcoords='offset points')
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Host gene L1 load vs poly(A)', fontsize=9, loc='center')

fig4.savefig(FIGDIR / 'fig4_context_hostgene.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print("  Figure 4 saved")

# --- Figure 5: Hotspot consistency + bimodality (3 panels) ---
fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5.5))
fig5.subplots_adjust(wspace=0.35)

# 5A: Top 10 robust loci poly(A) across cell lines
ax = axes5[0]
top10_consist = consist_df.head(10)
loci_for_plot = top10_consist['locus'].tolist()

bp_data = []
bp_labels = []
bp_positions = []
pos = 0
for locus in loci_for_plot:
    d = data[data['transcript_id'] == locus]
    cl_present = sorted(d['cell_line'].unique())
    for cl in cl_present:
        vals = d[d['cell_line'] == cl]['polya_length'].dropna().values
        if len(vals) >= 2:
            bp_data.append(vals)
            bp_labels.append(cl)
            bp_positions.append(pos)
            pos += 1
    pos += 0.5  # gap between loci

bp = ax.boxplot(bp_data, positions=bp_positions, showfliers=False, widths=0.6,
                patch_artist=True, medianprops=dict(color='black', lw=1.2))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(CL_COLORS.get(bp_labels[i], '#999999'))
    patch.set_alpha(0.7)

# Add locus labels
locus_centers = []
center = 0
for locus in loci_for_plot:
    d = data[data['transcript_id'] == locus]
    n_cl = len([cl for cl in d['cell_line'].unique()
                if len(d[d['cell_line'] == cl]['polya_length'].dropna()) >= 2])
    locus_centers.append(center + (n_cl - 1) / 2)
    center += n_cl + 0.5

ax.set_xticks(locus_centers)
locus_short = [f"{l.split('_')[0][:8]}" for l in loci_for_plot]
ax.set_xticklabels(locus_short, rotation=45, ha='right', fontsize=6.5)
ax.set_ylabel('Poly(A) length (nt)')
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Hotspot poly(A) by cell line', fontsize=9, loc='center')

# 5B: Between-CL CV distribution
ax = axes5[1]
cv_vals = consist_df['between_cl_cv'].dropna()
ax.hist(cv_vals, bins=30, color='#4C72B0', alpha=0.8, edgecolor='white')
ax.axvline(cv_vals.median(), color='#C44E52', ls='--', lw=1.5)
ax.text(cv_vals.median() + 0.01, ax.get_ylim()[1]*0.9,
        f'median CV={cv_vals.median():.3f}', fontsize=8, color='#C44E52')
ax.set_xlabel('Between-CL poly(A) CV')
ax.set_ylabel('Number of loci')
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Hotspot poly(A) variation', fontsize=9, loc='center')

# 5C: Bimodality examples
ax = axes5[2]
# Top 3 bimodal candidates by BC (with dip test if available)
if has_diptest:
    bimod_cands = bimod_df[(bimod_df['dip_p'] < 0.05) & (bimod_df['bimodality_coeff'] > 0.555)]
    if len(bimod_cands) == 0:
        bimod_cands = bimod_df.nlargest(3, 'bimodality_coeff')
    else:
        bimod_cands = bimod_cands.nlargest(3, 'bimodality_coeff')
else:
    bimod_cands = bimod_df.nlargest(3, 'bimodality_coeff')

colors_bimod = ['#E24A33', '#348ABD', '#8EBA42']
for i, (_, row) in enumerate(bimod_cands.iterrows()):
    vals = data[data['transcript_id'] == row['locus']]['polya_length'].dropna().clip(upper=400)
    ax.hist(vals, bins=np.arange(0, 401, 15), alpha=0.5, density=True,
            color=colors_bimod[i],
            label=f"{row['locus'][:20]} (BC={row['bimodality_coeff']:.2f}, n={int(row['n_reads'])})")
ax.set_xlabel('Poly(A) length (nt)')
ax.set_ylabel('Density')
ax.legend(fontsize=6.5, loc='upper right')
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Bimodal poly(A) candidates', fontsize=9, loc='center')

fig5.savefig(FIGDIR / 'fig5_consistency_bimodality.png', dpi=300, bbox_inches='tight')
plt.close(fig5)
print("  Figure 5 saved")

# --- Figure 6: Replicate poly(A) PCA (3 panels) ---
fig6 = plt.figure(figsize=(16, 5.5))
gs6 = fig6.add_gridspec(1, 3, wspace=0.35)

REP_MARKERS = ['o', 's', '^', 'D']
def get_rep_idx(sample):
    cl = sample_meta[sample]
    return CELL_LINES[cl].index(sample)

# 6A: PCA
ax = fig6.add_subplot(gs6[0, 0])
for cl in CELL_LINES:
    mask = pca_polya_df['cell_line'] == cl
    sub = pca_polya_df[mask]
    for _, row in sub.iterrows():
        ri = get_rep_idx(row['sample'])
        ax.scatter(row['PC1'], row['PC2'], c=CL_COLORS[cl], s=70,
                   marker=REP_MARKERS[ri], edgecolors='black', linewidth=0.5, zorder=3)
    pts = sub[['PC1', 'PC2']].values
    if len(pts) > 1:
        centroid = pts.mean(axis=0)
        for pt in pts:
            ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]],
                    c=CL_COLORS[cl], alpha=0.4, lw=1, zorder=2)

ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)')
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('PCA: poly(A) distribution', fontsize=10, loc='center')

# 6B: UMAP
ax = fig6.add_subplot(gs6[0, 1])
for cl in CELL_LINES:
    idxs = [i for i, s in enumerate(samples) if sample_meta[s] == cl]
    for idx in idxs:
        ri = get_rep_idx(samples[idx])
        ax.scatter(umap_emb[idx, 0], umap_emb[idx, 1], c=CL_COLORS[cl], s=70,
                   marker=REP_MARKERS[ri], edgecolors='black', linewidth=0.5, zorder=3)
    if len(idxs) > 1:
        pts = umap_emb[idxs]
        centroid = pts.mean(axis=0)
        for pt in pts:
            ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]],
                    c=CL_COLORS[cl], alpha=0.4, lw=1, zorder=2)

for i, s in enumerate(samples):
    ax.annotate(s, (umap_emb[i, 0], umap_emb[i, 1]),
                fontsize=5, alpha=0.5, xytext=(3, 3), textcoords='offset points')

ax.set_xlabel(f'{method_name}1')
ax.set_ylabel(f'{method_name}2')
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title(f'{method_name}: poly(A) distribution', fontsize=10, loc='center')

# 6C: JS similarity heatmap
ax = fig6.add_subplot(gs6[0, 2])
dist_condensed = squareform(js_dist, checks=False)
link = linkage(dist_condensed, method='average')
order = leaves_list(link)

sim_ordered = js_sim[np.ix_(order, order)]
sample_labels = [samples[i] for i in order]
cl_labels = [sample_meta[s] for s in sample_labels]

offdiag = js_sim[np.triu_indices(n_s, k=1)]
vmin_js = max(0, np.percentile(offdiag, 2) - 0.01)
vmax_js = min(1, np.percentile(offdiag, 98) + 0.01)

im = ax.imshow(sim_ordered, cmap='YlOrRd', vmin=vmin_js, vmax=vmax_js, aspect='auto')
ax.set_xticks(range(len(sample_labels)))
ax.set_xticklabels(sample_labels, rotation=90, fontsize=5)
ax.set_yticks(range(len(sample_labels)))
ax.set_yticklabels(sample_labels, fontsize=5)
for i, cl in enumerate(cl_labels):
    ax.plot(-1.5, i, 's', color=CL_COLORS[cl], markersize=4, clip_on=False)
plt.colorbar(im, ax=ax, label='JS similarity', shrink=0.8)
ax.text(0.5, -0.28,
        f'Within-CL: {np.mean(within_js):.3f} | Between-CL: {np.mean(between_js):.3f} | '
        f'Ratio: {js_ratio:.2f}x',
        transform=ax.transAxes, fontsize=7, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('JS similarity (poly(A) dist.)', fontsize=9, loc='center')

# Legend
handles = [mpatches.Patch(color=CL_COLORS[cl], label=cl) for cl in CELL_LINES]
fig6.legend(handles=handles, loc='lower center', ncol=6, fontsize=7.5,
            frameon=True, bbox_to_anchor=(0.5, -0.08))

fig6.savefig(FIGDIR / 'fig6_polya_pca.png', dpi=300, bbox_inches='tight')
plt.close(fig6)
print("  Figure 6 saved")

# =========================================================================
# Summary
# =========================================================================
print("\n=== SUMMARY ===")
print(f"1. Intronic vs Intergenic: delta={context_df['delta'].mean():+.1f} nt, p={mw_context.pvalue:.2e}")
print(f"   Host genes: {len(host_stats):,}, top=CCDC170 (718 reads, MCF7-specific)")
print(f"2. Hotspot consistency: {len(consist_df)} loci, median between-CL CV={consist_df['between_cl_cv'].median():.3f}")
print(f"3. Bimodality: {n_bimodal}/{len(bimod_df)} loci with BC>0.555")
if has_diptest:
    print(f"   Dip test significant: {n_dip_sig}/{len(bimod_df)}")
print(f"4. Poly(A) PCA: PC1={var_exp[0]:.1f}%, JS ratio={js_ratio:.2f}x")
print(f"\nAll outputs saved to {OUTDIR}")
print("Done!")
