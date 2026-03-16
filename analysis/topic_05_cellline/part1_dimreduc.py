#!/usr/bin/env python3
"""
Part 1 supplement: Dimensionality reduction of L1 loci expression patterns.
- Build loci x sample count matrix
- Depth-matched subsampling: subsample each sample to min L1 read count
- Binary presence/absence → PCA, UMAP (Jaccard), sample-sample heatmap
- Bootstrap (100 iterations) for robust Jaccard statistics
"""

import pandas as pd
import numpy as np
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
FIGDIR = OUTDIR / 'pdf_figures_v2'
FIGDIR.mkdir(exist_ok=True)

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
REP_MARKERS = ['o', 's', '^', 'D']

N_BOOTSTRAP = 100
RNG = np.random.RandomState(42)

# =========================================================================
# 1. Load raw reads per sample
# =========================================================================
print("Loading raw reads per sample...")
sample_meta = {}
sample_reads = {}  # sample -> list of transcript_ids (one per read)

for cl, groups in CELL_LINES.items():
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS']
        sample_reads[g] = df['transcript_id'].values
        sample_meta[g] = cl
        print(f"  {g}: {len(df)} reads, {df['transcript_id'].nunique()} loci")

samples = sorted(sample_reads.keys())
read_counts = {s: len(sample_reads[s]) for s in samples}
min_depth = min(read_counts.values())
min_sample = min(read_counts, key=read_counts.get)
print(f"\nSubsampling target: {min_depth} reads (= {min_sample})")

# =========================================================================
# 2. Helper: subsample → binary matrix
# =========================================================================
def subsample_to_binary(sample_reads, samples, target_n, rng, min_samples=2):
    """Subsample each sample to target_n reads, return binary loci matrix."""
    all_loci = set()
    sub_counts = {}
    for s in samples:
        reads = rng.choice(sample_reads[s], size=target_n, replace=False)
        unique, counts = np.unique(reads, return_counts=True)
        sub_counts[s] = dict(zip(unique, counts))
        all_loci.update(unique)

    loci = sorted(all_loci)
    mat = pd.DataFrame(0, index=loci, columns=samples, dtype=int)
    for s in samples:
        for loc, cnt in sub_counts[s].items():
            mat.loc[loc, s] = cnt

    # Filter: detected in >= min_samples
    n_detected = (mat > 0).sum(axis=1)
    mat_filt = mat[n_detected >= min_samples]
    mat_bin = (mat_filt > 0).astype(int)
    return mat_bin

# =========================================================================
# 3. Single representative subsampling (seed=42) for figures
# =========================================================================
print("\nRunning representative subsampling...")
mat_bin = subsample_to_binary(sample_reads, samples, min_depth, np.random.RandomState(42))
print(f"  Subsampled binary matrix: {mat_bin.shape[0]} loci x {mat_bin.shape[1]} samples")

# PCA
X_bin = mat_bin.T.values.astype(float)
X_scaled = StandardScaler().fit_transform(X_bin)
pca = PCA(n_components=min(len(samples), 10))
pcs = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_ * 100
print(f"  PC1: {var_explained[0]:.1f}%, PC2: {var_explained[1]:.1f}%, PC3: {var_explained[2]:.1f}%")

pca_df = pd.DataFrame({
    'sample': samples,
    'cell_line': [sample_meta[s] for s in samples],
    'n_reads_original': [read_counts[s] for s in samples],
    'n_reads_subsampled': min_depth,
    'n_loci_detected': [(mat_bin[s] > 0).sum() for s in samples],
    'PC1': pcs[:, 0], 'PC2': pcs[:, 1], 'PC3': pcs[:, 2],
})
pca_df.to_csv(OUTDIR / 'part1_dimreduc_pca.tsv', sep='\t', index=False)

# UMAP
print("  Running UMAP...")
try:
    import umap
    reducer = umap.UMAP(n_neighbors=min(8, len(samples)-1), min_dist=0.3,
                        metric='jaccard', random_state=42)
    umap_emb = reducer.fit_transform(X_bin)
    method_name = 'UMAP'
except ImportError:
    from sklearn.manifold import TSNE
    umap_emb = TSNE(n_components=2, perplexity=min(8, len(samples)//2),
                     random_state=42).fit_transform(X_bin)
    method_name = 't-SNE'

# Jaccard similarity
jaccard_dist = pairwise_distances(X_bin, metric='jaccard')
jaccard_sim = 1 - jaccard_dist

# =========================================================================
# 4. Bootstrap Jaccard ratio (within/between)
# =========================================================================
print(f"\nBootstrapping Jaccard ratio ({N_BOOTSTRAP} iterations)...")
within_means = []
between_means = []
ratios = []

for b in range(N_BOOTSTRAP):
    mb = subsample_to_binary(sample_reads, samples, min_depth, np.random.RandomState(b))
    xb = mb.T.values.astype(float)
    jd = pairwise_distances(xb, metric='jaccard')
    js = 1 - jd

    w, bw = [], []
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            if sample_meta[samples[i]] == sample_meta[samples[j]]:
                w.append(js[i, j])
            else:
                bw.append(js[i, j])
    within_means.append(np.mean(w))
    between_means.append(np.mean(bw))
    ratios.append(np.mean(w) / np.mean(bw))

print(f"  Within-CL Jaccard: {np.mean(within_means):.3f} ± {np.std(within_means):.3f}")
print(f"  Between-CL Jaccard: {np.mean(between_means):.3f} ± {np.std(between_means):.3f}")
print(f"  Ratio (within/between): {np.mean(ratios):.2f} ± {np.std(ratios):.2f}")
print(f"  Ratio 95% CI: [{np.percentile(ratios, 2.5):.2f}, {np.percentile(ratios, 97.5):.2f}]")

# Per-CL within-rep Jaccard (from representative subsampling)
within_per_cl = {}
for i in range(len(samples)):
    for j in range(i+1, len(samples)):
        cli = sample_meta[samples[i]]
        clj = sample_meta[samples[j]]
        if cli == clj:
            within_per_cl.setdefault(cli, []).append(jaccard_sim[i, j])

# =========================================================================
# 5. Figure (6 panels)
# =========================================================================
print("\nGenerating figure...")
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

def get_rep_idx(sample):
    cl = sample_meta[sample]
    return CELL_LINES[cl].index(sample)

# --- 5A: PCA PC1 vs PC2 ---
ax = fig.add_subplot(gs[0, 0])
for cl in CELL_LINES:
    mask = pca_df['cell_line'] == cl
    sub = pca_df[mask]
    for _, row in sub.iterrows():
        ri = get_rep_idx(row['sample'])
        ax.scatter(row['PC1'], row['PC2'],
                   c=CL_COLORS[cl], s=70, marker=REP_MARKERS[ri],
                   edgecolors='black', linewidth=0.5, zorder=3)
    pts = sub[['PC1', 'PC2']].values
    if len(pts) > 1:
        centroid = pts.mean(axis=0)
        for pt in pts:
            ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]],
                    c=CL_COLORS[cl], alpha=0.4, lw=1, zorder=2)

ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')
ax.set_title('A', fontsize=13, fontweight='bold', loc='left')
ax.set_title(f'PCA (subsampled to {min_depth} reads)', fontsize=10, loc='center')
ax.axhline(0, color='gray', lw=0.3, ls='--')
ax.axvline(0, color='gray', lw=0.3, ls='--')

# --- 5B: Depth independence proof ---
ax = fig.add_subplot(gs[0, 1])
orig_depths = [read_counts[s] for s in samples]
pc1_vals = pcs[:, 0]
from scipy.stats import spearmanr
r_depth_pc1, p_depth_pc1 = spearmanr(orig_depths, pc1_vals)
r_depth_pc2, p_depth_pc2 = spearmanr(orig_depths, pcs[:, 1])

for i, s in enumerate(samples):
    cl = sample_meta[s]
    ri = get_rep_idx(s)
    ax.scatter(orig_depths[i], pc1_vals[i],
               c=CL_COLORS[cl], s=70, marker=REP_MARKERS[ri],
               edgecolors='black', linewidth=0.5, zorder=3)

ax.set_xlabel('Original L1 read count')
ax.set_ylabel(f'PC1 ({var_explained[0]:.1f}%)')
ax.set_title('B', fontsize=13, fontweight='bold', loc='left')
ax.set_title('Depth independence', fontsize=10, loc='center')

# Annotate correlation
ax.text(0.97, 0.05,
        f'Spearman r = {r_depth_pc1:.3f}\np = {p_depth_pc1:.3f}',
        transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Fit line
z = np.polyfit(orig_depths, pc1_vals, 1)
x_line = np.linspace(min(orig_depths), max(orig_depths), 50)
ax.plot(x_line, np.polyval(z, x_line), '--', color='gray', lw=1, alpha=0.5)

# --- 5C: Variance explained + bootstrap ratio ---
ax = fig.add_subplot(gs[0, 2])
n_pcs = min(10, len(var_explained))
ax.bar(range(1, n_pcs+1), var_explained[:n_pcs], color='#4C72B0', alpha=0.8)
cumvar = np.cumsum(var_explained[:n_pcs])
ax.plot(range(1, n_pcs+1), cumvar, 'o-', color='#C44E52', markersize=5, zorder=3)
for i in range(min(3, n_pcs)):
    ax.text(i+1, var_explained[i] + 0.5, f'{var_explained[i]:.1f}%',
            ha='center', fontsize=7, color='#4C72B0')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax_r = ax.twinx()
ax_r.plot(range(1, n_pcs+1), cumvar, 'o-', color='#C44E52', markersize=5)
ax_r.set_ylabel('Cumulative (%)', color='#C44E52')
ax_r.set_ylim(0, 105)
ax_r.tick_params(axis='y', labelcolor='#C44E52')
ax.set_title('C', fontsize=13, fontweight='bold', loc='left')
ax.set_title('Variance explained', fontsize=10, loc='center')

# --- 5D: UMAP ---
ax = fig.add_subplot(gs[1, 0])
for cl in CELL_LINES:
    idxs = [i for i, s in enumerate(samples) if sample_meta[s] == cl]
    for idx in idxs:
        ri = get_rep_idx(samples[idx])
        ax.scatter(umap_emb[idx, 0], umap_emb[idx, 1],
                   c=CL_COLORS[cl], s=70, marker=REP_MARKERS[ri],
                   edgecolors='black', linewidth=0.5, zorder=3)
    if len(idxs) > 1:
        pts = umap_emb[idxs]
        centroid = pts.mean(axis=0)
        for pt in pts:
            ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]],
                    c=CL_COLORS[cl], alpha=0.4, lw=1, zorder=2)

for i, s in enumerate(samples):
    ax.annotate(s, (umap_emb[i, 0], umap_emb[i, 1]),
                fontsize=5, alpha=0.55, xytext=(3, 3),
                textcoords='offset points')

ax.set_xlabel(f'{method_name}1')
ax.set_ylabel(f'{method_name}2')
ax.set_title('D', fontsize=13, fontweight='bold', loc='left')
ax.set_title(f'{method_name} (Jaccard, subsampled)', fontsize=10, loc='center')

# --- 5E: Sample-sample Jaccard heatmap ---
ax = fig.add_subplot(gs[1, 1:])

dist_condensed = squareform(jaccard_dist, checks=False)
link = linkage(dist_condensed, method='average')
order = leaves_list(link)

sim_ordered = jaccard_sim[np.ix_(order, order)]
sample_labels = [samples[i] for i in order]
cl_labels = [sample_meta[s] for s in sample_labels]

offdiag = jaccard_sim[np.triu_indices(len(samples), k=1)]
vmin = max(0, np.percentile(offdiag, 2) - 0.01)
vmax = min(1, np.percentile(offdiag, 98) + 0.01)

im = ax.imshow(sim_ordered, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(sample_labels)))
ax.set_xticklabels(sample_labels, rotation=90, fontsize=6.5)
ax.set_yticks(range(len(sample_labels)))
ax.set_yticklabels(sample_labels, fontsize=6.5)

for i, cl in enumerate(cl_labels):
    ax.plot(-1.8, i, 's', color=CL_COLORS[cl], markersize=5.5, clip_on=False)
    ax.plot(i, -1.8, 's', color=CL_COLORS[cl], markersize=5.5, clip_on=False)

cbar = plt.colorbar(im, ax=ax, label='Jaccard similarity', shrink=0.7)
ax.set_title('E', fontsize=13, fontweight='bold', loc='left')
ax.set_title(f'Jaccard similarity (subsampled to {min_depth} reads)', fontsize=10, loc='center')

# Annotation box with bootstrap stats
stats_text = (
    f'Within-CL: {np.mean(within_means):.3f} ± {np.std(within_means):.3f}\n'
    f'Between-CL: {np.mean(between_means):.3f} ± {np.std(between_means):.3f}\n'
    f'Ratio: {np.mean(ratios):.2f}x '
    f'[{np.percentile(ratios, 2.5):.2f}-{np.percentile(ratios, 97.5):.2f}]'
)
ax.text(1.0, -0.15, stats_text, transform=ax.transAxes, fontsize=8,
        va='top', ha='right', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# --- Legend ---
handles = [mpatches.Patch(color=CL_COLORS[cl], label=cl) for cl in CELL_LINES]
fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8.5,
           frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.02))

fig.savefig(FIGDIR / 'fig_dimreduc.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Figure saved: {FIGDIR / 'fig_dimreduc.png'}")

# =========================================================================
# 6. Summary stats
# =========================================================================
print("\n=== Summary ===")
print(f"Subsampled depth: {min_depth} reads/sample ({min_sample})")
print(f"Binary loci: {mat_bin.shape[0]}")
print(f"PC1: {var_explained[0]:.1f}%, PC2: {var_explained[1]:.1f}%, PC3: {var_explained[2]:.1f}%")
print(f"Top 5 PCs: {sum(var_explained[:5]):.1f}%")
print(f"Bootstrap ({N_BOOTSTRAP}x):")
print(f"  Within-CL Jaccard: {np.mean(within_means):.3f} ± {np.std(within_means):.3f}")
print(f"  Between-CL Jaccard: {np.mean(between_means):.3f} ± {np.std(between_means):.3f}")
print(f"  Ratio: {np.mean(ratios):.2f}x [{np.percentile(ratios,2.5):.2f}-{np.percentile(ratios,97.5):.2f}]")
print(f"\nPer-CL within-replicate Jaccard (representative):")
for cl in sorted(within_per_cl.keys()):
    vals = within_per_cl[cl]
    print(f"  {cl}: {np.mean(vals):.3f} (n={len(vals)})")

# Save stats
stats_rows = [
    ('subsample_depth', min_depth),
    ('n_loci_binary', mat_bin.shape[0]),
    ('n_samples', mat_bin.shape[1]),
    ('PC1_var_pct', var_explained[0]),
    ('PC2_var_pct', var_explained[1]),
    ('PC3_var_pct', var_explained[2]),
    ('top5_cumvar_pct', sum(var_explained[:5])),
    ('bootstrap_n', N_BOOTSTRAP),
    ('within_CL_jaccard_mean', np.mean(within_means)),
    ('within_CL_jaccard_sd', np.std(within_means)),
    ('between_CL_jaccard_mean', np.mean(between_means)),
    ('between_CL_jaccard_sd', np.std(between_means)),
    ('ratio_mean', np.mean(ratios)),
    ('ratio_ci_lo', np.percentile(ratios, 2.5)),
    ('ratio_ci_hi', np.percentile(ratios, 97.5)),
    ('depth_vs_PC1_spearman_r', r_depth_pc1),
    ('depth_vs_PC1_spearman_p', p_depth_pc1),
    ('depth_vs_PC2_spearman_r', r_depth_pc2),
    ('depth_vs_PC2_spearman_p', p_depth_pc2),
]
pd.DataFrame(stats_rows, columns=['metric', 'value']).to_csv(
    OUTDIR / 'part1_dimreduc_stats.tsv', sep='\t', index=False, float_format='%.4f')

mat_bin.to_csv(OUTDIR / 'part1_loci_sample_matrix.tsv', sep='\t')
print(f"\nDone!")
