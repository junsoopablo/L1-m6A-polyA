#!/usr/bin/env python3
"""
PCA from two perspectives:
  (1) L1 subtype composition (gene_id fractions per replicate)
  (2) Ancient L1 loci expression pattern (loci read counts per replicate)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
OUT_DIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
MIN_READS = 200  # Minimum reads per replicate to include

# Excluded: HEK293 (1 rep, 194 reads), HEK293T (1 valid rep after filtering), THP1 (1 valid rep)
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

# Colors
base_colors = {
    'A549': '#1f77b4', 'H9': '#ff7f0e', 'Hct116': '#2ca02c',
    'HeLa': '#8c564b', 'HepG2': '#e377c2', 'HEYA8': '#7f7f7f',
    'K562': '#bcbd22', 'MCF7': '#17becf', 'SHSY5Y': '#aec7e8',
}
treatment_colors = {'HeLa-Ars': '#ff0000', 'MCF7-EV': '#ff00ff'}

def get_color(cl):
    return treatment_colors.get(cl, base_colors.get(cl, '#333333'))

# =========================================================================
# Load all data
# =========================================================================
print("Loading data...")
all_dfs = {}
for cl, groups in CELL_LINES.items():
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        if len(df) < MIN_READS:
            print(f"  Skipping {g}: {len(df)} reads < MIN_READS={MIN_READS}")
            continue
        df['cell_line'] = cl
        df['l1_age'] = df['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
        all_dfs[g] = df
        print(f"  {g}: {len(df)} reads, {df['gene_id'].nunique()} subtypes, "
              f"{df['transcript_id'].nunique()} loci")

# =====================================================================
# PART 1: SUBTYPE COMPOSITION PCA
# =====================================================================
print(f"\n{'='*90}")
print("PART 1: L1 Subtype Composition PCA")
print(f"{'='*90}")

# Get all subtypes
all_subtypes = set()
for g, df in all_dfs.items():
    all_subtypes.update(df['gene_id'].unique())
all_subtypes = sorted(all_subtypes)
print(f"Total subtypes across all: {len(all_subtypes)}")

# Build composition matrix (fraction of reads per subtype)
subtype_matrix = []
group_labels = []
cl_labels = []
for g, df in sorted(all_dfs.items()):
    counts = df['gene_id'].value_counts()
    fracs = counts / counts.sum()
    row = [fracs.get(st, 0.0) for st in all_subtypes]
    subtype_matrix.append(row)
    group_labels.append(g)
    cl_labels.append(df['cell_line'].iloc[0])

X_sub = np.array(subtype_matrix)
group_labels = np.array(group_labels)
cl_labels = np.array(cl_labels)

# Filter rare subtypes (present in < 30% of replicates)
subtype_prevalence = (X_sub > 0).mean(axis=0)
keep = subtype_prevalence >= 0.3
X_sub_filt = X_sub[:, keep]
subtypes_kept = [st for st, k in zip(all_subtypes, keep) if k]
print(f"Subtypes kept (>30% prevalence): {len(subtypes_kept)}/{len(all_subtypes)}")

# CLR transform (compositional data)
X_clr = np.log(X_sub_filt + 1e-6)
X_clr = X_clr - X_clr.mean(axis=1, keepdims=True)

scaler_sub = StandardScaler()
X_sub_scaled = scaler_sub.fit_transform(X_clr)

pca_sub = PCA()
X_pca_sub = pca_sub.fit_transform(X_sub_scaled)

print(f"\nPCA explained variance:")
for i in range(min(5, len(pca_sub.explained_variance_ratio_))):
    cumvar = pca_sub.explained_variance_ratio_[:i+1].sum() * 100
    print(f"  PC{i+1}: {pca_sub.explained_variance_ratio_[i]*100:.1f}% (cumulative {cumvar:.1f}%)")

# Top loadings
print(f"\nPC1 top loadings (subtypes):")
pc1_load = pd.Series(pca_sub.components_[0], index=subtypes_kept).sort_values()
for st, val in list(pc1_load.items())[:5]:
    print(f"  {st:<15} {val:+.3f}")
print("  ...")
for st, val in list(pc1_load.items())[-5:]:
    print(f"  {st:<15} {val:+.3f}")

print(f"\nPC2 top loadings (subtypes):")
pc2_load = pd.Series(pca_sub.components_[1], index=subtypes_kept).sort_values()
for st, val in list(pc2_load.items())[:5]:
    print(f"  {st:<15} {val:+.3f}")
print("  ...")
for st, val in list(pc2_load.items())[-5:]:
    print(f"  {st:<15} {val:+.3f}")

# =====================================================================
# PART 2: ANCIENT LOCI EXPRESSION PCA
# =====================================================================
print(f"\n{'='*90}")
print("PART 2: Ancient L1 Loci Expression PCA")
print(f"{'='*90}")

# Build loci x replicate matrix (ancient only)
loci_counts = {}
for g, df in sorted(all_dfs.items()):
    anc = df[df['l1_age'] == 'ancient']
    loci_counts[g] = anc['transcript_id'].value_counts()

# Get all ancient loci
all_loci = set()
for lc in loci_counts.values():
    all_loci.update(lc.index)
all_loci = sorted(all_loci)
print(f"Total ancient loci across all: {len(all_loci)}")

# Filter: keep loci present in ≥ 5 replicates
loci_presence = np.zeros(len(all_loci))
for lc in loci_counts.values():
    for i, loc in enumerate(all_loci):
        if loc in lc.index:
            loci_presence[i] += 1

min_reps = 5
keep_loci = loci_presence >= min_reps
loci_kept = [loc for loc, k in zip(all_loci, keep_loci) if k]
print(f"Loci present in >= {min_reps} replicates: {len(loci_kept)}/{len(all_loci)}")

# Build matrix: rows=replicates, cols=loci (read count proportions)
loci_matrix = []
loci_group_labels = []
loci_cl_labels = []
for g in sorted(all_dfs.keys()):
    lc = loci_counts[g]
    total = lc.sum()
    row = [lc.get(loc, 0) / total for loc in loci_kept]
    loci_matrix.append(row)
    loci_group_labels.append(g)
    loci_cl_labels.append(all_dfs[g]['cell_line'].iloc[0])

X_loci = np.array(loci_matrix)
loci_group_labels = np.array(loci_group_labels)
loci_cl_labels = np.array(loci_cl_labels)

# CLR transform
X_loci_clr = np.log(X_loci + 1e-6)
X_loci_clr = X_loci_clr - X_loci_clr.mean(axis=1, keepdims=True)

scaler_loci = StandardScaler()
X_loci_scaled = scaler_loci.fit_transform(X_loci_clr)

pca_loci = PCA()
X_pca_loci = pca_loci.fit_transform(X_loci_scaled)

print(f"\nPCA explained variance:")
for i in range(min(5, len(pca_loci.explained_variance_ratio_))):
    cumvar = pca_loci.explained_variance_ratio_[:i+1].sum() * 100
    print(f"  PC{i+1}: {pca_loci.explained_variance_ratio_[i]*100:.1f}% (cumulative {cumvar:.1f}%)")

# Top loading loci for PC1
print(f"\nPC1 top loading loci (ancient):")
pc1_loci_load = pd.Series(pca_loci.components_[0], index=loci_kept).sort_values()
for loc, val in list(pc1_loci_load.items())[:5]:
    print(f"  {loc:<30} {val:+.4f}")
print("  ...")
for loc, val in list(pc1_loci_load.items())[-5:]:
    print(f"  {loc:<30} {val:+.4f}")

# =====================================================================
# PART 3: PLOTTING
# =====================================================================

def plot_pca(ax, X_pca, cl_arr, grp_arr, pca_obj, title):
    """Helper to make a PCA scatter with convex hulls."""
    # Base cell lines
    for i in range(len(cl_arr)):
        cl = cl_arr[i]
        if cl in treatment_colors:
            continue
        ax.scatter(X_pca[i, 0], X_pca[i, 1],
                   c=get_color(cl), marker='o', s=80, alpha=0.7,
                   edgecolors='k', linewidth=0.5, zorder=2)
    # Treatment variants
    for i in range(len(cl_arr)):
        cl = cl_arr[i]
        if cl not in treatment_colors:
            continue
        ax.scatter(X_pca[i, 0], X_pca[i, 1],
                   c=get_color(cl), marker='*', s=250, alpha=0.9,
                   edgecolors='k', linewidth=1, zorder=3)
    # Labels
    for i in range(len(cl_arr)):
        ax.annotate(grp_arr[i], (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=5.5, alpha=0.6, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')
    # Convex hulls
    for cl in set(cl_arr):
        idx = [i for i, c in enumerate(cl_arr) if c == cl]
        if len(idx) >= 3:
            points = X_pca[idx, :2]
            try:
                hull = ConvexHull(points)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(points[hull_pts, 0], points[hull_pts, 1],
                        alpha=0.08, color=get_color(cl))
                ax.plot(points[hull_pts, 0], points[hull_pts, 1],
                        '-', alpha=0.3, color=get_color(cl), linewidth=1)
            except Exception:
                pass

    ax.set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# Build legend
legend_elements = []
for cl in sorted(set(list(cl_labels) + list(loci_cl_labels))):
    color = get_color(cl)
    marker = '*' if cl in treatment_colors else 'o'
    size = 12 if cl in treatment_colors else 8
    legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                   markerfacecolor=color, markersize=size,
                                   markeredgecolor='k', markeredgewidth=0.5,
                                   label=cl))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

plot_pca(ax1, X_pca_sub, cl_labels, group_labels, pca_sub,
         'PCA: L1 Subtype Composition')
ax1.legend(handles=legend_elements, loc='best', fontsize=6.5, ncol=2)

plot_pca(ax2, X_pca_loci, loci_cl_labels, loci_group_labels, pca_loci,
         'PCA: Ancient L1 Loci Expression')
ax2.legend(handles=legend_elements, loc='best', fontsize=6.5, ncol=2)

plt.tight_layout()
fig.savefig(OUT_DIR / 'cellline_subtype_loci_pca.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR / 'cellline_subtype_loci_pca.png'}")

# =====================================================================
# PART 4: Distance analysis for both
# =====================================================================
def distance_analysis(X_pca, cl_arr, label):
    print(f"\n--- {label} ---")
    cl_centroids = {}
    for cl in set(cl_arr):
        idx = [i for i, c in enumerate(cl_arr) if c == cl]
        cl_centroids[cl] = X_pca[idx, :2].mean(axis=0)

    base_only = [cl for cl in cl_centroids if cl not in treatment_colors]
    dists = []
    for i, cl1 in enumerate(base_only):
        for cl2 in base_only[i+1:]:
            d = np.linalg.norm(cl_centroids[cl1] - cl_centroids[cl2])
            dists.append((cl1, cl2, d))
    dist_vals = [d[2] for d in dists]

    print(f"  Base CL pairwise distances: mean={np.mean(dist_vals):.2f}, "
          f"median={np.median(dist_vals):.2f}, range=[{np.min(dist_vals):.2f}, {np.max(dist_vals):.2f}]")

    for treat, parent in [('HeLa-Ars', 'HeLa'), ('MCF7-EV', 'MCF7')]:
        if treat in cl_centroids and parent in cl_centroids:
            d_treat = np.linalg.norm(cl_centroids[treat] - cl_centroids[parent])
            pct = (np.array(dist_vals) < d_treat).mean() * 100
            print(f"  {treat} vs {parent}: dist={d_treat:.2f}, "
                  f"{pct:.0f}% base pairs closer → ", end='')
            if pct > 75:
                print("OUTLIER")
            elif pct > 50:
                print("moderately different")
            else:
                print("within normal variation")

    # Replicate spread
    print(f"  Replicate spread:")
    for cl in sorted(set(cl_arr)):
        idx = [i for i, c in enumerate(cl_arr) if c == cl]
        if len(idx) >= 2:
            max_d = 0
            for ii in range(len(idx)):
                for jj in range(ii+1, len(idx)):
                    d = np.linalg.norm(X_pca[idx[ii], :2] - X_pca[idx[jj], :2])
                    max_d = max(max_d, d)
            print(f"    {cl:<12} spread={max_d:.2f} ({len(idx)} reps)")

print(f"\n{'='*90}")
print("Distance Analysis")
print(f"{'='*90}")

distance_analysis(X_pca_sub, cl_labels, "Subtype Composition")
distance_analysis(X_pca_loci, loci_cl_labels, "Ancient Loci Expression")

# =====================================================================
# PART 5: Subtype composition heatmap
# =====================================================================

# Per-cell-line mean subtype fractions
cl_subtype = pd.DataFrame(X_sub[:, keep], index=group_labels, columns=subtypes_kept)
cl_subtype['cell_line'] = cl_labels
cl_mean_sub = cl_subtype.groupby('cell_line')[subtypes_kept].mean()

# Top 20 most abundant subtypes
top20 = cl_mean_sub.mean().sort_values(ascending=False).head(20).index.tolist()

fig2, ax3 = plt.subplots(figsize=(12, 7))
# Sort cell lines by PC1
cl_pc1_sub = {}
for cl in cl_mean_sub.index:
    vals = cl_mean_sub.loc[[cl], subtypes_kept].values
    vals_clr = np.log(vals + 1e-6)
    vals_clr = vals_clr - vals_clr.mean(axis=1, keepdims=True)
    cl_pc1_sub[cl] = pca_sub.transform(scaler_sub.transform(vals_clr))[0, 0]
sort_order = sorted(cl_pc1_sub, key=lambda x: cl_pc1_sub[x])

data_hm = cl_mean_sub.loc[sort_order, top20].values
# Z-score across cell lines
data_z = (data_hm - data_hm.mean(axis=0)) / (data_hm.std(axis=0) + 1e-8)

im = ax3.imshow(data_z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax3.set_yticks(range(len(sort_order)))
ax3.set_yticklabels(sort_order, fontsize=9)
ax3.set_xticks(range(len(top20)))
ax3.set_xticklabels(top20, fontsize=7, rotation=45, ha='right')
plt.colorbar(im, ax=ax3, label='Z-score', shrink=0.8)
ax3.set_title('Top 20 L1 Subtype Composition by Cell Line (Z-scored)')

for i, cl in enumerate(sort_order):
    if cl in treatment_colors:
        ax3.get_yticklabels()[i].set_color(get_color(cl))
        ax3.get_yticklabels()[i].set_fontweight('bold')

plt.tight_layout()
fig2.savefig(OUT_DIR / 'cellline_subtype_heatmap.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'cellline_subtype_heatmap.png'}")

# =====================================================================
# PART 6: Top shared loci heatmap
# =====================================================================

# Per-cell-line mean loci fractions
cl_loci_df = pd.DataFrame(X_loci, index=loci_group_labels, columns=loci_kept)
cl_loci_df['cell_line'] = loci_cl_labels
cl_mean_loci = cl_loci_df.groupby('cell_line')[loci_kept].mean()

# Top 30 most expressed loci
top30_loci = cl_mean_loci.mean().sort_values(ascending=False).head(30).index.tolist()

fig3, ax4 = plt.subplots(figsize=(14, 7))
cl_pc1_loci = {}
for cl in cl_mean_loci.index:
    vals = cl_mean_loci.loc[[cl], loci_kept].values
    vals_clr = np.log(vals + 1e-6)
    vals_clr = vals_clr - vals_clr.mean(axis=1, keepdims=True)
    cl_pc1_loci[cl] = pca_loci.transform(scaler_loci.transform(vals_clr))[0, 0]
sort_order_loci = sorted(cl_pc1_loci, key=lambda x: cl_pc1_loci[x])

data_loci_hm = cl_mean_loci.loc[sort_order_loci, top30_loci].values
data_loci_z = (data_loci_hm - data_loci_hm.mean(axis=0)) / (data_loci_hm.std(axis=0) + 1e-8)

im2 = ax4.imshow(data_loci_z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax4.set_yticks(range(len(sort_order_loci)))
ax4.set_yticklabels(sort_order_loci, fontsize=9)
ax4.set_xticks(range(len(top30_loci)))
ax4.set_xticklabels([l.replace('_dup', '\n') for l in top30_loci],
                     fontsize=5.5, rotation=45, ha='right')
plt.colorbar(im2, ax=ax4, label='Z-score', shrink=0.8)
ax4.set_title('Top 30 Ancient L1 Loci Expression by Cell Line (Z-scored)')

for i, cl in enumerate(sort_order_loci):
    if cl in treatment_colors:
        ax4.get_yticklabels()[i].set_color(get_color(cl))
        ax4.get_yticklabels()[i].set_fontweight('bold')

plt.tight_layout()
fig3.savefig(OUT_DIR / 'cellline_loci_heatmap.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'cellline_loci_heatmap.png'}")

print("\nDone!")
