#!/usr/bin/env python3
"""
Dimensionality reduction (PCA + UMAP) of L1 landscape features per replicate.

Features per replicate:
  - Poly(A): median, mean, IQR, fraction short (<80nt), fraction long (>200nt)
  - m6A rate, psi rate
  - Read length: median
  - Young L1 fraction
  - Ancient L1: poly(A) median, m6A rate, psi rate
  - Young L1: poly(A) median (if enough reads)

Goal: see if HeLa-Ars and MCF7-EV are outliers or within normal cell-line variation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
OUT_DIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
MIN_READS = 200  # Minimum reads per replicate to include

# Cell line → replicate groups
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

# =========================================================================
# 1. Build per-replicate feature matrix
# =========================================================================
print("Building per-replicate feature matrix...")

records = []
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

        df['l1_age'] = df['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
        anc = df[df['l1_age'] == 'ancient']
        yng = df[df['l1_age'] == 'young']

        rec = {
            'group': g,
            'cell_line': cl,
            'n_reads': len(df),
            # All L1
            'polya_median': df['polya_length'].median(),
            'polya_mean': df['polya_length'].mean(),
            'polya_iqr': df['polya_length'].quantile(0.75) - df['polya_length'].quantile(0.25),
            'polya_frac_short': (df['polya_length'] < 80).mean(),
            'polya_frac_long': (df['polya_length'] > 200).mean(),
            'm6a_rate': df['m6A'].mean(),
            'psi_rate': df['psi'].mean(),
            'rdlen_median': df['read_length'].median(),
            'young_frac': (df['l1_age'] == 'young').mean(),
            # Ancient L1
            'anc_polya_median': anc['polya_length'].median() if len(anc) >= 10 else np.nan,
            'anc_m6a_rate': anc['m6A'].mean() if len(anc) >= 10 else np.nan,
            'anc_psi_rate': anc['psi'].mean() if len(anc) >= 10 else np.nan,
            'anc_rdlen_median': anc['read_length'].median() if len(anc) >= 10 else np.nan,
        }
        # Young L1 features (may be sparse)
        if len(yng) >= 10:
            rec['yng_polya_median'] = yng['polya_length'].median()
            rec['yng_m6a_rate'] = yng['m6A'].mean()
            rec['yng_psi_rate'] = yng['psi'].mean()
        else:
            rec['yng_polya_median'] = np.nan
            rec['yng_m6a_rate'] = np.nan
            rec['yng_psi_rate'] = np.nan

        # Control poly(A)
        ctrl_path = PROJECT / f'results_group/{g}/i_control/{g}_control_summary.tsv'
        if ctrl_path.exists():
            ctrl = pd.read_csv(ctrl_path, sep='\t')
            ctrl = ctrl[ctrl['qc_tag'] == 'PASS']
            if len(ctrl) >= 10:
                rec['ctrl_polya_median'] = ctrl['polya_length'].median()
                rec['l1_vs_ctrl_delta'] = rec['polya_median'] - ctrl['polya_length'].median()
            else:
                rec['ctrl_polya_median'] = np.nan
                rec['l1_vs_ctrl_delta'] = np.nan
        else:
            rec['ctrl_polya_median'] = np.nan
            rec['l1_vs_ctrl_delta'] = np.nan

        records.append(rec)
        print(f"  {g}: {len(df)} reads")

feat_df = pd.DataFrame(records)
print(f"\nTotal replicates: {len(feat_df)}")

# =========================================================================
# 2. Select features and handle NaN
# =========================================================================
# Use features available for all replicates (no young-only features for sparse CL)
feature_cols = [
    'polya_median', 'polya_mean', 'polya_iqr',
    'polya_frac_short', 'polya_frac_long',
    'm6a_rate', 'psi_rate',
    'rdlen_median', 'young_frac',
    'anc_polya_median', 'anc_m6a_rate', 'anc_psi_rate',
    'anc_rdlen_median',
]
# Do NOT include control features - MCF7-EV lacks control data

# Drop rows with NaN in feature columns
valid = feat_df.dropna(subset=feature_cols).copy()
print(f"Replicates with complete features: {len(valid)}/{len(feat_df)}")

X = valid[feature_cols].values
labels = valid['group'].values
cell_lines = valid['cell_line'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================================
# 3. PCA
# =========================================================================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA explained variance:")
for i in range(min(5, len(pca.explained_variance_ratio_))):
    cumvar = pca.explained_variance_ratio_[:i+1].sum() * 100
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% (cumulative {cumvar:.1f}%)")

# Feature loadings
print(f"\nPC1 top loadings:")
pc1_load = pd.Series(pca.components_[0], index=feature_cols).sort_values()
for feat, val in list(pc1_load.items())[:3] + list(pc1_load.items())[-3:]:
    print(f"  {feat:<25} {val:+.3f}")

print(f"\nPC2 top loadings:")
pc2_load = pd.Series(pca.components_[1], index=feature_cols).sort_values()
for feat, val in list(pc2_load.items())[:3] + list(pc2_load.items())[-3:]:
    print(f"  {feat:<25} {val:+.3f}")

# =========================================================================
# 4. UMAP (if available)
# =========================================================================
try:
    import umap
    has_umap = True
    reducer = umap.UMAP(n_neighbors=min(8, len(X_scaled)-1), min_dist=0.3,
                        random_state=42, metric='euclidean')
    X_umap = reducer.fit_transform(X_scaled)
    print("\nUMAP computed successfully")
except ImportError:
    has_umap = False
    print("\nUMAP not available, skipping")

# =========================================================================
# 5. Plotting
# =========================================================================

# Color map: parent cell lines share base color, treatment variants highlighted
base_colors = {
    'A549': '#1f77b4',
    'H9': '#ff7f0e',
    'Hct116': '#2ca02c',
    'HeLa': '#8c564b',
    'HepG2': '#e377c2',
    'HEYA8': '#7f7f7f',
    'K562': '#bcbd22',
    'MCF7': '#17becf',
    'SHSY5Y': '#aec7e8',
}
# Treatment variants
treatment_colors = {
    'HeLa-Ars': '#ff0000',  # bright red
    'MCF7-EV': '#ff00ff',   # bright magenta
}

def get_color(cl):
    if cl in treatment_colors:
        return treatment_colors[cl]
    return base_colors.get(cl, '#333333')

def get_marker(cl):
    if cl in treatment_colors:
        return 's'  # square for treatment
    return 'o'  # circle for base

# --- PCA Plot ---
n_plots = 2 if has_umap else 1
fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 7))
if n_plots == 1:
    axes = [axes]

ax = axes[0]
# Plot base cell lines first (smaller)
for i in range(len(valid)):
    cl = cell_lines[i]
    if cl in treatment_colors:
        continue
    ax.scatter(X_pca[i, 0], X_pca[i, 1],
               c=get_color(cl), marker='o', s=80, alpha=0.7,
               edgecolors='k', linewidth=0.5, zorder=2)

# Plot treatment variants on top (larger, with star)
for i in range(len(valid)):
    cl = cell_lines[i]
    if cl not in treatment_colors:
        continue
    ax.scatter(X_pca[i, 0], X_pca[i, 1],
               c=get_color(cl), marker='*', s=250, alpha=0.9,
               edgecolors='k', linewidth=1, zorder=3)

# Add labels
for i in range(len(valid)):
    ax.annotate(labels[i], (X_pca[i, 0], X_pca[i, 1]),
                fontsize=6, alpha=0.7, ha='left', va='bottom',
                xytext=(3, 3), textcoords='offset points')

# Draw convex hulls for cell lines with ≥3 replicates
from scipy.spatial import ConvexHull
for cl in set(cell_lines):
    idx = [i for i, c in enumerate(cell_lines) if c == cl]
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

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA of L1 Landscape Features (per replicate)')
ax.grid(True, alpha=0.3)

# Legend
from matplotlib.lines import Line2D
legend_elements = []
for cl in sorted(set(cell_lines)):
    color = get_color(cl)
    marker = '*' if cl in treatment_colors else 'o'
    size = 12 if cl in treatment_colors else 8
    legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                   markerfacecolor=color, markersize=size,
                                   markeredgecolor='k', markeredgewidth=0.5,
                                   label=cl))
ax.legend(handles=legend_elements, loc='best', fontsize=7, ncol=2)

# --- UMAP Plot ---
if has_umap:
    ax2 = axes[1]
    for i in range(len(valid)):
        cl = cell_lines[i]
        if cl in treatment_colors:
            continue
        ax2.scatter(X_umap[i, 0], X_umap[i, 1],
                    c=get_color(cl), marker='o', s=80, alpha=0.7,
                    edgecolors='k', linewidth=0.5, zorder=2)
    for i in range(len(valid)):
        cl = cell_lines[i]
        if cl not in treatment_colors:
            continue
        ax2.scatter(X_umap[i, 0], X_umap[i, 1],
                    c=get_color(cl), marker='*', s=250, alpha=0.9,
                    edgecolors='k', linewidth=1, zorder=3)
    for i in range(len(valid)):
        ax2.annotate(labels[i], (X_umap[i, 0], X_umap[i, 1]),
                     fontsize=6, alpha=0.7, ha='left', va='bottom',
                     xytext=(3, 3), textcoords='offset points')
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')
    ax2.set_title('UMAP of L1 Landscape Features (per replicate)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_elements, loc='best', fontsize=7, ncol=2)

plt.tight_layout()
fig.savefig(OUT_DIR / 'cellline_pca_umap.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR / 'cellline_pca_umap.png'}")

# =========================================================================
# 6. PCA feature heatmap
# =========================================================================
fig2, ax3 = plt.subplots(figsize=(10, 8))

# Create per-cell-line mean feature matrix (pooled replicates)
cl_means = valid.groupby('cell_line')[feature_cols].mean()
cl_means_scaled = pd.DataFrame(
    scaler.transform(cl_means.values),
    index=cl_means.index, columns=feature_cols
)

# Sort by PC1 score
cl_pc1 = {}
for cl in cl_means.index:
    cl_pc1[cl] = pca.transform(scaler.transform(cl_means.loc[[cl]].values))[0, 0]
sort_order = sorted(cl_pc1, key=lambda x: cl_pc1[x])

im = ax3.imshow(cl_means_scaled.loc[sort_order].values,
                aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax3.set_yticks(range(len(sort_order)))
ax3.set_yticklabels(sort_order, fontsize=9)
ax3.set_xticks(range(len(feature_cols)))
ax3.set_xticklabels([f.replace('_', '\n') for f in feature_cols],
                     fontsize=7, rotation=45, ha='right')
plt.colorbar(im, ax=ax3, label='Z-score', shrink=0.8)
ax3.set_title('L1 Feature Heatmap by Cell Line (Z-scored)')

# Highlight treatment variants
for i, cl in enumerate(sort_order):
    if cl in treatment_colors:
        ax3.get_yticklabels()[i].set_color(get_color(cl))
        ax3.get_yticklabels()[i].set_fontweight('bold')

plt.tight_layout()
fig2.savefig(OUT_DIR / 'cellline_feature_heatmap.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'cellline_feature_heatmap.png'}")

# =========================================================================
# 7. Quantify: distance of treatment variants from parent
# =========================================================================
print(f"\n{'='*90}")
print("Distance Analysis: Treatment vs Parent Cell Line")
print(f"{'='*90}")

# Compute centroid for each base cell line in PCA space
cl_centroids = {}
for cl in set(cell_lines):
    idx = [i for i, c in enumerate(cell_lines) if c == cl]
    cl_centroids[cl] = X_pca[idx, :2].mean(axis=0)

# HeLa-Ars vs HeLa
if 'HeLa-Ars' in cl_centroids and 'HeLa' in cl_centroids:
    d_ars = np.linalg.norm(cl_centroids['HeLa-Ars'] - cl_centroids['HeLa'])
    print(f"\n  HeLa-Ars centroid distance from HeLa: {d_ars:.2f}")

# MCF7-EV vs MCF7
if 'MCF7-EV' in cl_centroids and 'MCF7' in cl_centroids:
    d_ev = np.linalg.norm(cl_centroids['MCF7-EV'] - cl_centroids['MCF7'])
    print(f"  MCF7-EV centroid distance from MCF7:   {d_ev:.2f}")

# All pairwise distances between cell line centroids
base_only = [cl for cl in cl_centroids if cl not in treatment_colors]
dists = []
for i, cl1 in enumerate(base_only):
    for cl2 in base_only[i+1:]:
        d = np.linalg.norm(cl_centroids[cl1] - cl_centroids[cl2])
        dists.append(d)

if dists:
    print(f"\n  Base cell line pairwise distances (PCA space):")
    print(f"    Mean: {np.mean(dists):.2f}")
    print(f"    Median: {np.median(dists):.2f}")
    print(f"    Range: [{np.min(dists):.2f}, {np.max(dists):.2f}]")

    if 'HeLa-Ars' in cl_centroids and 'HeLa' in cl_centroids:
        pct = (np.array(dists) < d_ars).mean() * 100
        print(f"\n  HeLa-Ars vs HeLa distance ({d_ars:.2f}):")
        print(f"    {pct:.0f}% of base CL pairs are closer → ", end='')
        if pct > 75:
            print("HeLa-Ars is an OUTLIER (farther than most CL pairs)")
        elif pct > 50:
            print("HeLa-Ars is MODERATELY different")
        else:
            print("HeLa-Ars is within normal CL variation")

    if 'MCF7-EV' in cl_centroids and 'MCF7' in cl_centroids:
        pct_ev = (np.array(dists) < d_ev).mean() * 100
        print(f"\n  MCF7-EV vs MCF7 distance ({d_ev:.2f}):")
        print(f"    {pct_ev:.0f}% of base CL pairs are closer → ", end='')
        if pct_ev > 75:
            print("MCF7-EV is an OUTLIER (farther than most CL pairs)")
        elif pct_ev > 50:
            print("MCF7-EV is MODERATELY different")
        else:
            print("MCF7-EV is within normal CL variation")

# Within-cell-line replicate distances
print(f"\n  Within-cell-line replicate spread (max pairwise dist):")
for cl in sorted(set(cell_lines)):
    idx = [i for i, c in enumerate(cell_lines) if c == cl]
    if len(idx) >= 2:
        max_d = 0
        for i_idx in range(len(idx)):
            for j_idx in range(i_idx+1, len(idx)):
                d = np.linalg.norm(X_pca[idx[i_idx], :2] - X_pca[idx[j_idx], :2])
                max_d = max(max_d, d)
        print(f"    {cl:<12} max spread: {max_d:.2f} ({len(idx)} reps)")

# =========================================================================
# 8. Bar plot: PC1 scores by cell line
# =========================================================================
fig3, ax4 = plt.subplots(figsize=(10, 5))

# Per-replicate PC1 scores
cl_order = sorted(set(cell_lines),
                  key=lambda cl: np.mean([X_pca[i, 0] for i, c in enumerate(cell_lines) if c == cl]))

x_pos = []
x_labels = []
pos = 0
for cl in cl_order:
    idx = [i for i, c in enumerate(cell_lines) if c == cl]
    scores = X_pca[idx, 0]
    color = get_color(cl)
    for s in scores:
        ax4.bar(pos, s, color=color, edgecolor='k', linewidth=0.5, width=0.8)
        pos += 1
    x_pos.append(pos - len(idx)/2 - 0.5)
    x_labels.append(cl)
    pos += 0.5  # gap between cell lines

ax4.set_xticks(x_pos)
ax4.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('PC1 score')
ax4.set_title('PC1 Score by Cell Line Replicate')
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.grid(True, axis='y', alpha=0.3)

# Highlight treatment variants
for i, lab in enumerate(ax4.get_xticklabels()):
    cl = x_labels[i]
    if cl in treatment_colors:
        lab.set_color(get_color(cl))
        lab.set_fontweight('bold')

plt.tight_layout()
fig3.savefig(OUT_DIR / 'cellline_pc1_barplot.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR / 'cellline_pc1_barplot.png'}")

# Save feature matrix
feat_df.to_csv(OUT_DIR / 'cellline_feature_matrix.tsv', sep='\t', index=False)
print(f"Saved: {OUT_DIR / 'cellline_feature_matrix.tsv'}")

print("\nDone!")
