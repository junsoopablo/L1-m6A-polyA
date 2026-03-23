"""
Shared figure style for L1 m6A manuscript.
Publication-quality defaults — muted, clean, data-forward.

Design principles (from Ten Simple Rules, Nature Cell Bio checklist, etc.):
  - Muted colorblind-friendly palette (Paul Tol inspired)
  - Show raw data points alongside summaries
  - Minimal chartjunk: no unnecessary fills, gradients, thick borders
  - Soft text color (#373737) instead of pure black
  - 600 DPI for publication; vector PDF output
  - Font hierarchy: panel labels 8pt > axis labels 7pt > ticks 7pt (Nature spec)
  - Print-friendly: all elements visible at 100% on A4/letter paper
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# scienceplots removed — all rcParams set explicitly below.
# (scienceplots 'no-latex' style sets Roboto, which triggers
#  "Font family 'Roboto' not found" warnings on most systems.)

# ── Figure dimensions (Nature: 183mm double, 89mm single) ──
MM = 1 / 25.4
FULL_WIDTH  = 165 * MM   # ~6.50 in  (matches LaTeX textwidth at 1in margins)
HALF_WIDTH  = 80 * MM    # ~3.15 in  (half of FULL_WIDTH minus gap)
PANEL_HEIGHT = HALF_WIDTH * 0.85  # ~2.98 in — standard panel height for uniform compose scaling

# ── Color palette — muted, colorblind-safe ──
# Based on Paul Tol "muted" + hand-tuned for L1 manuscript context
C_L1       = '#CC6677'   # muted rose — L1 primary
C_CTRL     = '#88CCEE'   # soft cyan — control
C_YOUNG    = '#44AA99'   # teal — young L1
C_ANCIENT  = '#DDCC77'   # sand — ancient L1
C_CATB     = '#AA4499'   # muted purple — Category B
C_STRESS   = '#882255'   # wine — arsenite stress
C_NORMAL   = '#4477AA'   # steel — normal/unstressed
C_GREY     = '#BBBBBB'   # neutral grey — ns / background
C_HIGHLIGHT = '#EE8866'  # warm peach — key highlight (sparing use)
C_TEXT     = '#373737'   # soft near-black for all text

# ── Semantic font sizes (print legibility at 183mm) ──
FS_ANNOT       = 8      # annotation text, stat results
FS_ANNOT_SMALL = 7      # smaller annotations, significance
FS_LEGEND      = 7.5    # in-plot legends
FS_LEGEND_SMALL = 6.5   # compact legends
FS_CBAR        = 7.5    # colorbar labels

# ── Line widths (scaled for FULL_WIDTH=165mm) ──
LW_AXIS        = 0.6    # axes & spines
LW_TICK        = 0.5    # tick marks
LW_DATA        = 1.0    # data lines (ECDF, regression)
LW_DATA_SEC    = 0.8    # secondary data lines (dashed, reference)
LW_REF         = 0.6    # reference/guide lines (y=1, x=0)
LW_BRACKET     = 0.6    # significance brackets
LW_MEDIAN      = 0.8    # median lines in violins
LW_CONNECT     = 0.5    # connecting lines (paired dots)

# ── Scatter marker sizes ──
S_POINT        = 30     # standard data point
S_POINT_SMALL  = 2      # point cloud / density scatter
S_POINT_LARGE  = 50     # emphasized data point

# Semantic aliases
C_HELA     = C_NORMAL
C_ARS      = C_STRESS

# ── rcParams ──
def setup_style():
    """Set publication rcParams."""
    plt.rcParams.update({
        # Fonts — Nature mandates sans-serif (Helvetica/Arial)
        # Sized for print at 183mm width
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.2,

        # Lines — calibrated for FULL_WIDTH=165mm
        'axes.linewidth': LW_AXIS,
        'xtick.major.width': LW_TICK,
        'ytick.major.width': LW_TICK,
        'xtick.minor.width': LW_TICK * 0.7,
        'ytick.minor.width': LW_TICK * 0.7,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.top': False,          # SciencePlots sets True; disable for clean L-shape
        'ytick.right': False,        # SciencePlots sets True; disable for clean L-shape
        'xtick.minor.visible': False, # SciencePlots enables; disable to avoid top-edge dots
        'ytick.minor.visible': False,
        'lines.linewidth': LW_DATA,
        'lines.markersize': 4.0,

        # Spines — clean L-shape
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Colors — soft near-black text
        'text.color': C_TEXT,
        'axes.labelcolor': C_TEXT,
        'xtick.color': C_TEXT,
        'ytick.color': C_TEXT,
        'axes.edgecolor': C_TEXT,

        # Layout — constrained_layout for stable panel spacing
        'figure.constrained_layout.use': True,

        # Output
        'figure.dpi': 150,       # screen preview
        'savefig.dpi': 600,      # publication quality
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'pdf.fonttype': 42,      # TrueType in PDF (editable)
        'ps.fonttype': 42,

        # Background
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
    })

# ── Helper functions ──
def panel_label(ax, letter, x=-0.18, y=1.08):
    """Add bold lowercase panel label (Nature style, 8pt)."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left',
            color='black')

def significance_bracket(ax, x1, x2, y, h, text, fontsize=7, color='#373737'):
    """Draw a thin significance bracket."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=LW_BRACKET, color=color)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom',
            fontsize=fontsize, color=color)

def significance_text(p):
    """Return significance string from p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'

def save_figure(fig, name, formats=('pdf', 'svg')):
    """Save figure in specified formats (default: PDF + SVG)."""
    for fmt in formats:
        fig.savefig(f'{name}.{fmt}', format=fmt)
    plt.close(fig)

def despine(ax):
    """Remove top and right spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def add_strip(ax, data_list, positions, colors=None, size=1.5, alpha=0.25,
              jitter=0.12, seed=42):
    """Overlay jittered strip of raw data points on a violin/box.

    Parameters
    ----------
    data_list : list of arrays
    positions : list of x-positions (matching data_list)
    colors : list of colors or single color
    """
    rng = np.random.RandomState(seed)
    if colors is None:
        colors = [C_GREY] * len(data_list)
    elif isinstance(colors, str):
        colors = [colors] * len(data_list)
    for i, (data, pos) in enumerate(zip(data_list, positions)):
        n = len(data)
        if n > 2000:
            # subsample for readability
            idx = rng.choice(n, 2000, replace=False)
            data = np.asarray(data)[idx]
            n = 2000
        x_jitter = pos + rng.uniform(-jitter, jitter, n)
        ax.scatter(x_jitter, data, s=size, alpha=alpha, color=colors[i],
                   edgecolors='none', rasterized=True, zorder=1)

def median_line(ax, data, pos, width=0.25, color='black', lw=1.2):
    """Draw a horizontal median line at a violin/strip position."""
    med = np.median(data)
    ax.hlines(med, pos - width, pos + width, color=color, lw=lw, zorder=4)
    return med

def ecdf_plot(ax, data, color, label, lw=1.2, ls='-'):
    """Plot ECDF. Efficient for large n via step function."""
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.step(sorted_data, y, where='post', color=color, lw=lw, ls=ls, label=label)


def dumbbell_plot(ax, labels, val1, val2, color1, color2,
                  label1='', label2='', horizontal=True,
                  marker_size=50, line_color='#BDC3C7', line_width=1.5,
                  marker1='o', marker2='o'):
    """Connected dot plot for paired comparisons (dumbbell chart).

    Parameters
    ----------
    labels : list of str — category labels
    val1, val2 : arrays of paired values
    color1, color2 : colors for each series
    horizontal : if True, categories on y-axis, values on x-axis
    """
    n = len(labels)
    positions = np.arange(n)

    if horizontal:
        for i in range(n):
            ax.plot([val1[i], val2[i]], [i, i], color=line_color,
                    lw=line_width, zorder=1)
        ax.scatter(val1, positions, s=marker_size, color=color1,
                   edgecolors='white', linewidths=0.5, zorder=3,
                   marker=marker1, label=label1 if label1 else None)
        ax.scatter(val2, positions, s=marker_size, color=color2,
                   edgecolors='white', linewidths=0.5, zorder=3,
                   marker=marker2, label=label2 if label2 else None)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
    else:
        for i in range(n):
            ax.plot([i, i], [val1[i], val2[i]], color=line_color,
                    lw=line_width, zorder=1)
        ax.scatter(positions, val1, s=marker_size, color=color1,
                   edgecolors='white', linewidths=0.5, zorder=3,
                   marker=marker1, label=label1 if label1 else None)
        ax.scatter(positions, val2, s=marker_size, color=color2,
                   edgecolors='white', linewidths=0.5, zorder=3,
                   marker=marker2, label=label2 if label2 else None)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)


def forest_plot(ax, labels, estimates, ci_lo, ci_hi, colors=None,
                ref_line=1.0, horizontal=True, marker_size=50):
    """Forest plot: point estimates with CI whiskers.

    Parameters
    ----------
    estimates : array of point estimates
    ci_lo, ci_hi : arrays of CI lower/upper bounds
    ref_line : reference line value (None to skip)
    """
    n = len(labels)
    positions = np.arange(n)
    if colors is None:
        colors = [C_L1] * n
    elif isinstance(colors, str):
        colors = [colors] * n

    cap = 0.12

    if horizontal:
        for i in range(n):
            ax.plot([ci_lo[i], ci_hi[i]], [i, i], color=colors[i],
                    lw=1.5, zorder=2, solid_capstyle='butt')
            ax.plot([ci_lo[i], ci_lo[i]], [i - cap, i + cap],
                    color=colors[i], lw=1.0, zorder=2)
            ax.plot([ci_hi[i], ci_hi[i]], [i - cap, i + cap],
                    color=colors[i], lw=1.0, zorder=2)
        ax.scatter(estimates, positions, s=marker_size, color=colors,
                   edgecolors='white', linewidths=0.5, zorder=3)
        if ref_line is not None:
            ax.axvline(ref_line, color='#CCCCCC', lw=0.7, ls='--', zorder=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
    else:
        for i in range(n):
            ax.plot([i, i], [ci_lo[i], ci_hi[i]], color=colors[i],
                    lw=1.5, zorder=2, solid_capstyle='butt')
            ax.plot([i - cap, i + cap], [ci_lo[i], ci_lo[i]],
                    color=colors[i], lw=1.0, zorder=2)
            ax.plot([i - cap, i + cap], [ci_hi[i], ci_hi[i]],
                    color=colors[i], lw=1.0, zorder=2)
        ax.scatter(positions, estimates, s=marker_size, color=colors,
                   edgecolors='white', linewidths=0.5, zorder=3)
        if ref_line is not None:
            ax.axhline(ref_line, color='#CCCCCC', lw=0.7, ls='--', zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)


def lollipop_plot(ax, labels, values, colors=None, horizontal=True,
                  marker_size=50, stem_width=1.5, ref_value=0):
    """Lollipop/Cleveland dot plot: stem from ref_value to point.

    Parameters
    ----------
    labels : list of str
    values : array of values
    colors : array of colors or single color
    horizontal : if True, categories on y-axis
    ref_value : baseline for stems
    """
    n = len(labels)
    positions = np.arange(n)
    if colors is None:
        colors = [C_L1] * n
    elif isinstance(colors, str):
        colors = [colors] * n

    if horizontal:
        for i in range(n):
            ax.plot([ref_value, values[i]], [i, i], color=colors[i],
                    lw=stem_width, zorder=1, solid_capstyle='butt')
        ax.scatter(values, positions, s=marker_size, color=colors,
                   edgecolors='white', linewidths=0.5, zorder=3)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
    else:
        for i in range(n):
            ax.plot([i, i], [ref_value, values[i]], color=colors[i],
                    lw=stem_width, zorder=1, solid_capstyle='butt')
        ax.scatter(positions, values, s=marker_size, color=colors,
                   edgecolors='white', linewidths=0.5, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
