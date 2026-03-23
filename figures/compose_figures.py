#!/usr/bin/env python3
"""
Compose individual panel PDFs into composite figures using LaTeX standalone.
Uses HEIGHT-MATCHING: all panels in a row share the same height,
widths auto-calculated from aspect ratios to fill 183mm total.

Outputs: fig1.pdf, fig2.pdf, fig3.pdf
"""
import subprocess, os, re

FIGDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(FIGDIR)

TOTAL_WIDTH = 165  # mm (matches LaTeX textwidth at 1in margins — no scaling)
GAP = 4            # mm between panels


def get_pdf_dims(path):
    """Read PDF MediaBox to get width/height in mm."""
    with open(path, 'rb') as f:
        content = f.read()
    match = re.search(
        rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]',
        content
    )
    if match:
        x1, y1, x2, y2 = [float(x) for x in match.groups()]
        w_mm = (x2 - x1) * 25.4 / 72
        h_mm = (y2 - y1) * 25.4 / 72
        return w_mm, h_mm
    raise ValueError(f"Could not read MediaBox from {path}")


def calc_row_height(panels, total_w, gap):
    """Calculate row height so panels fill total width.

    panels: list of (filename,) — aspect ratios read from PDFs.
    Returns (height_mm, list of (filename, width_mm)).
    """
    ratios = []
    for fname in panels:
        w, h = get_pdf_dims(fname)
        ratios.append(w / h)

    n_gaps = len(panels) - 1
    available = total_w - n_gaps * gap
    row_h = available / sum(ratios)

    result = []
    for fname, r in zip(panels, ratios):
        result.append((fname, row_h * r))

    return row_h, result


def make_tex(rows_spec, row_gap='3mm'):
    """Generate LaTeX source. rows_spec: list of [(fname, width_mm), ...]"""
    lines = [
        r'\documentclass[border=1mm]{standalone}',
        r'\usepackage{graphicx}',
        r'\usepackage{xcolor}',
        r'\pagecolor{white}',
        r'\begin{document}%',
        rf'\begin{{minipage}}{{{TOTAL_WIDTH}mm}}%',
    ]
    for ri, row in enumerate(rows_spec):
        for pi, (fname, wmm) in enumerate(row):
            lines.append(rf'\begin{{minipage}}[b]{{{wmm:.1f}mm}}%')
            lines.append(rf'  \includegraphics[width=\textwidth]{{{fname}}}%')
            lines.append(r'\end{minipage}%')
            if pi < len(row) - 1:
                lines.append(rf'\hspace{{{GAP}mm}}%')
        if ri < len(rows_spec) - 1:
            lines.append('')
            lines.append(rf'\vspace{{{row_gap}}}')
            lines.append('')
    lines.append(r'\end{minipage}%')
    lines.append(r'\end{document}')
    return '\n'.join(lines)


def compile_and_rename(texfile, figname):
    """Compile .tex, rename output, clean up."""
    pdflatex_bin = '/blaze/apps/envs/texlive/20250808/bin/x86_64-linux/pdflatex'
    result = subprocess.run(
        [pdflatex_bin, '-interaction=nonstopmode', texfile],
        capture_output=True, text=True, cwd=FIGDIR
    )
    if result.returncode != 0:
        print(f"  ERROR compiling {texfile}")
        # Show last few lines for debugging
        for line in result.stdout.split('\n')[-10:]:
            if line.strip():
                print(f"    {line}")
        return False

    base = texfile.replace('.tex', '')
    for ext in ['.aux', '.log', '.tex']:
        p = os.path.join(FIGDIR, f'{base}{ext}')
        if os.path.exists(p):
            os.remove(p)

    src = os.path.join(FIGDIR, f'{base}.pdf')
    dst = os.path.join(FIGDIR, f'{figname}.pdf')
    if os.path.exists(src):
        os.rename(src, dst)
    return True


# ══════════════════════════════════════════════
# Figure 1 (NEW): m6A enrichment + arsenite shortening + mechanism
# [a=old1a | b=old2a] / [c=old2c | d=old2e] / [e=old2f | f=old2d]
# ══════════════════════════════════════════════
print("Fig 1 (new):")
h1, row1 = calc_row_height(['fig1a.pdf', 'fig2a.pdf'], TOTAL_WIDTH, GAP)
h2, row2 = calc_row_height(['fig2c.pdf', 'fig2e.pdf'], TOTAL_WIDTH, GAP)
h3, row3 = calc_row_height(['fig2f.pdf', 'fig2d.pdf'], TOTAL_WIDTH, GAP)
print(f"  Row 1: h={h1:.1f}mm — " + ", ".join(f"{f}={w:.1f}mm" for f, w in row1))
print(f"  Row 2: h={h2:.1f}mm — " + ", ".join(f"{f}={w:.1f}mm" for f, w in row2))
print(f"  Row 3: h={h3:.1f}mm — " + ", ".join(f"{f}={w:.1f}mm" for f, w in row3))

tex1 = make_tex([row1, row2, row3])
with open('compose_fig1.tex', 'w') as f:
    f.write(tex1)
compile_and_rename('compose_fig1.tex', 'fig1')

# ══════════════════════════════════════════════
# Figure 2 (NEW): m6A-poly(A) coupling under stress
# [a=ECDF | b=slope | c=heatmap] — single row, 3 columns
# ══════════════════════════════════════════════
print("\nFig 2 (new):")
h1, row1 = calc_row_height(['fig3a.pdf', 'fig3b.pdf'], TOTAL_WIDTH, GAP)
print(f"  Row 1: h={h1:.1f}mm — " + ", ".join(f"{f}={w:.1f}mm" for f, w in row1))

tex2 = make_tex([row1])
with open('compose_fig2.tex', 'w') as f:
    f.write(tex2)
compile_and_rename('compose_fig2.tex', 'fig2')

# ══════════════════════════════════════════════
# Figure 4: [a | b] / [c | d]
# Sequence features of young L1 stress immunity
# ══════════════════════════════════════════════
print("\nFig 4:")
h1, row1 = calc_row_height(['fig4a_immunity_features.pdf', 'fig4b_feature_immunity.pdf'], TOTAL_WIDTH, GAP)
h2, row2 = calc_row_height(['fig4c_immunity_score.pdf', 'fig4d_motif_landscape.pdf'], TOTAL_WIDTH, GAP)
print(f"  Row 1: h={h1:.1f}mm — " + ", ".join(f"{f}={w:.1f}mm" for f, w in row1))
print(f"  Row 2: h={h2:.1f}mm — " + ", ".join(f"{f}={w:.1f}mm" for f, w in row2))

tex4 = make_tex([row1, row2])
with open('compose_fig4.tex', 'w') as f:
    f.write(tex4)
compile_and_rename('compose_fig4.tex', 'fig4')

# ══════════════════════════════════════════════
# Figure S14: [a | b]  (scatter + Δpoly(A) bar)
# ══════════════════════════════════════════════
print("\nFig S14:")
h1, row1 = calc_row_height(['figS14a.pdf', 'figS14b.pdf'], TOTAL_WIDTH, GAP)
print(f"  Row 1: h={h1:.1f}mm — " + ", ".join(f"{f}={w:.1f}mm" for f, w in row1))

texS14 = make_tex([row1])
with open('compose_figS14.tex', 'w') as f:
    f.write(texS14)
compile_and_rename('compose_figS14.tex', 'figS14')

print("\nDone.")
