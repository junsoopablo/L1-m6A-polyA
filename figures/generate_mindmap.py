#!/usr/bin/env python3
"""
Paper logic mind map v2 — incorporates ongoing analyses + data strategy.
Page 1: Current logic + weakness diagnosis
Page 2: Data strategy — what's coming, what to get, how each fills gaps
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── Korean font ──
import matplotlib.font_manager as fm
for _fpath in ['/qbio/junsoopablo/.fonts/NotoSansKR-Regular.otf',
               '/qbio/junsoopablo/.fonts/NotoSansKR-Bold.otf']:
    if os.path.exists(_fpath):
        fm.fontManager.addfont(_fpath)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Noto Sans KR', 'DejaVu Sans', 'Arial'],
    'font.size': 8,
    'pdf.fonttype': 42,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# Colors
C_STRONG = '#2E86AB'
C_MEDIUM = '#F6AE2D'
C_WEAK   = '#E84855'
C_BG     = '#F8F9FA'
C_TITLE  = '#1B2838'
C_LINE   = '#ADB5BD'
C_GREEN  = '#2A9D8F'
C_PURPLE = '#7B2D8E'
C_NEW    = '#FF6B35'  # orange for new data
C_COMING = '#3D5A80'  # steel blue for in-progress

def strength_color(stars):
    if stars >= 4.5: return C_STRONG
    elif stars >= 3.5: return '#5FAD56'
    elif stars >= 2.5: return C_MEDIUM
    else: return C_WEAK

def draw_box(ax, x, y, w, h, text, color='#FFFFFF', edge_color='#333333',
             fontsize=7, fontweight='normal', alpha=1.0, text_color='#333333',
             ha='center', va='center', zorder=3, lw=0.8, rounded=True, ls='-'):
    if rounded:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor=edge_color,
                             linewidth=lw, alpha=alpha, zorder=zorder,
                             linestyle=ls)
    else:
        box = mpatches.Rectangle((x - w/2, y - h/2), w, h,
                                  facecolor=color, edgecolor=edge_color,
                                  linewidth=lw, alpha=alpha, zorder=zorder,
                                  linestyle=ls)
    ax.add_patch(box)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=zorder+1,
            linespacing=1.3)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='#ADB5BD', lw=1.0, style='->', zorder=1):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)

OUTDIR = os.path.dirname(os.path.abspath(__file__))

with PdfPages(f'{OUTDIR}/mindmap.pdf') as pdf:

    # ══════════════════════════════════════════════════════════════
    # PAGE 1: Current Paper Logic + Weakness Diagnosis
    # ══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 15))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 15)
    ax.axis('off')

    # Title
    ax.text(5.5, 14.6, 'Page 1: 현재 논문 로직 + 약점 진단', fontsize=14, fontweight='bold',
            ha='center', va='center', color=C_TITLE)
    ax.text(5.5, 14.25, 'm6A marks LINE-1 for stress-specific poly(A) tail protection',
            fontsize=9, ha='center', va='center', color='#666666', style='italic')

    # ── THESIS BOX ──
    draw_box(ax, 5.5, 13.5, 9.0, 0.6,
             'THESIS: m6A -> L1 poly(A) tail stress protection\n'
             '"m6A enrichment (Part1) + selective shortening (Part2) + dose-response (Part3)"',
             color='#E8F4FD', edge_color=C_STRONG, fontsize=8, fontweight='bold',
             text_color=C_TITLE, lw=1.5)

    # Arrows from thesis to 3 parts
    for xp in [2.5, 5.5, 8.5]:
        draw_arrow(ax, xp, 13.15, xp, 12.55, color=C_STRONG, lw=1.5)

    # ═══ PART 1 ═══
    draw_box(ax, 2.5, 12.2, 3.5, 0.55,
             'Part 1: L1 m6A Enrichment\n현재 강도: 5/5',
             color='#D4EDDA', edge_color='#28A745', fontsize=8, fontweight='bold',
             text_color='#155724', lw=1.2)

    y1 = 11.35
    p1_ev = [
        ('Per-site 1.33x\nP~0, 115K reads', 5),
        ('23/23 libraries\nabove diagonal', 5),
        ('18/18 motifs\nenriched', 5),
        ('Threshold-free\n1.15x->1.50x', 5),
    ]
    for i, (text, sc) in enumerate(p1_ev):
        xi = 1.1 + i * 0.95
        draw_box(ax, xi, y1, 0.85, 0.6, text,
                 color=strength_color(sc), edge_color='#666', fontsize=5,
                 text_color='white', lw=0.4)
        draw_arrow(ax, xi, 11.7, xi, y1 + 0.33, color='#999', lw=0.5)

    # Part 1 gap
    draw_box(ax, 2.5, 10.55, 3.5, 0.35,
             'GAP: Writer 미확인. METTL3-dependent인지 불명',
             color='#FFF3CD', edge_color=C_MEDIUM, fontsize=6, lw=0.6,
             text_color='#856404')
    draw_arrow(ax, 2.5, y1 - 0.33, 2.5, 10.75, color='#999', lw=0.5)

    # ═══ PART 2 ═══
    draw_box(ax, 5.5, 12.2, 3.5, 0.55,
             'Part 2: Arsenite Poly(A) Shortening\n현재 강도: 4/5',
             color='#D4EDDA', edge_color='#28A745', fontsize=8, fontweight='bold',
             text_color='#155724', lw=1.2)

    p2_ev = [
        ('L1-specific\nCtrl unchanged', 5),
        ('Post-tx\n9 CL validation', 5),
        ('Young immune\nn=292', 4),
        ('ChromHMM\n4-axis model', 3),
    ]
    for i, (text, sc) in enumerate(p2_ev):
        xi = 4.1 + i * 0.95
        draw_box(ax, xi, y1, 0.85, 0.6, text,
                 color=strength_color(sc), edge_color='#666', fontsize=5,
                 text_color='white', lw=0.4)
        draw_arrow(ax, xi, 11.7, xi, y1 + 0.33, color='#999', lw=0.5)

    draw_box(ax, 5.5, 10.55, 3.5, 0.35,
             'GAP: Decay pathway 미확인. XRN1? Exosome? PARN?',
             color='#FFF3CD', edge_color=C_MEDIUM, fontsize=6, lw=0.6,
             text_color='#856404')
    draw_arrow(ax, 5.5, y1 - 0.33, 5.5, 10.75, color='#999', lw=0.5)

    # ═══ PART 3 ═══
    draw_box(ax, 8.5, 12.2, 3.5, 0.55,
             'Part 3: m6A-Poly(A) Protection\n현재 강도: 3/5',
             color='#FFF3CD', edge_color=C_MEDIUM, fontsize=8, fontweight='bold',
             text_color='#856404', lw=1.2)

    p3_ev = [
        ('OLS interaction\nP=2.7e-5', 4),
        ('Quartile\nQ1->Q4 +64nt', 4),
        ('Decay zone\n1.9x (P=8e-9)', 3.5),
        ('Correlation\nr=0.18 only', 2),
    ]
    for i, (text, sc) in enumerate(p3_ev):
        xi = 7.1 + i * 0.95
        draw_box(ax, xi, y1, 0.85, 0.6, text,
                 color=strength_color(sc), edge_color='#666', fontsize=5,
                 text_color='white', lw=0.4)
        draw_arrow(ax, xi, 11.7, xi, y1 + 0.33, color='#999', lw=0.5)

    draw_box(ax, 8.5, 10.55, 3.5, 0.35,
             'GAP: Causal 증거 없음. Correlation only. KO 필요',
             color='#FFDADA', edge_color=C_WEAK, fontsize=6, lw=0.6,
             text_color='#721C24')
    draw_arrow(ax, 8.5, y1 - 0.33, 8.5, 10.75, color='#999', lw=0.5)

    # ═══ REVIEWER ATTACK POINTS ═══
    y_prob = 9.4
    ax.text(5.5, y_prob + 0.5, 'Reviewer 예상 공격점 (4대 약점)', fontsize=11,
            fontweight='bold', ha='center', color=C_WEAK)

    problems = [
        ('1. "Protection" = 인과 주장\n   증거는 상관관계만\n'
         '   m6A writer KO 필요',
         '치명적', C_WEAK),
        ('2. HeLa 1 cell line\n   1 timepoint, 1 stress\n'
         '   일반화 가능성 의문',
         '심각', C_WEAK),
        ('3. DRS survivorship bias\n   분해된 low-m6A RNA는\n'
         '   애초에 관측 불가',
         '심각', '#D4750F'),
        ('4. Decay pathway 불명\n   XRN1? Exosome? PARN?\n'
         '   메커니즘 제시 없음',
         '중간', C_MEDIUM),
    ]
    for i, (text, sev, color) in enumerate(problems):
        xi = 1.5 + i * 2.5
        draw_box(ax, xi, y_prob - 0.5, 2.2, 1.1, text,
                 color=color, edge_color=color, fontsize=6,
                 text_color='white', lw=0.8, alpha=0.85)
        ax.text(xi, y_prob + 0.1, sev, fontsize=7, fontweight='bold',
                ha='center', color=color)

    # ═══ HOW NEW DATA ADDRESSES EACH ═══
    y_fix = 7.4
    ax.text(5.5, y_fix + 0.45, '진행중/계획중 데이터로 해결 가능한 약점', fontsize=11,
            fontweight='bold', ha='center', color=C_GREEN)

    fixes = [
        ('METTL3 KO\n(PRJEB40872, 진행중)\n\n'
         'KO시 m6A 소실 -> Writer 확인\n'
         'KO시 poly(A) 변화 -> 인과 증거\n'
         '-> 약점 1번 직접 해결',
         C_NEW),
        ('XRN1 KD MAFIA 완료\n(PRJNA842344, 진행중)\n\n'
         'XRN1 KD시 m6A 불변 확인\n'
         '-> m6A는 decay 상류\n'
         '-> 약점 4번 부분 해결',
         C_COMING),
        ('MCF7 STM2457\n(GSE269294, 공개)\n\n'
         'MCF7 = 우리 CL!\n'
         'METTL3 inhibitor 5 reps\n'
         '-> 약점 2번 부분 해결',
         '#5FAD56'),
    ]
    for i, (text, color) in enumerate(fixes):
        xi = 2.0 + i * 3.3
        draw_box(ax, xi, y_fix - 0.65, 2.8, 1.5, text,
                 color='#FFFFFF', edge_color=color, fontsize=6,
                 text_color='#333', lw=1.5, ls='-')
        # Arrow from fix to problem
        prob_xi = 1.5 + [0, 3, 1][i] * 2.5  # maps to problem 1, 4, 2
        draw_arrow(ax, xi, y_fix + 0.1, prob_xi, y_prob - 1.1,
                   color=color, lw=1.2, style='->')

    # ═══ EVIDENCE vs NOVELTY MATRIX (updated) ═══
    y_mat = 4.5
    ax.text(5.5, y_mat + 0.85, '증거 vs 신규성 매트릭스 (데이터 추가시 예상 변화)', fontsize=10,
            fontweight='bold', ha='center', color=C_TITLE)

    # Draw axes
    ax.annotate('', xy=(9.8, y_mat - 1.6), xytext=(1.5, y_mat - 1.6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1))
    ax.annotate('', xy=(1.5, y_mat + 0.5), xytext=(1.5, y_mat - 1.6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1))
    ax.text(5.5, y_mat - 1.85, 'Novelty ->', ha='center', fontsize=7, color='#333')
    ax.text(1.0, y_mat - 0.5, 'Evidence', ha='center', fontsize=7, color='#333',
            rotation=90)

    # Current positions (filled)
    items_now = [
        (3.5, y_mat + 0.2, 'Part 1\n(now)', C_STRONG),
        (6.0, y_mat + 0.15, 'Part 2\n(now)', C_STRONG),
        (8.5, y_mat - 1.0, 'Part 3\n(now)', C_WEAK),
    ]
    for x, y, label, color in items_now:
        draw_box(ax, x, y, 1.2, 0.45, label,
                 color=color, edge_color=color, fontsize=6,
                 text_color='white', lw=0.5, alpha=0.8)

    # Future positions (dashed)
    items_future = [
        (4.0, y_mat + 0.35, 'Part 1\n+METTL3 KO', C_STRONG),
        (6.5, y_mat + 0.0, 'Part 2\n+XRN1', '#5FAD56'),
        (8.5, y_mat - 0.2, 'Part 3\n+METTL3 KO', C_MEDIUM),
    ]
    for x, y, label, color in items_future:
        draw_box(ax, x, y, 1.2, 0.45, label,
                 color='#FFFFFF', edge_color=color, fontsize=5.5,
                 text_color=color, lw=1.2, ls='--')

    # Arrows showing movement
    draw_arrow(ax, 3.8, y_mat + 0.2, 4.0, y_mat + 0.35, color=C_NEW, lw=1.5)
    draw_arrow(ax, 6.2, y_mat + 0.15, 6.5, y_mat + 0.0, color=C_COMING, lw=1.5)
    draw_arrow(ax, 8.5, y_mat - 0.75, 8.5, y_mat - 0.2, color=C_NEW, lw=1.5, style='->')

    # Ideal zone
    ax.add_patch(mpatches.Rectangle((7.5, y_mat - 0.1), 2.5, 0.7,
                 facecolor='#D4EDDA', edgecolor='#28A745', lw=0.8,
                 alpha=0.25, linestyle='--', zorder=1))
    ax.text(8.75, y_mat + 0.5, 'Ideal zone', fontsize=6,
            ha='center', color='#28A745', style='italic')

    # ═══ CONCLUSION ═══
    y_c = 1.8
    draw_box(ax, 5.5, y_c, 9.5, 1.6,
             '핵심 진단\n\n'
             'Part 3 (m6A protection)이 논문의 핵심 novelty이나 가장 약함.\n'
             'METTL3 KO가 Part 1 + Part 3을 동시에 보강하는 최우선 데이터.\n'
             'XRN1 KD MAFIA 완료시 decay pathway 명확화 가능.\n'
             'MCF7 STM2457 (GSE269294)은 cell line 일반화에 기여.',
             color='#F0F4FF', edge_color=C_STRONG, fontsize=7.5,
             text_color='#333', lw=1.2, fontweight='normal')

    # Legend
    for i, (color, label) in enumerate([
        (C_STRONG, '5/5 - 매우 강함'), ('#5FAD56', '4/5 - 강함'),
        (C_MEDIUM, '3/5 - 보통'), (C_WEAK, '2/5 - 약함'),
        (C_NEW, '진행중 (METTL3 KO)'), (C_COMING, '진행중 (XRN1 KD)')
    ]):
        ax.add_patch(mpatches.Rectangle((0.3, 0.4 - i * 0.2), 0.25, 0.14,
                     facecolor=color, edgecolor='#666', lw=0.4))
        ax.text(0.7, 0.47 - i * 0.2, label, fontsize=5.5, va='center', color='#333')

    pdf.savefig(fig)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════
    # PAGE 2: Data Strategy
    # ══════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(11, 15))
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 15)
    ax2.axis('off')

    ax2.text(5.5, 14.6, 'Page 2: 데이터 전략 (보강 우선순위)', fontsize=14,
             fontweight='bold', ha='center', color=C_TITLE)

    # ═══ TIER 1: IN PROGRESS ═══
    y_t1 = 13.7
    ax2.text(0.7, y_t1, 'Tier 1: 진행중 (최우선)', fontsize=11, fontweight='bold',
             color=C_NEW)
    ax2.plot([0.7, 10.3], [y_t1 - 0.15, y_t1 - 0.15], color=C_NEW, lw=1.5)

    # METTL3 KO
    draw_box(ax2, 3.0, y_t1 - 1.0, 4.3, 1.3,
             'METTL3 KO (PRJEB40872)\n'
             'HEK293T, WT 3rep + KO 3rep, RNA002\n'
             'Phase 1 진행중 (download + basecall)\n\n'
             '예상 결과:\n'
             '  m6A density KO < WT (0.91x)\n'
             '  poly(A) KO < WT? (인과 증거)',
             color='#FFF5EE', edge_color=C_NEW, fontsize=6.5,
             text_color='#333', lw=1.5)

    # What it addresses
    draw_box(ax2, 8.0, y_t1 - 0.55, 3.5, 0.55,
             'Part 1 보강: Writer 확인\n(5/5 -> 5+/5)',
             color='#D4EDDA', edge_color='#28A745', fontsize=7,
             text_color='#155724', lw=1.0)
    draw_arrow(ax2, 5.2, y_t1 - 0.55, 6.2, y_t1 - 0.55, color=C_NEW, lw=1.2)

    draw_box(ax2, 8.0, y_t1 - 1.25, 3.5, 0.55,
             'Part 3 보강: 인과 증거 제공\n(3/5 -> 4/5)',
             color='#FFF3CD', edge_color=C_MEDIUM, fontsize=7,
             text_color='#856404', lw=1.0)
    draw_arrow(ax2, 5.2, y_t1 - 1.25, 6.2, y_t1 - 1.25, color=C_NEW, lw=1.2)

    # XRN1 KD
    y_xrn = y_t1 - 2.8
    draw_box(ax2, 3.0, y_xrn, 4.3, 1.3,
             'XRN1 KD (PRJNA842344)\n'
             'HeLa, mock/XRN1 x unstressed/Ars, n=2\n'
             'MAFIA 3/8 완료, nanopolish 부분 완료\n\n'
             'Expression: INCONCLUSIVE (batch effect)\n'
             '남은 분석: m6A level + poly(A) (완료분)',
             color='#F0F4FF', edge_color=C_COMING, fontsize=6.5,
             text_color='#333', lw=1.5)

    draw_box(ax2, 8.0, y_xrn + 0.3, 3.5, 0.55,
             'Part 2 보강: Decay pathway\nXRN1이 partial -> 다경로 증거',
             color='#D4EDDA', edge_color='#5FAD56', fontsize=6.5,
             text_color='#155724', lw=1.0)
    draw_arrow(ax2, 5.2, y_xrn + 0.3, 6.2, y_xrn + 0.3, color=C_COMING, lw=1.2)

    draw_box(ax2, 8.0, y_xrn - 0.4, 3.5, 0.55,
             'Part 3 보강: XRN1 KD에서\nm6A 불변이면 -> m6A upstream',
             color='#FFF3CD', edge_color=C_MEDIUM, fontsize=6.5,
             text_color='#856404', lw=1.0)
    draw_arrow(ax2, 5.2, y_xrn - 0.4, 6.2, y_xrn - 0.4, color=C_COMING, lw=1.2)

    # ═══ TIER 2: AVAILABLE PUBLIC DATA ═══
    y_t2 = 8.8
    ax2.text(0.7, y_t2, 'Tier 2: 공개 데이터 (다운로드 가능)', fontsize=11,
             fontweight='bold', color='#5FAD56')
    ax2.plot([0.7, 10.3], [y_t2 - 0.15, y_t2 - 0.15], color='#5FAD56', lw=1.5)

    pub_data = [
        ('GSE269294\nMCF7 + STM2457\n(METTL3 inhibitor)\n5 reps, RNA002',
         '-> MCF7 = 우리 CL!\n-> 2nd cell line 검증\n-> Part 1,3 cross-CL',
         '#5FAD56', 'HIGH'),
        ('GSE230936 (NanoSPA)\nHEK293T METTL3 KD\n+ TRUB1 KD\nNat Biotech 2024',
         '-> m6A + psi 동시 KD\n-> Part 1 cross-validation\n-> psi writer 검증',
         '#5FAD56', 'HIGH'),
        ('PRJNA1135158\nMOLM13 METTL3 KD\n34M reads (deep)\nCell Genomics 2025',
         '-> 3rd cell line (AML)\n-> deep coverage\n-> Part 1 일반화',
         C_MEDIUM, 'MEDIUM'),
    ]

    for i, (dataset, impact, color, priority) in enumerate(pub_data):
        xi = 1.8 + i * 3.2
        draw_box(ax2, xi, y_t2 - 0.95, 2.8, 1.1, dataset,
                 color='#FFFFFF', edge_color=color, fontsize=6,
                 text_color='#333', lw=1.2)
        draw_box(ax2, xi, y_t2 - 1.85, 2.8, 0.7, impact,
                 color=color, edge_color=color, fontsize=5.5,
                 text_color='white', lw=0.5, alpha=0.85)
        ax2.text(xi + 1.2, y_t2 - 0.35, priority, fontsize=6, fontweight='bold',
                 color=color, ha='center')

    # ═══ STRATEGIC ANALYSIS ═══
    y_s = 5.8
    ax2.text(0.7, y_s, '전략 분석: 각 데이터의 논문 임팩트', fontsize=11,
             fontweight='bold', color=C_TITLE)
    ax2.plot([0.7, 10.3], [y_s - 0.15, y_s - 0.15], color=C_TITLE, lw=1.0)

    # Impact matrix
    col_headers = ['데이터', 'Part 1\nm6A Enrich', 'Part 2\nArsenite', 'Part 3\nm6A Protect', '노력', '우선순위']
    col_x = [1.5, 3.5, 5.0, 6.5, 8.0, 9.3]
    col_w = [2.3, 1.2, 1.2, 1.2, 1.0, 1.2]

    for j, (header, cx) in enumerate(zip(col_headers, col_x)):
        draw_box(ax2, cx, y_s - 0.5, col_w[j], 0.4, header,
                 color='#E8E8E8', edge_color='#999', fontsize=5.5,
                 fontweight='bold', text_color='#333', lw=0.5)

    rows = [
        ['METTL3 KO\n(PRJEB40872)',  'Writer\n확인',   '-',        'm6A 제거시\npoly(A) 감소?', '진행중',  '1순위'],
        ['XRN1 KD MAFIA\n(PRJNA842344)', '-',    'Decay\npathway', 'm6A level\n확인',       '진행중',  '1순위'],
        ['MCF7 STM2457\n(GSE269294)', '2nd CL\n검증', '-',        'MCF7에서\nm6A->poly(A)?', '2주',    '2순위'],
        ['NanoSPA KD\n(GSE230936)',   'm6A + psi\n동시 KD', '-',   'psi 대조군', '1주',    '2순위'],
        ['MOLM13 deep\n(PRJNA1135158)', '3rd CL',  '-',        '-',         '2주',    '3순위'],
    ]

    row_colors = [C_NEW, C_COMING, '#5FAD56', '#5FAD56', C_MEDIUM]

    for ri, (row, rc) in enumerate(zip(rows, row_colors)):
        y_row = y_s - 1.05 - ri * 0.55
        for j, (cell, cx) in enumerate(zip(row, col_x)):
            bg = '#FFFFFF' if j > 0 else '#F8F9FA'
            ec = '#DDD' if j > 0 else rc
            fw = 'bold' if j == 0 or j == 5 else 'normal'
            tc = rc if j == 5 else '#333'
            draw_box(ax2, cx, y_row, col_w[j], 0.48, cell,
                     color=bg, edge_color=ec, fontsize=5,
                     fontweight=fw, text_color=tc, lw=0.4)

    # ═══ BEST-CASE SCENARIO ═══
    y_best = 2.4
    ax2.text(0.7, y_best + 0.5, 'Best-case 시나리오 (모든 데이터 추가시)', fontsize=11,
             fontweight='bold', color=C_GREEN)
    ax2.plot([0.7, 10.3], [y_best + 0.35, y_best + 0.35], color=C_GREEN, lw=1.0)

    scenarios = [
        ('METTL3 KO: m6A 50% 감소\n+ poly(A) 15nt 감소',
         'Part 1: 5+ (writer 확인)\nPart 3: 4/5 (인과 근거)\n-> "m6A-dependent protection"',
         '#28A745'),
        ('METTL3 KO: m6A 변화 없음\n또는 poly(A) 불변',
         'Part 3 약화 -> 논문 재구성\n"L1 m6A enrichment 자체"\n-> 최소한 Part1은 탄탄',
         C_MEDIUM),
        ('XRN1 KD: m6A 불변\n+ poly(A) 약간 연장',
         'Part 2: decay pathway 제시\n"m6A upstream, XRN1 partial"\n-> 메커니즘 명확화',
         '#5FAD56'),
    ]

    for i, (condition, outcome, color) in enumerate(scenarios):
        xi = 2.0 + i * 3.0
        draw_box(ax2, xi, y_best - 0.3, 2.6, 0.55, condition,
                 color='#FFFFFF', edge_color=color, fontsize=5.5,
                 text_color='#333', lw=1.0)
        draw_box(ax2, xi, y_best - 1.05, 2.6, 0.7, outcome,
                 color=color, edge_color=color, fontsize=5.5,
                 text_color='white', lw=0.5, alpha=0.85)
        draw_arrow(ax2, xi, y_best - 0.6, xi, y_best - 0.68, color=color, lw=1.0)

    # Bottom summary
    draw_box(ax2, 5.5, 0.5, 9.5, 0.7,
             '핵심 전략: METTL3 KO 완료가 최우선. m6A 감소시 Part 1+3 동시 보강.\n'
             '병렬로 XRN1 MAFIA 분석 + MCF7 STM2457 다운로드 진행.\n'
             'METTL3 KO 결과에 따라 논문 framing 조정 (protection vs association).',
             color='#E8F4FD', edge_color=C_STRONG, fontsize=7,
             text_color='#333', lw=1.2)

    pdf.savefig(fig2)
    plt.close(fig2)

print("Mind map v2 (2 pages) saved as mindmap.pdf")
