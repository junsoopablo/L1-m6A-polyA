#!/usr/bin/env python3
"""Part 3: L1 RNA Modification Landscape - Analysis & Figure Generation

Analyses:
  1. L1 vs Control modification density (psi/kb, m6A/kb) - within-sample comparison
  2. Young vs Ancient L1 + cross-cell-line variation
  3. Positional distribution along transcript body
  4. m6A-psi co-occurrence (L1 vs Control, within-sample)
  5. Per-locus modification consistency + intronic vs intergenic
  6. Motif enrichment (L1 vs Control)
"""

import pandas as pd
import numpy as np
import pysam
import ast
from pathlib import Path
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================
PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC_03 = PROJECT / 'analysis/01_exploration/topic_03_m6a_psi'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
RESULTS = PROJECT / 'results_group'
OUTDIR = TOPIC_05 / 'pdf_figures_part3'
OUTDIR.mkdir(exist_ok=True)

PROB_THRESHOLD = 204  # 80% on 0-255 scale (changed from 128/50%)
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

CELL_LINES = {
    'A549': ['A549_4','A549_5','A549_6'],
    'H9': ['H9_2','H9_3','H9_4'],
    'Hct116': ['Hct116_3','Hct116_4'],
    'HeLa': ['HeLa_1','HeLa_2','HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1','HeLa-Ars_2','HeLa-Ars_3'],
    'HepG2': ['HepG2_5','HepG2_6'],
    'HEYA8': ['HEYA8_1','HEYA8_2','HEYA8_3'],
    'K562': ['K562_4','K562_5','K562_6'],
    'MCF7': ['MCF7_2','MCF7_3','MCF7_4'],
    'MCF7-EV': ['MCF7-EV_1'],
    'SHSY5Y': ['SHSY5Y_1','SHSY5Y_2','SHSY5Y_3'],
}

CL_ORDER = ['A549','H9','Hct116','HeLa','HeLa-Ars','HepG2','HEYA8','K562','MCF7','MCF7-EV','SHSY5Y']
CL_COLORS = {
    'A549':'#1f77b4','H9':'#ff7f0e','Hct116':'#2ca02c','HeLa':'#d62728',
    'HeLa-Ars':'#9467bd','HepG2':'#8c564b','HEYA8':'#e377c2','K562':'#7f7f7f',
    'MCF7':'#bcbd22','MCF7-EV':'#17becf','SHSY5Y':'#aec7e8',
}

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 200})

# =============================================================================
# Helper functions
# =============================================================================
def parse_mm_ml_tags(mm_tag, ml_tag):
    result = {'m6A': [], 'psi': []}
    if mm_tag is None or ml_tag is None:
        return result
    mod_blocks = mm_tag.rstrip(';').split(';')
    ml_idx = 0
    for block in mod_blocks:
        if not block:
            continue
        parts = block.split(',')
        mod_type = parts[0]
        # Match BOTH strands: N+code (fwd-mapped) and N-code (rev-mapped)
        # DRS reads from reverse-strand genes get N- tags
        if '17802' in mod_type or 'U+p' in mod_type or 'T+p' in mod_type:
            mod_key = 'psi'  # ChEBI:17802 = pseudouridine
        elif '21891' in mod_type or 'A+a' in mod_type or 'A+m' in mod_type:
            mod_key = 'm6A'  # ChEBI:21891 = N6-methyladenosine
        else:
            ml_idx += len(parts) - 1
            continue
        current_pos = 0
        for pos_str in parts[1:]:
            if pos_str:
                current_pos += int(pos_str)
                if ml_idx < len(ml_tag):
                    result[mod_key].append((current_pos, ml_tag[ml_idx]))
                ml_idx += 1
    return result


def parse_bam_per_read(bam_path):
    records = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam:
            mm = read.get_tag("MM") if read.has_tag("MM") else None
            ml = read.get_tag("ML") if read.has_tag("ML") else None
            mods = parse_mm_ml_tags(mm, ml)
            rl = read.query_length or read.infer_query_length() or 0
            if rl < 50:
                continue
            m6a_high = [(p, prob) for p, prob in mods['m6A'] if prob >= PROB_THRESHOLD]
            psi_high = [(p, prob) for p, prob in mods['psi'] if prob >= PROB_THRESHOLD]
            records.append({
                'read_id': read.query_name,
                'read_length': rl,
                'm6a_sites_high': len(m6a_high),
                'psi_sites_high': len(psi_high),
                'm6a_positions': str([p for p, _ in m6a_high]),
                'psi_positions': str([p for p, _ in psi_high]),
            })
    return pd.DataFrame(records)


def safe_parse_list(val):
    if pd.isna(val) or val == '[]' or val == '':
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except:
        return []


# =============================================================================
# Load L1 per-read data (re-parse BAMs with corrected N+/N- parser)
# =============================================================================
print("=" * 60)
print("Loading L1 per-read data (re-parsing BAMs for N+/N- fix)...")
l1_cache = TOPIC_05 / 'part3_l1_per_read_cache'
l1_cache.mkdir(exist_ok=True)

l1_reads = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        # L1 summary for metadata + PASS filter
        sum_path = RESULTS / g / 'g_summary' / f'{g}_L1_summary.tsv'
        if not sum_path.exists():
            continue
        sm = pd.read_csv(sum_path, sep='\t')
        sm = sm[sm['qc_tag'] == 'PASS'].copy()
        if len(sm) == 0:
            continue

        # Parse L1 MAFIA BAM (cached)
        cache_path = l1_cache / f'{g}_l1_per_read.tsv'
        if cache_path.exists():
            pr = pd.read_csv(cache_path, sep='\t')
        else:
            bam_path = RESULTS / g / 'h_mafia' / f'{g}.mAFiA.reads.bam'
            if not bam_path.exists():
                print(f"  SKIP: {bam_path.name}")
                continue
            print(f"  Parsing {g} L1 BAM...")
            pr = parse_bam_per_read(bam_path)
            if len(pr) > 0:
                pr.to_csv(cache_path, sep='\t', index=False)

        if len(pr) == 0:
            continue

        # Merge with L1 summary (PASS reads only)
        merged = pr.merge(
            sm[['read_id', 'gene_id', 'transcript_id', 'read_length',
                'TE_group', 'overlapping_genes']],
            on='read_id', how='inner', suffixes=('_bam', '')
        )
        merged['group'] = g
        merged['cell_line'] = cl
        merged['l1_age'] = merged['gene_id'].apply(
            lambda x: 'young' if x in YOUNG else 'ancient'
        )
        l1_reads.append(merged)

l1 = pd.concat(l1_reads, ignore_index=True)
if 'read_length_bam' in l1.columns:
    l1['read_length'] = l1['read_length'].fillna(l1['read_length_bam'])
    l1.drop(columns=['read_length_bam'], inplace=True, errors='ignore')

l1['m6a_per_kb'] = l1['m6a_sites_high'] / (l1['read_length'] / 1000)
l1['psi_per_kb'] = l1['psi_sites_high'] / (l1['read_length'] / 1000)
print(f"  {len(l1):,} L1 reads, {l1['group'].nunique()} groups, {l1['cell_line'].nunique()} CL")

# =============================================================================
# Load / parse Control per-read data
# =============================================================================
print("\nLoading Control per-read data...")
ctrl_cache = TOPIC_05 / 'part3_ctrl_per_read_cache'
ctrl_cache.mkdir(exist_ok=True)

ctrl_reads = []
for cl, groups in CELL_LINES.items():
    if cl == 'MCF7-EV':
        continue
    for g in groups:
        cache_path = ctrl_cache / f'{g}_ctrl_per_read.tsv'
        if cache_path.exists():
            df = pd.read_csv(cache_path, sep='\t')
        else:
            bam_path = RESULTS / g / 'i_control' / 'mafia' / f'{g}.control.mAFiA.reads.bam'
            if not bam_path.exists():
                print(f"  SKIP: {bam_path.name}")
                continue
            print(f"  Parsing {g} control BAM...")
            df = parse_bam_per_read(bam_path)
            if len(df) > 0:
                df.to_csv(cache_path, sep='\t', index=False)
        if len(df) == 0:
            continue
        # Filter to PASS reads using control summary
        ctrl_sum_path = RESULTS / g / 'i_control' / f'{g}_control_summary.tsv'
        if ctrl_sum_path.exists():
            cs = pd.read_csv(ctrl_sum_path, sep='\t')
            pass_ids = set(cs[cs['qc_tag'] == 'PASS']['read_id'])
            df = df[df['read_id'].isin(pass_ids)].copy()
        df['group'] = g
        df['cell_line'] = cl
        ctrl_reads.append(df)

ctrl = pd.concat(ctrl_reads, ignore_index=True)
ctrl['m6a_per_kb'] = ctrl['m6a_sites_high'] / (ctrl['read_length'] / 1000)
ctrl['psi_per_kb'] = ctrl['psi_sites_high'] / (ctrl['read_length'] / 1000)
print(f"  {len(ctrl):,} Control reads, {ctrl['group'].nunique()} groups")

# =============================================================================
# Analysis 1: L1 vs Control Modification Density (within-sample)
# =============================================================================
print("\n" + "=" * 60)
print("Analysis 1: L1 vs Control density...")

density_rows = []
for g in l1['group'].unique():
    l1g = l1[l1['group'] == g]
    cl = l1g['cell_line'].iloc[0]
    cg = ctrl[ctrl['group'] == g] if g in ctrl['group'].values else pd.DataFrame()
    if len(cg) < 20:
        continue
    density_rows.append({
        'group': g, 'cell_line': cl,
        # Fraction with any modification (binary rate)
        'l1_psi_frac': (l1g['psi_sites_high'] > 0).mean(),
        'ctrl_psi_frac': (cg['psi_sites_high'] > 0).mean(),
        'l1_m6a_frac': (l1g['m6a_sites_high'] > 0).mean(),
        'ctrl_m6a_frac': (cg['m6a_sites_high'] > 0).mean(),
        # Mean sites/kb (captures density among all reads)
        'l1_psi_per_kb': l1g['psi_per_kb'].mean(),
        'ctrl_psi_per_kb': cg['psi_per_kb'].mean(),
        'l1_m6a_per_kb': l1g['m6a_per_kb'].mean(),
        'ctrl_m6a_per_kb': cg['m6a_per_kb'].mean(),
        'l1_n': len(l1g), 'ctrl_n': len(cg),
    })

dens = pd.DataFrame(density_rows)
dens['delta_psi_frac'] = dens['l1_psi_frac'] - dens['ctrl_psi_frac']
dens['delta_m6a_frac'] = dens['l1_m6a_frac'] - dens['ctrl_m6a_frac']
dens['delta_psi_kb'] = dens['l1_psi_per_kb'] - dens['ctrl_psi_per_kb']
dens['delta_m6a_kb'] = dens['l1_m6a_per_kb'] - dens['ctrl_m6a_per_kb']

# Wilcoxon signed-rank test (paired by group) on fraction
def safe_wilcoxon(x, y):
    try:
        return stats.wilcoxon(x, y)
    except ValueError:
        return type('obj', (object,), {'pvalue': 1.0, 'statistic': 0})()

wsr_psi_frac = safe_wilcoxon(dens['l1_psi_frac'], dens['ctrl_psi_frac'])
wsr_m6a_frac = safe_wilcoxon(dens['l1_m6a_frac'], dens['ctrl_m6a_frac'])
wsr_psi_kb = safe_wilcoxon(dens['l1_psi_per_kb'], dens['ctrl_psi_per_kb'])
wsr_m6a_kb = safe_wilcoxon(dens['l1_m6a_per_kb'], dens['ctrl_m6a_per_kb'])

print(f"  psi fraction: L1={dens['l1_psi_frac'].mean():.3f}, "
      f"Ctrl={dens['ctrl_psi_frac'].mean():.3f}, "
      f"Wilcoxon p={wsr_psi_frac.pvalue:.2e}")
print(f"  m6A fraction: L1={dens['l1_m6a_frac'].mean():.3f}, "
      f"Ctrl={dens['ctrl_m6a_frac'].mean():.3f}, "
      f"Wilcoxon p={wsr_m6a_frac.pvalue:.2e}")
print(f"  psi/kb mean: L1={dens['l1_psi_per_kb'].mean():.2f}, "
      f"Ctrl={dens['ctrl_psi_per_kb'].mean():.2f}, "
      f"Wilcoxon p={wsr_psi_kb.pvalue:.2e}")
print(f"  m6A/kb mean: L1={dens['l1_m6a_per_kb'].mean():.2f}, "
      f"Ctrl={dens['ctrl_m6a_per_kb'].mean():.2f}, "
      f"Wilcoxon p={wsr_m6a_kb.pvalue:.2e}")
dens.to_csv(OUTDIR / 'part3_l1_vs_ctrl_density.tsv', sep='\t', index=False)

# =============================================================================
# Analysis 2: Young vs Ancient + cross-CL
# =============================================================================
print("\nAnalysis 2: Age stratification + cross-CL...")

age_rows = []
for cl in CL_ORDER:
    for age in ['young', 'ancient']:
        sub = l1[(l1['cell_line'] == cl) & (l1['l1_age'] == age)]
        if len(sub) < 10:
            continue
        age_rows.append({
            'cell_line': cl, 'l1_age': age, 'n': len(sub),
            'psi_per_kb_median': sub['psi_per_kb'].median(),
            'psi_per_kb_mean': sub['psi_per_kb'].mean(),
            'm6a_per_kb_median': sub['m6a_per_kb'].median(),
            'm6a_per_kb_mean': sub['m6a_per_kb'].mean(),
        })

age_df = pd.DataFrame(age_rows)
# Overall young vs ancient
yng = l1[l1['l1_age'] == 'young']
anc = l1[l1['l1_age'] == 'ancient']
mw_psi_age = stats.mannwhitneyu(yng['psi_per_kb'], anc['psi_per_kb'])
print(f"  Young psi/kb median={yng['psi_per_kb'].median():.2f} (n={len(yng):,})")
print(f"  Ancient psi/kb median={anc['psi_per_kb'].median():.2f} (n={len(anc):,})")
print(f"  MW p={mw_psi_age.pvalue:.2e}")
age_df.to_csv(OUTDIR / 'part3_age_density.tsv', sep='\t', index=False)

# =============================================================================
# Analysis 3: Positional distribution
# =============================================================================
print("\nAnalysis 3: Positional distribution...")

def collect_positions(df, label):
    """Collect fractional positions for all high-conf sites."""
    rows = []
    for _, r in df.iterrows():
        rl = r['read_length']
        if rl < 100:
            continue
        for mod in ['psi', 'm6a']:
            positions = safe_parse_list(r.get(f'{mod}_positions', []))
            for pos in positions:
                frac = pos / rl
                if 0 <= frac <= 1:
                    rows.append({'frac_pos': frac, 'mod': mod, 'source': label})
    return rows

# Sample for efficiency (max 5000 reads per source)
np.random.seed(42)
l1_sample = l1.sample(min(5000, len(l1)))
ctrl_sample = ctrl.sample(min(5000, len(ctrl)))

pos_data = collect_positions(l1_sample, 'L1') + collect_positions(ctrl_sample, 'Control')
pos_df = pd.DataFrame(pos_data)
print(f"  L1 position entries: {(pos_df['source']=='L1').sum():,}")
print(f"  Ctrl position entries: {(pos_df['source']=='Control').sum():,}")

# KS test for uniformity
for mod in ['psi', 'm6a']:
    l1_pos = pos_df[(pos_df['source'] == 'L1') & (pos_df['mod'] == mod)]['frac_pos']
    ctrl_pos = pos_df[(pos_df['source'] == 'Control') & (pos_df['mod'] == mod)]['frac_pos']
    if len(l1_pos) > 10:
        ks_l1 = stats.kstest(l1_pos, 'uniform')
        print(f"  L1 {mod}: KS vs uniform p={ks_l1.pvalue:.2e}, mean_pos={l1_pos.mean():.3f}")
    if len(ctrl_pos) > 10:
        ks_ctrl = stats.kstest(ctrl_pos, 'uniform')
        print(f"  Ctrl {mod}: KS vs uniform p={ks_ctrl.pvalue:.2e}, mean_pos={ctrl_pos.mean():.3f}")

# =============================================================================
# Analysis 4: m6A-psi Co-occurrence (within-sample, L1 vs Control)
# =============================================================================
print("\nAnalysis 4: m6A-psi co-occurrence...")

def compute_cooccurrence(df):
    """Compute m6A-psi co-occurrence stats."""
    has_m6a = (df['m6a_sites_high'] > 0).astype(int)
    has_psi = (df['psi_sites_high'] > 0).astype(int)
    # 2x2 table: [m6a-/psi-, m6a-/psi+, m6a+/psi-, m6a+/psi+]
    a = ((has_m6a == 0) & (has_psi == 0)).sum()  # neither
    b = ((has_m6a == 0) & (has_psi == 1)).sum()  # psi only
    c = ((has_m6a == 1) & (has_psi == 0)).sum()  # m6a only
    d = ((has_m6a == 1) & (has_psi == 1)).sum()  # both
    n = len(df)
    # Odds ratio
    if a > 0 and b > 0 and c > 0 and d > 0:
        OR = (a * d) / (b * c)
        log_se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    else:
        OR = np.nan
        log_se = np.nan
    cooc_rate = d / n if n > 0 else 0
    m6a_rate = (c + d) / n if n > 0 else 0
    psi_rate = (b + d) / n if n > 0 else 0
    expected_cooc = m6a_rate * psi_rate  # under independence
    return {
        'n': n, 'neither': a, 'psi_only': b, 'm6a_only': c, 'both': d,
        'OR': OR, 'log_OR_se': log_se,
        'cooc_rate': cooc_rate, 'expected_cooc': expected_cooc,
        'm6a_rate': m6a_rate, 'psi_rate': psi_rate,
    }

cooc_rows = []
for g in sorted(l1['group'].unique()):
    cl = l1[l1['group'] == g]['cell_line'].iloc[0]
    # L1
    l1g = l1[l1['group'] == g]
    res_l1 = compute_cooccurrence(l1g)
    res_l1.update({'group': g, 'cell_line': cl, 'source': 'L1'})
    cooc_rows.append(res_l1)
    # Control
    if g in ctrl['group'].values:
        cg = ctrl[ctrl['group'] == g]
        res_ctrl = compute_cooccurrence(cg)
        res_ctrl.update({'group': g, 'cell_line': cl, 'source': 'Control'})
        cooc_rows.append(res_ctrl)

cooc_df = pd.DataFrame(cooc_rows)
cooc_df.to_csv(OUTDIR / 'part3_cooccurrence.tsv', sep='\t', index=False)

# Summary
l1_cooc = cooc_df[cooc_df['source'] == 'L1']
ctrl_cooc = cooc_df[cooc_df['source'] == 'Control']
print(f"  L1 co-occurrence rate: {l1_cooc['cooc_rate'].median():.3f} "
      f"(expected: {l1_cooc['expected_cooc'].median():.3f})")
print(f"  Ctrl co-occurrence rate: {ctrl_cooc['cooc_rate'].median():.3f} "
      f"(expected: {ctrl_cooc['expected_cooc'].median():.3f})")
print(f"  L1 OR median: {l1_cooc['OR'].median():.2f}")
print(f"  Ctrl OR median: {ctrl_cooc['OR'].median():.2f}")

# Paired test of OR (L1 vs Control, same group)
paired = l1_cooc[['group','OR']].merge(
    ctrl_cooc[['group','OR']], on='group', suffixes=('_l1','_ctrl')
).dropna()
if len(paired) > 5:
    wsr_or = stats.wilcoxon(paired['OR_l1'], paired['OR_ctrl'])
    print(f"  Paired Wilcoxon OR: p={wsr_or.pvalue:.3e}")

# =============================================================================
# Analysis 5: Per-locus modification consistency + intronic vs intergenic
# =============================================================================
print("\nAnalysis 5: Locus consistency + genomic context...")

# 5A: Per-locus psi modification fraction across replicates
locus_data = []
for cl, groups in CELL_LINES.items():
    if len(groups) < 2:
        continue
    for g in groups:
        sub = l1[(l1['group'] == g)]
        for _, r in sub.iterrows():
            locus_data.append({
                'locus': r['transcript_id'],
                'cell_line': cl,
                'group': g,
                'has_psi': int(r['psi_sites_high'] > 0),
                'psi_per_kb': r['psi_per_kb'],
            })

locus_df = pd.DataFrame(locus_data)
# Per-locus: fraction of reads with psi, per cell line
locus_summary = locus_df.groupby(['locus', 'cell_line']).agg(
    n_reads=('has_psi', 'count'),
    psi_frac=('has_psi', 'mean'),
    n_groups=('group', 'nunique'),
).reset_index()

# Loci with reads in >= 2 replicates
multi_rep = locus_summary[locus_summary['n_groups'] >= 2].copy()
# Per-locus consistency: compute per-replicate psi_frac and get CV
locus_by_rep = locus_df.groupby(['locus', 'cell_line', 'group']).agg(
    rep_psi_frac=('has_psi', 'mean'),
    rep_n=('has_psi', 'count'),
).reset_index()
# Only loci with >=2 reads in >=2 replicates
rep_counts = locus_by_rep[locus_by_rep['rep_n'] >= 2].groupby(
    ['locus', 'cell_line']).size().reset_index(name='n_reps_qual')
qual_loci = rep_counts[rep_counts['n_reps_qual'] >= 2]

consistency_rows = []
for _, row in qual_loci.iterrows():
    sub = locus_by_rep[(locus_by_rep['locus'] == row['locus']) &
                       (locus_by_rep['cell_line'] == row['cell_line']) &
                       (locus_by_rep['rep_n'] >= 2)]
    fracs = sub['rep_psi_frac'].values
    if len(fracs) >= 2:
        consistency_rows.append({
            'locus': row['locus'],
            'cell_line': row['cell_line'],
            'n_reps': len(fracs),
            'mean_psi_frac': fracs.mean(),
            'std_psi_frac': fracs.std(),
            'range_psi_frac': fracs.max() - fracs.min(),
        })

consist_df = pd.DataFrame(consistency_rows)
print(f"  Loci with >=2 reads in >=2 replicates: {len(consist_df)}")
if len(consist_df) > 0:
    print(f"  Mean psi_frac: {consist_df['mean_psi_frac'].mean():.3f}")
    print(f"  Mean range across reps: {consist_df['range_psi_frac'].mean():.3f}")
consist_df.to_csv(OUTDIR / 'part3_locus_consistency.tsv', sep='\t', index=False)

# 5B: Intronic vs Intergenic modification
context_rows = []
for ctx in ['intronic', 'intergenic']:
    sub = l1[l1['TE_group'] == ctx]
    context_rows.append({
        'context': ctx, 'n': len(sub),
        'psi_per_kb_median': sub['psi_per_kb'].median(),
        'psi_per_kb_mean': sub['psi_per_kb'].mean(),
        'm6a_per_kb_median': sub['m6a_per_kb'].median(),
        'm6a_per_kb_mean': sub['m6a_per_kb'].mean(),
        'read_length_median': sub['read_length'].median(),
    })

context_df = pd.DataFrame(context_rows)
mw_ctx_psi = stats.mannwhitneyu(
    l1[l1['TE_group'] == 'intronic']['psi_per_kb'],
    l1[l1['TE_group'] == 'intergenic']['psi_per_kb']
)
print(f"  Intronic psi/kb: {context_df[context_df['context']=='intronic']['psi_per_kb_median'].iloc[0]:.2f}")
print(f"  Intergenic psi/kb: {context_df[context_df['context']=='intergenic']['psi_per_kb_median'].iloc[0]:.2f}")
print(f"  MW p={mw_ctx_psi.pvalue:.2e}")
context_df.to_csv(OUTDIR / 'part3_context_modification.tsv', sep='\t', index=False)

# =============================================================================
# Analysis 6: Motif enrichment (L1 vs Control, using pileup data)
# =============================================================================
print("\nAnalysis 6: Motif enrichment...")

def load_pileup(bed_path):
    if not bed_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(bed_path, sep='\t')
    # Ensure numeric columns
    for col in ['coverage', 'modRatio', 'confidence']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Aggregate pileup across all groups (excluding treatment conditions for clean baseline)
BASE_CL = ['A549', 'H9', 'Hct116', 'HeLa', 'HepG2', 'HEYA8', 'K562', 'MCF7', 'SHSY5Y']
l1_pileups, ctrl_pileups = [], []

for cl in BASE_CL:
    for g in CELL_LINES[cl]:
        # L1 pileup
        l1_bed = RESULTS / g / 'h_mafia' / f'{g}.mAFiA.sites.bed'
        df = load_pileup(l1_bed)
        if len(df) > 0:
            df['group'] = g
            df['cell_line'] = cl
            l1_pileups.append(df)
        # Control pileup
        ctrl_bed = RESULTS / g / 'i_control' / 'mafia' / f'{g}.control.mAFiA.sites.bed'
        df = load_pileup(ctrl_bed)
        if len(df) > 0:
            df['group'] = g
            df['cell_line'] = cl
            ctrl_pileups.append(df)

l1_pile = pd.concat(l1_pileups, ignore_index=True) if l1_pileups else pd.DataFrame()
ctrl_pile = pd.concat(ctrl_pileups, ignore_index=True) if ctrl_pileups else pd.DataFrame()
print(f"  L1 pileup sites: {len(l1_pile):,}")
print(f"  Ctrl pileup sites: {len(ctrl_pile):,}")

# Define ALL canonical motifs (not just top-N by frequency)
ALL_DRACH = set()
for d in 'AGT':
    for r in 'AG':
        for h in 'ACT':
            ALL_DRACH.add(d + r + 'A' + 'C' + h)
ALL_PSI = {'GTTCA','GTTCC','GTTCG','GTTCT','AGTGG','GGTGG','TGTGG',
           'TGTAG','GGTCC','CATAA','TATAA','CATCC','CTTTA','ATTTG','GATGC','CCTCC'}
CANONICAL_MOTIFS = {'m6A': ALL_DRACH, 'psi': ALL_PSI}

# Per-motif modification rate (coverage-weighted mean modRatio)
motif_rows = []
for mod_type in ['m6A', 'psi']:
    for source, pile in [('L1', l1_pile), ('Control', ctrl_pile)]:
        if len(pile) == 0:
            continue
        sub = pile[(pile['name'] == mod_type) & (pile['coverage'] >= 5)]
        if len(sub) == 0:
            continue
        # Use ALL canonical motifs, not just top-N
        for motif in sorted(CANONICAL_MOTIFS[mod_type]):
            m = sub[sub['ref5mer'] == motif]
            if len(m) < 5:
                continue
            motif_rows.append({
                'mod_type': mod_type,
                'source': source,
                'motif': motif,
                'n_sites': len(m),
                'mean_modRatio': np.average(m['modRatio'], weights=m['coverage']),
                'median_modRatio': m['modRatio'].median(),
                'mean_coverage': m['coverage'].mean(),
            })

motif_df = pd.DataFrame(motif_rows)
motif_df.to_csv(OUTDIR / 'part3_motif_enrichment.tsv', sep='\t', index=False)

# Print top m6A motifs
for mod_type in ['m6A', 'psi']:
    print(f"\n  Top {mod_type} motifs (L1 vs Control):")
    l1m = motif_df[(motif_df['mod_type'] == mod_type) & (motif_df['source'] == 'L1')].sort_values(
        'mean_modRatio', ascending=False).head(5)
    for _, r in l1m.iterrows():
        ctrl_r = motif_df[(motif_df['mod_type'] == mod_type) &
                          (motif_df['source'] == 'Control') &
                          (motif_df['motif'] == r['motif'])]
        ctrl_val = ctrl_r['mean_modRatio'].iloc[0] if len(ctrl_r) > 0 else float('nan')
        print(f"    {r['motif']}: L1={r['mean_modRatio']:.1f}%, Ctrl={ctrl_val:.1f}%, "
              f"n_sites={r['n_sites']}")


# =============================================================================
# FIGURE GENERATION
# =============================================================================
print("\n" + "=" * 60)
print("Generating figures...")

# --- Figure 1: L1 vs Control Modification Density ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# 1A: psi fraction (binary: any high-conf site)
ax = axes[0]
dens_base = dens[~dens['cell_line'].isin(['HeLa-Ars', 'MCF7-EV'])].copy()
dens_plot = dens_base.sort_values('cell_line')
x = np.arange(len(dens_plot))
w = 0.35
ax.bar(x - w/2, dens_plot['l1_psi_frac'] * 100, w, color='#d62728', alpha=0.8, label='L1')
ax.bar(x + w/2, dens_plot['ctrl_psi_frac'] * 100, w, color='#1f77b4', alpha=0.8, label='Control')
ax.set_xticks(x)
ax.set_xticklabels(dens_plot['group'], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Reads with psi (%)')
ax.set_title('A. Pseudouridine detection rate')
ax.legend(fontsize=8)
ax.text(0.02, 0.95, f'Wilcoxon p={wsr_psi_frac.pvalue:.1e}',
        transform=ax.transAxes, fontsize=8, va='top')

# 1B: psi sites/kb (mean, captures density)
ax = axes[1]
ax.bar(x - w/2, dens_plot['l1_psi_per_kb'], w, color='#d62728', alpha=0.8, label='L1')
ax.bar(x + w/2, dens_plot['ctrl_psi_per_kb'], w, color='#1f77b4', alpha=0.8, label='Control')
ax.set_xticks(x)
ax.set_xticklabels(dens_plot['group'], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Mean psi sites/kb')
ax.set_title('B. Pseudouridine density (sites/kb)')
ax.legend(fontsize=8)
ax.text(0.02, 0.95, f'Wilcoxon p={wsr_psi_kb.pvalue:.1e}',
        transform=ax.transAxes, fontsize=8, va='top')

plt.tight_layout()
fig.savefig(OUTDIR / 'fig1_l1_vs_ctrl_density.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Figure 1 saved")

# --- Figure 2: Young vs Ancient + Cross-CL heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# 2A: Young vs Ancient boxplot
ax = axes[0]
plot_data = l1[l1['cell_line'].isin(BASE_CL)].copy()
bp_data = [plot_data[plot_data['l1_age'] == 'young']['psi_per_kb'],
           plot_data[plot_data['l1_age'] == 'ancient']['psi_per_kb']]
bp = ax.boxplot(bp_data, labels=['Young L1', 'Ancient L1'], widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops={'color': 'black', 'linewidth': 1.5})
bp['boxes'][0].set_facecolor('#ff7f0e')
bp['boxes'][1].set_facecolor('#1f77b4')
ax.set_ylabel('Pseudouridine sites/kb')
ax.set_title('A. Young vs Ancient L1')
delta_psi = bp_data[0].median() - bp_data[1].median()
ax.text(0.02, 0.95, f'Young median={bp_data[0].median():.2f}\n'
        f'Ancient median={bp_data[1].median():.2f}\n'
        f'delta={delta_psi:+.2f} sites/kb',
        transform=ax.transAxes, fontsize=8, va='top')

# 2B: Cross-CL heatmap of delta(L1-Control) fraction
ax = axes[1]
cl_delta = dens.groupby('cell_line').agg(
    delta_psi_frac=('delta_psi_frac', 'mean'),
    delta_m6a_frac=('delta_m6a_frac', 'mean'),
).reindex([c for c in CL_ORDER if c not in ['HeLa-Ars', 'MCF7-EV']])

x2 = np.arange(len(cl_delta))
ax.bar(x2 - 0.2, cl_delta['delta_psi_frac'] * 100, 0.35, color='#9467bd',
       alpha=0.8, label='psi')
ax.bar(x2 + 0.2, cl_delta['delta_m6a_frac'] * 100, 0.35, color='#2ca02c',
       alpha=0.8, label='m6A')
ax.set_xticks(x2)
ax.set_xticklabels(cl_delta.index, rotation=45, ha='right', fontsize=8)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.set_ylabel('Delta (L1 - Control) fraction (%)')
ax.set_title('B. L1 excess modification by cell line')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUTDIR / 'fig2_age_crosscl.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Figure 2 saved")

# --- Figure 3: Positional distribution ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for i, mod in enumerate(['psi', 'm6a']):
    ax = axes[i]
    for src, color in [('L1', '#d62728'), ('Control', '#1f77b4')]:
        vals = pos_df[(pos_df['mod'] == mod) & (pos_df['source'] == src)]['frac_pos']
        if len(vals) > 10:
            ax.hist(vals, bins=50, density=True, alpha=0.5, color=color, label=f'{src} (n={len(vals):,})')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5, label='Uniform')
    ax.set_xlabel('Fractional position (0=3\' end, 1=5\' end)')
    ax.set_ylabel('Density')
    mod_label = 'Pseudouridine' if mod == 'psi' else 'm6A'
    ax.set_title(f'{"A" if i==0 else "B"}. {mod_label} position along read')
    ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(OUTDIR / 'fig3_positional.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Figure 3 saved")

# --- Figure 4: m6A-psi Co-occurrence ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# 4A: Co-occurrence rate (observed vs expected) for L1 and Control
ax = axes[0]
for src, color, offset in [('L1', '#d62728', -0.15), ('Control', '#1f77b4', 0.15)]:
    sub = cooc_df[cooc_df['source'] == src].copy()
    sub = sub[~sub['cell_line'].isin(['HeLa-Ars', 'MCF7-EV'])]
    ax.scatter(sub['expected_cooc'], sub['cooc_rate'], c=color, alpha=0.7,
               s=40, label=src, edgecolors='white', linewidth=0.5)
# Diagonal
max_val = max(cooc_df['expected_cooc'].max(), cooc_df['cooc_rate'].max()) * 1.1
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Expected co-occurrence (independent)')
ax.set_ylabel('Observed co-occurrence rate')
ax.set_title('A. m6A-psi co-occurrence')
ax.legend(fontsize=8)

# 4B: Paired OR comparison (L1 vs Control)
ax = axes[1]
if len(paired) > 0:
    x3 = np.arange(len(paired))
    ax.bar(x3 - 0.2, paired['OR_l1'], 0.35, color='#d62728', alpha=0.8, label='L1')
    ax.bar(x3 + 0.2, paired['OR_ctrl'], 0.35, color='#1f77b4', alpha=0.8, label='Control')
    ax.set_xticks(x3)
    ax.set_xticklabels(paired['group'], rotation=45, ha='right', fontsize=7)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Odds Ratio (m6A-psi co-occurrence)')
    ax.set_title('B. Within-sample odds ratio')
    ax.legend(fontsize=8)
    if wsr_or.pvalue < 0.05:
        ax.text(0.02, 0.95, f'Paired Wilcoxon p={wsr_or.pvalue:.2e}',
                transform=ax.transAxes, fontsize=8, va='top')
    else:
        ax.text(0.02, 0.95, f'Paired Wilcoxon p={wsr_or.pvalue:.2f} (ns)',
                transform=ax.transAxes, fontsize=8, va='top')

plt.tight_layout()
fig.savefig(OUTDIR / 'fig4_cooccurrence.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Figure 4 saved")

# --- Figure 5: Locus consistency + Genomic context ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# 5A: Locus modification consistency
ax = axes[0]
if len(consist_df) > 0:
    ax.hist(consist_df['mean_psi_frac'], bins=30, color='#9467bd', alpha=0.7,
            edgecolor='white')
    ax.axvline(consist_df['mean_psi_frac'].median(), color='red',
               linestyle='--', linewidth=1.5, label=f'median={consist_df["mean_psi_frac"].median():.2f}')
    ax.set_xlabel('Mean psi fraction across replicates')
    ax.set_ylabel('Number of loci')
    ax.set_title(f'A. Per-locus psi consistency (n={len(consist_df)} loci)')
    ax.legend(fontsize=8)
    ax.text(0.02, 0.88, f'Mean range={consist_df["range_psi_frac"].mean():.2f}',
            transform=ax.transAxes, fontsize=8, va='top')

# 5B: Intronic vs Intergenic
ax = axes[1]
ctx_data = []
for ctx in ['intronic', 'intergenic']:
    sub = l1[l1['TE_group'] == ctx]
    ctx_data.append(sub['psi_per_kb'])

bp2 = ax.boxplot(ctx_data, labels=['Intronic', 'Intergenic'], widths=0.5,
                 patch_artist=True, showfliers=False,
                 medianprops={'color': 'black', 'linewidth': 1.5})
bp2['boxes'][0].set_facecolor('#2ca02c')
bp2['boxes'][1].set_facecolor('#ff7f0e')
ax.set_ylabel('Pseudouridine sites/kb')
ax.set_title('B. Genomic context')
ax.text(0.02, 0.95,
        f'Intronic: {ctx_data[0].median():.2f} (n={len(ctx_data[0]):,})\n'
        f'Intergenic: {ctx_data[1].median():.2f} (n={len(ctx_data[1]):,})\n'
        f'MW p={mw_ctx_psi.pvalue:.1e}',
        transform=ax.transAxes, fontsize=8, va='top')

plt.tight_layout()
fig.savefig(OUTDIR / 'fig5_consistency_context.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Figure 5 saved")

# --- Figure 6: Motif enrichment ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, mod_type in enumerate(['m6A', 'psi']):
    ax = axes[i]
    # Get top motifs by L1 modRatio
    l1m = motif_df[(motif_df['mod_type'] == mod_type) & (motif_df['source'] == 'L1')].copy()
    ctrlm = motif_df[(motif_df['mod_type'] == mod_type) & (motif_df['source'] == 'Control')].copy()
    if len(l1m) == 0:
        continue
    top = l1m.nlargest(10, 'mean_modRatio')
    motifs_list = top['motif'].tolist()

    x4 = np.arange(len(motifs_list))
    l1_vals = [top[top['motif'] == m]['mean_modRatio'].iloc[0] for m in motifs_list]
    ctrl_vals = []
    for m in motifs_list:
        c = ctrlm[ctrlm['motif'] == m]
        ctrl_vals.append(c['mean_modRatio'].iloc[0] if len(c) > 0 else 0)

    ax.barh(x4 + 0.2, l1_vals, 0.35, color='#d62728', alpha=0.8, label='L1')
    ax.barh(x4 - 0.2, ctrl_vals, 0.35, color='#1f77b4', alpha=0.8, label='Control')
    ax.set_yticks(x4)
    ax.set_yticklabels(motifs_list, fontsize=8, fontfamily='monospace')
    ax.set_xlabel('Coverage-weighted mean modRatio (%)')
    mod_label = 'm6A' if mod_type == 'm6A' else 'Pseudouridine'
    ax.set_title(f'{"A" if i==0 else "B"}. {mod_label} motif enrichment')
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUTDIR / 'fig6_motif.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Figure 6 saved")

print("\n" + "=" * 60)
print("All analyses complete!")
print(f"Figures saved to: {OUTDIR}")
