#!/usr/bin/env python3
"""Cat B (lncRNA/pseudogene exon-overlapping) L1 reads vs PASS L1 reads comparison.

Compares across all major analysis dimensions:
  1. Basic stats: read count, read length, young/ancient, subfamily
  2. Poly(A) tail length
  3. m6A/psi modification (MAFIA BAM → per-read, per-site rate)
  4. Arsenite stress response (HeLa vs HeLa-Ars)
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
OUTDIR = TOPIC_07 / 'catB_vs_pass_analysis'
OUTDIR.mkdir(exist_ok=True)

PROB_THRESHOLD = 128  # 50% on 0-255 scale
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

# =============================================================================
# MAFIA MM/ML parser (verified: 17802=psi, 21891=m6A, N+/N- both)
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
        if '17802' in mod_type or 'U+p' in mod_type or 'T+p' in mod_type:
            mod_key = 'psi'
        elif '21891' in mod_type or 'A+a' in mod_type or 'A+m' in mod_type:
            mod_key = 'm6A'
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
    """Parse MAFIA BAM → per-read modification counts."""
    records = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam:
            mm = read.get_tag("MM") if read.has_tag("MM") else None
            ml = read.get_tag("ML") if read.has_tag("ML") else None
            mods = parse_mm_ml_tags(mm, ml)
            rl = read.query_length or read.infer_query_length() or 0
            if rl < 50:
                continue
            m6a_high = [p for p, prob in mods['m6A'] if prob >= PROB_THRESHOLD]
            psi_high = [p for p, prob in mods['psi'] if prob >= PROB_THRESHOLD]
            records.append({
                'read_id': read.query_name,
                'read_length': rl,
                'm6a_sites_high': len(m6a_high),
                'psi_sites_high': len(psi_high),
                'm6a_all': len(mods['m6A']),
                'psi_all': len(mods['psi']),
            })
    return pd.DataFrame(records)


# =============================================================================
# 1. Load Cat B metadata
# =============================================================================
print("=" * 70)
print("Loading Cat B read metadata...")
catB_meta_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = TOPIC_07 / f'catB_reads_{grp}.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['group'] = grp
            df['cell_line'] = group
            catB_meta_list.append(df)

catB_meta = pd.concat(catB_meta_list, ignore_index=True)
catB_meta['is_young'] = catB_meta['subfamily'].isin(YOUNG)
print(f"  Cat B total reads: {len(catB_meta)}")
print(f"  Young: {catB_meta['is_young'].sum()} ({catB_meta['is_young'].mean()*100:.1f}%)")

# =============================================================================
# 2. Load PASS L1 metadata (from L1_summary)
# =============================================================================
print("\nLoading PASS L1 metadata...")
pass_meta_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['group'] = grp
            df['cell_line'] = group
            # Extract subfamily from gene_id (transcript_id column)
            if 'gene_id' in df.columns:
                df['subfamily'] = df['gene_id']
            elif 'transcript_id' in df.columns:
                # transcript_id is like L1MC4_dup15 → subfamily = L1MC4
                df['subfamily'] = df['transcript_id'].str.replace(r'_dup\d+$', '', regex=True)
            df['is_young'] = df['subfamily'].isin(YOUNG)
            pass_meta_list.append(df)

pass_meta = pd.concat(pass_meta_list, ignore_index=True)
print(f"  PASS L1 total reads: {len(pass_meta)}")
print(f"  Young: {pass_meta['is_young'].sum()} ({pass_meta['is_young'].mean()*100:.1f}%)")

# =============================================================================
# 3. Parse Cat B MAFIA BAMs
# =============================================================================
print("\nParsing Cat B MAFIA BAMs...")
catB_mod_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        bam = RESULTS / grp / 'j_catB' / 'mafia' / f'{grp}.catB.mAFiA.reads.bam'
        if bam.exists():
            df = parse_bam_per_read(bam)
            df['group'] = grp
            df['cell_line'] = group
            catB_mod_list.append(df)
            print(f"  {grp}: {len(df)} reads with MAFIA data")

catB_mod = pd.concat(catB_mod_list, ignore_index=True)
catB_mod['m6a_per_kb'] = catB_mod['m6a_sites_high'] / (catB_mod['read_length'] / 1000)
catB_mod['psi_per_kb'] = catB_mod['psi_sites_high'] / (catB_mod['read_length'] / 1000)

# Merge with metadata to get age
catB_mod = catB_mod.merge(
    catB_meta[['read_id', 'group', 'subfamily', 'is_young', 'age']].drop_duplicates(),
    on=['read_id', 'group'], how='left'
)
print(f"  Cat B mod total: {len(catB_mod)} reads")

# =============================================================================
# 4. Load PASS L1 Part3 cache (already parsed)
# =============================================================================
print("\nLoading PASS L1 Part3 cache...")
pass_mod_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = TOPIC_05 / 'part3_l1_per_read_cache' / f'{grp}_l1_per_read.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['group'] = grp
            df['cell_line'] = group
            pass_mod_list.append(df)

pass_mod = pd.concat(pass_mod_list, ignore_index=True)
pass_mod['m6a_per_kb'] = pass_mod['m6a_sites_high'] / (pass_mod['read_length'] / 1000)
pass_mod['psi_per_kb'] = pass_mod['psi_sites_high'] / (pass_mod['read_length'] / 1000)

# Merge with PASS metadata for age
pass_mod = pass_mod.merge(
    pass_meta[['read_id', 'group', 'subfamily', 'is_young']].drop_duplicates(),
    on=['read_id', 'group'], how='left'
)
pass_mod['age'] = np.where(pass_mod['is_young'], 'young', 'ancient')
print(f"  PASS mod total: {len(pass_mod)} reads")

# =============================================================================
# 5. Load Control Part3 cache
# =============================================================================
print("\nLoading Control Part3 cache...")
ctrl_mod_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = TOPIC_05 / 'part3_ctrl_per_read_cache' / f'{grp}_ctrl_per_read.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['group'] = grp
            df['cell_line'] = group
            ctrl_mod_list.append(df)

ctrl_mod = pd.concat(ctrl_mod_list, ignore_index=True)
ctrl_mod['m6a_per_kb'] = ctrl_mod['m6a_sites_high'] / (ctrl_mod['read_length'] / 1000)
ctrl_mod['psi_per_kb'] = ctrl_mod['psi_sites_high'] / (ctrl_mod['read_length'] / 1000)
print(f"  Control mod total: {len(ctrl_mod)} reads")

# =============================================================================
# 6. Load Cat B poly(A)
# =============================================================================
print("\nLoading Cat B nanopolish poly(A)...")
catB_polya_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = RESULTS / grp / 'j_catB' / f'{grp}.catB.nanopolish.polya.tsv.gz'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df = df[df['qc_tag'] == 'PASS'].copy()
            df['group'] = grp
            df['cell_line'] = group
            df = df.rename(columns={'readname': 'read_id', 'polya_length': 'polya'})
            catB_polya_list.append(df[['read_id', 'group', 'cell_line', 'polya']])

catB_polya = pd.concat(catB_polya_list, ignore_index=True)
# Merge with metadata for age
catB_polya = catB_polya.merge(
    catB_meta[['read_id', 'group', 'subfamily', 'is_young', 'age']].drop_duplicates(),
    on=['read_id', 'group'], how='left'
)
print(f"  Cat B poly(A): {len(catB_polya)} reads")

# =============================================================================
# 7. Load PASS L1 poly(A) (from L1_summary)
# =============================================================================
print("\nLoading PASS L1 poly(A)...")
pass_polya_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            if 'polya_length' in df.columns:
                df2 = df[df['polya_length'].notna()].copy()
                df2['group'] = grp
                df2['cell_line'] = group
                df2 = df2.rename(columns={'polya_length': 'polya'})
                if 'gene_id' in df2.columns:
                    df2['subfamily'] = df2['gene_id']
                elif 'transcript_id' in df2.columns:
                    df2['subfamily'] = df2['transcript_id'].str.replace(r'_dup\d+$', '', regex=True)
                df2['is_young'] = df2['subfamily'].isin(YOUNG)
                df2['age'] = np.where(df2['is_young'], 'young', 'ancient')
                pass_polya_list.append(df2[['read_id', 'group', 'cell_line', 'polya', 'subfamily', 'is_young', 'age']])

pass_polya = pd.concat(pass_polya_list, ignore_index=True)
print(f"  PASS poly(A): {len(pass_polya)} reads")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 1: BASIC STATISTICS")
print("=" * 70)

# --- 1a. Read counts per cell line ---
print("\n--- 1a. Read Counts per Cell Line ---")
rows = []
for cl in CL_ORDER:
    n_catB = len(catB_meta[catB_meta['cell_line'] == cl])
    n_pass = len(pass_meta[pass_meta['cell_line'] == cl])
    rows.append({'cell_line': cl, 'catB': n_catB, 'PASS': n_pass,
                 'catB_pct_of_pass': n_catB / n_pass * 100 if n_pass else 0})

counts_df = pd.DataFrame(rows)
print(counts_df.to_string(index=False))
counts_df.to_csv(OUTDIR / 'read_counts_per_cl.tsv', sep='\t', index=False)

# --- 1b. Young/Ancient proportion ---
print("\n--- 1b. Young vs Ancient Proportion ---")
catB_young_pct = catB_meta.groupby('cell_line')['is_young'].mean() * 100
pass_young_pct = pass_meta.groupby('cell_line')['is_young'].mean() * 100
age_df = pd.DataFrame({'catB_young_pct': catB_young_pct, 'PASS_young_pct': pass_young_pct}).loc[CL_ORDER]
print(age_df.to_string())
age_df.to_csv(OUTDIR / 'young_pct_per_cl.tsv', sep='\t')

# --- 1c. Read length ---
print("\n--- 1c. Read Length (from MAFIA data) ---")
catB_rl = catB_mod.groupby('cell_line')['read_length'].median()
pass_rl = pass_mod.groupby('cell_line')['read_length'].median()
rl_df = pd.DataFrame({'catB_median_rl': catB_rl, 'PASS_median_rl': pass_rl}).loc[CL_ORDER]
rl_df['ratio'] = rl_df['catB_median_rl'] / rl_df['PASS_median_rl']
print(rl_df.to_string())
rl_df.to_csv(OUTDIR / 'read_length_per_cl.tsv', sep='\t')

# --- 1d. Top subfamilies ---
print("\n--- 1d. Top 10 Subfamilies ---")
catB_subfam = catB_meta['subfamily'].value_counts().head(10)
pass_subfam = pass_meta['subfamily'].value_counts().head(10)
print("Cat B:")
print(catB_subfam.to_string())
print("\nPASS:")
print(pass_subfam.to_string())

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: POLY(A) TAIL")
print("=" * 70)

# --- 2a. Overall poly(A) ---
print("\n--- 2a. Overall Poly(A) Length ---")
catB_med = catB_polya['polya'].median()
pass_med = pass_polya['polya'].median()
u_stat, u_p = stats.mannwhitneyu(catB_polya['polya'].dropna(), pass_polya['polya'].dropna(), alternative='two-sided')
print(f"  Cat B median: {catB_med:.1f} nt")
print(f"  PASS  median: {pass_med:.1f} nt")
print(f"  Difference: {catB_med - pass_med:+.1f} nt")
print(f"  Mann-Whitney p: {u_p:.2e}")

# --- 2b. Poly(A) per cell line ---
print("\n--- 2b. Poly(A) per Cell Line ---")
rows = []
for cl in CL_ORDER:
    cb = catB_polya[catB_polya['cell_line'] == cl]['polya']
    ps = pass_polya[pass_polya['cell_line'] == cl]['polya']
    if len(cb) > 5 and len(ps) > 5:
        _, p = stats.mannwhitneyu(cb.dropna(), ps.dropna(), alternative='two-sided')
        rows.append({'cell_line': cl, 'catB_n': len(cb), 'catB_median': cb.median(),
                     'PASS_n': len(ps), 'PASS_median': ps.median(),
                     'delta': cb.median() - ps.median(), 'p': p})
polya_cl = pd.DataFrame(rows)
print(polya_cl.to_string(index=False))
polya_cl.to_csv(OUTDIR / 'polya_per_cl.tsv', sep='\t', index=False)

# --- 2c. Poly(A) by age ---
print("\n--- 2c. Poly(A) by Age ---")
for cat_label, df in [('Cat B', catB_polya), ('PASS', pass_polya)]:
    for age in ['young', 'ancient']:
        sub = df[df['age'] == age]['polya']
        print(f"  {cat_label} {age}: n={len(sub)}, median={sub.median():.1f}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MODIFICATION (m6A/psi)")
print("=" * 70)

# --- 3a. Overall m6A/kb and psi/kb ---
print("\n--- 3a. Overall m6A/kb, psi/kb ---")
for label, df in [('Cat B', catB_mod), ('PASS', pass_mod), ('Control', ctrl_mod)]:
    m6a = df['m6a_per_kb'].median()
    psi = df['psi_per_kb'].median()
    print(f"  {label:8s}: m6A/kb={m6a:.2f}, psi/kb={psi:.2f}, n={len(df)}")

# --- 3b. m6A/kb per cell line ---
print("\n--- 3b. m6A/kb per Cell Line ---")
rows = []
for cl in CL_ORDER:
    cb = catB_mod[catB_mod['cell_line'] == cl]['m6a_per_kb']
    ps = pass_mod[pass_mod['cell_line'] == cl]['m6a_per_kb']
    ct = ctrl_mod[ctrl_mod['cell_line'] == cl]['m6a_per_kb']
    if len(cb) > 5 and len(ps) > 5:
        _, p = stats.mannwhitneyu(cb.dropna(), ps.dropna(), alternative='two-sided')
        rows.append({'cell_line': cl,
                     'catB_m6a_kb': cb.median(), 'PASS_m6a_kb': ps.median(),
                     'ctrl_m6a_kb': ct.median() if len(ct) > 0 else np.nan,
                     'catB_vs_PASS_ratio': cb.median() / ps.median() if ps.median() > 0 else np.nan,
                     'p': p})
m6a_cl = pd.DataFrame(rows)
print(m6a_cl.to_string(index=False))
m6a_cl.to_csv(OUTDIR / 'm6a_per_kb_per_cl.tsv', sep='\t', index=False)

# --- 3c. psi/kb per cell line ---
print("\n--- 3c. psi/kb per Cell Line ---")
rows = []
for cl in CL_ORDER:
    cb = catB_mod[catB_mod['cell_line'] == cl]['psi_per_kb']
    ps = pass_mod[pass_mod['cell_line'] == cl]['psi_per_kb']
    ct = ctrl_mod[ctrl_mod['cell_line'] == cl]['psi_per_kb']
    if len(cb) > 5 and len(ps) > 5:
        _, p = stats.mannwhitneyu(cb.dropna(), ps.dropna(), alternative='two-sided')
        rows.append({'cell_line': cl,
                     'catB_psi_kb': cb.median(), 'PASS_psi_kb': ps.median(),
                     'ctrl_psi_kb': ct.median() if len(ct) > 0 else np.nan,
                     'catB_vs_PASS_ratio': cb.median() / ps.median() if ps.median() > 0 else np.nan,
                     'p': p})
psi_cl = pd.DataFrame(rows)
print(psi_cl.to_string(index=False))
psi_cl.to_csv(OUTDIR / 'psi_per_kb_per_cl.tsv', sep='\t', index=False)

# --- 3d. Modification by age ---
print("\n--- 3d. m6A/kb by Age ---")
for label, df in [('Cat B', catB_mod), ('PASS', pass_mod)]:
    for age_val in ['young', 'ancient']:
        if 'age' in df.columns:
            sub = df[df['age'] == age_val]
        elif 'is_young' in df.columns:
            sub = df[df['is_young'] == (age_val == 'young')]
        else:
            continue
        if len(sub) > 0:
            print(f"  {label:8s} {age_val:8s}: m6A/kb={sub['m6a_per_kb'].median():.2f}, "
                  f"psi/kb={sub['psi_per_kb'].median():.2f}, n={len(sub)}")

# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: ARSENITE STRESS RESPONSE")
print("=" * 70)

# --- 4a. HeLa vs HeLa-Ars poly(A) ---
print("\n--- 4a. HeLa vs HeLa-Ars Poly(A) ---")
for label, polya_df in [('Cat B', catB_polya), ('PASS', pass_polya)]:
    hela = polya_df[polya_df['cell_line'] == 'HeLa']['polya']
    ars = polya_df[polya_df['cell_line'] == 'HeLa-Ars']['polya']
    if len(hela) > 5 and len(ars) > 5:
        _, p = stats.mannwhitneyu(hela.dropna(), ars.dropna(), alternative='two-sided')
        delta = ars.median() - hela.median()
        print(f"  {label:8s}: HeLa={hela.median():.1f}, Ars={ars.median():.1f}, "
              f"Δ={delta:+.1f} nt, p={p:.2e}, n_HeLa={len(hela)}, n_Ars={len(ars)}")

# --- 4b. Arsenite by age ---
print("\n--- 4b. Arsenite Effect by Age ---")
for label, polya_df in [('Cat B', catB_polya), ('PASS', pass_polya)]:
    for age_val in ['young', 'ancient']:
        hela = polya_df[(polya_df['cell_line'] == 'HeLa') & (polya_df['age'] == age_val)]['polya']
        ars = polya_df[(polya_df['cell_line'] == 'HeLa-Ars') & (polya_df['age'] == age_val)]['polya']
        if len(hela) >= 3 and len(ars) >= 3:
            _, p = stats.mannwhitneyu(hela.dropna(), ars.dropna(), alternative='two-sided')
            delta = ars.median() - hela.median()
            print(f"  {label:8s} {age_val:8s}: HeLa={hela.median():.1f}, Ars={ars.median():.1f}, "
                  f"Δ={delta:+.1f} nt, p={p:.2e}")
        else:
            print(f"  {label:8s} {age_val:8s}: too few reads (HeLa={len(hela)}, Ars={len(ars)})")

# --- 4c. HeLa vs HeLa-Ars m6A ---
print("\n--- 4c. HeLa vs HeLa-Ars m6A/kb ---")
for label, mod_df in [('Cat B', catB_mod), ('PASS', pass_mod)]:
    hela = mod_df[mod_df['cell_line'] == 'HeLa']['m6a_per_kb']
    ars = mod_df[mod_df['cell_line'] == 'HeLa-Ars']['m6a_per_kb']
    if len(hela) > 5 and len(ars) > 5:
        _, p = stats.mannwhitneyu(hela.dropna(), ars.dropna(), alternative='two-sided')
        print(f"  {label:8s}: HeLa={hela.median():.2f}, Ars={ars.median():.2f}, p={p:.2e}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Cat B vs PASS L1 Key Metrics")
print("=" * 70)

summary_rows = []

# Read count
summary_rows.append({
    'metric': 'Total reads',
    'catB': len(catB_meta),
    'PASS': len(pass_meta),
    'ratio_or_delta': f"{len(catB_meta)/len(pass_meta)*100:.1f}%"
})

# Young %
summary_rows.append({
    'metric': 'Young L1 %',
    'catB': f"{catB_meta['is_young'].mean()*100:.1f}%",
    'PASS': f"{pass_meta['is_young'].mean()*100:.1f}%",
    'ratio_or_delta': ''
})

# Read length
summary_rows.append({
    'metric': 'Read length (median)',
    'catB': f"{catB_mod['read_length'].median():.0f}",
    'PASS': f"{pass_mod['read_length'].median():.0f}",
    'ratio_or_delta': f"{catB_mod['read_length'].median()/pass_mod['read_length'].median():.2f}x"
})

# Poly(A)
summary_rows.append({
    'metric': 'Poly(A) (median)',
    'catB': f"{catB_polya['polya'].median():.1f}",
    'PASS': f"{pass_polya['polya'].median():.1f}",
    'ratio_or_delta': f"{catB_polya['polya'].median() - pass_polya['polya'].median():+.1f} nt"
})

# m6A/kb
summary_rows.append({
    'metric': 'm6A/kb (median)',
    'catB': f"{catB_mod['m6a_per_kb'].median():.2f}",
    'PASS': f"{pass_mod['m6a_per_kb'].median():.2f}",
    'ratio_or_delta': f"{catB_mod['m6a_per_kb'].median()/pass_mod['m6a_per_kb'].median():.2f}x"
})

# psi/kb
summary_rows.append({
    'metric': 'psi/kb (median)',
    'catB': f"{catB_mod['psi_per_kb'].median():.2f}",
    'PASS': f"{pass_mod['psi_per_kb'].median():.2f}",
    'ratio_or_delta': f"{catB_mod['psi_per_kb'].median()/pass_mod['psi_per_kb'].median():.2f}x"
})

# Arsenite poly(A)
cb_hela = catB_polya[catB_polya['cell_line'] == 'HeLa']['polya'].median()
cb_ars = catB_polya[catB_polya['cell_line'] == 'HeLa-Ars']['polya'].median()
ps_hela = pass_polya[pass_polya['cell_line'] == 'HeLa']['polya'].median()
ps_ars = pass_polya[pass_polya['cell_line'] == 'HeLa-Ars']['polya'].median()
summary_rows.append({
    'metric': 'Ars poly(A) Δ',
    'catB': f"{cb_ars - cb_hela:+.1f} nt",
    'PASS': f"{ps_ars - ps_hela:+.1f} nt",
    'ratio_or_delta': ''
})

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(OUTDIR / 'summary_catB_vs_pass.tsv', sep='\t', index=False)

print(f"\nAll results saved to: {OUTDIR}")
print("Done!")
