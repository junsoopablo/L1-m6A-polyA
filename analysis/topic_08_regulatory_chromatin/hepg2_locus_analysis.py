#!/usr/bin/env python3
"""
HepG2 Top Loci Analysis: Why is HepG2 an outlier?
- Concentration of reads in top loci (Gini coefficient)
- m6A/kb, poly(A), read length for top loci vs overall
- Comparison with K562 and A549
- Sensitivity analysis: effect of removing top loci on overall m6A/kb

NOTE: transcript_id = unique locus (e.g., L1PA7_dup11216), gene_id = subfamily (e.g., L1PA7)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ============================================================
# Config
# ============================================================
BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
SUMMARY_DIR = BASE / 'results_group'
CACHE_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
OUT_DIR = BASE / 'analysis/01_exploration/topic_08_regulatory_chromatin/hepg2_locus_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

CELL_LINES = {
    'HepG2': ['HepG2_5', 'HepG2_6'],
    'K562':  ['K562_4', 'K562_5', 'K562_6'],
    'A549':  ['A549_4', 'A549_5', 'A549_6'],
}

# ============================================================
# Helper functions
# ============================================================
def load_summaries(groups):
    """Load and combine L1 summary files for given groups."""
    dfs = []
    for g in groups:
        fp = SUMMARY_DIR / g / 'g_summary' / f'{g}_L1_summary.tsv'
        if fp.exists():
            df = pd.read_csv(fp, sep='\t')
            df['group'] = g
            dfs.append(df)
        else:
            print(f"  WARNING: {fp} not found")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_part3_cache(groups):
    """Load Part3 per-read cache for given groups."""
    dfs = []
    for g in groups:
        fp = CACHE_DIR / f'{g}_l1_per_read.tsv'
        if fp.exists():
            df = pd.read_csv(fp, sep='\t')
            df['group'] = g
            dfs.append(df)
        else:
            print(f"  WARNING: {fp} not found")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def gini_coefficient(values):
    """Calculate Gini coefficient of an array of values."""
    values = np.array(values, dtype=float)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


def classify_age(subfamily):
    """Classify L1 subfamily as young or ancient."""
    return 'young' if subfamily in YOUNG_SUBFAMILIES else 'ancient'


# ============================================================
# Main analysis
# ============================================================
def analyze_cell_line(cl_name, groups):
    """Full analysis for one cell line, at locus level (transcript_id)."""
    print(f"\n{'='*80}")
    print(f" {cl_name} LOCUS ANALYSIS (transcript_id = unique locus)")
    print(f"{'='*80}")

    # --- 1. Load summary ---
    summary = load_summaries(groups)
    if summary.empty:
        print("  No summary data found!")
        return None

    # Filter PASS reads with valid poly(A)
    pass_reads = summary[summary['qc_tag'] == 'PASS'].copy()
    print(f"\n  Total reads: {len(summary)}, PASS reads: {len(pass_reads)}")

    # --- 2. Load Part3 cache ---
    cache = load_part3_cache(groups)
    if cache.empty:
        print("  No Part3 cache found!")
        return None
    print(f"  Part3 cache reads: {len(cache)}")

    # --- 3. Merge ---
    merged = pass_reads.merge(cache[['read_id', 'psi_sites_high', 'm6a_sites_high']],
                               on='read_id', how='inner')
    # Compute m6A/kb, psi/kb
    merged['m6a_per_kb'] = merged['m6a_sites_high'] / (merged['read_length'] / 1000)
    merged['psi_per_kb'] = merged['psi_sites_high'] / (merged['read_length'] / 1000)
    # subfamily = gene_id, locus = transcript_id
    merged['subfamily'] = merged['gene_id']
    merged['age'] = merged['subfamily'].apply(classify_age)
    # Valid poly(A)
    merged_polya = merged[merged['polya_length'] > 0]
    print(f"  Merged reads (PASS + Part3): {len(merged)}")
    print(f"  With valid poly(A): {len(merged_polya)}")

    # --- 4. Per-locus read count (using transcript_id) ---
    locus_counts = merged.groupby('transcript_id').agg(
        read_count=('read_id', 'count'),
        subfamily=('gene_id', 'first'),
    ).reset_index()

    # Get TE_group (most common per locus)
    te_group_mode = merged.groupby('transcript_id')['TE_group'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown')
    locus_counts['TE_group'] = locus_counts['transcript_id'].map(te_group_mode)
    locus_counts['age'] = locus_counts['subfamily'].apply(classify_age)

    locus_counts = locus_counts.sort_values('read_count', ascending=False).reset_index(drop=True)
    total_reads = locus_counts['read_count'].sum()
    locus_counts['cum_pct'] = locus_counts['read_count'].cumsum() / total_reads * 100

    # Gini
    gini = gini_coefficient(locus_counts['read_count'].values)
    print(f"\n  Total loci: {len(locus_counts)}")
    print(f"  Gini coefficient: {gini:.3f}")
    print(f"  Top 1 locus: {locus_counts.iloc[0]['transcript_id']} "
          f"({locus_counts.iloc[0]['read_count']} reads, {locus_counts.iloc[0]['cum_pct']:.1f}%)")
    if len(locus_counts) >= 10:
        print(f"  Top 10 loci: {locus_counts.iloc[9]['cum_pct']:.1f}% cumulative")
    if len(locus_counts) >= 30:
        print(f"  Top 30 loci: {locus_counts.iloc[29]['cum_pct']:.1f}% cumulative")

    # --- 5. Top 30 table ---
    n_show = min(30, len(locus_counts))
    print(f"\n  {'='*100}")
    print(f"  TOP {n_show} LOCI BY READ COUNT")
    print(f"  {'='*100}")
    print(f"  {'Rank':<5} {'Locus':<30} {'Reads':<8} {'Subfamily':<12} {'Age':<8} {'TE_group':<12} {'Cum%':<8}")
    print(f"  {'-'*100}")
    for i, row in locus_counts.head(n_show).iterrows():
        rank = i + 1
        print(f"  {rank:<5} {row['transcript_id']:<30} {row['read_count']:<8} {row['subfamily']:<12} "
              f"{row['age']:<8} {row['TE_group']:<12} {row['cum_pct']:<8.1f}")

    # --- 6. Top 10 loci: m6A/kb, poly(A), read length ---
    top10_loci = locus_counts.head(min(10, len(locus_counts)))['transcript_id'].tolist()
    overall_medians = {
        'm6a_per_kb': merged['m6a_per_kb'].median(),
        'polya_length': merged_polya['polya_length'].median(),
        'read_length': merged['read_length'].median(),
        'psi_per_kb': merged['psi_per_kb'].median(),
    }
    print(f"\n  {cl_name} OVERALL MEDIANS:")
    print(f"    m6A/kb = {overall_medians['m6a_per_kb']:.2f}")
    print(f"    psi/kb = {overall_medians['psi_per_kb']:.2f}")
    print(f"    poly(A) = {overall_medians['polya_length']:.1f} nt")
    print(f"    read length = {overall_medians['read_length']:.0f} bp")

    print(f"\n  {'='*120}")
    print(f"  TOP 10 LOCI DETAILS vs OVERALL")
    print(f"  {'='*120}")
    print(f"  {'Locus':<30} {'N':<6} {'Age':<8} {'m6A/kb':<10} {'psi/kb':<10} {'poly(A)':<10} {'RL':<10} {'TE_grp':<12} {'host_gene':<20}")
    print(f"  {'-'*120}")

    top10_results = []
    for locus in top10_loci:
        locus_data = merged[merged['transcript_id'] == locus]
        locus_polya = locus_data[locus_data['polya_length'] > 0]
        n = len(locus_data)
        age = classify_age(locus_data['subfamily'].iloc[0])
        m6a = locus_data['m6a_per_kb'].median()
        psi = locus_data['psi_per_kb'].median()
        polya = locus_polya['polya_length'].median() if len(locus_polya) > 0 else np.nan
        rl = locus_data['read_length'].median()
        te_grp = locus_data['TE_group'].mode().iloc[0] if len(locus_data) > 0 else '?'
        genes = locus_data['overlapping_genes'].dropna().unique()
        gene_str = ','.join([g for g in genes if g != '']) if len(genes) > 0 else 'intergenic'
        if gene_str == '':
            gene_str = 'intergenic'
        print(f"  {locus:<30} {n:<6} {age:<8} {m6a:<10.2f} {psi:<10.2f} "
              f"{polya:<10.1f} {rl:<10.0f} {te_grp:<12} {gene_str:<20}")
        top10_results.append({
            'locus': locus, 'n': n, 'age': age, 'm6a_per_kb': m6a,
            'psi_per_kb': psi, 'polya': polya, 'read_length': rl,
            'TE_group': te_grp, 'host_gene': gene_str
        })

    # --- 7. Sensitivity: removing top N loci ---
    print(f"\n  {'='*90}")
    print(f"  SENSITIVITY: EFFECT OF REMOVING TOP LOCI ON m6A/kb AND poly(A)")
    print(f"  {'='*90}")
    print(f"  {'Removed':<15} {'N_reads':<10} {'N_loci':<10} {'med_m6A/kb':<12} {'delta_m6A':<12} {'med_polyA':<12} {'delta_polyA':<12}")
    print(f"  {'-'*90}")

    baseline_m6a = merged['m6a_per_kb'].median()
    baseline_polya = merged_polya['polya_length'].median()
    print(f"  {'None':<15} {len(merged):<10} {len(locus_counts):<10} {baseline_m6a:<12.2f} {'--':<12} {baseline_polya:<12.1f} {'--':<12}")

    sensitivity_results = [{'removed': 'None', 'n': len(merged),
                            'm6a_per_kb': baseline_m6a, 'polya': baseline_polya}]

    for n_remove in [1, 3, 5, 10, 20, 30, 50, 100]:
        if n_remove > len(locus_counts):
            break
        top_n_loci = set(locus_counts.head(n_remove)['transcript_id'].tolist())
        remaining = merged[~merged['transcript_id'].isin(top_n_loci)]
        remaining_polya = remaining[remaining['polya_length'] > 0]
        if len(remaining) == 0:
            break
        m6a_val = remaining['m6a_per_kb'].median()
        polya_val = remaining_polya['polya_length'].median() if len(remaining_polya) > 0 else np.nan
        delta_m6a = m6a_val - baseline_m6a
        delta_polya = polya_val - baseline_polya if not np.isnan(polya_val) else np.nan
        top_n_reads = locus_counts.head(n_remove)['read_count'].sum()
        n_remaining_loci = len(locus_counts) - n_remove
        print(f"  Top {n_remove:<10} {len(remaining):<10} {n_remaining_loci:<10} {m6a_val:<12.2f} {delta_m6a:<+12.2f} "
              f"{polya_val:<12.1f} {delta_polya:<+12.1f}")
        sensitivity_results.append({
            'removed': f'Top {n_remove}', 'n': len(remaining),
            'm6a_per_kb': m6a_val, 'polya': polya_val,
            'removed_reads': top_n_reads,
            'removed_pct': top_n_reads / total_reads * 100
        })

    # --- Return data for cross-CL comparison ---
    return {
        'cl': cl_name,
        'total_reads': total_reads,
        'total_loci': len(locus_counts),
        'gini': gini,
        'top1_pct': locus_counts.iloc[0]['cum_pct'],
        'top1_locus': locus_counts.iloc[0]['transcript_id'],
        'top1_reads': locus_counts.iloc[0]['read_count'],
        'top10_pct': locus_counts.iloc[min(9, len(locus_counts)-1)]['cum_pct'] if len(locus_counts) >= 10 else np.nan,
        'overall_m6a': overall_medians['m6a_per_kb'],
        'overall_polya': overall_medians['polya_length'],
        'overall_rl': overall_medians['read_length'],
        'sensitivity': sensitivity_results,
        'locus_counts': locus_counts,
        'merged': merged,
    }


# ============================================================
# Run for all 3 cell lines
# ============================================================
results = {}
for cl_name, groups in CELL_LINES.items():
    results[cl_name] = analyze_cell_line(cl_name, groups)

# ============================================================
# Cross-CL comparison summary
# ============================================================
print(f"\n\n{'='*100}")
print(f" CROSS-CELL-LINE COMPARISON")
print(f"{'='*100}")
print(f"  {'CL':<10} {'Reads':<8} {'Loci':<8} {'Gini':<8} {'Top1%':<8} {'Top10%':<8} "
      f"{'m6A/kb':<10} {'polyA':<10} {'RL':<10}")
print(f"  {'-'*100}")
for cl_name in CELL_LINES:
    r = results[cl_name]
    if r is None:
        continue
    top10_str = f"{r['top10_pct']:.1f}" if not np.isnan(r.get('top10_pct', np.nan)) else 'N/A'
    print(f"  {cl_name:<10} {r['total_reads']:<8} {r['total_loci']:<8} {r['gini']:<8.3f} "
          f"{r['top1_pct']:<8.1f} {top10_str:<8} "
          f"{r['overall_m6a']:<10.2f} {r['overall_polya']:<10.1f} {r['overall_rl']:<10.0f}")

# Top 1 locus details
print(f"\n  TOP 1 LOCUS DETAILS:")
for cl_name in CELL_LINES:
    r = results[cl_name]
    if r is None:
        continue
    print(f"    {cl_name}: {r['top1_locus']} ({r['top1_reads']} reads, {r['top1_pct']:.1f}%)")

# ============================================================
# Deep dive: HepG2 #1 locus vs rest
# ============================================================
print(f"\n\n{'='*80}")
print(f" HEPG2 TOP LOCUS DEEP DIVE")
print(f"{'='*80}")
r = results['HepG2']
if r is not None:
    top1 = r['top1_locus']
    merged = r['merged']
    top1_data = merged[merged['transcript_id'] == top1]
    rest_data = merged[merged['transcript_id'] != top1]

    top1_polya = top1_data[top1_data['polya_length'] > 0]
    rest_polya = rest_data[rest_data['polya_length'] > 0]

    print(f"\n  Top locus: {top1}")
    print(f"  Subfamily: {top1_data['gene_id'].iloc[0]}")
    print(f"  Reads in top locus: {len(top1_data)} ({len(top1_data)/len(merged)*100:.1f}%)")

    # Genomic coordinates
    first_read = top1_data.iloc[0]
    print(f"  Location: {first_read['chr']}:{first_read['te_start']}-{first_read['te_end']} ({first_read['te_strand']})")
    print(f"  TE_group: {first_read['TE_group']}")
    genes = top1_data['overlapping_genes'].dropna().unique()
    gene_str = ','.join([g for g in genes if g != '']) if len(genes) > 0 else 'intergenic'
    print(f"  Host gene: {gene_str if gene_str else 'intergenic'}")

    for metric, label in [('m6a_per_kb', 'm6A/kb'), ('psi_per_kb', 'psi/kb'), ('read_length', 'Read length (bp)')]:
        top_med = top1_data[metric].median()
        rest_med = rest_data[metric].median()
        u_stat, u_p = stats.mannwhitneyu(top1_data[metric], rest_data[metric], alternative='two-sided')
        print(f"\n  {label}:")
        print(f"    Top1: median={top_med:.2f}, mean={top1_data[metric].mean():.2f}")
        print(f"    Rest: median={rest_med:.2f}, mean={rest_data[metric].mean():.2f}")
        print(f"    Ratio (top1/rest): {top_med/rest_med:.2f}x")
        print(f"    MWU p={u_p:.2e}")

    # Poly(A) comparison
    if len(top1_polya) > 0 and len(rest_polya) > 0:
        top_med = top1_polya['polya_length'].median()
        rest_med = rest_polya['polya_length'].median()
        u_stat, u_p = stats.mannwhitneyu(top1_polya['polya_length'], rest_polya['polya_length'],
                                          alternative='two-sided')
        print(f"\n  Poly(A) length:")
        print(f"    Top1: median={top_med:.1f}, mean={top1_polya['polya_length'].mean():.1f}")
        print(f"    Rest: median={rest_med:.1f}, mean={rest_polya['polya_length'].mean():.1f}")
        print(f"    Ratio (top1/rest): {top_med/rest_med:.2f}x")
        print(f"    MWU p={u_p:.2e}")

    # Genomic location of top 10
    print(f"\n  GENOMIC LOCATION OF TOP 10 LOCI:")
    for locus in r['locus_counts'].head(10)['transcript_id']:
        ldata = merged[merged['transcript_id'] == locus].iloc[0]
        subfamily = ldata['gene_id']
        strand = ldata.get('te_strand', ldata.get('read_strand', '?'))
        chr_val = ldata['chr']
        start = ldata['te_start'] if 'te_start' in ldata.index else ldata['start']
        end = ldata['te_end'] if 'te_end' in ldata.index else ldata['end']
        te_grp = ldata['TE_group']
        genes_raw = ldata.get('overlapping_genes', '')
        gene_display = genes_raw if pd.notna(genes_raw) and genes_raw != '' else 'intergenic'
        n = len(merged[merged['transcript_id'] == locus])
        print(f"    {locus:<28} {subfamily:<10} {chr_val}:{start}-{end} ({strand}) {te_grp:<12} "
              f"host={gene_display:<20} n={n}")

# ============================================================
# HepG2 top locus: check if present in K562 and A549
# ============================================================
print(f"\n\n{'='*80}")
print(f" HEPG2 TOP LOCUS IN OTHER CELL LINES")
print(f"{'='*80}")
if results['HepG2'] is not None:
    hepg2_top1 = results['HepG2']['top1_locus']
    print(f"\n  HepG2 top locus: {hepg2_top1}")
    for cl_name in ['K562', 'A549']:
        r = results[cl_name]
        if r is None:
            continue
        lc = r['locus_counts']
        match = lc[lc['transcript_id'] == hepg2_top1]
        if len(match) > 0:
            rank = match.index[0] + 1
            count = match.iloc[0]['read_count']
            pct = count / r['total_reads'] * 100
            print(f"    {cl_name}: rank #{rank}, {count} reads ({pct:.1f}%)")
        else:
            print(f"    {cl_name}: NOT FOUND in PASS reads")

# ============================================================
# Singleton analysis
# ============================================================
print(f"\n\n{'='*80}")
print(f" SINGLETON ANALYSIS (1 read per locus)")
print(f"{'='*80}")
for cl_name in CELL_LINES:
    r = results[cl_name]
    if r is None:
        continue
    lc = r['locus_counts']
    n_singleton = (lc['read_count'] == 1).sum()
    n_multi = (lc['read_count'] > 1).sum()
    n_10plus = (lc['read_count'] >= 10).sum()
    pct_singleton = n_singleton / len(lc) * 100
    print(f"\n  {cl_name}:")
    print(f"    Total loci: {len(lc)}")
    print(f"    Singletons: {n_singleton} ({pct_singleton:.1f}%)")
    print(f"    Multi-read: {n_multi} ({n_multi/len(lc)*100:.1f}%)")
    print(f"    >=10 reads: {n_10plus} ({n_10plus/len(lc)*100:.1f}%)")
    # How many reads are in singletons vs multi
    singleton_reads = n_singleton  # 1 read each
    multi_reads = r['total_reads'] - singleton_reads
    print(f"    Reads in singletons: {singleton_reads} ({singleton_reads/r['total_reads']*100:.1f}%)")
    print(f"    Reads in multi-read loci: {multi_reads} ({multi_reads/r['total_reads']*100:.1f}%)")

# ============================================================
# Age composition of top loci
# ============================================================
print(f"\n\n{'='*80}")
print(f" AGE COMPOSITION: TOP LOCI vs ALL")
print(f"{'='*80}")
for cl_name in CELL_LINES:
    r = results[cl_name]
    if r is None:
        continue
    lc = r['locus_counts']
    all_young_pct = (lc['age'] == 'young').sum() / len(lc) * 100
    all_young_read_pct = lc[lc['age'] == 'young']['read_count'].sum() / r['total_reads'] * 100

    top30 = lc.head(min(30, len(lc)))
    top30_young_pct = (top30['age'] == 'young').sum() / len(top30) * 100
    top30_young_read_pct = top30[top30['age'] == 'young']['read_count'].sum() / top30['read_count'].sum() * 100

    print(f"\n  {cl_name}:")
    print(f"    All loci: {all_young_pct:.1f}% young loci, {all_young_read_pct:.1f}% young reads")
    print(f"    Top 30:   {top30_young_pct:.1f}% young loci, {top30_young_read_pct:.1f}% young reads")

# ============================================================
# Subfamily breakdown of top loci
# ============================================================
print(f"\n\n{'='*80}")
print(f" SUBFAMILY BREAKDOWN IN TOP 30 LOCI")
print(f"{'='*80}")
for cl_name in CELL_LINES:
    r = results[cl_name]
    if r is None:
        continue
    top30 = r['locus_counts'].head(30)
    sub_counts = top30.groupby('subfamily')['read_count'].agg(['count', 'sum']).sort_values('sum', ascending=False)
    total_top30_reads = top30['read_count'].sum()
    print(f"\n  {cl_name} (top 30 loci):")
    for sub, row in sub_counts.head(10).iterrows():
        print(f"    {sub:<15} {int(row['count'])} loci, {int(row['sum'])} reads ({row['sum']/total_top30_reads*100:.1f}%)")

# ============================================================
# Save results
# ============================================================
# Save top 30 for each cell line
for cl_name in CELL_LINES:
    r = results[cl_name]
    if r is None:
        continue
    lc = r['locus_counts'].head(30).copy()
    lc.to_csv(OUT_DIR / f'{cl_name}_top30_loci.tsv', sep='\t', index=False)

# Save sensitivity analysis
sens_rows = []
for cl_name in CELL_LINES:
    r = results[cl_name]
    if r is None:
        continue
    for s in r['sensitivity']:
        s_copy = s.copy()
        s_copy['cell_line'] = cl_name
        sens_rows.append(s_copy)
pd.DataFrame(sens_rows).to_csv(OUT_DIR / 'sensitivity_analysis.tsv', sep='\t', index=False)

print(f"\n\nResults saved to: {OUT_DIR}")
print("Done.")
