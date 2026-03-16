#!/usr/bin/env python3
"""
Natural mutagenesis screen: which L1 consensus positions, when mutated,
correlate with stress vulnerability?

Approach:
1. Extract genomic sequence of each ancient L1 in our dataset
2. Align to L1HS consensus using minimap2
3. Per consensus position: compare mutation rate between
   stress-sensitive (short poly(A)) vs stress-resistant (long poly(A))
4. Generate a "stress protection map" across L1 consensus
"""
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
import tempfile
import os
import pysam

OUT_DIR = 'topic_08_sequence_features'
REF = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta'
L1_CONS = f'{OUT_DIR}/L1HS_consensus.fa'

# ── Load data ──
CACHE_DIR = 'topic_05_cellline/part3_l1_per_read_cache'
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

dfs = []
for grp in ['HeLa_1','HeLa_2','HeLa_3','HeLa-Ars_1','HeLa-Ars_2','HeLa-Ars_3']:
    cache = pd.read_csv(f'{CACHE_DIR}/{grp}_l1_per_read.tsv', sep='\t')
    cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)
    summ = pd.read_csv(f'../../results_group/{grp}/g_summary/{grp}_L1_summary.tsv', sep='\t')
    summ = summ[summ['qc_tag'] == 'PASS']
    merged = summ.merge(cache[['read_id', 'm6a_sites_high', 'm6a_per_kb']], on='read_id', how='inner')
    merged['group'] = grp
    merged['is_stress'] = 1 if 'Ars' in grp else 0
    dfs.append(merged)
df = pd.concat(dfs, ignore_index=True)
df['subfamily'] = df['gene_id']
df['age'] = df['subfamily'].apply(lambda x: 'Young' if x in YOUNG else 'Ancient')
df['te_length'] = df['te_end'] - df['te_start']

# Focus on ancient stressed reads with reasonable TE size
anc_st = df[(df['age'] == 'Ancient') & (df['is_stress'] == 1)].copy()
# Need TE elements ≥300bp for meaningful alignment
anc_st = anc_st[anc_st['te_length'] >= 300].copy()
print(f"Ancient stressed reads (TE≥300bp): {len(anc_st)}")

# Classify by poly(A): bottom 25% = sensitive, top 25% = resistant
q25, q75 = anc_st['polya_length'].quantile([0.25, 0.75])
anc_st['stress_group'] = 'middle'
anc_st.loc[anc_st['polya_length'] <= q25, 'stress_group'] = 'sensitive'
anc_st.loc[anc_st['polya_length'] >= q75, 'stress_group'] = 'resistant'
print(f"Sensitive (poly(A)≤{q25:.0f}): {(anc_st['stress_group']=='sensitive').sum()}")
print(f"Resistant (poly(A)≥{q75:.0f}): {(anc_st['stress_group']=='resistant').sum()}")

# ── Step 1: Extract genomic sequences of L1 elements ──
# Create BED file of unique L1 elements
l1_loci = anc_st[['te_chr', 'te_start', 'te_end', 'transcript_id', 'te_strand']].drop_duplicates()
# Add stress_group info per locus (use the read with most reads or the mean)
locus_group = anc_st.groupby('transcript_id').agg(
    mean_polya=('polya_length', 'mean'),
    n_reads=('read_id', 'count'),
    mean_m6a=('m6a_per_kb', 'mean'),
    te_chr=('te_chr', 'first'),
    te_start=('te_start', 'first'),
    te_end=('te_end', 'first'),
    te_strand=('te_strand', 'first'),
).reset_index()
locus_group['locus_q25'] = locus_group['mean_polya'].quantile(0.25)
locus_group['locus_q75'] = locus_group['mean_polya'].quantile(0.75)
locus_group['locus_group'] = 'middle'
locus_group.loc[locus_group['mean_polya'] <= locus_group['locus_q25'], 'locus_group'] = 'sensitive'
locus_group.loc[locus_group['mean_polya'] >= locus_group['locus_q75'], 'locus_group'] = 'resistant'

print(f"\nUnique loci: {len(locus_group)}")
print(f"Sensitive loci: {(locus_group['locus_group']=='sensitive').sum()}")
print(f"Resistant loci: {(locus_group['locus_group']=='resistant').sum()}")

# Write BED for bedtools getfasta
bed_path = f'{OUT_DIR}/l1_loci_for_alignment.bed'
with open(bed_path, 'w') as f:
    for _, row in locus_group.iterrows():
        # BED format: chr start end name score strand
        f.write(f"{row['te_chr']}\t{row['te_start']}\t{row['te_end']}\t{row['transcript_id']}\t0\t{row['te_strand']}\n")

# Extract sequences (strand-aware)
fasta_path = f'{OUT_DIR}/l1_loci_sequences.fa'
BEDTOOLS = '/blaze/apps/envs/bedtools/2.31.0/bin/bedtools'
cmd = f"{BEDTOOLS} getfasta -fi {REF} -bed {bed_path} -s -name -fo {fasta_path}"
subprocess.run(cmd, shell=True, check=True)
print(f"Extracted {sum(1 for l in open(fasta_path) if l.startswith('>'))//1} sequences")

# ── Step 2: Align to L1HS consensus with minimap2 ──
bam_path = f'{OUT_DIR}/l1_loci_vs_consensus.bam'
MINIMAP2 = '/blaze/apps/envs/minimap2/2.28/bin/minimap2'
SAMTOOLS = '/blaze/apps/envs/samtools/1.23/bin/samtools'
cmd = f"{MINIMAP2} -a --eqx --MD -t 4 {L1_CONS} {fasta_path} 2>/dev/null | {SAMTOOLS} sort -o {bam_path} && {SAMTOOLS} index {bam_path}"
subprocess.run(cmd, shell=True, check=True)
print(f"Alignment complete: {bam_path}")

# ── Step 3: Per-position mutation analysis ──
L1_LEN = 6064  # L1HS consensus length

# For each aligned locus, get per-position match/mismatch
bam = pysam.AlignmentFile(bam_path, 'rb')

# Initialize per-position counters
pos_match_sensitive = np.zeros(L1_LEN)
pos_mismatch_sensitive = np.zeros(L1_LEN)
pos_match_resistant = np.zeros(L1_LEN)
pos_mismatch_resistant = np.zeros(L1_LEN)
pos_coverage_sensitive = np.zeros(L1_LEN)
pos_coverage_resistant = np.zeros(L1_LEN)

# Map locus name to group
locus_to_group = dict(zip(locus_group['transcript_id'], locus_group['locus_group']))

n_aligned = 0
for read in bam.fetch():
    # Get locus name from query name (bedtools adds (strand) suffix)
    qname = read.query_name
    # bedtools getfasta -name adds ::chr:start-end(strand)
    locus_id = qname.split('::')[0] if '::' in qname else qname

    group = locus_to_group.get(locus_id)
    if group not in ('sensitive', 'resistant'):
        continue

    if read.is_unmapped:
        continue

    n_aligned += 1

    # Use CIGAR with --eqx to get per-position match/mismatch
    # reference positions on L1HS consensus
    pairs = read.get_aligned_pairs(with_seq=True)

    for qpos, rpos, ref_base in pairs:
        if rpos is None or qpos is None:
            continue
        if rpos < 0 or rpos >= L1_LEN:
            continue

        query_base = read.query_sequence[qpos] if qpos is not None else None

        if group == 'sensitive':
            pos_coverage_sensitive[rpos] += 1
            if ref_base.isupper():  # match
                pos_match_sensitive[rpos] += 1
            else:  # mismatch (lowercase in pysam)
                pos_mismatch_sensitive[rpos] += 1
        else:
            pos_coverage_resistant[rpos] += 1
            if ref_base.isupper():
                pos_match_resistant[rpos] += 1
            else:
                pos_mismatch_resistant[rpos] += 1

bam.close()
print(f"\nAligned loci processed: {n_aligned}")

# ── Step 4: Per-position mutation rate comparison ──
# Mutation rate = mismatch / (match + mismatch)
min_cov = 20  # minimum coverage at a position to include

mut_rate_sens = np.full(L1_LEN, np.nan)
mut_rate_resi = np.full(L1_LEN, np.nan)

for i in range(L1_LEN):
    total_s = pos_match_sensitive[i] + pos_mismatch_sensitive[i]
    total_r = pos_match_resistant[i] + pos_mismatch_resistant[i]
    if total_s >= min_cov:
        mut_rate_sens[i] = pos_mismatch_sensitive[i] / total_s
    if total_r >= min_cov:
        mut_rate_resi[i] = pos_mismatch_resistant[i] / total_r

# Smoothed (50bp window)
def smooth(arr, window=50):
    result = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        start = max(0, i - window // 2)
        end = min(len(arr), i + window // 2)
        valid = arr[start:end]
        valid = valid[~np.isnan(valid)]
        if len(valid) >= 5:
            result[i] = np.mean(valid)
    return result

mut_smooth_s = smooth(mut_rate_sens)
mut_smooth_r = smooth(mut_rate_resi)
mut_diff = mut_smooth_s - mut_smooth_r  # positive = more mutated in sensitive

# ── Step 5: Identify significant regions ──
# Sliding window test: 100bp windows, Fisher's exact test
window = 100
step = 50
results = []

for start in range(0, L1_LEN - window, step):
    end = start + window

    match_s = int(pos_match_sensitive[start:end].sum())
    mis_s = int(pos_mismatch_sensitive[start:end].sum())
    match_r = int(pos_match_resistant[start:end].sum())
    mis_r = int(pos_mismatch_resistant[start:end].sum())

    total_s = match_s + mis_s
    total_r = match_r + mis_r

    if total_s < 100 or total_r < 100:
        continue

    rate_s = mis_s / total_s if total_s > 0 else 0
    rate_r = mis_r / total_r if total_r > 0 else 0

    _, p = stats.fisher_exact([[match_s, mis_s], [match_r, mis_r]])

    results.append({
        'cons_start': start, 'cons_end': end,
        'mut_rate_sensitive': rate_s, 'mut_rate_resistant': rate_r,
        'diff': rate_s - rate_r, 'ratio': rate_s / rate_r if rate_r > 0 else np.inf,
        'p': p, 'n_sensitive': total_s, 'n_resistant': total_r
    })

res_df = pd.DataFrame(results)
res_df['p_adj'] = res_df['p'] * len(res_df)  # Bonferroni
res_df['sig'] = res_df['p_adj'] < 0.05

# Annotate domains
def annotate_domain(start):
    if start < 908: return '5UTR'
    elif start < 1990: return 'ORF1'
    elif start < 5817: return 'ORF2'
    else: return '3UTR'
res_df['domain'] = res_df['cons_start'].apply(annotate_domain)

print("\n" + "="*80)
print("MUTATION SENSITIVITY MAP: Regions where sensitive L1 are MORE mutated")
print("="*80)
print(f"\n{'Position':<15} {'Domain':<8} {'Mut_Sens':>10} {'Mut_Resi':>10} {'Diff':>8} {'P':>12} {'Sig':>5}")
print("-"*75)

sig_regions = res_df[res_df['sig']].sort_values('diff', ascending=False)
for _, r in sig_regions.head(20).iterrows():
    print(f"  {r['cons_start']}-{r['cons_end']:<10} {r['domain']:<8} {r['mut_rate_sensitive']:>10.4f} {r['mut_rate_resistant']:>10.4f} {r['diff']:>+8.4f} {r['p']:>12.2e}   *")

print(f"\nTotal significant windows (Bonferroni): {res_df['sig'].sum()} / {len(res_df)}")

# Regions where resistant L1 are MORE mutated (protective mutations?)
print("\n" + "="*80)
print("Regions where RESISTANT L1 are more mutated (protective mutations?)")
print("="*80)
sig_protective = res_df[res_df['sig'] & (res_df['diff'] < 0)].sort_values('diff')
for _, r in sig_protective.head(10).iterrows():
    print(f"  {r['cons_start']}-{r['cons_end']:<10} {r['domain']:<8} {r['mut_rate_sensitive']:>10.4f} {r['mut_rate_resistant']:>10.4f} {r['diff']:>+8.4f} {r['p']:>12.2e}   *")

# ── Summary by domain ──
print("\n" + "="*80)
print("SUMMARY BY DOMAIN")
print("="*80)
for domain in ['5UTR', 'ORF1', 'ORF2', '3UTR']:
    dom_data = res_df[res_df['domain'] == domain]
    if len(dom_data) > 0:
        mean_diff = dom_data['diff'].mean()
        n_sig = dom_data['sig'].sum()
        print(f"  {domain:<8} windows={len(dom_data):>4}  mean_diff={mean_diff:>+.4f}  sig_windows={n_sig}")

# Save
res_df.to_csv(f'{OUT_DIR}/l1_mutation_sensitivity_windows.tsv', sep='\t', index=False)

# Per-position data
pos_df = pd.DataFrame({
    'position': range(L1_LEN),
    'mut_rate_sensitive': mut_rate_sens,
    'mut_rate_resistant': mut_rate_resi,
    'coverage_sensitive': pos_coverage_sensitive,
    'coverage_resistant': pos_coverage_resistant,
    'mut_smooth_sensitive': mut_smooth_s,
    'mut_smooth_resistant': mut_smooth_r,
    'diff_smooth': mut_diff,
})
pos_df.to_csv(f'{OUT_DIR}/l1_perposition_mutation_rates.tsv', sep='\t', index=False)
print(f"\nSaved: l1_mutation_sensitivity_windows.tsv, l1_perposition_mutation_rates.tsv")
