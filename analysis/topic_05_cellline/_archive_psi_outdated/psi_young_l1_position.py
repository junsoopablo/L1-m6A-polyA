#!/usr/bin/env python3
"""
Map psi positions onto young L1 structural domains.

Young L1 (L1HS/L1PA2/L1PA3) have conserved structure:
  5'UTR (0-908) | ORF1 (909-1923) | spacer (1924-1989) | ORF2 (1990-5816) | 3'UTR (5817-6068)

Focus on full-length elements (>5kb) to enable accurate domain mapping.
DRS has 3' bias so most reads cover 3'UTR + ORF2 region.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import ast
import glob

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPICDIR = PROJECT / 'analysis/01_exploration'
CACHE_DIR = TOPICDIR / 'topic_05_cellline/part3_l1_per_read_cache'
OUTDIR = TOPICDIR / 'topic_05_cellline/psi_young_l1_position'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA2', 'L1PA3'}  # L1PA1 has 0 reads

# L1 consensus domains (L1HS, ~6069bp)
L1_CONSENSUS_LEN = 6069
DOMAINS = [
    ("5'UTR", 0, 909),
    ("ORF1", 909, 1924),
    ("spacer", 1924, 1990),
    ("ORF2", 1990, 5817),
    ("3'UTR", 5817, 6069),
]

# =========================================================================
# Load data
# =========================================================================
print("Loading data...")

# Load all L1 summaries
sum_dfs = []
for p in sorted(glob.glob(str(PROJECT / 'results_group/*/g_summary/*_L1_summary.tsv'))):
    df = pd.read_csv(p, sep='\t')
    df = df[df['qc_tag'] == 'PASS']
    group = Path(p).stem.replace('_L1_summary', '')
    df['group'] = group
    sum_dfs.append(df)
all_sum = pd.concat(sum_dfs, ignore_index=True)

# Filter young L1
young_sum = all_sum[all_sum['gene_id'].isin(YOUNG)].copy()
young_sum['te_length'] = young_sum['te_end'] - young_sum['te_start']
print(f"Young L1 reads (all): {len(young_sum):,}")

# Load part3 cache (psi positions)
cache_dfs = []
for p in sorted(CACHE_DIR.glob('*_l1_per_read.tsv')):
    df = pd.read_csv(p, sep='\t')
    df['psi_positions'] = df['psi_positions'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    df['m6a_positions'] = df['m6a_positions'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    cache_dfs.append(df)
all_cache = pd.concat(cache_dfs, ignore_index=True)

# Merge
merged = young_sum.merge(
    all_cache[['read_id', 'psi_positions', 'm6a_positions', 'psi_sites_high', 'm6a_sites_high']],
    on='read_id', how='inner')
print(f"Young L1 reads with mod data: {len(merged):,}")

# Full-length elements (>5kb)
fl = merged[merged['te_length'] > 5000].copy()
print(f"Full-length (>5kb) young L1 reads: {len(fl):,}")
print(f"Full-length loci: {fl['transcript_id'].nunique()}")

# =========================================================================
# Map psi positions to L1 coordinates
# =========================================================================
print("\nMapping psi positions to L1 coordinates...")

def read_pos_to_l1_pos(row):
    """Convert read-coordinate psi positions to L1-element positions (from 5' end)."""
    psi_l1_positions = []
    for p in row['psi_positions']:
        if row['read_strand'] == '+':
            # + strand: read pos 0 = genomic start (leftmost)
            genomic_pos = row['start'] + p
            l1_pos = genomic_pos - row['te_start']
        else:
            # - strand: read pos 0 = genomic end (rightmost), reverse complemented
            genomic_pos = row['end'] - p
            # For - strand L1, 5' end is at te_end
            l1_pos = row['te_end'] - genomic_pos

        # Normalize to consensus coordinates
        l1_pos_norm = l1_pos / row['te_length'] * L1_CONSENSUS_LEN

        # Keep only positions within the element
        if 0 <= l1_pos_norm <= L1_CONSENSUS_LEN:
            psi_l1_positions.append(l1_pos_norm)

    return psi_l1_positions


def read_pos_to_l1_pos_m6a(row):
    """Same for m6A positions."""
    m6a_l1_positions = []
    for p in row['m6a_positions']:
        if row['read_strand'] == '+':
            genomic_pos = row['start'] + p
            l1_pos = genomic_pos - row['te_start']
        else:
            genomic_pos = row['end'] - p
            l1_pos = row['te_end'] - genomic_pos

        l1_pos_norm = l1_pos / row['te_length'] * L1_CONSENSUS_LEN
        if 0 <= l1_pos_norm <= L1_CONSENSUS_LEN:
            m6a_l1_positions.append(l1_pos_norm)

    return m6a_l1_positions


fl['psi_l1_pos'] = fl.apply(read_pos_to_l1_pos, axis=1)
fl['m6a_l1_pos'] = fl.apply(read_pos_to_l1_pos_m6a, axis=1)

# Collect all psi positions
all_psi_pos = []
for _, row in fl.iterrows():
    all_psi_pos.extend(row['psi_l1_pos'])
all_psi_pos = np.array(all_psi_pos)
print(f"Total psi sites mapped: {len(all_psi_pos):,}")

all_m6a_pos = []
for _, row in fl.iterrows():
    all_m6a_pos.extend(row['m6a_l1_pos'])
all_m6a_pos = np.array(all_m6a_pos)
print(f"Total m6A sites mapped: {len(all_m6a_pos):,}")

# Also collect read coverage across L1 (to normalize)
# For each read, which L1 positions does it cover?
coverage_bins = np.zeros(200)  # 200 bins across L1
BIN_SIZE = L1_CONSENSUS_LEN / 200

for _, row in fl.iterrows():
    if row['read_strand'] == '+':
        l1_start = (row['start'] - row['te_start']) / row['te_length'] * L1_CONSENSUS_LEN
        l1_end = (row['end'] - row['te_start']) / row['te_length'] * L1_CONSENSUS_LEN
    else:
        l1_start = (row['te_end'] - row['end']) / row['te_length'] * L1_CONSENSUS_LEN
        l1_end = (row['te_end'] - row['start']) / row['te_length'] * L1_CONSENSUS_LEN

    l1_start = max(0, l1_start)
    l1_end = min(L1_CONSENSUS_LEN, l1_end)

    bin_start = int(l1_start / BIN_SIZE)
    bin_end = int(l1_end / BIN_SIZE)
    bin_start = max(0, min(bin_start, 199))
    bin_end = max(0, min(bin_end, 199))
    coverage_bins[bin_start:bin_end+1] += 1

# =========================================================================
# Domain-level analysis
# =========================================================================
print("\n=== Psi distribution across L1 domains ===")

def classify_domain(pos):
    for name, start, end in DOMAINS:
        if start <= pos < end:
            return name
    return 'outside'

psi_domains = [classify_domain(p) for p in all_psi_pos]
m6a_domains = [classify_domain(p) for p in all_m6a_pos]

# Count psi per domain
print(f"\n{'Domain':10s} {'Psi sites':>10s} {'%':>6s} {'m6A sites':>10s} {'%':>6s} {'Domain bp':>10s} {'% of L1':>8s}")
for name, start, end in DOMAINS:
    n_psi = sum(1 for d in psi_domains if d == name)
    n_m6a = sum(1 for d in m6a_domains if d == name)
    domain_len = end - start
    domain_frac = domain_len / L1_CONSENSUS_LEN
    psi_pct = n_psi / len(all_psi_pos) * 100 if len(all_psi_pos) > 0 else 0
    m6a_pct = n_m6a / len(all_m6a_pos) * 100 if len(all_m6a_pos) > 0 else 0
    print(f"{name:10s} {n_psi:10d} {psi_pct:5.1f}% {n_m6a:10d} {m6a_pct:5.1f}% {domain_len:10d} {domain_frac*100:7.1f}%")

# Coverage-normalized (per-read psi density per domain)
print(f"\n=== Coverage-normalized psi density per domain ===")
domain_results = []
for name, d_start, d_end in DOMAINS:
    domain_len = d_end - d_start
    # Count reads covering this domain
    n_reads_covering = 0
    total_psi_in_domain = 0
    total_m6a_in_domain = 0
    for _, row in fl.iterrows():
        # Check if read covers this domain
        if row['read_strand'] == '+':
            l1_start = (row['start'] - row['te_start']) / row['te_length'] * L1_CONSENSUS_LEN
            l1_end = (row['end'] - row['te_start']) / row['te_length'] * L1_CONSENSUS_LEN
        else:
            l1_start = (row['te_end'] - row['end']) / row['te_length'] * L1_CONSENSUS_LEN
            l1_end = (row['te_end'] - row['start']) / row['te_length'] * L1_CONSENSUS_LEN

        # Overlap with domain
        overlap_start = max(l1_start, d_start)
        overlap_end = min(l1_end, d_end)
        if overlap_end > overlap_start:
            overlap_len = overlap_end - overlap_start
            # Count as covering if >50% of domain or >200bp overlap
            if overlap_len > min(domain_len * 0.5, 200):
                n_reads_covering += 1
                psi_in = sum(1 for p in row['psi_l1_pos'] if d_start <= p < d_end)
                m6a_in = sum(1 for p in row['m6a_l1_pos'] if d_start <= p < d_end)
                total_psi_in_domain += psi_in
                total_m6a_in_domain += m6a_in

    psi_per_kb = total_psi_in_domain / (n_reads_covering * domain_len / 1000) if n_reads_covering > 0 else 0
    m6a_per_kb = total_m6a_in_domain / (n_reads_covering * domain_len / 1000) if n_reads_covering > 0 else 0

    domain_results.append({
        'domain': name, 'domain_start': d_start, 'domain_end': d_end,
        'domain_len': domain_len,
        'n_reads_covering': n_reads_covering,
        'psi_total': total_psi_in_domain, 'm6a_total': total_m6a_in_domain,
        'psi_per_kb': psi_per_kb, 'm6a_per_kb': m6a_per_kb,
    })
    print(f"  {name:10s}: n_reads={n_reads_covering:5d}, psi/kb={psi_per_kb:.2f}, m6a/kb={m6a_per_kb:.2f}")

domain_df = pd.DataFrame(domain_results)

# =========================================================================
# By subfamily
# =========================================================================
print("\n=== Per-subfamily domain density ===")
for sub in ['L1HS', 'L1PA2', 'L1PA3']:
    sub_fl = fl[fl['gene_id'] == sub]
    if len(sub_fl) < 10:
        continue
    sub_psi = []
    for _, row in sub_fl.iterrows():
        sub_psi.extend(row['psi_l1_pos'])
    sub_psi = np.array(sub_psi)

    print(f"\n  {sub} (n={len(sub_fl)} reads, {len(sub_psi)} psi sites):")
    for name, d_start, d_end in DOMAINS:
        n = sum(1 for p in sub_psi if d_start <= p < d_end)
        pct = n / len(sub_psi) * 100 if len(sub_psi) > 0 else 0
        domain_frac = (d_end - d_start) / L1_CONSENSUS_LEN * 100
        enrichment = (pct / domain_frac) if domain_frac > 0 else 0
        print(f"    {name:10s}: {n:5d} sites ({pct:5.1f}%), domain={domain_frac:.1f}%, enrichment={enrichment:.2f}x")

# =========================================================================
# Figure: Psi distribution across young L1 structure
# =========================================================================
print("\nGenerating figure...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 3, 1.5]})

# Color scheme for domains
domain_colors = {
    "5'UTR": '#ff9999',
    "ORF1": '#66b3ff',
    "spacer": '#cccccc',
    "ORF2": '#99ff99',
    "3'UTR": '#ffcc99',
}

# --- Panel A: Psi density histogram with domain annotation ---
ax = axes[0]
bins_pos = np.linspace(0, L1_CONSENSUS_LEN, 201)
bin_centers = (bins_pos[:-1] + bins_pos[1:]) / 2

# Histogram of psi positions
psi_hist, _ = np.histogram(all_psi_pos, bins=bins_pos)
# Normalize by read coverage at each bin
psi_density = np.where(coverage_bins > 0, psi_hist / coverage_bins, 0)
# Smooth with rolling average
window = 5
psi_smooth = np.convolve(psi_density, np.ones(window)/window, mode='same')

ax.fill_between(bin_centers, psi_smooth, alpha=0.3, color='#d62728')
ax.plot(bin_centers, psi_smooth, color='#d62728', linewidth=1.5, label='Psi density')

# m6A overlay
m6a_hist, _ = np.histogram(all_m6a_pos, bins=bins_pos)
m6a_density = np.where(coverage_bins > 0, m6a_hist / coverage_bins, 0)
m6a_smooth = np.convolve(m6a_density, np.ones(window)/window, mode='same')
ax.plot(bin_centers, m6a_smooth, color='#1f77b4', linewidth=1.5, alpha=0.7, label='m6A density')

# Domain background
for name, d_start, d_end in DOMAINS:
    ax.axvspan(d_start, d_end, alpha=0.1, color=domain_colors[name])

ax.set_ylabel('Modification density\n(sites per read per bin)', fontsize=11)
ax.set_title('A. Psi & m6A Distribution Across Young L1 (Full-Length, >5kb)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0, L1_CONSENSUS_LEN)

# Domain labels
for name, d_start, d_end in DOMAINS:
    mid = (d_start + d_end) / 2
    ax.text(mid, ax.get_ylim()[1] * 0.95, name, ha='center', va='top', fontsize=8,
            fontweight='bold', color='gray')

# --- Panel B: Read coverage ---
ax = axes[1]
ax.fill_between(bin_centers, coverage_bins, alpha=0.4, color='#888888')
ax.plot(bin_centers, coverage_bins, color='#555555', linewidth=1)

for name, d_start, d_end in DOMAINS:
    ax.axvspan(d_start, d_end, alpha=0.1, color=domain_colors[name])

ax.set_ylabel('Read coverage\n(n reads)', fontsize=11)
ax.set_title('B. DRS Read Coverage Across L1', fontsize=12, fontweight='bold')
ax.set_xlim(0, L1_CONSENSUS_LEN)

# --- Panel C: Domain-level bar ---
ax = axes[2]
domain_names = [d['domain'] for d in domain_results]
psi_densities = [d['psi_per_kb'] for d in domain_results]
m6a_densities = [d['m6a_per_kb'] for d in domain_results]

x = np.arange(len(domain_names))
w = 0.35
bars1 = ax.bar(x - w/2, psi_densities, w, color='#d62728', alpha=0.7, label='Psi/kb')
bars2 = ax.bar(x + w/2, m6a_densities, w, color='#1f77b4', alpha=0.7, label='m6A/kb')

for bar, val in zip(bars1, psi_densities):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.1f}', ha='center', fontsize=8)
for bar, val in zip(bars2, m6a_densities):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.1f}', ha='center', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(domain_names, fontsize=10)
ax.set_ylabel('Sites/kb', fontsize=11)
ax.set_title('C. Coverage-Normalized Modification Density per Domain', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUTDIR / 'young_l1_psi_position.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Figure saved: {OUTDIR / 'young_l1_psi_position.png'}")

# =========================================================================
# Save TSV
# =========================================================================
domain_df.to_csv(OUTDIR / 'young_l1_domain_density.tsv', sep='\t', index=False)

# Per-position density for detailed analysis
pos_df = pd.DataFrame({
    'bin_center': bin_centers,
    'psi_count': psi_hist,
    'm6a_count': m6a_hist.astype(int) if len(all_m6a_pos) > 0 else np.zeros(200, dtype=int),
    'coverage': coverage_bins.astype(int),
    'psi_density': psi_density,
    'm6a_density': m6a_density,
})
pos_df.to_csv(OUTDIR / 'young_l1_position_density.tsv', sep='\t', index=False)

# =========================================================================
# Statistical tests: domain enrichment
# =========================================================================
print("\n=== Domain enrichment tests ===")
# For each domain, test if psi density is higher/lower than ORF2 (largest domain, reference)
orf2_density = domain_df[domain_df['domain']=='ORF2']['psi_per_kb'].values[0]
print(f"\nReference: ORF2 psi/kb = {orf2_density:.2f}")
for _, row in domain_df.iterrows():
    if row['domain'] == 'ORF2' or row['domain'] == 'spacer':
        continue
    ratio = row['psi_per_kb'] / orf2_density if orf2_density > 0 else 0
    print(f"  {row['domain']:10s}: psi/kb={row['psi_per_kb']:.2f}, ratio vs ORF2={ratio:.2f}x")

# Chi-square: observed vs expected (proportional to domain length)
print("\n=== Chi-square: psi distribution vs uniform ===")
obs = []
exp = []
domain_names_chi = []
for name, d_start, d_end in DOMAINS:
    if name == 'spacer':
        continue  # too small
    n = sum(1 for p in all_psi_pos if d_start <= p < d_end)
    expected = len(all_psi_pos) * (d_end - d_start) / L1_CONSENSUS_LEN
    obs.append(n)
    exp.append(expected)
    domain_names_chi.append(name)
    print(f"  {name:10s}: observed={n:5d}, expected={expected:.0f}, O/E={n/expected:.2f}")

# Normalize expected to match observed sum (spacer excluded)
exp_arr = np.array(exp, dtype=float)
exp_arr = exp_arr * (sum(obs) / sum(exp_arr))
chi2, p_chi = stats.chisquare(obs, exp_arr)
print(f"  Chi-square: {chi2:.1f}, p = {p_chi:.2e}")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Young L1 full-length reads: {len(fl):,}")
print(f"Psi sites mapped: {len(all_psi_pos):,}")
print(f"m6A sites mapped: {len(all_m6a_pos):,}")
print(f"\nDomain psi density (coverage-normalized):")
for _, row in domain_df.iterrows():
    print(f"  {row['domain']:10s}: {row['psi_per_kb']:.2f} sites/kb (n_reads={row['n_reads_covering']})")
print(f"\nOutput: {OUTDIR}")
print("Done!")
