#!/usr/bin/env python3
"""L1 modification motif frequency enrichment & conservation analysis.

Analysis 1: Motif frequency enrichment
  - Count m6A (DRACH) and psi motifs per kb in L1 sequences
  - Compare with non-L1 intronic/intergenic sequences (size-matched)
  - Also compute expected motif frequency from base composition (null model)

Analysis 2: Conservation (PhyloP) at motif positions
  - Within L1 elements: compare PhyloP scores at modification motif positions
    vs non-motif positions
  - If motif positions are more conserved → purifying selection to maintain
    modification potential

Usage: conda run -n research python l1_motif_enrichment_conservation.py
"""
import os
import re
import random
from bisect import bisect_left, bisect_right
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyBigWig
import pysam
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
REF_FASTA = BASE / 'reference/Human.fasta'
L1_TE_BED = BASE / 'reference/L1_TE_L1_family.bed'
PHYLOP_BW = Path('/qbio/junsoopablo/02_Projects/06_Metabolic_labeling/reference/hg38.phyloP30way.bw')
OUT_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/l1_motif_conservation'

# m6A DRACH motifs: D={A,G,T}, R={A,G}, H={A,C,T}
M6A_DRACH = set()
for d in 'AGT':
    for r in 'AG':
        for h in 'ACT':
            M6A_DRACH.add(d + r + 'A' + 'C' + h)

# Psi motifs (from MAFIA classifier)
PSI_MOTIFS = {
    'GTTCA', 'GTTCC', 'GTTCG', 'GTTCT',
    'AGTGG', 'GGTGG', 'TGTGG',
    'TGTAG', 'GGTCC',
    'CATAA', 'TATAA',
    'CATCC', 'CTTTA', 'ATTTG', 'GATGC', 'CCTCC',
}

ALL_MOTIFS = M6A_DRACH | PSI_MOTIFS
CHROMS = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
CHROM_SET = set(CHROMS)

random.seed(42)
np.random.seed(42)

# Precompute reverse complements
RC_TABLE = str.maketrans('ACGT', 'TGCA')


def revcomp(s):
    return s[::-1].translate(RC_TABLE)


# Build combined motif set including reverse complements for fast lookup
def build_motif_lookup(motif_set):
    """Build motif set that includes reverse complements."""
    full = set(motif_set)
    for m in motif_set:
        full.add(revcomp(m))
    return full


def load_l1_regions():
    """Load L1 TE BED, filter to standard chroms."""
    regions = []
    with open(L1_TE_BED) as f:
        for line in f:
            fields = line.strip().split('\t')
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if chrom in CHROM_SET and end - start >= 50:
                subfamily = fields[3] if len(fields) > 3 else ''
                regions.append((chrom, start, end, subfamily))
    return regions


def build_l1_index(l1_regions):
    """Build sorted interval index for fast overlap checking."""
    idx = defaultdict(list)
    for chrom, start, end, _ in l1_regions:
        idx[chrom].append((start, end))
    for chrom in idx:
        idx[chrom].sort()
    return idx


def overlaps_l1(chrom, start, end, l1_idx):
    """Check if region overlaps any L1 using binary search."""
    intervals = l1_idx.get(chrom, [])
    if not intervals:
        return False
    # Find first interval that could overlap (start < end of query)
    i = bisect_right([iv[0] for iv in intervals], end) - 1
    # Check backwards from there
    while i >= 0 and intervals[i][0] < end:
        if intervals[i][1] > start:
            return True
        i -= 1
    # Also check forward
    i = bisect_left([iv[0] for iv in intervals], start)
    if i < len(intervals) and intervals[i][0] < end:
        return True
    return False


def sample_non_l1_regions(l1_regions, fai_path, l1_idx, n_sample=50000):
    """Sample random non-L1 regions with similar size distribution."""
    chrom_sizes = {}
    with open(fai_path) as f:
        for line in f:
            fields = line.strip().split('\t')
            chrom = fields[0]
            if chrom in CHROM_SET:
                chrom_sizes[chrom] = int(fields[1])

    sizes = [end - start for _, start, end, _ in l1_regions]
    size_sample = random.choices(sizes, k=n_sample)

    chroms_list = list(chrom_sizes.keys())
    chrom_weights = [chrom_sizes[c] for c in chroms_list]

    non_l1 = []
    attempts = 0
    while len(non_l1) < n_sample and attempts < n_sample * 10:
        chrom = random.choices(chroms_list, weights=chrom_weights, k=1)[0]
        size = size_sample[len(non_l1) % len(size_sample)]
        max_start = chrom_sizes[chrom] - size
        if max_start <= 0:
            attempts += 1
            continue
        start = random.randint(0, max_start)
        end = start + size
        if not overlaps_l1(chrom, start, end, l1_idx):
            non_l1.append((chrom, start, end))
        attempts += 1

    return non_l1


def count_all_motifs_batch(fasta, regions, motif_sets_dict, max_regions=None):
    """Count all motif occurrences in one pass through sequences.

    motif_sets_dict: {'m6A_DRACH': set_of_motifs, 'psi': set_of_motifs}
    Returns dict of {motif_name: {motif_pattern: (total_count, total_bp)}}
    and per-region densities.
    """
    # Build lookup: each 5mer -> list of (set_name, original_motif)
    kmer_to_sets = defaultdict(list)
    for set_name, motif_set in motif_sets_dict.items():
        for motif in motif_set:
            kmer_to_sets[motif].append((set_name, motif))
            rc = revcomp(motif)
            if rc != motif:
                kmer_to_sets[rc].append((set_name, motif))

    # Per-set totals
    set_total_count = {name: 0 for name in motif_sets_dict}
    set_total_bp = {name: 0 for name in motif_sets_dict}
    set_per_region = {name: [] for name in motif_sets_dict}

    # Per-individual-motif totals
    indiv_count = {}  # (set_name, motif) -> count
    indiv_bp = {}
    for set_name, motif_set in motif_sets_dict.items():
        for motif in motif_set:
            indiv_count[(set_name, motif)] = 0
            indiv_bp[(set_name, motif)] = 0

    if max_regions and len(regions) > max_regions:
        regions = regions[:max_regions]

    for idx, region in enumerate(regions):
        chrom, start, end = region[0], region[1], region[2]
        try:
            seq = fasta.fetch(chrom, start, end).upper()
        except Exception:
            continue
        if len(seq) < 5:
            continue

        # Count per set for this region
        region_counts = {name: 0 for name in motif_sets_dict}
        # Count per individual motif for this region
        region_indiv = defaultdict(int)

        for i in range(len(seq) - 4):
            kmer = seq[i:i+5]
            if kmer in kmer_to_sets:
                for set_name, orig_motif in kmer_to_sets[kmer]:
                    region_counts[set_name] += 1
                    region_indiv[(set_name, orig_motif)] += 1

        seq_len = len(seq)
        for name in motif_sets_dict:
            set_total_count[name] += region_counts[name]
            set_total_bp[name] += seq_len
            density = region_counts[name] / (seq_len / 1000) if seq_len > 0 else 0
            set_per_region[name].append(density)

        for key, cnt in region_indiv.items():
            indiv_count[key] += cnt
            indiv_bp[key] += seq_len

        if (idx + 1) % 50000 == 0:
            print(f"    Processed {idx+1:,} regions...")

    # Compute densities
    set_density = {}
    for name in motif_sets_dict:
        bp = set_total_bp[name]
        set_density[name] = set_total_count[name] / (bp / 1000) if bp > 0 else 0

    indiv_density = {}
    for key in indiv_count:
        bp = indiv_bp[key]
        indiv_density[key] = indiv_count[key] / (bp / 1000) if bp > 0 else 0

    return {
        'set_density': set_density,
        'set_per_region': {name: np.array(v) for name, v in set_per_region.items()},
        'set_total': set_total_count,
        'set_bp': set_total_bp,
        'indiv_density': indiv_density,
    }


def expected_motif_freq(base_comp, motif_set):
    """Expected motif frequency per kb given base composition."""
    expected = 0
    for motif in motif_set:
        prob = 1.0
        for base in motif:
            prob *= base_comp.get(base, 0.25)
        expected += prob
    # Also add reverse complement
    for motif in motif_set:
        rc = revcomp(motif)
        if rc != motif:
            prob = 1.0
            for base in rc:
                prob *= base_comp.get(base, 0.25)
            expected += prob
    return expected * 1000


def get_base_composition(fasta, regions, max_bp=50_000_000):
    """Get base composition from regions."""
    counts = Counter()
    bp = 0
    for region in regions:
        if bp >= max_bp:
            break
        chrom, start, end = region[0], region[1], region[2]
        try:
            seq = fasta.fetch(chrom, start, end).upper()
        except Exception:
            continue
        counts.update(seq)
        bp += len(seq)

    total = sum(counts[b] for b in 'ACGT')
    return {b: counts[b] / total for b in 'ACGT'} if total > 0 else {}


def phylop_at_motif_sites(bw, fasta, regions, motif_set, max_regions=20000):
    """Get PhyloP scores at motif positions vs non-motif positions within regions.

    Uses subsampling of non-motif positions to keep memory manageable.
    """
    full_motifs = build_motif_lookup(motif_set)
    motif_scores = []
    nonmotif_scores = []

    sampled = random.sample(regions, min(max_regions, len(regions)))

    for idx, region in enumerate(sampled):
        chrom, start, end = region[0], region[1], region[2]
        try:
            seq = fasta.fetch(chrom, start, end).upper()
            scores = bw.values(chrom, start, end)
        except Exception:
            continue

        if scores is None or len(scores) != len(seq):
            continue

        scores_arr = np.array(scores, dtype=np.float32)
        valid = ~np.isnan(scores_arr)

        # Find motif positions
        is_motif = np.zeros(len(seq), dtype=bool)
        for i in range(len(seq) - 4):
            kmer = seq[i:i+5]
            if kmer in full_motifs:
                is_motif[i:i+5] = True

        motif_mask = is_motif & valid
        nonmotif_mask = ~is_motif & valid

        if motif_mask.any():
            motif_scores.append(scores_arr[motif_mask])
        if nonmotif_mask.any():
            # Subsample non-motif to avoid memory explosion
            nm_scores = scores_arr[nonmotif_mask]
            if len(nm_scores) > 500:
                nm_scores = np.random.choice(nm_scores, 500, replace=False)
            nonmotif_scores.append(nm_scores)

        if (idx + 1) % 5000 == 0:
            print(f"    PhyloP: processed {idx+1:,}/{len(sampled):,} regions...")

    if motif_scores:
        motif_scores = np.concatenate(motif_scores)
    else:
        motif_scores = np.array([])
    if nonmotif_scores:
        nonmotif_scores = np.concatenate(nonmotif_scores)
    else:
        nonmotif_scores = np.array([])

    return motif_scores, nonmotif_scores


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    fasta = pysam.FastaFile(str(REF_FASTA))
    bw = pyBigWig.open(str(PHYLOP_BW))
    l1_regions = load_l1_regions()
    print(f"  L1 regions: {len(l1_regions):,}")

    l1_idx = build_l1_index(l1_regions)

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 1: Motif frequency enrichment
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ANALYSIS 1: Motif frequency enrichment (L1 vs non-L1)")
    print("="*70)

    # Sample non-L1 regions
    print("  Sampling non-L1 regions...")
    non_l1_regions = sample_non_l1_regions(l1_regions, str(REF_FASTA) + '.fai', l1_idx)
    print(f"  Non-L1 regions sampled: {len(non_l1_regions):,}")

    # Base composition
    print("  Computing base composition...")
    l1_comp = get_base_composition(fasta, l1_regions)
    non_l1_comp = get_base_composition(fasta, non_l1_regions)
    print(f"  L1 base comp: A={l1_comp.get('A',0):.3f} C={l1_comp.get('C',0):.3f} "
          f"G={l1_comp.get('G',0):.3f} T={l1_comp.get('T',0):.3f}")
    print(f"  Non-L1 comp:  A={non_l1_comp.get('A',0):.3f} C={non_l1_comp.get('C',0):.3f} "
          f"G={non_l1_comp.get('G',0):.3f} T={non_l1_comp.get('T',0):.3f}")

    # Count all motifs in one pass
    motif_sets = {'m6A_DRACH': M6A_DRACH, 'psi': PSI_MOTIFS}

    print("\n  Counting motifs in L1 regions (single pass)...")
    l1_results = count_all_motifs_batch(fasta, l1_regions, motif_sets, max_regions=200000)

    print("  Counting motifs in non-L1 regions (single pass)...")
    nl1_results = count_all_motifs_batch(fasta, non_l1_regions, motif_sets)

    results = {}
    for motif_name, motif_set in motif_sets.items():
        l1_density = l1_results['set_density'][motif_name]
        non_l1_density = nl1_results['set_density'][motif_name]
        l1_expected = expected_motif_freq(l1_comp, motif_set)
        non_l1_expected = expected_motif_freq(non_l1_comp, motif_set)

        l1_arr = l1_results['set_per_region'][motif_name]
        non_l1_arr = nl1_results['set_per_region'][motif_name]
        stat, pval = stats.mannwhitneyu(l1_arr, non_l1_arr, alternative='two-sided')

        enrichment = l1_density / non_l1_density if non_l1_density > 0 else float('inf')
        obs_exp_l1 = l1_density / l1_expected if l1_expected > 0 else float('inf')
        obs_exp_nonl1 = non_l1_density / non_l1_expected if non_l1_expected > 0 else float('inf')

        print(f"\n  --- {motif_name} motifs ({len(motif_set)} patterns) ---")
        print(f"  L1:     {l1_density:.2f} motifs/kb (obs/exp={obs_exp_l1:.2f}x)")
        print(f"  Non-L1: {non_l1_density:.2f} motifs/kb (obs/exp={obs_exp_nonl1:.2f}x)")
        print(f"  Enrichment (L1/non-L1): {enrichment:.2f}x")
        print(f"  MW p-value: {pval:.2e}")
        print(f"  L1 median per-region: {np.median(l1_arr):.2f}/kb")
        print(f"  Non-L1 median per-region: {np.median(non_l1_arr):.2f}/kb")

        results[motif_name] = {
            'l1_density': l1_density, 'non_l1_density': non_l1_density,
            'enrichment': enrichment, 'pval': pval,
            'l1_obs_exp': obs_exp_l1, 'non_l1_obs_exp': obs_exp_nonl1,
            'l1_per_region': l1_arr, 'non_l1_per_region': non_l1_arr,
        }

    # Individual motif breakdown (already computed in batch)
    print("\n  --- Individual motif frequencies (per kb) ---")
    indiv_results = []
    for set_name, motif_set in motif_sets.items():
        for motif in sorted(motif_set):
            key = (set_name, motif)
            l1_d = l1_results['indiv_density'].get(key, 0)
            nl_d = nl1_results['indiv_density'].get(key, 0)
            enrich = l1_d / nl_d if nl_d > 0 else float('inf')
            indiv_results.append({
                'mod_type': set_name, 'motif': motif,
                'l1_per_kb': l1_d, 'non_l1_per_kb': nl_d,
                'enrichment': enrich,
            })

    indiv_df = pd.DataFrame(indiv_results)
    indiv_df.to_csv(OUT_DIR / 'individual_motif_frequency.tsv', sep='\t', index=False)

    for mod in ['m6A_DRACH', 'psi']:
        mdf = indiv_df[indiv_df['mod_type'] == mod].sort_values('enrichment', ascending=False)
        print(f"\n  {mod} top enriched motifs in L1:")
        print(f"  {'Motif':<8} {'L1/kb':>7} {'Non-L1/kb':>9} {'Enrichment':>10}")
        for _, row in mdf.head(10).iterrows():
            print(f"  {row['motif']:<8} {row['l1_per_kb']:>6.2f} {row['non_l1_per_kb']:>8.2f} "
                  f"{row['enrichment']:>9.2f}x")

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 2: Conservation at motif positions within L1
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ANALYSIS 2: PhyloP conservation at motif vs non-motif positions within L1")
    print("="*70)

    # Compute PhyloP once per motif type and cache
    phylop_cache = {}
    conservation_results = []

    for motif_name, motif_set in [('m6A_DRACH', M6A_DRACH), ('psi', PSI_MOTIFS),
                                   ('combined', ALL_MOTIFS)]:
        print(f"\n  --- {motif_name} ---")
        motif_scores, nonmotif_scores = phylop_at_motif_sites(
            bw, fasta, l1_regions, motif_set, max_regions=20000)
        phylop_cache[motif_name] = (motif_scores, nonmotif_scores)

        if len(motif_scores) == 0 or len(nonmotif_scores) == 0:
            print("  No data")
            continue

        stat, pval = stats.mannwhitneyu(motif_scores, nonmotif_scores, alternative='two-sided')

        print(f"  Motif positions:     n={len(motif_scores):,}, "
              f"mean={motif_scores.mean():.4f}, median={np.median(motif_scores):.4f}")
        print(f"  Non-motif positions: n={len(nonmotif_scores):,}, "
              f"mean={nonmotif_scores.mean():.4f}, median={np.median(nonmotif_scores):.4f}")
        print(f"  Delta (motif - non-motif): {motif_scores.mean() - nonmotif_scores.mean():.4f}")
        print(f"  MW p-value: {pval:.2e}")

        motif_conserved = (motif_scores > 0).mean()
        nonmotif_conserved = (nonmotif_scores > 0).mean()
        print(f"  Fraction conserved (PhyloP > 0): motif={motif_conserved:.3f}, "
              f"non-motif={nonmotif_conserved:.3f}")

        conservation_results.append({
            'motif_type': motif_name,
            'n_motif': len(motif_scores), 'n_nonmotif': len(nonmotif_scores),
            'mean_motif': motif_scores.mean(), 'mean_nonmotif': nonmotif_scores.mean(),
            'median_motif': np.median(motif_scores), 'median_nonmotif': np.median(nonmotif_scores),
            'delta': motif_scores.mean() - nonmotif_scores.mean(),
            'pval': pval,
            'frac_conserved_motif': motif_conserved,
            'frac_conserved_nonmotif': nonmotif_conserved,
        })

    cons_df = pd.DataFrame(conservation_results)
    cons_df.to_csv(OUT_DIR / 'phylop_conservation.tsv', sep='\t', index=False)

    # Conservation by L1 age (young vs ancient)
    print("\n  --- Conservation by L1 age ---")
    YOUNG_SF = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
    young_regions = [(c, s, e, sf) for c, s, e, sf in l1_regions if sf in YOUNG_SF]
    ancient_regions = [(c, s, e, sf) for c, s, e, sf in l1_regions if sf not in YOUNG_SF]
    print(f"  Young L1: {len(young_regions):,}, Ancient L1: {len(ancient_regions):,}")

    age_phylop = {}
    for age_label, age_regions in [('young', young_regions), ('ancient', ancient_regions)]:
        ms, nms = phylop_at_motif_sites(bw, fasta, age_regions, ALL_MOTIFS, max_regions=20000)
        age_phylop[age_label] = (ms, nms)
        if len(ms) == 0:
            continue
        delta = ms.mean() - nms.mean()
        stat, pval = stats.mannwhitneyu(ms, nms, alternative='two-sided')
        print(f"  {age_label}: motif PhyloP={ms.mean():.4f}, non-motif={nms.mean():.4f}, "
              f"delta={delta:+.4f}, p={pval:.2e}")

    # ═══════════════════════════════════════════════════════════
    # FIGURES (using cached data)
    # ═══════════════════════════════════════════════════════════
    print("\nGenerating figures...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fig A: Motif frequency enrichment
    ax = axes[0, 0]
    motif_types = ['m6A_DRACH', 'psi']
    x = np.arange(len(motif_types))
    w = 0.35
    l1_vals = [results[m]['l1_density'] for m in motif_types]
    nl1_vals = [results[m]['non_l1_density'] for m in motif_types]
    bars1 = ax.bar(x - w/2, l1_vals, w, label='L1', color='#e74c3c')
    bars2 = ax.bar(x + w/2, nl1_vals, w, label='Non-L1 (random)', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(['m6A (DRACH)', 'psi (16 motifs)'])
    ax.set_ylabel('Motifs per kb')
    ax.set_title('A. Modification Motif Frequency')
    ax.legend()
    for bar, val, enr in zip(bars1, l1_vals, [results[m]['enrichment'] for m in motif_types]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{enr:.2f}x', ha='center', va='bottom', fontsize=9)

    # Fig B: Top individual motif enrichment
    ax = axes[0, 1]
    top_motifs = indiv_df.sort_values('enrichment', ascending=False).head(15)
    colors = ['#e74c3c' if row['mod_type'] == 'm6A_DRACH' else '#9b59b6'
              for _, row in top_motifs.iterrows()]
    bars = ax.barh(range(len(top_motifs)), top_motifs['enrichment'], color=colors)
    ax.set_yticks(range(len(top_motifs)))
    ax.set_yticklabels(top_motifs['motif'], fontsize=8)
    ax.axvline(1, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Enrichment (L1 / non-L1)')
    ax.set_title('B. Top Enriched Motifs in L1')
    ax.invert_yaxis()

    # Fig C: PhyloP distribution at motif vs non-motif within L1 (from cache)
    ax = axes[1, 0]
    bins = np.arange(-3, 5, 0.2)
    for motif_name, color in [('m6A_DRACH', '#e74c3c'), ('psi', '#9b59b6')]:
        ms, _ = phylop_cache[motif_name]
        if len(ms) > 0:
            ax.hist(ms, bins=bins, alpha=0.4, density=True,
                    label=f'{motif_name} motif', color=color)
    _, nms_all = phylop_cache['combined']
    if len(nms_all) > 0:
        ax.hist(nms_all, bins=bins, alpha=0.4, density=True,
                label='Non-motif L1', color='#95a5a6')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('PhyloP 30way score')
    ax.set_ylabel('Density')
    ax.set_title('C. Conservation within L1: Motif vs Non-motif')
    ax.legend(fontsize=8)

    # Fig D: Conservation by L1 age (from cache)
    ax = axes[1, 1]
    age_data = []
    for age_label, color in [('young', '#e74c3c'), ('ancient', '#3498db')]:
        ms, nms = age_phylop[age_label]
        if len(ms) > 0:
            age_data.append((age_label.capitalize() + ' L1', ms.mean(), nms.mean(), color))

    if age_data:
        x = np.arange(len(age_data))
        w = 0.35
        ax.bar(x - w/2, [d[1] for d in age_data], w, label='Motif positions',
               color=[d[3] for d in age_data], alpha=0.7)
        ax.bar(x + w/2, [d[2] for d in age_data], w, label='Non-motif positions',
               color=[d[3] for d in age_data], alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([d[0] for d in age_data])
        ax.set_ylabel('Mean PhyloP score')
        ax.set_title('D. Conservation: Motif vs Non-motif by L1 Age')
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'motif_enrichment_conservation.png', dpi=150)
    plt.close()
    print(f"Figure saved: {OUT_DIR / 'motif_enrichment_conservation.png'}")

    bw.close()
    fasta.close()
    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
