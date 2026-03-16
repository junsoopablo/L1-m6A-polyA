#!/usr/bin/env python3
"""
Regulatory L1 DRACH motif × Disease Variant intersection analysis.

Steps:
1. Extract unique regulatory L1 loci (Enhancer/Promoter from ChromHMM)
2. Get reference sequences at those loci → find DRACH motif positions
3. Intersect DRACH positions with ClinVar pathogenic/likely-pathogenic variants
4. Intersect DRACH positions with GWAS catalog SNPs
5. Also check ALL L1 loci (not just regulatory) for comparison
6. Report results
"""

import pandas as pd
import numpy as np
import pysam
import gzip
import os
import re
from collections import defaultdict

# ── Paths ──
BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
CHROMHMM_FILE = f'{BASE}/analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv'
REF_GENOME = f'{BASE}/reference/Human.fasta'
OUTDIR = f'{BASE}/analysis/01_exploration/topic_09_disease_variants'
CLINVAR_VCF = f'{OUTDIR}/clinvar_grch38.vcf.gz'
GWAS_FILE = f'{OUTDIR}/gwas_catalog_full.tsv'

os.makedirs(OUTDIR, exist_ok=True)

# ── DRACH motif definition ──
# D = A/G/T (not C), R = A/G, A, C, H = A/C/T (not G)
DRACH_PATTERN = re.compile(r'(?=([AGT][AG]AC[ACT]))')

def find_drach_positions(seq, offset=0):
    """Find all DRACH motif positions in sequence. Returns list of (start, end, motif)."""
    positions = []
    for m in DRACH_PATTERN.finditer(seq.upper()):
        pos = m.start()
        motif = m.group(1)
        # The A in DRACH is at position 2 (0-indexed) — the m6A target site
        m6a_pos = offset + pos + 2  # genomic position of the A (m6A candidate)
        positions.append({
            'motif_start': offset + pos,
            'motif_end': offset + pos + 5,
            'm6a_pos': m6a_pos,
            'motif': motif
        })
    return positions


def parse_clinvar_vcf(vcf_path, chrom_positions):
    """Parse ClinVar VCF and check overlap with DRACH positions.
    chrom_positions: dict of {chr: set of positions}
    """
    hits = []
    n_pathogenic = 0
    n_total = 0

    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            n_total += 1
            fields = line.strip().split('\t')
            chrom = fields[0]
            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom
            pos = int(fields[1])  # 1-based
            ref = fields[3]
            alt = fields[4]
            info = fields[7]

            # Filter for pathogenic/likely pathogenic
            clnsig = ''
            for item in info.split(';'):
                if item.startswith('CLNSIG='):
                    clnsig = item[7:]
                    break

            is_pathogenic = any(s in clnsig.lower() for s in
                              ['pathogenic', 'likely_pathogenic'])

            if not is_pathogenic:
                continue
            n_pathogenic += 1

            # Check if this position overlaps any DRACH motif
            if chrom in chrom_positions:
                if pos in chrom_positions[chrom]:
                    # Get disease info
                    clndn = ''
                    for item in info.split(';'):
                        if item.startswith('CLNDN='):
                            clndn = item[6:]
                            break

                    hits.append({
                        'chr': chrom,
                        'pos': pos,
                        'ref': ref,
                        'alt': alt,
                        'clnsig': clnsig,
                        'disease': clndn,
                        'info_short': info[:200]
                    })

    print(f"  ClinVar: {n_total:,} total variants, {n_pathogenic:,} pathogenic/likely_pathogenic")
    return hits


def parse_gwas_catalog(gwas_path, chrom_positions):
    """Parse GWAS catalog and check overlap with DRACH positions."""
    hits = []

    try:
        df = pd.read_csv(gwas_path, sep='\t', low_memory=False,
                        on_bad_lines='skip')
    except Exception as e:
        print(f"  GWAS catalog parse error: {e}")
        return hits

    # Key columns
    chr_col = 'CHR_ID'
    pos_col = 'CHR_POS'
    trait_col = 'DISEASE/TRAIT'
    pval_col = 'P-VALUE'
    snp_col = 'SNPS'

    if chr_col not in df.columns:
        print(f"  Available columns: {df.columns[:20].tolist()}")
        return hits

    for _, row in df.iterrows():
        try:
            chrom = f"chr{row[chr_col]}"
            pos = int(float(row[pos_col]))
        except (ValueError, TypeError):
            continue

        if chrom in chrom_positions and pos in chrom_positions[chrom]:
            hits.append({
                'chr': chrom,
                'pos': pos,
                'snp': row.get(snp_col, ''),
                'trait': row.get(trait_col, ''),
                'pvalue': row.get(pval_col, ''),
                'study': row.get('STUDY', ''),
                'mapped_gene': row.get('MAPPED_GENE', '')
            })

    print(f"  GWAS catalog: {len(df):,} associations checked")
    return hits


def main():
    print("=" * 70)
    print("Regulatory L1 DRACH × Disease Variant Analysis")
    print("=" * 70)

    # ── Step 1: Load regulatory L1 loci ──
    print("\n[1] Loading ChromHMM-annotated L1 loci...")
    df = pd.read_csv(CHROMHMM_FILE, sep='\t')

    # Get ALL unique L1 loci
    all_loci = df.drop_duplicates(subset=['chr', 'start', 'end'])[['chr', 'start', 'end', 'gene_id', 'l1_age', 'chromhmm_group']].copy()
    print(f"  Total unique L1 loci: {len(all_loci):,}")

    # Get regulatory loci
    reg_loci = all_loci[all_loci['chromhmm_group'].isin(['Enhancer', 'Promoter'])].copy()
    print(f"  Regulatory L1 loci (Enhancer+Promoter): {len(reg_loci):,}")
    print(f"    Enhancer: {(reg_loci['chromhmm_group']=='Enhancer').sum():,}")
    print(f"    Promoter: {(reg_loci['chromhmm_group']=='Promoter').sum():,}")

    # ── Step 2: Extract reference sequences and find DRACH motifs ──
    print("\n[2] Extracting reference sequences and scanning DRACH motifs...")
    ref = pysam.FastaFile(REF_GENOME)

    # Extend loci by 50bp on each side to capture flanking DRACH motifs
    FLANK = 50

    # Store DRACH positions for each locus type
    results = {}

    for label, loci_df in [('regulatory', reg_loci), ('all', all_loci)]:
        drach_positions = []
        chrom_pos_set = defaultdict(set)  # for fast lookup
        n_motifs_total = 0

        for _, row in loci_df.iterrows():
            chrom = row['chr']
            start = max(0, int(row['start']) - FLANK)
            end = int(row['end']) + FLANK

            try:
                seq = ref.fetch(chrom, start, end)
            except (ValueError, KeyError):
                # Try without 'chr' prefix or with it
                alt_chrom = chrom.replace('chr', '') if chrom.startswith('chr') else f'chr{chrom}'
                try:
                    seq = ref.fetch(alt_chrom, start, end)
                except:
                    continue

            motifs = find_drach_positions(seq, offset=start)
            n_motifs_total += len(motifs)

            for m in motifs:
                drach_positions.append({
                    'chr': chrom,
                    'locus_start': int(row['start']),
                    'locus_end': int(row['end']),
                    'l1_subfamily': row['gene_id'],
                    'chromhmm': row['chromhmm_group'],
                    **m
                })
                # Store all 5 positions of the DRACH motif for variant lookup
                for p in range(m['motif_start'], m['motif_end']):
                    chrom_pos_set[chrom].add(p + 1)  # convert to 1-based for VCF

        results[label] = {
            'drach_positions': drach_positions,
            'chrom_pos_set': chrom_pos_set,
            'n_motifs': n_motifs_total,
            'n_loci': len(loci_df)
        }

        # Total base pairs covered
        total_bp = sum(chrom_pos_set_sizes := {c: len(p) for c, p in chrom_pos_set.items()}.values())
        print(f"  {label}: {n_motifs_total:,} DRACH motifs across {len(loci_df):,} loci")
        print(f"    Total DRACH-covered positions: {total_bp:,} bp (1-based, for VCF matching)")

    ref.close()

    # ── Step 3: Intersect with ClinVar ──
    print("\n[3] Intersecting with ClinVar pathogenic variants...")

    for label in ['regulatory', 'all']:
        print(f"\n  --- {label.upper()} L1 ---")
        if os.path.exists(CLINVAR_VCF):
            clinvar_hits = parse_clinvar_vcf(CLINVAR_VCF, results[label]['chrom_pos_set'])
            results[label]['clinvar_hits'] = clinvar_hits
            print(f"  ClinVar hits in DRACH motifs: {len(clinvar_hits)}")
            if clinvar_hits:
                for h in clinvar_hits[:20]:
                    print(f"    {h['chr']}:{h['pos']} {h['ref']}>{h['alt']} | {h['clnsig']} | {h['disease'][:80]}")
        else:
            print(f"  ClinVar VCF not found: {CLINVAR_VCF}")
            results[label]['clinvar_hits'] = []

    # ── Step 4: Intersect with GWAS catalog ──
    print("\n[4] Intersecting with GWAS catalog...")

    for label in ['regulatory', 'all']:
        print(f"\n  --- {label.upper()} L1 ---")
        if os.path.exists(GWAS_FILE):
            gwas_hits = parse_gwas_catalog(GWAS_FILE, results[label]['chrom_pos_set'])
            results[label]['gwas_hits'] = gwas_hits
            print(f"  GWAS hits in DRACH motifs: {len(gwas_hits)}")
            if gwas_hits:
                # Group by trait
                trait_counts = defaultdict(int)
                for h in gwas_hits:
                    trait_counts[h['trait']] += 1
                print(f"  Unique traits: {len(trait_counts)}")
                for trait, count in sorted(trait_counts.items(), key=lambda x: -x[1])[:20]:
                    print(f"    {trait}: {count} SNPs")
        else:
            print(f"  GWAS catalog not found: {GWAS_FILE}")
            results[label]['gwas_hits'] = []

    # ── Step 5: Summary statistics ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for label in ['regulatory', 'all']:
        r = results[label]
        print(f"\n{label.upper()} L1:")
        print(f"  Loci: {r['n_loci']:,}")
        print(f"  DRACH motifs: {r['n_motifs']:,}")
        print(f"  Motifs/locus: {r['n_motifs']/r['n_loci']:.1f}")
        print(f"  ClinVar pathogenic hits: {len(r.get('clinvar_hits', []))}")
        print(f"  GWAS catalog hits: {len(r.get('gwas_hits', []))}")

    # ── Step 6: Expected vs observed (baseline rate) ──
    print("\n\n[6] Expected overlap estimation...")
    # ClinVar has ~100K pathogenic/LP variants in ~3.1 Gbp genome
    # Our DRACH positions cover X bp
    # Expected = (pathogenic_count / genome_size) * our_bp

    genome_size = 3.1e9
    for label in ['regulatory', 'all']:
        r = results[label]
        total_bp = sum(len(p) for p in r['chrom_pos_set'].values())
        n_clinvar = len(r.get('clinvar_hits', []))
        # rough estimate
        expected_random = total_bp / genome_size * 100000  # assuming ~100K pathogenic variants
        print(f"\n  {label.upper()}: {total_bp:,} bp DRACH coverage")
        print(f"    Observed ClinVar hits: {n_clinvar}")
        print(f"    Expected by chance (rough): {expected_random:.2f}")
        if n_clinvar > 0 and expected_random > 0:
            print(f"    Enrichment: {n_clinvar / expected_random:.1f}x")

    # ── Step 7: Save results ──
    for label in ['regulatory', 'all']:
        r = results[label]
        # Save DRACH positions
        if r['drach_positions']:
            pd.DataFrame(r['drach_positions']).to_csv(
                f'{OUTDIR}/{label}_l1_drach_positions.tsv', sep='\t', index=False)

        # Save hits
        if r.get('clinvar_hits'):
            pd.DataFrame(r['clinvar_hits']).to_csv(
                f'{OUTDIR}/{label}_l1_clinvar_hits.tsv', sep='\t', index=False)
        if r.get('gwas_hits'):
            pd.DataFrame(r['gwas_hits']).to_csv(
                f'{OUTDIR}/{label}_l1_gwas_hits.tsv', sep='\t', index=False)

    print(f"\nResults saved to {OUTDIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
