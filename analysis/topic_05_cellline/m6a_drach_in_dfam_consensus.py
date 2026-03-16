#!/usr/bin/env python3
"""Count DRACH motifs in Dfam L1 consensus sequences by positional bins.

For each consensus sequence, divide into positional bins and count DRACH density.
Compare L1HS (young) vs L1M (oldest) vs L1MC/L1ME (intermediate).

Key question: Did L1M consensus ORIGINALLY have higher DRACH in ORF1 vs ORF2?

Dfam L1 structure:
  _5end: 5'UTR + ORF1 + beginning of ORF2 (L1HS: 2136bp covers 0-35% of full element)
  _orf2: middle/late ORF2
  _3end: end of ORF2 + 3'UTR

L1HS full consensus (6064bp):
  5'UTR: 1-910 (0-15%), ORF1: 911-1924 (15-32%), ORF2: 1991-5817 (33-96%), 3'UTR: 5818-6064 (96-100%)
  L1HS_5end = 2136bp ≈ first 35% = 5'UTR + ORF1 + ~200bp of ORF2
"""

import re
from pathlib import Path
from collections import defaultdict

CONS_DIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/l1_consensus_sequences')
DRACH_PATTERN = re.compile(r'(?=([AGT][AG]AC[ACT]))')

def read_fasta(fpath):
    """Read FASTA file, return dict of name -> sequence."""
    seqs = {}
    name = None
    seq_parts = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name:
                    seqs[name] = ''.join(seq_parts)
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
    if name:
        seqs[name] = ''.join(seq_parts)
    return seqs

def count_drach(seq):
    """Count DRACH motifs in sequence."""
    return len(DRACH_PATTERN.findall(seq))

def drach_per_kb(seq):
    """DRACH density per kb."""
    if len(seq) == 0:
        return 0
    return count_drach(seq) / len(seq) * 1000

# =====================================================================
# 1. Load all consensus sequences
# =====================================================================
all_seqs = {}
for fa_file in CONS_DIR.glob('*.fa'):
    all_seqs.update(read_fasta(fa_file))

print(f"Loaded {len(all_seqs)} consensus sequences\n")

# =====================================================================
# 2. L1HS full-length analysis (from custom_LINE_reference)
# =====================================================================
# Also load a few full-length L1 instances from custom_LINE_reference for comparison
CUSTOM_REF = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT/reference/custom_LINE_reference.fasta')

print("=" * 80)
print("SECTION 1: FULL-LENGTH L1 INSTANCES — DRACH/kb BY REGION")
print("=" * 80)

# Read full-length L1 instances from custom reference
# These are near-full-length genomic copies used as alignment references
custom_seqs = read_fasta(CUSTOM_REF)
print(f"\nLoaded {len(custom_seqs)} full-length L1 reference sequences")

# Group by subfamily prefix
from collections import Counter
subfam_counts = Counter()
for name in custom_seqs:
    subfam = name.split('_chr')[0]
    subfam_counts[subfam] += 1

# Show top subfamilies
print("\nSubfamily counts (top 20):")
for sf, cnt in subfam_counts.most_common(20):
    print(f"  {sf}: {cnt}")

# For full-length analysis, use sequences > 5000bp (near-full-length)
# Use L1HS boundaries as reference (scaled by actual sequence length)
L1HS_REGIONS = [
    ("5'UTR", 0, 0.15),
    ("ORF1", 0.15, 0.32),
    ("ORF2", 0.32, 0.96),
    ("3'UTR", 0.96, 1.0),
]

print(f"\n{'Subfamily':<12s} {'n_full':>6s} {'5UTR':>8s} {'ORF1':>8s} {'ORF2':>8s} {'3UTR':>8s} {'Overall':>8s}")
print("-" * 60)

target_subfams = ['L1HS', 'L1PA2', 'L1PA3', 'L1PA4', 'L1PA5', 'L1PA6', 'L1PA7',
                  'L1PB1', 'L1MC1', 'L1MC2', 'L1MC3', 'L1MC4',
                  'L1ME1', 'L1ME2', 'L1MEa',
                  'L1MA1', 'L1MA2', 'L1MA4', 'L1MA5', 'L1MA9']

for sf in target_subfams:
    # Get full-length sequences for this subfamily
    sf_seqs = {k: v for k, v in custom_seqs.items()
               if k.split('_chr')[0] == sf and len(v) >= 5000}
    if not sf_seqs:
        continue

    region_drach = defaultdict(list)
    for name, seq in sf_seqs.items():
        seq_len = len(seq)
        for label, lo, hi in L1HS_REGIONS:
            start = int(seq_len * lo)
            end = int(seq_len * hi)
            region_seq = seq[start:end]
            region_drach[label].append(drach_per_kb(region_seq))

    # Average across instances
    avg_vals = {}
    for label in ["5'UTR", "ORF1", "ORF2", "3'UTR"]:
        vals = region_drach.get(label, [])
        avg_vals[label] = sum(vals) / len(vals) if vals else 0

    overall_seqs = [drach_per_kb(seq) for seq in sf_seqs.values()]
    overall = sum(overall_seqs) / len(overall_seqs)

    utr5_v = avg_vals["5'UTR"]
    orf1_v = avg_vals['ORF1']
    orf2_v = avg_vals['ORF2']
    utr3_v = avg_vals["3'UTR"]
    print(f"  {sf:<12s} {len(sf_seqs):>6d} {utr5_v:>8.1f} {orf1_v:>8.1f} "
          f"{orf2_v:>8.1f} {utr3_v:>8.1f} {overall:>8.1f}")

# =====================================================================
# 3. Dfam consensus sequences — DRACH analysis
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 2: Dfam CONSENSUS SEQUENCES — DRACH/kb")
print("=" * 80)

print(f"\n{'Name':<20s} {'Length':>7s} {'Region':>7s} {'DRACH':>6s} {'DRACH/kb':>9s}")
print("-" * 55)

# Sort by known order
dfam_order = [
    # 5end sequences (5'UTR + ORF1 + ORF2 start)
    'L1HS_5end', 'L1P1_5end', 'L1MEa_5end',
    'L1M1_5end', 'L1M2_5end', 'L1M4_5end',
    # orf2 sequences (ORF2 middle)
    'L1M3_orf2', 'L1M5_orf2',
    # 3end sequences (ORF2 end + 3'UTR)
    'L1HS_3end', 'L1PA2_3end', 'L1PA3_3end', 'L1PA4_3end',
    'L1PA5_3end', 'L1PA6_3end', 'L1PA7_3end',
    'L1MA1_3end', 'L1MA5_3end', 'L1MA10_3end',
    'L1MC1_3end', 'L1MC2_3end', 'L1MC3_3end', 'L1MC4_3end',
    'L1ME1_3end', 'L1ME2_3end', 'L1ME3_3end',
]

for name in dfam_order:
    if name not in all_seqs:
        continue
    seq = all_seqs[name]
    n_drach = count_drach(seq)
    density = drach_per_kb(seq)
    region = name.split('_')[-1]
    print(f"  {name:<20s} {len(seq):>7d} {region:>7s} {n_drach:>6d} {density:>9.1f}")

# =====================================================================
# 4. L1HS 5end: divide into ORF1-region vs ORF2-region
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 3: L1HS_5end — ORF1 vs ORF2 PORTION")
print("=" * 80)

# L1HS_5end is 2136bp, covering ~first 35% of 6064bp full consensus
# Within this 2136bp:
#   5'UTR: 0-910 bp (0-42.6% of 5end)
#   ORF1: 911-1924 bp (42.6-90.1% of 5end)
#   ORF2 start: 1991-2136 bp (93.2-100% of 5end)

if 'L1HS_5end' in all_seqs:
    seq = all_seqs['L1HS_5end']
    print(f"\nL1HS_5end total: {len(seq)} bp, {count_drach(seq)} DRACH, {drach_per_kb(seq):.1f}/kb")

    # Exact L1HS boundaries
    utr5 = seq[0:910]
    orf1 = seq[910:1924]
    orf2_start = seq[1990:2136]

    print(f"  5'UTR (0-910):    {len(utr5)} bp, {count_drach(utr5)} DRACH, {drach_per_kb(utr5):.1f}/kb")
    print(f"  ORF1 (911-1924):  {len(orf1)} bp, {count_drach(orf1)} DRACH, {drach_per_kb(orf1):.1f}/kb")
    print(f"  ORF2 start (1991-2136): {len(orf2_start)} bp, {count_drach(orf2_start)} DRACH, {drach_per_kb(orf2_start):.1f}/kb")

# =====================================================================
# 5. L1M _5end: positional DRACH profile
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 4: L1M 5end CONSENSUS — POSITIONAL DRACH PROFILE")
print("=" * 80)

for name in ['L1HS_5end', 'L1M1_5end', 'L1M2_5end', 'L1M4_5end']:
    if name not in all_seqs:
        continue
    seq = all_seqs[name]
    seq_len = len(seq)
    print(f"\n{name} ({seq_len} bp):")

    # Divide into 10 equal bins
    bin_size = seq_len // 10
    for i in range(10):
        start = i * bin_size
        end = (i + 1) * bin_size if i < 9 else seq_len
        chunk = seq[start:end]
        d = drach_per_kb(chunk)
        bar = '█' * int(d / 2)
        pct_lo = start / seq_len * 100
        pct_hi = end / seq_len * 100
        print(f"  {pct_lo:5.1f}-{pct_hi:5.1f}% ({start:>4d}-{end:>4d}): {drach_per_kb(chunk):>6.1f}/kb {bar}")

# =====================================================================
# 6. Compare 5end first-half (≈5'UTR+ORF1) vs second-half (≈ORF2) across subfamilies
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 5: 5end FIRST-HALF vs SECOND-HALF DRACH/kb")
print("=" * 80)

print(f"\n{'Name':<20s} {'Length':>7s} {'1st half':>9s} {'2nd half':>9s} {'ratio':>7s}")
print("-" * 55)

for name in ['L1HS_5end', 'L1P1_5end', 'L1MEa_5end',
             'L1M1_5end', 'L1M2_5end', 'L1M4_5end']:
    if name not in all_seqs:
        continue
    seq = all_seqs[name]
    mid = len(seq) // 2
    first = seq[:mid]
    second = seq[mid:]
    d1 = drach_per_kb(first)
    d2 = drach_per_kb(second)
    ratio = d1 / d2 if d2 > 0 else float('inf')
    print(f"  {name:<20s} {len(seq):>7d} {d1:>9.1f} {d2:>9.1f} {ratio:>7.2f}x")

# =====================================================================
# 7. _orf2 sequences (pure ORF2)
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 6: ORF2 CONSENSUS DRACH/kb (L1M)")
print("=" * 80)

for name in ['L1M3_orf2', 'L1M5_orf2']:
    if name not in all_seqs:
        continue
    seq = all_seqs[name]
    print(f"\n{name} ({len(seq)} bp): {count_drach(seq)} DRACH, {drach_per_kb(seq):.1f}/kb")

    # Positional profile
    bin_size = len(seq) // 5
    for i in range(5):
        start = i * bin_size
        end = (i + 1) * bin_size if i < 4 else len(seq)
        chunk = seq[start:end]
        d = drach_per_kb(chunk)
        bar = '█' * int(d / 2)
        print(f"  {start:>4d}-{end:>4d}: {d:>6.1f}/kb {bar}")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Compare L1HS vs L1M ORIGINAL consensus DRACH distribution:

If L1M consensus shows ORF1 > ORF2 DRACH → ancestral feature preserved
If L1M consensus shows ORF1 ≈ ORF2 or ORF1 < ORF2 → random drift in genomic copies

Key comparison:
  L1HS_5end: ORF1 region DRACH vs ORF2 beginning DRACH
  L1M1/M2/M4_5end: first-half (5'UTR+ORF1) vs second-half (ORF2 start)
  L1M3/M5_orf2: pure ORF2 DRACH density
""")

print("Done!")
