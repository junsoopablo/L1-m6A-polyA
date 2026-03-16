#!/usr/bin/env python3
"""
5' Adapter (REL5) Detection Analysis — Complete Summary
========================================================

BACKGROUND:
- HeLa/HeLa-Ars data (PRJNA842344, Dar et al. eLife 2024) used TERA-seq protocol
- REL5 5' adapter (58bp DNA) ligated to 5' cap of RNA → presence = full-length read
- DRS sequences 3'→5', so adapter appears at the start of basecalled reads

APPROACH:
- Re-basecalled 10 FAST5 files (SRR20445030, HeLa rep1) with `guppy --trim_strategy none`
- 37,384 PASS reads obtained
- Searched for REL5 adapter (RNA-converted) in first 80bp of each read

KEY FINDINGS:

1. ADAPTER IS DETECTABLE (1.2% of reads at ≥0.70 match threshold)
   - RNA basecaller skips first ~12bp of DNA adapter (DNA/RNA signal mismatch)
   - Adapter positions 12-58 basecalled as RNA with ~80-100% accuracy
   - 132 reads (0.35%) at ≥0.90 threshold (high confidence)

2. ADAPTER-POSITIVE READS ARE SHORTER, NOT LONGER
   - Full-length (adapter+): median 776bp
   - Truncated (adapter-): median 975bp
   - Ratio: 0.80x (full-length 20% shorter!)
   - Highest adapter rate in 200-500bp bin (3.9%)
   - Explanation: short mRNAs are more easily fully captured by DRS
     (pore more likely to read entire 500bp molecule than 3000bp one)

3. IMPLICATION FOR L1 ANALYSIS
   - L1 full-length = ~6kb — probability of DRS reaching 5' end ≈ 0%
   - 0/55 reads ≥5kb had adapter; 4/959 reads ≥3kb had adapter (0.4%)
   - The adapter tells us about DRS READ completeness, NOT RNA INTEGRITY
   - A read without adapter could be: (a) intact RNA not fully sequenced by DRS,
     or (b) degraded RNA fragment — CANNOT DISTINGUISH

4. WHY DETECTION RATE IS LOW (~1.2%)
   - ONT DRS 5' capture efficiency inherently low (reported 1-4% in literature)
   - RNA basecaller partially mis-reads DNA adapter signal
   - SRA FAST5 file reformatting may further reduce signal quality

CONCLUSION:
- The 5' adapter approach CANNOT help classify full-length vs degraded L1 reads
- Our existing methods are correct: read length, dist_to_3prime, consensus position mapping
- For L1, "full-length" assessment is best done by checking if reads reach the
  5' end of the L1 consensus (~15% region), not by adapter detection

CUTADAPT PARAMETERS (from original TERA-seq pipeline):
  cutadapt -g XAATGATACGGCGACCACCGAGATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT \
    --overlap 31 --error-rate 0.29 --minimum-length 25

GUPPY COMMAND (preserve adapter):
  guppy_basecaller -c rna_r9.4.1_70bps_hac.cfg --trim_strategy none

DATA:
  Pilot: /vault/.../PRJNA842344.../adapter_pilot/
  Original: topic_09_5prime_adapter/
"""
