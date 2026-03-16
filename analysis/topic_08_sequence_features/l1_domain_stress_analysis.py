#!/usr/bin/env python3
"""
Analyze stress vulnerability by L1 functional domain.

Maps each L1 read to the functional domain(s) it covers based on
RepeatMasker consensus coordinates, then checks whether reads from
specific domains are more stress-sensitive.

L1HS consensus (6064 bp) domain boundaries:
  5' UTR:   1 - 908
  ORF1:   909 - 1924
  Inter-ORF: 1925 - 1990  (small linker, merged with ORF1/ORF2 overlap)
  ORF2:  1991 - 5817
  3' UTR: 5818 - 6064

Note: Ancient L1 have degenerate domains, but consensus coordinates
from RepeatMasker still indicate which region of the ancestral element
the genomic copy derives from.
"""
import pandas as pd
import numpy as np
from scipy import stats

# ── L1HS domain boundaries (1-based) ──
DOMAINS = {
    "5UTR":     (1, 908),
    "ORF1":     (909, 1990),    # includes inter-ORF linker
    "ORF2":     (1991, 5817),
    "3UTR":     (5818, 6064),
}
L1HS_LENGTH = 6064

# ── Load rmsk with consensus positions ──
rmsk = pd.read_csv(
    'topic_08_sequence_features/hg38_L1_rmsk_consensus.tsv', sep='\t'
)
print(f"rmsk entries: {len(rmsk)}")

# Fix repStart for minus strand: UCSC stores as -(repLeft) for minus strand
# repStart on - strand is negative (= -(consensus_length - repEnd_on_plus))
# repLeft on + strand is negative (= -(consensus_length - repEnd))
# We need to normalize to get actual consensus start/end

# For + strand: repStart and repEnd are direct consensus coords
# For - strand: repStart is stored as -(consensus_length - repEnd), repEnd is direct
# repLeft: + strand = -(consensus_length - repEnd), - strand = direct repStart

# Let's compute consensus_start and consensus_end uniformly
def normalize_consensus_coords(row):
    """Convert UCSC rmsk coords to 1-based consensus start/end."""
    if row['strand'] == '+':
        # + strand: repStart = consensus start (0-based), repEnd = consensus end
        # repLeft = -(remaining after repEnd)
        cs = row['repStart'] + 1  # convert to 1-based
        ce = row['repEnd']
    else:
        # - strand: repStart = -(remaining before repStart) = negative
        # repEnd = consensus end on + strand
        # repLeft = consensus start on + strand (0-based)
        cs = row['repLeft'] + 1  # convert to 1-based
        ce = row['repEnd']
    return pd.Series({'cons_start': cs, 'cons_end': ce})

rmsk[['cons_start', 'cons_end']] = rmsk.apply(normalize_consensus_coords, axis=1)

# Create a unique key: chr:start-end for matching with L1 summary
rmsk['te_key'] = rmsk['genoName'] + ':' + rmsk['genoStart'].astype(str) + '-' + rmsk['genoEnd'].astype(str)

# ── Load L1 reads ──
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
dfs = []
for grp in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    path = f'../../results_group/{grp}/g_summary/{grp}_L1_summary.tsv'
    try:
        d = pd.read_csv(path, sep='\t')
        d['group'] = grp
        d['is_stress'] = 1 if 'Ars' in grp else 0
        dfs.append(d)
    except Exception as e:
        print(f"  Skip {grp}: {e}")
df = pd.concat(dfs, ignore_index=True)
df = df[df['qc_tag'] == 'PASS'].copy()
df['subfamily'] = df['gene_id']
df['age'] = df['subfamily'].apply(lambda x: 'Young' if x in YOUNG else 'Ancient')

# Create matching key
df['te_key'] = df['te_chr'] + ':' + df['te_start'].astype(str) + '-' + df['te_end'].astype(str)
print(f"PASS reads: {len(df)}")

# Merge
rmsk_slim = rmsk[['te_key', 'cons_start', 'cons_end', 'repName']].drop_duplicates('te_key')
df = df.merge(rmsk_slim, on='te_key', how='left')
matched = df['cons_start'].notna().sum()
print(f"Matched to rmsk: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
df = df[df['cons_start'].notna()].copy()
df['cons_start'] = df['cons_start'].astype(int)
df['cons_end'] = df['cons_end'].astype(int)

# ── Assign domain based on what the L1 ELEMENT covers ──
def get_domains_covered(cs, ce):
    """Return set of domains that this L1 element spans."""
    covered = set()
    for name, (ds, de) in DOMAINS.items():
        # Check overlap
        overlap = min(ce, de) - max(cs, ds)
        if overlap > 0:
            covered.add(name)
    return covered

def get_primary_domain(cs, ce):
    """Return the domain with the most coverage."""
    best_name, best_overlap = 'unknown', 0
    for name, (ds, de) in DOMAINS.items():
        overlap = min(ce, de) - max(cs, ds)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
    return best_name

def get_domain_coverage(cs, ce):
    """Return fraction of each domain covered."""
    result = {}
    for name, (ds, de) in DOMAINS.items():
        domain_len = de - ds + 1
        overlap = max(0, min(ce, de) - max(cs, ds))
        result[f'covers_{name}'] = overlap / domain_len
    return result

df['domains_covered'] = df.apply(lambda r: get_domains_covered(r['cons_start'], r['cons_end']), axis=1)
df['primary_domain'] = df.apply(lambda r: get_primary_domain(r['cons_start'], r['cons_end']), axis=1)
df['n_domains'] = df['domains_covered'].apply(len)

# Domain coverage fractions
cov = df.apply(lambda r: get_domain_coverage(r['cons_start'], r['cons_end']), axis=1, result_type='expand')
df = pd.concat([df, cov], axis=1)

# ── Analysis 1: Domain distribution ──
print("\n" + "="*70)
print("Analysis 1: Which domains do our L1 reads derive from?")
print("="*70)
for age in ['Ancient', 'Young']:
    sub = df[df['age'] == age]
    print(f"\n  {age} (N={len(sub)}):")
    for domain in ['5UTR', 'ORF1', 'ORF2', '3UTR']:
        has = sub[f'covers_{domain}'] > 0
        print(f"    {domain}: {has.sum():>5} ({has.mean()*100:>5.1f}%) | mean coverage: {sub[f'covers_{domain}'].mean()*100:.1f}%")
    print(f"  Primary domain distribution:")
    for domain, cnt in sub['primary_domain'].value_counts().items():
        print(f"    {domain}: {cnt} ({cnt/len(sub)*100:.1f}%)")

# ── Analysis 2: Stress delta by primary domain (Ancient only) ──
print("\n" + "="*70)
print("Analysis 2: Stress poly(A) delta by L1 domain (Ancient only)")
print("="*70)
anc = df[df['age'] == 'Ancient']
print(f"\n{'Domain':<12} {'N_un':>6} {'N_st':>6} {'Med_un':>8} {'Med_st':>8} {'Delta':>8} {'P':>12}")
print("-"*70)

domain_results = []
for domain in ['5UTR', 'ORF1', 'ORF2', '3UTR', 'unknown']:
    un = anc[(anc['primary_domain'] == domain) & (anc['is_stress'] == 0)]['polya_length']
    st = anc[(anc['primary_domain'] == domain) & (anc['is_stress'] == 1)]['polya_length']
    if len(un) >= 10 and len(st) >= 10:
        delta = st.median() - un.median()
        _, p = stats.mannwhitneyu(st, un, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {domain:<12} {len(un):>6} {len(st):>6} {un.median():>8.1f} {st.median():>8.1f} {delta:>+8.1f} {p:>12.2e} {sig}")
        domain_results.append({
            'domain': domain, 'n_un': len(un), 'n_st': len(st),
            'med_un': un.median(), 'med_st': st.median(), 'delta': delta, 'p': p
        })

# ── Analysis 3: By domain coverage pattern (which combination) ──
print("\n" + "="*70)
print("Analysis 3: Domain combination patterns (Ancient, stressed)")
print("="*70)
anc_st = anc[anc['is_stress'] == 1]
anc_un = anc[anc['is_stress'] == 0]

# Create domain pattern string
def domain_pattern(domains):
    order = ['5UTR', 'ORF1', 'ORF2', '3UTR']
    return '+'.join([d for d in order if d in domains])

anc['domain_pattern'] = anc['domains_covered'].apply(domain_pattern)

print(f"\n{'Pattern':<30} {'N_un':>6} {'N_st':>6} {'Med_un':>8} {'Med_st':>8} {'Delta':>8} {'P':>12}")
print("-"*85)

for pat in sorted(anc['domain_pattern'].unique()):
    un = anc[(anc['domain_pattern'] == pat) & (anc['is_stress'] == 0)]['polya_length']
    st = anc[(anc['domain_pattern'] == pat) & (anc['is_stress'] == 1)]['polya_length']
    if len(un) >= 15 and len(st) >= 15:
        delta = st.median() - un.median()
        _, p = stats.mannwhitneyu(st, un, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {pat:<30} {len(un):>6} {len(st):>6} {un.median():>8.1f} {st.median():>8.1f} {delta:>+8.1f} {p:>12.2e} {sig}")

# ── Analysis 4: Full-length (covers all 4 domains) vs partial ──
print("\n" + "="*70)
print("Analysis 4: Full-length (all domains) vs partial elements")
print("="*70)
anc['is_full'] = anc['n_domains'] == 4

for label, mask in [('Full-length (4 domains)', anc['is_full']), ('Partial (<4 domains)', ~anc['is_full'])]:
    sub = anc[mask]
    un = sub[sub['is_stress'] == 0]['polya_length']
    st = sub[sub['is_stress'] == 1]['polya_length']
    if len(un) >= 10 and len(st) >= 10:
        delta = st.median() - un.median()
        _, p = stats.mannwhitneyu(st, un, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:<30} N={len(un):>5}/{len(st):>5}  un={un.median():>7.1f}  st={st.median():>7.1f}  Δ={delta:>+7.1f}  P={p:.2e} {sig}")

# ── Analysis 5: m6A/kb by domain ──
print("\n" + "="*70)
print("Analysis 5: m6A sites by domain (Ancient, all conditions)")
print("="*70)
for domain in ['5UTR', 'ORF1', 'ORF2', '3UTR']:
    has = anc[anc[f'covers_{domain}'] > 0.5]  # >50% domain coverage
    no = anc[anc[f'covers_{domain}'] <= 0.1]   # minimal coverage
    if len(has) >= 20 and len(no) >= 20:
        _, p = stats.mannwhitneyu(has['m6A'], no['m6A'], alternative='two-sided')
        print(f"  Covers >{50}% {domain:<6}: N={len(has):>5}, median m6A={has['m6A'].median():.1f}")

# ── Analysis 6: 3' end only (most common DRS pattern) vs 5' containing ──
print("\n" + "="*70)
print("Analysis 6: 3'-biased reads vs 5'-containing reads")
print("="*70)
# DRS reads are 3'-biased, so most reads cover 3'UTR+ORF2
anc['has_5utr'] = anc['covers_5UTR'] > 0
anc['has_orf1'] = anc['covers_ORF1'] > 0.1

for label, mask in [
    ("Contains 5'UTR", anc['has_5utr']),
    ("Contains ORF1", anc['has_orf1']),
    ("3' only (no 5UTR/ORF1)", ~anc['has_5utr'] & ~anc['has_orf1']),
]:
    sub = anc[mask]
    un = sub[sub['is_stress'] == 0]['polya_length']
    st = sub[sub['is_stress'] == 1]['polya_length']
    if len(un) >= 10 and len(st) >= 10:
        delta = st.median() - un.median()
        _, p = stats.mannwhitneyu(st, un, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:<30} N={len(un):>5}/{len(st):>5}  un={un.median():>7.1f}  st={st.median():>7.1f}  Δ={delta:>+7.1f}  P={p:.2e} {sig}")

# ── Save results ──
pd.DataFrame(domain_results).to_csv(
    'topic_08_sequence_features/l1_domain_stress_results.tsv', sep='\t', index=False
)
print("\nSaved: l1_domain_stress_results.tsv")
