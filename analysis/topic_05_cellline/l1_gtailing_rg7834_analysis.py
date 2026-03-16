#!/usr/bin/env python3
"""
L1 G-tailing (TENT4A/B) Analysis: RNA004 Polyaglot + RG7834

Analyses:
  1. Polyaglot spike rate × RG7834 (poly(A)-matched, G vs U/C, subfamily)
  2. RNA002 Ninetails cross-chemistry validation
  3. Signal-level spike characteristics (HIGH vs LOW)

Spike orientation (ninetails paper): 0=LOW=U/C (pyrimidine), 1=HIGH=G (guanosine)
"""

import sys
import ast
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────
ISOTENT = '/qbio/junsoopablo/02_Projects/05_IsoTENT'
ISOTENT_L1 = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'

SPIKE_HCT = f'{ISOTENT}/polyaglot/HCT116.polyaglot.spike.tsv'
SPIKE_RG  = f'{ISOTENT}/polyaglot/HCT116_RG7834.polyaglot.spike.tsv'
L1_HCT    = f'{ISOTENT}/alignment/HCT116/d_LINE_quantification/HCT116_L1_reads.tsv'
L1_RG     = f'{ISOTENT}/alignment/HCT116_RG7834/d_LINE_quantification/HCT116_RG7834_L1_reads.tsv'

# RNA002 Hct116 ninetails
NT_RESIDUES = {
    'Hct116_3': f'{ISOTENT_L1}/results_group/Hct116_3/f_ninetails/2026-02-03_20-10-14_Hct116_3_nonadenosine_residues.tsv',
    'Hct116_4': f'{ISOTENT_L1}/results_group/Hct116_4/f_ninetails/2026-02-03_20-08-30_Hct116_4_nonadenosine_residues.tsv',
}
NT_CLASSES = {
    'Hct116_3': f'{ISOTENT_L1}/results_group/Hct116_3/f_ninetails/2026-02-03_20-10-14_Hct116_3_read_classes.tsv',
    'Hct116_4': f'{ISOTENT_L1}/results_group/Hct116_4/f_ninetails/2026-02-03_20-08-30_Hct116_4_read_classes.tsv',
}
L1_RNA002 = {
    'Hct116_3': f'{ISOTENT_L1}/results_group/Hct116_3/g_summary/Hct116_3_L1_summary.tsv',
    'Hct116_4': f'{ISOTENT_L1}/results_group/Hct116_4/g_summary/Hct116_4_L1_summary.tsv',
}

YOUNG_L1 = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# ── Helper Functions ───────────────────────────────────────────────────

def parse_list_col(s):
    """Parse string representation of list column."""
    if pd.isna(s) or s == '[]':
        return []
    try:
        return ast.literal_eval(s)
    except:
        return []


def fisher_exact_with_ci(a, b, c, d, alpha=0.05):
    """Fisher exact test with odds ratio and 95% CI (log method)."""
    table = np.array([[a, b], [c, d]])
    oddsratio, pvalue = stats.fisher_exact(table)
    # Log odds ratio CI
    if 0 in [a, b, c, d]:
        ci_lo, ci_hi = np.nan, np.nan
    else:
        log_or = np.log(oddsratio)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        z = stats.norm.ppf(1 - alpha/2)
        ci_lo = np.exp(log_or - z * se)
        ci_hi = np.exp(log_or + z * se)
    return oddsratio, pvalue, ci_lo, ci_hi


def prop_test_with_ci(x1, n1, x2, n2):
    """Two-proportion z-test with difference CI."""
    p1 = x1 / n1 if n1 > 0 else 0
    p2 = x2 / n2 if n2 > 0 else 0
    diff = p1 - p2
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2) if n1 > 0 and n2 > 0 else np.nan
    ci_lo = diff - 1.96 * se
    ci_hi = diff + 1.96 * se
    # z-test
    p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if n1 > 0 and n2 > 0 and p_pool > 0 else np.nan
    z = diff / se_pool if se_pool and se_pool > 0 else np.nan
    pval = 2 * stats.norm.sf(abs(z)) if not np.isnan(z) else np.nan
    return p1, p2, diff, ci_lo, ci_hi, pval


def print_section(title):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── Load & Merge Data ──────────────────────────────────────────────────

def load_rna004_data():
    """Load and merge polyaglot spike + L1 reads for both samples."""
    samples = {}
    for name, spike_path, l1_path in [
        ('HCT116', SPIKE_HCT, L1_HCT),
        ('HCT116_RG7834', SPIKE_RG, L1_RG),
    ]:
        spike = pd.read_csv(spike_path, sep='\t')
        l1 = pd.read_csv(l1_path, sep='\t')

        # Parse list columns
        for col in ['spike_positions', 'spike_lengths', 'spike_orientations', 'relative_spike_positions']:
            spike[col] = spike[col].apply(parse_list_col)

        # Merge on read_id
        merged = spike.merge(l1, on='read_id', how='inner')

        # Classify Young vs Ancient
        merged['age'] = merged['gene_id'].apply(lambda x: 'Young' if x in YOUNG_L1 else 'Ancient')

        # Spike presence flag
        merged['has_spike'] = merged['spike_count'] > 0

        # Expand spike orientations for per-spike analysis
        spike_rows = []
        for _, row in merged[merged['has_spike']].iterrows():
            for i, orient in enumerate(row['spike_orientations']):
                spike_rows.append({
                    'read_id': row['read_id'],
                    'gene_id': row['gene_id'],
                    'age': row['age'],
                    'polya_length': row['polya_length'],
                    'spike_orientation': orient,
                    'spike_type': 'G (HIGH)' if orient == 1 else 'U/C (LOW)',
                    'spike_position': row['spike_positions'][i] if i < len(row['spike_positions']) else np.nan,
                    'spike_length': row['spike_lengths'][i] if i < len(row['spike_lengths']) else np.nan,
                    'relative_position': row['relative_spike_positions'][i] if i < len(row['relative_spike_positions']) else np.nan,
                })
        spikes_df = pd.DataFrame(spike_rows) if spike_rows else pd.DataFrame()

        samples[name] = {
            'merged': merged,
            'spikes': spikes_df,
            'n_total': len(merged),
            'n_spike': merged['has_spike'].sum(),
            'n_l1_only': len(l1),
            'n_spike_only': len(spike),
        }
    return samples


def load_rna002_data():
    """Load RNA002 Hct116 ninetails data."""
    rna002 = {}
    for sample in ['Hct116_3', 'Hct116_4']:
        # L1 summary with all read info
        l1 = pd.read_csv(L1_RNA002[sample], sep='\t')
        # Apply dist_to_3prime <= 100 filter to match IsoTENT filter
        l1_filtered = l1[l1['dist_to_3prime'] <= 100].copy()

        # Read classes (ninetails)
        classes = pd.read_csv(NT_CLASSES[sample], sep='\t')

        # Non-adenosine residues
        residues = pd.read_csv(NT_RESIDUES[sample], sep='\t')

        # Merge L1 reads with read_classes
        l1_classes = l1_filtered.merge(
            classes, left_on='read_id', right_on='readname', how='inner'
        )

        # L1 decorated reads (class_y from ninetails read_classes after merge)
        class_col = 'class_y' if 'class_y' in l1_classes.columns else 'class'
        l1_decorated = l1_classes[l1_classes[class_col] == 'decorated']

        # Residue info for L1 reads
        l1_residues = residues[residues['readname'].isin(l1_filtered['read_id'])]

        rna002[sample] = {
            'l1': l1_filtered,
            'l1_classes': l1_classes,
            'l1_decorated': l1_decorated,
            'l1_residues': l1_residues,
            'n_l1': len(l1_filtered),
            'n_decorated': len(l1_decorated),
        }
    return rna002


# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: Polyaglot Spike Rate × RG7834
# ══════════════════════════════════════════════════════════════════════

def analysis1_spike_rate(samples):
    """Overall spike rate, G vs U/C, poly(A)-matched, subfamily."""
    print_section("ANALYSIS 1: Polyaglot Spike Rate × RG7834")

    hct = samples['HCT116']
    rg = samples['HCT116_RG7834']

    # ── 1a. Merge QC ──
    print("\n--- 1a. Merge QC ---")
    for name, s in [('HCT116', hct), ('HCT116_RG7834', rg)]:
        print(f"  {name}: spike.tsv={s['n_spike_only']} reads, L1_reads.tsv={s['n_l1_only']} reads, "
              f"merged={s['n_total']} reads, with_spike={s['n_spike']} ({100*s['n_spike']/s['n_total']:.1f}%)")

    # ── 1b. Overall spike rate comparison ──
    print("\n--- 1b. Overall Spike Rate ---")
    a, b = hct['n_spike'], hct['n_total'] - hct['n_spike']
    c, d = rg['n_spike'], rg['n_total'] - rg['n_spike']
    OR, p, ci_lo, ci_hi = fisher_exact_with_ci(a, b, c, d)
    print(f"  HCT116:      {hct['n_spike']}/{hct['n_total']} = {100*hct['n_spike']/hct['n_total']:.1f}%")
    print(f"  HCT116_RG:   {rg['n_spike']}/{rg['n_total']} = {100*rg['n_spike']/rg['n_total']:.1f}%")
    print(f"  Fisher exact: OR={OR:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={p:.4g}")

    # ── 1c. G (HIGH) vs U/C (LOW) spike composition ──
    print("\n--- 1c. G vs U/C Spike Composition ---")
    for name, s in [('HCT116', hct), ('HCT116_RG7834', rg)]:
        if len(s['spikes']) > 0:
            g_count = (s['spikes']['spike_orientation'] == 1).sum()
            uc_count = (s['spikes']['spike_orientation'] == 0).sum()
            total = g_count + uc_count
            print(f"  {name}: G={g_count} ({100*g_count/total:.1f}%), U/C={uc_count} ({100*uc_count/total:.1f}%), total_spikes={total}")

    # Fisher: G vs U/C between conditions
    g_hct = (hct['spikes']['spike_orientation'] == 1).sum() if len(hct['spikes']) > 0 else 0
    uc_hct = (hct['spikes']['spike_orientation'] == 0).sum() if len(hct['spikes']) > 0 else 0
    g_rg = (rg['spikes']['spike_orientation'] == 1).sum() if len(rg['spikes']) > 0 else 0
    uc_rg = (rg['spikes']['spike_orientation'] == 0).sum() if len(rg['spikes']) > 0 else 0
    OR2, p2, ci2_lo, ci2_hi = fisher_exact_with_ci(g_hct, uc_hct, g_rg, uc_rg)
    print(f"  Fisher (G fraction HCT vs RG): OR={OR2:.3f} [{ci2_lo:.3f}, {ci2_hi:.3f}], p={p2:.4g}")

    # Absolute G spikes per 1000 reads
    g_per1k_hct = 1000 * g_hct / hct['n_total']
    g_per1k_rg = 1000 * g_rg / rg['n_total']
    uc_per1k_hct = 1000 * uc_hct / hct['n_total']
    uc_per1k_rg = 1000 * uc_rg / rg['n_total']
    print(f"\n  Absolute rates per 1000 L1 reads:")
    print(f"    G:   HCT={g_per1k_hct:.1f}, RG={g_per1k_rg:.1f}  (ratio={g_per1k_rg/g_per1k_hct:.2f}x)" if g_per1k_hct > 0 else "")
    print(f"    U/C: HCT={uc_per1k_hct:.1f}, RG={uc_per1k_rg:.1f}  (ratio={uc_per1k_rg/uc_per1k_hct:.2f}x)" if uc_per1k_hct > 0 else "")

    # ── 1d. Poly(A)-length-matched spike rate ──
    print("\n--- 1d. Poly(A)-Length-Matched Spike Rate ---")
    hct_m = hct['merged'].copy()
    rg_m = rg['merged'].copy()

    # Filter reads with polya_length > 0 for binning
    hct_m = hct_m[hct_m['polya_length'] > 0].copy()
    rg_m = rg_m[rg_m['polya_length'] > 0].copy()

    # Create 50nt bins
    bins = list(range(0, 401, 50))
    labels = [f"{b}-{b+50}" for b in bins[:-1]]
    hct_m['pa_bin'] = pd.cut(hct_m['polya_length'], bins=bins, labels=labels, right=False)
    rg_m['pa_bin'] = pd.cut(rg_m['polya_length'], bins=bins, labels=labels, right=False)

    print(f"  {'Bin':>10s}  {'HCT':>6s}  {'HCT_sp':>7s}  {'HCT%':>6s}  {'RG':>6s}  {'RG_sp':>7s}  {'RG%':>6s}  {'Fisher_p':>9s}")
    for label in labels:
        h_bin = hct_m[hct_m['pa_bin'] == label]
        r_bin = rg_m[rg_m['pa_bin'] == label]
        h_n, h_sp = len(h_bin), h_bin['has_spike'].sum()
        r_n, r_sp = len(r_bin), r_bin['has_spike'].sum()
        if h_n >= 5 and r_n >= 5:
            _, fp, _, _ = fisher_exact_with_ci(h_sp, h_n - h_sp, r_sp, r_n - r_sp)
            h_pct = 100 * h_sp / h_n if h_n > 0 else 0
            r_pct = 100 * r_sp / r_n if r_n > 0 else 0
            print(f"  {label:>10s}  {h_n:>6d}  {h_sp:>7d}  {h_pct:>5.1f}%  {r_n:>6d}  {r_sp:>7d}  {r_pct:>5.1f}%  {fp:>9.4g}")
        elif h_n > 0 or r_n > 0:
            h_pct = 100 * h_sp / h_n if h_n > 0 else 0
            r_pct = 100 * r_sp / r_n if r_n > 0 else 0
            print(f"  {label:>10s}  {h_n:>6d}  {h_sp:>7d}  {h_pct:>5.1f}%  {r_n:>6d}  {r_sp:>7d}  {r_pct:>5.1f}%       n<5")

    # Cochran-Mantel-Haenszel-like approach: pool across bins
    print("\n  Poly(A)-length-stratified summary (bins with n>=5):")
    tot_h_sp, tot_h_nsp, tot_r_sp, tot_r_nsp = 0, 0, 0, 0
    for label in labels:
        h_bin = hct_m[hct_m['pa_bin'] == label]
        r_bin = rg_m[rg_m['pa_bin'] == label]
        h_n, h_sp = len(h_bin), h_bin['has_spike'].sum()
        r_n, r_sp = len(r_bin), r_bin['has_spike'].sum()
        if h_n >= 5 and r_n >= 5:
            tot_h_sp += h_sp
            tot_h_nsp += (h_n - h_sp)
            tot_r_sp += r_sp
            tot_r_nsp += (r_n - r_sp)
    OR3, p3, ci3_lo, ci3_hi = fisher_exact_with_ci(tot_h_sp, tot_h_nsp, tot_r_sp, tot_r_nsp)
    print(f"  Pooled (length-qualifying reads): HCT {tot_h_sp}/{tot_h_sp+tot_h_nsp}, RG {tot_r_sp}/{tot_r_sp+tot_r_nsp}")
    print(f"  Fisher: OR={OR3:.3f} [{ci3_lo:.3f}, {ci3_hi:.3f}], p={p3:.4g}")

    # ── 1e. G vs U/C composition poly(A)-matched ──
    print("\n--- 1e. G vs U/C Composition Poly(A)-Matched ---")
    if len(hct['spikes']) > 0 and len(rg['spikes']) > 0:
        hct_sp = hct['spikes'][hct['spikes']['polya_length'] > 0].copy()
        rg_sp = rg['spikes'][rg['spikes']['polya_length'] > 0].copy()
        hct_sp['pa_bin'] = pd.cut(hct_sp['polya_length'], bins=bins, labels=labels, right=False)
        rg_sp['pa_bin'] = pd.cut(rg_sp['polya_length'], bins=bins, labels=labels, right=False)

        print(f"  {'Bin':>10s}  {'HCT_G':>6s}  {'HCT_UC':>7s}  {'HCT_G%':>7s}  {'RG_G':>6s}  {'RG_UC':>7s}  {'RG_G%':>7s}")
        for label in labels:
            h_bin = hct_sp[hct_sp['pa_bin'] == label]
            r_bin = rg_sp[rg_sp['pa_bin'] == label]
            h_g = (h_bin['spike_orientation'] == 1).sum()
            h_uc = (h_bin['spike_orientation'] == 0).sum()
            r_g = (r_bin['spike_orientation'] == 1).sum()
            r_uc = (r_bin['spike_orientation'] == 0).sum()
            if (h_g + h_uc) > 0 or (r_g + r_uc) > 0:
                h_pct = 100 * h_g / (h_g + h_uc) if (h_g + h_uc) > 0 else 0
                r_pct = 100 * r_g / (r_g + r_uc) if (r_g + r_uc) > 0 else 0
                print(f"  {label:>10s}  {h_g:>6d}  {h_uc:>7d}  {h_pct:>6.1f}%  {r_g:>6d}  {r_uc:>7d}  {r_pct:>6.1f}%")

    # ── 1f. Young vs Ancient L1 spike rate ──
    print("\n--- 1f. Young vs Ancient L1 Spike Rate ---")
    for name, s in [('HCT116', hct), ('HCT116_RG7834', rg)]:
        m = s['merged']
        for age in ['Young', 'Ancient']:
            sub = m[m['age'] == age]
            n = len(sub)
            sp = sub['has_spike'].sum()
            pct = 100 * sp / n if n > 0 else 0
            print(f"  {name:>15s} {age:>8s}: {sp}/{n} = {pct:.1f}%")

    # Fisher: Young spike rate HCT vs RG
    for age in ['Young', 'Ancient']:
        h_sub = hct['merged'][hct['merged']['age'] == age]
        r_sub = rg['merged'][rg['merged']['age'] == age]
        h_sp, h_n = h_sub['has_spike'].sum(), len(h_sub)
        r_sp, r_n = r_sub['has_spike'].sum(), len(r_sub)
        if h_n > 0 and r_n > 0:
            OR_age, p_age, ci_lo_age, ci_hi_age = fisher_exact_with_ci(
                h_sp, h_n - h_sp, r_sp, r_n - r_sp)
            print(f"  {age} HCT vs RG: OR={OR_age:.3f} [{ci_lo_age:.3f}, {ci_hi_age:.3f}], p={p_age:.4g}")

    # ── 1g. Spike count per read distribution ──
    print("\n--- 1g. Spike Count Per Read (reads with spike) ---")
    for name, s in [('HCT116', hct), ('HCT116_RG7834', rg)]:
        spiked = s['merged'][s['merged']['has_spike']]
        counts = spiked['spike_count'].value_counts().sort_index()
        print(f"  {name}: {dict(counts)}")
        print(f"    mean={spiked['spike_count'].mean():.2f}, median={spiked['spike_count'].median():.0f}")
    # Mann-Whitney on spike counts (spiked reads only)
    hct_sc = hct['merged'][hct['merged']['has_spike']]['spike_count']
    rg_sc = rg['merged'][rg['merged']['has_spike']]['spike_count']
    if len(hct_sc) > 0 and len(rg_sc) > 0:
        u, p_mw = stats.mannwhitneyu(hct_sc, rg_sc, alternative='two-sided')
        print(f"  Mann-Whitney U: U={u:.0f}, p={p_mw:.4g}")

    # ── 1h. Spike relative position distribution ──
    print("\n--- 1h. Spike Relative Position (within poly(A)) ---")
    for name, s in [('HCT116', hct), ('HCT116_RG7834', rg)]:
        if len(s['spikes']) > 0:
            rel_pos = s['spikes']['relative_position'].dropna()
            if len(rel_pos) > 0:
                print(f"  {name}: n={len(rel_pos)}, mean={rel_pos.mean():.3f}, "
                      f"median={rel_pos.median():.3f}, "
                      f"Q1={rel_pos.quantile(0.25):.3f}, Q3={rel_pos.quantile(0.75):.3f}")
    # K-S test for position distributions
    hct_rp = hct['spikes']['relative_position'].dropna() if len(hct['spikes']) > 0 else pd.Series()
    rg_rp = rg['spikes']['relative_position'].dropna() if len(rg['spikes']) > 0 else pd.Series()
    if len(hct_rp) > 5 and len(rg_rp) > 5:
        ks_stat, ks_p = stats.ks_2samp(hct_rp, rg_rp)
        print(f"  K-S test: D={ks_stat:.3f}, p={ks_p:.4g}")

    # Position by spike type
    print("\n  Position by spike type:")
    for name, s in [('HCT116', hct), ('HCT116_RG7834', rg)]:
        if len(s['spikes']) > 0:
            for st in ['G (HIGH)', 'U/C (LOW)']:
                sub = s['spikes'][s['spikes']['spike_type'] == st]['relative_position'].dropna()
                if len(sub) > 0:
                    print(f"    {name} {st}: n={len(sub)}, median={sub.median():.3f}")

    # ── 1i. Poly(A) length distribution comparison ──
    print("\n--- 1i. Poly(A) Length Distribution ---")
    for name, s in [('HCT116', hct), ('HCT116_RG7834', rg)]:
        pa = s['merged']['polya_length']
        pa_pos = pa[pa > 0]
        print(f"  {name}: n={len(pa_pos)}, mean={pa_pos.mean():.1f}, median={pa_pos.median():.1f}, "
              f"Q1={pa_pos.quantile(0.25):.1f}, Q3={pa_pos.quantile(0.75):.1f}")
    # K-S test
    hct_pa = hct['merged']['polya_length']
    rg_pa = rg['merged']['polya_length']
    hct_pa_pos = hct_pa[hct_pa > 0]
    rg_pa_pos = rg_pa[rg_pa > 0]
    if len(hct_pa_pos) > 5 and len(rg_pa_pos) > 5:
        ks_stat, ks_p = stats.ks_2samp(hct_pa_pos, rg_pa_pos)
        print(f"  K-S test: D={ks_stat:.3f}, p={ks_p:.4g}")

    return samples


# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: RNA002 Ninetails Cross-Chemistry Validation
# ══════════════════════════════════════════════════════════════════════

def analysis2_rna002_cross(samples, rna002):
    """Cross-chemistry comparison: RNA002 ninetails vs RNA004 polyaglot."""
    print_section("ANALYSIS 2: RNA002 Ninetails Cross-Chemistry Validation")

    # ── 2a. RNA002 L1 decorated rate ──
    print("\n--- 2a. RNA002 Hct116 L1 Decorated Rate (dist_to_3prime ≤ 100) ---")
    for sample, data in rna002.items():
        n_total = data['n_l1']
        # Only count PASS reads for fair comparison
        l1_cls = data['l1_classes']
        qc_col = 'qc_tag_y' if 'qc_tag_y' in l1_cls.columns else 'qc_tag'
        cls_col = 'class_y' if 'class_y' in l1_cls.columns else 'class'
        l1_pass = l1_cls[l1_cls[qc_col] == 'PASS']
        n_pass = len(l1_pass)
        n_dec = len(l1_pass[l1_pass[cls_col] == 'decorated']) if n_pass > 0 else 0
        pct = 100 * n_dec / n_pass if n_pass > 0 else 0
        print(f"  {sample}: total_L1={n_total}, PASS={n_pass}, decorated={n_dec} ({pct:.1f}%)")

    # ── 2b. G/U/C composition in RNA002 ──
    print("\n--- 2b. RNA002 L1 Non-Adenosine Residue Composition ---")
    for sample, data in rna002.items():
        residues = data['l1_residues']
        if len(residues) > 0:
            nuc_counts = residues['prediction'].value_counts()
            total = nuc_counts.sum()
            print(f"  {sample}: total_residues={total}")
            for nuc in ['G', 'U', 'C']:
                n = nuc_counts.get(nuc, 0)
                pct = 100 * n / total if total > 0 else 0
                print(f"    {nuc}: {n} ({pct:.1f}%)")

    # ── 2c. Cross-chemistry comparison table ──
    print("\n--- 2c. Cross-Chemistry Comparison (RNA004 polyaglot vs RNA002 ninetails) ---")
    # RNA004 polyaglot composition
    hct = samples['HCT116']
    if len(hct['spikes']) > 0:
        g_n = (hct['spikes']['spike_orientation'] == 1).sum()
        uc_n = (hct['spikes']['spike_orientation'] == 0).sum()
        total_rna004 = g_n + uc_n
        print(f"\n  RNA004 HCT116 (polyaglot spikes):")
        print(f"    G (HIGH=1): {g_n}/{total_rna004} = {100*g_n/total_rna004:.1f}%")
        print(f"    U/C (LOW=0): {uc_n}/{total_rna004} = {100*uc_n/total_rna004:.1f}%")

    # RNA002 composition (combined Hct116_3 + Hct116_4)
    all_residues = pd.concat([rna002[s]['l1_residues'] for s in rna002])
    if len(all_residues) > 0:
        nuc_counts = all_residues['prediction'].value_counts()
        total_r2 = nuc_counts.sum()
        g_r2 = nuc_counts.get('G', 0)
        u_r2 = nuc_counts.get('U', 0)
        c_r2 = nuc_counts.get('C', 0)
        pyrimidine_r2 = u_r2 + c_r2
        print(f"\n  RNA002 Hct116 combined (ninetails residues):")
        print(f"    G: {g_r2}/{total_r2} = {100*g_r2/total_r2:.1f}%")
        print(f"    U: {u_r2}/{total_r2} = {100*u_r2/total_r2:.1f}%")
        print(f"    C: {c_r2}/{total_r2} = {100*c_r2/total_r2:.1f}%")
        print(f"    Pyrimidine (U+C): {pyrimidine_r2}/{total_r2} = {100*pyrimidine_r2/total_r2:.1f}%")
        print(f"    G vs Pyrimidine ratio: {100*g_r2/total_r2:.1f}% vs {100*pyrimidine_r2/total_r2:.1f}%")

    # ── 2d. Decorated rate cross-chemistry ──
    print("\n--- 2d. Decorated Rate Cross-Chemistry ---")
    # RNA004
    hct_spike_rate = 100 * hct['n_spike'] / hct['n_total'] if hct['n_total'] > 0 else 0
    print(f"  RNA004 HCT116 polyaglot spike rate: {hct['n_spike']}/{hct['n_total']} = {hct_spike_rate:.1f}%")

    # RNA002 (combined, PASS only, dist_to_3prime <= 100)
    for sample, data in rna002.items():
        l1_classes = data['l1_classes']
        # Check column name for qc_tag (might be qc_tag_x or qc_tag_y after merge)
        qc_col = 'qc_tag_y' if 'qc_tag_y' in l1_classes.columns else 'qc_tag'
        cls_col = 'class_y' if 'class_y' in l1_classes.columns else 'class'
        pass_reads = l1_classes[l1_classes[qc_col] == 'PASS']
        n_pass = len(pass_reads)
        n_dec = len(pass_reads[pass_reads[cls_col] == 'decorated'])
        pct = 100 * n_dec / n_pass if n_pass > 0 else 0
        print(f"  RNA002 {sample} ninetails decorated rate (PASS, d3p≤100): {n_dec}/{n_pass} = {pct:.1f}%")


# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: Signal-Level Spike Characteristics
# ══════════════════════════════════════════════════════════════════════

def analysis3_signal_characteristics(samples):
    """Spike signal analysis: length, position by orientation."""
    print_section("ANALYSIS 3: Spike Signal Characteristics")

    # ── 3a. Spike length by orientation ──
    print("\n--- 3a. Spike Length by Orientation ---")
    for name, s in [('HCT116', samples['HCT116']), ('HCT116_RG7834', samples['HCT116_RG7834'])]:
        if len(s['spikes']) > 0:
            for st in ['G (HIGH)', 'U/C (LOW)']:
                sub = s['spikes'][s['spikes']['spike_type'] == st]['spike_length'].dropna()
                if len(sub) > 0:
                    print(f"  {name} {st}: n={len(sub)}, mean={sub.mean():.1f}, "
                          f"median={sub.median():.1f}, std={sub.std():.1f}")

    # Mann-Whitney between G and U/C spike lengths
    print("\n  Spike length comparison (G vs U/C):")
    for name, s in [('HCT116', samples['HCT116']), ('HCT116_RG7834', samples['HCT116_RG7834'])]:
        if len(s['spikes']) > 0:
            g_len = s['spikes'][s['spikes']['spike_type'] == 'G (HIGH)']['spike_length'].dropna()
            uc_len = s['spikes'][s['spikes']['spike_type'] == 'U/C (LOW)']['spike_length'].dropna()
            if len(g_len) > 2 and len(uc_len) > 2:
                u, p_mw = stats.mannwhitneyu(g_len, uc_len, alternative='two-sided')
                print(f"  {name}: G median={g_len.median():.1f} vs U/C median={uc_len.median():.1f}, MW p={p_mw:.4g}")

    # ── 3b. Spike length change with RG7834 ──
    print("\n--- 3b. Spike Length Change (HCT vs RG7834) ---")
    hct_sp = samples['HCT116']['spikes']
    rg_sp = samples['HCT116_RG7834']['spikes']
    if len(hct_sp) > 0 and len(rg_sp) > 0:
        for st in ['G (HIGH)', 'U/C (LOW)']:
            h_len = hct_sp[hct_sp['spike_type'] == st]['spike_length'].dropna()
            r_len = rg_sp[rg_sp['spike_type'] == st]['spike_length'].dropna()
            if len(h_len) > 2 and len(r_len) > 2:
                u, p_mw = stats.mannwhitneyu(h_len, r_len, alternative='two-sided')
                print(f"  {st}: HCT median={h_len.median():.1f} vs RG median={r_len.median():.1f}, MW p={p_mw:.4g}")

    # ── 3c. Poly(A) position of spikes (5' vs 3' within tail) ──
    print("\n--- 3c. Spike Position Within Poly(A) Tail ---")
    print("  (relative_position: 0=5' end of poly(A), 1=3' end)")
    for name, s in [('HCT116', samples['HCT116']), ('HCT116_RG7834', samples['HCT116_RG7834'])]:
        if len(s['spikes']) > 0:
            for st in ['G (HIGH)', 'U/C (LOW)']:
                sub = s['spikes'][s['spikes']['spike_type'] == st]['relative_position'].dropna()
                if len(sub) > 0:
                    # Fraction in 5' half vs 3' half
                    n_5p = (sub < 0.5).sum()
                    n_3p = (sub >= 0.5).sum()
                    print(f"  {name} {st}: n={len(sub)}, 5'half={n_5p} ({100*n_5p/len(sub):.0f}%), "
                          f"3'half={n_3p} ({100*n_3p/len(sub):.0f}%), median_pos={sub.median():.3f}")


# ══════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════

def print_summary(samples, rna002):
    """Print summary table for quick reference."""
    print_section("SUMMARY TABLE")

    hct = samples['HCT116']
    rg = samples['HCT116_RG7834']

    # Spike counts
    g_hct = (hct['spikes']['spike_orientation'] == 1).sum() if len(hct['spikes']) > 0 else 0
    uc_hct = (hct['spikes']['spike_orientation'] == 0).sum() if len(hct['spikes']) > 0 else 0
    g_rg = (rg['spikes']['spike_orientation'] == 1).sum() if len(rg['spikes']) > 0 else 0
    uc_rg = (rg['spikes']['spike_orientation'] == 0).sum() if len(rg['spikes']) > 0 else 0

    print(f"\n  {'Metric':<35s}  {'HCT116':>10s}  {'RG7834':>10s}  {'Change':>10s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*10}")

    # Total L1 reads
    print(f"  {'Total L1 reads':<35s}  {hct['n_total']:>10d}  {rg['n_total']:>10d}  "
          f"{rg['n_total']/hct['n_total']:.2f}x")

    # Spiked reads
    h_rate = 100*hct['n_spike']/hct['n_total']
    r_rate = 100*rg['n_spike']/rg['n_total']
    print(f"  {'Spiked reads (%)':<35s}  {h_rate:>9.1f}%  {r_rate:>9.1f}%  "
          f"{r_rate-h_rate:>+.1f}pp")

    # G spikes
    g_h_pct = 100*g_hct/(g_hct+uc_hct) if (g_hct+uc_hct) > 0 else 0
    g_r_pct = 100*g_rg/(g_rg+uc_rg) if (g_rg+uc_rg) > 0 else 0
    print(f"  {'G fraction of spikes (%)':<35s}  {g_h_pct:>9.1f}%  {g_r_pct:>9.1f}%  "
          f"{g_r_pct-g_h_pct:>+.1f}pp")

    # Absolute rates
    g_per1k_h = 1000*g_hct/hct['n_total'] if hct['n_total'] > 0 else 0
    g_per1k_r = 1000*g_rg/rg['n_total'] if rg['n_total'] > 0 else 0
    uc_per1k_h = 1000*uc_hct/hct['n_total'] if hct['n_total'] > 0 else 0
    uc_per1k_r = 1000*uc_rg/rg['n_total'] if rg['n_total'] > 0 else 0
    print(f"  {'G spikes per 1000 reads':<35s}  {g_per1k_h:>10.1f}  {g_per1k_r:>10.1f}  "
          f"{g_per1k_r/g_per1k_h:.2f}x" if g_per1k_h > 0 else "")
    print(f"  {'U/C spikes per 1000 reads':<35s}  {uc_per1k_h:>10.1f}  {uc_per1k_r:>10.1f}  "
          f"{uc_per1k_r/uc_per1k_h:.2f}x" if uc_per1k_h > 0 else "")

    # RNA002 comparison
    all_residues = pd.concat([rna002[s]['l1_residues'] for s in rna002])
    if len(all_residues) > 0:
        nuc_counts = all_residues['prediction'].value_counts()
        total_r2 = nuc_counts.sum()
        g_r2 = nuc_counts.get('G', 0)
        print(f"\n  RNA002 Hct116 L1 G fraction: {g_r2}/{total_r2} = {100*g_r2/total_r2:.1f}%")
        print(f"  RNA004 HCT116 L1 G (HIGH) fraction: {g_hct}/{g_hct+uc_hct} = {g_h_pct:.1f}%")


# ══════════════════════════════════════════════════════════════════════
#  EXPORT RESULTS
# ══════════════════════════════════════════════════════════════════════

def export_results(samples, rna002):
    """Export key results as TSV."""
    outdir = f'{ISOTENT_L1}/analysis/01_exploration/topic_05_cellline'

    hct = samples['HCT116']
    rg = samples['HCT116_RG7834']

    # Per-read spike summary
    rows = []
    for name, s in [('HCT116', hct), ('HCT116_RG7834', rg)]:
        m = s['merged']
        for _, row in m.iterrows():
            g_spikes = sum(1 for o in row['spike_orientations'] if o == 1)
            uc_spikes = sum(1 for o in row['spike_orientations'] if o == 0)
            rows.append({
                'sample': name,
                'read_id': row['read_id'],
                'gene_id': row['gene_id'],
                'age': row['age'],
                'polya_length': row['polya_length'],
                'spike_count': row['spike_count'],
                'g_spike_count': g_spikes,
                'uc_spike_count': uc_spikes,
                'has_spike': row['has_spike'],
            })
    out_df = pd.DataFrame(rows)
    out_path = f'{outdir}/l1_gtailing_rg7834_per_read.tsv'
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\n  Exported: {out_path} ({len(out_df)} reads)")

    # Per-spike detail
    spike_rows = []
    for name, s in [('HCT116', hct), ('HCT116_RG7834', rg)]:
        if len(s['spikes']) > 0:
            sp = s['spikes'].copy()
            sp['sample'] = name
            spike_rows.append(sp)
    if spike_rows:
        spike_df = pd.concat(spike_rows)
        spike_path = f'{outdir}/l1_gtailing_rg7834_per_spike.tsv'
        spike_df.to_csv(spike_path, sep='\t', index=False)
        print(f"  Exported: {spike_path} ({len(spike_df)} spikes)")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("L1 G-tailing (TENT4A/B) Analysis: RNA004 Polyaglot + RG7834")
    print(f"{'='*70}")

    # Load data
    print("\nLoading RNA004 polyaglot + L1 data...")
    samples = load_rna004_data()

    print("Loading RNA002 ninetails data...")
    rna002 = load_rna002_data()

    # Run analyses
    analysis1_spike_rate(samples)
    analysis2_rna002_cross(samples, rna002)
    analysis3_signal_characteristics(samples)

    # Summary & Export
    print_summary(samples, rna002)
    export_results(samples, rna002)

    print(f"\n{'='*70}")
    print("  Analysis complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
