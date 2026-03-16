#!/usr/bin/env python3
"""
fix_part3_cache_swap.py

Fix chemodCode swap bug in Part3 cache files.

Original bug: part3_analysis.py line 80-83 had 17802(psi)->m6A, 21891(m6A)->psi.
This means all cache files have m6a/psi columns SWAPPED.

Fix: swap column names (not values) so that:
  old 'm6a_sites_high' (actually psi data) -> new 'psi_sites_high'
  old 'psi_sites_high' (actually m6a data) -> new 'm6a_sites_high'
  old 'm6a_positions'  (actually psi data) -> new 'psi_positions'
  old 'psi_positions'  (actually m6a data) -> new 'm6a_positions'

Creates backups before modifying.
"""

import os
import glob
import shutil

TOPICDIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline"

# Column rename mapping (swap m6a <-> psi)
RENAME_MAP = {
    'm6a_sites_high': 'psi_sites_high_TEMP',
    'psi_sites_high': 'm6a_sites_high',
    'm6a_positions': 'psi_positions_TEMP',
    'psi_positions': 'm6a_positions',
}
# Second pass to finalize temps
RENAME_MAP2 = {
    'psi_sites_high_TEMP': 'psi_sites_high',
    'psi_positions_TEMP': 'psi_positions',
}


def swap_columns_in_file(filepath):
    """Swap m6a/psi column names in a TSV file header."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        print(f"  SKIP (empty): {filepath}")
        return False

    header = lines[0].rstrip('\n')
    cols = header.split('\t')

    # Check if already fixed (look for TEMP markers or already correct)
    if 'psi_sites_high_TEMP' in header:
        print(f"  SKIP (partially fixed): {filepath}")
        return False

    # Apply rename map (two-pass to avoid collision)
    new_cols = []
    for c in cols:
        new_cols.append(RENAME_MAP.get(c, c))

    # Second pass
    final_cols = []
    for c in new_cols:
        final_cols.append(RENAME_MAP2.get(c, c))

    new_header = '\t'.join(final_cols)

    if new_header == header:
        print(f"  SKIP (no m6a/psi columns): {filepath}")
        return False

    # Create backup
    backup = filepath + '.bak_preswap'
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"  Backup: {backup}")

    # Write fixed file
    lines[0] = new_header + '\n'
    with open(filepath, 'w') as f:
        f.writelines(lines)

    print(f"  FIXED: {os.path.basename(filepath)}")
    print(f"    Old header: {header}")
    print(f"    New header: {new_header}")
    return True


def main():
    print("=" * 70)
    print("Fix Part3 Cache: Swap m6A <-> psi column names")
    print("=" * 70)

    # L1 cache
    l1_cache_dir = os.path.join(TOPICDIR, "part3_l1_per_read_cache")
    ctrl_cache_dir = os.path.join(TOPICDIR, "part3_ctrl_per_read_cache")

    n_fixed = 0
    for cache_dir in [l1_cache_dir, ctrl_cache_dir]:
        print(f"\nDirectory: {cache_dir}")
        files = sorted(glob.glob(os.path.join(cache_dir, "*.tsv")))
        for f in files:
            if f.endswith('.bak_preswap'):
                continue
            if swap_columns_in_file(f):
                n_fixed += 1

    # Also fix the state classification file
    state_file = os.path.join(
        "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_04_state",
        "l1_state_classification.tsv"
    )
    if os.path.exists(state_file):
        print(f"\nState classification file:")
        # State file has more columns to swap
        with open(state_file, 'r') as f:
            lines = f.readlines()

        header = lines[0].rstrip('\n')

        # Extended rename for state classification
        state_rename = {
            'm6a_sites_high': '__PSI_SITES_HIGH__',
            'psi_sites_high': 'm6a_sites_high',
            '__PSI_SITES_HIGH__': 'psi_sites_high',
            'm6a_sites_total': '__PSI_SITES_TOTAL__',
            'psi_sites_total': 'm6a_sites_total',
            '__PSI_SITES_TOTAL__': 'psi_sites_total',
            'm6a_per_kb': '__PSI_PER_KB__',
            'psi_per_kb': 'm6a_per_kb',
            '__PSI_PER_KB__': 'psi_per_kb',
            'has_m6a': '__HAS_PSI__',
            'has_psi': 'has_m6a',
            '__HAS_PSI__': 'has_psi',
        }

        # Three-pass rename to handle collisions
        cols = header.split('\t')
        # Pass 1: m6a -> temp
        new_cols = []
        for c in cols:
            if c == 'm6a_sites_high': new_cols.append('__PSI_SITES_HIGH__')
            elif c == 'm6a_sites_total': new_cols.append('__PSI_SITES_TOTAL__')
            elif c == 'm6a_per_kb': new_cols.append('__PSI_PER_KB__')
            elif c == 'has_m6a': new_cols.append('__HAS_PSI__')
            else: new_cols.append(c)

        # Pass 2: psi -> m6a
        new_cols2 = []
        for c in new_cols:
            if c == 'psi_sites_high': new_cols2.append('m6a_sites_high')
            elif c == 'psi_sites_total': new_cols2.append('m6a_sites_total')
            elif c == 'psi_per_kb': new_cols2.append('m6a_per_kb')
            elif c == 'has_psi': new_cols2.append('has_m6a')
            else: new_cols2.append(c)

        # Pass 3: temp -> psi
        final_cols = []
        for c in new_cols2:
            if c == '__PSI_SITES_HIGH__': final_cols.append('psi_sites_high')
            elif c == '__PSI_SITES_TOTAL__': final_cols.append('psi_sites_total')
            elif c == '__PSI_PER_KB__': final_cols.append('psi_per_kb')
            elif c == '__HAS_PSI__': final_cols.append('has_psi')
            else: final_cols.append(c)

        new_header = '\t'.join(final_cols)

        if new_header != header:
            backup = state_file + '.bak_preswap'
            if not os.path.exists(backup):
                shutil.copy2(state_file, backup)
                print(f"  Backup: {backup}")

            lines[0] = new_header + '\n'
            with open(state_file, 'w') as f:
                f.writelines(lines)

            print(f"  FIXED: l1_state_classification.tsv")
            print(f"    Old: {header}")
            print(f"    New: {new_header}")
            n_fixed += 1

    print(f"\n{'='*70}")
    print(f"Total files fixed: {n_fixed}")
    print("=" * 70)


if __name__ == '__main__':
    main()
