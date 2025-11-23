#!/usr/bin/env python3
"""
Process a single subject: load its H5 files from a metadata CSV, extract features,
and save a per-subject CSV. Intended to be called from the notebook in parallel
via subprocess to avoid pickling issues.
"""
from pathlib import Path
import argparse
import pandas as pd
import sys


def process_subject(subject: str, meta_csv: Path, out_dir: Path, eloc_path: Path, fs: int = 256):
    try:
        from scripts.features.comprehensive_features import load_epochs_from_h5, extract_all_features_for_epochs
    except Exception as e:
        print(f"✗ Could not import comprehensive_features: {e}")
        return 1

    meta = pd.read_csv(meta_csv)
    subj_meta = meta[meta['subject_id'].astype(str) == str(subject)]
    h5_files = sorted(subj_meta['path_h5'].unique())

    rows = []
    for p in h5_files:
        pth = Path(p)
        if not pth.is_absolute():
            # assume processed directory under project if relative
            pth = Path.cwd() / 'data' / 'processed' / pth.name

        if not pth.exists():
            print(f"  ⚠ Missing file for subject {subject}: {pth}")
            continue

        session_id = pth.stem.split('_')[1] if '_' in pth.stem else 'unknown'
        try:
            epochs = load_epochs_from_h5(pth, fs=fs, eloc_path=eloc_path)
            df_feats = extract_all_features_for_epochs(epochs, session_id, extract_functional=True)
            rows.append(df_feats)
            print(f"  ✓ Extracted {len(df_feats)} rows from {pth.name}")
        except Exception as e:
            print(f"  ✗ Error processing {pth.name}: {e}")
            continue

    if rows:
        df_sub = pd.concat(rows, ignore_index=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"comprehensive_subject_{subject}.csv"
        df_sub.to_csv(out_path, index=False)
        print(f"✓ Saved {len(df_sub)} rows for subject {subject} to {out_path}")
        return 0
    else:
        print(f"⚠ No features extracted for subject {subject}")
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject', required=True)
    p.add_argument('--meta', required=True, help='Path to eeg_metadata.csv')
    p.add_argument('--out_dir', required=True, help='Directory to write per-subject CSV')
    p.add_argument('--eloc', required=True, help='Path to eloc file')
    p.add_argument('--fs', type=int, default=256)
    args = p.parse_args()

    rc = process_subject(args.subject, Path(args.meta), Path(args.out_dir), Path(args.eloc), fs=args.fs)
    sys.exit(rc)


if __name__ == '__main__':
    main()
