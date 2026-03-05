"""Run a smoke test of comprehensive feature extraction on 3 H5 files.

This script selects the first 3 unique `path_h5` entries from
`data/interim/eeg_metadata.csv`, writes a small subset CSV, and calls
`iterate_all_files_and_extract` from `scripts.features.comprehensive_features`.

Result is saved to `data/interim/comprehensive_sample.csv`.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path if needed (scripts are in repo root)
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    from scripts.features.comprehensive_features import iterate_all_files_and_extract
except Exception as e:
    print(f"✗ Could not import comprehensive extractor: {e}")
    raise


def main():
    interim = project_root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    meta_csv = interim / "eeg_metadata.csv"
    if not meta_csv.exists():
        print(f"⚠ Metadata CSV not found at {meta_csv}")
        return 1

    meta = pd.read_csv(meta_csv)
    if 'path_h5' not in meta.columns:
        print("⚠ `path_h5` column missing from metadata")
        return 1

    unique_paths = meta['path_h5'].dropna().unique().tolist()
    if not unique_paths:
        print("⚠ No H5 paths found in metadata")
        return 1

    # Choose first 3 unique paths
    sample_paths = unique_paths[:3]
    print(f"Selected {len(sample_paths)} files for smoke test:\n  " + "\n  ".join(sample_paths))

    # Build subset metadata dataframe
    subset_meta = meta[meta['path_h5'].isin(sample_paths)].copy()
    subset_csv = interim / "eeg_metadata_subset3.csv"
    subset_meta.to_csv(subset_csv, index=False)

    # Eloc path (use eloc used elsewhere in repo)
    eloc_path = project_root / "src" / "io" / "ebneuro.locs"

    print(f"\nRunning comprehensive extractor on sample metadata: {subset_csv}")
    try:
        df_feats = iterate_all_files_and_extract(subset_csv, fs=256, eloc_path=eloc_path)
    except Exception as e:
        print(f"✗ Error during extraction: {e}")
        raise

    out_csv = interim / "comprehensive_sample.csv"
    df_feats.to_csv(out_csv, index=False)

    print(f"\n✓ Saved sample features to {out_csv} ({len(df_feats)} rows, {len(df_feats.columns)} cols)")
    print("\nPreview (first 5 rows):")
    print(df_feats.head().to_string(index=False))

    return 0


if __name__ == '__main__':
    exit(main())
