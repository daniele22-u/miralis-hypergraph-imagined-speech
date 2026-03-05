"""
EEG Bandpower Extraction Script
--------------------------------
Computes absolute and relative band powers for all EEG epochs stored in HDF5 files.
Loads true channel names from a .eloc montage file for correct electrode labeling.
"""

import h5py, mne, numpy as np, pandas as pd
from pathlib import Path
from typing import Optional

def load_channel_names_from_eloc(eloc_path: Path):
    names = []
    try:
        # Read eloc file line-by-line to avoid pandas CSV parsing issues
        with open(eloc_path, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                # assume last token is the channel label (common .eloc convention)
                if parts:
                    names.append(parts[-1])
    except Exception:
        # fallback to pandas as a last resort
        try:
            df = pd.read_csv(eloc_path, sep=r"\s+", header=None, engine="python", comment="#")
            names = df.iloc[:, -1].astype(str).tolist()
        except Exception:
            names = []
    return names

def load_epochs_from_h5(path_h5: Path, fs: int = 256, eloc_path: Optional[Path] = None):
    with h5py.File(path_h5, "r") as f:
        data = f["data"][:] # type: ignore
        labels = f["labels"][:] # type: ignore
        subj = f["subject"][()] # type: ignore
    # Ensure labels length matches number of epochs in data
    n_epochs = data.shape[0]
    if len(labels) != n_epochs:
        print(f"⚠ Inconsistent H5: {path_h5.name} has data epochs={n_epochs} but labels len={len(labels)}; synchronizing labels to data")
        if len(labels) > n_epochs:
            labels = labels[:n_epochs]
        else:
            # pad with empty labels
            pad = [''] * (n_epochs - len(labels))
            try:
                labels = list(labels) + pad
            except Exception:
                labels = pad
    if eloc_path and eloc_path.exists():
        ch_names = load_channel_names_from_eloc(eloc_path)
    else:
        ch_names = [f"EEG{i+1}" for i in range(data.shape[1])] # type: ignore
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    ep = mne.EpochsArray(data, info)
    lbl = [l.decode("utf-8") if isinstance(l, (bytes, bytearray)) else str(l) for l in labels] # type: ignore
    ep.metadata = pd.DataFrame({"label_name": lbl, "subject_id": str(subj), "epoch_idx": np.arange(len(lbl))})
    return ep

def compute_psd_epochs(epochs, fmin=1., fmax=45., n_fft=512):
    n_times = epochs.get_data().shape[-1]
    n_fft = min(n_fft, n_times)
    spec = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
    return spec.get_data(), spec.freqs

def bandpowers_from_psd(psd, freqs, bands):
    bp = {}
    for name, (f1, f2) in bands.items():
        idx = (freqs >= f1) & (freqs <= f2)
        bp[name] = np.trapezoid(psd[..., idx], freqs[idx], axis=-1)
    total = np.trapezoid(psd, freqs, axis=-1)
    bp_rel = {f"{k}_rel": (v / np.maximum(total, 1e-12)) for k, v in bp.items()}
    return bp | bp_rel, total


def _sync_metadata_with_h5(meta_df: pd.DataFrame, meta_csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Ensure metadata rows for each `path_h5` do not exceed the number of epochs in the H5 file.
    If metadata has more rows than the H5 `labels` length, trim the metadata to the H5 epoch count.
    """
    rows = []
    for path, grp in meta_df.groupby('path_h5'):
        p = Path(path)
        if not p.exists() and meta_csv_path is not None:
            # attempt to resolve relative/common location used in this repo
            candidate = meta_csv_path.parent.parent / 'processed' / Path(path).name
            if candidate.exists():
                p = candidate
        if not p.exists():
            # fallback: keep group as-is and warn
            print(f"⚠ H5 not found for metadata path '{path}', keeping metadata rows ({len(grp)})")
            rows.append(grp)
            continue
        try:
            with h5py.File(p, 'r') as f:
                n_epochs = int(f['labels'].shape[0])
        except Exception as e:
            print(f"⚠ Could not open {p}: {e} — keeping metadata rows ({len(grp)})")
            rows.append(grp)
            continue

        if len(grp) > n_epochs:
            print(f"⚠ Trimming metadata for {p.name}: {len(grp)} -> {n_epochs}")
            grp = grp.sort_values('epoch_idx').head(n_epochs)
        elif len(grp) < n_epochs:
            print(f"ℹ Metadata shorter than H5 for {p.name}: {len(grp)} < {n_epochs}")
        rows.append(grp)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return meta_df

def extract_bandpowers_for_epochs(epochs, bands, session_id):
    psd, freqs = compute_psd_epochs(epochs, fmin=min(b[0] for b in bands.values()),
                                    fmax=max(b[1] for b in bands.values()), n_fft=384)
    bp_dict, total = bandpowers_from_psd(psd, freqs, bands)
    n_epochs, n_ch = psd.shape[:2]
    rows = []
    for e in range(n_epochs):
        for ch in range(n_ch):
            row = {
                "subject_id": epochs.metadata["subject_id"].iloc[e],
                "session_id": session_id,
                "epoch_idx": e,
                "channel": epochs.info["ch_names"][ch],
                "label_name": epochs.metadata["label_name"].iloc[e],
                "total_power": total[e, ch]
            }
            for k, v in bp_dict.items():
                row[k] = v[e, ch]
            row["alpha_beta_ratio"] = row["alpha_rel"] / np.maximum(row["beta_rel"], 1e-12)
            row["theta_alpha_ratio"] = row["theta_rel"] / np.maximum(row["alpha_rel"], 1e-12)
            rows.append(row)
    return pd.DataFrame(rows)

def iterate_all_files_and_compute(meta_csv_path: Path, fs=256, eloc_path: Optional[Path] = None):
    # Read metadata robustly to avoid parsing errors caused by odd delimiters/quotes
    try:
        meta = pd.read_csv(meta_csv_path, engine="python", skipinitialspace=True, comment="#")
    except Exception:
        meta = pd.read_csv(meta_csv_path)
    meta["subject_id"] = meta["subject_id"].astype(str).str.strip()

    # Synchronize metadata with actual H5 epoch counts to avoid mismatches
    try:
        meta = _sync_metadata_with_h5(meta, meta_csv_path)
    except Exception as e:
        print(f"⚠ Error while synchronizing metadata with H5 files: {e}")
    bands = {"delta": (1,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30), "gamma": (30,45)}
    all_feat = []
    for subj, grp in meta.groupby("subject_id"):
        for p in sorted(grp["path_h5"].unique()):
            pth = Path(p)
            if not pth.is_absolute():
                pth = meta_csv_path.parent.parent / "processed" / Path(p).name
            if not pth.exists():
                print(f"Missing file: {pth}")
                continue
            session_id = pth.stem.split("_")[1] if "_" in pth.stem else "unknown"
            epochs = load_epochs_from_h5(pth, fs=fs, eloc_path=eloc_path)
            df_feats = extract_bandpowers_for_epochs(epochs, bands, session_id)
            all_feat.append(df_feats)
            print(f"{subj} | {session_id}: {len(df_feats)} rows")
    return pd.concat(all_feat, ignore_index=True) if all_feat else pd.DataFrame()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    meta_csv = project_root / "data" / "interim" / "eeg_metadata.csv"
    eloc_path = project_root / "src" / "io" / "ebneuro.locs"
    df_band = iterate_all_files_and_compute(meta_csv, eloc_path=eloc_path)
    df_band.to_csv(project_root / "data" / "interim" / "bandpowers.csv", index=False)
