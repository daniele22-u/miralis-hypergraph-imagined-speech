"""
EEG Bandpower Extraction Script
--------------------------------
Computes absolute and relative band powers for all EEG epochs stored in HDF5 files.
For each subject, session, and epoch, the script calculates channel-wise power across
canonical frequency bands (delta, theta, alpha, beta, gamma) and saves the results.
"""

import h5py, mne, numpy as np, pandas as pd
from pathlib import Path

def load_epochs_from_h5(path_h5: Path, fs: int = 256):
    with h5py.File(path_h5, "r") as f:
        data = f["data"][:] # type: ignore
        labels = f["labels"][:] # type: ignore
        subj = f["subject"][()] # type: ignore
    ch_names = [f"EEG{i+1}" for i in range(data.shape[1])] # type: ignore
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    ep = mne.EpochsArray(data, info)
    lbl = [l.decode("utf-8") if isinstance(l, (bytes, bytearray)) else str(l) for l in labels] # type: ignore
    ep.metadata = pd.DataFrame({"label_name": lbl, "subject_id": str(subj), "epoch_idx": np.arange(len(lbl))})
    return ep

def compute_psd_epochs(epochs, fmin=1., fmax=45., n_fft=512):
    n_times = epochs.get_data().shape[-1]
    n_fft = min(n_fft, n_times)
    try:
        spec = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
        psd = spec.get_data()
        freqs = spec.freqs
    except AttributeError:
        from mne.time_frequency import psd_welch # type: ignore
        psd, freqs = psd_welch(epochs, fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
    return psd, freqs

def bandpowers_from_psd(psd, freqs, bands):
    bp = {}
    for name, (f1, f2) in bands.items():
        idx = (freqs >= f1) & (freqs <= f2)
        bp[name] = np.trapezoid(psd[..., idx], freqs[idx], axis=-1)
    total = np.trapezoid(psd, freqs, axis=-1)
    bp_rel = {f"{k}_rel": (v / np.maximum(total, 1e-12)) for k, v in bp.items()}
    return bp | bp_rel, total

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

def iterate_all_files_and_compute(meta_csv_path: Path, fs=256):
    meta = pd.read_csv(meta_csv_path)
    meta["subject_id"] = meta["subject_id"].astype(str).str.strip()
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
            epochs = load_epochs_from_h5(pth, fs=fs)
            df_feats = extract_bandpowers_for_epochs(epochs, bands, session_id)
            all_feat.append(df_feats)
            print(f"{subj} | {session_id}: {len(df_feats)} rows")
    return pd.concat(all_feat, ignore_index=True) if all_feat else pd.DataFrame()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    meta_csv = project_root / "data" / "interim" / "eeg_metadata.csv"
    df_band = iterate_all_files_and_compute(meta_csv)
    df_band.to_csv(project_root / "data" / "interim" / "bandpowers.csv", index=False)
    print(f"Saved bandpowers to {project_root / 'data' / 'interim' / 'bandpowers.csv'}")