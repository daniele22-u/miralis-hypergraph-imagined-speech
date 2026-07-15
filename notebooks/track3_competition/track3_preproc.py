"""
track3_preproc.py — Preprocessing del segnale EEG per Track#3.

Pipeline (tutti gli step sono opzionali e configurabili):
  1. bandpass (+ notch opzionale)         filtro zero-phase (MNE FIR)
  2. baseline correction                  sottrae media della finestra pre-cue (-500..0 ms)
  3. crop                                 seleziona la finestra di imagined speech (0..2000 ms)
  4. standardize (z-score per canale)     statistiche stimate SOLO sul train del soggetto
                                          (subject-dependent: niente leakage da val/test)
  5. resample                             solo per REVE (256 -> 200 Hz)

Input/Output sempre in orientamento (trials, channels, time), float32.

Feature per i modelli a grafo (DGCNN/DHSLP): differential entropy (DE) o band power per
banda e per canale -> (trials, channels, n_bands).
"""
from __future__ import annotations
import numpy as np
import track3_config as C

# Bande standard EEG (Hz). gamma limitato da Nyquist (128 Hz a 256; 100 Hz a 200).
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ---------------------------------------------------------------------------
# Filtri
# ---------------------------------------------------------------------------
def bandpass(X: np.ndarray, fs: float, l_freq: float, h_freq: float,
             notch: float | None = None) -> np.ndarray:
    """Filtro passa-banda zero-phase sull'asse tempo (ultimo). X=(trials,ch,time)."""
    import mne
    Xf = X.astype(np.float64)
    if notch is not None:
        Xf = mne.filter.notch_filter(Xf, fs, np.array([notch]), verbose="error")
    Xf = mne.filter.filter_data(Xf, sfreq=fs, l_freq=l_freq, h_freq=h_freq,
                                verbose="error")
    return Xf.astype(np.float32)


# ---------------------------------------------------------------------------
# Baseline / crop
# ---------------------------------------------------------------------------
def _time_mask(t: np.ndarray, ms_range) -> np.ndarray:
    lo, hi = ms_range
    return (t >= lo) & (t < hi)


def baseline_correct(X: np.ndarray, t: np.ndarray, baseline_ms=C.BASELINE_MS) -> np.ndarray:
    if baseline_ms is None:
        return X
    m = _time_mask(t, baseline_ms)
    base = X[:, :, m].mean(axis=2, keepdims=True)
    return X - base


def crop(X: np.ndarray, t: np.ndarray, crop_ms=C.CROP_MS):
    """Ritorna (X_cropped, t_cropped)."""
    if crop_ms is None:
        return X, t
    m = _time_mask(t, crop_ms)
    return X[:, :, m], t[m]


# ---------------------------------------------------------------------------
# Standardize (subject-dependent, fit su train)
# ---------------------------------------------------------------------------
class ChannelScaler:
    """z-score per canale. Statistiche stimate su (trials, time) del train."""
    def __init__(self):
        self.mean_ = None   # (1, ch, 1)
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-7
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Resample (per REVE: 256 -> 200 Hz)
# ---------------------------------------------------------------------------
def resample(X: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return X
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(fs_out), int(fs_in))
    up, down = int(fs_out) // g, int(fs_in) // g
    Xr = resample_poly(X, up, down, axis=2)
    return Xr.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline completa per un soggetto
# ---------------------------------------------------------------------------
def preprocess_arrays(X, t, fs, *, bandpass_hz=C.BANDPASS_HZ, notch=C.NOTCH_HZ,
                      baseline_ms=C.BASELINE_MS, crop_ms=C.CROP_MS):
    """Applica filtro -> baseline -> crop. NON standardizza (fatto dopo, con stats del train)."""
    if bandpass_hz is not None:
        X = bandpass(X, fs, bandpass_hz[0], bandpass_hz[1], notch=notch)
    X = baseline_correct(X, t, baseline_ms)
    X, t2 = crop(X, t, crop_ms)
    return X, t2


def preprocess_subject(subject: int, *, merge_val_into_train=False,
                       standardize=True, resample_to=None, **pp_kwargs):
    """
    Ritorna dict con train/val/test già preprocessati e standardizzati (subject-dependent).
    resample_to: es. 200 per REVE. pp_kwargs -> preprocess_arrays.
    """
    import track3_io as io
    tr, va, te = io.load_subject_all(subject, merge_val_into_train=merge_val_into_train)

    def _pp(sd):
        X, t2 = preprocess_arrays(sd.X, sd.t, sd.fs, **pp_kwargs)
        return X, t2

    Xtr, t2 = _pp(tr)
    Xva, _ = _pp(va)
    Xte, _ = _pp(te)

    fs_eff = tr.fs
    if resample_to is not None:
        Xtr = resample(Xtr, int(tr.fs), resample_to)
        Xva = resample(Xva, int(va.fs), resample_to)
        Xte = resample(Xte, int(te.fs), resample_to)
        fs_eff = resample_to

    scaler = None
    if standardize:
        scaler = ChannelScaler().fit(Xtr)
        Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    return {
        "subject": subject, "fs": fs_eff, "t": t2, "clab": tr.clab, "pos_3d": tr.pos_3d,
        "X_train": Xtr, "y_train": tr.y,
        "X_val": Xva, "y_val": va.y,
        "X_test": Xte, "y_test": te.y,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Feature a banda per i modelli a grafo (nodi = canali)
# ---------------------------------------------------------------------------
def band_features(X: np.ndarray, fs: float, bands=BANDS, kind="de") -> np.ndarray:
    """
    X=(trials,ch,time) -> (trials, ch, n_bands).
    kind='de'  -> differential entropy  0.5*log(2*pi*e*var)  (standard DGCNN)
    kind='pow' -> log band power
    """
    import mne
    feats = []
    for (lo, hi) in bands.values():
        Xb = mne.filter.filter_data(X.astype(np.float64), fs, lo, hi, verbose="error")
        var = Xb.var(axis=2) + 1e-8              # (trials, ch)
        if kind == "de":
            f = 0.5 * np.log(2 * np.pi * np.e * var)
        else:
            f = np.log(var)
        feats.append(f)
    return np.stack(feats, axis=-1).astype(np.float32)   # (trials, ch, n_bands)


# ---------------------------------------------------------------------------
# Salvataggio tensori preprocessati
# ---------------------------------------------------------------------------
def save_subject_tensors(subject: int, tag: str, **pp_kwargs):
    """Preprocessa e salva un .pt per soggetto in INTERIM_DIR (per riuso rapido nei notebook)."""
    import torch
    d = preprocess_subject(subject, **pp_kwargs)
    out = C.INTERIM_DIR / f"S{subject:02d}_{tag}.pt"
    torch.save({k: v for k, v in d.items() if k != "scaler"}, out)
    return out


if __name__ == "__main__":
    import numpy as np
    d = preprocess_subject(1)
    print("subject 1 preprocessed:")
    print("  X_train", d["X_train"].shape, "fs", d["fs"], "t", d["t"][[0, -1]])
    print("  mean~0 std~1 :", float(d["X_train"].mean()), float(d["X_train"].std()))
    bf = band_features(d["X_train"], d["fs"])
    print("  band_features", bf.shape)
    dr = preprocess_subject(1, resample_to=C.REVE_FS)
    print("  REVE resample X_train", dr["X_train"].shape, "fs", dr["fs"])
