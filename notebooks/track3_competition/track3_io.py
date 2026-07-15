"""
track3_io.py — Caricamento dei file .mat del Track#3 e delle etichette di test.

Complicazioni gestite qui (verificate ispezionando i file reali):
  * Train/Validation set  -> MATLAB v7  (scipy.io.loadmat), epo.x = (time, channels, trials)
  * Test set              -> MATLAB v7.3 (HDF5, h5py),        epo.x = (trials, channels, time)
    e le clab/className sono reference HDF5 da dereferenziare.
  * epo.y nel Test set è OSCURATO (dummy): le TRUE label si leggono da
    'Track3_Answer Sheet_Test.xlsx' (nella cartella Test set).
  * epo.y in train/val è one-hot (5, n_trials) -> convertito ad argmax (0..4).

Tutte le funzioni restituiscono X con orientamento UNIFICATO: (trials, channels, time), float32.
Le label sono 0..4 (0=Hello,1=Helpme,2=Stop,3=Thankyou,4=Yes), coerenti con CLASS_NAMES.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.io as sio

from dataclasses import dataclass
import track3_config as C


@dataclass
class SubjectData:
    X: np.ndarray          # (trials, channels, time) float32
    y: np.ndarray          # (trials,) int in 0..4
    fs: float
    t: np.ndarray          # (time,) ms
    clab: list[str]        # 64 nomi canale
    pos_3d: np.ndarray | None  # (channels, 3) coord 3D elettrodi (da mnt), o None
    split: str
    subject: int


# ---------------------------------------------------------------------------
# Helpers HDF5 (per i file v7.3 del Test set)
# ---------------------------------------------------------------------------
def _h5_str(h5file, ref) -> str:
    """Dereferenzia un riferimento HDF5 a stringa MATLAB (array di char codes uint16)."""
    arr = np.array(h5file[ref]).flatten()
    return "".join(chr(int(c)) for c in arr if int(c) != 0)


def _h5_clab(h5file, epo_group) -> list[str]:
    refs = np.array(epo_group["clab"]).flatten()
    return [_h5_str(h5file, r) for r in refs]


def _load_v73(path, epo_key: str):
    import h5py
    with h5py.File(path, "r") as h:
        epo = h[epo_key]
        x = np.array(epo["x"])           # (trials, channels, time) in h5py
        # y può essere oscurato: proviamo comunque a leggerlo
        try:
            y_raw = np.array(epo["y"])
        except Exception:
            y_raw = None
        fs = float(np.array(epo["fs"]).squeeze())
        t = np.array(epo["t"]).squeeze()
        clab = _h5_clab(h, epo)
        pos_3d = None
        if "mnt" in h:
            try:
                pos_3d = np.array(h["mnt"]["pos_3d"])
                if pos_3d.shape[0] != C.N_CHANNELS and pos_3d.shape[-1] == C.N_CHANNELS:
                    pos_3d = pos_3d.T
            except Exception:
                pos_3d = None
    return x.astype(np.float32), y_raw, fs, t, clab, pos_3d


def _load_v7(path, epo_key: str):
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    epo = m[epo_key]
    x = np.array(epo.x)                  # (time, channels, trials)
    # -> (trials, channels, time)
    x = np.transpose(x, (2, 1, 0)).astype(np.float32)
    y_raw = np.array(epo.y)              # (5, trials) one-hot
    fs = float(epo.fs)
    t = np.array(epo.t)
    clab = [str(c) for c in epo.clab]
    pos_3d = None
    if "mnt" in m:
        mnt = m["mnt"]
        try:
            pos_3d = np.array(mnt.pos_3d)
            if pos_3d.shape[0] != C.N_CHANNELS and pos_3d.shape[-1] == C.N_CHANNELS:
                pos_3d = pos_3d.T
        except Exception:
            pos_3d = None
    return x, y_raw, fs, t, clab, pos_3d


def _onehot_to_labels(y_raw) -> np.ndarray:
    """(5, N) one-hot -> (N,) argmax in 0..4."""
    y = np.asarray(y_raw)
    if y.ndim == 2:
        # atteso (5, N); se arriva (N,5) trasponi
        if y.shape[0] != C.N_CLASSES and y.shape[1] == C.N_CLASSES:
            y = y.T
        return y.argmax(axis=0).astype(np.int64)
    return y.astype(np.int64).ravel()


# ---------------------------------------------------------------------------
# Answer sheet del Test set (TRUE label)
# ---------------------------------------------------------------------------
def load_test_labels() -> dict[int, np.ndarray]:
    """
    Legge 'Track3_Answer Sheet_Test.xlsx' e restituisce {subject: array(50,) label 0..4}.
    Layout del foglio (header=None):
      riga 1: Data_Sample01 in col1, Data_Sample02 in col3, ...  (Data_Sample k in col 2k-1)
      riga 2: 'Trial #' | 'True Label' ripetuto per soggetto
      righe 3+: 50 righe con (trial#, label 1..5) per soggetto
    Per il soggetto k: colonna trial = 2k-1, colonna label = 2k.
    """
    path = C.data_dir("test") / C.ANSWER_SHEET_TEST
    df = pd.read_excel(path, sheet_name="Track3", header=None)
    out: dict[int, np.ndarray] = {}
    for k in C.SUBJECTS:
        lab_col = 2 * k               # 0-indexed: label del soggetto k
        labels = pd.to_numeric(df.iloc[:, lab_col], errors="coerce").dropna()
        labels = labels[labels.between(1, 5)].to_numpy().astype(np.int64)
        if len(labels) != C.N_TRIALS["test"]:
            # fallback: prendi le prime 50 valide
            labels = labels[: C.N_TRIALS["test"]]
        out[k] = labels - 1           # event code 1..5 -> label 0..4
    return out


_TEST_LABELS_CACHE: dict[int, np.ndarray] | None = None


def _get_test_labels(subject: int) -> np.ndarray:
    global _TEST_LABELS_CACHE
    if _TEST_LABELS_CACHE is None:
        _TEST_LABELS_CACHE = load_test_labels()
    return _TEST_LABELS_CACHE[subject]


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------
def load_subject(split: str, subject: int) -> SubjectData:
    """Carica un soggetto per uno split ('train'|'val'|'test'). X = (trials, ch, time)."""
    assert split in ("train", "val", "test"), split
    path = C.subject_file(split, subject)
    epo_key = C.EPO_KEY[split]
    # prova scipy (v7), fallback h5py (v7.3)
    try:
        x, y_raw, fs, t, clab, pos_3d = _load_v7(path, epo_key)
    except NotImplementedError:
        x, y_raw, fs, t, clab, pos_3d = _load_v73(path, epo_key)

    if split == "test":
        y = _get_test_labels(subject)           # true label dall'answer sheet
        if len(y) != x.shape[0]:
            y = y[: x.shape[0]]
    else:
        y = _onehot_to_labels(y_raw)

    return SubjectData(X=x, y=y, fs=fs, t=np.asarray(t).ravel(),
                       clab=list(clab), pos_3d=pos_3d, split=split, subject=subject)


def load_subject_all(subject: int, merge_val_into_train: bool = False):
    """
    Comodo: ritorna (train, val, test) SubjectData per un soggetto.
    Se merge_val_into_train=True, concatena val dentro train (consentito dal regolamento).
    """
    tr = load_subject("train", subject)
    va = load_subject("val", subject)
    te = load_subject("test", subject)
    if merge_val_into_train:
        tr = SubjectData(
            X=np.concatenate([tr.X, va.X], 0),
            y=np.concatenate([tr.y, va.y], 0),
            fs=tr.fs, t=tr.t, clab=tr.clab, pos_3d=tr.pos_3d,
            split="train+val", subject=subject,
        )
    return tr, va, te


def canonical_clab() -> list[str]:
    """Ordine canonico dei 64 canali (letto dal train di S01)."""
    return load_subject("train", 1).clab


def canonical_positions() -> tuple[list[str], np.ndarray]:
    """(clab, pos_3d) da mnt del train S01, per topomap e REVE."""
    sd = load_subject("train", 1)
    return sd.clab, sd.pos_3d


if __name__ == "__main__":
    print(C.summary())
    if C.DATA_ROOT is None:
        raise SystemExit(C._no_data_msg())
    tr, va, te = load_subject_all(1)
    for sd in (tr, va, te):
        print(f"{sd.split:6s} X={sd.X.shape} y={sd.y.shape} classi={np.bincount(sd.y, minlength=5)}")
    print("clab[:6]:", tr.clab[:6], "... pos_3d:", None if tr.pos_3d is None else tr.pos_3d.shape)
