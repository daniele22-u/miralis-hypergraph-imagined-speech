"""
build_tensors.py — Pipeline SSH per la creazione dei tensori EEG
================================================================
Sostituisce l'esecuzione sequenziale di:
  EEG_01_pipeline_metadata_features_analysis.ipynb  (step 1: metadata CSV)
  EEG_02_tensors_and_graph.ipynb                     (step 2: tensori .pt)

Usage:
    python scripts/build_tensors.py --h5_dir /path/to/h5_files
    python scripts/build_tensors.py --h5_dir /path/to/h5 --mode aggregated --resume
    python scripts/build_tensors.py --h5_dir /path/to/h5 --mode both --n_jobs 4

Argomenti:
    --h5_dir    Path alla directory contenente i file .h5 (es. 00_01.h5, 00_02.h5, ...)
    --mode      Tipo di tensori da creare: aggregated | time | both  (default: aggregated)
    --resume    Se specificato, salta soggetti già processati
    --n_jobs    Numero di worker paralleli per la feature extraction (default: 1)
    --out_dir   Override per la directory di output (default: data/processed/)
    --skip_meta Se specificato, salta la creazione del metadata CSV (usa quello esistente)

Output:
    data/interim/eeg_metadata.csv
    data/processed/subject_tensors/subject_tensors_aggregated_epoch/subject_XX.pt
    data/processed/subject_tensors/subject_tensors_time/subject_XX.pt
"""

import argparse
import json
import re
import sys
import unicodedata
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import h5py
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 0. Project root detection
# ─────────────────────────────────────────────────────────────────────────────

def find_project_root() -> Path:
    """Risale dalla directory corrente fino a trovare il .git."""
    cwd = Path(__file__).resolve().parent.parent  # scripts/ -> project_root
    candidates = [cwd] + list(cwd.parents)
    for p in candidates:
        if (p / ".git").exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"
CONFIGS_DIR  = PROJECT_ROOT / "configs" / "label_schemes"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Label canonicalization (da EEG_01)
# ─────────────────────────────────────────────────────────────────────────────

TRUE_LABELS = [
    "arrivare","andare","aspettare","avere","capire","chiamare","chiedere","conoscere","dare","dire","dovere",
    "essere","fare","mettere","potere","prendere","sapere","sentire","trovare","venire","aprire","chiudere",
    "mangiare","bere","accendere","spegnere","volere","bene","si","no","più","poco","molto","sempre","adesso",
    "poi","male","sopra","sotto","destra","sinistra","avanti","indietro","oggi","domani","ieri","forse","prima",
    "perchè","anche","come","però","quindi","quando","dove","se","oppure","io","lui","lei","noi","voi","loro",
    "tu","questo","quello","buono","bello","cattivo","brutto","grande","piccolo","nuovo","vecchio","cosa","parte",
    "anno","casa","problema","aiuto","tempo","lavoro","persona","acqua","cibo","bisogno","donna","uomo","gruppo",
    "guerra","idea","macchina","mano","oggetto","telefono","computer","domanda","uno","mille","paura","ansia",
    "gioia","felicità","tristezza","serenità","amore","morte","bagno","dolore","riposo",
]


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _robust_decode(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                s = x.decode(enc)
                break
            except Exception:
                continue
        else:
            s = x.decode("utf-8", "ignore")
    else:
        s = str(x)
    s = s.replace("\x00", "").strip()
    if "Ã" in s or "Â" in s:
        try:
            s = s.encode("latin-1", "ignore").decode("utf-8", "ignore")
        except Exception:
            pass
    return unicodedata.normalize("NFC", s)


def canonicalize_label(raw) -> str:
    s = _robust_decode(raw).lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"_img$", "", s)
    s = (s.replace("\u2018", "'").replace("\u2019", "'")
          .replace("\u201c", '"').replace("\u201d", '"'))
    s = s.replace("''", "'").replace("´", "'").replace("`", "'").strip()
    fixes = {
        "perche": "perchè", "perche'": "perchè", "perch'e": "perchè",
        "pero": "però", "piu": "più", "serenita": "serenità", "felicita": "felicità",
    }
    if s in fixes:
        return fixes[s]
    if s in TRUE_LABELS:
        return s
    s_ascii = _strip_diacritics(s)
    candidates_ascii = [_strip_diacritics(t) for t in TRUE_LABELS]
    m1 = difflib.get_close_matches(s_ascii, candidates_ascii, n=1, cutoff=0.7)
    if m1:
        return TRUE_LABELS[candidates_ascii.index(m1[0])]
    m2 = difflib.get_close_matches(s, TRUE_LABELS, n=1, cutoff=0.6)
    if m2:
        return m2[0]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 2. Step 1 — Build metadata CSV (da EEG_01)
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata(h5_dir: Path, label2idx: Dict[str, int]) -> pd.DataFrame:
    """
    Scansiona tutti i file .h5 in h5_dir e costruisce il DataFrame dei metadati.
    Ogni riga = una epoch con subject_id, session_id, epoch_idx, label_name, ...
    """
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        print(f"⚠  Nessun file .h5 trovato in {h5_dir}")
        return pd.DataFrame()

    print(f"Trovati {len(h5_files)} file .h5 in {h5_dir}")
    rows = []
    skipped = []

    for file in h5_files:
        stem = file.stem
        if "_" not in stem:
            skipped.append(file.name)
            print(f"  skip (nome inatteso): {file.name}")
            continue
        subject_id, session_id = stem.rsplit("_", 1)
        try:
            with h5py.File(file, "r") as f:
                if "data" not in f or "labels" not in f:
                    skipped.append(file.name)
                    print(f"  skip (key mancanti): {file.name}")
                    continue
                labels = f["labels"][:]
                n_epochs, n_channels, n_samples = f["data"].shape

            for i, lbl_raw in enumerate(labels):
                label_name = canonicalize_label(lbl_raw)
                rows.append({
                    "subject_id":  subject_id,
                    "session_id":  session_id,
                    "epoch_idx":   i,
                    "label_name":  label_name,
                    "label_idx":   int(label2idx.get(label_name, -1)),
                    "n_channels":  n_channels,
                    "n_samples":   n_samples,
                    "fs":          256,
                    "path_h5":     str(file),
                })
            print(f"  ✓ {file.name}: {n_epochs} epochs")
        except Exception as e:
            print(f"  ✗ {file.name}: {e}")
            skipped.append(file.name)

    if skipped:
        print(f"\n⚠  Skippati {len(skipped)} file")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature extraction (da EEG_02 / comprehensive_features.py)
# ─────────────────────────────────────────────────────────────────────────────

def _import_feature_extractors():
    """Importa le funzioni di feature extraction dal progetto."""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from scripts.features.comprehensive_features import (
            extract_temporal_features,
            extract_spectral_features,
            extract_functional_features,
        )
        return extract_temporal_features, extract_spectral_features, extract_functional_features
    except ImportError as e:
        print(f"✗ Impossibile importare comprehensive_features: {e}")
        print("  Assicurati che scripts/features/comprehensive_features.py esista.")
        sys.exit(1)


_FEATURE_COLS = None

def extract_40_features_per_channel(x_win: np.ndarray, fs: int = 256):
    """
    x_win: (59, n_samples)
    return: feats (59, 40), feature_cols list[str]
    """
    global _FEATURE_COLS
    ext_temp, ext_spec, ext_func = _import_feature_extractors()

    n_ch = x_win.shape[0]
    rows = []
    for ch in range(n_ch):
        sig = x_win[ch]
        d = {}
        d.update(ext_temp(sig))
        d.update(ext_spec(sig, fs=fs))
        d.update(ext_func(x_win, ch))
        rows.append(d)

    if _FEATURE_COLS is None:
        _FEATURE_COLS = sorted(rows[0].keys())
        if len(_FEATURE_COLS) != 40:
            raise ValueError(f"Expected 40 features, got {len(_FEATURE_COLS)}")

    feats = np.zeros((n_ch, 40), dtype=np.float32)
    for ch in range(n_ch):
        feats[ch] = np.array([rows[ch][k] for k in _FEATURE_COLS], dtype=np.float32)
    return feats, _FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# 4. Canali (da EEG_02)
# ─────────────────────────────────────────────────────────────────────────────

def load_channel_names(project_root: Path) -> Tuple[List[int], List[str]]:
    """Legge ebneuro.locs e restituisce keep_idx, keep_names (escluso A1/A2)."""
    locs_path = project_root / "src" / "io" / "ebneuro.locs"
    names_61 = []
    with open(locs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                names_61.append(parts[3])
    names_61 = names_61[:61]
    EXCLUDE = {"A1", "A2"}
    keep_idx   = [i for i, n in enumerate(names_61) if n not in EXCLUDE]
    keep_names = [names_61[i] for i in keep_idx]
    return keep_idx, keep_names


def parse_session_id(v) -> int:
    if isinstance(v, str):
        v = v.strip().replace("S", "")
    return int(v)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Step 2 — Build tensori aggregated-epoch (da EEG_02)
# ─────────────────────────────────────────────────────────────────────────────

def build_aggregated_tensors(meta: pd.DataFrame, keep_idx: List[int], keep_names: List[str],
                              out_dir: Path, resume: bool, fs: int = 256):
    """
    Crea tensori (n_epochs, 59, 40) per soggetto.
    Salva in out_dir/subject_XX.pt
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label_col = "label_idx" if "label_idx" in meta.columns else "label_id"
    subjects = sorted(meta["subject_id"].astype(int).unique())
    print(f"\n=== Aggregated tensors → {out_dir} ===")
    print(f"Soggetti da processare: {len(subjects)}")

    total_kept = 0
    total_skipped_subj = 0

    for subject_id in subjects:
        out_path = out_dir / f"subject_{int(subject_id):02d}.pt"
        if resume and out_path.exists():
            total_skipped_subj += 1
            print(f"  ↷ skip subject {int(subject_id):02d} (già esiste)")
            continue

        df_s = meta[meta["subject_id"].astype(int) == subject_id]
        X_epochs, y_list, subj_list, sess_list, ep_list = [], [], [], [], []
        global_feature_cols = None
        skipped_epochs = 0

        for _, r in df_s.iterrows():
            epoch_idx = int(r["epoch_idx"])
            h5_path   = r["path_h5"]
            try:
                with h5py.File(h5_path, "r") as f:
                    n_in_file = f["data"].shape[0]
                    if epoch_idx < 0 or epoch_idx >= n_in_file:
                        skipped_epochs += 1
                        continue
                    x = f["data"][epoch_idx].astype(np.float32)
            except Exception as e:
                print(f"    ✗ {h5_path}: {e}")
                skipped_epochs += 1
                continue

            x = x[keep_idx, :]  # (59, T)
            feats, feature_cols = extract_40_features_per_channel(x, fs)
            if global_feature_cols is None and feature_cols is not None:
                global_feature_cols = list(feature_cols)

            feats = np.asarray(feats, dtype=np.float32)
            if feats.shape != (len(keep_idx), 40):
                skipped_epochs += 1
                continue

            X_epochs.append(torch.from_numpy(feats).float())
            y_list.append(int(r[label_col]))
            subj_list.append(int(r["subject_id"]))
            sess_list.append(parse_session_id(r["session_id"]))
            ep_list.append(epoch_idx)

        if not X_epochs:
            print(f"  ✗ subject {subject_id}: zero epoch valide, non salvato")
            continue

        X = torch.stack(X_epochs, dim=0)
        save_obj = {
            "X":                X,
            "y":                torch.tensor(y_list, dtype=torch.long),
            "subject_id":       torch.tensor(subj_list, dtype=torch.long),
            "session_id":       torch.tensor(sess_list, dtype=torch.long),
            "epoch_idx":        torch.tensor(ep_list, dtype=torch.long),
            "fs":               fs,
            "ch_names":         keep_names,
            "excluded_channels": ["A1", "A2"],
            "mode":             "aggregated_epoch_single_window",
        }
        if global_feature_cols:
            save_obj["feature_cols"] = global_feature_cols
        torch.save(save_obj, out_path)
        total_kept += len(X_epochs)
        print(f"  ✓ subject_{int(subject_id):02d}.pt | X={tuple(X.shape)} | skipped_epochs={skipped_epochs}")

    print(f"\nAggregated done. Skipped subjects (già esistevano): {total_skipped_subj} | Kept epochs: {total_kept}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Step 2b — Build tensori time-resolved (da EEG_02)
# ─────────────────────────────────────────────────────────────────────────────

def build_time_tensors(meta: pd.DataFrame, keep_idx: List[int], keep_names: List[str],
                        out_dir: Path, resume: bool, fs: int = 256,
                        win_len_s: float = 0.5, step_s: float = 0.25):
    """
    Crea tensori (n_epochs, n_windows, 59, 40) per soggetto.
    Salva in out_dir/subject_XX.pt
    """
    win  = int(round(win_len_s * fs))
    step = int(round(step_s * fs))
    out_dir.mkdir(parents=True, exist_ok=True)
    label_col = "label_idx" if "label_idx" in meta.columns else "label_id"
    subjects = sorted(meta["subject_id"].astype(int).unique())
    print(f"\n=== Time-resolved tensors → {out_dir} ===")
    print(f"win={win} samples ({win_len_s}s) | step={step} samples ({step_s}s)")
    print(f"Soggetti da processare: {len(subjects)}")

    total_kept = 0
    total_skipped_subj = 0

    for subject_id in subjects:
        out_path = out_dir / f"subject_{int(subject_id):02d}.pt"
        if resume and out_path.exists():
            total_skipped_subj += 1
            print(f"  ↷ skip subject {int(subject_id):02d} (già esiste)")
            continue

        df_s = meta[meta["subject_id"].astype(int) == subject_id]
        X_epochs, y_list, subj_list, sess_list, ep_list = [], [], [], [], []
        global_feature_cols = None
        skipped_epochs = 0

        for _, r in df_s.iterrows():
            epoch_idx = int(r["epoch_idx"])
            h5_path   = r["path_h5"]
            try:
                with h5py.File(h5_path, "r") as f:
                    n_in_file = f["data"].shape[0]
                    if epoch_idx < 0 or epoch_idx >= n_in_file:
                        skipped_epochs += 1
                        continue
                    x = f["data"][epoch_idx].astype(np.float32)
            except Exception as e:
                skipped_epochs += 1
                continue

            x = x[keep_idx, :]  # (59, T)
            T = x.shape[1]
            starts = list(range(0, T - win + 1, step))
            if not starts:
                skipped_epochs += 1
                continue

            feat_seq = []
            bad = False
            for s in starts:
                x_win = x[:, s:s + win]
                feats, feature_cols = extract_40_features_per_channel(x_win, fs)
                if global_feature_cols is None and feature_cols:
                    global_feature_cols = list(feature_cols)
                feats = np.asarray(feats, dtype=np.float32)
                if feats.shape != (len(keep_idx), 40):
                    bad = True
                    break
                feat_seq.append(feats)

            if bad or not feat_seq:
                skipped_epochs += 1
                continue

            feat_seq_np = np.stack(feat_seq, axis=0)  # (n_windows, 59, 40)
            X_epochs.append(torch.from_numpy(feat_seq_np).float())
            y_list.append(int(r[label_col]))
            subj_list.append(int(r["subject_id"]))
            sess_list.append(parse_session_id(r["session_id"]))
            ep_list.append(epoch_idx)

        if not X_epochs:
            print(f"  ✗ subject {subject_id}: zero epoch valide, non salvato")
            continue

        X = torch.stack(X_epochs, dim=0)  # (n_epochs, n_windows, 59, 40)
        save_obj = {
            "X":                X,
            "y":                torch.tensor(y_list, dtype=torch.long),
            "subject_id":       torch.tensor(subj_list, dtype=torch.long),
            "session_id":       torch.tensor(sess_list, dtype=torch.long),
            "epoch_idx":        torch.tensor(ep_list, dtype=torch.long),
            "fs":               fs,
            "win_len_s":        win_len_s,
            "step_s":           step_s,
            "ch_names":         keep_names,
            "excluded_channels": ["A1", "A2"],
            "mode":             "time_resolved_windowed_features",
        }
        if global_feature_cols:
            save_obj["feature_cols"] = global_feature_cols
        torch.save(save_obj, out_path)
        total_kept += len(X_epochs)
        print(f"  ✓ subject_{int(subject_id):02d}.pt | X={tuple(X.shape)} | skipped_epochs={skipped_epochs}")

    print(f"\nTime done. Skipped subjects (già esistevano): {total_skipped_subj} | Kept epochs: {total_kept}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline SSH: H5 → metadata CSV → tensori .pt",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--h5_dir",    required=True,
                        help="Directory contenente i file .h5 (es. 00_01.h5, 01_03.h5, ...)")
    parser.add_argument("--mode",      default="aggregated",
                        choices=["aggregated", "time", "both"],
                        help="Tipo di tensori: aggregated (59,40) | time (n_win,59,40) | both")
    parser.add_argument("--resume",    action="store_true",
                        help="Salta soggetti già processati (file .pt esistenti)")
    parser.add_argument("--skip_meta", action="store_true",
                        help="Salta la creazione del metadata CSV (usa quello esistente)")
    parser.add_argument("--out_dir",   default=None,
                        help="Override directory di output (default: data/processed/subject_tensors/)")
    parser.add_argument("--n_jobs",    type=int, default=1,
                        help="Worker paralleli (attualmente non utilizzato — riservato)")
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    if not h5_dir.exists():
        print(f"✗ h5_dir non trovato: {h5_dir}")
        sys.exit(1)

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"H5 directory: {h5_dir}")
    print(f"Mode: {args.mode}  |  Resume: {args.resume}")

    # ── Crea data/interim/ se non esiste ─────────────────────────────────────
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # ── Carica label2idx da configs/ o interim/ ───────────────────────────────
    label2idx_path = CONFIGS_DIR / "label2idx.json"
    if not label2idx_path.exists():
        label2idx_path = INTERIM_DIR / "label2idx.json"
    if label2idx_path.exists():
        with open(label2idx_path, encoding="utf-8") as f:
            label2idx = {canonicalize_label(k): int(v) for k, v in json.load(f).items()}
        print(f"✓ Caricato label2idx da {label2idx_path} ({len(label2idx)} entries)")
    else:
        label2idx = {canonicalize_label(lbl): idx for idx, lbl in enumerate(TRUE_LABELS)}
        print(f"✓ label2idx costruito da TRUE_LABELS ({len(label2idx)} entries)")

    # ── Step 1: Metadata CSV ──────────────────────────────────────────────────
    meta_csv = INTERIM_DIR / "eeg_metadata.csv"
    if args.skip_meta and meta_csv.exists():
        print(f"\n─── Step 1 saltato (--skip_meta) — uso {meta_csv} ───")
        meta = pd.read_csv(meta_csv)
    else:
        print("\n─── Step 1: Build metadata CSV ───")
        meta = build_metadata(h5_dir, label2idx)
        if meta.empty:
            print("✗ Nessun dato trovato. Controlla --h5_dir.")
            sys.exit(1)
        meta.to_csv(meta_csv, index=False)
        print(f"✓ Salvato {meta_csv} ({len(meta)} righe, {meta['subject_id'].nunique()} soggetti)")

    # Normalizza subject_id come intero
    meta["subject_id"] = pd.to_numeric(meta["subject_id"], errors="coerce").astype("Int64")
    meta = meta.dropna(subset=["subject_id", "epoch_idx"])
    meta["subject_id"] = meta["subject_id"].astype(int)

    # ── Canali ────────────────────────────────────────────────────────────────
    keep_idx, keep_names = load_channel_names(PROJECT_ROOT)
    print(f"\n✓ Canali: {len(keep_names)} (esclusi A1, A2)")

    # ── Step 2: Tensori ───────────────────────────────────────────────────────
    base_out = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "data" / "processed" / "subject_tensors"

    if args.mode in ("aggregated", "both"):
        out_agg = base_out / "subject_tensors_aggregated_epoch"
        build_aggregated_tensors(meta, keep_idx, keep_names, out_agg, args.resume)

    if args.mode in ("time", "both"):
        out_time = base_out / "subject_tensors_time"
        build_time_tensors(meta, keep_idx, keep_names, out_time, args.resume)

    print("\n✅ Pipeline completata.")
    print(f"   Output: {base_out}")
    print(f"   Metadata: {meta_csv}")


if __name__ == "__main__":
    main()
