"""
build_graphs.py — Pre-costruzione grafi ed ipergrafi per-trial

Eseguire una volta sola. Salva file .pt in data/interim/graphs/.
EEG_08, EEG_09, EEG_10, EEG_11 caricano da questi file.

OUTPUT
------
graphs/dataset_pcc_k{K}.pt          — lista Data(x, edge_index, y, subj)   [grafi]
graphs/dataset_plv_k{K}.pt          — stessa struttura, PLV-based
graphs/dataset_wpli_k{K}.pt         — stessa struttura, wPLI-based
graphs/dataset_hgnn_k{K}.pt         — lista HypergraphData(x, hyperedge_index, ...)

PRUNING
-------
  --prune-threshold T  — rimuove archi/iperspigoli dove |PCC| < T
                         (default: 0.0 = nessun pruning)
  --prune-channels     — rimuove canali con connettività media < mean - 2*std
                         (default: disabilitato)

USO
---
  conda activate daniele_311
  python scripts/build_graphs.py                     # default: k=6, nessun pruning
  python scripts/build_graphs.py --k 4 6 8           # multi-k
  python scripts/build_graphs.py --prune-threshold 0.3
  python scripts/build_graphs.py --hgnn-only         # solo ipergrafi
  python scripts/build_graphs.py --graph-only        # solo grafi semplici
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as sp_hilbert
from torch_geometric.data import Data
from tqdm import tqdm

# ── Path setup ──────────────────────────────────────────────
project_root = next(
    (p for p in [Path(__file__).resolve().parent.parent] + list(Path(__file__).resolve().parents)
     if (p / ".git").exists()),
    Path(__file__).resolve().parent.parent,
)
sys.path.insert(0, str(project_root / "scripts"))
from utils import get_keep_channels

META_CSV   = project_root / "data" / "interim" / "eeg_metadata.csv"
ELOC_PATH  = project_root / "src" / "io" / "ebneuro.locs"
GRAPHS_DIR = project_root / "data" / "interim" / "graphs"
SFREQ      = 256
PLV_BANDS  = [(4, 8), (8, 13)]   # theta + alpha


# ════════════════════════════════════════════════════════════
# FUNZIONI COSTRUZIONE GRAFO
# ════════════════════════════════════════════════════════════

def pcc_matrix(x_np: np.ndarray) -> np.ndarray:
    """Matrice |PCC| [N, N] da segnale grezzo [N, T]."""
    pcc = np.abs(np.corrcoef(x_np))
    np.fill_diagonal(pcc, 0.0)
    return pcc


def knn_edge_index(matrix: np.ndarray, k: int,
                   threshold: float = 0.0) -> torch.LongTensor:
    """
    k-NN edge_index da matrice di connettività [N, N].
    threshold: rimuovi archi dove matrix[i,j] < threshold (pruning).
    """
    N = matrix.shape[0]
    src, dst = [], []
    for i in range(N):
        row = matrix[i].copy()
        if threshold > 0.0:
            row[row < threshold] = 0.0
        top_k = np.argsort(row)[::-1][:k]
        for j in top_k:
            if row[j] > 0.0:
                src.extend([i, int(j)])
                dst.extend([int(j), i])
    if not src:
        # Fallback: grafo vuoto (tutti nodi isolati)
        return torch.zeros((2, 0), dtype=torch.long)
    # Deduplica
    edges = set(zip(src, dst))
    s, d = zip(*sorted(edges))
    return torch.tensor([list(s), list(d)], dtype=torch.long)


def plv_matrix(x_np: np.ndarray, sfreq: int = SFREQ,
               bands=None) -> np.ndarray:
    """PLV medio su bande theta+alpha [N, N]."""
    if bands is None:
        bands = PLV_BANDS
    nyq = sfreq / 2.0
    N = x_np.shape[0]
    plv_sum = np.zeros((N, N), dtype=np.float64)
    for flo, fhi in bands:
        b, a = butter(4, [flo / nyq, fhi / nyq], btype="band")
        x_f  = filtfilt(b, a, x_np, axis=1).astype(np.float32)
        phi  = np.angle(sp_hilbert(x_f, axis=1))
        exp_phi = np.exp(1j * phi)
        plv_sum += np.abs(exp_phi @ exp_phi.conj().T) / x_np.shape[1]
    plv = plv_sum / len(bands)
    np.fill_diagonal(plv, 0.0)
    return plv.astype(np.float32)


def wpli_matrix(x_np: np.ndarray, sfreq: int = SFREQ,
                bands=None) -> np.ndarray:
    """wPLI medio su bande theta+alpha [N, N]."""
    if bands is None:
        bands = PLV_BANDS
    nyq = sfreq / 2.0
    N = x_np.shape[0]
    wpli_sum = np.zeros((N, N), dtype=np.float64)
    for flo, fhi in bands:
        b, a = butter(4, [flo / nyq, fhi / nyq], btype="band")
        x_f  = filtfilt(b, a, x_np, axis=1).astype(np.float32)
        analytic = sp_hilbert(x_f, axis=1)
        for i in range(N):
            for jj in range(i + 1, N):
                cs = analytic[i] * np.conj(analytic[jj])
                im = np.imag(cs)
                denom = np.mean(np.abs(im))
                w = np.mean(im * np.sign(im)) / (denom + 1e-8)
                wpli_sum[i, jj] += abs(w)
                wpli_sum[jj, i] += abs(w)
    wpli = wpli_sum / len(bands)
    np.fill_diagonal(wpli, 0.0)
    return wpli.astype(np.float32)


# ════════════════════════════════════════════════════════════
# FUNZIONI COSTRUZIONE IPERGRAFO
# ════════════════════════════════════════════════════════════

def pcc_hyperedge_index(x_np: np.ndarray, k: int,
                        threshold: float = 0.0) -> torch.LongTensor:
    """
    Hyperedge_index per-trial da |PCC|.
    Per ogni nodo i: e_i = {i} ∪ {top-k più correlati}.
    threshold: salta l'arco se |PCC| < threshold (pruning).
    Ritorna: [2, N*(k+1)] LongTensor
    """
    pcc = pcc_matrix(x_np)
    N = pcc.shape[0]
    vertex_list, edge_list = [], []
    for i in range(N):
        row = pcc[i].copy()
        if threshold > 0.0:
            row[row < threshold] = 0.0
        top_k = np.argsort(row)[::-1][:k]
        members = [i] + [int(j) for j in top_k if row[j] > 0.0]
        for v in members:
            vertex_list.append(v)
            edge_list.append(i)
    return torch.tensor([vertex_list, edge_list], dtype=torch.long)


# ════════════════════════════════════════════════════════════
# CHANNEL PRUNING
# ════════════════════════════════════════════════════════════

def find_weak_channels(meta_df: pd.DataFrame, keep_idx: list,
                       n_sample: int = 500, seed: int = 42,
                       n_sigma: float = 2.0) -> list:
    """
    Identifica canali con connettività PCC media sotto mean - n_sigma*std.
    Restituisce lista di indici (dentro keep_idx) da considerare per rimozione.
    """
    rng = np.random.RandomState(seed)
    sample = meta_df.sample(min(n_sample, len(meta_df)), random_state=rng)
    paths_map = defaultdict(list)
    for _, row in sample.iterrows():
        paths_map[row["path_h5"]].append(int(row["epoch_idx"]))

    conn_sum = np.zeros(len(keep_idx), dtype=np.float64)
    count = 0
    for path, epoch_idxs in tqdm(paths_map.items(), desc="Channel pruning scan", leave=False):
        with h5py.File(path, "r") as f:
            for e_idx in epoch_idxs:
                x_np = f["data"][e_idx][keep_idx, :].astype(np.float32)
                pcc = pcc_matrix(x_np)
                conn_sum += pcc.mean(axis=1)
                count += 1

    mean_conn = conn_sum / count
    mu, sigma = mean_conn.mean(), mean_conn.std()
    threshold = mu - n_sigma * sigma
    weak = [i for i, c in enumerate(mean_conn) if c < threshold]
    print(f"Channel pruning: mu={mu:.3f} sigma={sigma:.3f} threshold={threshold:.3f}")
    print(f"  Weak channels: {weak} → nomi: {[keep_idx[i] for i in weak]}")
    return weak


# ════════════════════════════════════════════════════════════
# BUILD LOOP
# ════════════════════════════════════════════════════════════

def build_graph_dataset(meta_df, keep_idx, method, k,
                        threshold=0.0, save_path=None):
    """
    Costruisce lista di Data(x, edge_index, y, subj) per tutti i trial.
    x è grezzo (non normalizzato) — normalizzazione applicata in EEG_08.
    """
    print(f"\n── {method.upper()} k={k} threshold={threshold} ──")
    records = meta_df.to_dict("records")
    paths_map = defaultdict(list)
    for i, r in enumerate(records):
        paths_map[r["path_h5"]].append((i, int(r["epoch_idx"])))

    tmp = [None] * len(records)
    t0 = time.time()
    for path, items in tqdm(paths_map.items(), desc=f"Build {method}", leave=False):
        with h5py.File(path, "r") as f:
            for idx, e_idx in items:
                x_np = f["data"][e_idx][keep_idx, :].astype(np.float32)
                if method == "pcc":
                    mat = pcc_matrix(x_np)
                elif method == "plv":
                    mat = plv_matrix(x_np)
                elif method == "wpli":
                    mat = wpli_matrix(x_np)
                else:
                    raise ValueError(f"Metodo sconosciuto: {method}")
                ei = knn_edge_index(mat, k, threshold=threshold)
                r = records[idx]
                try:
                    subj = int(r["subject_id"])
                except (ValueError, KeyError):
                    continue
                tmp[idx] = Data(
                    x          = torch.tensor(x_np),
                    edge_index = ei,
                    y          = torch.tensor(int(r["label_idx"]), dtype=torch.long),
                    subj       = torch.tensor(subj, dtype=torch.long),
                )
    dataset = [d for d in tmp if d is not None]
    elapsed = int(time.time() - t0)
    print(f"  {len(dataset)} trial | tempo={elapsed}s")
    print(f"  Edges medi per trial: {np.mean([d.edge_index.shape[1] for d in dataset]):.0f}")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, save_path)
        size_mb = save_path.stat().st_size / 1e6
        print(f"  Salvato: {save_path} ({size_mb:.0f} MB)")
    return dataset


def build_hypergraph_dataset(meta_df, keep_idx, k,
                             threshold=0.0, save_path=None):
    """
    Costruisce lista di HypergraphData(x, hyperedge_index, y, subj, num_hyperedges).
    """
    try:
        from torch_geometric.data import Data as HypergraphData
    except ImportError:
        HypergraphData = Data

    print(f"\n── HGNN k={k} threshold={threshold} ──")
    N_HYPER = len(keep_idx)  # un iperspigolo per canale
    records = meta_df.to_dict("records")
    paths_map = defaultdict(list)
    for i, r in enumerate(records):
        paths_map[r["path_h5"]].append((i, int(r["epoch_idx"])))

    tmp = [None] * len(records)
    t0 = time.time()
    for path, items in tqdm(paths_map.items(), desc="Build HGNN", leave=False):
        with h5py.File(path, "r") as f:
            for idx, e_idx in items:
                x_np = f["data"][e_idx][keep_idx, :].astype(np.float32)
                he   = pcc_hyperedge_index(x_np, k, threshold=threshold)
                r = records[idx]
                try:
                    subj = int(r["subject_id"])
                except (ValueError, KeyError):
                    continue
                tmp[idx] = Data(
                    x               = torch.tensor(x_np),
                    hyperedge_index = he,
                    num_hyperedges  = torch.tensor(N_HYPER, dtype=torch.long),
                    y               = torch.tensor(int(r["label_idx"]), dtype=torch.long),
                    subj            = torch.tensor(subj, dtype=torch.long),
                )
    dataset = [d for d in tmp if d is not None]
    elapsed = int(time.time() - t0)
    print(f"  {len(dataset)} trial | tempo={elapsed}s")
    print(f"  Entries he medi per trial: {np.mean([d.hyperedge_index.shape[1] for d in dataset]):.0f}")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, save_path)
        size_mb = save_path.stat().st_size / 1e6
        print(f"  Salvato: {save_path} ({size_mb:.0f} MB)")
    return dataset


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pre-build grafi ed ipergrafi per-trial")
    parser.add_argument("--k",               type=int,   nargs="+", default=[6],
                        help="Valori di k per k-NN (default: 6)")
    parser.add_argument("--methods",         type=str,   nargs="+",
                        default=["pcc", "plv", "wpli"],
                        help="Metodi grafi semplici (default: pcc plv wpli)")
    parser.add_argument("--prune-threshold", type=float, default=0.0,
                        help="Rimuovi archi dove connettività < threshold (default: 0)")
    parser.add_argument("--prune-channels",  action="store_true",
                        help="Rimuovi canali con connettività media < mean-2sigma")
    parser.add_argument("--graph-only",      action="store_true",
                        help="Solo grafi semplici (no ipergrafi)")
    parser.add_argument("--hgnn-only",       action="store_true",
                        help="Solo ipergrafi (no grafi semplici)")
    parser.add_argument("--n-sample-prune",  type=int,   default=500,
                        help="Trial da campionare per channel pruning (default: 500)")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    keep_idx, keep_names = get_keep_channels(ELOC_PATH)
    print(f"Canali EEG: {len(keep_idx)} ({keep_names[:5]}...)")

    meta = pd.read_csv(META_CSV)
    meta = meta[~(
        (meta["path_h5"].str.contains("08_05.h5") & (meta["epoch_idx"] >= 110)) |
        (meta["path_h5"].str.contains("46_01.h5") & (meta["epoch_idx"] >= 110)) |
        (meta["path_h5"].str.contains("46_03.h5") & (meta["epoch_idx"] >= 110)) |
        (meta["path_h5"].str.contains("46_04.h5") & (meta["epoch_idx"] >= 34))
    )].copy()
    meta = meta[pd.to_numeric(meta["subject_id"], errors="coerce").notna()].copy()
    meta["subject_id"] = meta["subject_id"].astype(int).astype(str).str.zfill(2)
    print(f"Trial totali: {len(meta)}")

    # ── Channel pruning (opzionale) ───────────────────────
    if args.prune_channels:
        weak = find_weak_channels(meta, keep_idx, n_sample=args.n_sample_prune)
        if weak:
            print(f"⚠️  Canali deboli trovati: {weak}")
            resp = input("Rimuovere questi canali? [y/N] ").strip().lower()
            if resp == "y":
                keep_idx   = [keep_idx[i]   for i in range(len(keep_idx))   if i not in weak]
                keep_names = [keep_names[i] for i in range(len(keep_names)) if i not in weak]
                print(f"Canali dopo pruning: {len(keep_idx)}")

    threshold = args.prune_threshold
    suffix = f"_thr{threshold:.2f}".replace(".", "") if threshold > 0 else ""

    # ── Build grafi semplici ───────────────────────────────
    if not args.hgnn_only:
        for method in args.methods:
            for k in args.k:
                name = f"dataset_{method}_k{k}{suffix}.pt"
                save_path = GRAPHS_DIR / name
                if save_path.exists():
                    print(f"  Già esistente, skip: {name}")
                    continue
                build_graph_dataset(meta, keep_idx, method, k,
                                    threshold=threshold, save_path=save_path)

    # ── Build ipergrafi ────────────────────────────────────
    if not args.graph_only:
        for k in args.k:
            name = f"dataset_hgnn_k{k}{suffix}.pt"
            save_path = GRAPHS_DIR / name
            if save_path.exists():
                print(f"  Già esistente, skip: {name} (elimina per ricostruire)")
                continue
            build_hypergraph_dataset(meta, keep_idx, k,
                                     threshold=threshold, save_path=save_path)

    print("\n✅  Build completato.")
    print(f"   File in: {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
