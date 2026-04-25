"""
modify_eeg07e.py
Riscrive EEG_07e_build_graphs_tensors.ipynb con:
  - CPCCabs e CPCCim come metriche di connettività
  - Cache disco per le matrici di connettività (conn_cache/)
  - PLV/wPLI/CPCC vettorizzati (no for-loop O(N²))
  - Consensus filter anche nel pruning
"""

import json

NB_PATH = "/home/daniele_u/miralis-hypergraph-imagined-speech/notebooks/EEG_07e_build_graphs_tensors.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ──────────────────────────────────────────────────────────────
# NUOVE CELLE (lista di dict {cell_type, source, metadata, outputs})
# ──────────────────────────────────────────────────────────────

MD_INTRO = {
    "cell_type": "markdown",
    "metadata": {},
    "source": (
        "# EEG_07e — Pre-costruzione Tensori Grafo / Ipergrafo da CSV\n"
        "\n"
        "Legge i CSV di Paolo (`data/raw_csv/training_set/PXXX_SYYY/parola_img.csv`)\n"
        "e costruisce tensori PyG `Data(x, edge_index, y, subj, sess)` in `data/interim/graphs/`.\n"
        "\n"
        "**Struttura CSV**: 61 righe × 384 colonne (canali × campioni), nessun header.\n"
        "\n"
        "Metriche supportate: **PCC**, **PLV**, **wPLI**, **CPCCabs**, **CPCCim**\n"
        "\n"
        "| Metrica | Descrizione | Robusto VolumeConduction |\n"
        "|---------|-------------|---------------------------|\n"
        "| PCC | Pearson |r| | ✗ |\n"
        "| PLV | Phase Locking Value | ✗ |\n"
        "| wPLI | Weighted Phase Lag Index | ✓ |\n"
        "| CPCCabs | \\|mean(e^{iΔφ})\\| — ampiezza coupling | ✗ |\n"
        "| CPCCim | \\|mean(sin(Δφ))\\| — componente lag | ✓ |\n"
        "\n"
        "### Consensus filter — Iacomi et al. 2026\n"
        "\n"
        "Iacomi et al. (2026) analizzano la connettività EEG durante l'imagined speech su\n"
        "**questo stesso dataset** con PLV, wPLI, CPCCabs, CPCCim e trovano che mantenere\n"
        "solo gli archi presenti in **≥ 2/4 metriche** riduce i falsi positivi e produce\n"
        "grafi più stabili.\n"
        "\n"
        "### Cache matrici\n"
        "\n"
        "Le matrici di connettività (N×N float32) sono salvate in\n"
        "`data/interim/conn_cache/<CSV_stem>_<method>.npy` per non ricalcolarle ad ogni run.\n"
        "Elimina la cartella o imposta `USE_CONN_CACHE = False` per forzare il ricalcolo."
    )
}

CELL_CONFIG = """# ============================================================
# CONFIGURAZIONE
# ============================================================

# Metodi connettività — grafi e ipergrafi
# disponibili: "pcc" | "plv" | "wpli" | "cpcc_abs" | "cpcc_im"
METHODS       = ["pcc", "cpcc_abs", "cpcc_im"]
METHODS_HGNN  = ["pcc", "cpcc_abs", "cpcc_im"]

# K-vicini per k-NN graph / ipergrafo
K_VALUES = [6]
K_HYPER  = 6

# Soglia edge: rimuove archi con peso < threshold (0.0 = nessuna)
EDGE_THRESHOLD = 0.0

# Schema label da applicare
CLUSTER_SCHEME = "concr4"   # "concr4" | "ward4" | "sem5" | "pos4" | "raw110"

# Forza ricostruzione anche se file già esiste
FORCE_REBUILD = False

# Costruisci anche ipergrafo (per EEG_12)
BUILD_HGNN = True

# ============================================================
# CACHE MATRICI CONNETTIVITÀ
# ============================================================
# Le matrici (N×N) vengono salvate in conn_cache/<stem>_<method>.npy
# e ricaricate direttamente ai run successivi invece di ricalcolarle.
USE_CONN_CACHE = True    # False = disabilita cache (ricalcola sempre)

# ============================================================
# CONSENSUS FILTER — Iacomi et al. 2026 (stesso dataset)
# ============================================================
# Un arco (i,j) viene mantenuto solo se appare nel k-NN di
# ≥ CONSENSUS_MIN delle CONSENSUS_METHODS metriche.
#
# Raccomandazione paper: ≥ 2/4 metriche (PLV, wPLI, CPCCabs, CPCCim).
# Con tutte e 5 le metriche disponibili usa CONSENSUS_MIN=3.
# ⚠️  PLV/wPLI/CPCC sono più lenti di PCC — abilita cache per velocizzare.
CONSENSUS_FILTER   = False
CONSENSUS_METHODS  = ["pcc", "plv", "wpli", "cpcc_abs", "cpcc_im"]
CONSENSUS_MIN      = 3        # arco mantenuto se presente in ≥ N metriche

# ============================================================
# CONSENSUS FILTER NEL PRUNING
# ============================================================
# Dopo il pruning canali/archi, ricostruisce edge_index con consensus.
# Usa le matrici già in cache → molto più veloce del build iniziale.
CONSENSUS_PRUNING       = False
CONSENSUS_PRUNING_METHS = ["pcc", "cpcc_abs", "cpcc_im"]
CONSENSUS_PRUNING_MIN   = 2

print("Config OK")
if CONSENSUS_FILTER:
    print(f"  Build consensus: {CONSENSUS_METHODS}, min={CONSENSUS_MIN}/{len(CONSENSUS_METHODS)}")
if CONSENSUS_PRUNING:
    print(f"  Pruning consensus: {CONSENSUS_PRUNING_METHS}, min={CONSENSUS_PRUNING_MIN}")
"""

CELL_IMPORTS = """# ============================================================
# IMPORT E PATHS
# ============================================================

import os, sys, json, hashlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data

# Trova project root
project_root = next(
    (p for p in [Path().resolve()] + list(Path().resolve().parents)
     if (p / ".git").exists()),
    Path().resolve()
)
sys.path.insert(0, str(project_root / "scripts"))
from utils import load_label_scheme

# Paths
CSV_ROOT      = project_root / "data" / "raw_csv" / "training_set"
GRAPHS_DIR    = project_root / "data" / "interim" / "graphs"
CONN_CACHE_DIR = project_root / "data" / "interim" / "conn_cache"
CONFIGS       = project_root / "configs" / "label_schemes"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
CONN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_CHANS   = 61
N_SAMPLES = 384
SFREQ     = 256

# Mapping parola → label_id
with open(CONFIGS / "label2idx.json") as f:
    word2labelid = json.load(f)

# Schema cluster
labelid2cluster, N_CLASSES, cluster_names = load_label_scheme(
    CLUSTER_SCHEME, project_root / "data" / "interim"
)

session_dirs = sorted(CSV_ROOT.iterdir())
print(f"Project root   : {project_root}")
print(f"CSV root       : {CSV_ROOT}")
print(f"Cartelle sess  : {len(session_dirs)}")
print(f"N_CHANS        : {N_CHANS}")
print(f"Schema         : {CLUSTER_SCHEME} ({N_CLASSES} classi)")
print(f"Conn cache dir : {CONN_CACHE_DIR}")
"""


# Because Python source with triple-quotes inside a Python string is tricky,
# let's build the cell source as a list of lines for CELL_PARSING
CELL_PARSING_LINES = [
    "# ============================================================\n",
    "# PARSING NOMI CARTELLA  +  LOAD CSV\n",
    "# ============================================================\n",
    "\n",
    "def parse_folder(folder_name: str):\n",
    "    \"\"\"'P003_S002' -> (3, 2)\"\"\"\n",
    "    parts = folder_name.split(\"_\")\n",
    "    return int(parts[0][1:]), int(parts[1][1:])\n",
    "\n",
    "def load_csv_trial(csv_path) -> np.ndarray:\n",
    "    \"\"\"Legge CSV 61×384 senza header. Restituisce array float32 (61, 384).\"\"\"\n",
    "    return pd.read_csv(csv_path, header=None).values.astype(np.float32)\n",
    "\n",
    "# Test\n",
    "test_dir = session_dirs[0]\n",
    "test_csv = sorted(test_dir.iterdir())[0]\n",
    "x_test   = load_csv_trial(test_csv)\n",
    "subj_t, sess_t = parse_folder(test_dir.name)\n",
    "word_t = test_csv.stem.replace(\"_img\", \"\")\n",
    "\n",
    "print(f\"Cartella: {test_dir.name} → subj={subj_t}, sess={sess_t}\")\n",
    "print(f\"File    : {test_csv.name} → parola='{word_t}', label_id={word2labelid.get(word_t, '??')}\")\n",
    "print(f\"Shape   : {x_test.shape}  (atteso: (61, 384))\")\n",
    "assert x_test.shape == (N_CHANS, N_SAMPLES)\n",
    "print(\"✅ Parsing OK\")",
]

CELL_CONN_FN = """# ============================================================
# FUNZIONI DI CONNETTIVITÀ — vettorizzate + cache disco
# ============================================================
# Tutte le funzioni restituiscono matrici (N, N) float32 con diagonale=0.
# CPCC = Complex Phase Coupling Coefficient (Iacomi et al. 2026):
#   CPCCabs = |mean(exp(iΔφ))|      ← identico a PLV in valore assoluto
#   CPCCim  = |mean(sin(Δφ))|       ← componente lag, robusto a volume conduction
# ============================================================

from scipy.signal import hilbert as _hilbert

def pcc_matrix(x_np: np.ndarray) -> np.ndarray:
    \"\"\"Pearson |PCC| vettorizzato. O(N²) NumPy puro.\"\"\"
    pcc = np.abs(np.corrcoef(x_np)).astype(np.float32)
    np.fill_diagonal(pcc, 0.0)
    return pcc


def _phase_matrix(x_np: np.ndarray) -> np.ndarray:
    \"\"\"
    Fase istantanea via Hilbert. Restituisce matrice (N, T) float32.
    Usata internamente da PLV, wPLI, CPCCabs, CPCCim (calcolo unico).
    \"\"\"
    analytic = _hilbert(x_np.astype(np.float64), axis=1)
    return np.angle(analytic).astype(np.float32)


def plv_matrix(x_np: np.ndarray) -> np.ndarray:
    \"\"\"
    Phase Locking Value — completamente vettorizzato.
    PLV[i,j] = |mean(exp(i*(phi_i - phi_j)))|
    Equivalente a CPCCabs (stessa formula, diverso nome).
    \"\"\"
    phi = _phase_matrix(x_np)                # (N, T)
    # exp(i*phi) per ogni canale
    ep  = np.exp(1j * phi)                   # (N, T) complex128
    # PLV[i,j] = |mean_t(ep[i] * conj(ep[j]))|
    plv = np.abs(ep @ ep.conj().T / x_np.shape[1]).astype(np.float32)
    np.fill_diagonal(plv, 0.0)
    return plv


def wpli_matrix(x_np: np.ndarray) -> np.ndarray:
    \"\"\"
    Weighted Phase Lag Index — vettorizzato (no for-loop O(N²)).
    wPLI[i,j] = |mean(Im(Cs[i,j]))| / mean(|Im(Cs[i,j])|)
    dove Cs[i,j,t] = analytic[i,t] * conj(analytic[j,t])
    \"\"\"
    analytic = _hilbert(x_np.astype(np.float64), axis=1).astype(np.complex128)
    # cross-spectrum: shape (N, N, T) → troppo grande; procediamo riga per riga
    N, T  = x_np.shape
    wpli  = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        cs_row = analytic[i] * np.conj(analytic)  # (N, T)
        im_row = np.imag(cs_row)                   # (N, T)
        num    = np.abs(im_row.mean(axis=1))        # (N,)
        den    = np.abs(im_row).mean(axis=1) + 1e-9
        wpli[i] = (num / den).astype(np.float32)
    np.fill_diagonal(wpli, 0.0)
    return wpli


def cpcc_abs_matrix(x_np: np.ndarray) -> np.ndarray:
    \"\"\"
    CPCCabs = |mean_t(exp(i*Δφ))| — ampiezza del coupling di fase.
    Numericamente identico a PLV; nome separato per chiarezza paper.
    \"\"\"
    return plv_matrix(x_np)


def cpcc_im_matrix(x_np: np.ndarray) -> np.ndarray:
    \"\"\"
    CPCCim = |mean_t(sin(Δφ))| — componente immaginaria del CPCC.
    Robusta al volume conduction (analogo di iCoh), vettorizzata.
    CPCCim[i,j] = |mean(Im(exp(i*(phi_i - phi_j))))|
                = |mean(sin(phi_i - phi_j))|
    \"\"\"
    phi = _phase_matrix(x_np)               # (N, T) float32
    ep  = np.exp(1j * phi.astype(np.float64))
    cim = np.abs(np.imag(ep @ ep.conj().T) / x_np.shape[1]).astype(np.float32)
    np.fill_diagonal(cim, 0.0)
    return cim


# Registro globale
CONN_FN = {
    "pcc":      pcc_matrix,
    "plv":      plv_matrix,
    "wpli":     wpli_matrix,
    "cpcc_abs": cpcc_abs_matrix,
    "cpcc_im":  cpcc_im_matrix,
}


# ============================================================
# CACHE DISCO — salva/carica matrici (N×N) per csv_path + method
# ============================================================

def _cache_key(csv_path, method: str) -> Path:
    \"\"\"Restituisce il path del file .npy in CONN_CACHE_DIR.\"\"\"
    stem = Path(csv_path).stem
    return CONN_CACHE_DIR / f"{stem}_{method}.npy"


def get_conn_matrix(x_np: np.ndarray, method: str,
                    csv_path=None) -> np.ndarray:
    \"\"\"
    Restituisce la matrice di connettività (N, N) float32.
    Se USE_CONN_CACHE=True e il file .npy esiste, lo carica direttamente.
    Altrimenti calcola e salva.

    Args:
        x_np     : segnale (N_CHANS, N_SAMPLES)
        method   : chiave in CONN_FN
        csv_path : Path del CSV originale (usato come chiave cache)
    \"\"\"
    if USE_CONN_CACHE and csv_path is not None:
        cache_file = _cache_key(csv_path, method)
        if cache_file.exists():
            return np.load(cache_file)
        mat = CONN_FN[method](x_np)
        np.save(cache_file, mat)
        return mat
    return CONN_FN[method](x_np)


# ============================================================
# K-NN e CONSENSUS
# ============================================================

def knn_edge_index(matrix: np.ndarray, k: int,
                   threshold: float = 0.0) -> torch.LongTensor:
    \"\"\"k-NN graph da matrice connettività. Restituisce edge_index (2, E).\"\"\"
    N = matrix.shape[0]
    rows, cols = [], []
    for i in range(N):
        row = matrix[i].copy(); row[i] = -1.0
        if threshold > 0.0:
            row[row < threshold] = 0.0
        top_k = np.argsort(row)[-k:]
        for j in top_k:
            if row[j] > 0.0 or threshold == 0.0:
                rows += [i, j]; cols += [j, i]
    return torch.tensor([rows, cols], dtype=torch.long)


def consensus_edge_index(x_np: np.ndarray, k: int,
                          methods: list,
                          min_consensus: int,
                          threshold: float = 0.0,
                          csv_path=None) -> torch.LongTensor:
    \"\"\"
    Consensus filter — Iacomi et al. 2026.
    Un arco (i,j) viene mantenuto solo se appare nel k-NN di
    ≥ min_consensus delle metriche in `methods`.
    Usa la cache disco se USE_CONN_CACHE=True.
    \"\"\"
    N = x_np.shape[0]
    vote_matrix = np.zeros((N, N), dtype=np.int8)

    for method in methods:
        mat = get_conn_matrix(x_np, method, csv_path)
        for i in range(N):
            row = mat[i].copy(); row[i] = -1.0
            if threshold > 0.0:
                row[row < threshold] = 0.0
            top_k = np.argsort(row)[-k:]
            for j in top_k:
                if row[j] > 0.0 or threshold == 0.0:
                    vote_matrix[i, j] += 1
                    vote_matrix[j, i] += 1

    src, dst = np.where(vote_matrix >= min_consensus)
    mask = src != dst
    return torch.tensor([src[mask].tolist(), dst[mask].tolist()], dtype=torch.long)


def hyperedge_index_fn(x_np: np.ndarray, k: int,
                       method: str = "pcc",
                       threshold: float = 0.0,
                       csv_path=None) -> torch.LongTensor:
    \"\"\"
    Ipergrafo k-NN per-trial. Ogni nodo i è centro di un'iperedge.
    Usa cache disco se disponibile.
    \"\"\"
    mat = get_conn_matrix(x_np, method, csv_path)
    N   = mat.shape[0]
    vertex_list, edge_list = [], []
    for e_id in range(N):
        row = mat[e_id].copy()
        if threshold > 0.0:
            row[row < threshold] = 0.0
        top_k   = np.argsort(row)[-k:]
        members = [e_id] + [j for j in top_k
                            if (row[j] > 0.0 or threshold == 0.0)]
        for v in members:
            vertex_list.append(v); edge_list.append(e_id)
    return torch.tensor([vertex_list, edge_list], dtype=torch.long)


print("Funzioni connettività OK")
print(f"  Disponibili: {list(CONN_FN.keys())}")
print(f"  Cache: {'ON → ' + str(CONN_CACHE_DIR) if USE_CONN_CACHE else 'OFF'}")
if CONSENSUS_FILTER:
    print(f"  Consensus build: {CONSENSUS_METHODS}, min={CONSENSUS_MIN}")
"""

CELL_ITER_TRIALS = """# ============================================================
# HELPER: iteratore tutti i trial da CSV
# ============================================================

def iter_all_trials():
    \"\"\"
    Generator: yield (x_np, label_id, cluster_id, subj_id, sess_id, csv_path).
    Salta file con parola non in label2idx o label_id non in labelid2cluster.
    \"\"\"
    for sess_dir in sorted(CSV_ROOT.iterdir()):
        if not sess_dir.is_dir():
            continue
        subj_id, sess_id = parse_folder(sess_dir.name)
        for csv_path in sorted(sess_dir.iterdir()):
            if csv_path.suffix != ".csv":
                continue
            word = csv_path.stem.replace("_img", "")
            if word not in word2labelid:
                continue
            label_id = word2labelid[word]
            if label_id not in labelid2cluster:
                continue
            cluster_id = labelid2cluster[label_id]
            x_np = load_csv_trial(csv_path)
            yield x_np, label_id, cluster_id, subj_id, sess_id, csv_path

n_total = sum(1 for _ in iter_all_trials())
print(f"Trial totali validi: {n_total}")
print(f"(atteso ~{len(session_dirs) * 110} = {len(session_dirs)} sess × 110 parole)")
"""

CELL_BUILD_GRAPHS = """# ============================================================
# BUILD GRAPH TENSORS (PCC / PLV / wPLI / CPCCabs / CPCCim / Consensus)
# Output: data/interim/graphs/graph_{method}_k{k}.pt
#         oppure: data/interim/graphs/graph_consensus{min}of{n}_k{k}.pt
# Le matrici di connettività vengono salvate in conn_cache/ (riuso rapido).
# ============================================================

if CONSENSUS_FILTER:
    _n_methods = len(CONSENSUS_METHODS)
    _cons_tag  = f"consensus{CONSENSUS_MIN}of{_n_methods}"
    print(f"Modalità CONSENSUS FILTER: {CONSENSUS_METHODS}, min={CONSENSUS_MIN}/{_n_methods}")

    for k in K_VALUES:
        out_path = GRAPHS_DIR / f"graph_{_cons_tag}_k{k}.pt"
        if out_path.exists() and not FORCE_REBUILD:
            print(f"Skip: {out_path.name}")
            continue

        print(f"\\nCostruendo {out_path.name}...")
        data_list = []
        n_edges_log = []

        for x_np, label_id, cluster_id, subj_id, sess_id, csv_path in tqdm(
                iter_all_trials(), total=n_total, desc=f"consensus k={k}"):

            edge_index = consensus_edge_index(
                x_np, k=k,
                methods=CONSENSUS_METHODS,
                min_consensus=CONSENSUS_MIN,
                threshold=EDGE_THRESHOLD,
                csv_path=csv_path,
            )
            n_edges_log.append(edge_index.shape[1] // 2)

            data_list.append(Data(
                x          = torch.tensor(x_np, dtype=torch.float32),
                edge_index = edge_index,
                y          = torch.tensor(cluster_id, dtype=torch.long),
                label_id   = torch.tensor(label_id,   dtype=torch.long),
                subj       = torch.tensor(subj_id,    dtype=torch.long),
                sess       = torch.tensor(sess_id,    dtype=torch.long),
            ))

        torch.save(data_list, out_path)
        print(f"  ✅ Salvato: {out_path.name} — {len(data_list)} grafi")
        print(f"     Archi medi: {np.mean(n_edges_log):.1f} "
              f"(min={min(n_edges_log)}, max={max(n_edges_log)})")

else:
    for method in METHODS:
        for k in K_VALUES:
            out_path = GRAPHS_DIR / f"graph_{method}_k{k}.pt"
            if out_path.exists() and not FORCE_REBUILD:
                print(f"Skip: {out_path.name}")
                continue

            print(f"\\nCostruendo {out_path.name}...")
            data_list = []

            for x_np, label_id, cluster_id, subj_id, sess_id, csv_path in tqdm(
                    iter_all_trials(), total=n_total, desc=f"{method} k={k}"):

                matrix     = get_conn_matrix(x_np, method, csv_path)
                edge_index = knn_edge_index(matrix, k=k, threshold=EDGE_THRESHOLD)

                data_list.append(Data(
                    x          = torch.tensor(x_np, dtype=torch.float32),
                    edge_index = edge_index,
                    y          = torch.tensor(cluster_id, dtype=torch.long),
                    label_id   = torch.tensor(label_id,   dtype=torch.long),
                    subj       = torch.tensor(subj_id,    dtype=torch.long),
                    sess       = torch.tensor(sess_id,    dtype=torch.long),
                ))

            torch.save(data_list, out_path)
            print(f"  ✅ Salvato: {out_path.name} — {len(data_list)} grafi  "
                  f"x.shape={data_list[0].x.shape}")
"""

CELL_BUILD_HGRAPH = """# ============================================================
# BUILD HYPERGRAPH TENSORS (tutte le metriche in METHODS_HGNN)
# Output: data/interim/graphs/hgraph_{method}_k{k}.pt
# ============================================================

if BUILD_HGNN:
    for method in METHODS_HGNN:
        for k in [K_HYPER]:
            out_path = GRAPHS_DIR / f"hgraph_{method}_k{k}.pt"
            if out_path.exists() and not FORCE_REBUILD:
                print(f"Skip: {out_path.name}")
                continue

            print(f"\\nCostruendo {out_path.name}...")
            data_list = []

            for x_np, label_id, cluster_id, subj_id, sess_id, csv_path in tqdm(
                    iter_all_trials(), total=n_total,
                    desc=f"hgraph {method} k={k}"):

                he_index     = hyperedge_index_fn(x_np, k=k,
                                                  method=method,
                                                  threshold=EDGE_THRESHOLD,
                                                  csv_path=csv_path)
                n_hyperedges = he_index[1].max().item() + 1

                data_list.append(Data(
                    x               = torch.tensor(x_np, dtype=torch.float32),
                    hyperedge_index = he_index,
                    num_hyperedges  = n_hyperedges,
                    y               = torch.tensor(cluster_id, dtype=torch.long),
                    label_id        = torch.tensor(label_id,   dtype=torch.long),
                    subj            = torch.tensor(subj_id,    dtype=torch.long),
                    sess            = torch.tensor(sess_id,    dtype=torch.long),
                ))

            torch.save(data_list, out_path)
            print(f"  ✅ Salvato: {out_path.name} — {len(data_list)} ipergrafi")
"""

CELL_SANITY = r"""# ============================================================
# SANITY CHECK + COMPARISON PLOT GRAFO vs IPERGRAFO
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

print(f"=== File in {GRAPHS_DIR} ===")
graph_data  = {}
hgraph_data = {}

for pt_file in sorted(GRAPHS_DIR.glob("*.pt")):
    dl = torch.load(pt_file, weights_only=False)
    d0 = dl[0]
    subj_ids = sorted(set(d.subj.item() for d in dl))
    y_vals   = [d.y.item() for d in dl]
    is_hgraph = pt_file.name.startswith("hgraph")

    if not is_hgraph:
        ei = d0.edge_index
        print(f"  {pt_file.name}: {len(dl)} grafi | "
              f"x={d0.x.shape} | edge_index={ei.shape if ei is not None else 'None'} | "
              f"n_subj={len(subj_ids)} | y_range=[{min(y_vals)},{max(y_vals)}]")
        graph_data[pt_file.stem] = dl
    else:
        he = d0.hyperedge_index
        print(f"  {pt_file.name}: {len(dl)} ipergrafi | "
              f"x={d0.x.shape} | he={he.shape if he is not None else 'None'} | "
              f"n_he={d0.num_hyperedges} | y_range=[{min(y_vals)},{max(y_vals)}]")
        hgraph_data[pt_file.stem] = dl

if not graph_data and not hgraph_data:
    print("⚠️  Nessun file .pt trovato — esegui prima le celle build.")
else:
    print(f"\n✅ Sanity check OK — {N_CHANS} canali, schema={CLUSTER_SCHEME}")

# ---- COMPARISON PLOT ----
if graph_data and hgraph_data:
    import networkx as nx
    g_key  = list(graph_data.keys())[0]
    hg_key = list(hgraph_data.keys())[0]
    d_g    = graph_data[g_key][0]
    d_hg   = hgraph_data[hg_key][0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. PCC heatmap
    x_np = d_g.x.numpy()
    pcc  = np.abs(np.corrcoef(x_np)); np.fill_diagonal(pcc, 0)
    im = axes[0].imshow(pcc, cmap="RdYlBu_r", vmin=0, vmax=0.5)
    axes[0].set_title("PCC matrice\n(trial campione)")
    axes[0].set_xlabel("Canale"); axes[0].set_ylabel("Canale")
    plt.colorbar(im, ax=axes[0])

    # 2. Grafo k-NN
    G = nx.Graph()
    G.add_nodes_from(range(d_g.x.shape[0]))
    ei = d_g.edge_index.numpy()
    for s, t in zip(ei[0], ei[1]):
        if s < t: G.add_edge(int(s), int(t))
    deg_g = [G.degree(n) for n in G.nodes()]
    pos   = nx.circular_layout(G)
    nx.draw_networkx(G, pos=pos, ax=axes[1],
                     node_size=80, node_color=deg_g, cmap="viridis",
                     with_labels=False, width=0.4, edge_color="gray")
    axes[1].set_title(f"Grafo k-NN ({g_key})\n"
                      f"{G.number_of_nodes()} nodi, {G.number_of_edges()} archi\n"
                      f"grado medio={np.mean(deg_g):.1f}")

    # 3. Ipergrafo
    he = d_hg.hyperedge_index.numpy()
    n_nodes  = d_hg.x.shape[0]
    n_he_val = d_hg.num_hyperedges
    node_deg = np.bincount(he[0], minlength=n_nodes)
    he_size  = np.bincount(he[1], minlength=n_he_val)
    axes[2].bar(range(n_nodes), node_deg, color="steelblue", alpha=0.7, label="Grado nodo")
    ax2t = axes[2].twinx()
    ax2t.hist(he_size, bins=20, color="orange", alpha=0.5, label="Dim. iperedge")
    axes[2].set_xlabel("Nodo (canale)")
    axes[2].set_ylabel("Grado nodo", color="steelblue")
    ax2t.set_ylabel("Freq. dim. iperedge", color="orange")
    axes[2].set_title(f"Ipergrafo ({hg_key})\n{n_nodes} nodi, {n_he_val} iperedge")
    axes[2].legend(loc="upper left"); ax2t.legend(loc="upper right")

    plt.tight_layout()
    fig_path = project_root / "figures" / "eeg07e_graph_vs_hypergraph.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Plot salvato: {fig_path}")
"""

CELL_CONN_ANALYSIS = r"""# ============================================================
# ANALISI CONNETTIVITÀ — media su tutti i trial
# Guida al pruning: distribuzioni archi, canali deboli.
# Usa la cache disco → molto più veloce al secondo run.
# ============================================================

import seaborn as sns

# Metodi da includere nell'analisi (anche cpcc_abs e cpcc_im)
METHODS_ANALYSIS = METHODS if METHODS else ["pcc", "cpcc_abs", "cpcc_im"]

VAL_SUBSAMPLE = 10   # campiona vals 1 trial su VAL_SUBSAMPLE

# Costruisce lista path trial
_subjects_all = sorted(set(d.name.split("_")[0] for d in CSV_ROOT.iterdir() if d.is_dir()))
_sessions_all = sorted(set(d.name.split("_")[1] for d in CSV_ROOT.iterdir() if d.is_dir()))
_words_all    = sorted(word2labelid.keys())

_trial_paths = [
    CSV_ROOT / f"{s}_{ss}" / f"{w}_img.csv"
    for s in _subjects_all
    for ss in _sessions_all
    for w in _words_all
    if (CSV_ROOT / f"{s}_{ss}" / f"{w}_img.csv").exists()
]
n_trials_analysis = len(_trial_paths)
print(f"Trial trovati  : {n_trials_analysis}")
print(f"Metodi analisi : {METHODS_ANALYSIS}")
if USE_CONN_CACHE:
    n_cached = sum(1 for p in _trial_paths
                   for m in METHODS_ANALYSIS
                   if _cache_key(p, m).exists())
    print(f"Matrici in cache: {n_cached}/{n_trials_analysis * len(METHODS_ANALYSIS)}")

# Accumulatori
results = {m: {"sum": np.zeros((N_CHANS, N_CHANS)), "vals": [], "n": 0}
           for m in METHODS_ANALYSIS}

for i, csv_path in enumerate(tqdm(_trial_paths, desc="Analisi connettività")):
    if not csv_path.exists():
        continue
    x_np = load_csv_trial(csv_path)
    for m in METHODS_ANALYSIS:
        mat = get_conn_matrix(x_np, m, csv_path)
        results[m]["sum"] += mat
        results[m]["n"]   += 1
        if i % VAL_SUBSAMPLE == 0:
            results[m]["vals"].extend(
                mat[np.triu_indices(N_CHANS, k=1)].tolist()
            )

for m in METHODS_ANALYSIS:
    results[m]["avg"]  = results[m]["sum"] / max(results[m]["n"], 1)
    results[m]["vals"] = np.array(results[m]["vals"])
    results[m]["conn"] = results[m]["avg"].sum(axis=1)

print(f"\n✅ Analisi completata — {results[METHODS_ANALYSIS[0]]['n']} trial")

# Nomi canali
_eloc_candidates = [
    project_root / "data" / "interim" / "ebneuro.csv",
    project_root / "data" / "interim" / "channel_mapping.csv",
]
_ch_names = [str(i) for i in range(N_CHANS)]
for _p in _eloc_candidates:
    if _p.exists():
        try:
            _ch_names = pd.read_csv(_p, sep=";", decimal=",")["labels"].tolist()[:N_CHANS]
            print(f"Nomi canali: {_p.name}")
        except Exception:
            pass
        break

# PLOT
n_methods = len(METHODS_ANALYSIS)
fig, axes = plt.subplots(3, n_methods, figsize=(6 * n_methods, 14))
if n_methods == 1:
    axes = axes[:, np.newaxis]
fig.suptitle(
    f"Analisi Connettività — {n_trials_analysis} trial "
    f"({len(_subjects_all)} sogg × {len(_sessions_all)} sess × {len(_words_all)} parole)",
    fontsize=13
)

for col, m in enumerate(METHODS_ANALYSIS):
    avg   = results[m]["avg"]
    conn  = results[m]["conn"]
    vals  = results[m]["vals"]
    mean_c, std_c = conn.mean(), conn.std()

    sns.heatmap(avg, ax=axes[0, col], cmap="RdYlBu_r",
                vmin=0, vmax=float(np.percentile(vals, 95)) if len(vals) else 1.0,
                xticklabels=False, yticklabels=False)
    axes[0, col].set_title(f"{m.upper()} — matrice media\n({results[m]['n']} trial)")

    bar_colors = [
        "red"      if c < mean_c - 2 * std_c else
        "orange"   if c < mean_c -     std_c else
        "steelblue"
        for c in conn
    ]
    axes[1, col].bar(range(N_CHANS), conn, color=bar_colors)
    axes[1, col].axhline(mean_c,             color="black",  ls="--", lw=1.2, label=f"μ={mean_c:.3f}")
    axes[1, col].axhline(mean_c -     std_c, color="orange", ls="--", lw=1.2, label=f"μ-1σ")
    axes[1, col].axhline(mean_c - 2 * std_c, color="red",    ls="--", lw=1.2, label=f"μ-2σ")
    axes[1, col].set_xlabel("Canale (idx)")
    axes[1, col].set_ylabel("Connettività totale")
    axes[1, col].set_title(f"{m.upper()} — per canale\n🔴<μ-2σ  🟠<μ-1σ")
    axes[1, col].legend(fontsize=7)

    if len(vals):
        axes[2, col].hist(vals, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        for pct in [25, 50, 75]:
            thr = float(np.percentile(vals, pct))
            axes[2, col].axvline(thr, color="red", ls="--", lw=1.2, label=f"p{pct}={thr:.3f}")
        axes[2, col].set_xlabel(f"|{m.upper()}|")
        axes[2, col].set_ylabel("Frequenza")
        axes[2, col].set_title(f"{m.upper()} — distribuzione archi\n(campionato 1/{VAL_SUBSAMPLE} trial)")
        axes[2, col].legend(fontsize=7)

plt.tight_layout()
fig_path = project_root / "figures" / "eeg07e_connectivity_analysis_full.png"
fig.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Plot salvato: {fig_path}")

# Riepilogo testuale
for m in METHODS_ANALYSIS:
    conn   = results[m]["conn"]
    vals   = results[m]["vals"]
    mean_c = conn.mean(); std_c = conn.std()
    print(f"\n{'='*50}  {m.upper()}")
    print(f"  N trial analizzati : {results[m]['n']}")
    print(f"  Connettività media : μ={mean_c:.4f}  σ={std_c:.4f}")
    weak = np.where(conn < mean_c - 2 * std_c)[0]
    if len(weak):
        names_str = ", ".join(f"idx={i} ({_ch_names[i]})" for i in weak)
        print(f"  ⚠️  Canali deboli (<μ-2σ={mean_c-2*std_c:.4f}): {names_str}")
    else:
        print(f"  ✅ Nessun canale debole (<μ-2σ)")
    if len(vals):
        print(f"  PRUNE_EDGE_THRESHOLD consigliato:")
        for pct in [25, 50, 75]:
            thr  = float(np.percentile(vals, pct))
            kept = float((vals >= thr).mean() * 100)
            print(f"    p{pct:2d} = {thr:.4f}  →  mantiene ~{kept:.0f}% archi")
"""

CELL_PRUNING_PARAMS = """# ============================================================
# PARAMETRI PRUNING — auto-estratti dall'analisi
# ============================================================

if "results" not in globals() or "conn" not in results.get(METHODS_ANALYSIS[0], {}):
    raise RuntimeError("❌ 'results' non definito — esegui prima cell Analisi Connettività.")

REF_METHOD = METHODS_ANALYSIS[0]
conn_ref   = results[REF_METHOD]["conn"]
vals_ref   = results[REF_METHOD]["vals"]
mean_c     = conn_ref.mean()
std_c      = conn_ref.std()

# 1. Canali deboli: sotto μ - 2σ
weak_idx            = np.where(conn_ref < mean_c - 2 * std_c)[0]
PRUNE_CHANNEL_NAMES = [_ch_names[i] for i in weak_idx]

# 2. Soglia archi: percentile scelto
USE_PERCENTILE = 50   # 0 | 25 | 50 | 75

if USE_PERCENTILE > 0 and len(vals_ref) > 0:
    PRUNE_EDGE_THRESHOLD = float(np.percentile(vals_ref, USE_PERCENTILE))
else:
    PRUNE_EDGE_THRESHOLD = 0.0

print("=" * 55)
print(f"PARAMETRI AUTO-ESTRATTI (Riferimento: {REF_METHOD.upper()})")
print("=" * 55)
if len(PRUNE_CHANNEL_NAMES) > 0:
    print(f"🔴 Canali deboli (<μ-2σ): {len(PRUNE_CHANNEL_NAMES)}")
    print(f"   PRUNE_CHANNEL_NAMES = {PRUNE_CHANNEL_NAMES}")
else:
    print("✅ Nessun canale debole (<μ-2σ).")

if USE_PERCENTILE > 0:
    print(f"✂️  Soglia archi (p{USE_PERCENTILE}): {PRUNE_EDGE_THRESHOLD:.4f}")
else:
    print("✅ Nessun pruning archi (USE_PERCENTILE=0).")
print("=" * 55)
"""

CELL_PRUNING_BUILD = """# ============================================================
# BUILD TENSORI PRUNATI
# 1. Carica .pt esistenti
# 2. Rimuove PRUNE_CHANNEL_NAMES + applica PRUNE_EDGE_THRESHOLD
# 3. (opz.) Applica consensus filter con CONSENSUS_PRUNING_METHS
#    sfruttando la cache disco già calcolata → molto veloce
#
# Output: data/interim/graphs/{name}_drop*_thr*[_cons].pt
# ============================================================

_prune_drop_idx = set()
for _name in PRUNE_CHANNEL_NAMES:
    if _name in _ch_names:
        _prune_drop_idx.add(_ch_names.index(_name))
    else:
        print(f"⚠️  Canale '{_name}' non trovato — saltato")

_pruned_keep = [i for i in range(N_CHANS) if i not in _prune_drop_idx]
N_PRUNED     = len(_pruned_keep)

print(f"Canali rimossi : {[_ch_names[i] for i in _prune_drop_idx] or 'nessuno'}")
print(f"Canali rimasti : {N_PRUNED}/{N_CHANS}")
print(f"Edge threshold : {PRUNE_EDGE_THRESHOLD}")
print(f"Consensus prune: {'ON, metriche=' + str(CONSENSUS_PRUNING_METHS) + ' min=' + str(CONSENSUS_PRUNING_MIN) if CONSENSUS_PRUNING else 'OFF'}")

if N_PRUNED == N_CHANS and PRUNE_EDGE_THRESHOLD == 0.0 and not CONSENSUS_PRUNING:
    print("\\n⚠️  Nessun pruning attivo — imposta parametri e riesegui.")
else:
    _suf_ch  = ("_drop" + "_".join(PRUNE_CHANNEL_NAMES)) if PRUNE_CHANNEL_NAMES else ""
    _suf_thr = f"_thr{PRUNE_EDGE_THRESHOLD:.3f}" if PRUNE_EDGE_THRESHOLD > 0 else ""
    _suf_con = f"_cons{CONSENSUS_PRUNING_MIN}of{len(CONSENSUS_PRUNING_METHS)}" if CONSENSUS_PRUNING else ""
    _suffix  = _suf_ch + _suf_thr + _suf_con

    # Mappa indice vecchio → nuovo nei canali prunati
    _idx_map = {old: new for new, old in enumerate(_pruned_keep)}

    for pt_file in sorted(GRAPHS_DIR.glob("*.pt")):
        if "_drop" in pt_file.name or "_thr" in pt_file.name or "_cons" in pt_file.name:
            continue

        is_hgraph = pt_file.name.startswith("hgraph")
        _k        = int(pt_file.stem.split("_k")[-1]) if "_k" in pt_file.stem else K_HYPER
        out_path  = GRAPHS_DIR / (pt_file.stem + _suffix + ".pt")

        if out_path.exists() and not FORCE_REBUILD:
            print(f"Skip (esiste): {out_path.name}")
            continue

        dl = torch.load(pt_file, weights_only=False)
        print(f"\\nPruning {pt_file.name} → {out_path.name} ...")
        pruned_list = []

        # Ricostruisci la lista trial path per la cache (necessaria per consensus)
        # usando un generatore leggero (solo path, senza caricare x_np)
        _trial_map = {}  # subj_sess_word → csv_path
        for sess_dir in sorted(CSV_ROOT.iterdir()):
            if not sess_dir.is_dir(): continue
            s_id, ss_id = parse_folder(sess_dir.name)
            for cp in sorted(sess_dir.iterdir()):
                if cp.suffix != ".csv": continue
                word = cp.stem.replace("_img", "")
                if word not in word2labelid: continue
                lid = word2labelid[word]
                if lid not in labelid2cluster: continue
                _trial_map[(s_id, ss_id, word)] = cp

        for idx_d, d in enumerate(tqdm(dl, desc=out_path.name, leave=False)):
            x_np     = d.x.numpy()                 # (N_CHANS, 384)
            x_pruned = x_np[_pruned_keep, :]       # (N_PRUNED, 384)

            # Recupera csv_path dal trial (per cache)
            # L'indice idx_d corrisponde all'ordine in iter_all_trials()
            # → usa _trial_map se disponibile, altrimenti None
            _csv_path_d = list(_trial_map.values())[idx_d] if idx_d < len(_trial_map) else None

            if not is_hgraph:
                if CONSENSUS_PRUNING:
                    # Consensus filter sulle metriche indicate, sfrutta cache disco
                    _ei = consensus_edge_index(
                        x_pruned, k=_k,
                        methods=CONSENSUS_PRUNING_METHS,
                        min_consensus=CONSENSUS_PRUNING_MIN,
                        threshold=PRUNE_EDGE_THRESHOLD,
                        csv_path=_csv_path_d,   # cache già presente per canali originali
                    )
                elif PRUNE_EDGE_THRESHOLD > 0.0:
                    _mat = pcc_matrix(x_pruned)
                    _ei  = knn_edge_index(_mat, k=_k, threshold=PRUNE_EDGE_THRESHOLD)
                else:
                    # Riproietta edge_index esistente
                    _old_ei = d.edge_index.numpy()
                    _mask   = [
                        (int(s) in _idx_map and int(t) in _idx_map)
                        for s, t in zip(_old_ei[0], _old_ei[1])
                    ]
                    _new_src = [_idx_map[int(s)] for s, ok in zip(_old_ei[0], _mask) if ok]
                    _new_dst = [_idx_map[int(t)] for t, ok in zip(_old_ei[1], _mask) if ok]
                    _ei      = torch.tensor([_new_src, _new_dst], dtype=torch.long)

                pruned_list.append(Data(
                    x          = torch.tensor(x_pruned, dtype=torch.float32),
                    edge_index = _ei,
                    y          = d.y,
                    label_id   = d.label_id,
                    subj       = d.subj,
                    sess       = d.sess,
                ))
            else:
                _he = hyperedge_index_fn(x_pruned, k=_k,
                                         threshold=PRUNE_EDGE_THRESHOLD,
                                         csv_path=_csv_path_d)
                pruned_list.append(Data(
                    x               = torch.tensor(x_pruned, dtype=torch.float32),
                    hyperedge_index = _he,
                    num_hyperedges  = int(_he[1].max()) + 1,
                    y               = d.y,
                    label_id        = d.label_id,
                    subj            = d.subj,
                    sess            = d.sess,
                ))

        torch.save(pruned_list, out_path)
        print(f"  ✅ {out_path.name}: {len(pruned_list)} oggetti, "
              f"x.shape={pruned_list[0].x.shape}")
"""

CELL_VIZ_INTERACTIVE = r"""# ============================================================
# VISUALIZZAZIONE INTERATTIVA: Grafo vs Ipergrafo
# ============================================================

import ipywidgets as widgets
from ipywidgets import interact, interactive_output, VBox, HBox
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial import ConvexHull

_eloc_candidates = [
    project_root / "data" / "interim" / "ebneuro.csv",
    project_root / "data" / "interim" / "channel_mapping.csv",
]
_eloc_df = None
for _p in _eloc_candidates:
    if _p.exists():
        try:
            _eloc_df = pd.read_csv(_p, sep=";", decimal=",")
        except Exception:
            pass
        break

def get_positions(n_nodes):
    if _eloc_df is not None and len(_eloc_df) >= n_nodes:
        df = _eloc_df.iloc[:n_nodes]
        theta_deg = df["theta"].values.astype(float)
        radius    = df["radius"].values.astype(float)
        theta_rad = np.deg2rad(theta_deg)
        x = radius * np.sin(theta_rad)
        y = radius * np.cos(theta_rad)
        return np.stack([x, y], axis=1)
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

_subjects  = sorted(set(d.name.split("_")[0] for d in CSV_ROOT.iterdir() if d.is_dir()))
_words     = sorted(word2labelid.keys())
_sessions  = sorted(set(d.name.split("_")[1] for d in CSV_ROOT.iterdir() if d.is_dir()))

w_subj     = widgets.Dropdown(options=_subjects, value=_subjects[0], description="Soggetto:")
w_word     = widgets.Dropdown(options=_words,    value=_words[0],    description="Parola:")
w_sess     = widgets.Dropdown(options=_sessions, value=_sessions[0], description="Sessione:")
w_k        = widgets.IntSlider(min=2, max=15, value=6, step=1, description="k vicini:")
w_method   = widgets.Dropdown(options=list(CONN_FN.keys()), value="pcc", description="Metrica:")
w_he_max   = widgets.IntSlider(min=5, max=61, value=20, step=1, description="Max iperedge:")

def draw_convex_blob(ax, pos, members, color, alpha=0.25, pad=0.04):
    pts = pos[members]
    if len(pts) < 3:
        cx, cy = pts.mean(axis=0)
        circle = plt.Circle((cx, cy), pad*3, color=color, alpha=alpha, zorder=1)
        ax.add_patch(circle); return
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        centroid = hull_pts.mean(axis=0)
        expanded = centroid + (hull_pts - centroid) * (
            1 + pad / (np.linalg.norm(hull_pts - centroid, axis=1).mean() + 1e-9))
        poly = plt.Polygon(np.vstack([expanded, expanded[0]]),
                           closed=True, color=color, alpha=alpha, zorder=1,
                           linewidth=1.2, edgecolor=color)
        ax.add_patch(poly)
    except Exception:
        pass

def plot_comparison(subj, word, session, k, method, max_he):
    csv_path = CSV_ROOT / f"{subj}_{session}" / f"{word}_img.csv"
    if not csv_path.exists():
        print(f"File non trovato: {csv_path}")
        return

    x_np = load_csv_trial(csv_path)
    N    = x_np.shape[0]
    pos  = get_positions(N)

    mat = get_conn_matrix(x_np, method, csv_path)
    np.fill_diagonal(mat, 0.0)
    ei  = knn_edge_index(mat, k=k).numpy()

    he  = hyperedge_index_fn(x_np, k=k, method=method, csv_path=csv_path).numpy()
    n_he = int(he[1].max()) + 1
    he_members = {}
    for v, e in zip(he[0], he[1]):
        he_members.setdefault(int(e), []).append(int(v))

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for s, t in zip(ei[0], ei[1]):
        if s < t: G.add_edge(int(s), int(t), weight=float(mat[s, t]))
    node_deg = np.array([G.degree(i) for i in range(N)], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Soggetto {subj}  |  Parola '{word}'  |  Sessione {session}  |  k={k}  |  {method.upper()}",
                 fontsize=13, fontweight="bold")

    ax1.set_title(f"Grafo k-NN — {method.upper()}  ({G.number_of_edges()} archi)", fontsize=11)
    edge_weights = [mat[u, v] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos={i: pos[i] for i in range(N)},
                           ax=ax1, edge_color=edge_weights,
                           edge_cmap=cm.Blues, width=1.5, alpha=0.7)
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1],
                      s=60 + node_deg * 20, c=node_deg, cmap="plasma",
                      zorder=5, edgecolors="black", linewidths=0.5)
    plt.colorbar(sc1, ax=ax1, label="Grado nodo")
    ax1.add_patch(plt.Circle((0, 0), 0.6, fill=False, color="gray",
                              linewidth=1.5, linestyle="--"))
    ax1.set_aspect("equal"); ax1.axis("off")

    ax2.set_title(f"Ipergrafo — {method.upper()}  ({n_he} iperedge, mostra prime {min(max_he, n_he)})", fontsize=11)
    colors_he = cm.tab20(np.linspace(0, 1, min(max_he, n_he)))
    for (e_id, members), color in zip(list(he_members.items())[:max_he], colors_he):
        draw_convex_blob(ax2, pos, members, color=color[:3], alpha=0.3)
    n_he_per_node = np.bincount(he[0], minlength=N)
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1],
                      s=60 + n_he_per_node * 5, c=n_he_per_node, cmap="YlOrRd",
                      zorder=5, edgecolors="black", linewidths=0.5)
    plt.colorbar(sc2, ax=ax2, label="N° iperedge per nodo")
    ax2.add_patch(plt.Circle((0, 0), 0.6, fill=False, color="gray",
                              linewidth=1.5, linestyle="--"))
    ax2.set_aspect("equal"); ax2.axis("off")

    plt.tight_layout(); plt.show()

out = interactive_output(
    plot_comparison,
    {"subj": w_subj, "word": w_word, "session": w_sess,
     "k": w_k, "method": w_method, "max_he": w_he_max}
)
display(VBox([
    HBox([w_subj, w_word, w_sess]),
    HBox([w_k, w_method, w_he_max]),
    out
]))
"""

# ──────────────────────────────────────────────────────────────
# COSTRUZIONE NUOVE CELLE
# ──────────────────────────────────────────────────────────────

def make_code_cell(source_str):
    """Crea una cella di codice con source come lista di righe."""
    lines = source_str.split("\n")
    src_lines = [l + "\n" for l in lines]
    if src_lines:
        src_lines[-1] = src_lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src_lines,
    }

def make_md_cell(source_str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_str,
    }

new_cells = [
    make_md_cell(MD_INTRO["source"]),
    make_code_cell(CELL_CONFIG),
    make_code_cell(CELL_IMPORTS),
    {   # Cella parsing — usa source come lista pre-costruita
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CELL_PARSING_LINES,
    },
    make_code_cell(CELL_CONN_FN),
    make_code_cell(CELL_ITER_TRIALS),
    make_code_cell(CELL_BUILD_GRAPHS),
    make_code_cell(CELL_BUILD_HGRAPH),
    make_code_cell(CELL_SANITY),
    make_code_cell(CELL_CONN_ANALYSIS),
    make_code_cell(CELL_PRUNING_PARAMS),
    make_code_cell(CELL_PRUNING_BUILD),
    make_code_cell(CELL_VIZ_INTERACTIVE),
]

nb["cells"] = new_cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ EEG_07e aggiornato — {len(new_cells)} celle scritte.")
