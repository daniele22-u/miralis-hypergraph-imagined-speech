import json
import sys

nb_path = "/home/daniele_u/miralis-hypergraph-imagined-speech/notebooks/EEG_07b_precompute_graphs.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1: Setup
setup_code = """from pathlib import Path
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as sp_hilbert
from torch_geometric.data import Data
import multiprocessing as mp

project_root = next(
    (p for p in [Path().resolve()] + list(Path().resolve().parents)
     if (p / ".git").exists()),
    Path().resolve()
)

sys.path.insert(0, str(project_root / "scripts"))
from utils import load_label_scheme

CSV_ROOT   = project_root / "data" / "raw_csv" / "training_set"
CONFIGS    = project_root / "configs" / "label_schemes"

def parse_folder(folder_name: str):
    \"\"\"'P003_S002' -> (3, 2)\"\"\"
    parts = folder_name.split("_")
    return int(parts[0][1:]), int(parts[1][1:])

ROOT     = project_root
DATA_INT = ROOT / 'data' / 'interim'
FIGURES  = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

ELOC_PATH  = project_root / "src" / "io" / "ebneuro.locs"
GRAPHS_DIR = project_root / "data" / "interim" / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

K_GRAPH  = 6
SFREQ    = 256
N_CHANS  = 59

# Metodi disponibili:
#   "pcc"      -- Pearson |r| (veloce)
#   "plv"      -- Phase Locking Value (bande theta+alpha)
#   "wpli"     -- Weighted Phase Lag Index (bande theta+alpha)
#   "cpcc_abs" -- CPCCabs = |mean(exp(iDeltaPhi))| (Iacomi et al. 2026)
#   "cpcc_im"  -- CPCCim  = |mean(sin(DeltaPhi))|  (Iacomi et al. 2026)
#   "learned" / "dynamic" -- sentinel edge_index (grafo appreso in forward)
METHODS = ["pcc", "plv", "wpli", "cpcc_abs", "cpcc_im", "learned", "dynamic"]

# Bande freq. per PLV e wPLI (cpcc_abs/cpcc_im usano hilbert broadband)
PLV_BANDS = [(4, 8), (8, 13)]   # theta + alpha

# Forza ricostruzione anche se il .pt esiste gia'
FORCE_REBUILD = False

print(f"project_root : {project_root}")
print(f"Output dir   : {GRAPHS_DIR}")
print(f"K_GRAPH      : {K_GRAPH}  |  SFREQ={SFREQ}")
print(f"Metodi       : {METHODS}")
"""

# Cell 2: Meta loading
load_code = """import json

def read_eloc_names(path):
    names = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                names.append(parts[3])
    return names[:61]

ch_names_61 = read_eloc_names(ELOC_PATH)
EXCLUDE     = {"A1", "A2"}
keep_idx    = [i for i, n in enumerate(ch_names_61) if n not in EXCLUDE]
assert len(keep_idx) == N_CHANS

with open(CONFIGS / "label2idx.json") as f:
    word2labelid = json.load(f)

records = []
for sess_dir in sorted(CSV_ROOT.iterdir()):
    if not sess_dir.is_dir(): continue
    subj_id, _ = parse_folder(sess_dir.name)
    trials = sorted([p for p in sess_dir.iterdir() if p.suffix == ".csv"])
    for csv_path in trials:
        word = csv_path.stem.replace("_img", "")
        if word not in word2labelid:
            continue
        label_id = word2labelid[word]
        records.append({
            "path_csv": str(csv_path),
            "label_idx": label_id,
            "subject_id": subj_id,
        })

print(f"Trial totali : {len(records)}")
print(f"Canali       : {len(keep_idx)}")
"""

run_code = """# ============================================================
# BUILD DATASET -- joblib loky (spawn, niente fork memory copy)
# Metodi: pcc | plv | wpli | cpcc_abs | cpcc_im | learned | dynamic
# ============================================================

import gc
from joblib import Parallel, delayed

N_WORKERS = max(1, mp.cpu_count() - 2)
print(f"Worker: {N_WORKERS} / {mp.cpu_count()} core  (backend: loky/spawn)")


def _compute_one(path_csv, subj_int, y_int, method, k,
                 keep_idx_local, plv_bands_local, sfreq_local):
    \"\"\"
    Worker loky -- completamente autonomo, nessuna variabile globale.

    Metodi:
      pcc      : Pearson |r| vettorizzato
      cpcc_abs : |mean(exp(i*delta_phi))| -- broadband Hilbert, vettorizzato
      cpcc_im  : |mean(sin(delta_phi))|   -- broadband Hilbert, vettorizzato
      plv      : Phase Locking Value su bande PLV_BANDS
      wpli     : Weighted Phase Lag Index su bande PLV_BANDS
      learned/dynamic : sentinel (self-loops), nessun calcolo matrice
    \"\"\"
    import numpy as np
    import pandas as pd
    from scipy.signal import butter, filtfilt, hilbert as sp_hilbert

    x_np = pd.read_csv(path_csv, header=None).values.astype(np.float32)
    x_np = x_np[keep_idx_local, :]   # (N_CHANS, T)
    N, T = x_np.shape

    def knn_from_matrix(matrix, k):
        edges = set()
        for i in range(matrix.shape[0]):
            for j in np.argsort(matrix[i])[::-1][:k]:
                edges.add((i, int(j))); edges.add((int(j), i))
        return list(zip(*sorted(edges)))

    if method == "pcc":
        mat = np.abs(np.corrcoef(x_np)).astype(np.float32)
        np.fill_diagonal(mat, 0.0)

    elif method in ("cpcc_abs", "cpcc_im"):
        # Fase instantanea broadband via Hilbert (nessun filtro a bande)
        analytic = sp_hilbert(x_np.astype(np.float64), axis=1)
        phi      = np.angle(analytic)              # (N, T)
        ep       = np.exp(1j * phi)                # (N, T) complex
        if method == "cpcc_abs":
            # |mean_t(exp(i*delta_phi))| -- ampiezza coupling di fase
            mat = np.abs(ep @ ep.conj().T / T).astype(np.float32)
        else:
            # |mean_t(sin(delta_phi))| -- componente lag, robusto volume cond.
            mat = np.abs(np.imag(ep @ ep.conj().T) / T).astype(np.float32)
        np.fill_diagonal(mat, 0.0)

    elif method in ("plv", "wpli"):
        nyq = sfreq_local / 2.0
        acc = np.zeros((N, N), dtype=np.float64)
        for flo, fhi in plv_bands_local:
            b, a     = butter(4, [flo / nyq, fhi / nyq], btype="band")
            x_filt   = filtfilt(b, a, x_np, axis=1).astype(np.float32)
            analytic = sp_hilbert(x_filt, axis=1)
            if method == "plv":
                ep  = np.exp(1j * np.angle(analytic))
                acc += np.abs(ep @ ep.conj().T) / T
            else:  # wpli -- vettorizzato riga per riga per contenere RAM
                for i in range(N):
                    cs_row = analytic[i] * np.conj(analytic)
                    im_row = np.imag(cs_row)
                    num    = np.abs(im_row.mean(axis=1))
                    den    = np.abs(im_row).mean(axis=1) + 1e-9
                    acc[i] += num / den
        mat = (acc / len(plv_bands_local)).astype(np.float32)
        np.fill_diagonal(mat, 0.0)

    elif method in ("learned", "dynamic"):
        # Sentinel: self-loops su tutti i nodi (grafo appreso in forward)
        idx = list(range(N))
        return (x_np, idx, idx, y_int, subj_int)

    else:
        raise ValueError(f"Metodo non supportato: {method}")

    src_dst = knn_from_matrix(mat, k)
    return (x_np, list(src_dst[0]), list(src_dst[1]), y_int, subj_int)


for method in METHODS:
    out_path = GRAPHS_DIR / f"dataset_{method}_k{K_GRAPH}.pt"
    if out_path.exists() and not FORCE_REBUILD:
        print(f"[{method}] gia' calcolato -- {out_path.name}  (skip)")
        continue

    print(f"\\n[{method}] Costruisco dataset ({len(records)} trial) con {N_WORKERS} worker (loky)...")

    results = Parallel(n_jobs=N_WORKERS, backend="loky", verbose=0)(
        delayed(_compute_one)(
            r["path_csv"], int(r["subject_id"]), int(r["label_idx"]),
            method, K_GRAPH, keep_idx, PLV_BANDS, SFREQ
        )
        for r in tqdm(records, desc=method)
    )

    print(f"  Assemblaggio Data objects...")
    data_list = [
        Data(
            x          = torch.tensor(x_np, dtype=torch.float32),
            edge_index = torch.tensor([src, dst], dtype=torch.long),
            y          = torch.tensor(y,    dtype=torch.long),
            subj       = torch.tensor(subj, dtype=torch.long),
        )
        for x_np, src, dst, y, subj in results
    ]

    del results; gc.collect()

    torch.save(data_list, out_path)
    size_gb = out_path.stat().st_size / 1e9
    print(f"[{method}] Salvato: {out_path.name}  ({len(data_list)} grafi, {size_gb:.2f} GB)")

    del data_list; gc.collect()
    print(f"  Memoria liberata.")

print("\\nBuild completato.")
"""

def split_and_newline(code):
    lines = [line + "\n" for line in code.split("\n")]
    if lines:
        lines[-1] = lines[-1].strip("\n")
    return lines

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "from pathlib import Path" in src and "import multiprocessing" in src:
            cell["source"] = split_and_newline(setup_code)
        elif "meta = pd.read_csv(META_CSV)" in src:
            cell["source"] = split_and_newline(load_code)
        elif "BUILD DATASET — joblib loky" in src:
            cell["source"] = split_and_newline(run_code)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook 07b modified successfully.")
