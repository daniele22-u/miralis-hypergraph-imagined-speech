import json
import sys

nb_path = "/home/daniele_u/miralis-hypergraph-imagined-speech/notebooks/EEG_08_subject_clustering.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find setup cell
for cell in nb["cells"]:
    if cell.get("id") == "01-setup" or "01-setup" in cell.get("metadata", {}).get("id", "") or "from pathlib import Path" in "".join(cell.get("source", [])):
        source = cell["source"]
        
        # We replace the root detection block up to feature printing
        setup_code = """import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, adjusted_rand_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import Patch

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print('umap-learn non trovato — skip UMAP (solo PCA).')

# ----- Root detection e Config path -----
import sys
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
    \"\"\"'P003_S002' → (3, 2)\"\"\"
    parts = folder_name.split("_")
    return int(parts[0][1:]), int(parts[1][1:])

ROOT     = project_root
DATA_INT = ROOT / 'data' / 'interim'
FIGURES  = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

ACC_CSV  = DATA_INT / 'eeg08c_ss_4_norm_results.csv'
OUT_CSV  = DATA_INT / 'eeg08_subject_profiles.csv'

print(f'ROOT     → {ROOT}')
print(f'CSV_ROOT → {CSV_ROOT}')
n_paolo = len(list(CSV_ROOT.glob('*')))
print(f'Soggetti trovati: {n_paolo}')
print(f'ACC_CSV  → {ACC_CSV.exists()}')

# ----- Feature -----
SPEC_FEAT = ['spec_delta','spec_theta','spec_alpha','spec_beta','spec_gamma',
             'spec_delta_rel','spec_theta_rel','spec_alpha_rel','spec_beta_rel','spec_gamma_rel',
             'spec_total_power','spec_alpha_beta_ratio','spec_theta_alpha_ratio',
             'spec_theta_beta_ratio','spec_edge_freq','spec_entropy',
             'spec_dominant_freq','spec_dominant_power','spec_mean_freq','spec_median_freq']
TEMP_FEAT = ['temp_std','temp_skewness','temp_kurtosis','temp_rms','temp_zcr',
             'temp_hjorth_activity','temp_hjorth_mobility','temp_hjorth_complexity']
FUNC_FEAT = ['func_mean_corr','func_max_corr','func_std_corr',
             'func_num_strong_conn','func_mean_plv','func_max_plv']
ALL_FEAT  = SPEC_FEAT + TEMP_FEAT + FUNC_FEAT

FISHER_FEAT = ['spec_delta_rel','spec_theta_rel','spec_alpha_rel','spec_beta_rel','spec_gamma_rel',
               'func_mean_corr','func_mean_plv','temp_rms','temp_hjorth_mobility']
ITC_FEAT = ['spec_gamma_rel','spec_alpha_rel','spec_beta_rel',
            'func_mean_corr','func_mean_plv','temp_rms']

print(f'\\nFeature totali: {len(ALL_FEAT)}  ({len(SPEC_FEAT)} spec + {len(TEMP_FEAT)} temp + {len(FUNC_FEAT)} func)')
print(f'Fisher feat: {len(FISHER_FEAT)}  |  ITC feat: {len(ITC_FEAT)}')

# ----- Stile -----
sns.set_theme(style='whitegrid', palette='tab10', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
"""
        lines = [line + "\n" for line in setup_code.split("\n")]
        lines[-1] = lines[-1].strip("\n")
        cell["source"] = lines
        break

# Find load cell
for cell in nb["cells"]:
    if cell.get("id") == "02-load" or "02-load" in cell.get("metadata", {}).get("id", "") or "CACHE_PARQUET" in "".join(cell.get("source", [])):
        
        load_code = """# Estrazione feature dai CSV di Paolo, salvando in cache locale

CACHE_PARQUET = DATA_INT / 'eeg08_raw_csv_features_cache.parquet'
FORCE_RELOAD  = False   # True per ricalcolare anche se cache esiste

if CACHE_PARQUET.exists() and not FORCE_RELOAD:
    print(f'Cache parquet trovata — carico direttamente...')
    df_raw = pd.read_parquet(CACHE_PARQUET)
    print(f'Shape: {df_raw.shape}')
else:
    import json
    from features.comprehensive_features import (
        extract_temporal_features,
        extract_spectral_features,
        extract_functional_features
    )
    from utils import load_channel_names_from_eloc

    print(f'Cache CSV feature non trovata. Calcolo partendo dai raw csv di Paolo...')
    
    with open(CONFIGS / "label2idx.json") as f:
        word2labelid = json.load(f)
    
    ch_names = load_channel_names_from_eloc(project_root / "src" / "io" / "ebneuro.locs")
    
    def load_csv_trial(csv_path: Path):
        return pd.read_csv(csv_path, header=None).values.astype(np.float32)

    rows = []
    # Itera sui subject directory in training_set
    for sess_dir in sorted(CSV_ROOT.iterdir()):
        if not sess_dir.is_dir(): continue
        subj_id, sess_id = parse_folder(sess_dir.name)
        
        trials = sorted([p for p in sess_dir.iterdir() if p.suffix == ".csv"])
        print(f"Estraggo subject {subj_id}, session {sess_id}: {len(trials)} trials...")
        
        for epoch_idx, csv_path in enumerate(trials):
            word = csv_path.stem.replace("_img", "")
            if word not in word2labelid:
                continue
                
            label_id = word2labelid[word]
            label_name = word # nome della parola e.g. "cane"
            
            data_2d = load_csv_trial(csv_path)  # (61, 384) o (num_canali, time)
            n_chans = data_2d.shape[0]
            
            for ch in range(n_chans):
                ch_name = ch_names[ch] if ch < len(ch_names) else f"EEG{ch}"
                signal_1d = data_2d[ch]
                
                row = {
                    'subject_id': subj_id,
                    'session_id': sess_id,
                    'epoch_idx': epoch_idx,
                    'channel': ch_name,
                    'label_name': label_name
                }
                
                # Calcola spec, temp, e func (sarebbe un po' lento per func ma richiesto dall'analisi)
                row.update(extract_temporal_features(signal_1d))
                row.update(extract_spectral_features(signal_1d, fs=256))
                
                # Funzionali (corr media vs altri canali, plv)
                row.update(extract_functional_features(data_2d, ch))
                
                rows.append(row)

    df_raw = pd.DataFrame(rows)
    print(f'Concatenazione completata. Salvo cache parquet in {CACHE_PARQUET.name}...')
    df_raw.to_parquet(CACHE_PARQUET, index=False)

df_raw['subject_id'] = pd.to_numeric(df_raw['subject_id'], errors='coerce')
df_raw = df_raw.dropna(subset=['subject_id'])
df_raw['subject_id'] = df_raw['subject_id'].astype(int)

# In caso il json e la pipeline non mettano "temp_mean" (usato nel display), aggiungiamolo per retro-compatibilita se manca
if 'temp_mean' not in df_raw.columns:
    df_raw['temp_mean'] = 0.0

print(f'\\nShape: {df_raw.shape}')
print(f'Soggetti : {df_raw["subject_id"].nunique()}')
print(f'Canali   : {df_raw["channel"].nunique()}')
print(f'Parole   : {df_raw["label_name"].nunique()}')
"""
        
        lines = [line + "\n" for line in load_code.split("\n")]
        lines[-1] = lines[-1].strip("\n")
        cell["source"] = lines
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated.")
