"""
Utility functions for EEG data processing
-----------------------------------------
Shared helper functions used across multiple scripts.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def load_channel_names_from_eloc(eloc_path: Path) -> List[str]:
    """
    Load channel names from .eloc montage file.
    
    Args:
        eloc_path: Path to .eloc file
        
    Returns:
        List of channel names
    """
    names = []
    try:
        with open(eloc_path, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if parts:
                    names.append(parts[-1])
    except Exception:
        try:
            df = pd.read_csv(eloc_path, sep=r"\s+", header=None, engine="python", comment="#")
            names = df.iloc[:, -1].astype(str).tolist()
        except Exception:
            names = []
    return names


def load_label_scheme(
    scheme: str,
    interim_dir: Path,
) -> Tuple[Dict[int, int], int, Dict[int, str]]:
    """
    Carica uno schema di etichettatura per i 110 label_id del dataset.

    Args:
        scheme: "raw110" | "sem5" | "pos4" | "ward4" | "ward5"
        interim_dir: path alla cartella data/interim/

    Returns:
        labelid2cluster  — dict {label_id (int) → cluster_id (int)}
        n_classes        — numero di classi distinte
        cluster_names    — dict {cluster_id (int) → nome (str)}

    Uso tipico in notebook:
        labelid2cluster, N_CLASSES, cluster_names = load_label_scheme("sem5", interim_dir)
    """
    if scheme == "raw110":
        return {i: i for i in range(110)}, 110, {i: str(i) for i in range(110)}

    # Cerca prima in configs/label_schemes/ (git-tracked, disponibile subito dopo clone),
    # poi in data/interim/ (generato localmente da EEG_00_labels_and_tasks.ipynb)
    _interim = Path(interim_dir)
    _configs = _interim.parent.parent / "configs" / "label_schemes"

    lmap_path = None
    for search_dir in [_interim, _configs]:
        candidate = search_dir / f"labelid2cluster_{scheme}.json"
        if candidate.exists():
            lmap_path = candidate
            break

    if lmap_path is None:
        raise FileNotFoundError(
            f"Schema '{scheme}' non trovato in data/interim/ né in configs/label_schemes/\n"
            f"Schemi word-based (affidabili):\n"
            f"  raw110, ward4, ward5, ward6, pos4, sem5, concr4, phon4\n"
            f"Schemi EEG-based (instabili cross-subject, ARI≈0 — solo per analisi):\n"
            f"  eeg_4, eeg_5, eeg_z4, eeg_z5, eeg_hdb2\n"
            f"I word-based sono in configs/label_schemes/ (git-tracked).\n"
            f"Gli EEG-based richiedono l'esecuzione di EEG_00_labels_and_tasks.ipynb"
        )

    with open(lmap_path, "r", encoding="utf-8") as f:
        labelid2cluster = {int(k): int(v) for k, v in json.load(f).items()}

    n_classes = len(set(labelid2cluster.values()))

    # Cerca cluster_names nelle stesse directory
    names_path = None
    for search_dir in [_interim, _configs]:
        candidate = search_dir / f"cluster_names_{scheme}.json"
        if candidate.exists():
            names_path = candidate
            break

    if names_path is not None:
        with open(names_path, "r", encoding="utf-8") as f:
            cluster_names = {int(k): v for k, v in json.load(f).items()}
    else:
        cluster_names = {i: f"C{i}" for i in range(n_classes)}

    return labelid2cluster, n_classes, cluster_names


def get_keep_channels(eloc_path: Path) -> Tuple[List[int], List[str]]:
    """
    Restituisce gli indici e i nomi dei 59 canali EEG validi.

    Logica:
      - Il file .locs ha 63 posizioni (il casco EBNeuro).
      - Il file H5 grezzo ha 61 canali: Pz e POz (posizioni 62-63 nel .locs)
        NON erano connessi al casco durante la registrazione → non presenti nell'H5.
      - Il codice prende i primi 61 nomi dal .locs (allineati con l'H5) e
        rimuove A1 e A2 (elettrodi di riferimento mastoideo, nessuna posizione
        spaziale reale — theta=0, radius=0 nel .locs).
      - Risultato: 59 canali EEG con posizioni scalari valide.

    Riepilogo:
        63 posizioni .locs
        - 2 non registrati (Pz, POz) → 61 canali H5
        - 2 riferimenti (A1, A2)     → 59 canali EEG

    Args:
        eloc_path: Path al file ebneuro.locs

    Returns:
        keep_idx   — lista di 59 indici (0-based) nel segnale H5
        keep_names — lista di 59 nomi canale corrispondenti
    """
    names_all = []
    with open(eloc_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                names_all.append(parts[3])

    # I primi 61 nomi corrispondono ai 61 canali presenti nell'H5
    # (Pz e POz — posizioni 62-63 — non erano connessi durante la registrazione)
    N_H5_CHANS  = 61
    REFERENCE   = {"A1", "A2"}   # mastoid refs, nessuna posizione spaziale
    ch_names_h5 = names_all[:N_H5_CHANS]

    keep_idx   = [i for i, n in enumerate(ch_names_h5) if n not in REFERENCE]
    keep_names = [ch_names_h5[i] for i in keep_idx]

    assert len(keep_idx) == 59, (
        f"Attesi 59 canali EEG, trovati {len(keep_idx)}. "
        f"Controlla il file .locs: {eloc_path}"
    )
    return keep_idx, keep_names


def decode_label(label_raw) -> str:
    """
    Decode label from various formats (bytes, string, etc.)
    
    Args:
        label_raw: Raw label value (bytes, str, etc.)
        
    Returns:
        Decoded label as string
    """
    if isinstance(label_raw, (bytes, bytearray)):
        # Try multiple encodings
        for encoding in ('utf-8', 'latin-1', 'cp1252'):
            try:
                return label_raw.decode(encoding).strip()
            except (UnicodeDecodeError, AttributeError):
                continue
        # Fallback
        return str(label_raw)
    else:
        return str(label_raw).strip()
