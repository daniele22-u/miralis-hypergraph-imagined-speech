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

    lmap_path = Path(interim_dir) / f"labelid2cluster_{scheme}.json"
    if not lmap_path.exists():
        raise FileNotFoundError(
            f"Schema '{scheme}' non trovato: {lmap_path}\n"
            f"Schemi word-based (affidabili):\n"
            f"  raw110, ward4, ward5, ward6, pos4, sem5, concr4, phon4\n"
            f"Schemi EEG-based (instabili cross-subject, ARI≈0 — solo per analisi):\n"
            f"  eeg_4, eeg_5, eeg_z4, eeg_z5, eeg_hdb2\n"
            f"Tutti richiedono l'esecuzione di EEG_00_labels_and_tasks.ipynb"
        )

    with open(lmap_path, "r", encoding="utf-8") as f:
        labelid2cluster = {int(k): int(v) for k, v in json.load(f).items()}

    n_classes = len(set(labelid2cluster.values()))

    names_path = Path(interim_dir) / f"cluster_names_{scheme}.json"
    if names_path.exists():
        with open(names_path, "r", encoding="utf-8") as f:
            cluster_names = {int(k): v for k, v in json.load(f).items()}
    else:
        cluster_names = {i: f"C{i}" for i in range(n_classes)}

    return labelid2cluster, n_classes, cluster_names


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
