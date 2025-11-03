"""
Utility functions for EEG data processing
-----------------------------------------
Shared helper functions used across multiple scripts.
"""

import pandas as pd
from pathlib import Path
from typing import List


def load_channel_names_from_eloc(eloc_path: Path) -> List[str]:
    """
    Load channel names from .eloc montage file.
    
    Args:
        eloc_path: Path to .eloc file
        
    Returns:
        List of channel names
    """
    df = pd.read_csv(eloc_path, sep=r"\s+", header=None, engine="python")
    names = df.iloc[:, -1].astype(str).tolist()
    return names


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
