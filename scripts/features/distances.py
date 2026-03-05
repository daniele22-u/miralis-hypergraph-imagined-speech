"""
Script: distances.py
Author: Daniele

Description:
    Computes pairwise distances between EEG electrodes based on their 3D positions
    defined in the file 'ebneuro.locs'. Produces two NxN distance matrices:

        1. Euclidean distances (straight-line distance between electrodes)
        2. Geodesic distances (along the scalp surface)

    Both matrices are saved as CSV files in:
        data/interim/

    Example output:
        - channel_pairwise_euclidean_distances.csv
        - channel_pairwise_geodesic_distances.csv

    The script also prints the geodesic distance between two reference electrodes (Cz–T4).

    To run from the project root:
        python scripts/features/distances.py
"""


import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from pathlib import Path
from typing import List, Tuple, Dict
import mne


def geodesic_distance_matrix_from_ch_pos(
    ch_pos: Dict[str, np.ndarray], 
    channel_order: List[str] = None,  # type: ignore
    radius: float = None # type: ignore
) -> Tuple[List[str], np.ndarray]:
    if channel_order is None:
        names = list(ch_pos.keys())
    else:
        names = [ch for ch in channel_order if ch in ch_pos]

    P = np.array([ch_pos[ch][:3] for ch in names], dtype=float)
    norms = np.linalg.norm(P, axis=1)

    if radius is None:
        if not np.all(np.isfinite(norms)) or np.any(norms == 0):
            raise ValueError("Invalid channel positions.")
        R = float(np.mean(norms))
    else:
        R = float(radius)

    U = P / norms[:, None]
    dot = np.clip(U @ U.T, -1.0, 1.0)
    theta = np.arccos(dot)
    D_geo = R * theta
    return names, D_geo


def main():
    project_root = Path(__file__).resolve().parents[2]

    locs_path = project_root / "src" / "io" / "ebneuro.locs"

    output_dir = project_root / "data" / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)

    montage = mne.channels.read_custom_montage(str(locs_path))
    ch_pos = montage.get_positions()['ch_pos']
    ch_names = list(ch_pos.keys()) # type: ignore
    pos_array = np.array([ch_pos[ch][:3] for ch in ch_names]) # type: ignore

    euclidean_distance = distance_matrix(pos_array, pos_array)
    df_euclidean = pd.DataFrame(euclidean_distance, index=ch_names, columns=ch_names)
    df_euclidean.to_csv(output_dir / "channel_pairwise_euclidean_distances.csv")

    names, D_geo = geodesic_distance_matrix_from_ch_pos(ch_pos, channel_order=ch_names, radius=0.095) # type: ignore
    df_geodesic = pd.DataFrame(D_geo, index=names, columns=names)
    df_geodesic.to_csv(output_dir / "channel_pairwise_geodesic_distances.csv")

    if 'Cz' in df_geodesic.index and 'T4' in df_geodesic.columns:
        print("Geodesic distance Cz–T4:", float(df_geodesic.loc['Cz', 'T4']), "m\n") # type: ignore
        print("Euclidean distance Cz–T4:", float(df_euclidean.loc['Cz', 'T4']), "m\n") # type: ignore
    else:
        print("Cz or T4 not found in channel names.")

    print("✅ Saved distance matrices in:", output_dir)


if __name__ == "__main__":
    main()
