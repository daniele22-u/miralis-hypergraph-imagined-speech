"""
Comprehensive EEG Feature Extraction Script
-------------------------------------------
Extracts temporal, spectral, and functional features from EEG epochs.
Features are computed per electrode and stored in a unified dataframe.

Feature Types:
1. Temporal: Statistical measures of the raw signal
2. Spectral: Power spectral density and band powers
3. Functional: Connectivity and synchronization metrics
"""

import h5py
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, stats
from typing import Dict, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_channel_names_from_eloc, decode_label


def load_epochs_from_h5(path_h5: Path, fs: int = 256, eloc_path: Optional[Path] = None):
    """Load EEG epochs from HDF5 file with proper channel names"""
    with h5py.File(path_h5, "r") as f:
        data = f["data"][:]  # type: ignore
        labels = f["labels"][:]  # type: ignore
        # Try to read a subject identifier from the file; if it's
        # missing or formatted oddly (e.g., an array), fall back to
        # deriving it from the filename `XX_YY.h5`.
        try:
            subj_ds = f.get("subject", None)
            subj_val = subj_ds[()] if subj_ds is not None else None
        except Exception:
            subj_val = None
    
    if eloc_path and eloc_path.exists():
        ch_names = load_channel_names_from_eloc(eloc_path)
    else:
        ch_names = [f"EEG{i+1}" for i in range(data.shape[1])]  # type: ignore
    
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    ep = mne.EpochsArray(data, info)
    lbl = [decode_label(l) for l in labels]  # type: ignore

    # Normalize subject id: prefer a scalar/string from the HDF5 `subject`
    # dataset when it's a simple scalar/bytes; otherwise fall back to the
    # filename convention `{subject}_{session}.h5`.
    subject_id = None
    if subj_val is not None:
        # If it's bytes or a scalar, try to decode/convert
        try:
            if isinstance(subj_val, (bytes, bytearray)):
                subject_id = subj_val.decode('utf-8', 'ignore')
            elif isinstance(subj_val, (str, int, float)):
                subject_id = str(subj_val)
            elif isinstance(subj_val, (np.ndarray, list)) and subj_val.size == 1:
                subject_id = str(subj_val.tolist()[0])
        except Exception:
            subject_id = None

    if not subject_id:
        # Fallback: derive subject from filename (part before first underscore)
        subject_id = path_h5.stem.split("_")[0] if "_" in path_h5.stem else path_h5.stem

    ep.metadata = pd.DataFrame({
        "label_name": lbl,
        "subject_id": str(subject_id),
        "epoch_idx": np.arange(len(lbl))
    })
    return ep


# ==================== TEMPORAL FEATURES ====================

def extract_temporal_features(signal_1d: np.ndarray) -> Dict[str, float]:
    """
    Extract temporal domain features from a 1D signal.
    
    Features:
    - Mean, Std, Variance
    - Skewness, Kurtosis
    - Min, Max, Range, Peak-to-peak
    - RMS (Root Mean Square)
    - Zero crossing rate
    - Hjorth parameters (Activity, Mobility, Complexity)
    """
    feats = {}
    
    # Basic statistics
    feats['temp_mean'] = np.mean(signal_1d)
    feats['temp_std'] = np.std(signal_1d)
    feats['temp_var'] = np.var(signal_1d)
    feats['temp_min'] = np.min(signal_1d)
    feats['temp_max'] = np.max(signal_1d)
    feats['temp_range'] = feats['temp_max'] - feats['temp_min']
    feats['temp_ptp'] = np.ptp(signal_1d)
    
    # Higher order moments
    feats['temp_skewness'] = stats.skew(signal_1d)
    feats['temp_kurtosis'] = stats.kurtosis(signal_1d)
    
    # RMS
    feats['temp_rms'] = np.sqrt(np.mean(signal_1d**2))
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(signal_1d)) != 0)
    feats['temp_zcr'] = zero_crossings / len(signal_1d)
    
    # Hjorth parameters
    # Activity: variance
    feats['temp_hjorth_activity'] = np.var(signal_1d)
    
    # Mobility: sqrt(var(derivative)/var(signal))
    first_deriv = np.diff(signal_1d)
    feats['temp_hjorth_mobility'] = np.sqrt(np.var(first_deriv) / np.maximum(np.var(signal_1d), 1e-12))
    
    # Complexity: mobility(derivative) / mobility(signal)
    second_deriv = np.diff(first_deriv)
    mobility_deriv = np.sqrt(np.var(second_deriv) / np.maximum(np.var(first_deriv), 1e-12))
    feats['temp_hjorth_complexity'] = mobility_deriv / np.maximum(feats['temp_hjorth_mobility'], 1e-12)
    
    return feats


# ==================== SPECTRAL FEATURES ====================

def compute_psd_welch(signal_1d: np.ndarray, fs: int = 256, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density using Welch's method"""
    freqs, psd = signal.welch(signal_1d, fs=fs, nperseg=min(nperseg, len(signal_1d)))
    return freqs, psd


def extract_spectral_features(signal_1d: np.ndarray, fs: int = 256) -> Dict[str, float]:
    """
    Extract spectral domain features.
    
    Features:
    - Band powers (delta, theta, alpha, beta, gamma) - absolute and relative
    - Spectral edge frequency (95% power)
    - Spectral entropy
    - Dominant frequency and its power
    - Band power ratios
    """
    feats = {}
    
    # Compute PSD
    freqs, psd = compute_psd_welch(signal_1d, fs=fs)
    
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Compute band powers
    total_power = np.trapezoid(psd, freqs)
    
    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.trapezoid(psd[idx], freqs[idx])
        feats[f'spec_{band_name}'] = band_power
        feats[f'spec_{band_name}_rel'] = band_power / np.maximum(total_power, 1e-12)
    
    feats['spec_total_power'] = total_power
    
    # Band ratios
    feats['spec_alpha_beta_ratio'] = feats['spec_alpha_rel'] / np.maximum(feats['spec_beta_rel'], 1e-12)
    feats['spec_theta_alpha_ratio'] = feats['spec_theta_rel'] / np.maximum(feats['spec_alpha_rel'], 1e-12)
    feats['spec_theta_beta_ratio'] = feats['spec_theta_rel'] / np.maximum(feats['spec_beta_rel'], 1e-12)
    
    # Spectral edge frequency (95%)
    cumsum_psd = np.cumsum(psd)
    edge_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
    feats['spec_edge_freq'] = freqs[edge_idx[0]] if len(edge_idx) > 0 else 0.0
    
    # Spectral entropy
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
    feats['spec_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
    
    # Dominant frequency
    dom_idx = np.argmax(psd)
    feats['spec_dominant_freq'] = freqs[dom_idx]
    feats['spec_dominant_power'] = psd[dom_idx]
    
    # Mean and median frequency
    feats['spec_mean_freq'] = np.sum(freqs * psd) / np.sum(psd)
    cumsum_norm = np.cumsum(psd) / np.sum(psd)
    median_idx = np.where(cumsum_norm >= 0.5)[0]
    feats['spec_median_freq'] = freqs[median_idx[0]] if len(median_idx) > 0 else 0.0
    
    return feats


# ==================== FUNCTIONAL FEATURES ====================

def extract_functional_features(data_2d: np.ndarray, channel_idx: int) -> Dict[str, float]:
    """
    Extract functional connectivity features for a single channel.
    
    Features:
    - Mean correlation with all other channels
    - Max correlation with any other channel
    - Number of strong connections (correlation > 0.7)
    - Phase locking value (PLV) with other channels
    
    Args:
        data_2d: (n_channels, n_samples) array
        channel_idx: Index of the current channel
    """
    feats = {}
    n_channels, n_samples = data_2d.shape
    
    # Correlation-based connectivity
    correlations = []
    for other_idx in range(n_channels):
        if other_idx != channel_idx:
            corr = np.corrcoef(data_2d[channel_idx], data_2d[other_idx])[0, 1]
            correlations.append(abs(corr))
    
    if len(correlations) > 0:
        feats['func_mean_corr'] = np.mean(correlations)
        feats['func_max_corr'] = np.max(correlations)
        feats['func_std_corr'] = np.std(correlations)
        feats['func_num_strong_conn'] = np.sum(np.array(correlations) > 0.7)
    else:
        feats['func_mean_corr'] = 0.0
        feats['func_max_corr'] = 0.0
        feats['func_std_corr'] = 0.0
        feats['func_num_strong_conn'] = 0
    
    # Phase Locking Value (simplified version using Hilbert transform)
    try:
        # Apply Hilbert transform to get instantaneous phase
        analytic_signal = signal.hilbert(data_2d[channel_idx])
        phase_current = np.angle(analytic_signal)
        
        plvs = []
        for other_idx in range(n_channels):
            if other_idx != channel_idx:
                analytic_other = signal.hilbert(data_2d[other_idx])
                phase_other = np.angle(analytic_other)
                phase_diff = phase_current - phase_other
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plvs.append(plv)
        
        if len(plvs) > 0:
            feats['func_mean_plv'] = np.mean(plvs)
            feats['func_max_plv'] = np.max(plvs)
        else:
            feats['func_mean_plv'] = 0.0
            feats['func_max_plv'] = 0.0
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        # Handle numerical errors in Hilbert transform or phase computation
        feats['func_mean_plv'] = 0.0
        feats['func_max_plv'] = 0.0
    
    return feats


# ==================== MAIN EXTRACTION FUNCTION ====================

def extract_all_features_for_epochs(epochs, session_id: str, extract_functional: bool = True):
    """
    Extract all features (temporal, spectral, functional) for all epochs and channels.
    
    Returns:
        DataFrame with one row per (epoch, channel) combination
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    n_epochs, n_channels, n_samples = data.shape
    fs = int(epochs.info['sfreq'])
    
    rows = []
    
    for e in range(n_epochs):
        epoch_data = data[e]  # (n_channels, n_samples)
        
        for ch in range(n_channels):
            channel_name = epochs.info['ch_names'][ch]
            signal_1d = epoch_data[ch]
            
            # Base info
            row = {
                'subject_id': epochs.metadata['subject_id'].iloc[e],
                'session_id': session_id,
                'epoch_idx': e,
                'channel': channel_name,
                'label_name': epochs.metadata['label_name'].iloc[e]
            }
            
            # Extract temporal features
            temp_feats = extract_temporal_features(signal_1d)
            row.update(temp_feats)
            
            # Extract spectral features
            spec_feats = extract_spectral_features(signal_1d, fs=fs)
            row.update(spec_feats)
            
            # Extract functional features
            if extract_functional:
                func_feats = extract_functional_features(epoch_data, ch)
                row.update(func_feats)
            
            rows.append(row)
        
        if (e + 1) % 50 == 0:
            print(f"  Processed {e + 1}/{n_epochs} epochs...")
    
    return pd.DataFrame(rows)


def iterate_all_files_and_extract(meta_csv_path: Path, fs: int = 256, eloc_path: Optional[Path] = None):
    """
    Iterate over all H5 files and extract comprehensive features.
    """
    meta = pd.read_csv(meta_csv_path)
    meta['subject_id'] = meta['subject_id'].astype(str).str.strip()
    
    all_features = []
    
    for subj, grp in meta.groupby('subject_id'):
        for p in sorted(grp['path_h5'].unique()):
            pth = Path(p)
            if not pth.is_absolute():
                pth = meta_csv_path.parent.parent / "processed" / Path(p).name
            
            if not pth.exists():
                print(f"Missing file: {pth}")
                continue
            
            session_id = pth.stem.split("_")[1] if "_" in pth.stem else "unknown"
            print(f"\nProcessing {subj} | {session_id}: {pth.name}")
            
            epochs = load_epochs_from_h5(pth, fs=fs, eloc_path=eloc_path)
            df_feats = extract_all_features_for_epochs(epochs, session_id, extract_functional=True)
            all_features.append(df_feats)
            
            print(f"  Extracted {len(df_feats)} feature rows")
    
    return pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    meta_csv = project_root / "data" / "interim" / "eeg_metadata.csv"
    eloc_path = project_root / "src" / "io" / "ebneuro.locs"
    
    print("Starting comprehensive feature extraction...")
    df_features = iterate_all_files_and_extract(meta_csv, eloc_path=eloc_path)
    
    output_path = project_root / "data" / "interim" / "comprehensive_features.csv"
    df_features.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df_features)} rows to {output_path}")
    print(f"✓ Feature columns: {len(df_features.columns)}")
