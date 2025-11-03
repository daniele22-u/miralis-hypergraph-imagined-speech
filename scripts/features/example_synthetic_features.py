"""
Example: Feature Extraction from Synthetic EEG Data
---------------------------------------------------
Demonstrates the feature extraction pipeline using synthetic EEG data.
This can be run without actual EEG datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.features.comprehensive_features import (
    extract_temporal_features,
    extract_spectral_features,
    extract_functional_features
)


def generate_synthetic_eeg(n_channels=10, n_samples=384, fs=256, noise_level=0.5):
    """
    Generate synthetic EEG-like data with realistic spectral characteristics.
    
    Args:
        n_channels: Number of EEG channels
        n_samples: Number of samples (at 256 Hz, 384 samples = 1.5 seconds)
        fs: Sampling frequency
        noise_level: Amount of noise to add
    
    Returns:
        np.ndarray: Synthetic EEG data (n_channels, n_samples)
    """
    t = np.arange(n_samples) / fs
    data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Add different frequency components (EEG bands)
        delta = 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz (delta)
        theta = 0.3 * np.sin(2 * np.pi * 6 * t)  # 6 Hz (theta)
        alpha = 0.8 * np.sin(2 * np.pi * 10 * t)  # 10 Hz (alpha)
        beta = 0.4 * np.sin(2 * np.pi * 20 * t)  # 20 Hz (beta)
        gamma = 0.2 * np.sin(2 * np.pi * 35 * t)  # 35 Hz (gamma)
        
        # Add channel-specific phase shifts
        phase_shift = ch * 0.1
        
        # Combine components
        signal = (delta + theta + alpha + beta + gamma) * np.cos(2 * np.pi * phase_shift)
        
        # Add noise
        noise = noise_level * np.random.randn(n_samples)
        
        data[ch] = signal + noise
    
    return data


def demonstrate_feature_extraction():
    """Demonstrate feature extraction on synthetic data"""
    
    print("=" * 70)
    print("COMPREHENSIVE EEG FEATURE EXTRACTION - DEMONSTRATION")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic EEG data...")
    n_channels = 10
    n_samples = 384
    fs = 256
    
    eeg_data = generate_synthetic_eeg(n_channels=n_channels, n_samples=n_samples, fs=fs)
    print(f"   Generated: {n_channels} channels × {n_samples} samples ({n_samples/fs:.1f} seconds)")
    
    # Extract features for one channel
    print("\n2. Extracting features for Channel 1...")
    channel_signal = eeg_data[0]
    
    # Temporal features
    print("\n   [Temporal Features]")
    temp_feats = extract_temporal_features(channel_signal)
    for feat_name, feat_value in list(temp_feats.items())[:5]:
        print(f"   - {feat_name}: {feat_value:.4f}")
    print(f"   ... and {len(temp_feats) - 5} more temporal features")
    
    # Spectral features
    print("\n   [Spectral Features]")
    spec_feats = extract_spectral_features(channel_signal, fs=fs)
    print(f"   - Total Power: {spec_feats['spec_total_power']:.4f} µV²")
    print(f"   - Alpha Power: {spec_feats['spec_alpha']:.4f} µV²")
    print(f"   - Beta Power: {spec_feats['spec_beta']:.4f} µV²")
    print(f"   - Dominant Frequency: {spec_feats['spec_dominant_freq']:.2f} Hz")
    print(f"   - Spectral Entropy: {spec_feats['spec_entropy']:.4f}")
    print(f"   ... and {len(spec_feats) - 5} more spectral features")
    
    # Functional features
    print("\n   [Functional Features]")
    # Pre-compute for efficiency (though not critical for single call)
    corr_matrix = np.corrcoef(eeg_data)
    try:
        from scipy import signal
        phase_data = np.angle(signal.hilbert(eeg_data, axis=1))
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        phase_data = None
    
    func_feats = extract_functional_features(eeg_data, channel_idx=0, 
                                            corr_matrix=corr_matrix, 
                                            phase_data=phase_data)
    print(f"   - Mean Correlation: {func_feats['func_mean_corr']:.4f}")
    print(f"   - Max Correlation: {func_feats['func_max_corr']:.4f}")
    print(f"   - Mean PLV: {func_feats['func_mean_plv']:.4f}")
    print(f"   - Strong Connections: {func_feats['func_num_strong_conn']}")
    
    # Extract features for all channels
    print("\n3. Extracting features for all channels...")
    all_features = []
    
    # Pre-compute correlation matrix and phase data for efficiency
    corr_matrix = np.corrcoef(eeg_data)
    try:
        from scipy import signal
        phase_data = np.angle(signal.hilbert(eeg_data, axis=1))
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        phase_data = None
    
    for ch in range(n_channels):
        features = {
            'channel': f'CH{ch+1}',
            'epoch_idx': 0
        }
        
        # Extract all feature types
        features.update(extract_temporal_features(eeg_data[ch]))
        features.update(extract_spectral_features(eeg_data[ch], fs=fs))
        features.update(extract_functional_features(eeg_data, channel_idx=ch, 
                                                    corr_matrix=corr_matrix, 
                                                    phase_data=phase_data))
        
        all_features.append(features)
    
    df = pd.DataFrame(all_features)
    print(f"   Created DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Display summary
    print("\n4. Feature Summary:")
    print(f"   - Temporal features: {len([c for c in df.columns if c.startswith('temp_')])}")
    print(f"   - Spectral features: {len([c for c in df.columns if c.startswith('spec_')])}")
    print(f"   - Functional features: {len([c for c in df.columns if c.startswith('func_')])}")
    
    # Visualize some features
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Raw signal
    axes[0, 0].plot(eeg_data[0], linewidth=0.8, color='darkblue')
    axes[0, 0].set_title('Raw EEG Signal - Channel 1', fontweight='bold')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude (µV)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Band powers across channels
    band_cols = ['spec_delta', 'spec_theta', 'spec_alpha', 'spec_beta', 'spec_gamma']
    band_data = df[band_cols].values.T
    
    axes[0, 1].bar(range(len(band_cols)), df[band_cols].mean().values, 
                   color=['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12'], alpha=0.7)
    axes[0, 1].set_xticks(range(len(band_cols)))
    axes[0, 1].set_xticklabels(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
    axes[0, 1].set_title('Average Band Powers', fontweight='bold')
    axes[0, 1].set_ylabel('Power (µV²)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Power distribution across channels
    axes[1, 0].bar(df['channel'], df['spec_total_power'], color='steelblue', alpha=0.7)
    axes[1, 0].set_title('Total Power Across Channels', fontweight='bold')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Total Power (µV²)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Connectivity matrix
    corr_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                corr_matrix[i, j] = np.corrcoef(eeg_data[i], eeg_data[j])[0, 1]
            else:
                corr_matrix[i, j] = 1.0
    
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Inter-Channel Correlation', fontweight='bold')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Channel')
    plt.colorbar(im, ax=axes[1, 1], label='Correlation')
    
    plt.suptitle('Synthetic EEG Feature Extraction Example', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "figures" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_eeg_features.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Visualization saved to: {output_path}")
    
    plt.show()
    
    # Save dataframe
    csv_path = output_dir / "synthetic_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n6. Features saved to: {csv_path}")
    
    # Display feature statistics
    print("\n7. Feature Statistics:")
    print("\n" + "="*70)
    print(df.describe()[['spec_total_power', 'spec_alpha_rel', 'temp_rms', 'func_mean_corr']].to_string())
    print("="*70)
    
    print("\n✓ Demonstration complete!")
    print("\nNext steps:")
    print("  1. Run with real EEG data: python scripts/features/comprehensive_features.py")
    print("  2. Generate visualizations: python scripts/graphs/feature_visualizations.py")
    print("  3. Explore the tutorial: notebooks/feature_extraction_tutorial.ipynb")
    print()


if __name__ == "__main__":
    demonstrate_feature_extraction()
