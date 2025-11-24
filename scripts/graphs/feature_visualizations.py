"""
EEG Feature Visualization Suite
-------------------------------
Creates comprehensive visualizations for EEG features:
1. Power variation of a single electrode across all epochs
2. Electrodes with highest power per epoch (dynamic)
3. Power variation during each epoch (temporal evolution)
4. Topographic maps of power distribution
5. Feature distributions and correlations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import h5py
import mne
from typing import List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_channel_names_from_eloc, decode_label


def load_comprehensive_features(csv_path: Path) -> pd.DataFrame:
    """Load the comprehensive features dataframe"""
    return pd.read_csv(csv_path)


# ==================== VISUALIZATION 1: Single Electrode Power Over Time ====================

def plot_electrode_power_across_epochs(df: pd.DataFrame, electrode: str, subject_id: str = None,
                                       output_path: Optional[Path] = None):
    """
    Visualize power variation of a single electrode across all epochs.
    Shows total power, band powers, and temporal features.
    """
    # Filter data
    df_elec = df[df['channel'] == electrode].copy()
    if subject_id:
        df_elec = df_elec[df_elec['subject_id'] == subject_id]
    
    df_elec = df_elec.sort_values('epoch_idx')
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Total power across epochs
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_elec['epoch_idx'], df_elec['spec_total_power'], 
             marker='o', markersize=3, linewidth=1, alpha=0.7)
    ax1.set_xlabel('Epoch Index', fontsize=11)
    ax1.set_ylabel('Total Power (µV²)', fontsize=11)
    ax1.set_title(f'Total Power Variation - Electrode {electrode}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Band powers stacked area
    ax2 = fig.add_subplot(gs[1, :])
    bands = ['spec_delta', 'spec_theta', 'spec_alpha', 'spec_beta', 'spec_gamma']
    band_labels = ['Delta (1-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 'Beta (13-30 Hz)', 'Gamma (30-45 Hz)']
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12']
    
    band_data = df_elec[bands].values.T
    ax2.stackplot(df_elec['epoch_idx'], band_data, labels=band_labels, colors=colors, alpha=0.7)
    ax2.set_xlabel('Epoch Index', fontsize=11)
    ax2.set_ylabel('Band Power (µV²)', fontsize=11)
    ax2.set_title(f'Band Power Distribution - Electrode {electrode}', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Relative band powers
    ax3 = fig.add_subplot(gs[2, 0])
    rel_bands = ['spec_delta_rel', 'spec_theta_rel', 'spec_alpha_rel', 'spec_beta_rel', 'spec_gamma_rel']
    for i, (band, label, color) in enumerate(zip(rel_bands, band_labels, colors)):
        ax3.plot(df_elec['epoch_idx'], df_elec[band], 
                label=label, color=color, linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Epoch Index', fontsize=11)
    ax3.set_ylabel('Relative Power', fontsize=11)
    ax3.set_title(f'Relative Band Powers - {electrode}', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Temporal features
    ax4 = fig.add_subplot(gs[2, 1])
    ax4_twin = ax4.twinx()
    
    ax4.plot(df_elec['epoch_idx'], df_elec['temp_rms'], 
            label='RMS', color='blue', linewidth=1.5, alpha=0.8)
    ax4_twin.plot(df_elec['epoch_idx'], df_elec['temp_hjorth_mobility'], 
                 label='Hjorth Mobility', color='red', linewidth=1.5, alpha=0.8, linestyle='--')
    
    ax4.set_xlabel('Epoch Index', fontsize=11)
    ax4.set_ylabel('RMS (µV)', fontsize=11, color='blue')
    ax4_twin.set_ylabel('Hjorth Mobility', fontsize=11, color='red')
    ax4.set_title(f'Temporal Features - {electrode}', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
    
    plt.suptitle(f'Comprehensive Power Analysis - Electrode {electrode}', 
                fontsize=15, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== VISUALIZATION 2: Top Electrodes Per Epoch ====================

def plot_top_electrodes_per_epoch(df: pd.DataFrame, subject_id: str = None, 
                                  top_n: int = 5, output_path: Optional[Path] = None):
    """
    Show which electrodes have the highest power in each epoch.
    Creates a heatmap-style visualization.
    """
    if subject_id:
        df = df[df['subject_id'] == subject_id]
    
    # Get unique epochs and channels
    epochs = sorted(df['epoch_idx'].unique())
    channels = sorted(df['channel'].unique())
    
    # Create matrix: epochs x channels
    power_matrix = np.zeros((len(epochs), len(channels)))
    
    for i, epoch in enumerate(epochs):
        epoch_data = df[df['epoch_idx'] == epoch]
        for j, channel in enumerate(channels):
            ch_data = epoch_data[epoch_data['channel'] == channel]
            if not ch_data.empty:
                power_matrix[i, j] = ch_data['spec_total_power'].values[0]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Heatmap of all powers
    im = ax1.imshow(power_matrix.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax1.set_xlabel('Epoch Index', fontsize=12)
    ax1.set_ylabel('Channel', fontsize=12)
    ax1.set_title('Power Distribution Across Epochs and Channels', fontsize=13, fontweight='bold')
    
    # Set ticks
    if len(epochs) > 50:
        tick_step = len(epochs) // 20
        ax1.set_xticks(range(0, len(epochs), tick_step))
        ax1.set_xticklabels(epochs[::tick_step], fontsize=8)
    else:
        ax1.set_xticks(range(0, len(epochs), max(1, len(epochs)//10)))
        ax1.set_xticklabels(epochs[::max(1, len(epochs)//10)], fontsize=8)
    
    if len(channels) > 30:
        tick_step = len(channels) // 15
        ax1.set_yticks(range(0, len(channels), tick_step))
        ax1.set_yticklabels(channels[::tick_step], fontsize=7)
    else:
        ax1.set_yticks(range(len(channels)))
        ax1.set_yticklabels(channels, fontsize=7)
    
    plt.colorbar(im, ax=ax1, label='Total Power (µV²)')
    
    # 2. Top N channels per epoch
    top_channels_per_epoch = []
    for i, epoch in enumerate(epochs):
        epoch_powers = power_matrix[i, :]
        top_indices = np.argsort(epoch_powers)[-top_n:][::-1]
        top_ch = [channels[idx] for idx in top_indices]
        top_channels_per_epoch.append(top_ch)
    
    # Count frequency of each channel being in top N
    channel_counts = {}
    for top_list in top_channels_per_epoch:
        for ch in top_list:
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
    
    sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    ch_names = [x[0] for x in sorted_channels]
    ch_counts = [x[1] for x in sorted_channels]
    
    ax2.barh(range(len(ch_names)), ch_counts, color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(ch_names)))
    ax2.set_yticklabels(ch_names, fontsize=9)
    ax2.set_xlabel(f'Frequency in Top {top_n} (out of {len(epochs)} epochs)', fontsize=11)
    ax2.set_title(f'Most Frequently High-Power Channels', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== VISUALIZATION 3: Power Evolution Within Epoch ====================

def plot_power_evolution_within_epochs(h5_path: Path, eloc_path: Path, 
                                       epoch_indices: List[int],
                                       output_path: Optional[Path] = None):
    """
    Show how power evolves during specific epochs using sliding windows.
    """
    # Load data
    with h5py.File(h5_path, "r") as f:
        data = f["data"][:]  # (n_epochs, n_channels, n_samples)
        labels = f["labels"][:]  # type: ignore
    
    # Load channel names
    ch_names = load_channel_names_from_eloc(eloc_path) if eloc_path.exists() else [f"EEG{i+1}" for i in range(data.shape[1])]
    
    fs = 256  # Sampling rate
    window_size = 64  # ~250 ms windows
    step_size = 32  # 50% overlap
    
    n_plots = len(epoch_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    for idx, epoch_idx in enumerate(epoch_indices):
        epoch_data = data[epoch_idx]  # (n_channels, n_samples)
        label = decode_label(labels[epoch_idx])  # type: ignore
        
        n_channels, n_samples = epoch_data.shape
        
        # Compute power in sliding windows
        n_windows = (n_samples - window_size) // step_size + 1
        time_points = np.arange(n_windows) * step_size / fs
        
        # Calculate average power across all channels in each window
        avg_powers = []
        for w in range(n_windows):
            start = w * step_size
            end = start + window_size
            window_data = epoch_data[:, start:end]
            # Average power across channels
            avg_power = np.mean(np.var(window_data, axis=1))
            avg_powers.append(avg_power)
        
        axes[idx].plot(time_points, avg_powers, linewidth=2, color='darkblue', alpha=0.8)
        axes[idx].fill_between(time_points, avg_powers, alpha=0.3, color='lightblue')
        axes[idx].set_xlabel('Time (s)', fontsize=11)
        axes[idx].set_ylabel('Average Power (µV²)', fontsize=11)
        axes[idx].set_title(f'Epoch {epoch_idx} - Label: {label}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Power Evolution Within Epochs', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== VISUALIZATION 4: Topographic Power Maps ====================

def plot_topographic_power_maps(df: pd.DataFrame, epoch_idx: int, 
                                montage_path: Path, output_path: Optional[Path] = None):
    """
    Create topographic maps showing spatial distribution of power for different bands.
    """
    # Filter data for specific epoch
    df_epoch = df[df['epoch_idx'] == epoch_idx]
    
    if df_epoch.empty:
        print(f"No data for epoch {epoch_idx}")
        return
    
    # Load montage
    try:
        montage = mne.channels.read_custom_montage(str(montage_path))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Could not load montage for topographic plotting: {e}")
        return
    
    # Create MNE info
    ch_names = df_epoch['channel'].tolist()
    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
    info.set_montage(montage, on_missing='ignore')
    
    # Create figure with subplots for each band
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_labels = ['Delta (1-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 
                   'Beta (13-30 Hz)', 'Gamma (30-45 Hz)']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    im = None  # Store the last image for colorbar
    
    for i, (band, label) in enumerate(zip(bands, band_labels)):
        power_col = f'spec_{band}'
        if power_col in df_epoch.columns:
            data = df_epoch.set_index('channel')[power_col].reindex(ch_names).values
            
            # Create evoked object for plotting
            evoked = mne.EvokedArray(data[:, np.newaxis], info, tmin=0)
            
            # Plot topomap
            im_temp, _ = mne.viz.plot_topomap(data, info, axes=axes[i], show=False,
                                        cmap='RdYlBu_r', contours=6)
            axes[i].set_title(label, fontsize=12, fontweight='bold')
            
            # Store the image object for colorbar
            if im is None:
                im = im_temp
    
    # Remove extra subplot and add colorbar
    fig.delaxes(axes[5])
    if im is not None:
        plt.colorbar(im, ax=axes.tolist(), label='Power (µV²)', fraction=0.046, pad=0.04)
    
    label_name = df_epoch['label_name'].iloc[0]
    plt.suptitle(f'Topographic Power Distribution - Epoch {epoch_idx} ({label_name})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== VISUALIZATION 5: Feature Distributions ====================

def plot_feature_distributions(df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Show distributions of key features across all electrodes and epochs.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    features = [
        ('spec_total_power', 'Total Power (µV²)', 'skyblue'),
        ('spec_alpha_rel', 'Alpha Relative Power', 'green'),
        ('spec_beta_rel', 'Beta Relative Power', 'orange'),
        ('temp_rms', 'RMS (µV)', 'purple'),
        ('temp_hjorth_mobility', 'Hjorth Mobility', 'red'),
        ('spec_dominant_freq', 'Dominant Frequency (Hz)', 'brown'),
        ('func_mean_corr', 'Mean Correlation', 'teal'),
        ('func_mean_plv', 'Mean PLV', 'magenta'),
        ('spec_entropy', 'Spectral Entropy', 'navy')
    ]
    
    for i, (feat, label, color) in enumerate(features):
        if feat in df.columns:
            axes[i].hist(df[feat].dropna(), bins=50, color=color, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(label, fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].set_title(f'Distribution of {label}', fontsize=11, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = df[feat].mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.2f}')
            axes[i].legend(fontsize=8)
    
    plt.suptitle('Feature Distributions Across All Epochs and Channels', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== VISUALIZATION 6: Feature Correlation Matrix ====================

def plot_feature_correlation_matrix(df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Show correlation between different feature types.
    """
    # Select key features
    feature_cols = [col for col in df.columns if any(
        col.startswith(prefix) for prefix in ['spec_', 'temp_', 'func_']
    )]
    
    # Sample features for readability
    selected_features = [
        'spec_total_power', 'spec_alpha', 'spec_beta', 'spec_gamma',
        'spec_alpha_rel', 'spec_beta_rel',
        'spec_dominant_freq', 'spec_entropy',
        'temp_mean', 'temp_std', 'temp_rms',
        'temp_hjorth_activity', 'temp_hjorth_mobility',
        'func_mean_corr', 'func_max_corr', 'func_mean_plv'
    ]
    
    selected_features = [f for f in selected_features if f in df.columns]
    
    # Compute correlation matrix
    corr_matrix = df[selected_features].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(len(selected_features)))
    ax.set_yticks(range(len(selected_features)))
    ax.set_xticklabels(selected_features, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(selected_features, fontsize=9)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    
    # Add correlation values
    for i in range(len(selected_features)):
        for j in range(len(selected_features)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== MAIN VISUALIZATION SCRIPT ====================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    features_csv = project_root / "data" / "processed" / "comprehensive_features_subjects.csv"
    output_dir = project_root / "figures" / "feature_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading features...")
    df = load_comprehensive_features(features_csv)
    print(f"Loaded {len(df)} feature rows")
    
    # Get subject and session info
    subject_id = df['subject_id'].iloc[0]
    
    # 1. Single electrode power over time
    print("\n1. Creating electrode power variation plot...")
    electrode = 'Cz'  # Central electrode
    if electrode in df['channel'].values:
        plot_electrode_power_across_epochs(
            df, electrode, subject_id,
            output_path=output_dir / f"power_variation_{electrode}.png"
        )
    
    # 2. Top electrodes per epoch
    print("\n2. Creating top electrodes heatmap...")
    plot_top_electrodes_per_epoch(
        df, subject_id, top_n=5,
        output_path=output_dir / "top_electrodes_per_epoch.png"
    )
    
    # 3. Power evolution within epochs
    print("\n3. Creating power evolution plots...")
    h5_file = project_root / "data" / "processed" / f"{subject_id}_S001.h5"
    eloc_path = project_root / "scripts" / "data_processing" / "Preprocessing" / "ebneuro.eloc"
    
    if h5_file.exists() and eloc_path.exists():
        sample_epochs = [0, 10, 20, 30, 40] if len(df['epoch_idx'].unique()) > 40 else [0, 5, 10]
        plot_power_evolution_within_epochs(
            h5_file, eloc_path, sample_epochs,
            output_path=output_dir / "power_evolution_within_epochs.png"
        )
    
    # 4. Topographic maps
    print("\n4. Creating topographic power maps...")
    # Try both .locs and .eloc file extensions
    montage_path_locs = project_root / "scripts" / "data_processing" / "Preprocessing" / "ebneuro.locs"
    montage_path_eloc = project_root / "scripts" / "data_processing" / "Preprocessing" / "ebneuro.eloc"
    
    montage_path = None
    if montage_path_locs.exists():
        montage_path = montage_path_locs
    elif montage_path_eloc.exists():
        montage_path = montage_path_eloc
    
    if montage_path:
        plot_topographic_power_maps(
            df, epoch_idx=0, montage_path=montage_path,
            output_path=output_dir / "topographic_power_maps_epoch0.png"
        )
    else:
        print("  Skipping topographic maps: montage file not found")
    
    # 5. Feature distributions
    print("\n5. Creating feature distribution plots...")
    plot_feature_distributions(
        df, output_path=output_dir / "feature_distributions.png"
    )
    
    # 6. Feature correlation matrix
    print("\n6. Creating feature correlation matrix...")
    plot_feature_correlation_matrix(
        df, output_path=output_dir / "feature_correlation_matrix.png"
    )
    
    print(f"\n✓ All visualizations saved to {output_dir}")
