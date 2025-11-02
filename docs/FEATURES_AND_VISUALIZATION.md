# EEG Comprehensive Feature Extraction and Visualization

This module provides tools for extracting comprehensive features from EEG data and creating rich visualizations to understand signal characteristics across electrodes and epochs.

## Overview

The system extracts three main types of features:
1. **Temporal Features**: Statistical and time-domain characteristics
2. **Spectral Features**: Frequency-domain and power spectral density features
3. **Functional Features**: Connectivity and synchronization metrics

## Scripts

### 1. Feature Extraction: `scripts/features/comprehensive_features.py`

Extracts comprehensive features from EEG epochs stored in HDF5 files.

#### Temporal Features (13 features)
- **Basic Statistics**: mean, std, variance, min, max, range, peak-to-peak
- **Higher Order Moments**: skewness, kurtosis
- **Signal Characteristics**: RMS (Root Mean Square), zero crossing rate
- **Hjorth Parameters**: activity, mobility, complexity

#### Spectral Features (22 features)
- **Band Powers** (absolute and relative):
  - Delta (1-4 Hz): slow-wave activity
  - Theta (4-8 Hz): memory processes, drowsiness
  - Alpha (8-13 Hz): relaxed wakefulness
  - Beta (13-30 Hz): active cognitive processing
  - Gamma (30-45 Hz): high-level cognition
- **Band Power Ratios**: alpha/beta, theta/alpha, theta/beta
- **Spectral Characteristics**: 
  - Spectral edge frequency (95% power)
  - Spectral entropy
  - Dominant frequency and power
  - Mean and median frequency

#### Functional Features (6 features)
- **Correlation-based Connectivity**:
  - Mean correlation with other channels
  - Max correlation
  - Standard deviation of correlations
  - Number of strong connections (r > 0.7)
- **Phase Locking Value (PLV)**:
  - Mean PLV across channels
  - Max PLV

#### Usage

```python
python scripts/features/comprehensive_features.py
```

**Input**: 
- `data/interim/eeg_metadata.csv` - metadata file pointing to H5 files
- HDF5 files with EEG epochs
- `scripts/data_processing/Preprocessing/ebneuro.eloc` - electrode montage

**Output**:
- `data/interim/comprehensive_features.csv` - unified dataframe with all features

**Output Format**:
Each row represents one (epoch, channel) combination with columns:
- `subject_id`, `session_id`, `epoch_idx`, `channel`, `label_name`
- All temporal features (prefix: `temp_`)
- All spectral features (prefix: `spec_`)
- All functional features (prefix: `func_`)

### 2. Visualization Suite: `scripts/graphs/feature_visualizations.py`

Creates comprehensive visualizations for understanding EEG features.

#### Visualization 1: Single Electrode Power Variation
**Function**: `plot_electrode_power_across_epochs()`

Shows how power varies across all epochs for a specific electrode:
- Total power timeline
- Stacked area chart of band powers
- Relative band power evolution
- Temporal features (RMS, Hjorth mobility)

**Example**:
```python
plot_electrode_power_across_epochs(df, electrode='Cz', subject_id='11')
```

#### Visualization 2: Top Electrodes Per Epoch
**Function**: `plot_top_electrodes_per_epoch()`

Identifies which electrodes show highest activity:
- Heatmap of power across all epochs and channels
- Bar chart of most frequently high-power channels

**Example**:
```python
plot_top_electrodes_per_epoch(df, subject_id='11', top_n=5)
```

#### Visualization 3: Power Evolution Within Epochs
**Function**: `plot_power_evolution_within_epochs()`

Shows temporal dynamics within individual epochs using sliding windows:
- Power variation during the 1.5s epoch
- Multiple epochs displayed for comparison

**Example**:
```python
plot_power_evolution_within_epochs(h5_path, eloc_path, epoch_indices=[0, 10, 20])
```

#### Visualization 4: Topographic Power Maps
**Function**: `plot_topographic_power_maps()`

Spatial distribution of power across the scalp:
- Separate topomaps for each frequency band
- Shows regional activation patterns

**Example**:
```python
plot_topographic_power_maps(df, epoch_idx=0, montage_path=montage_path)
```

#### Visualization 5: Feature Distributions
**Function**: `plot_feature_distributions()`

Statistical distributions of key features:
- Histograms for 9 key features
- Mean values indicated
- Helps understand feature ranges and outliers

#### Visualization 6: Feature Correlation Matrix
**Function**: `plot_feature_correlation_matrix()`

Correlation between different feature types:
- Heatmap showing inter-feature relationships
- Useful for feature selection and understanding dependencies

#### Running All Visualizations

```python
python scripts/graphs/feature_visualizations.py
```

**Output**: All visualizations saved to `figures/feature_visualizations/`

## Data Flow

```
Raw EEG Data (HDF5)
        ↓
[comprehensive_features.py]
        ↓
Comprehensive Features DataFrame
        ↓
[feature_visualizations.py]
        ↓
Multiple Visualization Outputs
```

## Feature Statistics

For a typical dataset with:
- 1 subject
- 5 sessions
- ~220 epochs per session
- 61 channels per epoch

**Total rows in features dataframe**: ~67,000 rows
**Total features per row**: ~45 features
- 5 metadata columns
- 13 temporal features
- 22 spectral features
- 6 functional features

## Physiological Interpretation

### Band Powers
- **High Delta/Theta**: Low vigilance, drowsy states
- **High Alpha**: Relaxed, eyes-closed resting
- **High Beta**: Active engagement, motor planning
- **High Gamma**: Fast cognitive processing

### Ratios
- **Alpha/Beta Ratio**: Higher in calm/relaxed conditions
- **Theta/Alpha Ratio**: Higher in fatigue or stress

### Hjorth Parameters
- **Activity**: Signal variance (overall power)
- **Mobility**: Mean frequency (how fast the signal changes)
- **Complexity**: Deviation from a pure sine wave

### Functional Features
- **High Correlation**: Strong functional connectivity between regions
- **High PLV**: Synchronized oscillations between channels

## Customization

### Adding New Features

To add custom features, modify `comprehensive_features.py`:

```python
def extract_custom_features(signal_1d: np.ndarray) -> Dict[str, float]:
    feats = {}
    # Add your feature computations
    feats['custom_feature_name'] = compute_feature(signal_1d)
    return feats

# Add to main extraction function:
custom_feats = extract_custom_features(signal_1d)
row.update(custom_feats)
```

### Adding New Visualizations

Create new visualization functions in `feature_visualizations.py`:

```python
def plot_custom_visualization(df: pd.DataFrame, output_path: Optional[Path] = None):
    # Your visualization code
    pass
```

## Dependencies

- `numpy`, `pandas`: Data manipulation
- `scipy`: Signal processing
- `matplotlib`, `seaborn`: Visualization
- `mne`: EEG-specific processing and topographic plotting
- `h5py`: Reading HDF5 files

## Notes

- **Processing Time**: Feature extraction can take several minutes depending on dataset size
- **Memory**: Functional features require loading full epoch data (all channels)
- **File Sizes**: The comprehensive features CSV can be large (>100 MB for full datasets)
- **Topographic Plots**: Require proper electrode montage file (`.locs` or `.eloc`)

## Future Enhancements

Potential additions:
1. **Time-Frequency Features**: Wavelet transforms, short-time Fourier transform
2. **Nonlinear Features**: Sample entropy, approximate entropy, fractal dimension
3. **Connectivity Networks**: Graph-theoretic measures (clustering coefficient, path length)
4. **Machine Learning Features**: Autoencoder embeddings, learned representations
5. **Interactive Visualizations**: Plotly/Dash dashboards for real-time exploration
6. **Animated Visualizations**: Power evolution videos across epochs

## References

- Hjorth, B. (1970). "EEG analysis based on time domain properties"
- Welch, P. (1967). "The use of fast Fourier transform for the estimation of power spectra"
- Lachaux, J.P. et al. (1999). "Measuring phase synchrony in brain signals"
