# H5 Data Loading Notebook Guide

## Overview

The `01_load_h5_data.ipynb` notebook provides a comprehensive framework for loading and organizing H5 data files from the imagined speech EEG dataset.

## Purpose

This notebook was created to:
- Load raw H5 files from a local directory structure
- Parse subject and trial IDs from filenames
- Build structured DataFrames for downstream analysis
- Validate data consistency and completeness
- Provide visualization of dataset structure

## File Naming Convention

The notebook expects H5 files to follow this naming pattern:

```
{subject_id}_{trial_id}.h5
```

**Examples:**
- `00_00.h5` → Subject 00, Trial 00
- `00_01.h5` → Subject 00, Trial 01
- `01_00.h5` → Subject 01, Trial 00
- etc.

Where:
- **Subject ID**: Two-digit identifier (00, 01, 02, ...)
- **Trial ID**: Two-digit identifier (00, 01, 02, ..., 05)

## Quick Start

### 1. Configure Data Path

Open the notebook and set the `BASE_PATH` variable in the second code cell:

```python
BASE_PATH = "/path/to/your/data/directory"
```

**Example:**
```python
BASE_PATH = "/Users/danieleuras/Library/CloudStorage/OneDrive-PolitecnicodiMilano/File di Francesco Iacomi - h5/data"
```

### 2. Run All Cells

Execute all cells in order (Cell → Run All in Jupyter).

### 3. Review Results

The notebook will:
- Discover all matching H5 files
- Load metadata from each file
- Create a structured DataFrame
- Perform sanity checks
- Generate visualizations
- Save metadata to CSV

## Expected H5 File Structure

The notebook expects H5 files to contain these keys:

- **`data`**: 3D array with shape `(n_epochs, n_channels, n_samples)`
- **`labels`**: Array of label strings or bytes (length = n_epochs)
- **`subject`** (optional): Subject identifier

**Example structure:**
```
file.h5
├── data        [100 epochs × 60 channels × 640 samples]
├── labels      [100 labels]
└── subject     "01"
```

## Output

### 1. Dataset DataFrame (`dataset_df`)

A pandas DataFrame containing:

| Column | Description |
|--------|-------------|
| `subject_id` | Subject identifier from filename |
| `trial_id` | Trial identifier from filename |
| `file_path` | Absolute path to H5 file |
| `file_name` | Filename only |
| `file_size_mb` | File size in megabytes |
| `n_epochs` | Number of epochs in file |
| `n_channels` | Number of EEG channels |
| `n_samples` | Number of time samples per epoch |
| `data_shape` | Shape tuple as string |
| `sample_label` | Example label from file |
| `available_keys` | H5 keys found in file |

### 2. Sanity Checks

The notebook performs these validations:

- ✓ Number of subjects detected
- ✓ Trials per subject (min/max/mean)
- ✓ Data shape consistency across files
- ✓ Missing trials detection
- ✓ Total dataset size
- ✓ Total number of epochs

### 3. Visualizations

Four plots are generated:

1. **Trials per Subject**: Bar chart showing trial count per subject
2. **File Size Distribution**: Histogram of file sizes
3. **Epochs Distribution**: Histogram of epoch counts
4. **Trial Availability Heatmap**: Shows which trials exist for each subject

### 4. Saved Metadata

The DataFrame is saved to:
```
data/interim/h5_dataset_metadata.csv
```

## Integration with Existing Code

This notebook follows repository conventions:

### Using Existing Utilities

```python
# Load channel names (from utils.py pattern)
from pathlib import Path
import pandas as pd

def load_channel_names_from_eloc(eloc_path: Path):
    df = pd.read_csv(eloc_path, sep=r"\s+", header=None, engine="python")
    names = df.iloc[:, -1].astype(str).tolist()
    return names

# Path to electrode file
eloc_path = project_root / "scripts" / "data_processing" / "Preprocessing" / "ebneuro.eloc"
channel_names = load_channel_names_from_eloc(eloc_path)
```

### Loading Data for Processing

```python
import h5py

# Example: Load data from a specific file
file_path = dataset_df.iloc[0]['file_path']

with h5py.File(file_path, 'r') as f:
    data = f['data'][:]      # Shape: (epochs, channels, samples)
    labels = f['labels'][:]  # Shape: (epochs,)
```

### Building MNE Epochs (Compatible with existing code)

```python
import mne
import numpy as np

# Load data
with h5py.File(file_path, 'r') as f:
    data = f['data'][:]
    labels = f['labels'][:]
    
# Create MNE info object
fs = 256  # Sampling frequency
info = mne.create_info(
    ch_names=channel_names or [f"EEG{i+1}" for i in range(data.shape[1])],
    sfreq=fs,
    ch_types='eeg'
)

# Create epochs
epochs = mne.EpochsArray(data, info)

# Decode labels
decoded_labels = [
    l.decode('utf-8') if isinstance(l, bytes) else str(l) 
    for l in labels
]

# Add metadata
epochs.metadata = pd.DataFrame({
    'label_name': decoded_labels,
    'epoch_idx': np.arange(len(decoded_labels))
})
```

## Example Workflows

### Workflow 1: Load All Data Into Memory

```python
# For small datasets
all_data = {}

for _, row in dataset_df.iterrows():
    subject_id = row['subject_id']
    trial_id = row['trial_id']
    
    with h5py.File(row['file_path'], 'r') as f:
        all_data[f"{subject_id}_{trial_id}"] = {
            'data': f['data'][:],
            'labels': [l.decode('utf-8') if isinstance(l, bytes) else str(l) 
                      for l in f['labels'][:]],
            'shape': f['data'].shape
        }
```

### Workflow 2: Process Files One at a Time

```python
# For large datasets - process without loading all into memory
for _, row in dataset_df.iterrows():
    print(f"Processing {row['subject_id']}_{row['trial_id']}")
    
    with h5py.File(row['file_path'], 'r') as f:
        # Process data in chunks or extract features
        data = f['data']  # Reference, not loaded yet
        
        # Process first epoch
        epoch_0 = data[0, :, :]
        
        # Your processing here...
```

### Workflow 3: Filter and Select Specific Data

```python
# Select only specific subjects
subjects_of_interest = ['00', '01', '05']
filtered_df = dataset_df[dataset_df['subject_id'].isin(subjects_of_interest)]

# Select only first 3 trials per subject
first_trials = dataset_df[dataset_df['trial_id'].astype(int) < 3]

# Get all files for one subject
subject_01_df = dataset_df[dataset_df['subject_id'] == '01']
```

## Troubleshooting

### Issue: "Data directory not found"

**Solution:** Check that `BASE_PATH` is set correctly and the directory exists.

```python
from pathlib import Path
data_dir = Path(BASE_PATH)
print(f"Directory exists: {data_dir.exists()}")
print(f"Is directory: {data_dir.is_dir()}")
```

### Issue: "No H5 files found"

**Solution:** Verify file naming matches the expected pattern:

```python
# List all H5 files
for f in sorted(Path(BASE_PATH).glob("*.h5")):
    print(f.name)
```

### Issue: "KeyError: 'data'" when loading

**Solution:** Check what keys are in the H5 file:

```python
import h5py
with h5py.File(file_path, 'r') as f:
    print("Available keys:", list(f.keys()))
```

### Issue: Unicode decode errors with labels

**Solution:** The notebook handles multiple encodings automatically. If issues persist:

```python
# Manual decoding
label_raw = f['labels'][0]
if isinstance(label_raw, bytes):
    for encoding in ('utf-8', 'latin-1', 'cp1252'):
        try:
            label = label_raw.decode(encoding)
            break
        except:
            continue
```

## Dependencies

The notebook requires these packages (all included in `environment.yml`):

- `h5py`: HDF5 file handling
- `pandas`: DataFrame operations
- `numpy`: Numerical operations
- `matplotlib`: Basic plotting
- `seaborn`: Statistical visualizations

## Compatibility

This notebook is designed to work seamlessly with:

- **`dataframe.ipynb`**: Uses same DataFrame structure
- **`feature_extraction_tutorial.ipynb`**: Compatible MNE epochs format
- **`scripts/features/bandpowers.py`**: Same H5 loading approach
- **`scripts/graphs/eeg_viewer.py`**: Same data structure expectations

## Advanced Usage

### Custom File Pattern

If your files use a different naming convention, modify the regex pattern:

```python
# In discover_h5_files function
# Default: r"^(\d{2})_(\d{2})\.h5$"

# Example: subject_trial.h5 format
pattern = r"^([a-z]+)_([0-9]+)\.h5$"
```

### Additional Metadata

Add custom metadata extraction:

```python
# In load_h5_file_info function
with h5py.File(file_path, "r") as f:
    # Add custom field
    if "sampling_rate" in f:
        metadata["fs"] = f["sampling_rate"][()]
    
    if "experiment_date" in f:
        metadata["date"] = f["experiment_date"][()]
```

## Notes

- **Memory efficient**: Metadata is loaded without reading full data arrays
- **Error handling**: Failed files are reported but don't stop processing
- **Flexible**: Works with different H5 structures (checks keys dynamically)
- **Documented**: Every function includes docstrings and type hints

## Contact

For issues or questions about this notebook:
- Check existing notebooks in `notebooks/` directory
- Review code in `scripts/` for similar patterns
- Refer to `README.md` for project overview

---

**Last Updated**: 2025-11-22  
**Version**: 1.0  
**Author**: GitHub Copilot Agent
