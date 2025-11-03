# 🧠 Imagined Speech EEG Decoding using Graph Convolutional Networks  
**Building a Neural Semantic Dictionary**

---

## 📄 Overview
This repository contains the code, data structure, and documentation for my Master's Thesis project at the *Politecnico di Milano*.

The goal is to explore **imagined speech decoding** from EEG signals using **Graph Convolutional Networks (GCN)** and **Graph Signal Processing (GSP)** techniques, with the final aim of building a **neural semantic dictionary** — a mapping between EEG patterns and conceptual categories (e.g., *food*, *objects*, *emotions*).

---

## 🎯 Objectives
1. **Analyze EEG patterns** associated with imagined speech tasks.  
2. **Construct EEG-based graph representations** capturing spatial and functional relationships between electrodes.  
3. **Train Graph Convolutional / Spatio-Temporal Graph Neural Networks** for semantic category classification.  
4. **Build an interpretable neural dictionary** linking stable EEG features to semantic concepts.

---

## 🧩 Methodology

### 1. Preprocessing
- Band-pass & notch filtering  
- Artifact removal (ICA / ASR)  
- Segmentation into epochs  
- Normalization and channel alignment  

### 2. Feature Extraction
- **Temporal Features**: Statistical measures, Hjorth parameters, zero-crossing rate, RMS
- **Spectral Features**: Power in δ, θ, α, β, γ bands (absolute and relative), spectral entropy, dominant frequency
- **Functional Features**: Connectivity metrics (correlation, PLV), inter-channel synchronization
- Graph construction based on electrode distances and connectivity  

📖 **See [Feature Extraction Guide](docs/FEATURES_AND_VISUALIZATION.md)** for detailed documentation

### 3. Graph Modeling
- Graph Signal Processing (GSP) for structural and functional representation  
- Graph Learning (GL) for adaptive connectivity inference  
- Graph Convolutional Networks (GCN / ST-GCN) for EEG decoding  

### 4. Semantic Classification
- Multi-class classification across semantic categories  
- Subject-independent validation  
- Visualization of learned brain topologies  

### 5. Neural Dictionary Construction
- Extraction of prototypical EEG patterns per category  
- Dimensionality reduction (t-SNE / UMAP)  
- Creation of a concept–pattern mapping  

---

## 🧠 Scientific Background

This work builds upon two key studies:

| Reference | Method | Contribution |
|------------|---------|--------------|
| **Einizade et al. (2022)** – *Neural decoding of imagined speech using Graph Signal Processing and Graph Learning* | GraphIS model (GSP + GL) | Enhanced decoding accuracy and right-hemisphere activation patterns |
| **Li et al. (2025)** – *EEG-based speech imagery decoding by dynamic hypergraph learning* | DHSLP / DHSLF | Hypergraph models capturing higher-order EEG relations (up to 78% accuracy) |

---
### 📦 Repository Structure
```text
miralis-hypergraph-imagined-speech/
├─ data/
│  ├─ raw/
│  ├─ interim/              # Metadata and extracted features
│  └─ processed/            # Preprocessed EEG epochs (.h5)
├─ scripts/
│  ├─ data_processing/      # Preprocessing pipeline
│  ├─ features/             # Feature extraction
│  │  ├─ bandpowers.py      # Band power extraction
│  │  └─ comprehensive_features.py  # Temporal, spectral, functional features
│  ├─ graphs/               # Visualization tools
│  │  ├─ eeg_viewer.py      # Interactive EEG viewer
│  │  └─ feature_visualizations.py  # Feature visualization suite
│  ├─ models/
│  ├─ predict/
│  └─ training/
├─ notebooks/
│  ├─ dataframe.ipynb       # Dataset exploration
│  └─ feature_extraction_tutorial.ipynb  # Feature extraction tutorial
├─ docs/
│  └─ FEATURES_AND_VISUALIZATION.md  # Feature extraction documentation
├─ figures/                 # Generated visualizations
├─ environment.yml          # Conda environment
├─ LICENSE
└─ README.md
```

## 🧰 Tech Stack
####
---

## 🚀 How to Run
To execute the preprocessing pipeline, make sure you have the following installed:
MATLAB (recommended version: 2024b or later)
Python–MATLAB Engine (official connector between Python and MATLAB)
👉 Installation guide
EEGLAB, toolbox for EEG analysis
👉 https://sccn.ucsd.edu/eeglab/download.php
After running the preprocessing function kindly shared by [Miralis] (MIT License), located at:
<project_root>/scripts/data_processing/Preprocessing/
the converted data will be available in:

<project_root>/data/processed/
These processed data can then be visualized and explored in the notebook:
notebooks/dataframe.ipynb

### Feature Extraction and Visualization
After preprocessing, extract comprehensive EEG features:

```bash
# Extract temporal, spectral, and functional features
python scripts/features/comprehensive_features.py

# Generate visualizations
python scripts/graphs/feature_visualizations.py
```

**Features extracted** (per electrode, per epoch):
- **13 Temporal features**: statistics, Hjorth parameters, RMS, zero-crossing rate
- **22 Spectral features**: band powers (δ, θ, α, β, γ), spectral entropy, dominant frequency
- **6 Functional features**: correlation-based connectivity, phase locking value (PLV)

**Visualizations created**:
- Power variation across epochs for individual electrodes
- Heatmap of high-power electrodes per epoch
- Temporal power evolution within epochs
- Topographic power distribution maps
- Feature distributions and correlation matrices

📖 See the [Feature Extraction Tutorial](notebooks/feature_extraction_tutorial.ipynb) for examples.

####
---

## 📊 Expected Results
- Improved imagined speech decoding accuracy (subject-independent).  
- Visualization of learned EEG connectivity graphs.  
- Stable EEG prototypes per semantic category.  
- Foundational “Neural Dictionary” for concept decoding.

---

## 👤 Author
**Daniele Uras**  
Master’s Thesis – *Politecnico di Milano*  
Supervisor: _[To be defined]_  

📧 daniele.uras@mail.polimi.it
🔗 [LinkedIn](www.linkedin.com/in/daniele-uras1) • [GitHub](https://github.com/daniele22-u)

---

## 📚 Citation
###
---
## 🌟 Acknowledgements
- Politecnico di Milano – DEIB Department  
