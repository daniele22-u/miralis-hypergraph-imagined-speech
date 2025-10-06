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
- Spectral power in δ, θ, α, β, γ bands  
- Functional connectivity metrics (correlation, PLV, coherence)  
- Graph construction based on electrode distances and connectivity  

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
imagined-speech-gcn/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ src/
│  ├─ preprocessing/
│  ├─ features/
│  ├─ graphs/
│  ├─ models/
│  ├─ utils/
│  └─ visualization/
├─ notebooks/
├─ configs/
├─ results/
├─ figures/
├─ environment.yml
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## 🧰 Tech Stack
####
---

## 🚀 How to Run
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
