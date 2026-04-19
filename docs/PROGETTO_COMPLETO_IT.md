# Progetto Completo: Decodifica dell'Imagined Speech da EEG con Graph Neural Networks

> Tesi Magistrale - Politecnico di Milano, DEIB
> Autore: Daniele Uras
> Ultimo aggiornamento: 19 aprile 2026

---

## 1. Obiettivo del Progetto

Costruire un **dizionario neurale semantico** che mappi pattern EEG a categorie concettuali (cibo, oggetti, emozioni, ecc.) attraverso la decodifica dell'*imagined speech* (parlato immaginato) utilizzando Graph Convolutional Networks (GCN) e Graph Signal Processing (GSP).

Il progetto si basa sulla letteratura scientifica recente:
- **Einizade et al. (2022)** - Modello GraphIS (GSP + Graph Learning)
- **Li et al. (2025)** - DHSLP/DHSLF (Hypergraph Learning, fino a 78% di accuratezza)

**Task di classificazione**: 110 parole immaginarie, multi-classe, con validazione subject-specific e subject-independent.

---

## 2. Pipeline dei Dati

### 2.1 Acquisizione e Preprocessing

- **Dati grezzi**: file XDF (BioSignal XML Data) da acquisizioni EEG
- **Preprocessing** (MATLAB/EEGLAB):
  - Filtraggio passa-banda e notch
  - Rimozione artefatti (ICA / ASR)
  - Segmentazione in epoche (~1.5s ciascuna a 256 Hz)
  - Normalizzazione e allineamento canali
- **Output**: file HDF5 in `data/processed/` con struttura `(n_epochs, n_channels, n_samples)`
- **Canali registrati nell'H5**: 61 su 63 presenti nel casco (Pz e POz non acquisiti dal sistema)
- **Canali validi nei tensori**: 59 (A1 e A2 esclusi — elettrodi di riferimento senza posizione spaziale)
- **Sessioni**: 5 per soggetto, ~220 epoche per sessione

### 2.2 Estrazione Feature (40 feature per elettrodo)

Script principale: `scripts/features/comprehensive_features.py`

#### Feature Temporali (13)
| Feature | Descrizione |
|---------|-------------|
| mean, std, variance | Statistiche di base del segnale |
| min, max, range, peak-to-peak | Ampiezza del segnale |
| skewness, kurtosis | Momenti di ordine superiore |
| RMS | Root Mean Square (potenza media) |
| Zero Crossing Rate | Frequenza passaggi per zero |
| Hjorth Activity | Varianza del segnale (potenza) |
| Hjorth Mobility | Frequenza media (mobilita) |
| Hjorth Complexity | Deviazione dalla forma sinusoidale |

#### Feature Spettrali (22)
| Feature | Descrizione |
|---------|-------------|
| Delta power (abs/rel) | Potenza banda 1-4 Hz (attivita onde lente) |
| Theta power (abs/rel) | Potenza banda 4-8 Hz (memoria, sonnolenza) |
| Alpha power (abs/rel) | Potenza banda 8-13 Hz (veglia rilassata) |
| Beta power (abs/rel) | Potenza banda 13-30 Hz (elaborazione cognitiva) |
| Gamma power (abs/rel) | Potenza banda 30-45 Hz (cognizione di alto livello) |
| Alpha/Beta, Theta/Alpha, Theta/Beta | Rapporti tra bande |
| Spectral Edge Frequency | Frequenza al 95% della potenza |
| Spectral Entropy | Complessita dello spettro |
| Dominant Frequency/Power | Frequenza e potenza dominanti |
| Mean/Median Frequency | Frequenza media e mediana |

#### Feature Funzionali (6)
| Feature | Descrizione |
|---------|-------------|
| Mean/Max/Std Correlation | Connettivita basata su correlazione |
| Strong Connections (r > 0.7) | Numero connessioni forti |
| Mean/Max PLV | Phase Locking Value (sincronizzazione di fase) |

**Output**: `data/interim/comprehensive_features.csv` (~67.000 righe per soggetto)

### 2.3 Costruzione Tensori

Due rappresentazioni sono state create (notebook `EEG_02_tensors_and_graph.ipynb`):

1. **Tensore Aggregato**: `(n_trials, 59, 40)` - un singolo valore per trial, canale e feature
2. **Tensore Time-Resolved**: `(n_trials, 5, 59, 40)` - l'epoca e divisa in 5 finestre temporali, le feature sono calcolate per ciascuna finestra

Salvati come file `.pt` (PyTorch) per ogni soggetto.

### 2.4 Costruzione Grafi

Tre approcci implementati:

**A) Grafo Spaziale Statico**
- Nodi = 59 elettrodi EEG
- Archi = k-nearest neighbors basati sulle posizioni degli elettrodi sullo scalpo
- Topologia fissa per tutti i trial
- Distanze calcolate: euclidee e geodesiche (`scripts/features/distances.py`)

**B) Grafo Feature-Similarity (adattivo per trial)**
- Nodi = 59 elettrodi
- Archi = cosine similarity tra vettori feature dei nodi
- Top-k connessioni piu simili
- La topologia cambia ad ogni trial

**C) Grafo Spazio-Temporale** (preparato)
- Ogni trial = sequenza di 5 grafi (uno per finestra temporale)
- Encoding con GCN, aggregazione temporale via mean pooling o GRU

---

## 3. Analisi Esplorativa

### 3.1 Visualizzazioni

**Suite di 6 visualizzazioni** (`scripts/graphs/feature_visualizations.py`):
1. **Potenza singolo elettrodo**: Timeline della potenza totale e per banda
2. **Top elettrodi per epoca**: Heatmap potenza e bar chart dei canali piu attivi
3. **Evoluzione potenza intra-epoca**: Dinamiche temporali con finestre scorrevoli
4. **Mappe topografiche**: Distribuzione spaziale per ogni banda di frequenza
5. **Distribuzioni feature**: Istogrammi delle 9 feature principali
6. **Matrice di correlazione**: Heatmap delle relazioni inter-feature

**Topomaps interattive** (notebook `EEG_03_visualization_topomaps.ipynb`):
- Slider temporale sulle 5 finestre
- Confronto tra soggetti con mappe differenza/similarita

**EEG Viewer interattivo** (`scripts/graphs/eeg_viewer.py`):
- Applicazione GUI PySide6
- Selezione soggetto, epoche, canali
- Visualizzazione tracce EEG sovrapposti o impilati

### 3.2 Analisi Embedding

Effettuata in `EEG_02_tensors_and_graph.ipynb` con t-SNE, UMAP e PCA:

- **Forte separazione per soggetto**: i trial si raggruppano fortemente per soggetto
- **Nessun clustering per parola**: le singole parole NON formano cluster compatti
- **Geometria del manifold**: strutture curve, ad anello o a ferro di cavallo, suggestive di un manifold neurale continuo
- **Effetto sessione**: debole, non spiega la struttura principale

### 3.3 Export TensorFlow Projector

Tre modalita di export (notebook `baseline_test_with_3_projector_exports.ipynb`):
1. **Aggregato**: `(n_trials, 59x40)` appiattito
2. **Time-concatenato**: `(n_trials, 5x59x40)` appiattito
3. **Window-level**: `(n_trials x 5, 59x40)` ogni finestra come osservazione separata

### 3.4 Analisi Statistica delle Feature

Notebook `EEG_00_feature_significance.ipynb`:

- **Metodo**: Test di Kruskal-Wallis con correzione Benjamini-Hochberg (FDR)
- **Misura effetto**: epsilon-squared
- **80 descrittori scalari** (media e std delle 40 feature)

**Risultati chiave**:
| Fattore | Effetto | Interpretazione |
|---------|---------|-----------------|
| **Parola** (label_id) | Trascurabile | Le feature aggregate non discriminano tra parole |
| **Soggetto** (subject_id) | MOLTO FORTE | Le feature dipendono fortemente dal soggetto |
| **Sessione** (session_id) | Debole | Non e la fonte principale di variabilita |

---

## 4. Esperimenti Baseline

### 4.1 Modelli Vettoriali

Notebook `baseline_test_logreg_mlp.ipynb`:

| Modello | Rappresentazione | Dimensione | Subject-Specific | Subject-Independent |
|---------|-----------------|------------|------------------|---------------------|
| Logistic Regression | Aggregato (media canali) | (40,) | ~Chance | ~Chance |
| MLP | Aggregato (media canali) | (40,) | ~Chance | ~Chance |
| MLP | Full flattened | (2.360,) | ~Chance | ~Chance |
| MLP | Time-resolved | (11.800,) | ~Chance | ~Chance |

**Conclusione**: la media dei canali perde informazione spaziale discriminativa. Anche con la rappresentazione completa, i modelli vettoriali falliscono.

### 4.2 Modelli Deep Learning End-to-End (EEG_05 / EEG_07)

Notebook: `EEG_05_braindecode_baselines.ipynb`, `EEG_07_braindecode_5fold.ipynb`

- Architetture Braindecode: EEGNet, EEG Conformer, ShallowFBCSPNet, Deep4Net, ATCNet, Labram
- Input: segnale EEG grezzo (59 canali × 384 campioni), schema concr4 (4 classi)
- **Risultato**: tutti i modelli a chance level esatto (test_bacc ≈ 25.0%, std ≤ 0.001)
- **Conclusione**: deep learning end-to-end standard non riesce a generalizzare cross-soggetto

### 4.3 GCN con Grafo PCC k-NN (EEG_08)

Notebook: `EEG_08_gcn_spatial_graph.ipynb`

- Architettura: Temporal Encoder (1D CNN per-nodo) + ChebConv(K=2)
- Grafo: PCC k-NN (k=6), 59 nodi, 420 archi — ispirato a Lun et al. 2022 (GCNs-Net)
- Schema: concr4 (4 classi), split subject-independent (train 0-49, val 50-59, test 60-73)
- Class weights per gestire sbilanciamento tra classi

| Modello | val_bacc | test_bacc | epochs |
|---------|----------|-----------|--------|
| ChebGCN_2L | 0.257 | 0.249 | 24 |
| ChebGCN_3L | 0.259 | 0.245 | 18 |
| ChebGCNSkip | 0.254 | **0.255** | 16 |

**Conclusione**: grafo PCC statico senza domain adaptation → chance level (1.02x).
Class collapse su STATO (~68% predizioni). Causa: grafo medio uguale per tutti i soggetti,
nessun meccanismo per gestire ε²(soggetto)=0.85.

### 4.4 Domain Adversarial GAT — Risultati Finali (EEG_09)

Notebook: `EEG_09_gat_domain_adversarial.ipynb`

- Architettura: Temporal Encoder + GATConv(8 heads) + Gradient Reversal Layer (GRL)
- Ispirato a: DAGAM (Xu et al. 2023, arXiv:2202.12948) + GAT (Veličković et al. 2018)
- Loss: L_task(parola) + λ_adv × L_subj(soggetto, con GRL)

**v1** — λ_adv=0.1, patience=15: early stop prima che alpha DANN faccia effetto (alpha ≈ 0.38 a epoch ~19, il GRL non ha tempo di agire)

**v2** — λ_adv=0.5, patience_adv=40 (run finale):

| Modello | val_bacc | test_bacc |
|---------|----------|-----------|
| GAT_2L (no ADV) | 0.262 | 0.250 |
| GAT_2L_ADV | 0.258 | 0.254 |
| GAT_3L_ADV | 0.258 | 0.250 |

**SMOKING GUN**: rimuovendo il segnale soggetto con GRL, il modello collassa a chance totale. L'unico segnale discriminativo che il modello imparava era l'identità del soggetto, non la categoria semantica. Confusion matrix GAT_2L_ADV: 0.81–0.82 di predizioni sulla classe 1 per qualsiasi classe reale — collasso completo verso una classe dominante.

**Conclusione**: la domain adversarial training conferma in modo strutturale che nel segnale EEG grezzo cross-soggetto non esiste informazione semantica estraibile con le architetture attuali.

### 4.5 Analisi Strutturale dei Grafi (EEG_07c / EEG_07d)

Notebook: `EEG_07c_graph_exploratory.ipynb`, `EEG_07d_graph_structure.ipynb`

Analisi pre-training per capire se la struttura del grafo EEG contenga informazione semantica discriminativa.

#### EEG_07c — Analisi Esplorativa

- 6 analisi statistiche sul grafo PCC prima del training
- ANOVA F-score degli archi ≈ 1.0 per tutte le categorie semantiche → gli archi del grafo non discriminano le categorie
- t-SNE sui grafi: cluster per SOGGETTO, non per parola immaginata
- Distanza intra-categoria ≈ distanza inter-categoria → il grafo non separa semanticamente

#### EEG_07d — Subject Fingerprint e Spettro Laplaciano

**Subject fingerprint** (cosine similarity tra grafi EEG):

| | Stessa categoria | Categoria diversa |
|--|------------------|-------------------|
| Stesso soggetto | 0.785 | 0.801 |
| Soggetto diverso | 0.705 | 0.709 |

- Effetto soggetto: +0.086 | Effetto categoria: −0.010 | **Ratio: 8.6×**
- L'identità del soggetto è 8.6× più discriminativa della categoria semantica nel grafo

**Spettro Laplaciano**: le 4 categorie producono spettri Laplaciani sovrapposti in modo perfetto. ChebGCN e HGNN operano su questo spettro — per definizione matematica non possono discriminare tra le categorie.

**Small-world**: σ=2.151, C_real/C_rand=2.1, L_real/L_rand=0.976 — il grafo EEG ha topologia small-world, ma questa proprietà è identica per tutte le categorie.

**Conclusione critica**: il fallimento di tutti i modelli GNN è strutturale e matematicamente dimostrabile — il grafo EEG non contiene informazione semantica discriminativa a livello di spettro Laplaciano.

### 4.6 Advanced GNN (EEG_10)

Notebook: `EEG_10_advanced_gnn.ipynb`

- 4 architetture avanzate, schema cross-subject, concr4 (4 classi)
- Obiettivo: verificare se architetture GNN più complesse superino il fallimento strutturale

| Modello | val_bacc | Note |
|---------|----------|------|
| ChebGCN (baseline EEG_08) | 0.261 | riferimento |
| LGGNet | 0.259 | loss converge a ln(4)=1.386 |
| AT-DGNN | 0.254 | |
| DiffPool-GNN | 0.253 | |

**Nota critica — ln(4)=1.386**: questo valore è il minimo teorico della CrossEntropy per output uniforme su 4 classi. Il modello trova la soluzione ottima per un task senza segnale: predire uniformemente tutte le classi.

**Conclusione**: il modello più semplice (ChebGCN) raggiunge la val_bacc più alta — segnale classico di assenza di pattern segnale. La complessità architetturale non compensa l'assenza di informazione discriminativa nel grafo.

### 4.7 Hypergraph Neural Networks (EEG_11)

Notebook: `EEG_11_hgnn_imagined_speech.ipynb`

Contributo principale della tesi: primo test sistematico di HGNN su imagined speech a grande scala (70 soggetti).

- Basato su: Feng et al. 2019 (HGNN), Li et al. 2025 (DHSLP), AllSet (Chien et al. 2022)
- 3 architetture HGNN, schema cross-subject, concr4
- Ottimizzazioni implementate: hyperedge_index globale, preload dati in RAM, dataset pre-built (~3.7 GB)

| Modello | test_bacc | Pattern collasso |
|---------|-----------|-----------------|
| HGNN_2L | 0.250 | Split classi 1+2 |
| HGNN_2L_DYN | 0.247 | Collasso su classe 3 |
| HGNN_ATT_2L | 0.250 | Collasso su classe 1 |

**Conclusione**: tutti a chance (25.0%). Il fallimento non è architetturale ma strutturale: dimostrato matematicamente da EEG_07d che lo spettro Laplaciano è identico tra categorie. Gli iperedge aggiungono ordine superiore ma non creano informazione semantica laddove non esiste nel segnale.

### 4.8 Subject-Specific Baseline (EEG_06)

Notebook: `EEG_06_subject_specific.ipynb`

- Schema Leave-One-Session-Out: train sessioni 1-4, test sessione 5
- 10 soggetti, 6 modelli Braindecode, concr4, instance norm
- Obiettivo: verificare se esiste segnale semantico a livello subject-specific (~330 trial/soggetto)

| Modello | mean_test_bacc | std |
|---------|----------------|-----|
| EEGNet | 0.2532 | 0.0196 |
| ShallowFBCSPNet | 0.2486 | 0.0200 |
| Deep4Net | 0.2561 | 0.0476 |
| EEGConformer | 0.2488 | 0.0162 |
| ATCNet | 0.2342 | 0.0343 |
| Labram | 0.2485 | 0.0045 |

**Outlier rilevante**: Deep4Net su un soggetto specifico raggiunge test_bacc=0.344 (1.38× chance) — suggerisce che per alcuni soggetti il segnale semantico esiste, ma non è generalizzabile con i dati disponibili (~330 trial/soggetto).

**Conclusione**: anche il setting subject-specific è a chance per la maggior parte dei soggetti. Il segnale esiste sporadicamente per pochi soggetti, ma l'insufficienza dei dati per soggetto (330 trial vs. migliaia tipicamente necessari) e la variabilità intra-individuale non permettono la generalizzazione.

### 4.9 Riepilogo Risultati

| Modello | Architettura | Subject-Independent (test_bacc) | Subject-Specific (test_bacc) |
|---------|-------------|--------------------------------|------------------------------|
| Logistic Regression / MLP | Feature manuali | ~Chance | ~Chance |
| EEGNet, Conformer, ecc. (EEG_05/07) | DL raw cross-subject | ~25.0% | — |
| EEGNet, Conformer, ecc. (EEG_06) | DL raw subject-specific | — | ~25.0%–25.6% (outlier: 34.4%) |
| ChebGCN (EEG_08) | GCN statico PCC | ~25.5% (1.02×) | — |
| GAT + GRL (EEG_09 v1) | GAT + adversarial (λ=0.1) | ~25.3% | — |
| GAT + GRL (EEG_09 v2) | GAT + adversarial (λ=0.5) | ~25.2% | — |
| LGGNet / AT-DGNN / DiffPool (EEG_10) | GNN avanzate | ~25.3–25.4% | — |
| HGNN_2L / HGNN_DYN / HGNN_ATT (EEG_11) | Hypergraph NN | ~25.0% | — |

**Chance level**: 25.0% (4 classi concr4)

**Conclusione trasversale**: nessuna architettura supera il chance level in modo statisticamente rilevante in setting cross-subject. EEG_07d dimostra matematicamente perché: il grafo EEG ha spettro Laplaciano identico tra categorie semantiche, e l'effetto soggetto è 8.6× l'effetto categoria. Il percorso promettente rimane il meta-learning subject-specific o il contrastive learning cross-soggetto con normalizzazione esplicita dell'identità soggetto.

---

## 5. Clustering Semantico

Notebook `EEG_00_labels_and_tasks.ipynb`:

- Clustering delle 110 parole in categorie semantiche usando word embeddings
- Clustering a 4 e 5 gruppi
- Output: `idx2label.json`, `word2cluster_4.json`, `word2cluster_5.json`
- Obiettivo: ridurre il task da 110 classi a 4-5 categorie semantiche

---

## 6. Struttura del Repository

```
miralis-hypergraph-imagined-speech/
├── docs/
│   ├── FEATURES_AND_VISUALIZATION.md      # Guida feature e visualizzazioni
│   ├── IMPLEMENTATION_SUMMARY.md          # Riepilogo implementazione
│   ├── METRICS_EXPLANATION.md             # Spiegazione metriche (EN)
│   ├── METRICS_EXPLANATION_IT.md          # Spiegazione metriche (IT)
│   └── checkpoints/7-03-26.md            # Checkpoint sperimentale
├── scripts/
│   ├── features/
│   │   ├── comprehensive_features.py      # Engine estrazione feature
│   │   ├── bandpowers.py                  # Estrazione potenze di banda
│   │   ├── distances.py                   # Distanze tra elettrodi
│   │   └── example_synthetic_features.py  # Demo con dati sintetici
│   ├── graphs/
│   │   ├── feature_visualizations.py      # Suite di visualizzazioni
│   │   ├── eeg_viewer.py                  # Viewer EEG interattivo (GUI)
│   │   └── run_eeg_viewer_conda.sh        # Script avvio viewer
│   ├── data_processing/                   # Preprocessing MATLAB
│   ├── utils.py                           # Utility condivise
│   ├── run_comprehensive_subject.py       # Elaborazione per soggetto
│   └── run_comprehensive_sample.py        # Elaborazione campione
├── notebooks/
│   ├── EEG_00_labels_and_tasks.ipynb      # Esplorazione dataset e clustering
│   ├── EEG_00_feature_significance.ipynb  # Analisi statistica feature
│   ├── EEG_01_pipeline_metadata_features_analysis.ipynb  # Pipeline feature
│   ├── EEG_02_tensors_and_graph.ipynb     # Tensori e grafi
│   ├── EEG_03_visualization_topomaps.ipynb  # Visualizzazioni topografiche
│   ├── EEG_05_braindecode_baselines.ipynb  # Baseline DL Braindecode cross-subject
│   ├── EEG_06_subject_specific.ipynb       # Baseline subject-specific LOSO
│   ├── EEG_07_braindecode_5fold.ipynb      # Braindecode 5-fold cross-subject
│   ├── EEG_07c_graph_exploratory.ipynb     # Analisi esplorativa grafi pre-training
│   ├── EEG_07d_graph_structure.ipynb       # Subject fingerprint + spettro Laplaciano
│   ├── EEG_08_gcn_spatial_graph.ipynb      # GCN ChebConv su grafo PCC k-NN
│   ├── EEG_09_gat_domain_adversarial.ipynb # GAT + Gradient Reversal Layer (DANN)
│   ├── EEG_10_advanced_gnn.ipynb           # LGGNet, AT-DGNN, DiffPool cross-subject
│   ├── EEG_11_hgnn_imagined_speech.ipynb   # Hypergraph Neural Networks (contributo tesi)
│   └── tests/
│       ├── baseline_test_logreg_mlp.ipynb
│       ├── baseline_test_gcn.ipynb
│       ├── EEG_GNN_baseline_spatial_graph.ipynb
│       ├── EEG_GNN_baseline_feature_similarity_graph.ipynb
│       ├── EEG_GNN_temporal_baseline_spatial_graph_FIXED.ipynb
│       └── baseline_test_with_3_projector_exports.ipynb
├── figures/                               # Visualizzazioni generate
├── src/                                   # Dockerfile e script avvio
├── environment.yml                        # Ambiente Conda
└── README.md                              # Documentazione principale
```

---

## 7. Stack Tecnologico

| Componente | Tecnologia |
|------------|------------|
| Linguaggio | Python 3.12 |
| Graph Neural Networks | PyTorch Geometric 2.7.0 |
| Analisi EEG | MNE-Python 1.10.2 |
| Signal Processing | SciPy, NumPy |
| Machine Learning | scikit-learn, PyTorch 2.8.0 |
| Gestione Dati | Pandas, h5py |
| Visualizzazione | Matplotlib, Plotly, Seaborn |
| GUI | PySide6 |
| Preprocessing | MATLAB Engine 25.2 |
| Sviluppo | JupyterLab 4.4.9 |
| Deploy | Docker + Conda |

---

## 8. Insight Chiave del Progetto

1. **La variabilita inter-soggetto domina** lo spazio delle feature: i trial si raggruppano per soggetto, non per parola immaginata (ε²=0.85 per soggetto vs. ε²=0.03 per parola, confermato da EEG_07d con ratio 8.6×).

2. **I modelli vettoriali sono insufficienti** per il task a 110 parole, anche con rappresentazioni ad alta dimensionalita (11.800 feature).

3. **I grafi spaziali statici non bastano**: lo smoothing basato sulla prossimita degli elettrodi non codifica informazione lessicale.

4. **I grafi adattivi mostrano una promessa limitata**: la feature-similarity cattura una piccola struttura intra-soggetto (~2× chance) ma non generalizza tra soggetti.

5. **La struttura temporale e critica**: la suddivisione in 5 finestre temporali preserva dinamiche importanti che l'aggregazione distrugge.

6. **Il fallimento delle GNN e strutturale, non architetturale**: EEG_07d dimostra matematicamente che lo spettro Laplaciano del grafo EEG e identico tra le 4 categorie semantiche. ChebGCN, GAT e HGNN operano su questo spettro — non possono discriminare per definizione matematica, indipendentemente dalla complessita dell'architettura.

7. **La domain adversarial training ha confermato il problema**: rimuovendo il segnale soggetto con GRL (EEG_09 v2), il modello collassa completamente — l'unico segnale discriminativo era l'identita del soggetto.

8. **Le Hypergraph Neural Networks sono a chance**: EEG_11 mostra che le relazioni di ordine superiore non creano informazione semantica laddove non esiste nel segnale. Il contributo HGNN della tesi e una dimostrazione negativa rigorosa.

9. **Il segnale semantico esiste per alcuni soggetti**: l'outlier di EEG_06 (Deep4Net, test_bacc=0.344 per un soggetto specifico) suggerisce che l'informazione e presente ma altamente soggetto-specifica e non generalizzabile con i dati disponibili.

10. **La prossima frontiera e il meta-learning e il contrastive learning**: l'approccio MAML/Prototypical Networks per adattamento rapido soggetto-specifico, o il contrastive learning cross-soggetto (Shen et al. 2022), sono le direzioni piu promettenti per superare il limite strutturale dimostrato.

---

*Documento generato automaticamente dall'analisi completa del repository.*
