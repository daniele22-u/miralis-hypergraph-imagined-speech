# Progetto Completo: Decodifica dell'Imagined Speech da EEG con Graph Neural Networks

> Tesi Magistrale - Politecnico di Milano, DEIB
> Autore: Daniele Uras
> Ultimo aggiornamento: 8 marzo 2026

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
- **Canali validi**: 59 (A1 e A2 esclusi come riferimento)
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

### 4.2 Modelli GNN

#### GCN con Grafo Spaziale Statico
Notebooks: `baseline_test_gcn.ipynb`, `EEG_GNN_baseline_spatial_graph.ipynb`

- Grafo k-NN basato su posizioni elettrodi
- **Risultato**: ~Chance level (sia subject-specific che independent)
- **Conclusione**: lo smoothing spaziale statico non e sufficiente

#### GCN con Grafo Feature-Similarity
Notebook: `EEG_GNN_baseline_feature_similarity_graph.ipynb`

- Grafo adattivo basato su cosine similarity tra feature
- **Risultato subject-specific**: ~2x chance level (piccolo miglioramento)
- **Risultato subject-independent**: ~Chance level
- **Conclusione**: i grafi adattivi catturano qualcosa della struttura intra-soggetto ma falliscono nella generalizzazione

#### GCN Spazio-Temporale
Notebook: `EEG_GNN_temporal_baseline_spatial_graph_FIXED.ipynb`

- Sequenza di 5 grafi (uno per finestra temporale) con connessioni spaziali
- Aggregazione temporale via pooling o GRU
- **Stato**: preparato per la fase sperimentale successiva

### 4.3 Riepilogo Risultati

| Baseline | Tipo Grafo | Subject-Specific | Subject-Independent |
|----------|-----------|------------------|---------------------|
| Logistic Regression | Nessuno | ~Chance | ~Chance |
| MLP | Nessuno | ~Chance | ~Chance |
| GCN Statico | Spaziale (k-NN) | ~Chance | ~Chance |
| GCN Adattivo | Feature-Similarity | ~2x Chance | ~Chance |
| GCN Spazio-Temporale | Spaziale + Tempo | In preparazione | In preparazione |

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

1. **La variabilita inter-soggetto domina** lo spazio delle feature: i trial si raggruppano per soggetto, non per parola immaginata.

2. **I modelli vettoriali sono insufficienti** per il task a 110 parole, anche con rappresentazioni ad alta dimensionalita (11.800 feature).

3. **I grafi spaziali statici non bastano**: lo smoothing basato sulla prossimita degli elettrodi non codifica informazione lessicale.

4. **I grafi adattivi mostrano una promessa limitata**: la feature-similarity cattura una piccola struttura intra-soggetto (~2x chance) ma non generalizza tra soggetti.

5. **La struttura temporale e critica**: la suddivisione in 5 finestre temporali preserva dinamiche importanti che l'aggregazione distrugge.

6. **Servono relazioni di ordine superiore**: la prossima frontiera sono le hypergraph neural networks, che possono modellare interazioni tra gruppi di elettrodi, non solo coppie.

---

*Documento generato automaticamente dall'analisi completa del repository.*
