# Progetto Completo: Decodifica dell'Imagined Speech da EEG con Graph Neural Networks

> Tesi Magistrale - Politecnico di Milano, DEIB
> Autore: Daniele Uras
> Ultimo aggiornamento: 26 aprile 2026

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

### 4.4 Domain Adversarial GAT (EEG_09)

Notebook: `EEG_09_gat_domain_adversarial.ipynb`

- Architettura: Temporal Encoder + GATConv(8 heads) + Gradient Reversal Layer
- Ispirato a: DAGAM (Xu et al. 2023, arXiv:2202.12948) + GAT (Veličković et al. 2018)
- Loss: L_task(parola) + λ_adv × L_subj(soggetto, con GRL)
- Prima run: λ_adv=0.1, patience=15 — early stop prima che alpha DANN faccia effetto

| Modello | val_bacc | test_bacc | epochs |
|---------|----------|-----------|--------|
| GAT_2L (no ADV) | 0.262 | 0.253 | 19 |
| GAT_2L_ADV | 0.255 | 0.253 | 20 |
| GAT_3L_ADV | 0.251 | 0.250 | 16 |

**Conclusione**: tutti a chance. Problema identificato: patience=15 causa early stop a epoch
~19 quando alpha DANN è ancora ~0.38 — il GRL non ha tempo di agire efficacemente.
Seconda run pianificata con λ_adv=0.5 e patience_adv=40.

### 4.5 GCN Subject-Specific Leave-One-Session-Out (EEG_08b/c)

Notebook: `EEG_08b_subject_specific_gcn.ipynb` — risultati in `data/interim/eeg08c_ss_4_norm_results.csv`

- Architettura: Temporal Encoder (1D CNN per-nodo) + ChebConv(K=2), come EEG_08
- Schema: **subject-specific** — ogni soggetto addestrato separatamente, Leave-One-Session-Out (LOSO)
- Grafo: PCC k-NN (k=6) — ⚠️ questa run usa ancora grafo statico per-soggetto (OLD code, da rieseguire con graph classification per-trial)
- Schema: concr4 (4 classi), normalizzazione instance-norm attiva
- 10 soggetti (00-04, 06-09, soggetto 05 escluso per dati mancanti)
- Figura: `figures/eeg08c_ss_4_norm_boxplot.png`

**Risultati medi (10 soggetti):**

| Modello | val_bacc (avg) | test_bacc (avg) | note |
|---------|----------------|-----------------|------|
| ChebGCN_2L | 0.298 | 0.247 | ≈ chance |
| ChebGCN_3L | 0.302 | 0.236 | sotto chance |
| ChebGCNSkip | 0.302 | **0.263** | 1.05x chance |

**Migliore singolo soggetto**: soggetto 02 con ChebGCNSkip → test_bacc = 0.315 (1.26x chance)

**Conclusione**: risultati a chance anche in setting subject-specific LOSO. Causa probabile:
grafo PCC statico calcolato sull'intera sessione di training (non per-trial) — non è vera graph
classification. Codice corretto in sessione 21/04 (EEG_08b ora usa `pcc_to_edge_index_trial`
per-trial). Da **rieseguire sulla VM** per confronto valido.

> ⚠️ Nota metodologica: i risultati EEG_08c sono stati generati con OLD code (grafo condiviso
> per-soggetto). La correzione graph classification per-trial è nel commit `515628b`.
> Re-run necessaria per risultati metodologicamente corretti.

### 4.6 Riepilogo Risultati

| Modello | Architettura | Subject-Independent (test_bacc) |
|---------|-------------|--------------------------------|
| Logistic Regression / MLP | Feature manuali | ~Chance |
| EEGNet, Conformer, ecc. | Deep learning raw | ~25.0% |
| ChebGCN (EEG_08) | GCN statico PCC | ~25.5% (1.02x) |
| GAT + GRL (EEG_09 v1) | GAT + adversarial | ~25.3% (λ=0.1, patience=15) |
| ChebGCN SS-LOSO (EEG_08b/c) | GCN subject-specific | ~26.3% (1.05x) ⚠️ old code |
| **GAT + GRL (EEG_09 v2)** | GAT + adversarial | 🔄 in corso (λ=0.5, patience=40) |

Chance level: **25.0%** (4 classi concr4)

### 4.7 Analisi Connettività Inter-Soggetti (EEG_07e — sessione 26/04)

Notebook: `EEG_07e_build_graphs_tensors.ipynb` — cella statica aggiunta (id `197633c8`)

Confronto inter-soggetti su trial singolo (`accendere_img.csv`) per 6 soggetti campione.
Quattro rappresentazioni di connettività confrontate:

| Rappresentazione | Descrizione |
|-----------------|-------------|
| PCC matrix | Matrice correlazione grezza — struttura a blocchi visibile |
| Adj threshold | Tutte le coppie con PCC > p50 (0.157) — non top-k |
| k-NN raw | Top-6 vicini per nodo — sempre 263±15 archi (fisso per costruzione) |
| Consensus | Arco sopravvive in ≥2/5 metriche (PCC/PLV/wPLI/CPCCabs/CPCCim) — cache-free |

**Risultati chiave:**

| Soggetto | Archi thr | Archi consensus | Gruppo |
|----------|-----------|----------------|--------|
| P002_S005 | **1507** | 194 | Alta connettività |
| P008_S004 | **1525** | 237 | Alta connettività |
| P011_S003 | 1285 | 202 | Alta connettività |
| P000_S001 | 942 | 208 | Media |
| P014_S001 | 738 | 185 | Bassa connettività |
| P005_S004 | 679 | 194 | Bassa connettività |

**Insight principali:**
- **Adj threshold è la metrica più discriminativa**: spread 679–1525 archi (~2x), riflette la forza assoluta della connettività EEG del soggetto
- **Consensus è stabile** (185–237 archi): il filtro multi-metrica comprime le differenze — NON utile per discriminare soggetti
- **k-NN non discrimina**: per costruzione ogni nodo ha sempre k=6 vicini
- **Due gruppi naturali** già visibili da trial singolo: alta connettività (P002/P008/P011) vs bassa (P005/P014)
- **HE co-membership**: differenze visive drammatiche tra i due gruppi — P002/P008 mostrano blocchi cyan brillanti, P005/P014 quasi bui
- **Bug cache identificato**: `_cache_key()` usa `Path(stem)` — stessa chiave per lo stesso word tra soggetti diversi; workaround implementato chiamando le funzioni di metrica direttamente

**Output salvati:**
- `figures/eeg07e_cross_subject_connectivity.png` — griglia 6×6 (PCC | adj_thr | k-NN | consensus | HE | bar)
- `figures/eeg07e_cross_subject_profiles.png` — profili sovrapposti + barchart archi

### 4.8 Subject Clustering EEG-First (EEG_08 — direzione Francesco)

Notebook: `EEG_08_subject_clustering.ipynb` — **in preparazione**

**Contesto**: Francesco ha chiuso l'approccio di EEG_08 precedente (usare l'accuracy come proxy per "capacità IS"). Motivazione: la varianza delle accuracies tra soggetti è troppo bassa e troppo vicina al baseline per essere un segnale primario affidabile.

**Nuova direzione (istruzione Francesco):**
> "Parti dai dati EEG grezzi, clusterizza i soggetti sulla base delle feature EEG, trova 2-3 cluster naturali. Solo DOPO vai a vedere come quei cluster si mappano sull'accuracy — usa l'accuracy per validare/interpretare i cluster, non per definirli."

**Piano implementativo:**
1. Feature per-soggetto (media multi-trial su campione di sessioni):
   - `thresh_density` = n_archi(PCC>p50) / n_coppie — la metrica più discriminativa trovata
   - `mean_pcc` = media off-diagonale matrice PCC
   - `pcc_block_strength` = primo autovalore del Laplaciano normalizzato (struttura a blocchi)
   - Band power medio per canale (delta/theta/alpha/beta/gamma)
2. Feature matrix: 70 soggetti × ~10 feature
3. K-Means e clustering gerarchico → 2–3 cluster
4. Visualizzazione: PCA/UMAP dello spazio soggetti, colorato per cluster
5. Post-hoc: sovrapposizione accuracy da EEG_06 — i cluster corrispondono a performance diversa?

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

1. **La variabilita inter-soggetto domina** lo spazio delle feature: i trial si raggruppano per soggetto, non per parola immaginata (ε²=0.85 per soggetto vs 0.03 per parola).

2. **I modelli vettoriali sono insufficienti** per il task a 110 parole, anche con rappresentazioni ad alta dimensionalita (11.800 feature).

3. **I grafi spaziali statici non bastano**: lo smoothing basato sulla prossimita degli elettrodi non codifica informazione lessicale.

4. **I grafi adattivi mostrano una promessa limitata**: la feature-similarity cattura una piccola struttura intra-soggetto (~2x chance) ma non generalizza tra soggetti.

5. **La struttura temporale e critica**: la suddivisione in 5 finestre temporali preserva dinamiche importanti che l'aggregazione distrugge.

6. **Servono relazioni di ordine superiore**: la prossima frontiera sono le hypergraph neural networks, che possono modellare interazioni tra gruppi di elettrodi, non solo coppie.

7. **L'accuracy non è segnale primario per la variabilità inter-soggetto**: la spread delle accuracies tra soggetti è troppo bassa (~21–27% in concr4, very small range) per distinguere soggetti "capaci IS" da non-capaci. L'approccio corretto è clustering EEG-first (features grezze → cluster → validation post-hoc con accuracy).

8. **La connettività threshold (adj_thr) è il segnale più discriminativo trovato**: il numero di coppie con PCC > p50 varia da ~679 a ~1525 tra soggetti (~2x), mentre consensus (185–237) e k-NN (~263 fisso) non discriminano. Due gruppi naturali emergono già da trial singolo.

---

*Documento generato automaticamente dall'analisi completa del repository.*
