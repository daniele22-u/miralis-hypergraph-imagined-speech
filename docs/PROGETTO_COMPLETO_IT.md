# Progetto Completo: Decodifica dell'Imagined Speech da EEG con Graph Neural Networks

> Tesi Magistrale - Politecnico di Milano, DEIB
> Autore: Daniele Uras
> Ultimo aggiornamento: 10 maggio 2026

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

| Modello | Architettura | Setting | test_bacc |
|---------|-------------|---------|-----------|
| Logistic Regression / MLP | Feature manuali | S-Indep | ~Chance |
| EEGNet, Conformer, ecc. | Deep learning raw | S-Indep | ~25.0% |
| ChebGCN (EEG_08) | GCN statico PCC | S-Indep | ~25.5% (1.02x) |
| GAT + GRL (EEG_09 v1 GAT) | GAT + adversarial PyG | S-Indep | ~25.3% |
| ChebGCN LOSO (EEG_08b) | GCN per-trial pruned | S-Spec | ~26.3% |
| GCN/GAT/DANN ablation (EEG_08) | 5 metriche pruned | S-Indep | ~25% |
| **HGNN S-Indep (EEG_09)** | HGNN 2L ipergrafi pruned | S-Indep | **~25.7%** (range 0.253–0.260) |
| **HGNN S-Spec LOSO (EEG_09b)** | HGNN 2L ipergrafi pruned | S-Spec | **top: 31.8%** (P031), mediana ~25% |
| **T-HGNN (EEG_10)** | CNN 1D per-nodo + HGNN | S-Indep | **~24.8%** (0.241–0.252) ↓ sotto chance |
| **W-HGNN S-Indep (EEG_11)** | H_pruned soft fisso, features per finestra | S-Indep | **~25.5%** (0.249–0.261) |
| **W-HGNN S-Spec LOSO (EEG_12)** | W-HGNN, K=8 finestre, LOSO | S-Spec | **top: 34.4%** (P007), mediana ~25%, delta medio vs 09b: ≈0 |

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

### 4.8 Subject Clustering EEG-First (EEG_08c — sessione 28/04–08/05)

Notebook: `EEG_08c_subject_quality_metrics.ipynb` — **completato**

**Contesto**: Francesco ha chiuso l'approccio precedente (usare l'accuracy come proxy per "capacità IS"). Motivazione: la varianza delle accuracies tra soggetti è troppo bassa e troppo vicina al baseline per essere un segnale primario affidabile.

**Nuova direzione (istruzione Francesco — implementata):**
> "Parti dai dati EEG grezzi, clusterizza i soggetti sulla base delle feature EEG, trova 2-3 cluster naturali. Solo DOPO vai a vedere come quei cluster si mappano sull'accuracy — usa l'accuracy per validare/interpretare i cluster, non per definirli."

**Implementazione (18 celle):**
- Feature per-soggetto: band power (5 bande × 61 canali = 305d) + node degree connectivity (61d) + ISC similarity (1d) = 367d totali
- PCA → riduzione a 20 componenti (spiegano ~85% varianza)
- K-Means k=2..6 + Silhouette + Elbow per selezione k → **k=3 ottimale** (silhouette = 0.253)
- Dendrogramma Ward linkage
- t-SNE 2D + PCA 2D scatter colorato per cluster
- Post-hoc accuracy overlay (carica `eeg08b_subject_ranking.csv`)
- Test statistici: Kruskal-Wallis + Spearman ISC vs accuracy

**Risultati (null result confermato):**

| Metrica | Valore |
|---------|--------|
| Silhouette k=3 | 0.253 |
| Kruskal-Wallis p | **0.939** (non significativo) |
| Spearman ISC vs accuracy | ~0 |

**Conclusione**: le feature EEG statiche (band power + connettività) **non clusterizzano** i soggetti in gruppi con performance di decodifica diverse. Il cluster k=3 corrisponde a 3 soggetti EEG-outlier (P017, P055, P063) con accuracy media. La struttura EEG inter-soggetto è reale (ε²=0.85) ma non predice la decodificabilità dei trial → confermato empiricamente il feedback di Francesco.

### 4.9 Dataset Grafi/Ipergrafi EEG_07f (sessione 26–28/04)

Notebook: `EEG_07f_build_graphs_hypergraphs.ipynb` — **completato**

**Build completo dataset graph classification:**
- 38.883 trial totali × 5 metriche (PCC, |PCC|, Im-PCC, wPLI, PLV) × 2 varianti (raw / consensus-pruned)
- 4 output per trial: grafo raw, grafo pruned, ipergrafo raw, ipergrafo pruned
- Shape per trial: `x=(61, 384)`, `adj=(61,61)`, `H=(61,E)` con E variabile dopo pruning
- Paradigma: 1 file `.pt` per trial — **GRAPH CLASSIFICATION**
- Output: `data/graphs_{metric}/`, `data/graphs_pruned_{metric}/`, `data/hypergraphs_{metric}/`, `data/hypergraphs_pruned_{metric}/`

**Analisi post-hoc aggiunta (3 celle):**
- Topomap connettività: grado pesato per canale per 4 varianti × 5 metriche (output: `figures/topomap_P000_acqua.png`)
- Mean adjacency per cluster semantico (concr4): matrici medie + 6 differenze pairwise TwoSlopeNorm
- Visualizzazione brain sLORETA dorsal view con overlay parcellazione Desikan-Killiany (fsaverage)

### 4.10 GCN/GAT/DANN Ablation su Grafi Pruned (EEG_08 — sessione 28/04)

Notebook: `EEG_08_gnn_classification.ipynb` — **configurato, da rieseguire con pruned**

**Configurazione aggiornata:**
- `GRAPH_DIRS` ora usa esclusivamente grafi pruned (`graphs_pruned_*`)
- 10 tipi di grafo: 5 metriche × pruned only
- 3 modelli: GCN, GAT, DANN (Domain Adversarial)
- Metrica: balanced accuracy (4 classi concr4, chance = 25%)
- Risultato atteso: ~25% (confermato nella run precedente con raw — grafo statico spaziale non sufficiente)

**EEG_08b subject-specific** aggiornato analogamente a pruned-only.

### 4.11 HGNN Subject-Independent Ablation (EEG_09 — sessione 28/04–08/05)

Notebook: `EEG_09_hgnn_classification.ipynb` — **creato, da eseguire sul server**

**Architettura: HGNN (Feng et al. 2019, AAAI)**
- Formula: `X' = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2} X Θ`
- Implementazione: batched bmm per (B,N,C) × (B,N,E)
- Dataset standard PyTorch (non PyG) — H denso (61,61) con padding consensus

**Configurazione:**
- 5 metriche (pcc, abs_pcc, im_pcc, wpli, plv) × pruned only = 5 run
- Split: TRAIN sogg 0–49, VAL 50–59, TEST 60–73
- MAX_EPOCHS=60, PATIENCE=12, HIDDEN=128, LR=1e-3
- W&B tracking con heatmap output
- Instance norm attiva (Bomatter 2024)

**Risultati (eseguito 08/05/2026):**

| Metrica | Val bAcc | Test bAcc |
|---------|----------|-----------|
| pcc     | 0.264    | **0.260** |
| abs_pcc | 0.262    | 0.253     |
| im_pcc  | 0.260    | 0.256     |
| wpli    | 0.261    | 0.255     |
| plv     | 0.255    | 0.260     |

Chance level: 25.0% — tutti i modelli a ~25–26%. Range test: 0.253–0.260 (spread <1%).

**Conclusione**: ipergrafo statico pruned in setting subject-independent → chance level, identico al GCN su grafi semplici. La topologia ipergraph statica non porta vantaggio rispetto al grafo semplice. Necessaria modellazione temporale esplicita.

**Note tecniche:**
- H pruned ha E variabile dopo consensus → `F.pad(H, (0, N_CHANNELS - H.shape[1]))` normalizza a (61,61)
- `VARIANTS = [True]` (pruned only, conforme a policy post-07f)

### 4.12 HGNN Subject-Specific LOSO (EEG_09b — sessione 28/04–08/05)

Notebook: `EEG_09b_hgnn_subject_specific.ipynb` — **completato e debuggato**

**Schema: subject-specific LOSO**
- Per ogni soggetto: test=ultima sessione, val=penultima, train=resto
- Stessa architettura HGNN di EEG_09

**Bug corretti:**
1. `N_CHANNELS` non definito nel CONFIG → aggiunto `N_CHANNELS = 61`, `N_SAMPLES = 384`
2. H shape mismatch (E variabile dopo pruning) → `F.pad(H, (0, N_CHANNELS - H.shape[1]))`
3. `KeyError: 'Test bAcc'` (SUBJECT_RESULTS vuoto) → guard `if not SUBJECT_RESULTS:`
4. `wandb.sdk.mailbox.MailboxClosedError` in loop Jupyter → `settings=wandb.Settings(start_method='thread')`

**Risultati:**

| Soggetto top | Test bAcc | Note |
|-------------|-----------|------|
| P031 | 0.318 | Miglior soggetto Top-1 |
| P015 | ~0.53 | Miglior soggetto Top-2 |
| Mediana | ~0.25 | Chance level |
| % sopra chance | ~47% (35/74) | BCI illiteracy confermata |

**Insight chiave:**
- Forte bias verso classe ASTRATTO nei soggetti top (recall 0.70–0.86)
- P034: class collapse su STATO (recall 0.86, altre classi ~0)
- ~53% dei soggetti sotto o a chance anche con HGNN subject-specific

### 4.13 T-HGNN: Temporal Encoder + HGNN (EEG_10 — sessione 08–10/05)

Notebook: `EEG_10_temporal_hgnn.ipynb` — **completato**

**Motivazione**: EEG_09 usa x=(61,384) come feature nodo dirette → nessuna modellazione temporale esplicita. EEG_10 introduce un encoder 1D CNN per canale prima dell'HGNN.

**Architettura (T-HGNN)**:
- `TemporalEncoder`: Conv1d(1,16,k=25,s=4) → BN+ELU → Conv1d(16,32,k=15,s=4) → BN+ELU → Conv1d(32,64,k=8,s=4) → BN+ELU → AdaptiveAvgPool1d(1) → (B,N,64)
- Input x=(B,61,384) reshape a (B×61,1,384), CNN, reshape a (B,61,64)
- HGNN 2L su feature temporali compressed (64d invece di 384d)
- Anti-collapse: WeightedRandomSampler + label_smoothing=0.1

**Risultati (5 metriche × pruned, S-Indep):**

| Metrica | Val bAcc | Test bAcc |
|---------|----------|-----------|
| pcc     | 0.262    | 0.248     |
| abs_pcc | 0.265    | 0.252     |
| im_pcc  | 0.258    | 0.250     |
| wpli    | 0.260    | 0.249     |
| plv     | 0.260    | 0.241     |

**Conclusione**: T-HGNN *peggiore* di EEG_09 (24.8% vs 25.7%). L'encoder CNN introduce overfitting — il segnale compresso 64d perde informazione rispetto ai 384 sample diretti nel setting subject-independent. Mild overfitting (val > test consistently).

**Lezione**: comprimere il segnale temporale prima dell'HGNN non aiuta — la variabilità inter-soggetto è troppo alta per imparare una compressione generalizzabile.

**Bug risolto**: class collapse ASTRATTO (WeightedRandomSampler + class_weights = double-counting → bias invertito). Fix: rimuovere `weight=class_weights` da CrossEntropyLoss — sampler già bilancia i batch.

---

### 4.14 W-HGNN: Windowed HGNN con H_pruned Fisso (EEG_11 — sessione 08–10/05)

Notebook: `EEG_11_dynamic_hgnn.ipynb` — **completato**

**Ragionamento evolutivo**:
- EEG_11 v1: H_k = |PCC(x_k)| per finestra → rumoroso (47 df per 61² correlazioni su 48 sample)
- EEG_11 v2: H_k = H_pruned ⊙ |PCC(x_k)| → tautologico (H_pruned già basato su PCC)
- **EEG_11 finale (W-HGNN)**: H_pruned soft come topologia FISSA, feature nodo variano per finestra

**Architettura (W-HGNN)**:
- K=8 finestre temporali (384/8 = 48 sample ≈ 188ms)
- Per ogni finestra k: `feat_k = Linear(x_k: 48→32)` + `out_k = HGNN(feat_k, H_pruned)` → H invariato
- `z = mean(out_1..K)` → Linear → logits(B,4)
- H_pruned soft (non binarizzato) — topologia consensus-validated noise-free

**Risultati (5 metriche × pruned, S-Indep):**

| Metrica | Val bAcc | Test bAcc |
|---------|----------|-----------|
| pcc     | 0.256    | 0.249     |
| abs_pcc | 0.253    | 0.260     |
| im_pcc  | 0.262    | 0.261     |
| wpli    | 0.260    | 0.252     |
| plv     | 0.261    | 0.253     |

**Conclusione**: W-HGNN ≈ HGNN statico (EEG_09). La variazione temporale nei feature dei nodi non introduce guadagno misurabile in S-Indep. Il bottleneck rimane la variabilità inter-soggetto (ε²=0.85), non la scelta di architettura temporale.

---

### 4.15 W-HGNN Subject-Specific LOSO (EEG_12 — sessione 10/05)

Notebook: `EEG_12_whgnn_subject_specific.ipynb` — **completato**

**Schema**: replica di EEG_09b (HGNN statico LOSO) con architettura W-HGNN. LOSO per sessione (test=ultima, val=penultima, train=resto). 74 soggetti totali.

**Risultati top-10:**

| Soggetto | W-HGNN (EEG_12) | HGNN (EEG_09b) | Delta |
|----------|-----------------|----------------|-------|
| P007 | **0.344** | 0.250 | +0.094 |
| P051 | 0.333 | 0.279 | +0.054 |
| P057 | 0.316 | 0.225 | +0.091 |
| P070 | 0.315 | 0.250 | +0.065 |
| P045 | 0.312 | 0.257 | +0.056 |

**Statistiche globali:**
- Best: P007 = 0.344 (nuovo record, vs 0.318 di P031 in EEG_09b, +8.2% relativo)
- Delta medio W-HGNN vs HGNN: **−0.004** (praticamente zero)
- ~50% soggetti sopra chance, ~50% sotto — BCI illiteracy invariata

**Insight confusion matrix**:
- P007: distribuzione più bilanciata tra classi
- P045: collasso su AZIONE (0.85 recall) — WeightedSampler non risolve tutto
- P051/P057: collasso parziale su ASTRATTO (0.61–0.66 recall)
- Bias per-soggetto è diverso → problema strutturale (identità soggetto), non architetturale

**Conclusione**: W-HGNN migliora il tetto dei soggetti top (+0.09 per i migliori) ma non cambia la distribuzione complessiva. Il delta medio ≈ 0 conferma che il guadagno temporale è marginale. Il soffitto attuale con questo paradigma è ~34% su 4 classi in setting subject-specific.

**Gap con target tesi**: Li et al. 2025 (DHSLP) raggiungono ~78% con dynamic hypergraph *appreso* end-to-end — differenza chiave: le iperedge non sono costruite da PCC ma imparate come parametri della rete. Prossimo step: EEG_13.

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
