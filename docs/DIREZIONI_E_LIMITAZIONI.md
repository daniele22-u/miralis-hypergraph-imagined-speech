# Direzioni Future e Limitazioni Attuali

> Ultimo aggiornamento: 21 aprile 2026

---

## 1. Limitazioni Attuali

### 1.1 Limitazioni dei Dati

| Limitazione | Dettaglio | Impatto |
|-------------|-----------|---------|
| **Variabilita inter-soggetto** | Le feature EEG sono dominate dall'identita del soggetto, non dalla parola immaginata | I modelli subject-independent sono al livello del caso |
| **Task a 110 classi** | Classificazione multi-classe con 110 parole distinte | Chance level ~0.9%, task estremamente difficile |
| **Variabilita inter-soggetto irriducibile con feature manuali** | RSA Mantel test: r=0.001, p=0.817 — le feature EEG estratte non correlano con struttura semantica | Feature manuali troppo crude; necessario approccio DL end-to-end |
| **Cluster EEG instabili inter-soggetto** | ARI cross-subject ≈ 0 (p=1.000) — clustering EEG-driven non replicabile tra soggetti | Solo schemi supervised (Ward/POS/Semantico-BCI) sono affidabili come target |
| **70 soggetti** | Dataset insolitamente grande (la maggior parte degli studi usa <15 soggetti) | **Vantaggio**: abundant cross-subject pairs per contrastive learning |
| **Epoche corte** | ~1.5 secondi per trial a 256 Hz | Informazione temporale limitata per trial |
| **Sbilanciamento** | Non tutte le parole hanno lo stesso numero di trial | Alcune classi sotto-rappresentate |

### 1.2 Limitazioni delle Feature

| Limitazione | Dettaglio | Impatto |
|-------------|-----------|---------|
| **Feature aggregate** | Media su tutta l'epoca perde dinamiche temporali fini | Informazione lessicale potenzialmente distrutta |
| **Nessuna feature non-lineare** | Mancano entropia campionaria, dimensione frattale, Lyapunov | Dinamiche non-lineari non catturate |
| **Nessuna feature tempo-frequenza** | Mancano wavelet, STFT, Hilbert-Huang | Transitori spettrali non modellati |
| **Feature manuali** | 40 feature progettate a mano, non apprese | Potenziale bias nella selezione delle feature |
| **PLV limitato** | Calcolato solo su banda broadband | Connettivita specifica per banda non estratta |

### 1.3 Limitazioni dei Modelli

| Limitazione | Dettaglio | Impatto |
|-------------|-----------|---------|
| **Solo grafi semplici** | Archi tra coppie di nodi, nessuna iperedge | Non modellano interazioni di ordine superiore |
| **GCN shallow** | Architetture con pochi layer | Aggregazione di vicinato limitata |
| **Nessun attention mechanism** | GAT implementato in EEG_09 ma non ancora efficace | Risultati a chance nella prima run |
| **Nessun domain adaptation efficace** | GRL implementato (EEG_09) ma patience troppo bassa (15 → 40) e λ troppo basso (0.1 → 0.5) | Prima run non ha dato tempo al GRL di agire |
| **Graph classification non rispettata** | EEG_08/08b/09 usavano grafo statico condiviso — non vera graph classification | Corretto in commit 515628b (21/04/2026); ~38K grafi per-trial ora obbligatori |
| **No augmentation** | Nessuna augmentation dei dati EEG | Dataset effettivo non espanso |
| **No pre-training** | Nessun pre-training self-supervised | Rappresentazioni non apprese in modo non supervisionato |

### 1.4 Limitazioni Infrastrutturali

| Limitazione | Dettaglio |
|-------------|-----------|
| **Preprocessing MATLAB** | Dipendenza da MATLAB/EEGLAB per il preprocessing, non interamente in Python |
| **Nessuna pipeline automatizzata** | I notebook richiedono esecuzione manuale e sequenziale |
| **Nessun experiment tracking** | Manca MLflow/W&B per tracciare esperimenti sistematicamente |
| **Nessun hyperparameter tuning** | Iperparametri scelti manualmente |

---

## 2. Cosa Possiamo Fare Ora

### 2.1 Azioni Immediate (pronte con l'infrastruttura attuale)

#### A0. Subject-Centering del Segnale EEG (priorità assoluta)
- **Cosa**: sottrarre la media soggetto da ogni trial — `X_trial -= X_subject_mean` — prima di qualsiasi altra operazione
- **Come**: nei notebook di caricamento dati, calcolare la media su tutti i trial del soggetto e sottrarla
- **Perché**: le analisi di questa sessione dimostrano che la struttura nei dati è dominata dall'identità del soggetto (ε²=0.85). Il subject-centering rimuove questo DC offset senza bisogno di modelli complessi
- **Riferimento**: tecnica standard in neuroimaging cognitiva (RSA, Pattern Analysis), analoga alla baseline correction in event-related analysis
- **Effort**: bassissimo — 2 righe di codice in ogni notebook
- **Status**: da implementare come primo passo prima di qualsiasi nuovo esperimento

#### A1. Instance Normalization nei Modelli DL
- **Cosa**: sostituire BatchNorm con InstanceNorm in tutti i modelli (`nn.InstanceNorm1d`)
- **Come**: normalizzazione per-trial invece che per-batch: ogni trial normalizzato indipendentemente attraverso i canali
- **Perché**: rimuove offset di ampiezza soggetto-specifici; Bomatter et al. 2024 mostrano miglioramenti significativi
- **Effort**: basso (1 riga per modello)
- **Status**: da implementare dopo subject-centering

#### A2. Completare il Baseline Spazio-Temporale
- **Cosa**: allenare il GCN spazio-temporale gia preparato in `EEG_GNN_temporal_baseline_spatial_graph_FIXED.ipynb`
- **Come**: sequenza di 5 grafi per trial, encoding GCN + aggregazione con mean pooling o GRU
- **Perche**: stabilire un upper bound per i modelli basati su grafi semplici prima di passare agli ipergrafi
- **Effort**: basso, il codice e gia pronto

#### B. Classificazione su Categorie Semantiche (4-5 classi)
- **Cosa**: rieseguire tutti i baseline sul task ridotto (4-5 cluster semantici invece di 110 parole)
- **Come**: usare `word2cluster_4.json` / `word2cluster_5.json` gia generati
- **Perche**: un task piu semplice potrebbe rivelare struttura nel segnale che il task a 110 classi nasconde
- **Effort**: basso, richiede solo modifica delle label

#### C. Analisi Per-Banda dei Grafi
- **Cosa**: costruire grafi separati per ogni banda di frequenza (delta, theta, alpha, beta, gamma)
- **Come**: filtrare le feature per banda e calcolare cosine similarity solo su quelle
- **Perche**: diverse bande EEG codificano informazioni cognitive differenti
- **Effort**: medio, richiede modifica della pipeline grafi

#### D. Feature Selection Guidata
- **Cosa**: applicare metodi di feature selection (mutual information, mRMR, LASSO) per ridurre le 40 feature
- **Come**: selezionare le feature piu discriminative per il task specifico
- **Perche**: ridurre rumore e dimensionalita, potenzialmente migliorare la classificazione
- **Effort**: basso-medio

### 2.2 Sviluppi a Medio Termine

#### E. Cross-Subject Contrastive Learning (Shen et al. 2022)
- **Cosa**: addestrare un encoder con coppie positive cross-soggetto — stessa parola, soggetti diversi
- **Come**: coppie positive `(trial parola W soggetto A, trial parola W soggetto B)`, coppie negative `(trial parola W, trial parola V≠W)`. Loss NT-Xent o SupCon. Con 70 soggetti abbiamo coppie abbondanti
- **Perché**: l'analisi RSA e ARI mostra che la struttura per parola è completamente sommersa dalla struttura per soggetto. Il contrastive cross-soggetto forza l'encoder ad ignorare l'identità del soggetto e a rappresentare il contenuto lessicale
- **Riferimento**: Shen et al. 2022, Zhao et al. 2023 (Thinking Race — specifico per imagined speech)
- **Effort**: medio

#### E2. DANN — Gradient Reversal su Subject Label
- **Cosa**: aggiungere un discriminatore soggetto con gradient reversal all'encoder condiviso
- **Come**: encoder → task classifier + (gradient reversal →) subject discriminator. Loss combinata forza l'encoder a produrre rappresentazioni che il discriminatore soggetto non riesce a classificare
- **Perché**: approccio complementare al contrastive learning. Il discriminatore esplicito forza la rimozione dell'informazione soggetto
- **Riferimento**: Zheng & Lu 2020 (DANN per EEG)
- **Effort**: medio

#### E3. Allineamento Riemanniano (Euclidean Alignment)
- **Cosa**: riallineare le matrici di covarianza dei soggetti prima di ogni esperimento
- **Come**: `pyriemann.utils.mean.mean_riemann` per calcolare la covarianza media; trasformare ogni trial `X → R_s^{-1/2} * X`
- **Perché**: preprocessing geometricamente principiato che riduce la distanza inter-soggetto nello spazio delle matrici di covarianza
- **Riferimento**: Jayaram & Barachant 2020
- **Effort**: basso (preprocessing aggiuntivo)

#### F. Hypergraph Neural Networks
- **Cosa**: implementare reti neurali su ipergrafi dove le iperedge connettono gruppi di elettrodi
- **Come**:
  - Costruire ipergrafi basati su regioni cerebrali (frontale, temporale, parietale, occipitale)
  - Iperedge da clustering funzionale (PLV per banda di frequenza)
  - Usare framework AllSet (Chien et al. 2022) o implementare HGNN custom con PyTorch Geometric
- **Perche**: le relazioni di ordine superiore tra gruppi di elettrodi possono catturare pattern che i grafi semplici non modellano
- **Riferimento**: Li et al. 2025 raggiungono 78% con dynamic hypergraph learning; per-trial hyperedge construction via PLV/coherence
- **Effort**: alto

#### G. Graph Attention Networks (GAT)
- **Cosa**: sostituire GCN con GAT per pesare dinamicamente i vicini
- **Come**: implementare GAT con PyTorch Geometric (`GATConv`)
- **Perche**: non tutti gli elettrodi vicini contribuiscono ugualmente alla classificazione
- **Effort**: medio

#### H. Domain Adaptation per Transfer Cross-Soggetto
- **Cosa**: implementare strategie di adattamento di dominio avanzate
- **Come**:
  - Allineamento delle distribuzioni (MMD, Deep CORAL)
  - Adversarial domain adaptation (DANN — vedi E2)
  - Adaptive Batch Normalization (Lee et al. 2022): aggiornare running stats BN con pochi trial del soggetto target
- **Perche**: la variabilita inter-soggetto e il problema principale; servono metodi per compensarla
- **Effort**: alto

#### I. Feature Apprese — EEGNet / EEG Conformer End-to-End
- **Cosa**: sostituire le 40 feature manuali con input raw (59, 384) direttamente ai modelli
- **Come**: EEGNet su raw EEG con subject-centering → baseline end-to-end; EEG Conformer per dipendenze temporali a lungo raggio
- **Perche**: la RSA dimostra che le feature manuali non correlano con la semantica. L'approccio end-to-end ha già il notebook `EEG_04_braindecode_raw_baselines.ipynb` pronto
- **Effort**: basso (infrastruttura già pronta)

#### J. Neural Prototype Clustering (contributo originale)
- **Cosa**: costruire prototipi neurali per parola (media dei trial z-scored per soggetto), clusterizzarli a 4-5 gruppi, confrontare con gli schemi semantici tramite ARI/V-measure
- **Motivazione**: la domanda — "lo spazio EEG recupera parzialmente la struttura semantica?" — è aperta nella letteratura per imagined speech. Pereira et al. 2018 lo hanno dimostrato per fMRI; l'analogo EEG non è stato pubblicato a grande scala
- **Come**: già implementato parzialmente in `EEG_00_labels_and_tasks.ipynb`. L'estensione è confrontare i cluster EEG-z con Ward-4/POS-4/Semantico-BCI-5 via ARI
- **Effort**: basso (codice base già presente)

### 2.3 Sviluppi a Lungo Termine

#### I. Feature Tempo-Frequenza
- **Cosa**: aggiungere wavelet transforms, STFT, Hilbert-Huang Transform
- **Perche**: catturano transitori spettrali che il metodo di Welch non rileva
- **Effort**: medio

#### J. Feature Non-Lineari
- **Cosa**: aggiungere sample entropy, approximate entropy, dimensione frattale, esponente di Lyapunov
- **Perche**: catturano la complessita non-lineare del segnale EEG
- **Effort**: medio

#### K. EEG Data Augmentation
- **Cosa**: generare dati sintetici per espandere il dataset
- **Come**: noise injection, time warping, channel dropout, mixup, GAN-based
- **Perche**: piu dati di training migliorano la generalizzazione
- **Effort**: medio

#### L. Pipeline Automatizzata End-to-End
- **Cosa**: creare una pipeline completa dalla raw data alla predizione
- **Come**: script Python unico o pipeline con DVC/Prefect
- **Perche**: riproducibilita e facilita di sperimentazione
- **Effort**: medio-alto

#### M. Experiment Tracking
- **Cosa**: integrare MLflow o Weights & Biases
- **Perche**: tracciare sistematicamente iperparametri, metriche e artefatti
- **Effort**: basso-medio

---

## 3. Roadmap Suggerita

### Fase 1: Correzione Fondamentale (immediato)
1. **Subject-centering** su tutti i notebook (`X_trial -= X_subject_mean`) — prerequisito per tutto il resto
2. **Instance Normalization** sostituisce BatchNorm in tutti i modelli DL
3. **EEGNet end-to-end** su segnale raw (59, 384) — già in `EEG_04_braindecode_raw_baselines.ipynb`
4. **Completare** training GCN spazio-temporale
5. **Rieseguire** tutti i baseline sul task a 4-5 categorie semantiche (schemi affidabili: Ward-4, POS-4, Semantico-BCI-5)

### Fase 2: Affrontare la Variabilità Inter-Soggetto (breve termine)
6. **Allineamento Riemanniano** (pyriemann Euclidean Alignment) — preprocessing aggiuntivo
7. **Cross-subject contrastive learning** (Shen et al. 2022) — sfrutta i 70 soggetti come vantaggio
8. **DANN** gradient reversal su subject label (Zheng & Lu 2020)
9. **GAT** (Graph Attention Networks) — `GATConv` di PyG

### Fase 3: Hypergraph e Analisi Avanzata (medio termine)
10. **Neural Prototype Clustering** — confronto EEG vs. semantica con ARI (contributo originale)
11. **Dynamic Hypergraph (DHSLP-style)** — per-trial PLV/coherence hyperedges
12. **Allineamento Riemanniano** per domain generalization

### Fase 4: Sistema Completo (lungo termine)
13. Feature tempo-frequenza (wavelet, STFT) — dopo aver stabilito baseline DL
14. Data augmentation sistematica (Rommel et al. 2022)
15. Pipeline automatizzata end-to-end
16. Validazione su dataset esterni (Nieto et al. Inner Speech)

---

## 4. Rischi e Considerazioni

| Rischio | Probabilita | Mitigazione |
|---------|-------------|-------------|
| Task a 110 classi intrinsecamente troppo difficile per EEG | Alta | Ridurre a categorie semantiche (4-5 classi) |
| Variabilita inter-soggetto irriducibile | Media-Alta | Domain adaptation, subject normalization |
| Overfitting su pochi soggetti | Alta | Cross-validation rigorosa, data augmentation |
| Feature manuali non sufficienti | Media | Rappresentazioni apprese (autoencoder, contrastive) |
| Costo computazionale degli ipergrafi | Media | Ottimizzazione, mini-batch training |
| Tempo di sviluppo eccessivo | Media | Prioritizzare azioni a basso effort e alto impatto |

---

## 5. Metriche di Successo

| Obiettivo | Metrica | Target |
|-----------|---------|--------|
| Baseline vettoriale | Accuracy 110 classi | > 2% (2x chance) |
| GNN subject-specific | Accuracy 110 classi | > 5% |
| GNN subject-independent | Accuracy 110 classi | > 2% |
| Categorie semantiche (subject-specific) | Accuracy 4-5 classi | > 40% |
| Categorie semantiche (subject-independent) | Accuracy 4-5 classi | > 30% |
| Hypergraph (riferimento Li et al.) | Accuracy comparabile | Avvicinare 78% |

---

*Documento generato automaticamente dall'analisi completa del repository.*
