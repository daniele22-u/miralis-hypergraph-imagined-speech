# Direzioni Future e Limitazioni Attuali

> Ultimo aggiornamento: 8 marzo 2026

---

## 1. Limitazioni Attuali

### 1.1 Limitazioni dei Dati

| Limitazione | Dettaglio | Impatto |
|-------------|-----------|---------|
| **Variabilita inter-soggetto** | Le feature EEG sono dominate dall'identita del soggetto, non dalla parola immaginata | I modelli subject-independent sono al livello del caso |
| **Task a 110 classi** | Classificazione multi-classe con 110 parole distinte | Chance level ~0.9%, task estremamente difficile |
| **Numero limitato di soggetti** | Dataset con pochi soggetti disponibili | Limita la generalizzabilita dei risultati |
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
| **Nessun attention mechanism** | Manca Graph Attention (GAT) | Tutti i vicini pesati ugualmente |
| **Nessun domain adaptation** | Nessuna strategia per il trasferimento tra soggetti | Variabilita inter-soggetto non compensata |
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

#### A. Completare il Baseline Spazio-Temporale
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

#### E. Hypergraph Neural Networks
- **Cosa**: implementare reti neurali su ipergrafi dove le iperedge connettono gruppi di elettrodi
- **Come**:
  - Costruire ipergrafi basati su regioni cerebrali (frontale, temporale, parietale, occipitale)
  - Iperedge da clustering funzionale (gruppi di elettrodi co-attivati)
  - Usare librerie come `hypergraph-nn` o implementare HGNN custom con PyTorch Geometric
- **Perche**: le relazioni di ordine superiore tra gruppi di elettrodi possono catturare pattern che i grafi semplici non modellano
- **Riferimento**: Li et al. (2025) raggiungono 78% con dynamic hypergraph learning
- **Effort**: alto

#### F. Graph Attention Networks (GAT)
- **Cosa**: sostituire GCN con GAT per pesare dinamicamente i vicini
- **Come**: implementare GAT con PyTorch Geometric (`GATConv`)
- **Perche**: non tutti gli elettrodi vicini contribuiscono ugualmente alla classificazione
- **Effort**: medio

#### G. Domain Adaptation per Transfer Cross-Soggetto
- **Cosa**: implementare strategie di adattamento di dominio
- **Come**:
  - Allineamento delle distribuzioni (MMD, CORAL)
  - Adversarial domain adaptation
  - Subject-specific normalization layers
- **Perche**: la variabilita inter-soggetto e il problema principale; servono metodi per compensarla
- **Effort**: alto

#### H. Feature Apprese (Deep Learning)
- **Cosa**: sostituire o integrare le feature manuali con rappresentazioni apprese
- **Come**:
  - Autoencoder convoluzionali sul segnale raw
  - Contrastive learning (SimCLR/BYOL adattato per EEG)
  - EEGNet o altri modelli end-to-end
- **Perche**: le feature manuali potrebbero non catturare i pattern piu discriminativi
- **Effort**: alto

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

### Fase 1: Consolidamento Baseline (immediato)
1. Completare training GCN spazio-temporale
2. Rieseguire tutti i baseline sul task a 4-5 categorie semantiche
3. Implementare experiment tracking (W&B o MLflow)

### Fase 2: Modelli Avanzati su Grafi (breve termine)
4. Implementare GAT (Graph Attention Networks)
5. Grafi per-banda (feature filtrate per frequenza)
6. Feature selection guidata

### Fase 3: Hypergraph e Transfer (medio termine)
7. Implementare Hypergraph Neural Networks
8. Domain adaptation per generalizzazione cross-soggetto
9. Contrastive learning per rappresentazioni apprese

### Fase 4: Sistema Completo (lungo termine)
10. Feature tempo-frequenza e non-lineari
11. Data augmentation
12. Pipeline automatizzata end-to-end
13. Validazione su dataset esterni

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
