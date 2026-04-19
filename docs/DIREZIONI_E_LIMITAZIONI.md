# Direzioni Future e Limitazioni Attuali

> Ultimo aggiornamento: 19 aprile 2026

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
| **Fallimento strutturale dimostrato** | EEG_07d: spettro Laplaciano identico tra 4 categorie semantiche — ChebGCN/GAT/HGNN non possono discriminare per definizione matematica | Nessuna architettura GNN standard può superare questo limite |
| **Effetto soggetto 8.6× l'effetto categoria** | Cosine similarity: stesso soggetto = 0.79 vs. soggetto diverso = 0.71; stessa categoria vs. diversa: differenza = 0.010 | Il segnale soggetto sovrasta il segnale semantico in qualsiasi rappresentazione grafo |
| **Domain adversarial training fallisce** | GAT + GRL (EEG_09 v2, λ=0.5): rimozione segnale soggetto → collasso totale del modello | L'unico segnale discriminativo era l'identità soggetto; rimosso quello, non resta nulla |
| **GNN avanzate non migliorano** | LGGNet, AT-DGNN, DiffPool (EEG_10): tutte a chance; modello più semplice = migliore | Segnale classico di assenza di pattern semantico nel grafo |
| **HGNN a chance** | EEG_11: HGNN_2L, HGNN_2L_DYN, HGNN_ATT_2L tutti a 25.0% | Le relazioni di ordine superiore non creano informazione semantica laddove non esiste |
| **Subject-specific insufficiente** | EEG_06: mean test_bacc ≈ 25.0–25.6% su 10 soggetti (outlier: 34.4%) | ~330 trial/soggetto non bastano; segnale presente solo per alcuni soggetti |
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

### 2.1 Completato

#### [COMPLETATO] A1. Instance Normalization nei Modelli DL
- Implementata in EEG_05, EEG_06, EEG_07, EEG_08, EEG_09, EEG_10, EEG_11
- Risultato: nessun miglioramento rilevante vs. chance

#### [COMPLETATO] B. Classificazione su Categorie Semantiche (4-5 classi, schema concr4)
- Rieseguiti tutti i baseline su concr4 (4 classi)
- Risultato: tutti a chance (25.0%)

#### [COMPLETATO] G. Graph Attention Networks (GAT)
- Implementato in EEG_09 con GATConv(8 heads)
- Risultato: a chance. La domain adversarial training (EEG_09 v2) ha confermato il fallimento strutturale

#### [COMPLETATO] E2. DANN — Gradient Reversal su Subject Label
- Implementato in EEG_09 v1 (λ=0.1, patience=15) e v2 (λ=0.5, patience_adv=40)
- Risultato: rimozione segnale soggetto → collasso totale. SMOKING GUN: nessuna informazione semantica residua senza il segnale soggetto

#### [COMPLETATO] F. Hypergraph Neural Networks
- Implementato in EEG_11 (HGNN_2L, HGNN_2L_DYN, HGNN_ATT_2L)
- Basato su Feng et al. 2019, Li et al. 2025, AllSet (Chien et al. 2022)
- Risultato: tutti a chance (25.0%). Dimostrazione negativa rigorosa confermata da analisi strutturale EEG_07d

#### [COMPLETATO] Analisi Strutturale dei Grafi (EEG_07c/07d)
- Spettro Laplaciano identico tra categorie, ratio effetto soggetto/categoria = 8.6×
- Risultato: spiegazione matematica del fallimento di tutte le architetture GNN testorate

### 2.2 Azioni Immediate (da implementare)

#### A. Cross-Subject Contrastive Learning (Shen et al. 2022) — PRIORITÀ ALTA
- **Cosa**: addestrare un encoder con coppie positive cross-soggetto — stessa categoria semantica, soggetti diversi
- **Come**: coppie positive `(trial categoria C soggetto A, trial categoria C soggetto B)`, coppie negative per categoria diversa. Loss NT-Xent o SupCon. Con 70 soggetti abbiamo coppie abbondanti
- **Perché**: il DANN ha dimostrato che rimuovere l'informazione soggetto in modo diretto non funziona (collasso totale). Il contrastive learning forza l'encoder a trovare rappresentazioni soggetto-invarianti che preservino il contenuto semantico
- **Riferimento**: Shen et al. 2022, Zhao et al. 2023 (Thinking Race — specifico per imagined speech)
- **Effort**: medio

#### B. Meta-Learning Subject-Specific (MAML / Prototypical Networks) — PRIORITÀ ALTA
- **Cosa**: addestrare un meta-modello che si adatta rapidamente a un nuovo soggetto con pochi trial
- **Come**: episodic training su soggetti visti, test di adattamento su soggetti nuovi con k-shot (k=5–20 trial)
- **Perché**: EEG_06 mostra che il segnale esiste per alcuni soggetti (outlier 34.4%). Il meta-learning può sfruttare pochi trial per adattarsi al profilo EEG specifico del soggetto
- **Riferimento**: Finn et al. 2017 (MAML), Snell et al. 2017 (Prototypical Networks), He et al. 2021 (meta-EEG)
- **Effort**: alto

#### C. Allineamento Riemanniano (Euclidean Alignment)
- **Cosa**: riallineare le matrici di covarianza dei soggetti prima di ogni esperimento
- **Come**: `pyriemann.utils.mean.mean_riemann` per calcolare la covarianza media; trasformare ogni trial `X → R_s^{-1/2} * X`
- **Perché**: preprocessing geometricamente principiato che riduce la distanza inter-soggetto nello spazio delle matrici di covarianza, senza bisogno di modelli complessi
- **Riferimento**: Jayaram & Barachant 2020
- **Effort**: basso (preprocessing aggiuntivo)

#### D. Neural Prototype Clustering (contributo originale)
- **Cosa**: costruire prototipi neurali per categoria (media dei trial z-scored per soggetto), confrontare con gli schemi semantici tramite ARI/V-measure
- **Motivazione**: la domanda — "lo spazio EEG recupera parzialmente la struttura semantica a livello soggetto-specifico?" — rimane aperta. Pereira et al. 2018 lo hanno dimostrato per fMRI; l'analogo EEG non è stato pubblicato a grande scala
- **Come**: già implementato parzialmente in `EEG_00_labels_and_tasks.ipynb`
- **Effort**: basso (codice base già presente)

### 2.3 Sviluppi a Medio Termine

#### E. Domain Adaptation Avanzata
- **Cosa**: implementare strategie di adattamento di dominio più sofisticate di DANN
- **Come**:
  - Allineamento delle distribuzioni (MMD, Deep CORAL)
  - Adaptive Batch Normalization (Lee et al. 2022): aggiornare running stats BN con pochi trial del soggetto target
  - Subject-specific fine-tuning da un modello pre-addestrato cross-soggetto
- **Perché**: il DANN puro ha fallito; servono approcci che adattino il modello al soggetto target piuttosto che rimuovere l'informazione soggetto
- **Effort**: alto

#### F. EEG Data Augmentation
- **Cosa**: generare dati sintetici per espandere il dataset (specialmente per setting subject-specific)
- **Come**: noise injection, time warping, channel dropout, mixup, GAN-based
- **Perché**: ~330 trial/soggetto sono insufficienti; l'augmentation può aiutare il meta-learning e il fine-tuning
- **Effort**: medio

### 2.4 Sviluppi a Lungo Termine

#### G. Feature Tempo-Frequenza
- **Cosa**: aggiungere wavelet transforms, STFT, Hilbert-Huang Transform
- **Perché**: catturano transitori spettrali che il metodo di Welch non rileva
- **Effort**: medio

#### H. Feature Non-Lineari
- **Cosa**: aggiungere sample entropy, approximate entropy, dimensione frattale, esponente di Lyapunov
- **Perché**: catturano la complessità non-lineare del segnale EEG
- **Effort**: medio

#### I. Pipeline Automatizzata End-to-End
- **Cosa**: creare una pipeline completa dalla raw data alla predizione
- **Come**: script Python unico o pipeline con DVC/Prefect
- **Perché**: riproducibilità e facilità di sperimentazione
- **Effort**: medio-alto

#### J. Experiment Tracking
- **Cosa**: integrare MLflow o Weights & Biases
- **Perché**: tracciare sistematicamente iperparametri, metriche e artefatti
- **Effort**: basso-medio

---

## 3. Roadmap Aggiornata

### Fase 1: Completata (aprile 2026)
1. [FATTO] **Instance Normalization** in tutti i modelli DL — nessun miglioramento vs. chance
2. [FATTO] **Baseline DL end-to-end** (EEGNet, Conformer, ecc.) cross-subject concr4 — tutti a chance
3. [FATTO] **Subject-specific baseline** (EEG_06) — a chance con outlier 34.4%
4. [FATTO] **GCN ChebConv** su grafo PCC k-NN (EEG_08) — 1.02× chance
5. [FATTO] **GAT + Domain Adversarial** (EEG_09 v1/v2) — collasso totale senza segnale soggetto
6. [FATTO] **Advanced GNN** (LGGNet, AT-DGNN, DiffPool — EEG_10) — tutti a chance
7. [FATTO] **Hypergraph Neural Networks** (EEG_11) — tutti a chance
8. [FATTO] **Analisi strutturale grafi** (EEG_07c/07d) — dimostrazione matematica del fallimento

### Fase 2: Superare il Limite Strutturale (breve termine — PRIORITÀ ATTUALE)
1. **Allineamento Riemanniano** (pyriemann Euclidean Alignment) — preprocessing che riduce la distanza inter-soggetto nello spazio delle covarianze
2. **Cross-subject contrastive learning** (Shen et al. 2022) — sfrutta i 70 soggetti come vantaggio; approccio complementare al DANN
3. **Neural Prototype Clustering** — confronto EEG vs. semantica con ARI per capire se esiste struttura soggetto-specifica

### Fase 3: Meta-Learning e Adattamento Soggetto-Specifico (medio termine)
4. **Meta-learning** (MAML / Prototypical Networks) — sfrutta l'outlier EEG_06 come segnale: il segnale esiste per alcuni soggetti, il meta-learning può adattarsi rapidamente
5. **Subject-specific fine-tuning** da modello pre-addestrato cross-soggetto
6. **Data augmentation** per setting subject-specific (~330 trial/soggetto sono insufficienti)

### Fase 4: Sistema Completo (lungo termine)
7. Feature tempo-frequenza (wavelet, STFT) — dopo aver stabilito baseline contrastive
8. Pipeline automatizzata end-to-end
9. Validazione su dataset esterni (Nieto et al. Inner Speech)

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

| Obiettivo | Metrica | Target | Risultato Attuale |
|-----------|---------|--------|-------------------|
| Baseline vettoriale | bacc 110 classi | > 2% (2× chance) | ~Chance — non raggiunto |
| DL end-to-end cross-subject | bacc 4 classi | > 30% | ~25.0% — non raggiunto |
| DL subject-specific | bacc 4 classi | > 40% | ~25.6% (outlier: 34.4%) — non raggiunto |
| GNN subject-independent | bacc 4 classi | > 30% | ~25.0–25.5% — non raggiunto |
| GAT + adversarial | bacc 4 classi | > 30% | ~25.2% — non raggiunto; SMOKING GUN: collasso senza segnale soggetto |
| Hypergraph (riferimento Li et al.) | bacc 4 classi | avvicinare 78% | ~25.0% — non raggiunto; fallimento strutturale dimostrato |
| Contrastive cross-soggetto | bacc 4 classi | > 30% | da implementare |
| Meta-learning subject-specific | bacc 4 classi | > 35% | da implementare |

**Nota**: il mancato raggiungimento dei target non è un fallimento sperimentale — è un risultato scientifico rilevante. La dimostrazione matematica del fallimento strutturale (EEG_07d) e la conferma da SMOKING GUN (EEG_09 v2) sono contributi originali alla letteratura su imagined speech decoding.

## 6. Future Work — Meta-Learning

Il meta-learning soggetto-specifico emerge come la direzione più promettente dopo i risultati delle fasi 1–3:

- **Motivazione**: l'outlier EEG_06 (test_bacc=0.344 per un soggetto specifico con Deep4Net) indica che il segnale semantico esiste per alcuni soggetti, ma è altamente soggetto-specifico. L'architettura che si adatta rapidamente al profilo EEG del singolo soggetto può sfruttare questo segnale.
- **Approcci**: MAML (Model-Agnostic Meta-Learning, Finn et al. 2017), Prototypical Networks (Snell et al. 2017), TaskNorm per EEG (He et al. 2021)
- **Vantaggi del dataset**: 70 soggetti × 5 sessioni = 350 episodi per meta-training, struttura ideale per episodic learning
- **Schema proposto**: meta-training su soggetti 0–59, meta-test su soggetti 60–69 con k-shot adaptation (k=10–50 trial)

---

*Documento generato automaticamente dall'analisi completa del repository.*
