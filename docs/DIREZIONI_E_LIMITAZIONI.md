# Direzioni Future e Limitazioni Attuali

> Ultimo aggiornamento: 24 giugno 2026

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

### 2.0 ✅ COMPLETATO: Subject Clustering EEG-First (EEG_08c)

> **Contesto**: l'approccio precedente (usare accuracy come proxy per "capacità IS") è stato chiuso da Francesco. La varianza delle accuracies tra soggetti (21–27% in concr4) è troppo bassa e troppo vicina al baseline per essere un segnale primario.

**Implementato in EEG_08c** (18 celle):
- Feature matrix 70 soggetti × 367d (band power + degree + ISC)
- KMeans k=2..6 + Silhouette + Elbow → k=3 ottimale
- Dendrogramma Ward + t-SNE + PCA 2D
- Post-hoc accuracy overlay + Kruskal-Wallis + Spearman

**Risultato**: **null result empirico** — KW p=0.939, silhouette 0.253. Le feature EEG statiche non separano soggetti per decodificabilità. Cluster k=3 = 3 outlier EEG (P017, P055, P063) con accuracy media. Francesco aveva ragione: la variabilità EEG inter-soggetto è reale ma non predice la performance del classificatore.

**Implicazione per la tesi**: non ha senso segmentare l'analisi per "capacità IS" — i modelli devono essere valutati sull'intera popolazione senza pre-selezione soggetti.

### 2.1 Azioni Immediate (pronte con l'infrastruttura attuale)

#### A-1. ✅ HGNN Subject-Specific LOSO (EEG_09b) — COMPLETATO
- Architettura HGNN (Feng 2019) su ipergrafi pruned
- 74 soggetti, LOSO split — top: P031 31.8%, mediana ~25%, ~47% sopra chance
- BCI illiteracy confermata: ~53% soggetti a/sotto chance anche con HGNN
- Bug risolti: N_CHANNELS mancante, H shape mismatch, wandb MailboxClosedError

#### A0. ✅ HGNN Subject-Independent Ablation (EEG_09) — COMPLETATO
- 5 metriche × pruned only → tutti a chance (test 0.253–0.260)
- Spread < 1% tra metriche — nessuna metrica si distingue
- **Conclusione**: ipergrafo statico pruned = grafo semplice in S-Indep → serve modellazione temporale

#### A1b. ✅ T-HGNN: Temporal Encoder + HGNN (EEG_10) — COMPLETATO
- CNN 1D per-nodo (384→64) + HGNN 2L, S-Indep
- Risultato: 0.241–0.252 test bAcc — *peggio* di EEG_09
- Causa: CNN overfitta la compressione temporale in S-Indep
- Lezione: compressione CNN prima di HGNN non generalizza cross-subject

#### A1c. ✅ W-HGNN: Windowed HGNN con H fisso (EEG_11) — COMPLETATO
- K=8 finestre, H_pruned soft fisso, feat nodo = Linear(48→32) per finestra
- Risolve tautologia H_mask ⊙ |PCC(x_k)| — H_pruned già PCC-based
- Risultato: 0.249–0.261 test bAcc — pari a EEG_09 (nessun guadagno)
- Conferma: bottleneck è variabilità inter-soggetto, non architettura temporale

#### A1d. ✅ W-HGNN Subject-Specific LOSO (EEG_12) — COMPLETATO
- Replica EEG_09b con W-HGNN, 74 soggetti, LOSO split
- Best: P007=0.344 (nuovo record, +8.2% vs P031=0.318 di EEG_09b)
- Delta medio vs HGNN statico: −0.004 (≈ zero)
- Soffitto attuale: ~34% su 4 classi, ~50% soggetti sotto chance (BCI illiteracy)
- Gap con target tesi (Li et al. 2025 ~78%): enorme — necessario salto architetturale

#### A0'. Subject-Centering del Segnale EEG (priorità assoluta)
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

#### F. ✅ Hypergraph Neural Networks (EEG_09–12) — COMPLETATO (topologia fissa)
- HGNN statico, T-HGNN, W-HGNN implementati e testati (EEG_09–12)
- Risultato: soffitto ~34% S-Spec, ~26% S-Indep — nessun breakthrough

#### F2. ✅ DHSLP/DHSLF — Dynamic Hypergraph con Iperedge Apprese (EEG_13/14/15 — COMPLETATO)
- Implementato in EEG_13 (S-Spec LOSO), EEG_14 (pretrain + calibrazione), EEG_15 (Algorithm 1 Li et al.)
- **Record assoluto**: P070 bAcc=**0.393** (C1, fronto-occipital) — primo soggetto a superare 0.39
- **Conclusione**: il soffitto subject-specific con DHSLP è ~39% per i migliori soggetti, ~25% mediana

#### F3. ✅ Scoperta Fenotipi Neurali C0/C1 (EEG_16b/17 — RISULTATO TESI)
- **Svolta concettuale**: cluster per connettività soggetto → 2 fenotipi stabili (ARI=0.933 sessione, ARI=0.796 trial)
- C0 (n=36): fronto-motor, imagery articolatoria — proficiency=STATO
- C1 (n=37): fronto-occipital, imagery visuo-linguistica — proficiency=TRATTO
- Validato da: permutation test, cross-metric PLV, firme spettrali differenziate (EEG_22/23)
- Split-half reliability (EEG_25): C0 r=−0.36 (stato), C1 r=+0.45 CI esclude zero (tratto)

#### F4. ✅ cVAE + Mixup Augmentation (EEG_28 — COMPLETATO)
- cVAE condizionale per generazione trial sintetici — plausibile ma guadagno marginale
- Mixup augmentation su trial stesso-label — test rapido, beneficio limitato
- Conclusione: il bottleneck è variabilità inter-soggetto, non dataset size

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

### Fase 0–2: ✅ COMPLETATE (maggio–giugno 2026)
- ✅ Subject clustering EEG-first (EEG_08c) — null result empirico
- ✅ HGNN ablation completa (EEG_09–12) — soffitto ~34% S-Spec
- ✅ DHSLP implementato (EEG_13–15) — best P070=0.393
- ✅ Scoperta fenotipi C0/C1 (EEG_16b/17/19) — ARI=0.933 sessione, 0.796 trial
- ✅ Caratterizzazione fenotipi (EEG_22/23/24/25) — firme spettrali, PLV, tratto/stato
- ✅ cVAE + mixup (EEG_28) — generazione sintetica, guadagno marginale

### Fase 3: Esperimenti Tesi da Eseguire (prossima settimana — server)
1. **EEG_30** — Trial-level decodability prediction (feature dinamiche, nested CV)
2. **EEG_31** — Riemannian alignment (pyriemann, re-fenotyping in tangent space)
3. **EEG_32** — CSD rephenotyping (Laplaciano spaziale, robustezza fenotipi)
4. **EEG_33** — Phenotype switch decodability (Cohen's d trial-level, confound control)
5. **EEG_34** — Information ceiling (MINE, Fano inequality, margine teorico)

### Fase 4: Scrittura Tesi (giugno–luglio 2026)
- Cap. 4: Fenotipi neurali — 6 livelli di evidenza (EEG_16b–25)
- Cap. 5: Modelli di decodifica — DHSLP best P070=0.393
- Cap. 6: Esperimenti di caratterizzazione (EEG_30–34)
- Cap. 7: Discussione — dual-stream, tratto/stato, ceiling informazionale

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
| Baseline vettoriale | bAcc 4 classi | > 25% | ✅ ~25% (EEGNet) |
| GNN subject-independent | bAcc 4 classi | > 26% | ✅ ~26% (DHSLP S-Indep) |
| GNN subject-specific mediana | bAcc 4 classi | > 30% | ⚠️ ~25% mediana |
| GNN subject-specific top | bAcc 4 classi | > 35% | ✅ **0.393** (P070, EEG_13b) |
| Fenotipi stabili | ARI sessione | > 0.8 | ✅ **0.933** (EEG_19) |
| Fenotipi stabili trial | ARI trial | > 0.5 | ✅ **0.796** (EEG_19 §9-13) |
| Hypergraph (ref. Li et al.) | bAcc comparabile | avvicinare 78% | ⚠️ 39% best soggetto |
| Ceiling informazionale | MI → max bAcc | da stimare | 🔄 EEG_34 §7 (null+CI per permutazione, in corso) |
| Replica fenotipi (coorte indipendente) | esistono su dataset nuovo | sì/no | ✅ **sì** (V5W: Riemann sil 0.376 p<0.001; cross-coorte r=0.895) |
| Fenotipi geometria-invarianti | Riemann = tesi |PCC| | match | ✅ **ARI 1.0** (V5W_09) |

---

## 6. Stato Scrittura (giugno 2026)

L'analisi è sostanzialmente completa. Pronti a scrivere:
- **Cap. 4 (Decoding)** — chance, blindato (DHSLP, ML Iacomi, Riemann, 2 dataset).
- **Cap. 6 (Fenotipi)** — validato da ogni angolo (V5W: geometria-invariante ARI 1.0, coorte-invariante r=0.895 + cross-modale 0.73, robusto agli artefatti).

Ultimo tassello analitico aperto:
- **Cap. 5 (Ceiling)** — MI/Fano: aggiunto il null per permutazione + CI (EEG_34 §7) per blindare "MI≈0" contro il bias dello stimatore. Da runnare sul server.

Minori (gestibili in scrittura): robustezza ε² alla normalizzazione; framing BCI-literate / concr4.

---

*Documento generato automaticamente dall'analisi completa del repository.*
