# CLAUDE.md — Istruzioni Persistenti per il Progetto

> Questo file guida Claude in ogni sessione di lavoro su questo repository.
> Leggilo sempre prima di fare qualsiasi cosa.

---

## 1. Contesto del Progetto

**Titolo**: Decodifica dell'Imagined Speech da Segnali EEG con Graph Neural Networks
**Autore**: Daniele Uras
**Istituzione**: Politecnico di Milano, DEIB
**Tipo**: Tesi Magistrale (2025-2026)

**Obiettivo**: Costruire un dizionario neurale semantico che mappi pattern EEG a categorie concettuali attraverso la decodifica dell'imagined speech, usando Graph Neural Networks e, in prospettiva, Hypergraph Neural Networks.

**Documenti di riferimento principali** — leggili sempre se hai dubbi sul progetto:

- `docs/PROGETTO_COMPLETO_IT.md` — tutto ciò che è stato fatto finora
- `docs/DIREZIONI_E_LIMITAZIONI.md` — roadmap, limitazioni, prossimi passi

### Vault del collega (stesso dataset)

Un collega lavora sullo **stesso identico dataset** (74 soggetti × 110 parole) da un angolo diverso: word-length decoding cross-subject. Il suo vault è in `/Users/danieleuras/Documents/ClaudeBrain-main` (symlinkato in Tesi-Wiki come `wiki/claudebrain-vault/`).

**REGOLA: quando Daniele fa una domanda di ricerca, confronta sempre i due approcci** — come l'ha affrontato lui (Tesi-Wiki) e come l'ha affrontato il collega (ClaudeBrain). Cerca punti di convergenza, divergenza, e metodologie riusabili. L'hub di tutti i collegamenti è `Tesi-Wiki/wiki/bridge/claudebrain.md`.

Note chiave del collega: `imagined-speech-ceiling-thesis` (8 linee per il data ceiling), `methodology-lessons-imagined-speech` (9 lezioni), `per-subject-vs-group-decoding-gap`, `subject-identification-brainid` (100% ID), `data-quality-headline-findings` (beta ucciso 98.86%), `hypergraph-eeg-exploration` (7 varianti NULL).
- `docs/papers_found.md` — letteratura di riferimento

---

## 2. Regola Fondamentale: Approccio Deep Learning

**L'approccio è deep learning end-to-end. Non serve feature engineering manuale.**

- Le 40 feature manuali (temporali, spettrali, funzionali) sono state usate **solo come baseline esplorativo**
- Il modello deve imparare rappresentazioni direttamente dal **segnale EEG grezzo** preprocessato
- Input atteso per i modelli DL: tensori `(n_trials, 59, n_samples)` — canali × campioni temporali
- Non proporre mai feature engineering manuale come soluzione principale

---

## 3. Ambiente di Sviluppo

Due ambienti conda disponibili:

| Parametro | `daniele_310` | `daniele_311` |
|-----------|---------------|---------------|
| **Python** | 3.10 | 3.11.15 |
| **PyTorch** | 2.8.0 | 2.5.0 |
| **Braindecode** | 1.2.0 (no CBraMod) | 1.3.2 (con CBraMod) |
| **PyG** | 2.7.0 | 2.7.0 |
| **MNE** | 1.10.2 | 1.11.0 |
| **Porta JupyterLab** | 8888 | 8889 |
| **Launch config** | `jupyter-lab` | `jupyter-lab-311` |

**Env raccomandato per nuovi notebook DL**: `daniele_311` (include CBraMod)

Per avviare JupyterLab 311: usa `preview_start` con il server `jupyter-lab-311`.
Per il token: `source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate daniele_311 && jupyter server list`

- **Non usare** `daniele_dl_thesis` come env — usare `daniele_310` o `daniele_311`

---

## 4. Dataset

- **Segnale**: EEG a 61 canali, 256 Hz, epoche ~1.5s (384 campioni)
- **Task**: 110 parole immaginarie (chance level ~0.9%)
- **Soggetti**: ~70, con 5 sessioni ciascuno
- **Clustering semantico**: `concr4` (4 categorie semantiche) e `gram4` (Verbi/Sostantivi/Aggettivi/Altro)

### ⚠️ REGOLA FONDAMENTALE: Sorgente Dati per i Modelli GNN

**I notebook EEG_09/10/12/13 leggono ESCLUSIVAMENTE dai CSV grezzi di Paolo.**
**NON usare mai i file H5 (`data/processed/`) — sono abbandonati.**

```
data/raw_csv/training_set/
    PXXX_SYYY/              ← soggetto X, sessione Y
        accendere_img.csv   ← shape (61, 384): righe=canali, colonne=samples
        acqua_img.csv
        ...                 ← 110 file CSV per sessione
```

- Caricamento: `pd.read_csv(path, header=None).values` → `(61, 384)` float32
- Nessun header, nessun indice: solo valori numerici
- Split subject-independent: SUBJ_TRAIN=0–49, SUBJ_VAL=50–59, SUBJ_TEST=60–73

### Mapping parole → label

- `configs/label_schemes/label2idx.json` — word → label_idx (0–109)
- `configs/label_schemes/labelid2cluster_concr4.json` — label_idx → cluster_id
- `configs/label_schemes/labelid2cluster_gram4.json` — label_idx → cluster_id (grammaticale)

**Finding critico**: i trial si raggruppano per **soggetto**, non per parola. La variabilità inter-soggetto domina (ε²=0.85 per soggetto vs 0.03 per parola).

---

## 5. Struttura Repository

```
miralis-hypergraph-imagined-speech/
├── CLAUDE.md                          ← questo file
├── docs/
│   ├── PROGETTO_COMPLETO_IT.md        ← AGGIORNA dopo ogni sessione
│   ├── DIREZIONI_E_LIMITAZIONI.md     ← AGGIORNA quando la roadmap cambia
│   ├── papers_found.md                ← letteratura di riferimento
│   ├── presentation_progetto.pptx     ← presentazione stato progetto
│   └── checkpoints/                   ← checkpoint datati delle sessioni
├── notebooks/                         ← PATHWAY A: strada principale tesi (GNN → Hypergraph)
│   ├── EEG_05_braindecode_raw_baseline_indipendant.ipynb
│   ├── EEG_06_subject_specific.ipynb
│   ├── EEG_07_cv_subject_independent.ipynb
│   ├── EEG_08_..._GNN.ipynb
│   ├── EEG_09_gat_subject_independent.ipynb
│   ├── EEG_10_hypergraph_nn.ipynb
│   ├── EEG_11_hgnn_subject_independent.ipynb
│   ├── EEG_12_subject_specific_gnn.ipynb
│   ├── EEG_13b_dhslp_subject_specific.ipynb  ← DHSLP soggetto-specifico, fix overfitting
│   ├── EEG_14_dhslp_pretrain_finetune.ipynb  ← pretrain S-indep + calibrazione per sogg.
│   ├── EEG_15_li_trial_hypergraph.ipynb      ← Li et al. 2025 fedele (⚠️ non ancora eseguito)
│   ├── EEG_16_subject_clustering.ipynb       ← clustering strutturale su adj pruned (v1)
│   ├── EEG_16b_subject_clustering_v2.ipynb  ← clustering su 1830-dim upper triangle, ipergrafo, XAI
│   ├── EEG_17_cluster_connectivity.ipynb     ← connettività media per cluster semantico
│   ├── EEG_18_functional_clustering.ipynb    ← clustering funzionale 4D→305D→1041D
│   ├── EEG_19_session_clustering.ipynb       ← test-retest: stabilità fenotipi tra sessioni
│   ├── EEG_20_dhslp_cluster_specific.ipynb  ← DHSLP cluster-specific C0/C1 (null result)
│   ├── EEG_21_subject_stability_analysis.ipynb ← stabilità top soggetti, per-class, learning curve
│   ├── EEG_22_trial_confidence_analysis.ipynb  ← band power trial-by-trial: firma spettrale C0 vs C1
│   ├── ...
│   ├── pathwayB/                      ← PATHWAY B: engineering (augmentation, domain adaptation)
│   │   ├── README.md
│   │   ├── B01_data_augmentation.ipynb
│   │   ├── B02_domain_adaptation_session.ipynb
│   │   ├── B03_domain_adaptation_subject.ipynb
│   │   └── B04_contrastive_learning.ipynb
│   └── tests/                         ← esperimenti baseline
├── scripts/
│   ├── features/                      ← pipeline feature (solo baseline)
│   ├── graphs/                        ← visualizzazioni e viewer EEG
│   └── utils.py
├── figures/                           ← plot generati
├── data/
│   ├── processed/                     ← HDF5 preprocessati
│   └── interim/                       ← tensori .pt e CSV feature
└── src/                               ← Dockerfile e script avvio
```

---

## 6. Stato Attuale degli Esperimenti

Vedere `docs/PROGETTO_COMPLETO_IT.md` per il dettaglio completo. Riepilogo:

| Modello | Subject-Specific | Subject-Independent |
|---------|-----------------|---------------------|
| Logistic Regression / MLP | ~Chance | ~Chance |
| GCN Grafo Statico (k-NN) | ~Chance | ~Chance |
| GCN Feature-Similarity | ~2x Chance ⚠️ | ~Chance |
| GCN Spazio-Temporale | completato | completato |

### Risultati recenti (maggio 2026)

**EEG_12/13b — DHSLP soggetto-specifico con fix overfitting**
- Modifiche: hidden 128→64, dropout 0.3→0.6, WD 1e-3→5e-3, label smoothing 0.1→0.2, mixup α=0.4
- Gap train/val: 0.67 → 0.04
- Migliori soggetti: ~34% bAcc (4 classi, chance=25%)

**EEG_14 — Pretrain subject-independent + calibrazione per soggetto**
- PRE-CAL: 26.1% mean bAcc (modello pretrained valutato direttamente)
- POST-CAL: 26.3% mean bAcc (fine-tune sess. 1-4, test sess. 5) — Δ≈0, 7/13 soggetti migliorati
- Conclusione: pretrain generalizza appena sopra chance; fine-tuning non aiuta sistematicamente

**EEG_15 — Li et al. 2025 fedele sui nostri dati**
- Split 300/200/50 (usa tutti i 550 trial), grid 3087 config, pipeline identica al paper
- ⚠️ Non ancora eseguito — risultati da aggiornare dopo run

**EEG_16 — Clustering strutturale su adj pruned**
- Feature: matrice adj pruned 61×61 per soggetto, PCA→KMeans(k=2): 52+22 soggetti
- Mann-Whitney U vs bAcc: **p=0.800 (ns)** — struttura topografica NON predice decodificabilità

**EEG_16b — Clustering soggetti su upper triangle completo (1830-dim)**
- Feature: upper triangle abs_pcc 61×61 = 1830-dim, media su tutti i trial (~550 × 5 sessioni)
- P022 escluso come outlier singleton confermato da PCA
- KMeans(k=2) su PCA 20 componenti: **C0 ~36 sogg (fronto-motor)** · **C1 ~37 sogg (fronto-occipital)**
- Silhouette ipergrafo=0.284, silhouette grafo=0.240 → ipergrafo superiore come clustering
- XAI Cohen's d: coppia top F3-PO8 d=−5.4; regione più discriminante: frontale (mean|d|=0.861)
- Permutation test N=5000: 0/5000 shuffle raggiunge d osservato → **non circolare**
- Cross-metric ARI: ARI(abs_pcc, PLV)=1.00 · ARI(abs_pcc, wPLI)=0.95 → struttura robusta
- Signal quality: varianza p=0.504 ns · kurtosi p=0.170 ns · gamma p=0.068 ns (borderline)
- **Null result bAcc**: Mann-Whitney p=0.800 (ns) — connettività strutturale ≠ performance IS

**EEG_19 — Test-retest: stabilità fenotipi tra sessioni**
- 73 soggetti × 5 sessioni = 365 vettori; stessa feature di EEG_16b (1830-dim abs_pcc)
- Ratio intra/inter soggetto = 0.425 → sessioni stesso soggetto molto più simili
- NMI subject recovery = 0.946 (vs random 0.618); 56/73 sogg (77%) perfetti 5/5 sessioni
- ARI fenotipo C0/C1 = 0.933 · concordanza = 98.4% → fenotipo EEG_16b presente in ogni sessione
- F3-PO8: p<0.05 in 5/5 sessioni → marker C1 stabile temporalmente
- **Conclusione**: i fenotipi sono caratteristica del soggetto, non artefatto della media; nessun drift temporale

**EEG_20 — DHSLP cluster-specific (C0 vs C1)**
- Ipotesi: modelli DHSLP addestrati separatamente su C0 e C1 specializzano le iperedge apprese sul fenotipo corrispondente
- Config potenziata vs EEG_13b: N_EDGES=32, HIDDEN=256, DROPOUT=0.3, WD=3e-3, MAX_EPOCHS=150, PATIENCE=25
- **Risultato**: tutti a chance — Model_C0 su C0_TEST=0.2510, su C1_TEST=0.2512; Model_C1 su C0_TEST=0.2500, su C1_TEST=0.2500
- Matrice 2×2 completamente piatta; diagonale NON > off-diagonal → specializzazione non confermata
- XAI topomaps: aree temporali/centrali (PO7, AF4, C2) — non i pattern fronto-motor/fronto-occipital attesi
- Solo 15% soggetti sopra chance in entrambi i modelli
- **Root cause**: ~25 soggetti train per cluster insufficienti per subject-independent learning (EEG_13b usa ~50)
- **Null result utile per tesi**: "cluster-specific training non migliora su S-Indep — il bottleneck è la scarsità di dati, non l'eterogeneità fenotipica"

**EEG_22 — Band power trial-by-trial: firma spettrale C0 vs C1**
- Per ogni sogg top (10 per cluster), inference DHSLP (checkpoint EEG_13b), Welch PSD per trial × elettrodo
- Mann-Whitney U + Cohen's d tra trial corretti e sbagliati
- **C0 (Fronto-motor)**: 1012 trial, 256 corretti (25.3%)
  - ALPHA dominante: 13/61 sig, max|d|=0.248 (corretti → più alpha frontale = motor preparation)
  - GAMMA inverso: 11/61 sig, **corretti hanno MENO gamma posteriore** (PO7 d=-0.205, FT7 d=-0.183, F7 d=-0.170) — soppressione rete visuo-spaziale
- **C1 (Fronto-occipital)**: 1081 trial, 298 corretti (27.6%)
  - THETA soppresso dominante: 21/61 sig, max|d|=0.225 (corretti hanno meno theta globale = stato cognitivo ottimale)
  - GAMMA occipitale: 2/61 sig **positivo** (P7 d=+0.135, O1 d=+0.123) — network F3-PO8 attivo
- **Firma opposta**: C0 sopprime gamma posteriore, C1 lo amplifica → meccanismi neurali distinti per fenotipo
- Primo notebook con effetto direzionale reale: biologicamente plausibile e topograficamente coerente con EEG_16b

**EEG_17 — Connettività media per cluster semantico**
- DEV[sogg,k] = deviazione individuale dalla grand mean connettività della popolazione
- Finding: deviazione è firma individuale stabile, non correlata a bAcc

**EEG_18 — Clustering funzionale (4D → 305D → 1041D)**
- 4 scalari (§1–§4): sil=0.18, KW p=0.93 (ns)
- 305D multi-band power (§10 A): sil=0.32, KW p=0.074 (ns)
- 732D Li gamma-temporal (§10 B): sil=0.624, KW p=0.023★ ma cluster 70+3 = outlier detection
- 1041D combined (§10 C): sil=0.589, KW p=0.023★ — stessa struttura outlier
- ARI strutturale vs funzionale ≈ 0 — ortogonali
- **Risultato negativo robusto**: 1041D di feature esaurite, nessuna predice bAcc

---

## 7. Roadmap Attiva

### ✅ Completati (maggio 2026)

- **EEG_13b**: fix overfitting subject-specific (gap train/val 0.67→0.04)
- **EEG_14**: pretrain S-indep + calibrazione per soggetto (26.1%→26.3% bAcc, Δ≈0)
- **EEG_15**: Li et al. 2025 fedele implementato (⚠️ non ancora eseguito)
- **EEG_16**: clustering strutturale su adj pruned (p=0.800 ns vs bAcc)
- **EEG_16b**: clustering 1830-dim con XAI, permutation test, cross-metric ARI, test-retest ready
- **EEG_17**: connettività media per cluster semantico (firma individuale, ns vs bAcc)
- **EEG_18**: clustering funzionale 4D→305D→1041D (tutto ns, ARI=0 vs strutturale)
- **EEG_19**: test-retest stabilità fenotipi (NMI=0.946, ARI=0.933, concordanza 98.4%)
- **EEG_20**: DHSLP cluster-specific (null result — ~25 sogg/cluster insufficienti per S-Indep)
- **EEG_22**: band power trial-by-trial C0 vs C1 — primo risultato direzionale reale (firme spettrali opposte)

### 🎯 Prossimi passi

1. **[IMMEDIATO]** EEG_13b long run — MAX_EPOCHS=300, PATIENCE=70; cerca il soffitto reale dei migliori soggetti
2. **[IMMEDIATO]** Eseguire EEG_15 sui dati completi — Li et al. 2025 fedele
3. **[BREVE]** Scrittura tesi — il quadro sperimentale inter-soggetto è esaurito
   - Cap. variabilità inter-soggetto: null results EEG_16b/EEG_18/EEG_20 SONO il contributo
   - Cap. metodi: EEG_13b, EEG_14, EEG_15, EEG_16b, EEG_19
4. **[BREVE]** Focus intra-soggetto: confronto sistematico W-HGNN vs DHSLP vs Li et al. per i migliori soggetti
5. **[MEDIO]** Dizionario neurale semantico dai migliori soggetti

---

## 8. Convenzioni di Codice

- **Lingua**: commenti e documentazione in **italiano**; codice (variabili, funzioni) in **inglese**
- **Notebook Pathway A**: prefisso `EEG_XX_` — pipeline principale tesi
- **Notebook Pathway B**: prefisso `B0X_` in `notebooks/pathwayB/` — engineering
- **Commit**: seguire lo stile esistente (`feat:`, `docs:`, `refactor:`)
- **Branch corrente**: `claude/elegant-neumann`
- **Nessuna feature engineering manuale** nei nuovi modelli

### 8.1 Regola Fondamentale: Experiment Tracking con W&B

**Ogni nuovo notebook di training DEVE integrare Weights & Biases (wandb.ai).**

- **Entity**: `uras-daniele22-politecnico-di-milano`
- **Project**: `miralis-imagined-speech`
- **Una run per modello** — `wandb.init(..., reinit=True)` dentro il loop sui modelli
- **Nome run**: `eegXX_{model_name}_{cluster_scheme}` (es. `eeg11_HGNN_2L_concr4`)

**Cosa loggare obbligatoriamente:**
```python
# Config (in wandb.init)
config = {
    "notebook": "EEG_XX_...", "model": model_name,
    "n_classes": N_CLASSES, "cluster_scheme": CLUSTER_SCHEME,
    "lr": LR, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
    "n_train_subj": len(SUBJ_TRAIN), ...
}

# Per ogni epoca (in training loop)
run.log({"train/loss": ..., "train/acc": ..., "val/bacc": ..., "lr": ..., "epoch": epoch})

# Summary finale
run.summary["val_bacc"]  = val_bacc
run.summary["test_bacc"] = test_bacc
run.log({"confusion_matrix": wandb.plot.confusion_matrix(...)})
run.finish()
```

**Installazione** (una tantum su ogni VM/env):
```bash
pip install wandb
wandb login  # inserire API key da wandb.ai/settings
```

---

### 8.2 Regola Fondamentale: Graph Classification

**Nei notebook GNN (EEG_08, EEG_09, EEG_10, EEG_11 e tutti i futuri), il paradigma è sempre GRAPH CLASSIFICATION:**

- **1 grafo per trial** — edge_index calcolato on-the-fly per ogni trial nel metodo `get()` del dataset
- **~38K grafi totali** — 70 soggetti × 5 sessioni × 110 parole
- **MAI usare un grafo statico condiviso** tra trial (né globale per tutti i soggetti, né per soggetto)
- Il grafo statico (`build_pcc_graph` o equivalente) va usato SOLO per display/log/topomap

Implementazione corretta in PyG:
```python
# ✓ CORRETTO — graph classification
def get(self, idx):
    x_np = load_trial(idx)
    edge_index = pcc_to_edge_index_trial(x_np, k=self.k)  # per-trial!
    return Data(x=x, edge_index=edge_index, y=label)

# ✗ SBAGLIATO — grafo condiviso (non è graph classification)
def get(self, idx):
    x = load_trial(idx)
    return Data(x=x, edge_index=self.shared_edge_index, y=label)
```

---

## 9. Obbligo di Aggiornamento Documenti

**Dopo ogni sessione di lavoro significativa, Claude DEVE:**

### 9.1 Aggiornare `docs/PROGETTO_COMPLETO_IT.md`

- Aggiungere i nuovi esperimenti nella sezione 4 (Esperimenti Baseline)
- Aggiornare il riepilogo risultati
- Aggiornare gli insight chiave se emergono nuove conclusioni
- Aggiornare la data in cima al documento

### 9.2 Aggiornare `docs/DIREZIONI_E_LIMITAZIONI.md`

- Spostare le azioni completate (da "Cosa Possiamo Fare" a risultati)
- Aggiornare le limitazioni se ne vengono superate
- Aggiornare le metriche di successo con i risultati reali
- Aggiornare la data

### 9.3 Creare un checkpoint datato

- Creare `docs/checkpoints/GG-MM-AA.md` con il riepilogo della sessione
- Includere: cosa è stato fatto, risultati numerici, conclusioni, prossimo step

### 9.4 Commit e Merge

- Fare sempre commit dopo aggiornamenti significativi
- Usare `docs:` o `feat:` come prefisso a seconda del tipo di modifica
- **Fare sempre merge su `main` dopo ogni sessione**, salvo indicazione esplicita contraria da parte di Daniele
- Il merge va fatto con `git checkout main && git pull origin main && git merge claude/elegant-neumann --no-ff && git push origin main`

---

## 10. Riferimenti Chiave dalla Letteratura

| Paper | Rilevanza |
|-------|-----------|
| Li et al. 2025 — DHSLP/DHSLF | TARGET: 78% accuracy con hypergraph dinamico su imagined speech |
| Einizade et al. 2022 — GraphIS | Punto di partenza GCN/GSP per imagined speech |
| Lawhern et al. 2018 — EEGNet | Baseline end-to-end da implementare subito |
| Song et al. 2023 — EEG Conformer | Architettura CNN+Transformer per segnale raw |
| Feng et al. 2019 — HGNN | Fondamenta teoriche degli hypergraph |
| Chien et al. 2022 — AllSet | Framework flessibile per hypergraph |
| Bomatter et al. 2024 — Instance Norm | 1 riga di codice per migliorare cross-subject |
| Shen et al. 2022 — Contrastive EEG | Contrastive learning subject-invariant |

Vedere `docs/papers_found.md` per la lista completa con abstract e valutazione.

---

## 11. Convenzioni di Visualizzazione

### Matrici di connettività
**Regola fissa**: ogni volta che si plotta una matrice di connettività (adj, PCC, wPLI, PLV, ecc.) gli elettrodi vanno **sempre ordinati per regione cerebrale**:

```
Frontale → Temporale → Centrale → Parietale → Occipitale
(AF*, F*, FT*, FC*) → (T*, TP*) → (C*, CP*) → (P*, PO*) → (O*)
```

Questo rende il pattern immediatamente leggibile senza dover cercare gli elettrodi singolarmente.

---

## 12. Wiki Knowledge Base (Obsidian)

**Path vault**: `/Users/danieleuras/Documents/Tesi-Wiki`

Quando hai bisogno di contesto non presente in questo progetto:
1. Leggi `wiki/hot.md` (contesto recente, ~500 parole)
2. Se non basta, leggi `wiki/index.md`
3. Per specifiche aree: `wiki/concetti/`, `wiki/esperimenti/`, `wiki/papers/`

**`/save` alla fine di ogni sessione** — la skill analizza la conversazione e archivia insights, decisioni e risultati nel vault come note strutturate. Aggiorna automaticamente `index`, `log` e `hot`.

Non leggere il wiki per domande generiche di coding — solo per contesto specifico della tesi.

---

## 13. Note Operative

- **Git worktree attivo**: `.claude/worktrees/elegant-neumann/` — lavora sempre qui
- **PR aperta**: #8 su GitHub
- **JupyterLab**: avviare con `preview_start jupyter-lab-311` per nuovi notebook DL (CBraMod), `jupyter-lab` per notebook legacy
- **Non usare** `daniele_dl_thesis` come env — usare `daniele_310` (legacy) o `daniele_311` (nuovi notebook)
- **Non committare** `node_modules/`, file `.pt` di grandi dimensioni, dati grezzi

---

## 12. Narrativa Scientifica della Tesi

**Il quadro è chiaro (maggio 2026).** Abbiamo esaurito sistematicamente le spiegazioni semplici della variabilità inter-soggetto EEG. I risultati negativi SONO il contributo scientifico.

### Cosa abbiamo trovato
- Struttura topografica (connettività pruned) NON predice decodificabilità: p=0.800
- Feature funzionali 1041D (band power + Li gamma-temporal) NON predicono bAcc: tutti ns
- Clustering strutturale e funzionale sono ortogonali: ARI≈0
- Fine-tuning da pretrain non aiuta sistematicamente: Δ≈0 su 13 soggetti
- Modelli subject-specific migliori arrivano a ~34% bAcc (4 classi, chance=25%)
- **EEG_16b**: due fenotipi neurali reali (C0 fronto-motor / C1 fronto-occipital), robusti a metrica e temporalmente stabili — ma NON correlati a bAcc
- **EEG_19**: i fenotipi sono proprietà del soggetto presenti in ogni singola sessione (NMI=0.946, concordanza 98.4%)

### Cosa significa
La variabilità inter-soggetto nella decodifica IS non è spiegata da:
- Topografia della connettività EEG (struttura statica del grafo)
- Pattern di potenza spettrale multi-banda (5 bande × 61 canali)
- Feature temporali gamma (Li et al. 2025, 12 feature × 61 canali)
- Combinazioni di tutto quanto sopra
- Fenotipo strutturale C0/C1 (confermato stabile nel tempo via EEG_19)

**Nota metodologica chiave (EEG_16b)**: analisi robustificata da permutation test N=5000 (anti-circular), cross-metric ARI≈1 (anti-artefatto metrica), signal quality ns (anti-noise). Il risultato negativo bAcc è quindi solido.

### Implicazione per la tesi
- Presentare il clustering negativo come "esaurimento sistematico" — più forte di un singolo null result
- I migliori soggetti (34% bAcc) esistono e sono riproducibili → modelli intra-soggetto hanno senso
- Il dizionario neurale semantico è realizzabile solo per soggetti ad alta capacità IS
- EEG_16b + EEG_19 = contributo metodologico in sé: caratterizzazione fenotipi neurali per IS

### EEG_22: primo risultato positivo — firma spettrale dei trial decodificabili
- Domanda: i trial che DHSLP classifica correttamente hanno una firma EEG diversa da quelli sbagliati?
- C0 (Fronto-motor): sì — alpha sync frontale + soppressione gamma occipitale (de-attivazione rete visiva durante IS motorio)
- C1 (Fronto-occipital): sì — theta soppresso globale (stato cognitivo ottimale) + gamma occipitale amplificato (network F3-PO8 attivo)
- Le ipotesi strutturali di EEG_16b trovano conferma dinamica in EEG_22: la firma strutturale del fenotipo lascia traccia nei trial
- **Non è un null result**: è il primo effetto direzionale biologicamente interpretabile dell'intero progetto

### EEG_20: ultimo tentativo inter-soggetto
- Ipotesi testata: modelli DHSLP separati per fenotipo C0/C1 specializzano le iperedge apprese
- Risultato: tutto a chance (0.2505), matrice 2×2 piatta, XAI topomaps non discriminative
- Causa: ~25 soggetti train per cluster troppo pochi — il segnale IS non è apprendibile S-Indep con così pochi soggetti
- **Conferma definitiva**: la variabilità inter-soggetto non si risolve né con clustering strutturale, né con clustering funzionale, né con modelli fenotipo-specifici

### Cosa NON fare
- Non proporre ulteriore clustering inter-soggetto su feature EEG statiche — già esaurito
- Non proporre modelli S-Indep cluster-specific — già testato (EEG_20), null result
- Non aspettarsi che un modello subject-independent superi chance su questo dataset
- Non proporre feature engineering manuale come soluzione principale
- Non concludere che C0/C1 implichi diversa capacità IS — il null result bAcc lo esclude
