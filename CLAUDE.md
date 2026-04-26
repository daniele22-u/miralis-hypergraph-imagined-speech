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
│   ├── EEG_08_..._GNN.ipynb           ← prossimo: GCN su grafo elettrodico
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
| GCN Spazio-Temporale | 🔄 In corso | 🔄 In corso |

---

## 7. Roadmap Attiva

Due percorsi paralleli. Vedere `docs/DIREZIONI_E_LIMITAZIONI.md` per il dettaglio completo.

### 🎓 Pathway A — Tesi (cartella `notebooks/`)
Priorità principale: contributo scientifico della tesi.

1. **[IMMEDIATO]** EEG_07: 5-fold CV subject-independent → validazione baseline
2. **[IMMEDIATO]** EEG_08: GCN su grafo elettrodico spaziale (PyG, nodi=elettrodi)
3. **[BREVE]** EEG_09: Graph Attention Networks (GAT)
4. **[MEDIO]** EEG_10: Hypergraph Neural Networks (DHSLP/DHSLF — obiettivo tesi)

### 🔧 Pathway B — Engineering (cartella `notebooks/pathwayB/`)
Miglioramenti ingegneristici ai baseline, utili come comparativi in tesi.

1. B01: Data augmentation per ridurre overfitting inter-sessione (SS)
2. B02: Domain adaptation cross-sessione (CORAL / MMD)
3. B03: Domain adaptation cross-soggetto (adversarial)
4. B04: Contrastive learning subject-invariant (Shen 2022)

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

**Nei notebook GNN (EEG_08, EEG_08b, EEG_09, EEG_10, EEG_11 e tutti i futuri), il paradigma è sempre GRAPH CLASSIFICATION:**

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

## 11. Note Operative

- **Git worktree attivo**: `.claude/worktrees/elegant-neumann/` — lavora sempre qui
- **PR aperta**: #8 su GitHub
- **JupyterLab**: avviare con `preview_start jupyter-lab-311` per nuovi notebook DL (CBraMod), `jupyter-lab` per notebook legacy
- **Non usare** `daniele_dl_thesis` come env — usare `daniele_310` (legacy) o `daniele_311` (nuovi notebook)
- **Non committare** `node_modules/`, file `.pt` di grandi dimensioni, dati grezzi
