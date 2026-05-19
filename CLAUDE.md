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
│   ├── EEG_08_..._GNN.ipynb
│   ├── EEG_09_gat_subject_independent.ipynb
│   ├── EEG_10_hypergraph_nn.ipynb
│   ├── EEG_11_hgnn_subject_independent.ipynb
│   ├── EEG_12_subject_specific_gnn.ipynb
│   ├── EEG_13b_dhslp_subject_specific.ipynb  ← DHSLP soggetto-specifico, fix overfitting
│   ├── EEG_14_dhslp_pretrain_finetune.ipynb  ← pretrain S-indep + calibrazione per sogg.
│   ├── EEG_15_li_trial_hypergraph.ipynb      ← Li et al. 2025 fedele (⚠️ non ancora eseguito)
│   ├── EEG_16_subject_clustering.ipynb       ← clustering strutturale su adj pruned
│   ├── EEG_17_cluster_connectivity.ipynb     ← connettività media per cluster semantico
│   ├── EEG_18_functional_clustering.ipynb    ← clustering funzionale 4D→305D→1041D
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
- **EEG_17**: connettività media per cluster semantico (firma individuale, ns vs bAcc)
- **EEG_18**: clustering funzionale 4D→305D→1041D (tutto ns, ARI=0 vs strutturale)

### 🎯 Prossimi passi

1. **[IMMEDIATO]** Eseguire EEG_15 sui dati completi — aggiornare risultati
2. **[IMMEDIATO]** Scrittura tesi — il quadro sperimentale è sostanzialmente completo
   - Cap. variabilità inter-soggetto: null results SONO il contributo
   - Cap. metodi: descrivere EEG_13b, EEG_14, EEG_15, EEG_16/17/18
3. **[BREVE]** Focus intra-soggetto
   - Ablation freeze encoder in EEG_14 (solo classificatore fine-tunato)
   - Confronto sistematico W-HGNN vs DHSLP vs Li et al. per soggetto
4. **[MEDIO]** Dizionario neurale semantico
   - Embedding per cluster semantico dai migliori soggetti
   - Separabilità nello spazio latente

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

## 11. Note Operative

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

### Cosa significa
La variabilità inter-soggetto nella decodifica IS non è spiegata da:
- Topografia della connettività EEG (struttura statica del grafo)
- Pattern di potenza spettrale multi-banda (5 bande × 61 canali)
- Feature temporali gamma (Li et al. 2025, 12 feature × 61 canali)
- Combinazioni di tutto quanto sopra

### Implicazione per la tesi
- Presentare il clustering negativo come "esaurimento sistematico" — più forte di un singolo null result
- I migliori soggetti (34% bAcc) esistono e sono riproducibili → modelli intra-soggetto hanno senso
- Il dizionario neurale semantico è realizzabile solo per soggetti ad alta capacità IS

### Cosa NON fare
- Non proporre ulteriore clustering inter-soggetto su feature EEG statiche — già esaurito
- Non aspettarsi che un modello subject-independent superi chance su questo dataset
- Non proporre feature engineering manuale come soluzione principale
