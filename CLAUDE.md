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

- **Segnale**: EEG a 59 canali validi, 256 Hz, epoche ~1.5s
  - File H5 (grezzo): **61 canali** (il casco ha 63 posizioni nel `.locs`, ma Pz e POz non sono stati registrati)
  - Rimossi A1 e A2 (elettrodi di riferimento, nessuna posizione spaziale) → **59 canali EEG**
- **Task**: 110 parole immaginarie (chance level ~0.9%)
- **Soggetti**: 70, con 5 sessioni ciascuno (~220 epoche/sessione)
- **Formato dati**: HDF5 in `data/processed/`, struttura `(n_epochs, n_channels, n_samples)`
- **Tensori PyTorch**: in `data/interim/` come file `.pt` per soggetto
- **Clustering semantico**: `word2cluster_4.json`, `word2cluster_5.json` (4-5 categorie)

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
├── notebooks/
│   ├── EEG_00_labels_and_tasks.ipynb
│   ├── EEG_00_feature_significance.ipynb
│   ├── EEG_01_pipeline_metadata_features_analysis.ipynb
│   ├── EEG_02_tensors_and_graph.ipynb
│   ├── EEG_03_visualization_topomaps.ipynb
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

Vedere `docs/DIREZIONI_E_LIMITAZIONI.md` per il dettaglio completo. Ordine di priorità:

1. **[IMMEDIATO]** Completare training GCN spazio-temporale (`EEG_GNN_temporal_baseline_spatial_graph_FIXED.ipynb`)
2. **[IMMEDIATO]** Rieseguire baseline su 4-5 categorie semantiche
3. **[BREVE]** EEGNet / EEG Conformer end-to-end (segnale raw come input)
4. **[BREVE]** Graph Attention Networks (GAT)
5. **[MEDIO]** Hypergraph Neural Networks (obiettivo principale della tesi)
6. **[MEDIO]** Domain adaptation cross-soggetto (MMD, CORAL, adversarial)
7. **[MEDIO]** Contrastive learning subject-invariant

---

## 8. Convenzioni di Codice

- **Lingua**: commenti e documentazione in **italiano**; codice (variabili, funzioni) in **inglese**
- **Notebook**: prefisso numerico `EEG_XX_` per la pipeline principale; `test_` per esperimenti
- **Commit**: seguire lo stile esistente (`feat:`, `docs:`, `refactor:`)
- **Branch corrente**: `claude/elegant-neumann`
- **Nessuna feature engineering manuale** nei nuovi modelli

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

### 9.4 Commit

- Fare sempre commit dopo aggiornamenti significativi
- Usare `docs:` o `feat:` come prefisso a seconda del tipo di modifica

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
