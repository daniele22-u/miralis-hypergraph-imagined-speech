# Pipeline `validation_5words`

Validazione esterna del progetto su un **dataset indipendente a 5 parole** (Francesco Iacomi).
Stessi metodi della tesi, soggetti **nuovi** → serve a (a) testare il decoding a vocabolario
ridotto in modo *nativo* (non sottocampionato come EEG_38) e (b) **replicare** i fenotipi C0/C1
su una coorte indipendente.

## Dataset sorgente
- Path: `~/Library/CloudStorage/OneDrive-PolitecnicodiMilano/File di Francesco Iacomi - 5words`
- 41 soggetti (`sub-P000` … `sub-P0040`), 5 sessioni, 5 parole: **acqua, aiuto, mangiare, no, si**
- Condizioni: **img** (imagined) + **read**
- **17/41** hanno epoche già pronte (`Epoche_selezionate_in_automatico/{parola}/S{n}_epoca_{parola}_{img|read}.set`, EEGLAB)
- Le epoche sono **già preprocessate** (band-pass 1–100, notch 50/100, resample 256 Hz, avg ref,
  `pop_clean_rawdata`) — equivalente al pipeline della tesi. **Niente ri-filtraggio** (sarebbe doppio).
  Shape epoca: **(61, 384) @ 256 Hz, 1.5 s** = formato canonico della tesi.

## Isolamento dei dati (IMPORTANTE)
La numerazione 5words (P000–P040) **collide** con i soggetti della tesi (P000–P090).
Per non sovrascrivere nulla, **tutto il dataset 5words vive sotto `data/5words_subjects/`**:
```
data/5words_subjects/
  P{id:03d}_S{sess:03d}/{parola}_{img|read}_{k:02d}.csv   ← stadio 1 (ingest, 61×384)
  graphs/{graphs,hypergraphs}_{pruned_}{metric}/P.._S../trial_###.pt   ← stadio 2
```

## Label scheme
`configs/label_schemes/label2idx_5words.json` → acqua=0, aiuto=1, mangiare=2, no=3, si=4.
Decoding **diretto delle 5 parole** (chance = **20%**). Niente concr4/gram4.

## Stadi
| # | Notebook | Cosa fa | Input → Output |
|---|----------|---------|----------------|
| 1 | `V5W_01_preprocessing.ipynb` | ingest .set → CSV canonici | OneDrive .set → `data/5words_subjects/P.._S../*.csv` |
| 2 | `V5W_02_graph_build.ipynb` | connettività + consensus → grafi/ipergrafi pruned (per-trial) | CSV → `data/5words_subjects/graphs/...` |
| 3 | `V5W_03_dhslp_subject_specific.ipynb` | DHSLP per-soggetto (5 classi) | grafi → bAcc per soggetto (chance 20%) |
| 4 | `V5W_04_dhslp_subject_independent.ipynb` | DHSLP cross-subject (5 classi) | grafi → test del ceiling |
| 5 | `V5W_05_phenotypes.ipynb` | clustering EEG-first → replica C0/C1 | grafi → ARI cross-metrica, topomap |

Env: **`daniele_311`** (serve MNE per leggere le .set).

## Stato build
- [x] Stadio 1 — ingest (runnabile)
- [x] Stadio 2 — graph build (adattato da `EEG_07f`, runnabile)
- [ ] Stadio 3 — DHSLP SS (adattare `EEG_13b`: N_CLASSES=5, label dirette, HG_ROOT 5words)
- [ ] Stadio 4 — DHSLP SI (adattare `EEG_13`: split sui soggetti 5words)
- [ ] Stadio 5 — fenotipi (adattare `EEG_16b`: ricostruire la cache feature dai grafi 5words)

Gli stadi 3–5 si costruiscono **dopo** aver verificato l'output dello stadio 2
(dipendono interamente dai grafi prodotti lì).
