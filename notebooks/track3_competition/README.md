# Track#3 — Multi-class Imagined Speech Classification

Pipeline completa (preprocessing → modelli → grafici) per il dataset della competizione
**Track#3** (imagined speech, 5 classi). Progettata per girare sulla macchina SSH `spinlabs-01`
(GPU, env `daniele_311`), con i dati che arrivano nella cartella **Downloads di Windows**.

> ⚠️ Dataset **diverso** da quello della tesi (74 soggetti × 110 parole). Qui: 15 soggetti,
> 5 classi, **subject-dependent**. I due non vanno mescolati.

---

## 1. Il dataset in breve (dal PDF `Data_description(Track3).pdf`)

| Proprietà | Valore |
|---|---|
| Classi | `Hello`, `Helpme`, `Stop`, `Thankyou`, `Yes` (event code 1–5) → **chance 20%** |
| Soggetti | 15 (S01–S15), **subject-dependent**: modello di Sxx-train valutato su Sxx-test |
| Canali | 64 (10-20), ref/ground Fpz/FCz |
| Frequenza | **256 Hz** |
| Epoca | `-500 … +2601 ms` (795 campioni); imagined speech = 0…2000 ms |
| Trial | train **300** (60/classe), val **50** (10/classe), test **50** (10/classe) |

**Dettagli tecnici gestiti dal codice:**
- Train/Val sono `.mat` v7 (`scipy.io`), con `epo.x = (time, channels, trials)`.
- Test è `.mat` **v7.3/HDF5** (`h5py`), con `epo.x = (trials, channels, time)` e clab da dereferenziare.
- Le **true label del Test** sono nel foglio `Test set/Track3_Answer Sheet_Test.xlsx` (nel `.mat` `epo.y` è oscurato).
- Tutti i loader restituiscono orientamento unificato **`(trials, channels, time)`**, label 0–4.

---

## 2. Struttura della cartella

```
track3_competition/
├── README.md                       ← questo file
├── track3_config.py                ← path dati (DATA_ROOT), costanti, device, W&B
├── track3_io.py                    ← loader .mat v7/v7.3 + parser answer sheet
├── track3_preproc.py               ← bandpass, baseline, crop, z-score, resample, band features
├── track3_models.py                ← EEGNet/Shallow/Deep4 (braindecode), DGCNN, DHSLP, REVE
├── track3_train.py                 ← loop subject-dependent, metriche, early stopping, plot
├── 00_data_exploration.ipynb       ← shape, bilanciamento, segnale, PSD, topomap
├── 01_preprocessing.ipynb          ← prima/dopo, pipeline, salvataggio tensori
├── 02_braindecode_baselines.ipynb  ← EEGNet, ShallowFBCSPNet, Deep4Net
├── 03_graph_models.ipynb           ← DGCNN + DHSLP (hypergraph dinamico)
├── 04_reve_foundation.ipynb        ← REVE (brain-bzh/reve-large, 200 Hz)
├── interim/                        ← (generata) tensori .pt preprocessati
└── results/                        ← (generata) metrics_*.csv e figures/
```

La logica pesante è nei moduli `.py` (testabili e riusabili); i notebook sono orchestrazione + grafici.

---

## 3. Dove mettere i dati (Windows Downloads ↔ WSL)

Sulla VM WSL, la cartella Downloads di Windows è montata sotto `/mnt/c/...`. Il codice cerca
`DATA_ROOT` in quest'ordine (il primo che esiste vince):

1. variabile d'ambiente `TRACK3_DATA`
2. i candidati in `_CANDIDATE_ROOTS` dentro `track3_config.py`

Su `spinlabs-01` l'utente Windows è **`students`**, quindi il path atteso è già tra i candidati:
`/mnt/c/Users/students/Downloads/Track#3 Imagined speech classification`.

**Opzione A — variabile d'ambiente (override esplicito):**
```bash
export TRACK3_DATA="/mnt/c/Users/students/Downloads/Track#3 Imagined speech classification"
```
(mettila in `~/.bashrc` per renderla permanente)

**Opzione B — symlink dentro la home WSL:**
```bash
ln -s "/mnt/c/Users/students/Downloads/Track#3 Imagined speech classification" ~/track3_data
```
`~/track3_data` è già tra i candidati.

Verifica con:
```bash
conda activate daniele_311
python track3_config.py     # deve stampare DATA_ROOT: /mnt/c/.../Track#3 ...
python track3_io.py         # deve caricare S01 e stampare le shape
```

> Quando mi dirai il tuo utente Windows, aggiorno io il candidato `/mnt/c/Users/.../Downloads/...`
> in `track3_config.py`.

---

## 4. Ambiente

Env `daniele_311` (Python 3.11, PyTorch 2.5, braindecode 1.3.2, PyG 2.7, MNE 1.11).
Stack già presente **tranne**:

```bash
pip install wandb        # per l'experiment tracking (opzionale ma consigliato)
wandb login              # API key da wandb.ai/settings
# transformers è già installato; serve solo per REVE (scarica i pesi al primo run)
```

Avvio JupyterLab: `preview_start jupyter-lab-311` (o come fai di solito da VSCode remoto).

---

## 5. Come eseguire (ordine)

1. **00_data_exploration** — sanity check dei dati (nessuna dipendenza esterna).
2. **01_preprocessing** — capisci/regola la pipeline; opzionalmente salva i tensori.
3. **02_braindecode_baselines** — EEGNet / ShallowFBCSPNet / Deep4Net (i 3 baseline forti).
4. **03_graph_models** — DGCNN + DHSLP (hypergraph dinamico, obiettivo tesi).
5. **04_reve_foundation** — REVE (prima il test su 1 soggetto per scaricare i pesi, poi il full run).

Ogni run subject-dependent produce `results/metrics_<model>.csv` e `results/figures/*.png`.

**Snippet minimo (da qualsiasi notebook):**
```python
import track3_train as T
df, res = T.run_subject_dependent("eegnet", use_wandb=False,
                                  train_kwargs=dict(epochs=200, patience=30))
T.plot_per_subject(df, "eegnet")
```

---

## 6. Modelli inclusi

| Modello | Input | Note |
|---|---|---|
| **EEGNet** | raw `(ch, time)` | baseline compatto, braindecode |
| **ShallowFBCSPNet** | raw | baseline forte per motor/speech imagery |
| **Deep4Net** | raw | ConvNet profonda |
| **DGCNN** | band features (DE, 5 bande) | adiacenza appresa + Chebyshev graph conv (Song 2018) |
| **DHSLP** | raw | **1 ipergrafo per trial**, iperarchi kNN dinamici (stile Li et al. 2025) |
| **REVE** | raw @ 200 Hz | foundation model `brain-bzh/reve-large`, congelato + testa lineare |

**W&B**: entity `uras-daniele22-politecnico-di-milano`, project `miralis-imagined-speech`,
run taggate `track3` + nome modello + soggetto. Attiva con `use_wandb=True`.

---

## 7. Note / caveat

- **REVE**: l'API esatta (forma dell'output di `model(eeg, positions)`) va verificata al primo run.
  Il wrapper `REVEClassifier` gestisce output tensore/dict in modo difensivo e costruisce la testa
  in modo lazy; se il forward fallisce, controlla la model card HF e adegua `_extract`.
- **DHSLP** qui è una implementazione *funzionale* nello spirito di Li et al. 2025 (hypergraph
  dinamico + structure learning). La riproduzione fedele gamma/semi-supervised è nei notebook
  EEG_13b/EEG_15 della tesi.
- **Preprocessing**: default bandpass 0.5–45 Hz, baseline -500..0 ms, crop 0..2000 ms. Cambia in
  `track3_config.py` o passa `pp_kwargs` a `run_subject_dependent`.
- **Validation come extra-train**: il regolamento consente di usare il val set come training
  aggiuntivo → passa `merge_val=True` a `run_subject_dependent` (in quel caso l'early stopping
  usa comunque il val, valuta se tenere uno split interno).
