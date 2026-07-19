# Track#3 — Risultati (imagined speech, 5 classi, chance 20%)

> Aggiornato: luglio 2026. Numeri = **test accuracy** (classi bilanciate → acc ≈ balanced acc).
> Preprocessing: **PP_MINIMAL** (solo z-score, nessun bandpass/baseline/crop).

## ⭐ Modello novel: HyperTempNet — ipergrafo TEMPORALE (batte il baseline)

**Contributo principale**: gli ipergrafi di connettività **spaziale** (sui canali) NON aiutano la
decodifica di imagined speech — la connettività codifica il **soggetto** (NMI 0.95), non la **parola**
(NMI 0.001), dimostrato con ablation su 5 architetture (DHSLP, HyperEEGNet, HyperAdaptNet, DGCNN,
membro-ensemble). Proponiamo invece un **ipergrafo sui SEGMENTI TEMPORALI** (dinamiche della parola),
ispirato agli inter-segment hyperedges di Hyper-MML (Kang et al. 2026), combinato con un front-end
multi-scala (EEG-Inception) e la costruzione dinamica dell'ipergrafo (cfr. DHSLP).

**Architettura**: raw → conv multi-scala (kernel 16/32/64/128) → conv spaziale → K=10 segmenti
temporali (nodi) → ipergrafo appreso sui segmenti (HGNN) → classificatore.

**Risultati (subject-dependent, 5 seed, Wilcoxon paired per-soggetto):**

| modello | test acc (media ± std) |
|---|---|
| **HyperTempNet (ipergrafo temporale ON)** | **0.701 ± 0.008** |
| HyperTempNet (ablation: ipergrafo OFF) | 0.655 ± 0.015 |
| Shallow (in-harness, confronto equo) | 0.554 ± 0.014 |

- **Contributo ipergrafo temporale** (on vs off): Δ=**+0.046**, **Wilcoxon p=0.0012**, 13/15 soggetti meglio.
- **vs baseline Shallow**: Δ=**+0.147**, **Wilcoxon p=0.0004**, 14/15 soggetti meglio.

**Confronto sui 3 protocolli** (HyperTempNet vs Shallow, stessa harness):

| protocollo | HyperTempNet | Shallow | Δ |
|---|---|---|---|
| subject-dependent | **0.711** | 0.556 | +0.155 |
| subject-mixed | **0.501** | 0.404 | +0.097 |
| subject-independent | 0.231 | 0.221 | +0.010 (entrambi ~chance) |

HyperTempNet vince su **dependent e mixed** (dove c'è segnale). Sul **mixed 0.501** batte tutti i baseline
(Shallow 0.456) e si avvicina a **CBraMod 0.537** (foundation model) senza pretraining. Cross-subject a
chance per entrambi (muro cross-subject invariato).

**Controllo dual** (ipergrafo temporale + spaziale in parallelo): lo **spaziale da solo = 0.24 (chance)**,
il dual ≈ temporale-solo → lo spaziale non aggiunge info sulla parola (conferma la storia).

**La storia della tesi**: *ipergrafi spaziali (connettività) falliscono perché catturano il soggetto;
ipergrafi temporali (segmenti) funzionano perché catturano le dinamiche della parola.* Negative +
positive, con meccanismo e ablation. È il cuore della novelty.

### Metriche stile CBraMod Table 9 (Optuna + subject-mixed, 5 seed)

Iperparametri di HyperTempNet ottimizzati con **Optuna** (30 trial TPE, obiettivo `val_bacc`, nessun
leakage): `F=16, K_seg=12, n_edges=8, hidden=96, dropout=0.3, lr=7.4e-4, wd=3.7e-5, batch=32`.
Metriche finali su test (5 seed, media ± std), sulle **stesse colonne** del paper CBraMod:

| Metodo | Params | Balanced Acc | Cohen's κ | Macro-F1 |
|---|---|---|---|---|
| EEGNet *(paper CBraMod)* | 0.003M | 0.4413 | 0.3016 | 0.4413 |
| LaBraM-Base *(paper)* | 5.8M | 0.5060 | 0.3800 | 0.5054 |
| CBraMod *(foundation, paper)* | 4.0M | 0.5373 ± 0.0108 | 0.4216 ± 0.0163 | 0.5383 ± 0.0096 |
| Shallow *(nostro, in-harness)* | 0.04M | 0.4021 ± 0.0133 | 0.2527 ± 0.0166 | 0.4013 ± 0.0134 |
| **⭐ HyperTempNet *(nostro, Optuna)*** | **0.04M** | **0.5555 ± 0.0169** | **0.4443 ± 0.0212** | **0.5551 ± 0.0170** |

**Lettura onesta**: HyperTempNet batte nettamente i baseline non-foundation (vs Shallow +0.153 bacc,
+0.192 κ) ed è **alla pari / marginalmente sopra CBraMod** su tutte e tre le metriche (+0.018 bacc,
+0.023 κ, +0.017 mF1) — **ma le barre d'errore si sovrappongono**, quindi il claim è "eguaglia un
foundation model", non "lo supera in modo significativo". Il valore sta nel **contesto**: **~100× meno
parametri** (40.549 = 0.04M vs 4.0M — stessa taglia della Shallow che batte) e **nessun pretraining**
(CBraMod è pre-addestrato su ~60k ore). Notebook `11_optuna_metrics.ipynb`.

## Finding principale: il bandpass FIR distruggeva il segnale

Il preprocessing iniziale (`PP_DEFAULT`: bandpass 0.5–45 Hz + baseline + crop 0–2000 ms)
teneva **tutti i modelli a chance**. Il colpevole isolato è il **filtro passa-banda FIR di MNE**
applicato a epoche corte (3 s): introduce artefatti che smaterializzano l'informazione
discriminativa. Togliendolo (`PP_MINIMAL`):

| EEGNet subject-dependent | test_acc |
|---|---|
| PP_DEFAULT (bandpass 0.5–45) | 0.24 (≈ chance) |
| **PP_MINIMAL (nessun filtro)** | **0.555** |

Isolamento pulito (subject-mixed): `PP_MINIMAL` 0.34 vs `PP_HIGHGAMMA` (bandpass 0.5–100) 0.24
→ differiscono **solo** per il bandpass. Conferma: è il filtro, non la banda di frequenza.
Allineato con CBraMod, che usa preprocessing minimo (solo resample + normalizzazione).

**Conseguenza**: `PP_MINIMAL` è ora lo **standard** (default dei runner). `PP_DEFAULT`/`PP_HIGHGAMMA`
restano nel codice solo per l'ablation.

## Tabella: 3 protocolli × 5 modelli (PP_MINIMAL)

| modello | subject-dependent | subject-mixed | subject-independent (holdout) |
|---|---|---|---|
| **ShallowFBCSPNet** | **0.575** | **0.456** | 0.225 |
| EEGNet | 0.555 | 0.335 | 0.198 |
| Deep4Net | 0.472 | 0.411 | 0.198 |
| DGCNN | 0.285 | 0.211 | 0.200 |
| DHSLP (dinamico) | 0.264 | 0.204 | 0.193 |
| *chance* | *0.20* | *0.20* | *0.20* |

*(subject-independent = holdout su S14–15; per il numero rigoroso serve LOSO.)*

## Letture

1. **Il pattern regge su tutti i modelli**: `dependent > mixed > independent`.
   - *dependent* migliore: il modello dedicato cattura i pattern del singolo soggetto.
   - *mixed* più basso: un modello unico per 15 soggetti è diluito dalla variabilità
     inter-soggetto (ε²≈0.85 per soggetto: i trial si raggruppano per soggetto, non per parola).
   - *independent* a chance: **l'imagined speech non trasferisce a soggetti nuovi**.
2. **ShallowFBCSPNet è il modello migliore** (dep 0.575, mixed 0.456). Il suo mixed **batte
   l'EEGNet di CBraMod Table 9 (0.44)** nello stesso protocollo.
3. **I modelli a grafo (DGCNN/DHSLP) su feature spettrali restano a chance**: le CNN sul segnale
   grezzo li dominano → conferma l'approccio "raw end-to-end, niente feature engineering".
4. **0.555 subject-dependent è competitivo** con la BCI Competition 2020 Track 3 (~0.5–0.65).

## Confronto con CBraMod (ICLR 2025, Table 9 — protocollo subject-MIXED)

| metodo | protocollo | balanced acc |
|---|---|---|
| EEGNet (CBraMod paper) | mixed | 0.44 |
| LaBraM-Base (CBraMod paper) | mixed | 0.506 |
| CBraMod | mixed | 0.537 |
| **ShallowFBCSPNet (noi, PP_MINIMAL)** | **mixed** | **0.456** |

CBraMod resta davanti (grazie al pretraining), ma il nostro Shallow con preprocessing minimo
è nel gruppo di testa dei baseline non-foundation.

## Modelli a grafo: retry con costruzione "base" (PCC per-trial + DHSLP EEG_13b)

Rifatti DGCNN e DHSLP con la tecnica corretta della tesi (non più band-power / kNN):
- **DGCNN**: grafo **PCC per-trial + pruning top-k** (`track3_graphs.py`), node features dal raw.
- **DHSLP**: **fedele a EEG_13b** — iperarchi appresi, incidenza soft `H=softmax(node·E)`, K finestre raw.

| modello | dependent | mixed | max_train (mixed) |
|---|---|---|---|
| DGCNN (grafo PCC) | 0.283 | 0.199 | **0.236** (non fitta il training) |
| DHSLP (EEG_13b) | 0.241 | 0.207 | 0.427 (fitta, non generalizza) |

**Verdetto (robusto su entrambi i protocolli)**: la struttura a grafo/ipergrafo **NON aggiunge
nulla** sopra le ConvNet sul raw.
- DGCNN **underfitta**: ridurre i 64 canali a connettività + mean-pool butta via l'informazione
  spazio-temporale che le ConvNet sfruttano (non fitta nemmeno 4500 trial di training).
- DHSLP **overfitta subject-dependent** (train→1.0, test chance) e resta a chance sul mixed:
  memorizza, la struttura appresa non trasferisce.
- Coerente con la filosofia "raw end-to-end": le ConvNet (Shallow 0.575, EEGNet 0.555) dominano.

## REVE (foundation model, brain-bzh/reve-large — CONGELATO + linear probe)

REVE usato come feature extractor congelato + testa lineare (nessun adattamento del backbone).

| protocollo | REVE (congelato) | miglior ConvNet |
|---|---|---|
| subject-dependent | 0.329 | Shallow 0.575 |
| subject-mixed | 0.288 | Shallow 0.456 |
| subject-independent (holdout) | 0.200 (chance) | ~0.225 |

**Letture**:
- REVE congelato ~0.33 (dependent): il linear probe è **ben sopra il chance** senza adattare il
  backbone → il pretraining (60k ore) contiene feature rilevanti per l'imagined speech.
- Ma **non batte le ConvNet task-specific** (Shallow 0.575): il transfer generico perde contro
  un modello addestrato sul task.
- **Cross-subject a chance anche per REVE**: nemmeno un foundation model da 60k ore rompe il muro
  → conferma fortissima che l'imagined speech non trasferisce tra soggetti.
- **Fine-tuning** (`freeze_backbone=False`), subject-independent holdout: `max_train_acc=0.807`,
  `test=0.189` (chance). Il backbone sbloccato **fitta i soggetti di training all'80%** ma sui
  soggetti mai visti resta a chance → **prova definitiva** che il muro cross-subject non è capacità
  né ottimizzazione: il pattern dell'imagined speech è **soggetto-specifico**. Nemmeno un foundation
  model da 60k ore, fine-tunato, lo supera.

## Quadro complessivo (subject-dependent, PP_MINIMAL, test acc)

ShallowFBCSPNet 0.575 > EEGNet 0.555 > Deep4 0.472 > **REVE (frozen) 0.329** > DGCNN 0.283 > DHSLP 0.241.
Cross-subject (independent): **tutti a chance (~0.20)**.

## TODO

- [x] REVE congelato (linear probe) — 0.329 dep / 0.288 mixed / 0.200 indep. Sotto le ConvNet.
- [x] REVE **fine-tuning** — indep: train 0.807 ma test 0.189 (chance). Muro cross-subject confermato.
- [ ] LOSO per i numeri subject-independent finali (gold standard cross-subject).
- [ ] Più seed per le medie ± std.
- [ ] (opzionale) DHSLP fedele a Li et al. 2025 (gamma + PCC + pruning + 1-NN semi-sup).
- [ ] Esperimento resample 200 Hz + normalizzazione /100µV per allinearsi esattamente a CBraMod.
