# Track#3 — Risultati (imagined speech, 5 classi, chance 20%)

> Aggiornato: luglio 2026. Numeri = **test accuracy** (classi bilanciate → acc ≈ balanced acc).
> Preprocessing: **PP_MINIMAL** (solo z-score, nessun bandpass/baseline/crop). 1 seed.

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

## TODO

- [ ] LOSO per i numeri subject-independent finali (gold standard cross-subject).
- [ ] Più seed per le medie ± std.
- [ ] REVE (foundation model) — dove ci si aspetta di guadagnare sul cross-subject.
- [ ] (opzionale) DHSLP fedele a Li et al. 2025 (gamma + PCC + pruning + 1-NN semi-sup).
- [ ] Esperimento resample 200 Hz + normalizzazione /100µV per allinearsi esattamente a CBraMod.
