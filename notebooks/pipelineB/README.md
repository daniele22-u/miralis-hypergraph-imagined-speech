# Pipeline B — Rest & Read Analysis

Notebook che sfruttano le nuove condizioni `_read` e `riposo_NNN` estratte dagli XDF di Francesco.

| Notebook | Obiettivo |
|---|---|
| **PB00** `compute_subject_bacc` | Calcola bacc per soggetto (LR su PSD, ~5s/sogg) → CSV |
| **PB01** `bci_illiteracy_from_rest` | Valida claim Blankertz: rest EEG predice BCI illiteracy |
| **PB02** `img_read_rest_signal_comparison` | Confronto ERP/PSD tra le tre condizioni |
| **PB03** `read_img_similarity_literacy` | Similarity read↔img come proxy di BCI literacy |
| **PB04** `rest_normalization` | Rest come baseline per normalizzare epoche img |
| **PB05** `read_pretrain` | Read come augmentation/pre-training per decoder img |

## Dipendenze

- `data/raw_csv/training_set/PXXX_SYYY/*_read.csv` — da EEG_39
- `data/raw_csv/training_set/PXXX_SYYY/riposo_NNN.csv` — da EEG_39
- `data/interim/subject_bacc_pipelineB.csv` — da PB00 (richiesto da PB01, PB03)

## Ordine di esecuzione consigliato

1. **PB00** — calcola bacc per tutti i soggetti (~10 min totali, ha resume capability)
2. **PB02** — esplorazione segnale, nessuna dipendenza, puoi partire subito
3. **PB01** — richiede PB00 e dati rest (riposo_NNN.csv)
4. **PB03** — richiede PB00 e dati read (_read.csv)
5. **PB04** — indipendente, richiede solo riposo_NNN.csv
6. **PB05** — più lento, richiede _read.csv e training PyTorch
