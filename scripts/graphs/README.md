# EEG Viewer — Istruzioni rapide

Questo file descrive come avviare l'interfaccia interattiva `eeg_viewer.py` evitando conflitti tra venv e Conda su macOS.

Problema comune
- Su macOS alcune build di Python sono collegate a una versione di Tcl/Tk incompatibile; chiamare `tk.Tk()` può causare un abort nativo (SIGABRT) e chiudere l'intero processo.

Cosa ho aggiunto nel progetto
- `run_eeg_viewer_conda.sh`: wrapper eseguibile che disattiva un eventuale `.venv` attivo e lancia il viewer dentro l'ambiente Conda `daniele_dl_thesis` (dove sono installati `tk`, `mne`, `matplotlib`, `h5py`).

Come avviare l'interfaccia (raccomandato)

1. Esegui il wrapper (dalla root del progetto):

```bash
./scripts/graphs/run_eeg_viewer_conda.sh
```

2. In alternativa, attiva manualmente l'ambiente Conda e lancia lo script:

```bash
conda activate daniele_dl_thesis
python scripts/graphs/eeg_viewer.py
```

Se vedi messaggi simili a "Detected Tk abort... falling back to static PNG output" significa che stai eseguendo lo script con un Python che ha un Tcl/Tk incompatibile: lo script salverà invece una PNG in `data/figures/`.

Se preferisci sempre il comportamento non-interattivo (solo PNG), lo script già salva un file di fallback quando Tk non è disponibile.

Suggerimenti per ripristinare l'interfaccia interattiva
- Usa una build di Python collegata a una versione aggiornata di Tcl/Tk (es. installer ufficiale Python.org, Homebrew Python o Conda). Per esempio con Conda:

```bash
conda install -n daniele_dl_thesis -c conda-forge tk matplotlib h5py mne
```

- Dopo l'installazione, riesegui il wrapper sopra.

Se hai bisogno, posso aggiungere ulteriori opzioni (es. nome env configurabile, avvio con Qt invece di Tk). Apri un issue o chiedi qui e provvedo.
