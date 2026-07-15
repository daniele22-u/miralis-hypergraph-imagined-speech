"""
track3_config.py — Configurazione centrale per il dataset Track#3 (imagined speech, 5 classi).

Tutto il codice (moduli e notebook) importa da qui costanti e path.
Pensato per girare sia in locale (Mac) sia sulla macchina SSH `spinlabs-01` (WSL/Windows),
dove i dati arrivano nella cartella Downloads di Windows.

--- Come impostare il path ai dati ---
Ordine di risoluzione di DATA_ROOT (il primo che esiste vince):
  1. variabile d'ambiente TRACK3_DATA  (es. export TRACK3_DATA=/mnt/c/Users/<user>/Downloads/Track3)
  2. i candidati elencati in _CANDIDATE_ROOTS qui sotto
Se nessuno esiste, DATA_ROOT resta None e i loader sollevano un errore esplicito con le istruzioni.

Su WSL i file di Windows sono montati sotto /mnt/c/... — quindi la cartella Downloads di Windows
tipicamente è: /mnt/c/Users/<TUO_UTENTE_WINDOWS>/Downloads/'Track#3 Imagined speech classification'
Quando linkeremo Downloads<->WSL basterà aggiungere quel path a _CANDIDATE_ROOTS o esportare TRACK3_DATA.
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Risoluzione del path ai dati
# ---------------------------------------------------------------------------
# Aggiungi qui i path candidati (in ordine di priorità). Il primo esistente vince.
_CANDIDATE_ROOTS = [
    # --- Mac locale di Daniele ---
    "/Users/danieleuras/Downloads/Track#3 Imagined speech classification",
    # --- Macchina SSH spinlabs-01 (WSL): Downloads di Windows montato sotto /mnt/c ---
    # NB: sostituisci <winuser> con il tuo utente Windows quando linkeremo Downloads<->WSL.
    "/mnt/c/Users/danieleuras/Downloads/Track#3 Imagined speech classification",
    # --- eventuale copia dentro la home WSL dopo il link ---
    os.path.expanduser("~/data/Track#3 Imagined speech classification"),
    os.path.expanduser("~/track3_data"),
]


def resolve_data_root() -> Path | None:
    env = os.environ.get("TRACK3_DATA")
    if env and Path(env).exists():
        return Path(env)
    for c in _CANDIDATE_ROOTS:
        if Path(c).exists():
            return Path(c)
    return None


DATA_ROOT = resolve_data_root()

# Sottocartelle dentro DATA_ROOT (nomi esatti dallo zip della competizione)
TRAIN_DIR_NAME = "Training set"
VAL_DIR_NAME = "Validation set"
TEST_DIR_NAME = "Test set"
FILE_PATTERN = "Data_Sample{:02d}.mat"      # Data_Sample01.mat ... Data_Sample15.mat
ANSWER_SHEET_TEST = "Track3_Answer Sheet_Test.xlsx"   # dentro Test set/, contiene le TRUE label

# Chiave dello struct `epo` dentro il .mat, dipende dal set
EPO_KEY = {"train": "epo_train", "val": "epo_validation", "test": "epo_test"}


def data_dir(split: str) -> Path:
    """split in {'train','val','test'} -> Path della sottocartella."""
    assert DATA_ROOT is not None, _no_data_msg()
    return DATA_ROOT / {"train": TRAIN_DIR_NAME, "val": VAL_DIR_NAME, "test": TEST_DIR_NAME}[split]


def subject_file(split: str, subject: int) -> Path:
    return data_dir(split) / FILE_PATTERN.format(subject)


def _no_data_msg() -> str:
    return (
        "DATA_ROOT non trovato. Imposta il path ai dati Track#3 in uno di questi modi:\n"
        "  export TRACK3_DATA=/percorso/della/cartella/Track#3\n"
        "oppure aggiungi il path a _CANDIDATE_ROOTS in track3_config.py.\n"
        f"Candidati provati: {_CANDIDATE_ROOTS}"
    )


# ---------------------------------------------------------------------------
# 2. Costanti del dataset (dal PDF Data_description Track3)
# ---------------------------------------------------------------------------
FS = 256                       # Hz, sampling frequency (epo.fs)
N_CHANNELS = 64                # elettrodi 10-20
N_CLASSES = 5
# event code -> nome classe (dal PDF, sezione Event codes)
CLASS_NAMES = ["Hello", "Helpme", "Stop", "Thankyou", "Yes"]   # ordine = event code 1..5 = label 0..4
EVENT_CODE = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}  # 1..5
CHANCE_LEVEL = 1.0 / N_CLASSES  # 0.20

# Timing epoca (epo.t): da -500 ms a ~2601 ms => 795 campioni a 256 Hz
T_MIN_MS = -500.0
T_MAX_MS = 2601.5625
N_SAMPLES = 795
# Fase di imagined speech: il cross mark sparisce a t=0 e l'immaginazione dura ~2 s
CUE_ONSET_MS = 0.0
IMAGERY_DUR_MS = 2000.0

# Numero trial atteso per set (subject-dependent)
N_TRIALS = {"train": 300, "val": 50, "test": 50}   # 5 classi x {60,10,10}
SUBJECTS = list(range(1, 16))   # S01..S15

# ---------------------------------------------------------------------------
# 3. Preprocessing di default (override nei notebook se serve)
# ---------------------------------------------------------------------------
BANDPASS_HZ = (0.5, 45.0)      # filtro passa-banda di default (mu/beta + basso gamma)
BASELINE_MS = (-500.0, 0.0)    # finestra per baseline correction (media sottratta)
CROP_MS = (0.0, 2000.0)        # finestra analizzata (imagined speech). None = usa tutta l'epoca
NOTCH_HZ = None                # es. 50.0 se serve togliere rete elettrica; None = off
REVE_FS = 200                  # REVE richiede 200 Hz -> resample da 256

# ---------------------------------------------------------------------------
# 4. Device
# ---------------------------------------------------------------------------
def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 5. Path di output (dentro il repo)
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PKG_DIR / "results"
INTERIM_DIR = PKG_DIR / "interim"   # tensori .pt preprocessati per soggetto
FIG_DIR = RESULTS_DIR / "figures"
for _d in (RESULTS_DIR, INTERIM_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 6. Weights & Biases
# ---------------------------------------------------------------------------
WANDB_ENTITY = "uras-daniele22-politecnico-di-milano"
WANDB_PROJECT = "miralis-imagined-speech"   # stesso progetto della tesi; run taggate 'track3'


def summary() -> str:
    root = str(DATA_ROOT) if DATA_ROOT else "NON TROVATO (vedi _no_data_msg)"
    return (
        f"Track#3 config\n"
        f"  DATA_ROOT   : {root}\n"
        f"  fs          : {FS} Hz | canali: {N_CHANNELS} | classi: {N_CLASSES} (chance {CHANCE_LEVEL:.1%})\n"
        f"  classi      : {CLASS_NAMES}\n"
        f"  epoca       : {T_MIN_MS:.0f}..{T_MAX_MS:.0f} ms ({N_SAMPLES} campioni)\n"
        f"  soggetti    : {len(SUBJECTS)} (subject-dependent)\n"
    )


if __name__ == "__main__":
    print(summary())
