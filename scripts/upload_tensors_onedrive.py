"""
upload_tensors_onedrive.py
==========================
Carica (o sincronizza) i file tensori EEG su OneDrive.

Uso rapido
----------
    python scripts/upload_tensors_onedrive.py

Opzioni principali
------------------
    --dry-run        mostra cosa verrebbe copiato senza farlo
    --dest DIR       cartella di destinazione OneDrive (override di ONEDRIVE_TENSORS_DIR)
    --what all|subj|epoch|interim
                     cosa caricare (default: subj)
    --force          ricopia anche i file già presenti con stessa dimensione
    --method mount|rclone
                     metodo di trasferimento (default: mount)

Variabili d'ambiente
--------------------
    ONEDRIVE_TENSORS_DIR   path completo della cartella OneDrive di destinazione
                           Se non impostata, viene usato il percorso auto-rilevato
                           (funziona su macOS con OneDrive client installato)

Esempio su nuova macchina (Linux / VM)
---------------------------------------
    # Con rclone configurato (rclone config → aggiungi remote "onedrive"):
    python scripts/upload_tensors_onedrive.py --method rclone --dest onedrive:EEG_tensors

    # Con OneDrive montato manualmente:
    export ONEDRIVE_TENSORS_DIR=/mnt/onedrive/EEG_tensors
    python scripts/upload_tensors_onedrive.py
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── project root (portabile, gestisce anche git worktree) ───────────────────
def _find_project_root() -> Path:
    """Trova la root del repo principale, anche se siamo in un worktree."""
    import subprocess
    try:
        # git worktree: --git-common-dir punta sempre al .git del repo principale
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            git_common = Path(result.stdout.strip())
            # .git/worktrees/../.. → repo root
            if "worktrees" in str(git_common):
                return git_common.parent.parent
            return git_common.parent
    except Exception:
        pass
    # fallback: cerca .git risalendo dal CWD
    return next(
        (p for p in [Path().resolve()] + list(Path().resolve().parents)
         if (p / ".git").is_dir()),
        Path().resolve()
    )

project_root = _find_project_root()

# ── cartelle sorgente da caricare ────────────────────────────────────────────
#
#  Categorie logiche:
#  - aggregated : subject_tensors_aggregated_epoch/  (352 MB) ← usato dai modelli DL
#  - time       : subject_tensors_time/              (1.7 GB) ← tensori temporali completi
#  - epoch      : eeg_epoch_tensors.pt               (369 MB) ← tensore unificato (alternativa)
#  - maps       : JSON/NPY piccoli in interim/        (~50 MB) ← cluster maps, embedding cache
#  - features   : CSV feature per soggetto in interim (~1.6 GB) ← ricalcolabili, solo se servono
#  - dl_ready   : aggregated + epoch + maps           (~780 MB) ← tutto ciò che serve per i modelli DL
#  - all        : tutto quanto

_proc = project_root / "data" / "processed"
_int  = project_root / "data" / "interim"

# file piccoli in interim (JSON, NPY, CSV piccoli) — esclusi i CSV pesanti
_maps_files = [
    f for f in _int.iterdir()
    if f.is_file() and f.suffix in (".json", ".npy", ".csv")
    and f.stat().st_size < 5 * 1024 * 1024  # < 5 MB
] if _int.exists() else []

SOURCES = {
    "aggregated": [_proc / "subject_tensors" / "subject_tensors_aggregated_epoch"],
    "time":       [_proc / "subject_tensors" / "subject_tensors_time"],
    "epoch":      [_proc / "eeg_epoch_tensors.pt"],
    "maps":       _maps_files,
    "features":   [f for f in _int.iterdir()
                   if f.is_file() and f.suffix == ".csv"
                   and f.stat().st_size >= 5 * 1024 * 1024
                  ] if _int.exists() else [],
}
SOURCES["dl_ready"] = SOURCES["aggregated"] + SOURCES["epoch"] + SOURCES["maps"]
SOURCES["all"] = (SOURCES["aggregated"] + SOURCES["time"] +
                  SOURCES["epoch"] + SOURCES["maps"] + SOURCES["features"])


# ── auto-rilevamento OneDrive su macOS ───────────────────────────────────────
def _find_onedrive_macos() -> Path | None:
    cloud = Path.home() / "Library" / "CloudStorage"
    if cloud.exists():
        candidates = sorted(cloud.glob("OneDrive*"))
        if candidates:
            return candidates[0]
    # fallback: cartella ~/OneDrive
    od = Path.home() / "OneDrive"
    return od if od.exists() else None


def resolve_dest(dest_arg: str | None) -> tuple[str, Path | None]:
    """
    Restituisce (method, path):
      - method = 'mount'  → path è un Path locale
      - method = 'rclone' → path è None (dest è la stringa rclone remote:path)
    """
    if dest_arg and ":" in dest_arg:
        # rclone remote (es. "onedrive:EEG_tensors")
        return "rclone", None

    env_dir = os.environ.get("ONEDRIVE_TENSORS_DIR")
    if dest_arg:
        return "mount", Path(dest_arg)
    if env_dir:
        return "mount", Path(env_dir)

    auto = _find_onedrive_macos()
    if auto:
        dest = auto / "EEG_tensors"
        print(f"[auto] OneDrive rilevato: {auto}")
        return "mount", dest

    return "mount", None


# ── copia locale (mount) ─────────────────────────────────────────────────────
def _copy_item(src: Path, dst_root: Path, force: bool, dry_run: bool) -> tuple[int, int]:
    """Copia src (file o cartella) in dst_root. Ritorna (copied, skipped)."""
    copied = skipped = 0

    if src.is_file():
        dst = dst_root / src.name
        if not force and dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"  skip  {src.name}")
            skipped += 1
        else:
            print(f"  copy  {src.name}  ({src.stat().st_size / 1e6:.1f} MB)")
            if not dry_run:
                dst_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied += 1

    elif src.is_dir():
        files = sorted(src.rglob("*"))
        files = [f for f in files if f.is_file()]
        print(f"\n  Cartella: {src.name}  ({len(files)} file)")
        for f in files:
            rel = f.relative_to(src.parent)
            dst = dst_root / rel
            if not force and dst.exists() and dst.stat().st_size == f.stat().st_size:
                skipped += 1
            else:
                size_mb = f.stat().st_size / 1e6
                print(f"    copy  {rel}  ({size_mb:.1f} MB)")
                if not dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                copied += 1

    return copied, skipped


def upload_mount(sources: list[Path], dest: Path, force: bool, dry_run: bool):
    print(f"\nDestinazione: {dest}")
    if dry_run:
        print("  [DRY RUN — nessun file verrà copiato]")
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    total_copied = total_skipped = 0
    for src in sources:
        if not src.exists():
            print(f"\n[WARN] sorgente non trovata: {src}")
            continue
        c, s = _copy_item(src, dest, force, dry_run)
        total_copied += c
        total_skipped += s

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Completato: {total_copied} file copiati, {total_skipped} saltati.")


# ── rclone ───────────────────────────────────────────────────────────────────
def upload_rclone(sources: list[Path], dest_str: str, dry_run: bool):
    if shutil.which("rclone") is None:
        print("[ERRORE] rclone non trovato. Installalo con: brew install rclone")
        print("         Poi configura OneDrive: rclone config")
        sys.exit(1)

    for src in sources:
        if not src.exists():
            print(f"[WARN] sorgente non trovata: {src}")
            continue

        if src.is_file():
            remote_path = f"{dest_str}/{src.name}"
            cmd = ["rclone", "copyto", str(src), remote_path, "--progress"]
        else:
            remote_path = f"{dest_str}/{src.name}"
            cmd = ["rclone", "copy", str(src), remote_path, "--progress"]

        if dry_run:
            cmd.append("--dry-run")

        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Carica i tensori EEG su OneDrive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dest", default=None,
                        help="Cartella di destinazione (path locale o remote rclone 'onedrive:path')")
    parser.add_argument("--what",
                        choices=["dl_ready", "aggregated", "time", "epoch", "maps", "features", "all"],
                        default="dl_ready",
                        help="Cosa caricare (default: dl_ready — aggregated+epoch+maps, ~780MB)")
    parser.add_argument("--force", action="store_true",
                        help="Ricopia anche i file già presenti con stessa dimensione")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra cosa verrebbe copiato senza farlo")
    parser.add_argument("--method", choices=["mount", "rclone"], default=None,
                        help="Metodo (default: auto)")
    args = parser.parse_args()

    sources = SOURCES[args.what]
    method, dest_path = resolve_dest(args.dest)

    if args.method:
        method = args.method

    # dimensione totale stimata
    total_bytes = 0
    for src in sources:
        if src.is_file():
            total_bytes += src.stat().st_size if src.exists() else 0
        elif src.is_dir():
            total_bytes += sum(f.stat().st_size for f in src.rglob("*") if f.is_file() and src.exists())
    print(f"Upload: {args.what}  |  {total_bytes / 1e9:.2f} GB stimati")

    if method == "rclone":
        dest_str = args.dest or os.environ.get("ONEDRIVE_TENSORS_DIR", "onedrive:EEG_tensors")
        upload_rclone(sources, dest_str, args.dry_run)
    else:
        if dest_path is None:
            print("\n[ERRORE] Impossibile trovare OneDrive automaticamente.")
            print("  Opzioni:")
            print("  1. Imposta ONEDRIVE_TENSORS_DIR=/path/to/onedrive/EEG_tensors")
            print("  2. Usa --dest /path/to/onedrive/EEG_tensors")
            print("  3. Usa --method rclone --dest onedrive:EEG_tensors")
            sys.exit(1)
        upload_mount(sources, dest_path, args.force, args.dry_run)


if __name__ == "__main__":
    main()
