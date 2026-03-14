#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_data_symlinks.sh — Collega i dati EEG esterni alla struttura data/ della repo
# ──────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   bash scripts/setup_data_symlinks.sh [SOURCE_DIR]
#
# Default SOURCE_DIR:
#   /mnt/c/Users/students/Desktop/Daniele/EEG_tensors_stuff
#
# Lo script crea symlink da:
#   data/processed/subject_tensors/subject_tensors_aggregated_epoch → SOURCE/subject_tensors_aggregated_epoch
#   data/processed/subject_tensors/subject_tensors_time             → SOURCE/subject_tensors_time
#   data/interim/<file>                                              → SOURCE/<file>  (per file di supporto)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_SOURCE="/mnt/c/Users/students/Desktop/Daniele/EEG_tensors_stuff"
SOURCE="${1:-$DEFAULT_SOURCE}"

if [ ! -d "$SOURCE" ]; then
    echo "✗ Directory sorgente non trovata: $SOURCE"
    echo "  Uso: bash scripts/setup_data_symlinks.sh /path/to/EEG_tensors_stuff"
    exit 1
fi

echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "SOURCE:       $SOURCE"
echo ""

# ── Crea le directory se non esistono ────────────────────────────────────────
mkdir -p "$PROJECT_ROOT/data/processed/subject_tensors"
mkdir -p "$PROJECT_ROOT/data/interim"

# ── 1. Symlink per i tensori ─────────────────────────────────────────────────
echo "─── Tensori ───"

for subdir in subject_tensors_aggregated_epoch subject_tensors_time; do
    target="$PROJECT_ROOT/data/processed/subject_tensors/$subdir"
    source="$SOURCE/$subdir"

    if [ ! -d "$source" ]; then
        echo "  ⚠ Sorgente non trovata: $source — skip"
        continue
    fi

    # Rimuovi directory esistente (non symlink) se presente
    if [ -d "$target" ] && [ ! -L "$target" ]; then
        echo "  ⚠ Rimuovo directory locale esistente: $target"
        rm -rf "$target"
    fi

    ln -sfn "$source" "$target"
    echo "  ✓ $subdir → $source"
done

# ── 2. Symlink per i file di supporto in data/interim/ ───────────────────────
echo ""
echo "─── File di supporto → data/interim/ ───"

INTERIM_FILES=(
    # Distances
    channel_mapping.csv
    channel_pairwise_euclidean_distances.csv
    channel_pairwise_geodesic_distances.csv
    geodesic_D_59.npy
    geodesic_D_59_channels.csv
    # Tensori globali
    eeg_epoch_tensors.pt
    # Embeddings
    emb_cache_paraphrase-multilingual-MiniLM-L12-v2.npy
    # Features CSV
    comprehensive_subject_73.csv
    # Label mappings
    label2idx.json
    idx2label.json
    # EEG-based clustering (non in configs/)
    cluster_names_eeg_4.json
    cluster_names_eeg_5.json
    cluster_names_eeg_hdb2.json
    cluster_names_eeg_z4.json
    cluster_names_eeg_z5.json
    cluster_names_pos4.json
    cluster_names_sem5.json
    labelid2cluster_eeg_4.json
    labelid2cluster_eeg_5.json
    labelid2cluster_eeg_hdb2.json
    labelid2cluster_eeg_z4.json
    labelid2cluster_eeg_z5.json
    labelid2cluster_pos4.json
    labelid2cluster_sem5.json
    # Word2cluster mappings
    word2cluster_4.json
    word2cluster_5.json
    word2cluster_eeg_4.json
    word2cluster_eeg_5.json
    word2cluster_eeg_hdb2.json
    word2cluster_eeg_z4.json
    word2cluster_eeg_z5.json
    word2cluster_fine.json
    word2cluster_pos4.json
    word2cluster_sem5.json
)

linked=0
skipped=0
for f in "${INTERIM_FILES[@]}"; do
    src="$SOURCE/$f"
    dst="$PROJECT_ROOT/data/interim/$f"

    if [ ! -f "$src" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    ln -sf "$src" "$dst"
    linked=$((linked + 1))
done

echo "  ✓ $linked file linkati, $skipped non trovati nella sorgente"

# ── Riepilogo ────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "✅ Setup completato!"
echo ""
echo "Tensori aggregated: OK"
echo "Tensori time:       OK"
echo "File interim:       OK"
echo "═══════════════════════════════════════════════"
