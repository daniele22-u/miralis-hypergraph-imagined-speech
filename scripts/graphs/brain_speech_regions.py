#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mappa 3D del cervello (fsaverage) con le regioni del dual-stream del parlato
evidenziate, per la figura di §2.2 della tesi.

Strumento: MNE-Python + fsaverage + parcellazione Desikan-Killiany (aparc),
lo stesso stack già usato in EEG_07f.

USO
----
Env consigliato: daniele_311 (MNE 1.11). Serve anche PyVista:
    pip install pyvista pyvistaqt
Esegui (in JupyterLab o da shell con display):
    python scripts/graphs/brain_speech_regions.py
Output:
    figures/brain_speech_regions.png  (PNG ad alta risoluzione)

NOTE
----
- La prima volta scarica fsaverage (~1 GB) via mne.datasets.fetch_fsaverage().
  Se il progetto l'ha già scaricato (EEG_07f), riusa la cache.
- Headless (server senza schermo): esporta prima
      export PYVISTA_OFF_SCREEN=true
  e, se serve, usa xvfb-run.
- Per angolare a piacere: cambia AZIMUTH/ELEVATION qui sotto, oppure in un
  notebook ruota interattivamente e poi chiama brain.save_image(...).
"""

import os
# render OFFSCREEN: salva il PNG senza aprire finestre/widget interattivi.
# DEVE stare prima di import mne/pyvista.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
from pathlib import Path

import mne

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SURF      = "inflated"          # 'inflated' (regioni chiare) | 'pial' (gyri realistici)
HEMI      = "lh"                # emisfero sinistro (rete del parlato)
SIZE      = (1600, 1200)        # risoluzione render
BG        = "white"
CORTEX    = "low_contrast"      # grigio tenue
ALPHA     = 0.85                # opacità delle regioni evidenziate

# Vista: laterale sinistra. Per fine-tuning sblocca AZIMUTH/ELEVATION.
VIEW      = "lateral"
AZIMUTH   = None                # es. 180
ELEVATION = None                # es. 90

# Colori coerenti con la figura TikZ (blu ~ bluepoli, arancio)
COL_DORSAL  = "#4a7a99"         # dorsal stream  (articolatorio / sensomotorio)
COL_VENTRAL = "#c8761f"         # ventral stream (lessicale-semantico)

# Regioni Desikan-Killiany (aparc), suffisso -lh aggiunto in automatico
DORSAL  = ["precentral", "caudalmiddlefrontal", "supramarginal", "parsopercularis"]
VENTRAL = ["superiortemporal", "middletemporal", "parstriangularis"]

OUT = Path(__file__).resolve().parents[2] / "figures" / "brain_speech_regions.png"


def main():
    # offscreen automatico se non c'è display
    if not os.environ.get("DISPLAY") and os.name != "nt":
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    print(">> fetch fsaverage (riusa la cache se presente)...")
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    subject = "fsaverage"

    try:
        mne.viz.set_3d_backend("pyvistaqt")
    except Exception:
        mne.viz.set_3d_backend("notebook")

    print(f">> rendering brain ({SURF}, {HEMI})...")
    Brain = mne.viz.get_brain_class()
    # show=False: niente display interattivo (evita il crash ipywidgets/trame da terminale)
    brain = Brain(subject, hemi=HEMI, surf=SURF, subjects_dir=subjects_dir,
                  background=BG, cortex=CORTEX, size=SIZE, show=False)

    labels = mne.read_labels_from_annot(subject, parc="aparc", hemi=HEMI,
                                        subjects_dir=subjects_dir, verbose=False)
    by_name = {lab.name: lab for lab in labels}

    def highlight(region_list, color):
        for r in region_list:
            name = f"{r}-{HEMI}"
            lab = by_name.get(name)
            if lab is None:
                print(f"   [warn] regione non trovata: {name}")
                continue
            brain.add_label(lab, color=color, alpha=ALPHA, borders=False)
            print(f"   + {name}")

    print(">> evidenzio dorsal stream...");  highlight(DORSAL,  COL_DORSAL)
    print(">> evidenzio ventral stream..."); highlight(VENTRAL, COL_VENTRAL)

    if AZIMUTH is not None or ELEVATION is not None:
        brain.show_view(azimuth=AZIMUTH, elevation=ELEVATION)
    else:
        brain.show_view(VIEW)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    brain.save_image(str(OUT))
    print(f">> salvato: {OUT}")
    print("   (dorsal=blu, ventral=arancio — legenda da mettere nella caption LaTeX)")


if __name__ == "__main__":
    main()
