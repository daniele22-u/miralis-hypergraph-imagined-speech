"""
Genera figura tesi: grafo elettrodico sovrapposto alla mappa del cranio.

Pannello sinistro  : matrice di adiacenza A (abs. PCC, pruned, media su N trial)
Pannello destro    : grafo sul cranio — archi pesati per coupling, nodi per grado

Sorgente dati (in ordine di priorità):
  1. data/hypergraphs_pruned_abs_pcc/  → .pt files (adj già calcolata + prunata)
  2. data/raw_csv/training_set/        → CSV raw   → calcola abs_pcc + prune

Output:
  figures/scalp_graph_thesis.pdf   (vettoriale, per LaTeX)
  figures/scalp_graph_thesis.png   (300 dpi, per preview)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
project_root = next(
    (p for p in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents)
     if (p / '.git').exists()),
    Path(__file__).resolve().parent.parent,
)

ELOC_FILE     = project_root / 'src' / 'io' / 'ebneuro.locs'
HG_DIR        = project_root / 'data' / 'hypergraphs_pruned_abs_pcc'
CSV_DIR       = project_root / 'data' / 'raw_csv' / 'training_set'
FIG_OUT       = project_root / 'figures' / 'scalp_graph_thesis'

SUBJ          = 0       # soggetto da usare (cambia se P000 non disponibile)
SESS          = 1       # sessione
MAX_TRIALS    = 30      # media su questi trial per una adj più stabile
EDGE_THR_PCT  = 95      # top 5% archi (coerente con EEG_07f)
N_CH          = 61      # canali EEG (escluse reference A1, A2)

BLUEPOLI      = '#005F9E'
BAD_CHANNELS  = {'A1', 'A2'}   # reference — escludi dalla visualizzazione

# ── 1. POSIZIONI 2D DAGLI ELETTRODI (.locs EEGLAB) ───────────────────────────
# Formato colonne: index  angle_deg  radius  name
# Convenzione EEGLAB: angle=0 → naso (y>0), angle=±90 → orecchie (x=±)
# 2D Cartesiane: x = radius * sin(angle_rad)  (dx, ±=right/left)
#                y = radius * cos(angle_rad)  (dy, + = front/nose)

locs_data = []
with open(ELOC_FILE) as f:
    for line in f:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            _, angle, radius, name = parts[0], float(parts[1]), float(parts[2]), parts[3]
        except ValueError:
            continue
        if name in BAD_CHANNELS or radius == 0.0:
            continue
        rad = np.radians(angle)
        x   = radius * np.sin(rad)
        y   = radius * np.cos(rad)
        locs_data.append({'name': name, 'x': x, 'y': y, 'radius': radius})

locs_data = locs_data[:N_CH]   # prendi solo i primi N_CH
ch_names  = [d['name'] for d in locs_data]
pos_xy    = np.array([[d['x'], d['y']] for d in locs_data])   # (N_CH, 2)

# Normalizza al cerchio: in EEGLAB il raggio 0.5 ≈ bordo orecchie
# Nota: il massimo raggio nel file è ~0.535, scalalo a 1.0
max_r     = max(d['radius'] for d in locs_data)
pos_norm  = pos_xy / max_r     # elettrodi periferici toccano ~cerchio unitario

print(f'Elettrodi caricati: {len(ch_names)}')

# ── 2. MATRICE DI ADIACENZA (media su trial) ─────────────────────────────────
adj_accum = np.zeros((N_CH, N_CH), dtype=np.float32)
n_loaded  = 0

# Opzione A: carica da .pt precomputati
pt_dir = HG_DIR / f'P{SUBJ:03d}_S{SESS:03d}'
if pt_dir.exists():
    pts = sorted(pt_dir.glob('trial_*.pt'))[:MAX_TRIALS]
    for pt_path in pts:
        d   = torch.load(pt_path, map_location='cpu')
        adj = d.get('adj')
        if adj is not None:
            a = adj.numpy().astype(np.float32)
            if a.shape == (N_CH, N_CH):
                adj_accum += a
                n_loaded  += 1
    if n_loaded:
        print(f'Caricati {n_loaded} .pt da {pt_dir}')

# Opzione B: calcola abs_pcc da CSV raw
if n_loaded == 0:
    sess_dir = CSV_DIR / f'P{SUBJ:03d}_S{SESS:03d}'
    csvs     = sorted(sess_dir.glob('*_img.csv'))[:MAX_TRIALS] if sess_dir.exists() else []
    for csv_path in csvs:
        x = pd.read_csv(csv_path, header=None).values[:N_CH].astype(np.float32)
        if x.shape[0] < N_CH:
            continue
        a_raw = np.abs(np.corrcoef(x)).astype(np.float32)
        np.fill_diagonal(a_raw, 0.0)
        thr = np.percentile(a_raw[a_raw > 0], EDGE_THR_PCT)
        a_raw[a_raw < thr] = 0.0
        adj_accum += a_raw
        n_loaded  += 1
    if n_loaded:
        print(f'Calcolato abs_pcc da {n_loaded} CSV in {sess_dir}')

if n_loaded == 0:
    raise FileNotFoundError(
        f'Nessun dato trovato per P{SUBJ:03d}_S{SESS:03d}.\n'
        f'Controlla HG_DIR={HG_DIR} o CSV_DIR={CSV_DIR}.\n'
        f'Cambia SUBJ/SESS nella config.'
    )

adj_mean = adj_accum / n_loaded   # (N_CH, N_CH) — media su trial

# ── 3. STATISTICHE ───────────────────────────────────────────────────────────
degree   = (adj_mean > 0).sum(axis=1).astype(float)
w_max    = adj_mean.max()

# ── 4. PLOT ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5.8))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.1], figure=fig, wspace=0.12)

# ── 4a. Matrice di adiacenza ─────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
im  = ax0.imshow(adj_mean, cmap='Blues', aspect='auto',
                 origin='upper', interpolation='nearest')
cb0 = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
cb0.set_label('avg. abs. PCC (pruned)', fontsize=10)
ax0.set_xlabel('Electrode index', fontsize=10)
ax0.set_ylabel('Electrode index', fontsize=10)
ax0.set_title(r'Adjacency matrix $\mathbf{A}$  '
              f'(mean over {n_loaded} trials)', fontsize=11)
ax0.tick_params(labelsize=8)

# ── 4b. Grafo sul cranio ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.set_aspect('equal')
ax1.axis('off')

# Testa (cerchio + naso + orecchie)
head = plt.Circle((0, 0), 1.06, color='#888888', fill=False,
                  linewidth=1.8, zorder=0)
ax1.add_patch(head)

nose_x = np.array([-0.10, 0, 0.10])
nose_y = np.array([1.06, 1.26, 1.06])
ax1.plot(nose_x, nose_y, color='#888888', linewidth=1.8, zorder=0)

for sign in (-1, 1):
    theta = np.linspace(np.radians(92), np.radians(268), 60)
    ax1.plot(sign * (1.06 + 0.10 * np.cos(theta)),
             0.10 * np.sin(theta),
             color='#888888', linewidth=1.8, zorder=0)

# Archi (linewidth e alpha proporzionali al peso medio)
for i in range(N_CH):
    for j in range(i + 1, N_CH):
        w = adj_mean[i, j] / w_max
        if w <= 0:
            continue
        ax1.plot([pos_norm[i, 0], pos_norm[j, 0]],
                 [pos_norm[i, 1], pos_norm[j, 1]],
                 color=BLUEPOLI,
                 alpha=0.20 + 0.70 * w,
                 linewidth=0.25 + 2.20 * w,
                 zorder=1)

# Nodi (colore = grado)
sc = ax1.scatter(pos_norm[:, 0], pos_norm[:, 1],
                 c=degree, cmap='Blues',
                 vmin=degree.min(), vmax=degree.max(),
                 s=60, zorder=3,
                 edgecolors='#222222', linewidths=0.55)

# Etichette elettrodi principali
KEY_LABELS = {'Cz', 'C3', 'C4', 'Fz', 'Pz', 'T3', 'T4',
              'Oz', 'F3', 'F4', 'P3', 'P4'}
for i, ch in enumerate(ch_names):
    if ch in KEY_LABELS:
        ax1.annotate(ch, pos_norm[i],
                     fontsize=7.5, ha='center', va='bottom',
                     xytext=(0, 6), textcoords='offset points',
                     color='#111111', fontweight='normal')

cb1 = fig.colorbar(sc, ax=ax1, fraction=0.038, pad=0.02, shrink=0.78)
cb1.set_label('Degree', fontsize=10)

ax1.set_title(r'Electrode graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$  '
              '(abs. PCC, pruned)', fontsize=11)
ax1.set_xlim(-1.38, 1.38)
ax1.set_ylim(-1.38, 1.48)

# ── 5. SALVA ──────────────────────────────────────────────────────────────────
(project_root / 'figures').mkdir(exist_ok=True)
for ext in ('pdf', 'png'):
    out = f'{FIG_OUT}.{ext}'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Salvato: {out}')

plt.show()
