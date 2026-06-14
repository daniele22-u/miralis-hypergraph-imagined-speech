"""
Genera le 3 figure illustrative per il capitolo Background della tesi (§2.4, §2.5).

Figure schematiche (NON dati reali) che mostrano la corrispondenza grafo <-> matrice
di adiacenza e l'astrazione a ipergrafo, usando le posizioni reali degli elettrodi
(proiezione polare EEGLAB, identica a mne.plot_topomap).

  thesis_graph.pdf       -> §2.4  grafo sul cranio (5 regioni anatomiche, archi pesati)
  thesis_adj_matrix.pdf  -> §2.4  matrice di adiacenza binaria 15x15 (blocchi 0/1)
  thesis_hypergraph.pdf  -> §2.5  ipergrafo (5 iper-archi = regioni)

Tutto il testo nelle figure e' in INGLESE (convenzione tesi).
Output: figures/{thesis_graph,thesis_adj_matrix,thesis_hypergraph}.{pdf,png}
Le figure vanno poi copiate in thesislatex/Images/ (graphicspath = ./Images/).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

project_root = next(
    (p for p in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents)
     if (p / '.git').exists()),
    Path(__file__).resolve().parent.parent,
)
ELOC_FILE = project_root / 'src' / 'io' / 'ebneuro.locs'
FIG_DIR   = project_root / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ── 1. Posizioni 2D elettrodi (proiezione polare EEGLAB) ──────────────────────
# Formato .locs: index angle_deg radius name. +y = naso, +x = destra.
BAD_CH = {'A1', 'A2'}   # reference, radius 0
locs   = []
with open(ELOC_FILE) as fh:
    for line in fh:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            angle, radius, name = float(parts[1]), float(parts[2]), parts[3]
        except ValueError:
            continue
        if name in BAD_CH or radius == 0:
            continue
        rad = np.radians(angle)
        locs.append({'name': name, 'x': radius * np.sin(rad), 'y': radius * np.cos(rad)})

locs     = locs[:61]
ch_names = [d['name'] for d in locs]
pos      = np.array([[d['x'], d['y']] for d in locs])
pos     /= np.sqrt((pos ** 2).sum(axis=1)).max()   # periferici -> cerchio unitario
N        = len(pos)

# ── 2. Regioni anatomiche (unica sorgente di verita' per le 3 figure) ─────────
def nearest(center, n):
    return list(np.argsort(np.linalg.norm(pos - center, axis=1))[:n])

regions = [
    (nearest(np.array([ 0.0,  0.80]), 7), '#4A90D9', 'Frontal'),
    (nearest(np.array([-0.60, 0.0 ]), 6), '#E67E22', 'Left Temporal'),
    (nearest(np.array([ 0.60, 0.0 ]), 6), '#27AE60', 'Right Temporal'),
    (nearest(np.array([ 0.0, -0.40]), 6), '#8E44AD', 'Parietal'),
    (nearest(np.array([ 0.0, -0.90]), 5), '#C0392B', 'Occipital'),
]

node_region, node_col = {}, {}
for ri, (members, color, _) in enumerate(regions):
    for m in members:
        node_region[m] = ri
        node_col[m]    = color

# ── 3. Sottoinsieme per la matrice: 3 elettrodi per regione = 15 nodi ─────────
sel_nodes, region_slices = [], []
for node_ids, color, name in regions:
    s = len(sel_nodes)
    sel_nodes += node_ids[:3]
    region_slices.append((s, s + 3, color, name))
n_s          = len(sel_nodes)
sel_labels   = [ch_names[i] for i in sel_nodes]
sel_col_list = [node_col[i] for i in sel_nodes]

# ── 4. Archi DETERMINISTICI (no seed) — ogni "1" della matrice = arco nel grafo
# within-region: triangolo completo tra i 3 sel-nodi; cross-region: 3 link sparsi.
edge_w = {}
for s, e, _, _ in region_slices:
    sn = sel_nodes[s:e]
    for a in range(len(sn)):
        for b in range(a + 1, len(sn)):
            edge_w[(min(sn[a], sn[b]), max(sn[a], sn[b]))] = 0.78
for i, j in [(sel_nodes[1], sel_nodes[4]),
             (sel_nodes[7], sel_nodes[9]),
             (sel_nodes[10], sel_nodes[13])]:
    edge_w[(min(i, j), max(i, j))] = 0.38

A_sub = np.zeros((n_s, n_s), dtype=int)
for ki, i in enumerate(sel_nodes):
    for kj, j in enumerate(sel_nodes):
        if ki != kj and (min(i, j), max(i, j)) in edge_w:
            A_sub[ki, kj] = 1

# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_head(ax, r=1.06, col='#555', lw=1.6):
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(th) * r, np.sin(th) * r, color=col, lw=lw, zorder=0)
    ax.plot([-0.09, 0, 0.09], [r, r * 1.17, r], color=col, lw=lw, zorder=0)
    for s in (-1, 1):
        t = np.linspace(np.radians(92), np.radians(268), 40)
        ax.plot(s * (r + 0.09 * np.cos(t)), 0.14 * np.sin(t), color=col, lw=lw, zorder=0)

def inflated_hull(pts, pad=0.13):
    if len(pts) < 3:
        return None
    hull = ConvexHull(pts)
    v    = pts[hull.vertices]
    d    = v - v.mean(axis=0)
    v2   = v + d / np.linalg.norm(d, axis=1, keepdims=True).clip(1e-9) * pad
    return np.vstack([v2, v2[0]])

KEY = {'Cz', 'C3', 'C4', 'Fz', 'Pz', 'Oz', 'T3', 'T4', 'FCz', 'CPz'}

def label_key(ax):
    for i, ch in enumerate(ch_names):
        if ch in KEY:
            ax.annotate(ch, pos[i], fontsize=7, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points',
                        color='#111', fontweight='bold')

def add_legend(ax):
    patches = [mpatches.Patch(facecolor=c, edgecolor='none', label=n)
               for _, c, n in regions]
    ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.04),
              ncol=5, fontsize=8, frameon=True, framealpha=0.95, edgecolor='#ccc',
              handlelength=1.2, handletextpad=0.4, columnspacing=0.7)

def scalp_ax(ax):
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.28, 1.42)

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    print(f'Saved: figures/{name}.*')

# ── FIGURA 1 — GRAFO (§2.4) ───────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(5.5, 6.2))
scalp_ax(ax1); draw_head(ax1)
ax1.scatter(pos[:, 0], pos[:, 1], c='#DDDDDD', s=22, zorder=2,
            edgecolors='#AAAAAA', linewidths=0.3)
for ki, i in enumerate(sel_nodes):
    for kj, j in enumerate(sel_nodes):
        if ki >= kj or A_sub[ki, kj] == 0:
            continue
        ri, rj = node_region.get(i), node_region.get(j)
        col = regions[ri][1] if ri == rj else '#2C3E50'
        lw  = 2.5 if ri == rj else 1.2
        ax1.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                 color=col, alpha=0.75, linewidth=lw, zorder=3, solid_capstyle='round')
ax1.scatter(pos[sel_nodes, 0], pos[sel_nodes, 1], c=sel_col_list, s=70, zorder=4,
            edgecolors='white', linewidths=1.2)
label_key(ax1); add_legend(ax1)
fig1.tight_layout(pad=0.3)
save(fig1, 'thesis_graph')

# ── FIGURA 2 — MATRICE DI ADIACENZA (§2.4) ────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7.0, 6.8))
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_xlim(-2.5, n_s + 2.2); ax2.set_ylim(n_s + 0.2, -2.0)
for ki in range(n_s):
    for kj in range(n_s):
        val = A_sub[ki, kj]
        if ki == kj:
            bg, fg, fw = '#EFEFEF', '#BBBBBB', 'normal'
        elif val == 1:
            ri_r = next((ri for ri, (s, e, c, _) in enumerate(region_slices) if s <= ki < e), None)
            ri_c = next((ri for ri, (s, e, c, _) in enumerate(region_slices) if s <= kj < e), None)
            bg, fg, fw = (regions[ri_r][1] if ri_r == ri_c else '#2C3E50'), 'white', 'bold'
        else:
            bg, fg, fw = '#FAFAFA', '#CCCCCC', 'normal'
        ax2.add_patch(Rectangle([kj - 0.5, ki - 0.5], 1, 1,
                                facecolor=bg, edgecolor='#DDDDDD', lw=0.5))
        if ki != kj:
            ax2.text(kj, ki, str(val), ha='center', va='center',
                     fontsize=9, color=fg, fontweight=fw, family='monospace')
for s, e, c, _ in region_slices[:-1]:
    ax2.axhline(e - 0.5, color='#888', lw=0.8, ls=':', zorder=6)
    ax2.axvline(e - 0.5, color='#888', lw=0.8, ls=':', zorder=6)
for s, e, c, _ in region_slices:
    ax2.add_patch(Rectangle([s - 0.5, s - 0.5], e - s, e - s,
                            facecolor='none', edgecolor=c, lw=2.5, zorder=7))
for kj, label in enumerate(sel_labels):
    rc = next((c for s, e, c, _ in region_slices if s <= kj < e), '#333')
    ax2.text(kj, -0.65, label, ha='center', va='bottom', fontsize=7,
             color=rc, fontweight='bold', rotation=90)
for ki, label in enumerate(sel_labels):
    rc = next((c for s, e, c, _ in region_slices if s <= ki < e), '#333')
    ax2.text(-0.65, ki, label, ha='right', va='center', fontsize=7.5,
             color=rc, fontweight='bold')
for s, e, c, name in region_slices:
    ax2.text(n_s + 0.3, (s + e - 1) / 2.0, name, ha='left', va='center',
             fontsize=7.5, color=c, fontweight='bold')
fig2.tight_layout(pad=0.3)
save(fig2, 'thesis_adj_matrix')

# ── FIGURA 3 — IPERGRAFO (§2.5) ───────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(5.5, 6.2))
scalp_ax(ax3); draw_head(ax3)
for node_ids, color, name in regions:
    hv = inflated_hull(pos[node_ids], pad=0.13)
    if hv is not None:
        ax3.fill(hv[:, 0], hv[:, 1], color=color, alpha=0.18, zorder=1)
        ax3.plot(hv[:, 0], hv[:, 1], color=color, lw=2.0, ls='--', zorder=2, alpha=0.80)
ax3.scatter(pos[:, 0], pos[:, 1], c='#DDDDDD', s=22, zorder=3,
            edgecolors='#AAAAAA', linewidths=0.3)
ax3.scatter(pos[sel_nodes, 0], pos[sel_nodes, 1], c=sel_col_list, s=70, zorder=5,
            edgecolors='white', linewidths=1.2)
label_key(ax3); add_legend(ax3)
fig3.tight_layout(pad=0.3)
save(fig3, 'thesis_hypergraph')

print('\nDone. Copia i PDF in thesislatex/Images/ per la tesi.')
