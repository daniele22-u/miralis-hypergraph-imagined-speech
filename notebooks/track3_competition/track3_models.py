"""
track3_models.py — Modelli per Track#3 (imagined speech, 5 classi, subject-dependent).

Contiene:
  * make_braindecode(name, ...)  factory per EEGNet / ShallowFBCSPNet / Deep4Net (segnale grezzo)
  * DGCNN                        Dynamical Graph CNN (Song et al. 2018) su feature di banda per canale
  * DHSLP                        Dynamic Hypergraph Structure Learning + Prediction (stile Li et al. 2025)
                                 -> 1 ipergrafo PER TRIAL con iperarchi kNN dinamici (graph classification)
  * REVEClassifier               wrapper del foundation model brain-bzh/reve-large (200 Hz) + testa lineare

Convenzioni input:
  - braindecode / DHSLP / REVE : segnale grezzo (batch, n_ch, time)
  - DGCNN                      : feature di banda   (batch, n_ch, n_feat)   [vedi track3_preproc.band_features]
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import track3_config as C


# ===========================================================================
# 1. Baseline end-to-end su segnale grezzo (braindecode)
# ===========================================================================
def make_braindecode(name: str, n_chans: int, n_times: int, n_classes: int = C.N_CLASSES,
                     sfreq: float = C.FS) -> nn.Module:
    """
    name in {'eegnet','shallow','deep4'}. Restituisce un nn.Module che mappa
    (batch, n_chans, n_times) -> LOGITS (batch, n_classes).

    IMPORTANTE: alcune versioni di braindecode terminano il modello con LogSoftmax.
    Con nn.CrossEntropyLoss il softmax verrebbe applicato DUE VOLTE (log_softmax(log_softmax(x)))
    -> gradiente schiacciato, il modello resta a chance. Qui garantiamo output = logits:
      1) se il costruttore supporta `add_log_softmax`, lo mettiamo False;
      2) in ogni caso rimuoviamo un eventuale strato finale (Log)Softmax.
    """
    import inspect
    try:
        from braindecode.models import EEGNet as _EEGNet          # nome nuovo (braindecode recenti)
    except ImportError:
        from braindecode.models import EEGNetv4 as _EEGNet        # alias vecchio (braindecode <1.12)
    from braindecode.models import ShallowFBCSPNet, Deep4Net
    name = name.lower()
    cls = {"eegnet": _EEGNet, "shallow": ShallowFBCSPNet, "deep4": Deep4Net}.get(name)
    if cls is None:
        raise ValueError(f"nome braindecode sconosciuto: {name}")
    kw = dict(n_chans=n_chans, n_outputs=n_classes, n_times=n_times, final_conv_length="auto")
    if "add_log_softmax" in inspect.signature(cls.__init__).parameters:
        kw["add_log_softmax"] = False
    model = cls(**kw)
    _strip_softmax(model)
    return model


def _strip_softmax(model: nn.Module) -> nn.Module:
    """Sostituisce con Identity ogni (Log)Softmax nel modello (garantisce output = logits)."""
    for parent in model.modules():
        for key, child in list(parent.named_children()):
            if isinstance(child, (nn.LogSoftmax, nn.Softmax)):
                setattr(parent, key, nn.Identity())
    return model


def output_is_logprob(model: nn.Module, x) -> bool:
    """Diagnostica: True se l'output somiglia a log-probabilità (exp somma ~1) invece che logits."""
    model.eval()
    with torch.no_grad():
        s = model(x[:8]).exp().sum(dim=1)
    return bool((s - 1).abs().mean() < 1e-2)


# ===========================================================================
# 2. DGCNN — Dynamical Graph Convolutional Neural Network (Song et al. 2018)
#    Matrice di adiacenza APPRESA (globale, condivisa), Chebyshev graph conv.
#    Nodi = canali; feature nodo = differential entropy per banda.
# ===========================================================================
import track3_graphs as G


class _TemporalEncoder(nn.Module):
    """Encoder condiviso per canale: Conv1d sul tempo -> embedding per nodo (canale)."""
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, stride=2, padding=12), nn.BatchNorm1d(16), nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=13, stride=2, padding=6), nn.BatchNorm1d(32), nn.ELU(),
            nn.Conv1d(32, emb_dim, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(emb_dim), nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.emb_dim = emb_dim

    def forward(self, x):
        b, c, t = x.shape
        h = self.net(x.reshape(b * c, 1, t)).reshape(b, c, self.emb_dim)
        return h   # (batch, n_ch, emb_dim)


class _GraphConvDense(nn.Module):
    """Graph conv tipo GCN su adiacenza densa normalizzata: out = A_hat (X W)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_hat):
        return torch.bmm(A_hat, self.lin(x))      # (B,N,out)


class DGCNN(nn.Module):
    """
    DGCNN con GRAFO PCC PER-TRIAL + pruning (tecnica base della tesi, vedi track3_graphs).
    Per ogni trial: adiacenza da connettività (PCC/PLV) tra canali -> pruning top-k ->
    normalizzazione GCN. Node features dal SEGNALE GREZZO (encoder temporale), NON band-power
    (le feature spettrali affossavano il modello). 1 grafo per trial, on-the-fly.
    """
    def __init__(self, n_ch: int, n_times: int, n_classes: int = C.N_CLASSES,
                 metric: str = "pcc", k_neighbors: int = 8,
                 emb_dim: int = 64, hid: int = 64, dropout: float = 0.5):
        super().__init__()
        self.metric, self.k = metric, k_neighbors
        self.encoder = _TemporalEncoder(emb_dim)
        self.gc1 = _GraphConvDense(emb_dim, hid)
        self.gc2 = _GraphConvDense(hid, hid)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(hid, 64), nn.ELU(), nn.Dropout(dropout),
                                 nn.Linear(64, n_classes))

    def forward(self, x):
        A_hat = G.build_adjacency(x, metric=self.metric, k=self.k)   # (B,N,N) per-trial
        h = self.encoder(x)                                          # (B,N,emb) dal raw
        h = self.dropout(self.act(self.gc1(h, A_hat)))
        h = self.act(self.gc2(h, A_hat))
        return self.cls(h.mean(1))                                   # global mean pool


# ===========================================================================
# 3. DHSLP — Dynamic Hypergraph, FEDELE a EEG_13b (tesi).
#    Iperarchi = embedding APPRENDIBILI; incidenza soft H = softmax(node·edge)
#    (structure learning end-to-end). Node features da FINESTRE RAW + pos-encoding.
#    1 ipergrafo per trial. Portato da notebooks/EEG_13b_dhslp_subject_specific.ipynb.
# ===========================================================================
class _HGNNConv(nn.Module):
    """HGNN layer (Feng et al. 2019) — batched, incidenza soft H (EEG_13b)."""
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_ch, out_ch))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, H):
        # X:(B,N,C)  H:(B,N,E) soft
        d_v = H.sum(dim=2).clamp(min=1e-6)
        d_e = H.sum(dim=1).clamp(min=1e-6)
        Dv = (1.0 / d_v.sqrt()).unsqueeze(-1)
        De = (1.0 / d_e).unsqueeze(1)
        out = Dv * (X @ self.weight)
        out = torch.bmm(H.transpose(1, 2), out)
        out = De.transpose(1, 2) * out
        out = torch.bmm(H, out)
        out = Dv * out
        if self.bias is not None:
            out = out + self.bias
        return out


class DHSLP(nn.Module):
    """
    Dynamic Hypergraph Structure Learning + Prediction (Li et al. 2025), come in EEG_13b.
    Iperarchi = E (n_edges, d_model) APPRENDIBILI. Per K finestre temporali:
      feat = node_proj(finestra_raw) + pos_enc ;  H = softmax(feat @ E.T)  (incidenza soft)
    HGNN conv su H, mean pool su nodi e finestre.
    """
    def __init__(self, n_ch: int, n_times: int, n_classes: int = C.N_CLASSES,
                 K: int = 4, n_edges: int = 16, d_model: int = 64, hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.K = K
        self.T_win = max(1, n_times // K)
        self.d_model = d_model
        self.E = nn.Parameter(torch.randn(n_edges, d_model) * 0.01)      # iperarchi appresi
        self.pos_enc = nn.Parameter(torch.randn(n_ch, d_model) * 0.01)   # pos-encoding elettrodi
        self.node_proj = nn.Sequential(nn.Linear(self.T_win, d_model), nn.LayerNorm(d_model), nn.ELU())
        dims = [d_model] + [hidden] * n_layers
        self.convs = nn.ModuleList([_HGNNConv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.clf = nn.Linear(hidden, n_classes)

    def build_dynamic_H(self, feat):
        scores = torch.matmul(feat, self.E.T) / (self.d_model ** 0.5)
        return torch.softmax(scores, dim=2)        # (B,N,n_edges) soft

    def forward(self, x):
        B, N, T = x.shape
        outs = []
        for k in range(self.K):
            x_k = x[:, :, k * self.T_win:(k + 1) * self.T_win]
            if x_k.shape[2] != self.T_win:         # scarta finestra finale incompleta
                continue
            feat = self.node_proj(x_k) + self.pos_enc     # (B,N,d)
            H_k = self.build_dynamic_H(feat)              # (B,N,n_edges)
            out = feat
            for conv, bn in zip(self.convs, self.bns):
                out = conv(out, H_k)
                out = bn(out.reshape(B * N, -1)).reshape(B, N, -1)
                out = F.relu(out)
                out = self.drop(out)
            outs.append(out.mean(dim=1))                  # (B, hidden)
        return self.clf(torch.stack(outs, dim=1).mean(dim=1))


# ===========================================================================
# 4. REVE — foundation model (brain-bzh/reve-large), input a 200 Hz
# ===========================================================================
class REVEClassifier(nn.Module):
    """
    Wrapper del foundation model REVE come feature extractor + testa lineare.
    Richiede: pip install transformers, e connessione per scaricare i pesi la prima volta.
    Input atteso: (batch, n_ch, time) a 200 Hz (usa preprocess_subject(resample_to=200)).

    NB: l'API esatta di REVE va verificata al primo run: la forward gestisce in modo
    difensivo output tensore / dict (last_hidden_state / pooler_output). La testa lineare
    viene costruita SUBITO in __init__ (con un forward fittizio per inferire la dim di
    embedding), così i suoi parametri entrano nell'optimizer creato da train_model.
    """
    def __init__(self, electrode_names: list[str], n_classes: int = C.N_CLASSES,
                 freeze_backbone: bool = True, dummy_time: int = 200):
        super().__init__()
        from transformers import AutoModel
        self.pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained("brain-bzh/reve-large", trust_remote_code=True)
        self.electrode_names = electrode_names
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # posizioni (n_ch, 3) calcolate una volta
        with torch.no_grad():
            pos = self.pos_bank(electrode_names)
            self.register_buffer("positions", pos if torch.is_tensor(pos) else torch.as_tensor(pos))
        self.n_classes = n_classes
        # --- testa costruita SUBITO via forward fittizio (fix: deve esistere prima
        #     dell'optimizer, altrimenti i suoi pesi non verrebbero mai allenati) ---
        with torch.no_grad():
            dummy = torch.zeros(1, len(electrode_names), dummy_time)
            emb = self._extract(self.backbone(dummy, self.positions.unsqueeze(0)))
        self.head = nn.Linear(emb.size(-1), n_classes)

    def _extract(self, out):
        """Normalizza l'output di REVE in un vettore (batch, D)."""
        if torch.is_tensor(out):
            emb = out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            emb = out[0]
        elif isinstance(out, dict):
            emb = out.get("pooler_output", out.get("last_hidden_state", list(out.values())[0]))
        else:
            raise TypeError(f"Output REVE non riconosciuto: {type(out)}")
        # pool su eventuali dimensioni token intermedie -> (batch, D)
        while emb.dim() > 2:
            emb = emb.mean(dim=1)
        return emb

    def forward(self, x):
        pos = self.positions.unsqueeze(0).expand(x.size(0), -1, -1).to(x.device)
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            out = self.backbone(x, pos)
        emb = self._extract(out)
        return self.head(emb)


# ===========================================================================
# Factory unica
# ===========================================================================
def build_model(name: str, *, n_ch: int, n_times: int | None = None,
                in_feat: int | None = None, electrode_names=None, **kw) -> nn.Module:
    name = name.lower()
    if name in ("eegnet", "shallow", "deep4"):
        assert n_times is not None
        return make_braindecode(name, n_ch, n_times, **kw)
    if name == "dgcnn":
        assert n_times is not None, "DGCNN (grafo PCC per-trial) richiede n_times (segnale raw)"
        return DGCNN(n_ch, n_times, **kw)
    if name == "dhslp":
        assert n_times is not None, "DHSLP (finestre raw) richiede n_times"
        return DHSLP(n_ch, n_times, **kw)
    if name == "reve":
        assert electrode_names is not None
        return REVEClassifier(electrode_names, **kw)
    raise ValueError(f"modello sconosciuto: {name}")


if __name__ == "__main__":
    b, n_ch, t = 4, 64, 512
    x = torch.randn(b, n_ch, t)
    print("EEGNet  :", make_braindecode("eegnet", n_ch, t)(x).shape)
    print("Shallow :", make_braindecode("shallow", n_ch, t)(x).shape)
    print("Deep4   :", make_braindecode("deep4", n_ch, t)(x).shape)
    print("DGCNN   :", DGCNN(n_ch, t)(x).shape, "(grafo PCC per-trial)")
    print("DHSLP   :", DHSLP(n_ch, t)(x).shape, "(ipergrafo learned, EEG_13b)")
    print("(REVE non testato offline: richiede download pesi)")
