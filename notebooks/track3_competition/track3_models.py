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
    from braindecode.models import EEGNetv4, ShallowFBCSPNet, Deep4Net
    name = name.lower()
    cls = {"eegnet": EEGNetv4, "shallow": ShallowFBCSPNet, "deep4": Deep4Net}.get(name)
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
def _normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """A_hat = D^-1/2 (relu(A)+I) D^-1/2, simmetrizzata."""
    A = F.relu(A)
    A = 0.5 * (A + A.t())
    n = A.size(0)
    A = A + torch.eye(n, device=A.device, dtype=A.dtype)
    d = A.sum(1)
    d_inv_sqrt = torch.pow(d + 1e-8, -0.5)
    Dm = torch.diag(d_inv_sqrt)
    return Dm @ A @ Dm


def _chebyshev(A_hat: torch.Tensor, K: int) -> list[torch.Tensor]:
    """Termini di Chebyshev T_0..T_{K-1} della matrice normalizzata."""
    n = A_hat.size(0)
    Tk = [torch.eye(n, device=A_hat.device, dtype=A_hat.dtype), A_hat]
    for k in range(2, K):
        Tk.append(2 * A_hat @ Tk[-1] - Tk[-2])
    return Tk[:K]


class DGCNN(nn.Module):
    def __init__(self, n_ch: int, in_feat: int, n_classes: int = C.N_CLASSES,
                 K: int = 3, hid: int = 64, dropout: float = 0.5):
        super().__init__()
        self.n_ch, self.K = n_ch, K
        # adiacenza appresa (dinamica nel senso di Song: parametro allenato)
        self.A = nn.Parameter(torch.full((n_ch, n_ch), 1e-3))
        # una proiezione lineare per ogni termine di Chebyshev
        self.theta = nn.ModuleList([nn.Linear(in_feat, hid, bias=False) for _ in range(K)])
        self.bn = nn.BatchNorm1d(n_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(n_ch * hid, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, n_ch, in_feat)
        x = self.bn(x)
        A_hat = _normalize_adj(self.A)
        Tk = _chebyshev(A_hat, self.K)
        out = 0.0
        for k in range(self.K):
            out = out + self.theta[k](Tk[k] @ x)   # (batch, n_ch, hid)
        out = self.dropout(self.act(out))
        return self.fc(out)


# ===========================================================================
# 3. DHSLP — Dynamic Hypergraph Structure Learning + Prediction
#    (stile Li et al. 2025). 1 IPERGRAFO PER TRIAL, iperarchi kNN dinamici
#    ricostruiti a ogni forward dalle embedding apprese -> graph classification.
# ===========================================================================
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
        # x: (batch, n_ch, time) -> (batch*n_ch, 1, time)
        b, c, t = x.shape
        h = self.net(x.reshape(b * c, 1, t)).reshape(b, c, self.emb_dim)
        return h   # (batch, n_ch, emb_dim)


def _dynamic_incidence(emb: torch.Tensor, k: int) -> torch.Tensor:
    """
    Costruisce l'incidenza H (batch, n_nodi, n_iperarchi) via kNN dinamico sulle embedding.
    Un iperarco per nodo: contiene il nodo + i suoi k vicini piu simili (distanza euclidea).
    """
    b, n, d = emb.shape
    dist = torch.cdist(emb, emb)                 # (b, n, n)
    idx = dist.topk(k + 1, dim=-1, largest=False).indices   # (b, n, k+1) include se stesso
    H = torch.zeros(b, n, n, device=emb.device, dtype=emb.dtype)
    H.scatter_(2, idx.transpose(1, 2), 1.0)      # colonna e = iperarco centrato sul nodo e
    return H                                     # (b, n_nodi, n_iperarchi=n)


class _HGNNConv(nn.Module):
    """Convoluzione su ipergrafo (Feng et al. 2019): X' = Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Theta."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim)

    def forward(self, x, H):
        # x:(b,n,in) H:(b,n,e)
        W = torch.ones(H.size(0), H.size(2), device=H.device, dtype=H.dtype)  # peso iperarchi = 1
        Dv = H.sum(2) + 1e-8                      # grado nodi (b,n)
        De = H.sum(1) + 1e-8                      # grado iperarchi (b,e)
        Dv_isqrt = torch.diag_embed(Dv.pow(-0.5))
        De_inv = torch.diag_embed((De).pow(-1.0))
        Wd = torch.diag_embed(W)
        theta_x = self.theta(x)                   # (b,n,out)
        G = Dv_isqrt @ H @ Wd @ De_inv @ H.transpose(1, 2) @ Dv_isqrt
        return G @ theta_x


class DHSLP(nn.Module):
    """
    Dynamic Hypergraph Structure Learning + Prediction.
    Encoder temporale -> embedding nodi (canali) -> iperarchi kNN dinamici ->
    2 layer HGNN -> global mean pool -> classificatore.
    """
    def __init__(self, n_ch: int, n_classes: int = C.N_CLASSES,
                 emb_dim: int = 64, hid: int = 64, k: int = 8, dropout: float = 0.5):
        super().__init__()
        self.k = k
        self.encoder = _TemporalEncoder(emb_dim)
        self.hg1 = _HGNNConv(emb_dim, hid)
        self.hg2 = _HGNNConv(hid, hid)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(hid, 64), nn.ELU(), nn.Dropout(dropout),
                                 nn.Linear(64, n_classes))

    def forward(self, x):
        emb = self.encoder(x)                     # (b, n_ch, emb)
        H = _dynamic_incidence(emb, self.k)       # (b, n_ch, n_edges)
        h = self.act(self.hg1(emb, H))
        h = self.dropout(h)
        h = self.act(self.hg2(h, H))
        h = h.mean(1)                             # global mean pool sui nodi
        return self.cls(h)


# ===========================================================================
# 4. REVE — foundation model (brain-bzh/reve-large), input a 200 Hz
# ===========================================================================
class REVEClassifier(nn.Module):
    """
    Wrapper del foundation model REVE come feature extractor + testa lineare.
    Richiede: pip install transformers, e connessione per scaricare i pesi la prima volta.
    Input atteso: (batch, n_ch, time) a 200 Hz (usa preprocess_subject(resample_to=200)).

    NB: l'API esatta di REVE va verificata al primo run: la forward gestisce in modo
    difensivo output tensore / dict (last_hidden_state / pooler_output) e costruisce la
    testa lineare in modo lazy alla prima chiamata.
    """
    def __init__(self, electrode_names: list[str], n_classes: int = C.N_CLASSES,
                 freeze_backbone: bool = True):
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
        self.head = None   # lazy

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
        if self.head is None:
            self.head = nn.Linear(emb.size(-1), self.n_classes).to(emb.device)
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
        assert in_feat is not None, "DGCNN richiede in_feat (n. bande)"
        return DGCNN(n_ch, in_feat, **kw)
    if name == "dhslp":
        return DHSLP(n_ch, **kw)
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
    feat = torch.randn(b, n_ch, 5)
    print("DGCNN   :", DGCNN(n_ch, 5)(feat).shape)
    print("DHSLP   :", DHSLP(n_ch)(x).shape)
    print("(REVE non testato offline: richiede download pesi)")
