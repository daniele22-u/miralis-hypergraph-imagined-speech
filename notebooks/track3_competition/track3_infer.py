"""
track3_infer — carica un checkpoint HyperTempNet salvato e fa inference/interpretabilita'
senza riaddestrare. Il checkpoint (results/checkpoints/hypertempnet_best.pt) contiene
state_dict + tutta la config necessaria a ricostruire il modello.

Uso tipico:
    import track3_infer as I
    model, meta = I.load_checkpoint()          # ricostruisce + carica i pesi
    logits = I.predict(model, X)               # (N, n_classes)
    H = I.incidence(model, X)                   # (N, K, n_edges)  <- incidenza soft (interpretabilita')
"""
import os
import numpy as np
import torch
import track3_config as C
import track3_models as M

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "results", "checkpoints", "hypertempnet_best.pt")


def load_checkpoint(path: str = DEFAULT_CKPT, device=None):
    """Ricostruisce il modello dalla config nel checkpoint e carica i pesi. Ritorna (model, meta)."""
    device = device or C.get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = M.build_model(ckpt["model_name"], n_ch=ckpt["n_ch"], n_times=ckpt["n_times"],
                          **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    meta["device"] = device
    return model, meta


def _as_tensor(X, device):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    return X.to(device)


@torch.no_grad()
def predict(model, X, batch_size: int = 128, device=None):
    """Logits (N, n_classes). X: (N, n_ch, n_times) numpy o tensor."""
    device = device or next(model.parameters()).device
    X = _as_tensor(X, device)
    out = [model(X[i:i + batch_size]).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out, 0).numpy()


@torch.no_grad()
def incidence(model, X, batch_size: int = 128, device=None):
    """
    Incidenza soft H (N, K, n_edges): per ogni trial, quanto ogni segmento temporale (K nodi)
    attiva ciascun iperarco appreso. E' il meccanismo interno con cui il modello discrimina.
    """
    device = device or next(model.parameters()).device
    X = _as_tensor(X, device)
    res = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        h = torch.cat([b(xb.unsqueeze(1)) for b in model.branches], 1)   # (B,Fc,C,T)
        h = model.spatial(h).squeeze(2)                                  # (B,Fc,T)
        B, Fc, Tt = h.shape
        seg = Tt // model.K
        h = h[:, :, :seg * model.K].reshape(B, Fc, model.K, seg).mean(3) # (B,Fc,K)
        feat = model.proj(h.transpose(1, 2))                             # (B,K,hidden)
        Hh = torch.softmax(feat @ model.E.T / (model.hidden ** 0.5), dim=2)
        res.append(Hh.cpu().numpy())
    return np.concatenate(res, 0)
