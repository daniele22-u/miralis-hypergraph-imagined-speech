"""
track3_graphs.py — Costruzione grafo PER-TRIAL + pruning (dense, batched).

Porta la tecnica "base" della tesi (grafo per-trial da connettività + pruning) in forma
dense, così i modelli restano compatibili con il trainer a tensori (niente PyG DataLoader).

Regola CLAUDE.md rispettata: 1 grafo per trial, adiacenza calcolata on-the-fly nel forward,
MAI un grafo statico condiviso.

Funzioni:
  pcc_adjacency(x)        (B,N,T) -> (B,N,N)  correlazione di Pearson tra canali, per trial
  plv_adjacency(x)        (B,N,T) -> (B,N,N)  phase-locking value (via Hilbert)
  topk_prune(A, k)        tiene i top-k |A| per riga, simmetrizza, azzera la diagonale
  normalize_adj(A)        D^-1/2 (A+I) D^-1/2  (batched, per graph conv tipo GCN)
"""
from __future__ import annotations
import torch


def pcc_adjacency(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pearson correlation tra canali, per ogni trial. x:(B,N,T) -> (B,N,N)."""
    xc = x - x.mean(dim=2, keepdim=True)
    std = xc.std(dim=2, keepdim=True).clamp(min=eps)
    xn = xc / std
    A = torch.bmm(xn, xn.transpose(1, 2)) / x.shape[2]     # (B,N,N) in [-1,1]
    return A


def plv_adjacency(x: torch.Tensor) -> torch.Tensor:
    """Phase-Locking Value tra canali via trasformata di Hilbert (approssimata su tutta la banda)."""
    # fase istantanea via Hilbert (torch non ha hilbert: usa FFT analytic signal)
    B, N, T = x.shape
    Xf = torch.fft.fft(x, dim=2)
    h = torch.zeros(T, device=x.device)
    if T % 2 == 0:
        h[0] = h[T // 2] = 1.0
        h[1:T // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(T + 1) // 2] = 2.0
    analytic = torch.fft.ifft(Xf * h.view(1, 1, T), dim=2)
    phase = torch.angle(analytic)                          # (B,N,T)
    # PLV(i,j) = |mean_t exp(i*(phi_i - phi_j))|
    z = torch.exp(1j * phase)                              # (B,N,T) complex
    A = (z @ z.conj().transpose(1, 2)).abs() / T           # (B,N,N)
    return A


def topk_prune(A: torch.Tensor, k: int) -> torch.Tensor:
    """
    Pruning: per ogni nodo tiene solo i k archi più forti (|A|), simmetrizza, azzera la diagonale.
    Ritorna l'adiacenza pesata (peso = |A| sugli archi tenuti).
    """
    B, N, _ = A.shape
    Aabs = A.abs()
    eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
    Aabs = Aabs - eye * 1e9                                # escludi self-loop dal top-k
    kk = min(k, N - 1)
    idx = Aabs.topk(kk, dim=2).indices                    # (B,N,kk)
    mask = torch.zeros_like(A)
    mask.scatter_(2, idx, 1.0)
    mask = ((mask + mask.transpose(1, 2)) > 0).to(A.dtype)  # simmetrico (arco se in almeno una direzione)
    return A.abs() * mask                                  # peso = |connettività|


def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """Normalizzazione simmetrica GCN: D^-1/2 (A+I) D^-1/2. Batched (B,N,N)."""
    B, N, _ = A.shape
    A = A + torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
    d = A.sum(2).clamp(min=1e-6)
    d_isqrt = d.pow(-0.5)
    return d_isqrt.unsqueeze(2) * A * d_isqrt.unsqueeze(1)


def build_adjacency(x: torch.Tensor, metric: str = "pcc", k: int = 8) -> torch.Tensor:
    """Pipeline completa: connettività -> pruning top-k -> normalizzazione. (B,N,T)->(B,N,N)."""
    A = pcc_adjacency(x) if metric == "pcc" else plv_adjacency(x)
    A = topk_prune(A, k)
    return normalize_adj(A)
