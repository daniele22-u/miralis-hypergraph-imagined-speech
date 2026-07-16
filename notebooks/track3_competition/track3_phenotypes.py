"""
track3_phenotypes.py — Teoria dei fenotipi (EEG_16/16b) portata su Track#3.

Idea: ogni soggetto ha una FIRMA DI CONNETTIVITÀ (matrice di adiacenza media tra
elettrodi). Clusterizzando le firme, i soggetti si dividono in fenotipi (nella tesi:
C0 "fronto-motor" vs C1 "fronto-occipital"). Qui replichiamo su Track#3 (15 soggetti,
64 canali) e verifichiamo se il fenotipo predice la decodabilità.

Pipeline (come EEG_16b):
  firma = mean_trial( connettività per-trial )  -> triangolo superiore (2016-dim per 64 ch)
  StandardScaler -> PCA -> KMeans(k=2)
  analisi: Cohen's d per coppia, permutation test, differenza C1-C0, quality-check.

NB: 15 soggetti sono POCHI (la tesi ne aveva 74) -> risultato esplorativo, non definitivo.
"""
from __future__ import annotations
import numpy as np
import track3_config as C


def subject_fingerprints(metric: str = "plv", prune_k: int | None = None,
                         subjects=None, band=None, abs_val: bool = True):
    """
    Firma di connettività per soggetto (media sui trial), triangolo superiore.
    metric: 'plv' o 'pcc'. prune_k: se int, applica pruning top-k prima di mediare.
    band: (lo,hi) Hz per filtrare prima della connettività (None = broadband).
    Ritorna (feats (n_subj, n_pairs), subj_ids, adj_mats (n_subj, 64, 64)).
    """
    import torch
    import track3_io as io
    import track3_preproc as P
    import track3_graphs as G
    subjects = subjects or C.SUBJECTS
    triu = np.triu_indices(C.N_CHANNELS, k=1)
    feats, ids, mats = [], [], []
    for s in subjects:
        tr, va, te = io.load_subject_all(s)
        X = np.concatenate([tr.X, va.X, te.X], 0)          # (N, 64, T) tutti i trial
        if band is not None:
            X = P.bandpass(X, tr.fs, band[0], band[1])
        x = torch.as_tensor(X, dtype=torch.float32)
        A = G.pcc_adjacency(x) if metric == "pcc" else G.plv_adjacency(x)   # (N,64,64)
        if prune_k:
            A = G.topk_prune(A, prune_k)
        if abs_val:
            A = A.abs()
        Am = A.mean(0).numpy()                              # (64,64) media sui trial
        feats.append(Am[triu]); ids.append(s); mats.append(Am)
    return np.array(feats), ids, np.array(mats)


def cluster_subjects(feats, k: int = 2, seed: int = 42, n_pca: int = 20):
    """StandardScaler -> PCA -> KMeans. Ritorna (labels, Z, pca, scaler)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feats)
    pca = PCA(n_components=min(n_pca, len(feats) - 1), random_state=seed)
    Z = pca.fit_transform(Xs)
    labels = KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(Z)
    return labels, Z, pca, scaler


def silhouette_scan(feats, ks=(2, 3, 4, 5), seed: int = 42):
    """Silhouette per vari k (per capire quanti fenotipi ci sono)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    Xs = StandardScaler().fit_transform(feats)
    Z = PCA(n_components=min(20, len(feats) - 1), random_state=seed).fit_transform(Xs)
    out = {}
    for k in ks:
        if k >= len(feats):
            continue
        lbl = KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(Z)
        out[k] = float(silhouette_score(Z, lbl))
    return out


def cohens_d_pairs(feats, labels):
    """Cohen's d per ogni coppia di elettrodi tra cluster 0 e 1. d>0 => C0 più forte."""
    a, b = feats[labels == 0], feats[labels == 1]
    na, nb = len(a), len(b)
    pooled = np.sqrt((a.var(0) * na + b.var(0) * nb) / max(na + nb, 1) + 1e-12)
    return (a.mean(0) - b.mean(0)) / (pooled + 1e-12)


def permutation_test(feats, labels, n_perm: int = 5000, seed: int = 0):
    """
    Permuta le label cluster n_perm volte; p = frazione di permutazioni con max|d| >= osservato.
    Chiude l'obiezione di circular analysis (come §18 di EEG_16b).
    """
    rng = np.random.default_rng(seed)
    obs = np.abs(cohens_d_pairs(feats, labels)).max()
    count = sum(np.abs(cohens_d_pairs(feats, rng.permutation(labels))).max() >= obs
                for _ in range(n_perm))
    return float(obs), (count + 1) / (n_perm + 1)


def vec_to_sym(vec, n=C.N_CHANNELS):
    """Vettore triangolo superiore -> matrice simmetrica (n,n)."""
    M = np.zeros((n, n))
    M[np.triu_indices(n, k=1)] = vec
    return M + M.T


def node_strength_diff(mats, labels):
    """Per elettrodo: differenza di forza media di connettività C1 - C0. Ritorna (64,)."""
    m0 = mats[labels == 0].mean(0)
    m1 = mats[labels == 1].mean(0)
    return (m1 - m0).mean(1)   # media su j


if __name__ == "__main__":
    feats, ids, mats = subject_fingerprints("plv", subjects=[1, 2, 3])
    print("fingerprints:", feats.shape, "ids:", ids, "mats:", mats.shape)
    lbl, Z, pca, _ = cluster_subjects(feats, k=2)
    print("labels:", lbl, "PCA var:", pca.explained_variance_ratio_[:3])
