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


# ===========================================================================
# RIEMANN — clustering su matrici di covarianza SPD (varietà, non euclideo).
# Portato da V5W_07_riemann.ipynb. La metrica affine-invariante ignora la scala
# globale => lo split è per STRUTTURA di connettività, non per ampiezza/SNR.
# ===========================================================================
def frontopolar_keep_idx(clab):
    """Indici canali TENUTI escludendo i frontopolari/EOG (per il controllo artifact-robust)."""
    drop = {"Fp1", "Fp2", "FPz", "Fpz", "AF7", "AF8", "AF3", "AF4", "AFz"}
    return [i for i, c in enumerate(clab) if c not in drop]


def load_covariances(subjects=None, band=None, estimator="oas", keep_idx=None):
    """
    Covarianze SPD per-trial (pyriemann) + media di Riemann per soggetto.
    band: (lo,hi) Hz per filtrare prima (es. (1,30) per artifact-robust). keep_idx: sottoinsieme canali.
    Ritorna dict: covs (Ntot,c,c), subj, split ('train'/'val'/'test'), y, M (n_subj,c,c), ids, clab.
    """
    import numpy as np
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann
    import track3_io as io
    import track3_preproc as P
    subjects = subjects or C.SUBJECTS
    ce = Covariances(estimator=estimator)
    covs_all, subj_all, split_all, y_all = [], [], [], []
    M, ids, clab_used = [], [], None
    for s in subjects:
        tr, va, te = io.load_subject_all(s)
        clab = tr.clab if keep_idx is None else [tr.clab[i] for i in keep_idx]
        clab_used = clab
        subj_covs = []
        for name, sd in (("train", tr), ("val", va), ("test", te)):
            X = sd.X.astype(np.float64)
            if band is not None:
                X = P.bandpass(X, sd.fs, band[0], band[1]).astype(np.float64)
            if keep_idx is not None:
                X = X[:, keep_idx, :]
            Ct = ce.transform(X)                       # (N, c, c) SPD
            covs_all.append(Ct); subj_all += [s] * len(Ct)
            split_all += [name] * len(Ct); y_all += sd.y.tolist()
            subj_covs.append(Ct)
        M.append(mean_riemann(np.concatenate(subj_covs))); ids.append(s)
    return dict(covs=np.concatenate(covs_all), subj=np.array(subj_all),
                split=np.array(split_all), y=np.array(y_all),
                M=np.stack(M), ids=ids, clab=clab_used)


def tangent_features(M):
    """Proietta le medie-soggetto nel tangent space di Riemann (feature euclidee valide lì)."""
    from pyriemann.tangentspace import TangentSpace
    return TangentSpace().fit(M).transform(M)          # (n_subj, n_pairs)


def riemann_cluster(M, k=2, seed=42):
    """Media-soggetto SPD -> tangent space -> StandardScaler/PCA/KMeans. Ritorna (labels, TS, Z)."""
    TS = tangent_features(M)
    labels, Z, _, _ = cluster_subjects(TS, k=k, seed=seed)
    return labels, TS, Z


def det1_normalize(M):
    """Normalizza ogni matrice a determinante 1 (rimuove la scala globale)."""
    import numpy as np
    out = []
    for Ci in M:
        # pre-scala a O(1) per evitare overflow di slogdet su covarianze in µV² (non cambia il det=1)
        Ci = Ci / (np.trace(Ci) / Ci.shape[0] + 1e-12)
        _, ld = np.linalg.slogdet(Ci)
        out.append(Ci * np.exp(-ld / Ci.shape[0]))
    return np.stack(out)


def riemann_decoding(data):
    """
    Decoding Riemann subject-specific (gold standard EEG classico): MDM e TangentSpace+LogReg.
    Train su (train+val), test su test. Ritorna lista (subj, bacc_MDM, bacc_TSLR).
    """
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace
    rows = []
    for s in data["ids"]:
        m = data["subj"] == s
        covs_s, split_s, y_s = data["covs"][m], data["split"][m], data["y"][m]
        tr = np.isin(split_s, ["train", "val"]); te = split_s == "test"
        Xtr, ytr, Xte, yte = covs_s[tr], y_s[tr], covs_s[te], y_s[te]
        try:
            b_mdm = balanced_accuracy_score(yte, MDM().fit(Xtr, ytr).predict(Xte))
        except Exception:
            b_mdm = np.nan
        try:
            pipe = make_pipeline(TangentSpace(), LogisticRegression(max_iter=1000, C=1.0))
            b_ts = balanced_accuracy_score(yte, pipe.fit(Xtr, ytr).predict(Xte))
        except Exception:
            b_ts = np.nan
        rows.append((int(s), float(b_mdm), float(b_ts)))
    return rows


def silhouette_permutation(TS, k=2, n_perm=1000, seed=42):
    """Permutation test sulla silhouette nel tangent space (chiude il circular analysis)."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    def _sil(feat):
        Z = PCA(min(20, feat.shape[0] - 1), random_state=seed).fit_transform(
            StandardScaler().fit_transform(feat))
        return silhouette_score(Z, KMeans(k, n_init=20, random_state=seed).fit_predict(Z))

    obs = _sil(TS)
    rng = np.random.default_rng(seed)
    n, p = TS.shape
    null = np.array([_sil(np.take_along_axis(TS, rng.random((n, p)).argsort(0), axis=0))
                     for _ in range(n_perm)])
    return float(obs), null, float((null >= obs).mean())


if __name__ == "__main__":
    feats, ids, mats = subject_fingerprints("plv", subjects=[1, 2, 3])
    print("fingerprints euclidee:", feats.shape, "ids:", ids)
    data = load_covariances(subjects=[1, 2, 3])
    print("covarianze:", data["covs"].shape, "M:", data["M"].shape)
    lbl, TS, Z = riemann_cluster(data["M"], k=2)
    print("riemann labels:", lbl, "TS:", TS.shape)
