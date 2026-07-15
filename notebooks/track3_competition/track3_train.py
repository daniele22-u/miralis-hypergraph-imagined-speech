"""
track3_train.py — Loop di training subject-dependent, metriche, W&B opzionale, plotting.

Protocollo competizione: SUBJECT-DEPENDENT.
  Per ogni soggetto: train sul suo Training set, (early stopping sul Validation set),
  valutazione finale sul suo Test set (true label dall'answer sheet).
  Le metriche vengono poi mediate sui 15 soggetti.

Uso tipico (dal notebook):
    import track3_train as T
    df, results = T.run_subject_dependent("eegnet", feature="raw")
    T.plot_per_subject(df, "eegnet")
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

import track3_config as C
import track3_preproc as P
import track3_models as M


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def _evaluate(model, loader, device, criterion):
    model.eval()
    preds, tgts, losses = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        losses.append(criterion(out, yb).item() * len(yb))
        preds.append(out.argmax(1).cpu().numpy())
        tgts.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds); y_true = np.concatenate(tgts)
    loss = float(np.sum(losses) / len(y_true))
    return y_true, y_pred, loss


# ---------------------------------------------------------------------------
# Training di un singolo modello/soggetto
# ---------------------------------------------------------------------------
def train_model(model, data, device, *, epochs=200, lr=1e-3, weight_decay=1e-4,
                batch_size=32, patience=30, label_smoothing=0.0,
                wandb_run=None, verbose=False):
    """
    data: dict con X_train,y_train,X_val,y_val,X_test,y_test (numpy).
    Early stopping sulla balanced accuracy di validation. Ripristina i pesi migliori.
    label_smoothing: passato a CrossEntropyLoss (CBraMod usa 0.1).
    Ritorna dict con history e metriche di test (inclusa max_train_acc: capacità di fit).
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tl = _to_loader(data["X_train"], data["y_train"], batch_size, True)
    vl = _to_loader(data["X_val"], data["y_val"], batch_size, False)
    tel = _to_loader(data["X_test"], data["y_test"], batch_size, False)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_bacc": []}
    best_bacc, best_state, best_epoch, since = -1.0, None, 0, 0

    for epoch in range(epochs):
        model.train()
        run_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(yb)
            tr_correct += (out.argmax(1) == yb).sum().item()   # train acc "gratis" dai batch
            tr_total += len(yb)
        sched.step()
        train_loss = run_loss / len(data["y_train"])
        train_acc_ep = tr_correct / tr_total

        yv, pv, vloss = _evaluate(model, vl, device, criterion)
        vbacc = balanced_accuracy_score(yv, pv)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc_ep)
        hist["val_loss"].append(vloss)
        hist["val_bacc"].append(vbacc)
        if wandb_run is not None:
            wandb_run.log({"train/loss": train_loss, "val/loss": vloss,
                           "val/bacc": vbacc, "lr": sched.get_last_lr()[0], "epoch": epoch})
        if vbacc > best_bacc:
            best_bacc, best_epoch = vbacc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break
        if verbose and epoch % 20 == 0:
            print(f"    ep{epoch:3d} train_loss={train_loss:.3f} val_bacc={vbacc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # train accuracy del modello migliore -> distingue underfitting da overfitting:
    #   train_acc ~ chance  => il modello NON impara (bug/preprocessing) [underfitting]
    #   train_acc alta, test ~ chance => problema di generalizzazione/protocollo [overfitting]
    ytr_e, ptr_e, _ = _evaluate(model, tl, device, criterion)
    yt, pt, tloss = _evaluate(model, tel, device, criterion)
    res = {
        "history": hist,
        "best_epoch": best_epoch,
        "val_bacc": float(best_bacc),
        "train_acc": float(accuracy_score(ytr_e, ptr_e)),
        "max_train_acc": float(max(hist["train_acc"])) if hist["train_acc"] else 0.0,
        "test_acc": float(accuracy_score(yt, pt)),
        "test_bacc": float(balanced_accuracy_score(yt, pt)),
        "test_loss": tloss,
        "y_true": yt, "y_pred": pt,
        "confusion": confusion_matrix(yt, pt, labels=list(range(C.N_CLASSES))),
    }
    if wandb_run is not None:
        wandb_run.summary["val_bacc"] = res["val_bacc"]
        wandb_run.summary["test_acc"] = res["test_acc"]
        wandb_run.summary["test_bacc"] = res["test_bacc"]
    return res


# ---------------------------------------------------------------------------
# Preparazione input per tipo di feature
# ---------------------------------------------------------------------------
def _prepare_inputs(d, feature):
    """d = output di preprocess_subject. feature in {'raw','band'}."""
    if feature == "raw":
        return {
            "X_train": d["X_train"], "y_train": d["y_train"],
            "X_val": d["X_val"], "y_val": d["y_val"],
            "X_test": d["X_test"], "y_test": d["y_test"],
        }, {"n_times": d["X_train"].shape[2]}
    if feature == "band":
        fs = d["fs"]
        return {
            "X_train": P.band_features(d["X_train"], fs), "y_train": d["y_train"],
            "X_val": P.band_features(d["X_val"], fs), "y_val": d["y_val"],
            "X_test": P.band_features(d["X_test"], fs), "y_test": d["y_test"],
        }, {"in_feat": len(P.BANDS)}
    raise ValueError(feature)


# mappa modello -> tipo di feature / resample richiesto
_MODEL_SPEC = {
    "eegnet":  {"feature": "raw",  "resample": None},
    "shallow": {"feature": "raw",  "resample": None},
    "deep4":   {"feature": "raw",  "resample": None},
    "dhslp":   {"feature": "raw",  "resample": None},
    "dgcnn":   {"feature": "raw",  "resample": None},   # ora grafo PCC per-trial sul raw
    "reve":    {"feature": "raw",  "resample": C.REVE_FS},
}


# ---------------------------------------------------------------------------
# Runner subject-dependent su tutti i soggetti
# ---------------------------------------------------------------------------
def run_subject_dependent(model_name, subjects=None, *, merge_val=False,
                          model_kwargs=None, train_kwargs=None, pp_kwargs=None,
                          use_wandb=False, device=None, seed=42, verbose=True):
    """
    Allena `model_name` in modalità subject-dependent su tutti i soggetti.
    Ritorna (df_metriche, results_per_soggetto).
    """
    model_name = model_name.lower()
    spec = _MODEL_SPEC[model_name]
    subjects = subjects or C.SUBJECTS
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}
    pp_kwargs = dict(C.PP_MINIMAL) if pp_kwargs is None else pp_kwargs  # MINIMAL = standard (vedi README)
    device = device or C.get_device()
    set_seed(seed)

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print("[wandb non installato: pip install wandb — continuo senza logging]")

    rows, results = [], {}
    for s in subjects:
        t0 = time.time()
        d = P.preprocess_subject(s, merge_val_into_train=merge_val,
                                 resample_to=spec["resample"], **pp_kwargs)
        data, auto_kw = _prepare_inputs(d, spec["feature"])

        build_kw = dict(n_ch=C.N_CHANNELS, **auto_kw, **model_kwargs)
        if model_name == "reve":
            build_kw["electrode_names"] = d["clab"]
        model = M.build_model(model_name, **build_kw)

        run = None
        if wandb is not None:
            run = wandb.init(entity=C.WANDB_ENTITY, project=C.WANDB_PROJECT,
                             name=f"track3_{model_name}_S{s:02d}", reinit=True,
                             tags=["track3", model_name, f"S{s:02d}"],
                             config={"notebook": "track3", "model": model_name, "subject": s,
                                     "feature": spec["feature"], "merge_val": merge_val,
                                     **{k: v for k, v in {**train_kwargs}.items()}})
        res = train_model(model, data, device, wandb_run=run, **train_kwargs)
        if run is not None:
            run.finish()

        results[s] = res
        rows.append({"subject": s, "val_bacc": res["val_bacc"],
                     "test_acc": res["test_acc"], "test_bacc": res["test_bacc"],
                     "best_epoch": res["best_epoch"], "sec": round(time.time() - t0, 1)})
        if verbose:
            print(f"S{s:02d}  train_acc={res['train_acc']:.3f}  test_acc={res['test_acc']:.3f}  "
                  f"test_bacc={res['test_bacc']:.3f}  (val_bacc={res['val_bacc']:.3f}, {rows[-1]['sec']}s)")

    df = pd.DataFrame(rows).set_index("subject")
    if verbose:
        print(f"\n== {model_name} ==  test_acc mean={df.test_acc.mean():.3f}±{df.test_acc.std():.3f} "
              f"| test_bacc mean={df.test_bacc.mean():.3f} | chance={C.CHANCE_LEVEL:.2f}")
    return df, results


# ---------------------------------------------------------------------------
# Runner SUBJECT-MIXED (protocollo di CBraMod Table 9)
# ---------------------------------------------------------------------------
def run_subject_mixed(model_name, subjects=None, *, merge_val=False,
                      model_kwargs=None, train_kwargs=None, pp_kwargs=None,
                      use_wandb=False, device=None, seed=42, verbose=True,
                      standardize_per_subject=True):
    """
    Protocollo SUBJECT-MIXED (come CBraMod, Table 9): i 15 soggetti vengono messi
    in un UNICO dataset e si allena UN SOLO modello.
      train pool = tutti i training trial dei soggetti (~4500)
      val pool   = tutti i validation trial (~750) -> early stopping affidabile
      test pool  = tutti i test trial (~750), degli STESSI soggetti visti in training

    NB: NON è il protocollo ufficiale della competizione (che è subject-dependent).
    Serve per confrontarsi con i numeri di Table 9 del paper CBraMod.

    Standardizzazione: per default z-score PER SOGGETTO (stats del suo train) prima del
    pooling -> rimuove le differenze di scala inter-soggetto senza leakage.

    Ritorna (summary_df, res) dove:
      - res contiene anche res['test_subject'] (id soggetto per ogni test trial)
      - summary_df ha l'accuratezza per-soggetto SOTTO IL MODELLO MIXED + riga 'ALL'
    """
    model_name = model_name.lower()
    spec = _MODEL_SPEC[model_name]
    subjects = subjects or C.SUBJECTS
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}
    pp_kwargs = dict(C.PP_MINIMAL) if pp_kwargs is None else pp_kwargs  # MINIMAL = standard (vedi README)
    device = device or C.get_device()
    set_seed(seed)

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print("[wandb non installato: pip install wandb — continuo senza logging]")

    # --- accumula e poola tutti i soggetti ---
    parts = {k: [] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte")}
    test_subj = []
    clab = None
    auto_kw = {}
    t0 = time.time()
    for s in subjects:
        d = P.preprocess_subject(s, merge_val_into_train=merge_val,
                                 resample_to=spec["resample"],
                                 standardize=standardize_per_subject, **pp_kwargs)
        data, auto_kw = _prepare_inputs(d, spec["feature"])
        parts["Xtr"].append(data["X_train"]); parts["ytr"].append(data["y_train"])
        parts["Xva"].append(data["X_val"]);   parts["yva"].append(data["y_val"])
        parts["Xte"].append(data["X_test"]);  parts["yte"].append(data["y_test"])
        test_subj.append(np.full(len(data["y_test"]), s))
        clab = d["clab"]

    pooled = {
        "X_train": np.concatenate(parts["Xtr"], 0), "y_train": np.concatenate(parts["ytr"], 0),
        "X_val": np.concatenate(parts["Xva"], 0),   "y_val": np.concatenate(parts["yva"], 0),
        "X_test": np.concatenate(parts["Xte"], 0),  "y_test": np.concatenate(parts["yte"], 0),
    }
    test_subj = np.concatenate(test_subj, 0)
    if verbose:
        print(f"pool: train={pooled['X_train'].shape[0]}  val={pooled['X_val'].shape[0]}  "
              f"test={pooled['X_test'].shape[0]}  (input {pooled['X_train'].shape[1:]})")

    # --- costruisci e allena UN modello ---
    build_kw = dict(n_ch=C.N_CHANNELS, **auto_kw, **model_kwargs)
    if model_name == "reve":
        build_kw["electrode_names"] = clab
    model = M.build_model(model_name, **build_kw)

    run = None
    if wandb is not None:
        run = wandb.init(entity=C.WANDB_ENTITY, project=C.WANDB_PROJECT,
                         name=f"track3mixed_{model_name}", reinit=True,
                         tags=["track3", "subject-mixed", model_name],
                         config={"notebook": "track3", "model": model_name,
                                 "protocol": "subject-mixed", "feature": spec["feature"],
                                 "n_train": pooled["X_train"].shape[0]})
    res = train_model(model, pooled, device, wandb_run=run, **train_kwargs)
    if run is not None:
        run.finish()
    res["test_subject"] = test_subj

    # --- breakdown per-soggetto sotto il modello mixed ---
    rows = []
    for s in subjects:
        m = test_subj == s
        rows.append({"subject": int(s),
                     "test_acc": float(accuracy_score(res["y_true"][m], res["y_pred"][m])),
                     "test_bacc": float(balanced_accuracy_score(res["y_true"][m], res["y_pred"][m]))})
    df = pd.DataFrame(rows).set_index("subject")
    df.loc["ALL"] = [res["test_acc"], res["test_bacc"]]
    res["elapsed_sec"] = round(time.time() - t0, 1)

    if verbose:
        print(f"\n== {model_name} SUBJECT-MIXED ==  max_train_acc={res['max_train_acc']:.3f}  "
              f"test_acc={res['test_acc']:.3f}  test_bacc={res['test_bacc']:.3f}  "
              f"(val_bacc={res['val_bacc']:.3f}, {res['elapsed_sec']}s, chance={C.CHANCE_LEVEL:.2f})")
        if res['max_train_acc'] < 0.35:
            print("  ⚠️ max_train_acc ~ chance => il modello NON riesce a fittare nemmeno il "
                  "training = problema di OTTIMIZZAZIONE (LR, ecc.), non di dati.")
        elif res['test_acc'] < 0.30:
            print("  ⚠️ fitta il training ma test ~ chance => è un SOFFITTO di generalizzazione "
                  "(imagined speech è così), non un bug.")
    return df, res


# ---------------------------------------------------------------------------
# Runner SUBJECT-INDEPENDENT (cross-subject: soggetti di test MAI visti)
# ---------------------------------------------------------------------------
def _subject_bundle(subject, feature, spec, pp_kwargs):
    """Tutti i trial di un soggetto (train+val+test concatenati), già preprocessati."""
    d = P.preprocess_subject(subject, resample_to=spec["resample"], **pp_kwargs)
    data, auto_kw = _prepare_inputs(d, feature)
    X = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], 0)
    y = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], 0)
    return X, y, auto_kw, d["clab"]


def run_subject_independent(model_name, *, mode="holdout",
                            train_subjects=None, val_subjects=None, test_subjects=None,
                            model_kwargs=None, train_kwargs=None, pp_kwargs=None,
                            use_wandb=False, device=None, seed=42, verbose=True):
    """
    SUBJECT-INDEPENDENT (cross-subject): i soggetti di TEST non compaiono MAI in training.
    È il protocollo più difficile: misura la vera generalizzazione tra soggetti.

    mode='holdout' (default, veloce): UN modello.
        train_subjects (def 1-11) -> training
        val_subjects   (def 12-13) -> early stopping (soggetti held-out)
        test_subjects  (def 14-15) -> test (soggetti held-out)
    mode='loso' (leave-one-subject-out, 15 modelli):
        per ogni soggetto s: test = s, val = 2 soggetti (diversi da s), train = i restanti 12.

    Ritorna (df, results). Standardizzazione per-soggetto (nessun leakage cross-subject).
    """
    model_name = model_name.lower()
    spec = _MODEL_SPEC[model_name]
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}
    pp_kwargs = dict(C.PP_MINIMAL) if pp_kwargs is None else pp_kwargs  # MINIMAL = standard (vedi README)
    device = device or C.get_device()
    set_seed(seed)

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print("[wandb non installato: pip install wandb — continuo senza logging]")

    # cache dei bundle per soggetto (evita di ri-preprocessare in LOSO)
    cache = {}
    clab = [None]

    def bundle(s):
        if s not in cache:
            X, y, auto_kw, cl = _subject_bundle(s, spec["feature"], spec, pp_kwargs)
            cache[s] = (X, y, auto_kw)
            clab[0] = cl
        return cache[s]

    def _pool(subjs):
        Xs, ys = [], []
        auto_kw = {}
        for s in subjs:
            X, y, auto_kw = bundle(s)
            Xs.append(X); ys.append(y)
        return np.concatenate(Xs, 0), np.concatenate(ys, 0), auto_kw

    def _fit_eval(tr_subj, va_subj, te_subj, tag):
        Xtr, ytr, auto_kw = _pool(tr_subj)
        Xva, yva, _ = _pool(va_subj)
        Xte, yte, _ = _pool(te_subj)
        build_kw = dict(n_ch=C.N_CHANNELS, **auto_kw, **model_kwargs)
        if model_name == "reve":
            build_kw["electrode_names"] = clab[0]
        model = M.build_model(model_name, **build_kw)
        run = None
        if wandb is not None:
            run = wandb.init(entity=C.WANDB_ENTITY, project=C.WANDB_PROJECT,
                             name=f"track3indep_{model_name}_{tag}", reinit=True,
                             tags=["track3", "subject-independent", model_name],
                             config={"model": model_name, "protocol": "subject-independent",
                                     "mode": mode, "test": list(te_subj)})
        res = train_model(model, {"X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva,
                                  "X_test": Xte, "y_test": yte}, device, wandb_run=run, **train_kwargs)
        if run is not None:
            run.finish()
        return res

    t0 = time.time()
    if mode == "holdout":
        train_subjects = train_subjects or list(range(1, 12))   # 1-11
        val_subjects = val_subjects or [12, 13]
        test_subjects = test_subjects or [14, 15]
        res = _fit_eval(train_subjects, val_subjects, test_subjects, "holdout")
        df = pd.DataFrame([{"test_subjects": str(test_subjects),
                            "max_train_acc": res["max_train_acc"],
                            "test_acc": res["test_acc"], "test_bacc": res["test_bacc"],
                            "val_bacc": res["val_bacc"]}])
        if verbose:
            print(f"== {model_name} SUBJECT-INDEPENDENT (holdout, test={test_subjects}) ==  "
                  f"max_train_acc={res['max_train_acc']:.3f}  test_acc={res['test_acc']:.3f}  "
                  f"test_bacc={res['test_bacc']:.3f}  ({round(time.time()-t0,1)}s, "
                  f"chance={C.CHANCE_LEVEL:.2f})")
        return df, {"holdout": res}

    elif mode == "loso":
        subs = list(C.SUBJECTS)
        rows, results = [], {}
        for i, s in enumerate(subs):
            others = [x for x in subs if x != s]
            va = [others[i % len(others)], others[(i + 1) % len(others)]]  # 2 soggetti come val
            tr = [x for x in others if x not in va]
            res = _fit_eval(tr, va, [s], f"loso_S{s:02d}")
            results[s] = res
            rows.append({"subject": s, "test_acc": res["test_acc"],
                         "test_bacc": res["test_bacc"], "val_bacc": res["val_bacc"]})
            if verbose:
                print(f"LOSO test S{s:02d}: test_acc={res['test_acc']:.3f} "
                      f"test_bacc={res['test_bacc']:.3f}")
        df = pd.DataFrame(rows).set_index("subject")
        df.loc["MEAN"] = [df.test_acc.mean(), df.test_bacc.mean(), df.val_bacc.mean()]
        if verbose:
            print(f"\n== {model_name} SUBJECT-INDEPENDENT (LOSO) ==  "
                  f"test_acc mean={df.loc['MEAN','test_acc']:.3f} | chance={C.CHANCE_LEVEL:.2f}")
        return df, results

    raise ValueError(f"mode sconosciuto: {mode} (usa 'holdout' o 'loso')")


# ---------------------------------------------------------------------------
# Baseline classico (sanity check: c'è segnale decodificabile senza deep learning?)
# ---------------------------------------------------------------------------
def classical_baseline(subjects=None, *, pp_kwargs=None, protocol="mixed", verbose=True):
    """
    LDA su band-power (differential entropy per canale/banda). Veloce, CPU.
    Se anche questo è a chance con un preprocessing ma sale con un altro,
    il problema è il PREPROCESSING, non il modello deep.
      protocol='mixed'      -> pool 15 soggetti, un LDA
      protocol='dependent'  -> un LDA per soggetto, media
    Ritorna il dict dei risultati.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    subjects = subjects or C.SUBJECTS
    pp_kwargs = dict(C.PP_MINIMAL) if pp_kwargs is None else pp_kwargs  # MINIMAL = standard (vedi README)

    def _feat(d, split):
        bf = P.band_features(d[f"X_{split}"], d["fs"])
        return bf.reshape(bf.shape[0], -1)

    if protocol == "dependent":
        accs = []
        for s in subjects:
            d = P.preprocess_subject(s, **pp_kwargs)
            clf = LDA().fit(_feat(d, "train"), d["y_train"])
            accs.append(clf.score(_feat(d, "test"), d["y_test"]))
        acc = float(np.mean(accs))
        if verbose:
            print(f"LDA band-power SUBJECT-DEPENDENT: test_acc medio={acc:.3f} "
                  f"(chance {C.CHANCE_LEVEL})")
        return {"test_acc": acc, "per_subject": accs}

    Xtr, ytr, Xte, yte = [], [], [], []
    for s in subjects:
        d = P.preprocess_subject(s, **pp_kwargs)
        Xtr.append(_feat(d, "train")); ytr.append(d["y_train"])
        Xte.append(_feat(d, "test"));  yte.append(d["y_test"])
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    Xte, yte = np.concatenate(Xte), np.concatenate(yte)
    clf = LDA().fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    if verbose:
        print(f"LDA band-power SUBJECT-MIXED: test_acc={acc:.3f} "
              f"(train {clf.score(Xtr, ytr):.3f}, chance {C.CHANCE_LEVEL})")
    return {"test_acc": acc}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(results, subject, ax=None):
    import matplotlib.pyplot as plt
    h = results[subject]["history"]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h["train_loss"], label="train loss")
    ax.plot(h["val_loss"], label="val loss")
    ax2 = ax.twinx()
    if h.get("train_acc"):
        ax2.plot(h["train_acc"], color="orange", label="train acc")
    ax2.plot(h["val_bacc"], color="green", label="val bAcc")
    ax2.axhline(C.CHANCE_LEVEL, color="grey", ls="--", lw=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax2.set_ylabel("balanced accuracy")
    ax.set_title(f"S{subject:02d} training")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    return ax


def plot_per_subject(df, model_name, ax=None, save=True):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df.index.astype(str), df.test_acc, color="#4C72B0")
    ax.axhline(C.CHANCE_LEVEL, color="red", ls="--", label=f"chance {C.CHANCE_LEVEL:.0%}")
    ax.axhline(df.test_acc.mean(), color="black", ls="-",
               label=f"mean {df.test_acc.mean():.3f}")
    ax.set_xlabel("subject"); ax.set_ylabel("test accuracy")
    ax.set_ylim(0, max(0.6, df.test_acc.max() * 1.15))
    ax.set_title(f"{model_name} — subject-dependent test accuracy (5 classes)")
    ax.legend()
    if save:
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(C.FIG_DIR / f"per_subject_{model_name}.png", dpi=130)
    return ax


def plot_confusion(results, subject=None, model_name="model", ax=None, save=True):
    """Confusion matrix aggregata (tutti i soggetti) o di un singolo soggetto."""
    import matplotlib.pyplot as plt
    if subject is not None:
        cm = results[subject]["confusion"]
        title = f"{model_name} — S{subject:02d}"
    else:
        cm = np.sum([r["confusion"] for r in results.values()], axis=0)
        title = f"{model_name} — aggregata (15 soggetti)"
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C.N_CLASSES)); ax.set_yticks(range(C.N_CLASSES))
    ax.set_xticklabels(C.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(C.CLASS_NAMES)
    for i in range(C.N_CLASSES):
        for j in range(C.N_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    if save:
        plt.tight_layout()
        plt.savefig(C.FIG_DIR / f"confusion_{model_name}.png", dpi=130)
    return ax


def save_metrics(df, model_name):
    out = C.RESULTS_DIR / f"metrics_{model_name}.csv"
    df.to_csv(out)
    return out
