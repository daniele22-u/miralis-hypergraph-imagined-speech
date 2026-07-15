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
                batch_size=32, patience=30, wandb_run=None, verbose=False):
    """
    data: dict con X_train,y_train,X_val,y_val,X_test,y_test (numpy).
    Early stopping sulla balanced accuracy di validation. Ripristina i pesi migliori.
    Ritorna dict con history e metriche di test.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tl = _to_loader(data["X_train"], data["y_train"], batch_size, True)
    vl = _to_loader(data["X_val"], data["y_val"], batch_size, False)
    tel = _to_loader(data["X_test"], data["y_test"], batch_size, False)

    hist = {"train_loss": [], "val_loss": [], "val_bacc": []}
    best_bacc, best_state, best_epoch, since = -1.0, None, 0, 0

    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(yb)
        sched.step()
        train_loss = run_loss / len(data["y_train"])

        yv, pv, vloss = _evaluate(model, vl, device, criterion)
        vbacc = balanced_accuracy_score(yv, pv)
        hist["train_loss"].append(train_loss)
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

    yt, pt, tloss = _evaluate(model, tel, device, criterion)
    res = {
        "history": hist,
        "best_epoch": best_epoch,
        "val_bacc": float(best_bacc),
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
    "dgcnn":   {"feature": "band", "resample": None},
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
    pp_kwargs = pp_kwargs or {}
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
            print(f"S{s:02d}  test_acc={res['test_acc']:.3f}  test_bacc={res['test_bacc']:.3f}  "
                  f"(val_bacc={res['val_bacc']:.3f}, {rows[-1]['sec']}s)")

    df = pd.DataFrame(rows).set_index("subject")
    if verbose:
        print(f"\n== {model_name} ==  test_acc mean={df.test_acc.mean():.3f}±{df.test_acc.std():.3f} "
              f"| test_bacc mean={df.test_bacc.mean():.3f} | chance={C.CHANCE_LEVEL:.2f}")
    return df, results


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
