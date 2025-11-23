"""
Interactive EEG Viewer for the ImaSpe thesis project.
Automatically finds the dataset path relative to the project root.
Allows to:
- Select the subject and load the corresponding .h5 file
- Choose the epoch and EEG channels to display
- Visualize EEG traces with real electrode names (10–20 system)
- Close safely without kernel crashes
"""

import numpy as np
import pandas as pd
import h5py
import mne
from pathlib import Path
import subprocess
import sys

# Try to import Tkinter and the TkAgg canvas. If that fails (commonly on macOS
# when the system Tcl/Tk is incompatible with the Python build), fall back to
# a non-interactive mode that saves a static PNG using the Agg backend.
TK_AVAILABLE = True
tk_import_error = None
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt

    # --- Visual style defaults ---
    # Use seaborn-whitegrid if available, otherwise fall back gracefully.
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        try:
            plt.style.use("seaborn")
        except Exception:
            plt.style.use("default")

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "lines.linewidth": 1.8,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": "#dddddd",
        "grid.alpha": 0.6,
    })

    from matplotlib import cm
    import numpy as _np


    def plot_eeg_traces(ax, times, data, ch_names, selected_idx,
                        stacked=True, spacing_factor=1, cmap="tab10", fill_alpha=0.06):
        """Helper to draw EEG traces with nicer aesthetics.

        - stacked: if True draw traces with vertical offsets and channel labels
        - else: overlay with legend
        """
        ax.clear()
        selected_idx = list(selected_idx)
        n = len(selected_idx)
        palette = cm.get_cmap(cmap, max(3, n))

        # Extract selected traces
        traces = _np.array([data[i, :] for i in selected_idx])

        if stacked and n > 1:
            # automatic spacing based on peak-to-peak
            p2p = _np.ptp(traces, axis=1)
            base_spacing = p2p.max() if p2p.max() > 0 else traces.std()
            spacing = base_spacing * spacing_factor
            offsets = _np.arange(n)[::-1] * spacing  # top-to-bottom

            for j, idx in enumerate(selected_idx):
                y = traces[j, :] + offsets[j]
                color = palette(j)
                ax.plot(times, y, color=color, linewidth=1.2, alpha=0.95)
                ax.fill_between(times, y, offsets[j], color=color, alpha=fill_alpha)

            # Label yticks with channel names at offsets
            ax.set_yticks(offsets)
            ax.set_yticklabels([ch_names[i] for i in selected_idx])
            ax.set_ylabel("")
            try:
                ax.spines["left"].set_visible(False)
            except Exception:
                pass
        else:
            for j, idx in enumerate(selected_idx):
                y = traces[j, :]
                color = palette(j)
                ax.plot(times, y, label=ch_names[idx], color=color, alpha=0.95,linewidth=1.2)
                ax.fill_between(times, y, y.min(), color=color, alpha=fill_alpha)
            ax.legend(loc="upper right")

        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_xlim(times[0], times[-1])
        ax.grid(True, which="both", axis="both", linestyle="-", linewidth=0.6)
        ax.margins(x=0)
        ax.tick_params(labelsize=9)


    def _check_tk_via_subprocess(python_executable: str) -> bool:
        """Try to create and destroy a Tk root in a separate process.

        This avoids aborting the current process when the system Tcl/Tk is incompatible.
        Returns True if the subprocess exits with code 0.
        """
        code = (
            "import tkinter as tk\n"
            "try:\n"
            "    r = tk.Tk()\n"
            "    r.update()\n"
            "    r.destroy()\n"
            "    print('TK_OK')\n"
            "except Exception as e:\n"
            "    print('TK_ERROR', e)\n"
            "    raise\n"
        )
        try:
            proc = subprocess.run([python_executable, "-c", code], capture_output=True, timeout=5)
            out = proc.stdout.decode(errors='ignore') + proc.stderr.decode(errors='ignore')
            return proc.returncode == 0
        except Exception:
            return False
except Exception as e:
    TK_AVAILABLE = False
    tk_import_error = e
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def load_epochs_from_h5(path_h5: Path, fs: int = 256):
    with h5py.File(path_h5, "r") as f:
        data = f["data"][:] # type: ignore
        labels = f["labels"][:] # type: ignore
        subject = f["subject"][()] # type: ignore

    n_epochs, n_channels, _ = data.shape # type: ignore
    # Try to read the custom montage; if it's missing, fall back to generic names.
    try:
        montage = mne.channels.read_custom_montage("src/io/ebneuro.locs")
        ch_names = montage.ch_names[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    except Exception:
        montage = None
        ch_names = [f"CH{i}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=["eeg"] * n_channels)

    epochs = mne.EpochsArray(data, info)
    labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in labels] # type: ignore
    epochs.metadata = pd.DataFrame({
        "label_name": labels,
        "subject_id": str(subject)
    })

    epochs.set_montage(montage, on_missing="ignore")
    return epochs


def show_eeg_viewer(meta_csv_path: Path):
    global TK_AVAILABLE, tk_import_error
    meta_df = pd.read_csv(meta_csv_path)
    meta_df["subject_id"] = meta_df["subject_id"].astype(str).str.strip()
    subjects = sorted(meta_df["subject_id"].unique().tolist())

    # Even if tkinter imported successfully, creating a Tk root can still
    # abort the process on macOS when Tcl/Tk is incompatible. Do a quick
    # subprocess check and fall back if it fails.
    if TK_AVAILABLE:
        ok = _check_tk_via_subprocess(sys.executable)
        if not ok:
            print("Detected Tk abort when creating GUI in subprocess; falling back to static PNG output.")
            TK_AVAILABLE = False
            tk_import_error = "subprocess Tk check failed"

    if not TK_AVAILABLE:
        # Provide a graceful fallback: save a static PNG of the first subject/epoch
        print("WARNING: Tkinter / TkAgg unavailable. Falling back to static PNG output.")
        print(f"Tk import error: {tk_import_error}")
        # Create a single static plot for the first subject and save it.
        try:
            row = meta_df.iloc[0]
            path_h5 = Path(row["path_h5"])
            if not path_h5.is_absolute():
                path_h5 = meta_csv_path.parent.parent / "processed" / path_h5.name
            epochs = load_epochs_from_h5(path_h5)
            ep_idx = 0
            data = epochs.get_data()[ep_idx]
            times = epochs.times
            selected_ch = epochs.ch_names[:4]
            ch_idx = [epochs.ch_names.index(ch) for ch in selected_ch]

            # Use the improved plotting helper for fallback as well and ensure
            # margins allow the title and y-labels to be visible.
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=False)
            label = epochs.metadata.iloc[ep_idx]["label_name"]
            subj = str(row["subject_id"]) if "subject_id" in row else str(epochs.metadata.iloc[ep_idx]["subject_id"])
            ax.set_title(f"Subject {subj} — Epoch {ep_idx} — Label: {label}")
            plot_eeg_traces(ax, times, data, epochs.ch_names, ch_idx, stacked=True)

            # Adjust margins to avoid clipping title / channel labels
            try:
                fig.tight_layout()
            except Exception:
                fig.subplots_adjust(left=0.18, right=0.98, top=0.88)

            out_dir = meta_csv_path.parent.parent / "figures"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"eeg_viewer_fallback_subject_{subj}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved static EEG plot to: {out_path}")
            print("To use the interactive viewer, install/update a compatible Tcl/Tk or use a Python build linked against a newer Tcl/Tk (e.g., python.org installer, Homebrew python, or conda).")
        except Exception as ex:
            print(f"Fallback failed: {ex}")
        return


    root = tk.Tk()
    root.title("EEG Dataset Viewer – ImaSpe Thesis")

    # Finestra fissa, non full-screen
    root.geometry("1400x800")   # puoi ridurre/es. "1200x700"

    ctrl = ttk.Frame(root, padding=10)
    ctrl.pack(side=tk.LEFT, fill=tk.Y)

    ttk.Label(ctrl, text="Select subject:", font=("Helvetica", 11, "bold")).pack(pady=(5, 5))
    subj_box = ttk.Combobox(ctrl, values=subjects, state="readonly", width=10)
    subj_box.current(0)
    subj_box.pack(pady=5)

    ttk.Label(ctrl, text="Label:", font=("Helvetica", 11, "bold")).pack(pady=(20, 5))
    label_box = ttk.Combobox(ctrl, values=[], state="readonly", width=20)
    label_box.pack(pady=5)

    ttk.Label(ctrl, text="EEG Channels:", font=("Helvetica", 11, "bold")).pack(pady=(20, 5))
    frame_channels = ttk.Frame(ctrl)
    frame_channels.pack(fill=tk.Y, expand=True)
    scrollbar = ttk.Scrollbar(frame_channels, orient="vertical")
    ch_list = tk.Listbox(
        frame_channels, selectmode="multiple",
        exportselection=False, width=20, height=25,
        yscrollcommand=scrollbar.set
    )
    scrollbar.config(command=ch_list.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    ch_list.pack(side=tk.LEFT, fill=tk.Y, expand=True)

    plot_frame = ttk.Frame(root)
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    current_epochs = None

    def load_subject(event=None):
        nonlocal current_epochs
        subj = subj_box.get()
        row = meta_df.query("subject_id == @subj").iloc[0]
        path_h5 = Path(row["path_h5"])
        if not path_h5.is_absolute():
            path_h5 = meta_csv_path.parent.parent / "processed" / path_h5.name
        print(f"Loading subject {subj}: {path_h5.name}")
        # remember previously selected channel names so selection can be
        # preserved across subject switches
        prev_selected = [ch_list.get(i) for i in ch_list.curselection()]

        try:
            current_epochs = load_epochs_from_h5(path_h5)
            # populate label combobox with unique labels for this subject
            labels = list(current_epochs.metadata["label_name"].unique())
            label_box["values"] = labels
            if labels:
                label_box.current(0)

            ch_list.delete(0, tk.END)
            for ch in current_epochs.ch_names:
                ch_list.insert(tk.END, ch)

            # restore previous channel selections where possible
            to_select = [i for i, ch in enumerate(current_epochs.ch_names) if ch in prev_selected]
            if to_select:
                for i in to_select:
                    ch_list.selection_set(i)
            else:
                # default selection: first four channels
                ch_list.selection_set(0, min(3, len(current_epochs.ch_names)-1))

            update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load file: {e}")

    def update_plot(event=None):
        if current_epochs is None:
            return
        # Determine epoch index by selected label (pick first matching epoch)
        selected_label = label_box.get() if label_box.get() else None
        if selected_label:
            try:
                ep_idx = int(next(i for i, l in enumerate(current_epochs.metadata["label_name"]) if l == selected_label))
            except StopIteration:
                ep_idx = 0
        else:
            ep_idx = 0
        selected = [ch_list.get(i) for i in ch_list.curselection()]
        if not selected:
            selected = current_epochs.ch_names[:4]
        ch_idx = [current_epochs.ch_names.index(ch) for ch in selected]
        data = current_epochs.get_data()[ep_idx]
        times = current_epochs.times
        label = current_epochs.metadata.iloc[ep_idx]["label_name"] # type: ignore

        label = current_epochs.metadata.iloc[ep_idx]["label_name"]  # type: ignore

        # Title: use a resilient figure-level text so it's not clipped by embedding/ttk
        title = f"Subject {subj_box.get()}  |  Label: {label}  |  Epoch: {ep_idx}"

        # Plot con helper
        plot_eeg_traces(ax, times, data, current_epochs.ch_names, ch_idx, stacked=True)

        # Ensure axis label sizes are reasonable (in case DPI scaling changed them)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

        # Layout: riduco il margine sinistro per avvicinare le tracce al pannello laterale
        # e riservo spazio in alto per il titolo tramite fig.text
        fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.18)

        # Remove previous figure texts (including prior title) and add centered title
        try:
            fig.texts.clear()
        except Exception:
            # older matplotlibs: reassign empty list
            try:
                fig.texts = []
            except Exception:
                pass
        try:
            fig.text(0.5, 0.92, title, ha="center", va="bottom", fontsize=11, fontweight="bold")
        except Exception:
            ax.set_title(title, fontsize=11, fontweight="bold")

        canvas.draw()
        # Forza aggiornamento del widget Tk per evitare problemi di rendering su macOS
        try:
            canvas.get_tk_widget().update_idletasks()
        except Exception:
            pass

    subj_box.bind("<<ComboboxSelected>>", load_subject)
    label_box.bind("<<ComboboxSelected>>", update_plot)
    ch_list.bind("<<ListboxSelect>>", update_plot)

    load_subject()

    def on_close():
        plt.close("all")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    meta_csv = project_root / "data" / "interim" / "eeg_metadata.csv"
    show_eeg_viewer(meta_csv)
