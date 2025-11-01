"""
Interactive EEG Viewer for the ImaSpe thesis project.
Automatically finds the dataset path relative to the project root.
Allows to:
- Select the subject and load the corresponding .h5 file
- Choose the epoch and EEG channels to display
- Visualize EEG traces with real electrode names (10–20 system)
- Close safely without kernel crashes
"""

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import mne
from pathlib import Path


def load_epochs_from_h5(path_h5: Path, fs: int = 256):
    with h5py.File(path_h5, "r") as f:
        data = f["data"][:] # type: ignore
        labels = f["labels"][:] # type: ignore
        subject = f["subject"][()] # type: ignore

    n_epochs, n_channels, _ = data.shape # type: ignore
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")

    epochs = mne.EpochsArray(data, info)
    labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in labels] # type: ignore
    epochs.metadata = pd.DataFrame({
        "label_name": labels,
        "subject_id": str(subject)
    })

    epochs.set_montage(montage, on_missing="ignore")
    return epochs


def show_eeg_viewer(meta_csv_path: Path):
    meta_df = pd.read_csv(meta_csv_path)
    meta_df["subject_id"] = meta_df["subject_id"].astype(str).str.strip()
    subjects = sorted(meta_df["subject_id"].unique().tolist())


    root = tk.Tk()
    root.title("EEG Dataset Viewer – ImaSpe Thesis")
    root.geometry("1400x900")
    root.configure(bg="#f7f7f7")

    ctrl = ttk.Frame(root, padding=10)
    ctrl.pack(side=tk.LEFT, fill=tk.Y)

    ttk.Label(ctrl, text="Select subject:", font=("Helvetica", 11, "bold")).pack(pady=(5, 5))
    subj_box = ttk.Combobox(ctrl, values=subjects, state="readonly", width=10)
    subj_box.current(0)
    subj_box.pack(pady=5)

    ttk.Label(ctrl, text="Epoch:", font=("Helvetica", 11, "bold")).pack(pady=(20, 5))
    epoch_box = ttk.Combobox(ctrl, values=[], state="readonly", width=10)
    epoch_box.pack(pady=5)

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

        try:
            current_epochs = load_epochs_from_h5(path_h5)
            epoch_box["values"] = list(range(len(current_epochs)))
            epoch_box.current(0)
            ch_list.delete(0, tk.END)
            for ch in current_epochs.ch_names:
                ch_list.insert(tk.END, ch)
            ch_list.selection_set(0, 3)
            update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load file: {e}")

    def update_plot(event=None):
        if current_epochs is None:
            return
        ep_idx = int(epoch_box.get()) if epoch_box.get() else 0
        selected = [ch_list.get(i) for i in ch_list.curselection()]
        if not selected:
            selected = current_epochs.ch_names[:4]
        ch_idx = [current_epochs.ch_names.index(ch) for ch in selected]
        data = current_epochs.get_data()[ep_idx]
        times = current_epochs.times
        label = current_epochs.metadata.iloc[ep_idx]["label_name"] # type: ignore

        ax.clear()
        for ci in ch_idx:
            ax.plot(times, data[ci, :], label=current_epochs.ch_names[ci])
        ax.set_title(f"Subject {subj_box.get()} — Epoch {ep_idx} — Label: {label}", fontsize=13)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude [µV]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        canvas.draw()

    subj_box.bind("<<ComboboxSelected>>", load_subject)
    epoch_box.bind("<<ComboboxSelected>>", update_plot)
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
