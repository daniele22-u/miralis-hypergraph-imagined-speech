"""
Interactive EEG Viewer for the ImaSpe thesis project.
Automatically finds the dataset path relative to the project root.
Allows to:
- Select the subject and load the corresponding .h5 file
- Choose the epoch and EEG channels to display
- Visualize EEG traces with real electrode names (10–20 system)
"""

import sys
import numpy as np
import pandas as pd
import h5py
import mne
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QListWidget, QAbstractItemView, QMessageBox
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib import cm

# --- Visual style defaults ---
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

def plot_eeg_traces(ax, times, data, ch_names, selected_idx,
                    stacked=True, spacing_factor=1, cmap="tab10", fill_alpha=0.06):
    """Helper to draw EEG traces with nicer aesthetics."""
    ax.clear()
    selected_idx = list(selected_idx)
    n = len(selected_idx)
    palette = plt.get_cmap(cmap, max(3, n))

    traces = np.array([data[i, :] for i in selected_idx])

    if stacked and n > 1:
        p2p = np.ptp(traces, axis=1)
        base_spacing = p2p.max() if p2p.max() > 0 else traces.std()
        spacing = base_spacing * spacing_factor
        offsets = np.arange(n)[::-1] * spacing  

        for j, idx in enumerate(selected_idx):
            y = traces[j, :] + offsets[j]
            color = palette(j)
            ax.plot(times, y, color=color, linewidth=1.2, alpha=0.95)
            ax.fill_between(times, y, offsets[j], color=color, alpha=fill_alpha)

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
            ax.plot(times, y, label=ch_names[idx], color=color, alpha=0.95, linewidth=1.2)
            ax.fill_between(times, y, y.min(), color=color, alpha=fill_alpha)
        ax.legend(loc="upper right")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_xlim(times[0], times[-1])
    ax.grid(True, which="both", axis="both", linestyle="-", linewidth=0.6)
    ax.margins(x=0)
    ax.tick_params(labelsize=9)


def load_epochs_from_h5(path_h5: Path, fs: int = 256):
    with h5py.File(path_h5, "r") as f:
        data = f["data"][:]
        labels = f["labels"][:]
        subject = f["subject"][()]

    n_epochs, n_channels, _ = data.shape
    try:
        montage = mne.channels.read_custom_montage("src/io/ebneuro.locs")
        ch_names = montage.ch_names[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    except Exception:
        montage = None
        ch_names = [f"CH{i}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=["eeg"] * n_channels)

    epochs = mne.EpochsArray(data, info)
    labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in labels]
    epochs.metadata = pd.DataFrame({
        "label_name": labels,
        "subject_id": str(subject)
    })

    epochs.set_montage(montage, on_missing="ignore")
    return epochs


class EEGViewer(QMainWindow):
    def __init__(self, meta_csv_path: Path):
        super().__init__()
        self.setWindowTitle("EEG Dataset Viewer – ImaSpe Thesis")
        self.resize(1400, 800)

        self.meta_csv_path = meta_csv_path
        self.meta_df = pd.read_csv(meta_csv_path)
        self.meta_df["subject_id"] = self.meta_df["subject_id"].astype(str).str.strip()
        self.subjects = sorted(self.meta_df["subject_id"].unique().tolist())
        
        self.current_epochs = None

        self._init_ui()
        self.load_subject()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        h_layout = QHBoxLayout(main_widget)

        # ---- Left Controls Panel ----
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setContentsMargins(10, 10, 10, 10)

        # Subject selection
        subj_label = QLabel("Select subject:")
        subj_label.setStyleSheet("font-weight: bold;")
        ctrl_layout.addWidget(subj_label)
        
        self.subj_combo = QComboBox()
        self.subj_combo.addItems(self.subjects)
        self.subj_combo.currentIndexChanged.connect(self.load_subject)
        ctrl_layout.addWidget(self.subj_combo)

        ctrl_layout.addSpacing(15)

        # Label selection
        lbl_label = QLabel("Label:")
        lbl_label.setStyleSheet("font-weight: bold;")
        ctrl_layout.addWidget(lbl_label)
        
        self.label_combo = QComboBox()
        self.label_combo.currentIndexChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.label_combo)

        ctrl_layout.addSpacing(15)

        # Channels selection
        ch_label = QLabel("EEG Channels:")
        ch_label.setStyleSheet("font-weight: bold;")
        ctrl_layout.addWidget(ch_label)

        self.ch_list = QListWidget()
        self.ch_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ch_list.itemSelectionChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.ch_list)

        ctrl_layout.addStretch()
        
        # Add controls to a widget to set a fixed width
        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl_layout)
        ctrl_widget.setFixedWidth(250)
        h_layout.addWidget(ctrl_widget)

        # ---- Right Plot Panel ----
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        h_layout.addWidget(self.canvas, stretch=1)

    def load_subject(self):
        subj = self.subj_combo.currentText()
        row = self.meta_df.query("subject_id == @subj").iloc[0]
        path_h5 = Path(row["path_h5"])
        
        if not path_h5.is_absolute():
            path_h5 = self.meta_csv_path.parent.parent / "processed" / path_h5.name
            
        print(f"Loading subject {subj}: {path_h5.name}")

        # Remember previous selection
        prev_selected = [i.text() for i in self.ch_list.selectedItems()]

        try:
            self.current_epochs = load_epochs_from_h5(path_h5)
            
            # Update labels
            self.label_combo.blockSignals(True)
            self.label_combo.clear()
            labels = list(self.current_epochs.metadata["label_name"].unique())
            self.label_combo.addItems(labels)
            if labels:
                self.label_combo.setCurrentIndex(0)
            self.label_combo.blockSignals(False)

            # Update channels
            self.ch_list.blockSignals(True)
            self.ch_list.clear()
            self.ch_list.addItems(self.current_epochs.ch_names)
            
            # Restore selection or select first 4
            to_select = [i for i in range(self.ch_list.count()) 
                         if self.ch_list.item(i).text() in prev_selected]
            
            if to_select:
                for idx in to_select:
                    self.ch_list.item(idx).setSelected(True)
            else:
                for idx in range(min(4, self.ch_list.count())):
                    self.ch_list.item(idx).setSelected(True)
            
            self.ch_list.blockSignals(False)

            self.update_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to load file:\n{str(e)}")

    def update_plot(self):
        if self.current_epochs is None:
            return

        selected_label = self.label_combo.currentText()
        if selected_label:
            try:
                ep_idx = int(next(i for i, l in enumerate(self.current_epochs.metadata["label_name"]) if l == selected_label))
            except StopIteration:
                ep_idx = 0
        else:
            ep_idx = 0

        selected_items = self.ch_list.selectedItems()
        if not selected_items:
            # Fallback if nothing selected but lists populated
            selected = self.current_epochs.ch_names[:4]
        else:
            selected = [item.text() for item in selected_items]
            
        ch_idx = [self.current_epochs.ch_names.index(ch) for ch in selected]
        
        data = self.current_epochs.get_data()[ep_idx]
        times = self.current_epochs.times
        label = self.current_epochs.metadata.iloc[ep_idx]["label_name"]

        subj = self.subj_combo.currentText()
        title = f"Subject {subj}  |  Label: {label}  |  Epoch: {ep_idx}"

        plot_eeg_traces(self.ax, times, data, self.current_epochs.ch_names, ch_idx, stacked=True)

        self.ax.xaxis.label.set_size(12)
        self.ax.yaxis.label.set_size(12)

        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.15)

        try:
            self.fig.texts.clear()
        except:
            self.fig.texts = []
            
        self.fig.text(0.5, 0.92, title, ha="center", va="bottom", fontsize=11, fontweight="bold")

        self.canvas.draw()


if __name__ == "__main__":
    import os
    import site
    import glob
    
    # Fix for Qt platform plugin "cocoa" error in Conda on macOS
    deps = []
    for p in site.getsitepackages():
        deps.extend(glob.glob(os.path.join(p, 'PySide6', 'Qt', 'plugins', 'platforms')))
    if deps:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = deps[0]

    project_root = Path(__file__).resolve().parents[2]
    meta_csv = project_root / "data" / "interim" / "eeg_metadata.csv"
    
    # Avoid Qt crash by setting an attribute before app creation
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    
    app = QApplication(sys.argv)
    viewer = EEGViewer(meta_csv)
    viewer.show()
    sys.exit(app.exec())
