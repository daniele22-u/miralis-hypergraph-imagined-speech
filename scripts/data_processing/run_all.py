"""
Convenience launcher for the end-to-end EEG preprocessing pipeline.

This script is a thin wrapper around
`scripts/data_processing/Preprocessing-main/src/Python/main.py`.  It wires
default paths for the dataset, MATLAB scripts, EEGLAB installation, and channel
locations, then delegates all heavy lifting to that CLI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict


def _default_paths() -> Dict[str, Path]:
    """
    Infer repository-relative defaults needed by the preprocessing CLI.
    """
    project_root = Path(__file__).resolve().parents[2]
    pipeline_root = project_root / "scripts" / "data_processing" / "Preprocessing-main"

    return {
        "project_root": project_root,
        "pipeline_root": pipeline_root,
        "main_script": pipeline_root / "src" / "Python" / "main.py",
        "matlab_scripts": pipeline_root / "src" / "Matlab",
        "chanlocs": pipeline_root / "ebneuro.locs",
        "data_dir": project_root / "data" / "raw_data",
        "output_dir": project_root / "data" / "processed",
        "eeglab": project_root / "scripts" / "data_processing" / "eeglab2025.1.0",
    }


def _build_parser(defaults: Dict[str, Path]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EEG preprocessing pipeline using repository defaults."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=defaults["data_dir"],
        help="Directory containing raw .xdf recordings (default: repo data/raw_data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults["output_dir"],
        help="Destination directory for generated .h5 files (default: repo data/processed).",
    )
    parser.add_argument(
        "--matlab-scripts",
        type=Path,
        default=defaults["matlab_scripts"],
        help="Path to MATLAB preprocessing scripts (default: Preprocessing-main/src/Matlab).",
    )
    parser.add_argument(
        "--eeglab",
        type=Path,
        default=defaults["eeglab"],
        help="Path to the EEGLAB installation (default: scripts/data_processing/eeglab2025.1.0).",
    )
    parser.add_argument(
        "--chanlocs",
        type=Path,
        default=defaults["chanlocs"],
        help="Channel locations file to use (default: Preprocessing-main/ebneuro.locs).",
    )

    return parser


def main() -> None:
    defaults = _default_paths()
    parser = _build_parser(defaults)
    args = parser.parse_args()

    main_script = defaults["main_script"]
    if not main_script.exists():
        print(f"❌ Cannot locate preprocessing entry point: {main_script}")
        sys.exit(1)

    missing_paths = [
        ("data directory", args.data_dir),
        ("MATLAB scripts", args.matlab_scripts),
        ("EEGLAB directory", args.eeglab),
        ("chanlocs file", args.chanlocs),
    ]

    for label, path in missing_paths:
        if not path.exists():
            print(f"⚠️  Warning: {label} not found at {path}")

    command = [
        sys.executable,
        str(main_script),
        str(args.data_dir),
        "-o",
        str(args.output_dir),
        "--matlab-scripts",
        str(args.matlab_scripts),
        "--eeglab",
        str(args.eeglab),
        "--chanlocs",
        str(args.chanlocs),
    ]

    print("🔹 Starting EEG preprocessing pipeline via CLI...")
    print(f"📂 Raw data: {args.data_dir}")
    print(f"📂 Output dir: {args.output_dir}")
    print(f"🧠 MATLAB scripts: {args.matlab_scripts}")
    print(f"🧩 EEGLAB: {args.eeglab}")
    print(f"📜 Chanlocs: {args.chanlocs}\n")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Pipeline failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)

    print("🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
