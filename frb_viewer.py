#!/usr/bin/env python3
"""
Simple FRB Viewer
=================

Browse .npy files containing frequency x time dynamic spectra, inspect the
burst, interactively select width/bandwidth, and run optional processing
functions from downsample.py and dedisperse.py.

Dependencies:
    pip install PySide6 matplotlib numpy

Run:
    python frb_viewer.py /path/to/npy_directory

You can also launch without an argument and choose the directory in the GUI.

Expected array shape:
    (n_frequency, n_time)

Optional integration:
---------------------
The GUI will try to import functions from downsample.py and dedisperse.py if
they are located beside this file. Because existing scripts often have
different APIs, the GUI includes an adapter section near the top. If your
functions have different names/signatures, edit the two adapter functions:

    downsample_array2d(...)
    run_dedisperse(...)

If the imports/functions cannot be used, the GUI falls back to built-in NumPy
downsampling and a no-op DM adjustment, while clearly indicating that in the
status bar.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover - optional dependency
    curve_fit = None

from fluxcal import fluxcal

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# Optional external-script adapters
# ---------------------------------------------------------------------------

def _load_local_module(filename: str, module_name: str):
    """Load a Python module from beside this script, if it exists."""
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DOWNSAMPLE_MODULE = _load_local_module("downsample.py", "_frb_downsample")



def _try_call(fn, candidates):
    """
    Try several common calling conventions.

    candidates is a list of (args, kwargs). Only TypeError is interpreted as
    a signature mismatch. Other exceptions are allowed to propagate.
    """
    last_error = None
    for args, kwargs in candidates:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No calling convention supplied.")


def _extract_array(result):
    """Extract an ndarray from common return formats."""
    if isinstance(result, np.ndarray):
        return result

    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, np.ndarray):
                return item

    if isinstance(result, dict):
        for key in ("array", "data", "result", "dedispersed", "output"):
            if key in result and isinstance(result[key], np.ndarray):
                return result[key]

    raise TypeError("Could not find a NumPy array in the function result.")


def _extract_float(result):
    """Extract the first scalar float from common return formats."""
    if isinstance(result, (float, int, np.floating, np.integer)):
        return float(result)

    if isinstance(result, dict):
        for key in ("dm", "optimized_dm", "best_dm", "opt_dm"):
            if key in result:
                try:
                    return float(result[key])
                except (TypeError, ValueError):
                    pass

    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, (float, int, np.floating, np.integer)):
                return float(item)

    return None


def build_fitburst_generic_input(
    data_full: np.ndarray,
    *,
    dm: float = 0.0,
    res_time: float = 40.96e-6,
    res_freq: float = 0.390625,
    freqs_bin0: float = 400.0,
    bad_chans: Optional[list[int]] = None,
    is_dedispersed: bool = False,
    arrival_time: Any = 0.0,
    ref_freq: float = 400.0,
    width_seconds: Optional[Any] = None,
    amplitude: Optional[Any] = None,
):
    """Build the fitburst-generic payload expected by fitburst backend DataReader."""
    data = np.asarray(data_full, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D dynamic spectrum, got shape {data.shape!r}")

    num_freq, num_time = data.shape
    if bad_chans is None:
        bad_chans = []
    else:
        bad_chans = [int(ch) for ch in bad_chans]

    metadata = {
        "bad_chans": bad_chans,
        "freqs_bin0": float(freqs_bin0),
        "is_dedispersed": bool(is_dedispersed),
        "num_freq": int(num_freq),
        "num_time": int(num_time),
        "times_bin0": 0.0,
        "res_freq": float(res_freq),
        "res_time": float(res_time),
    }

    # Accept either a single arrival/width or lists for multi-component bursts.
    if isinstance(arrival_time, (list, tuple, np.ndarray)):
        arrivals = [float(a) for a in arrival_time]
    else:
        arrivals = [float(arrival_time)]

    if width_seconds is None:
        # Default width: at least one sample, or the whole duration if that's larger.
        default_width = max(float(num_time) * float(res_time), float(res_time))
        widths = [default_width] * len(arrivals)
    elif isinstance(width_seconds, (list, tuple, np.ndarray)):
        widths = [float(w) for w in width_seconds]
    else:
        widths = [float(width_seconds)] * len(arrivals)

    if amplitude is None:
        amplitude_values = [float(np.log10(np.nanmax(np.abs(data)) + 1e-9)) for _ in arrivals]
    elif isinstance(amplitude, (list, tuple, np.ndarray)):
        amplitude_values = [float(a) for a in amplitude]
    else:
        amplitude_values = [float(amplitude) for _ in arrivals]

    # Ensure lengths match
    if len(widths) != len(arrivals):
        # If a single width provided for multiple arrivals, broadcast it.
        if len(widths) == 1:
            widths = widths * len(arrivals)
        else:
            raise ValueError("Length of arrival_time and width_seconds must match")
    if len(amplitude_values) != len(arrivals):
        if len(amplitude_values) == 1:
            amplitude_values = amplitude_values * len(arrivals)
        else:
            raise ValueError("Length of amplitude does not match arrival_time")

    burst_parameters = {
        "amplitude": amplitude_values,
        "arrival_time": [float(a) for a in arrivals],
        "burst_width": [float(w) for w in widths],
        "dm": [float(dm) for _ in arrivals],
        "dm_index": [-2.0 for _ in arrivals],
        "ref_freq": [float(ref_freq) for _ in arrivals],
        "scattering_index": [0.0 for _ in arrivals],
        "scattering_timescale": [0.0 for _ in arrivals],
        "spectral_index": [-2.0 for _ in arrivals],
        "spectral_running": [0.0 for _ in arrivals],
    }
    return data, metadata, burst_parameters


def downsample_array2d(array2d, downsample_factor_row=1, downsample_factor_col=1):
    """
    Downsample/average the rows of the 2D array by a factor of downsample_factor_row, and downsample/average the columns
    of the 2D array by a factor of downsample_factor_col.

    Parameters
    ----------
    array2d: 2D input array.
    downsample_factor_row: Downsampling factor in the row direction. Number of rows must be divisible by
    downsample_factor_row. Default is 1 (no downsampling).
    downsample_factor_col: Downsampling factor in the column direction. Number of columns must be divisible by
    downsample_factor_col. Default is 1 (no downsampling).

    Returns
    -------
    array2d_downsampled: Downsampled 2D output array.
    """

    nrow, ncol = np.shape(array2d)

    if downsample_factor_row < 1 or downsample_factor_col < 1:
        raise ValueError("Downsample factors must be >= 1")

    # Compute downsampled sizes using floor division so we can handle cases
    # where the input dimensions are not exactly divisible by the factors.
    ncol_downsampled = ncol // downsample_factor_col
    nrow_downsampled = nrow // downsample_factor_row

    if ncol_downsampled == 0 or nrow_downsampled == 0:
        raise ValueError("Downsample factor too large for array dimensions")

    # Trim any leftover rows/cols that don't fit evenly into the requested
    # downsampling shape. This is more robust than asserting and will match
    # common expectations (drop trailing samples).
    trim_cols = ncol_downsampled * downsample_factor_col
    trim_rows = nrow_downsampled * downsample_factor_row

    arr = np.asarray(array2d, dtype=float)[:trim_rows, :trim_cols].copy()

    # Downsample the array in the column direction.
    if downsample_factor_col > 1:
        arr = arr.reshape((trim_rows, ncol_downsampled, downsample_factor_col))
        arr = np.mean(arr, axis=2)

    # Downsample the array in the row direction.
    if downsample_factor_row > 1:
        arr = arr.reshape((nrow_downsampled, downsample_factor_row, ncol_downsampled))
        arr = np.mean(arr, axis=1)

    return arr

def dedisperse_incoherent(dyn_spec, dm, tsamp, df, k=(1.0/0.000241)):
    """Compatibility shim retained for older callers. This viewer intentionally does not
    perform dedispersion because the feature is not reliable for the loaded data.
    """
    return np.asarray(dyn_spec, dtype=float), dm


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        # Disable constrained_layout so we can control spacing precisely and
        # remove padding between subplots (we'll use subplots_adjust).
        self.figure = Figure(figsize=(10, 8), constrained_layout=False)
        super().__init__(self.figure)
        self.setParent(parent)

        # Layout: time series above dynamic spectrum, frequency spectrum to the right.
        # Use no spacing so the plots visually connect; axes are explicitly
        # shared so ticks/limits remain synchronized.
        gs = self.figure.add_gridspec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            hspace=0.0,
            wspace=0.0,
        )

        # Create the dynamic spectrum first, then attach the time-axis above
        # and the frequency-spectrum to the right so sharing is unambiguous.
        self.ax_dyn = self.figure.add_subplot(gs[1, 0])
        self.ax_time = self.figure.add_subplot(gs[0, 0], sharex=self.ax_dyn)
        # Frequency spectrum occupies the same row as the dynamic spectrum and
        # shares its y-axis. This makes the spectrum vertically align exactly
        # with the frequency axis of the dynamic spectrum.
        self.ax_freq = self.figure.add_subplot(gs[1, 1], sharey=self.ax_dyn)

        # Hide duplicate tick labels: the dyn spectrum shows the time ticks, so
        # the small time-series panel above should not duplicate numeric x labels.
        self.ax_time.xaxis.set_tick_params(labelbottom=False)

        # Remove adjoining spines so the panels appear connected.
        self.ax_time.spines["bottom"].set_visible(False)
        self.ax_dyn.spines["top"].set_visible(False)
        # Frequency spectrum sits to the right of dyn; hide its left spine so
        # it visually connects to the dyn panel.
        self.ax_freq.spines["left"].set_visible(False)
        # Show the shared frequency axis on the right for the frequency plot.
        self.ax_freq.yaxis.tick_right()
        self.ax_freq.yaxis.set_label_position("right")

        # Force zero padding so all three panels touch without gaps.
        # Small margins are kept to avoid clipping axis labels when enabled.
        self.figure.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.06, wspace=0.0, hspace=0.0)

        self._width_span: Optional[SpanSelector] = None
        self._band_span: Optional[SpanSelector] = None

        self._width_callback = None
        self._band_callback = None

    def clear(self):
        self.ax_dyn.clear()
        self.ax_time.clear()
        self.ax_freq.clear()
        self.draw_idle()

    def disable_selectors(self):
        if self._width_span is not None:
            self._width_span.set_active(False)
        if self._band_span is not None:
            self._band_span.set_active(False)
        self._width_span = None
        self._band_span = None

    def enable_width_selector(self, callback):
        self.disable_selectors()
        self._width_callback = callback
        self._width_span = SpanSelector(
            self.ax_time,
            self._on_width_selected,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.25),
            drag_from_anywhere=True,
        )

    def enable_bandwidth_selector(self, callback):
        self.disable_selectors()
        self._band_callback = callback
        self._band_span = SpanSelector(
            self.ax_freq,
            self._on_band_selected,
            "vertical",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.25),
            drag_from_anywhere=True,
        )

    def _on_width_selected(self, xmin, xmax):
        if self._width_callback is not None:
            self._width_callback(float(min(xmin, xmax)), float(max(xmin, xmax)))

    def _on_band_selected(self, ymin, ymax):
        if self._band_callback is not None:
            self._band_callback(float(min(ymin, ymax)), float(max(ymin, ymax)))


class FRBViewer(QMainWindow):
    DEFAULT_DM = 219.456

    @staticmethod
    def _is_bad_note(note: Any) -> bool:
        return str(note).strip().lower() == "bad"

    @staticmethod
    def _is_duplicate_note(note: Any) -> bool:
        return str(note).strip().lower() == "duplicate"

    def _is_excluded_note(self, note: Any) -> bool:
        return (
            (self.no_bad and self._is_bad_note(note))
            or (self.no_dups and self._is_duplicate_note(note))
        )

    def __init__(self, directory: Optional[str] = None, *, no_bad: bool = False, no_dups: bool = False):
        super().__init__()

        self.setWindowTitle("FRB Viewer")
        self.resize(1400, 1000)

        self.no_bad = no_bad
        self.no_dups = no_dups
        self.directory: Optional[Path] = None
        self.files: list[Path] = []
        self.index = 0

        self.original_array: Optional[np.ndarray] = None
        self.current_array: Optional[np.ndarray] = None

        # Per-burst metadata.
        self.results: dict[str, dict[str, Any]] = {}
        self._all_results: dict[str, dict[str, Any]] = {}

        # Current processing state.
        self.current_dm: Optional[float] = self.DEFAULT_DM
        # Default downsampling: time x16, frequency x4
        self.time_factor = 16
        self.freq_factor = 4

        # Physical resolutions: one frequency channel and one time sample.
        # Frequency units are assumed to be MHz unless you change the label.
        self.freq_resolution = 0.390625  # MHz per channel
        # Time resolution in seconds (40.96 microseconds)
        self.time_resolution = 40.96e-6  # seconds per sample

        # Interactive selections.
        self.width_start: Optional[float] = None
        self.width_end: Optional[float] = None
        self.band_start: Optional[float] = None
        self.band_end: Optional[float] = None

        # Last Gaussian fit results (stored to include in saved results).
        self._last_gaussian_fit: Optional[dict[str, Any]] = None
        # Multi-component fits: list of {'arrival_time': float, 'sigma_seconds': float}
        self._components: list[dict[str, float]] = []

        self._build_ui()

        if directory:
            self.load_directory(Path(directory))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top row: directory and burst identifier.
        top = QHBoxLayout()

        self.dir_label = QLabel("No directory selected")
        self.open_button = QPushButton("Open Directory")
        self.open_button.clicked.connect(self.choose_directory)

        self.burst_label = QLabel("No burst loaded")

        top.addWidget(self.open_button)
        top.addWidget(self.dir_label, 1)
        top.addWidget(self.burst_label)

        root.addLayout(top)

        # Plot.
        self.canvas = PlotCanvas(self)
        root.addWidget(self.canvas, 1)

        # Navigation.
        nav = QHBoxLayout()
        self.previous_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        self.previous_button.clicked.connect(self.previous_burst)
        self.next_button.clicked.connect(self.next_burst)

        self.position_label = QLineEdit("0 / 0")
        self.position_label.setAlignment(Qt.AlignCenter)
        self.position_label.setMaximumWidth(120)
        self.position_label.returnPressed.connect(self.go_to_position_from_text)
        self.position_label.editingFinished.connect(self.go_to_position_from_text)

        nav.addWidget(self.previous_button)
        nav.addWidget(self.position_label)
        nav.addWidget(self.next_button)
        root.addLayout(nav)

        # Selection buttons.
        selection = QHBoxLayout()

        self.width_button = QPushButton("Width Selection")
        self.width_button.clicked.connect(self.start_width_selection)

        self.band_button = QPushButton("Bandwidth Selection")
        self.band_button.clicked.connect(self.start_bandwidth_selection)

        self.accept_button = QPushButton("Accept Selection")
        self.accept_button.clicked.connect(self.accept_selection)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_selection)

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)

        selection.addWidget(self.width_button)
        selection.addWidget(self.band_button)
        selection.addWidget(self.accept_button)
        # Fit Gaussian button: measure TOA and width with uncertainties.
        self.fit_button = QPushButton("Fit Gaussian")
        self.fit_button.clicked.connect(self.fit_gaussian)
        selection.addWidget(self.fit_button)
        # Add Component: append the last fit as an additional burst component
        self.add_component_button = QPushButton("Add Component")
        self.add_component_button.clicked.connect(self.add_component)
        selection.addWidget(self.add_component_button)
        selection.addWidget(self.reset_button)
        selection.addWidget(self.zoom_in_button)
        selection.addWidget(self.zoom_out_button)
        selection.addWidget(self.reset_zoom_button)
        root.addLayout(selection)

        # Processing controls.
        processing = QHBoxLayout()

        processing.addWidget(QLabel("Time downsample:"))
        self.time_spin = QSpinBox()
        self.time_spin.setRange(1, 100000)
        self.time_spin.setValue(16)
        processing.addWidget(self.time_spin)

        self.time_button = QPushButton("Downsample Time")
        self.time_button.clicked.connect(self.downsample_time)
        processing.addWidget(self.time_button)

        processing.addWidget(QLabel("Freq. downsample:"))
        self.freq_spin = QSpinBox()
        self.freq_spin.setRange(1, 100000)
        self.freq_spin.setValue(4)
        processing.addWidget(self.freq_spin)

        self.freq_button = QPushButton("Downsample Freq.")
        self.freq_button.clicked.connect(self.downsample_frequency)
        processing.addWidget(self.freq_button)

        root.addLayout(processing)

        # DM.
        dm_row = QHBoxLayout()
        dm_row.addWidget(QLabel("DM:"))

        self.dm_spin = QDoubleSpinBox()
        self.dm_spin.setRange(-100000.0, 100000.0)
        self.dm_spin.setDecimals(6)
        self.dm_spin.setSingleStep(0.001)
        dm_row.addWidget(self.dm_spin)

        self.dm_button = QPushButton("Adjust DM")
        self.dm_button.clicked.connect(self.adjust_dm)
        dm_row.addWidget(self.dm_button)

        self.dm_display = QLabel("Current DM: —")
        dm_row.addWidget(self.dm_display, 1)

        root.addLayout(dm_row)

        # Notes.
        notes = QHBoxLayout()
        notes.addWidget(QLabel("Note:"))

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Enter a note for this burst...")
        notes.addWidget(self.note_edit, 1)

        self.note_button = QPushButton("Make a note")
        self.note_button.clicked.connect(self.save_note)
        notes.addWidget(self.note_button)

        self.export_fitburst_button = QPushButton("Export fitburst .npz")
        self.export_fitburst_button.clicked.connect(self.export_fitburst_npz)
        notes.addWidget(self.export_fitburst_button)

        self.fluxcal_button = QPushButton("Fluxcal")
        self.fluxcal_button.clicked.connect(self.run_fluxcal)
        notes.addWidget(self.fluxcal_button)

        root.addLayout(notes)

        # Measurement/status row.
        status = QHBoxLayout()
        self.width_label = QLabel("Width: —")
        self.band_label = QLabel("Bandwidth: —")
        self.fit_label = QLabel("Fit: —")
        self.processing_label = QLabel("")

        status.addWidget(self.width_label)
        status.addWidget(self.band_label)
        status.addWidget(self.fit_label)
        status.addWidget(self.processing_label, 1)

        root.addLayout(status)

        self.statusBar().showMessage("Choose a directory containing .npy files.")

        self._update_button_states()

    # ------------------------------------------------------------------
    # Directory / burst loading
    # ------------------------------------------------------------------

    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select directory containing .npy files",
        )
        if directory:
            self.load_directory(Path(directory))

    def load_directory(self, directory: Path):
        if not directory.is_dir():
            QMessageBox.critical(self, "Error", f"Not a directory:\n{directory}")
            return

        files = sorted(directory.glob("*.npy"))
        if not files:
            QMessageBox.warning(
                self,
                "No files found",
                f"No .npy files were found in:\n{directory}",
            )
            return

        self.directory = directory
        self.files = files

        if self.no_bad or self.no_dups:
            path = self._results_path()
            excluded_names = set()
            if path is not None and path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        excluded_names = {
                            key
                            for key, value in loaded.items()
                            if isinstance(value, dict) and self._is_excluded_note(value.get("note"))
                        }
                except (OSError, json.JSONDecodeError):
                    pass
            self.files = [p for p in files if p.name not in excluded_names]

        self.index = 0
        self._load_results()

        if not self.files:
            filter_names = []
            if self.no_bad:
                filter_names.append("--no-bad")
            if self.no_dups:
                filter_names.append("--no-dups")
            filter_text = " / ".join(filter_names) if filter_names else "the current filter"
            QMessageBox.warning(
                self,
                "No valid bursts",
                f"All .npy files in {directory} were excluded by the {filter_text} filter.",
            )
            return

        self.load_burst(0)

        self.dir_label.setText(str(directory))

    def _dm_delta_from_spin(self) -> float:
        return float(self.dm_spin.value()) if self.dm_spin is not None else 0.0

    def _dm_total_for_results(self) -> float:
        return float(self.DEFAULT_DM + self._dm_delta_from_spin())

    def _set_dm_spin_from_total(self, total_dm: float):
        self.dm_spin.setValue(float(total_dm) - self.DEFAULT_DM)

    def _load_results(self):
        """Load previously saved per-burst metadata from results JSON."""
        self.results = {}
        self._all_results = {}
        path = self._results_path()
        if path is None or not path.exists():
            return

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._all_results = loaded
                self.results = {}
                for key, value in loaded.items():
                    if not isinstance(value, dict):
                        self.results[key] = value
                        continue
                    if not self._is_excluded_note(value.get("note")):
                        self.results[key] = value
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt/partial results file should not prevent the bursts
            # themselves from being viewed.
            self.statusBar().showMessage(
                f"Could not load existing results file: {exc}"
            )

    def load_burst(self, index: int):
        if not self.files:
            return

        index = max(0, min(index, len(self.files) - 1))
        self.index = index

        path = self.files[index]

        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Could not load burst",
                f"Could not load:\n{path}\n\n{exc}",
            )
            return

        if array.ndim != 2:
            QMessageBox.critical(
                self,
                "Invalid array",
                f"{path.name} has shape {array.shape}.\n"
                "Expected a 2-D frequency × time array.",
            )
            return

        self.original_array = np.asarray(array, dtype=float)
        self.current_array = self.original_array.copy()

        entry = self.results.get(path.name, {})

        self.current_dm = entry.get("dm")
        # Use stored values when present, otherwise fall back to new defaults
        self.time_factor = int(entry.get("time_downsample", 16) or 16)
        self.freq_factor = int(entry.get("freq_downsample", 4) or 4)

        self.width_start = entry.get("width_start")
        self.width_end = entry.get("width_end")
        self.band_start = entry.get("band_start")
        self.band_end = entry.get("band_end")
        self._last_gaussian_fit = entry.get("gaussian_fit")
        # Load any previously saved components for this burst
        comps = entry.get("components", [])
        if isinstance(comps, list):
            cleaned = []
            for c in comps:
                if not isinstance(c, dict):
                    continue
                try:
                    component = {
                        "arrival_time": float(c.get("arrival_time")) if "arrival_time" in c else None,
                        "sigma_seconds": float(c.get("sigma_seconds")) if "sigma_seconds" in c else None,
                    }
                    for key in ("amplitude", "amplitude_err", "arrival_time_err", "sigma_seconds_err", "baseline", "baseline_err"):
                        if key in c:
                            component[key] = c[key]
                    if "gaussian_fit" in c:
                        component["gaussian_fit"] = c["gaussian_fit"]
                except Exception:
                    continue
                if component.get("arrival_time") is not None and component.get("sigma_seconds") is not None:
                    cleaned.append(component)
            self._components = cleaned
        else:
            self._components = []

        self.time_spin.setValue(self.time_factor)
        self.freq_spin.setValue(self.freq_factor)

        if self.current_dm is not None:
            self._set_dm_spin_from_total(float(self.current_dm))
        else:
            self.dm_spin.setValue(0.0)

        self.note_edit.setText(str(entry.get("note", "")))

        # Reapply stored processing to the original array.
        self._rebuild_current_array()

        self._plot()
        self._update_labels()
        self._update_button_states()

        self.statusBar().showMessage(
            f"Loaded {path.name} ({self.original_array.shape[0]} × {self.original_array.shape[1]})"
        )

    def _rebuild_current_array(self):
        if self.original_array is None:
            return

        arr = self.original_array.copy()

        if self.time_factor > 1:
            arr = downsample_array2d(arr, downsample_factor_col=self.time_factor)

        if self.freq_factor > 1:
            arr = downsample_array2d(arr, downsample_factor_row=self.freq_factor)

        self.current_array = np.asarray(arr)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(self):
        if self.current_array is None:
            return

        self.canvas.disable_selectors()
        # Clear previous image/lines before redrawing to avoid overplotting
        self.canvas.clear()

        data = np.asarray(self.current_array)

        # Robust display scaling.
        finite = data[np.isfinite(data)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [1, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = finite.min(), finite.max()
        else:
            vmin, vmax = 0.0, 1.0

        # Time and frequency coordinates are represented by sample/channel
        # indices because .npy files contain only arrays.
        nfreq, ntime = data.shape
        t = np.arange(ntime)
        f = np.arange(nfreq)

        self.canvas.ax_dyn.imshow(
            data,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[0, ntime - 1, 0, nfreq - 1],
            vmin=vmin,
            vmax=vmax,
        )
        # Avoid adding titles/labels that introduce padding; the UI shows
        # the burst name and measurement labels separately.
        self.canvas.ax_dyn.set_ylabel("")
        ts = np.nanmean(data, axis=0)
        fs = np.nanmean(data, axis=1)

        self.canvas.ax_time.plot(t, ts)
        # Keep the time/profile axes uncluttered to avoid creating layout gaps.
        self.canvas.ax_time.set_ylabel("")
        self.canvas.ax_time.set_xlabel("")

        self.canvas.ax_freq.plot(fs, f)
        self.canvas.ax_freq.set_ylabel("")
        self.canvas.ax_freq.set_xlabel("")

        # Formatters to show physical units instead of raw sample/channel indices.
        def time_formatter(value_seconds: float) -> str:
            # value_seconds is time in seconds; choose appropriate unit
            if abs(value_seconds) >= 1.0:
                return f"{value_seconds:.3f} s"
            if abs(value_seconds) >= 1e-3:
                return f"{value_seconds*1e3:.3f} ms"
            return f"{value_seconds*1e6:.3f} μs"

        def freq_formatter(value_mhz: float) -> str:
            # Show frequency in MHz with reasonable precision
            return f"{value_mhz:.3f} MHz"

        # Apply formatters: axis values are sample/channel indices, so multiply
        # by the per-sample/channel resolution. The dynamic spectrum shows the
        # time axis (x) so attach the time formatter to `ax_dyn` and the
        # frequency formatters to the dyn/freq y-axes.
        self.canvas.ax_dyn.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: time_formatter(x * self.time_resolution))
        )

        # Account for any frequency downsampling: each displayed channel
        # corresponds to `freq_factor` original channels. Map displayed
        # channel index `y` to a physical frequency in MHz such that the
        # top channel (index nfreq-1) corresponds to 800 MHz.
        freq_factor = max(1, int(self.freq_factor))
        # Force the displayed frequency axis to always span 400-800 MHz
        # regardless of masked or downsampled channels. Map displayed
        # channel index `y` in [0, nfreq-1] linearly to [400, 800] MHz.
        freq_hi = 800.0
        freq_lo = 400.0
        if nfreq > 1:
            df = (freq_hi - freq_lo) / (nfreq - 1)
        else:
            df = 0.0

        self.canvas.ax_dyn.yaxis.set_major_formatter(
            FuncFormatter(lambda y, pos, _lo=freq_lo, _df=df: freq_formatter(_lo + y * _df))
        )

        self.canvas.ax_freq.yaxis.set_major_formatter(
            FuncFormatter(lambda y, pos, _lo=freq_lo, _df=df: freq_formatter(_lo + y * _df))
        )

        # Ensure each subplot fills its available axis range.
        # Dynamic spectrum: full sample extents.
        self.canvas.ax_dyn.set_xlim(0, ntime - 1)
        self.canvas.ax_dyn.set_ylim(0, nfreq - 1)

        # Time profile: show full time range horizontally, autoscale vertically.
        self.canvas.ax_time.relim()
        self.canvas.ax_time.autoscale_view()
        # Ensure the shared x-axis (dynamic spectrum) covers the full time range.
        self.canvas.ax_dyn.set_xlim(0, ntime - 1)

        # Frequency profile: autoscale horizontally, show full frequency range vertically.
        self.canvas.ax_freq.relim()
        self.canvas.ax_freq.autoscale_view()
        # Shared y-axis with the dynamic spectrum controls the frequency range.
        self.canvas.ax_dyn.set_ylim(0, nfreq - 1)

        # Explicitly synchronize the frequency-spectrum y-axis with the
        # dynamic-spectrum y-axis so ticks/limits exactly match.
        dyn_ylim = self.canvas.ax_dyn.get_ylim()
        self.canvas.ax_freq.set_ylim(dyn_ylim)
        # Ensure matched inversion state (if any).
        if self.canvas.ax_dyn.yaxis_inverted() and not self.canvas.ax_freq.yaxis_inverted():
            self.canvas.ax_freq.invert_yaxis()
        if self.canvas.ax_freq.yaxis_inverted() and not self.canvas.ax_dyn.yaxis_inverted():
            self.canvas.ax_dyn.invert_yaxis()

        self._draw_measurement_lines()

        self.canvas.draw_idle()

    def _draw_measurement_lines(self):
        # Remove old guide lines.
        for ax in (self.canvas.ax_dyn, self.canvas.ax_time, self.canvas.ax_freq):
            for line in list(ax.lines):
                # Preserve actual spectrum/profile lines only where needed.
                if getattr(line, "_frb_guide", False):
                    line.remove()

        if self.width_start is not None:
            for ax in (self.canvas.ax_dyn, self.canvas.ax_time):
                line = ax.axvline(self.width_start, linestyle="--", color="red")
                line._frb_guide = True

        if self.width_end is not None:
            for ax in (self.canvas.ax_dyn, self.canvas.ax_time):
                line = ax.axvline(self.width_end, linestyle="--", color="red")
                line._frb_guide = True

        if self.band_start is not None:
            line = self.canvas.ax_dyn.axhline(self.band_start, linestyle="--", color="red")
            line._frb_guide = True

            line = self.canvas.ax_freq.axhline(self.band_start, linestyle="--", color="red")
            line._frb_guide = True

        if self.band_end is not None:
            line = self.canvas.ax_dyn.axhline(self.band_end, linestyle="--", color="red")
            line._frb_guide = True

            line = self.canvas.ax_freq.axhline(self.band_end, linestyle="--", color="red")
            line._frb_guide = True

        # Plot each fitted component on the time-profile if available.
        component_fits = list(self._components) if self._components else []
        if self._last_gaussian_fit is not None and not component_fits:
            component_fits = [self._last_gaussian_fit]

        if component_fits and self.current_array is not None:
            try:
                nfreq, ntime = self.current_array.shape
                time_factor = max(1, int(self.time_factor))
                t_samples = np.arange(ntime)

                profile = np.nanmean(self.current_array, axis=0)
                baseline_profile = float(np.nanpercentile(profile, 10)) if np.size(profile) else 0.0
                profile_min = float(np.nanmin(profile)) if np.size(profile) else 0.0
                profile_max = float(np.nanmax(profile)) if np.size(profile) else 1.0

                colors = ["lime", "cyan", "magenta", "orange", "gold"]
                for idx, fit in enumerate(component_fits):
                    A = fit.get("amplitude")
                    mu = fit.get("arrival_time")
                    sigma = fit.get("sigma_seconds")
                    baseline = fit.get("baseline") if fit.get("baseline") is not None else baseline_profile

                    if mu is None or sigma is None:
                        continue
                    if float(sigma) <= 0:
                        continue

                    mu_sample = float(mu) / (self.time_resolution * time_factor)
                    sigma_sample = float(sigma) / (self.time_resolution * time_factor)

                    if not np.isfinite(mu_sample) or not np.isfinite(sigma_sample):
                        continue

                    if A is not None:
                        amplitude = float(A)
                    else:
                        # Infer amplitude directly from the burst profile so the
                        # Gaussian stays in the same units as the visible time series.
                        half_width = max(1, int(np.ceil(3.0 * sigma_sample)))
                        lo = max(0, int(np.floor(mu_sample - half_width)))
                        hi = min(len(t_samples), int(np.ceil(mu_sample + half_width)))
                        local_profile = profile[lo:hi]
                        if local_profile.size:
                            amplitude = float(np.max(local_profile) - baseline)
                        else:
                            amplitude = max(profile_max - baseline, 1e-6)

                    gauss = float(baseline) + max(amplitude, 0.0) * np.exp(-0.5 * ((t_samples - mu_sample) / sigma_sample) ** 2)

                    color = colors[idx % len(colors)]
                    line, = self.canvas.ax_time.plot(t_samples, gauss, color=color, linewidth=1.5, alpha=0.8)
                    line._frb_guide = True

                    vline = self.canvas.ax_time.axvline(mu_sample, linestyle=":", color=color, alpha=0.9)
                    vline._frb_guide = True
                    vline2 = self.canvas.ax_dyn.axvline(mu_sample, linestyle=":", color=color, alpha=0.9)
                    vline2._frb_guide = True
            except Exception:
                # Non-fatal: plotting the fit should not break the viewer.
                pass

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def next_burst(self):
        self._save_current_state()
        if self.index < len(self.files) - 1:
            self.load_burst(self.index + 1)

    def previous_burst(self):
        self._save_current_state()
        if self.index > 0:
            self.load_burst(self.index - 1)

    def go_to_position_from_text(self):
        if not self.files:
            return

        text = self.position_label.text().strip()
        if not text:
            self._update_labels()
            return

        value_text = text.replace(" ", "")
        if "/" in value_text:
            left, right = value_text.split("/", 1)
            if left and right:
                try:
                    target_index = int(left) - 1
                    total_files = int(right)
                except ValueError:
                    self._update_labels()
                    return
            else:
                self._update_labels()
                return
        else:
            try:
                target_index = int(value_text) - 1
                total_files = len(self.files)
            except ValueError:
                self._update_labels()
                return

        if total_files <= 0:
            self._update_labels()
            return

        if target_index < 0:
            target_index = 0
        elif target_index >= len(self.files):
            target_index = len(self.files) - 1

        if target_index != self.index:
            self._save_current_state()
            self.load_burst(target_index)
        else:
            self._update_labels()

    # ------------------------------------------------------------------
    # Width / bandwidth selection
    # ------------------------------------------------------------------

    def start_width_selection(self):
        if self.current_array is None:
            return

        self.canvas.enable_width_selector(self.width_selected)
        self.processing_label.setText(
            "Drag across the burst in the time-series panel, then click Accept Selection."
        )
        self.statusBar().showMessage("Width selection active.")

    def width_selected(self, start, end):
        self.width_start = start
        self.width_end = end
        self._draw_measurement_lines()
        self.canvas.draw_idle()
        self._update_labels()

    def start_bandwidth_selection(self):
        if self.current_array is None:
            return

        self.canvas.enable_bandwidth_selector(self.bandwidth_selected)
        self.processing_label.setText(
            "Drag across the burst bandwidth in the frequency-spectrum panel, then click Accept Selection."
        )
        self.statusBar().showMessage("Bandwidth selection active.")

    def bandwidth_selected(self, start, end):
        self.band_start = start
        self.band_end = end
        self._draw_measurement_lines()
        self.canvas.draw_idle()
        self._update_labels()

    def accept_selection(self):
        self.canvas.disable_selectors()
        self._save_current_state()
        self.processing_label.setText("Selection accepted.")
        self.statusBar().showMessage("Selection saved.")

    def _time_center(self) -> Optional[float]:
        """Return the time-center (in displayed samples) to use for zooming.

        Prefer the center of the selected width if available, otherwise the
        midpoint of the full time axis.
        """
        if self.current_array is None:
            return None
        ntime = self.current_array.shape[1]
        if self.width_start is not None and self.width_end is not None:
            return (self.width_start + self.width_end) / 2.0
        return (ntime - 1) / 2.0

    def _apply_zoom(self, scale: float):
        if self.current_array is None:
            return
        ntime = self.current_array.shape[1]

        # Current view width in displayed samples.
        cur_xlim = self.canvas.ax_dyn.get_xlim()
        cur_width = max(1.0, abs(cur_xlim[1] - cur_xlim[0]))

        new_width = max(1.0, min(cur_width * scale, ntime - 1))

        # Keep zoom centered on the current view's midpoint so repeated
        # zooms stay focused where the user is looking. Fall back to the
        # width-selection center or full-array midpoint if needed.
        try:
            view_center = (cur_xlim[0] + cur_xlim[1]) / 2.0
        except Exception:
            view_center = None

        center = view_center if view_center is not None else (self._time_center() or (ntime - 1) / 2.0)

        left = center - new_width / 2.0
        right = center + new_width / 2.0

        # Clamp to valid range.
        if left < 0:
            left = 0.0
            right = min(new_width, ntime - 1)
        if right > ntime - 1:
            right = ntime - 1
            left = max(0.0, right - new_width)

        self.canvas.ax_dyn.set_xlim(left, right)
        try:
            self.canvas.ax_time.set_xlim(left, right)
        except Exception:
            pass
        self.canvas.draw_idle()

    def zoom_in(self):
        self._apply_zoom(0.5)

    def zoom_out(self):
        self._apply_zoom(2.0)

    def reset_zoom(self):
        if self.current_array is None:
            return
        ntime = self.current_array.shape[1]
        self.canvas.ax_dyn.set_xlim(0, ntime - 1)
        try:
            self.canvas.ax_time.set_xlim(0, ntime - 1)
        except Exception:
            pass
        self.canvas.draw_idle()

    def reset_selection(self):
        """Reset selections and processing for the current burst.

        Clears width/band selections, DM adjustments, and downsampling factors,
        restores the `current_array` to the original array, updates controls,
        redraws the plots, and saves the cleared state.
        """
        if not self.files:
            return

        # Disable interactive selectors
        self.canvas.disable_selectors()

        # Clear selections
        self.width_start = None
        self.width_end = None
        self.band_start = None
        self.band_end = None

        # Clear DM and downsampling
        self.current_dm = None
        # Restore to new defaults
        self.time_factor = 16
        self.freq_factor = 4

        # Clear any previously added multi-component fits.
        self._components = []
        self._last_gaussian_fit = None

        # Reset UI controls
        self.time_spin.setValue(16)
        self.freq_spin.setValue(4)
        self.dm_spin.setValue(0.0)

        # Restore original data
        if self.original_array is not None:
            self.current_array = self.original_array.copy()
        else:
            self.current_array = None

        # Update UI and save
        self._plot()
        self._update_labels()
        self._save_current_state()

        self.processing_label.setText("Reset: selections and processing cleared.")
        self.statusBar().showMessage("Reset complete.")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def downsample_time(self):
        if self.original_array is None:
            return

        factor = self.time_spin.value()
        try:
            self.time_factor = factor
            self._rebuild_current_array()
            self._plot()
            self._save_current_state()

            source = "downsample.py" if _DOWNSAMPLE_MODULE else "built-in NumPy"
            self.processing_label.setText(f"Time downsampling: {factor} ({source})")
            self.statusBar().showMessage("Time downsampling complete.")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Time downsampling failed",
                str(exc),
            )

    def downsample_frequency(self):
        if self.original_array is None:
            return

        factor = self.freq_spin.value()
        try:
            self.freq_factor = factor
            self._rebuild_current_array()
            self._plot()
            self._save_current_state()

            source = "downsample.py" if _DOWNSAMPLE_MODULE else "built-in NumPy"
            self.processing_label.setText(f"Frequency downsampling: {factor} ({source})")
            self.statusBar().showMessage("Frequency downsampling complete.")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Frequency downsampling failed",
                str(exc),
            )

    def adjust_dm(self):
        """DM correction is intentionally disabled because the dedispersion code is not reliable."""
        if self.original_array is None:
            return

        self.current_dm = self._dm_total_for_results()
        self._rebuild_current_array()
        self._set_dm_spin_from_total(float(self.current_dm))
        self._plot()
        self._save_current_state()

        self.processing_label.setText("DM adjustment: disabled (dedispersion removed)")
        self.statusBar().showMessage("DM correction disabled; raw data is displayed.")

    def fit_gaussian(self):
        """Fit a Gaussian + constant baseline to the time-profile and record results.

        Uses SciPy's `curve_fit` when available; otherwise falls back to
        moment-based estimates without uncertainties.
        """
        if self.current_array is None:
            return

        data = np.asarray(self.current_array)
        nfreq, ntime = data.shape
        ts = np.nanmean(data, axis=0)

        # Use selected width if present, otherwise full range.
        if self.width_start is not None and self.width_end is not None:
            start = int(max(0, np.floor(self.width_start)))
            end = int(min(ntime - 1, np.ceil(self.width_end)))
        else:
            start = 0
            end = ntime - 1

        if end <= start:
            QMessageBox.warning(self, "Fit failed", "Invalid width selection for fitting.")
            return

        inds = np.arange(start, end + 1)
        profile = ts[start : end + 1]

        # Physical time axis in seconds (account for downsampling factor)
        time_factor = max(1, int(self.time_factor))
        t = inds * self.time_resolution * time_factor

        # Baseline estimate and initial guesses
        baseline0 = float(np.nanpercentile(profile, 10))
        amp0 = float(np.nanmax(profile) - baseline0)
        mu0 = float((start + end) / 2.0 * self.time_resolution * time_factor)
        sigma0 = float(max(1.0, (end - start) / 6.0) * self.time_resolution * time_factor)

        result: dict[str, Any] = {}

        if curve_fit is not None:
            def _gauss(x, A, mu, sigma, c):
                return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c

            p0 = [amp0, mu0, sigma0, baseline0]
            try:
                popt, pcov = curve_fit(_gauss, t, profile, p0=p0, maxfev=20000)
                perr = np.sqrt(np.diag(pcov)) if pcov is not None else [None] * len(popt)

                A, mu, sigma, c = popt
                A_err, mu_err, sigma_err, c_err = perr

                result = {
                    "amplitude": float(A),
                    "amplitude_err": float(A_err) if A_err is not None else None,
                    "arrival_time": float(mu),
                    "arrival_time_err": float(mu_err) if mu_err is not None else None,
                    "sigma_seconds": float(abs(sigma)),
                    "sigma_seconds_err": float(abs(sigma_err)) if sigma_err is not None else None,
                    "baseline": float(c),
                    "baseline_err": float(c_err) if c_err is not None else None,
                }
            except Exception as exc:
                # Fall back to moments if curve_fit fails
                curve_fit_failed = True
        else:
            curve_fit_failed = True

        if 'result' not in locals() or (not result and 'curve_fit_failed' in locals() and curve_fit_failed):
            # Moment-based estimates (no uncertainties)
            prof_nonneg = profile - baseline0
            prof_nonneg = np.where(prof_nonneg > 0, prof_nonneg, 0.0)
            denom = prof_nonneg.sum()
            if denom <= 0:
                QMessageBox.warning(self, "Fit failed", "Profile contains no positive signal for fitting.")
                return
            mu = float((t * prof_nonneg).sum() / denom)
            var = float((prof_nonneg * (t - mu) ** 2).sum() / denom)
            sigma = float(np.sqrt(var))

            result = {
                "amplitude": float(profile.max()),
                "amplitude_err": None,
                "arrival_time": mu,
                "arrival_time_err": None,
                "sigma_seconds": sigma,
                "sigma_seconds_err": None,
                "baseline": float(baseline0),
                "baseline_err": None,
            }

        # Save and display
        self._last_gaussian_fit = result
        # Keep last fit available; do not automatically add to components.
        self._save_current_state()
        self._plot()
        self._update_labels()

        # Friendly processing label
        at_ms = result["arrival_time"] * 1000.0
        at_err_ms = (result.get("arrival_time_err") * 1000.0) if result.get("arrival_time_err") else None
        wid_ms = result["sigma_seconds"] * 1000.0
        wid_err_ms = (result.get("sigma_seconds_err") * 1000.0) if result.get("sigma_seconds_err") else None

        parts = [f"TOA: {at_ms:.4g} ms"]
        if at_err_ms is not None:
            parts[0] += f" ± {at_err_ms:.3g} ms"
        parts.append(f"Width(σ): {wid_ms:.4g} ms")
        if wid_err_ms is not None:
            parts[1] += f" ± {wid_err_ms:.3g} ms"

        self.processing_label.setText("; ".join(parts))
        self.statusBar().showMessage("Gaussian fit complete.")

    def add_component(self):
        """Append the last Gaussian fit as an additional component for multi-component exports."""
        if self._last_gaussian_fit is None:
            QMessageBox.warning(self, "No fit available", "No Gaussian fit available to add as a component.")
            return

        fit = dict(self._last_gaussian_fit)
        try:
            at = float(fit.get("arrival_time"))
            sig = float(fit.get("sigma_seconds"))
        except Exception:
            QMessageBox.warning(self, "Invalid fit", "The last fit does not contain valid arrival/width information.")
            return

        component = {
            "arrival_time": at,
            "sigma_seconds": sig,
            "gaussian_fit": fit,
        }
        for key in ("amplitude", "amplitude_err", "arrival_time_err", "sigma_seconds_err", "baseline", "baseline_err"):
            if key in fit:
                component[key] = fit[key]

        self._components.append(component)
        self._save_current_state()
        self._plot()
        self._update_labels()
        self.statusBar().showMessage(f"Added component: TOA={at:.6g} s, σ={sig:.6g} s")

    # ------------------------------------------------------------------
    # Notes and output
    # ------------------------------------------------------------------

    def save_note(self):
        if not self.files:
            return

        self._save_current_state()
        self._write_results()

        self.processing_label.setText("Note saved.")
        self.statusBar().showMessage("Note saved to results file.")

    def _save_current_state(self):
        if not self.files:
            return

        filename = self.files[self.index].name

        width = None
        if self.width_start is not None and self.width_end is not None:
            # width stored in original (pre-downsampled) samples
            width_display = abs(self.width_end - self.width_start)
            width = float(width_display * max(1, int(self.time_factor)))

        bandwidth = None
        if self.band_start is not None and self.band_end is not None:
            # bandwidth stored in original (pre-downsampled) frequency channels
            band_display = abs(self.band_end - self.band_start)
            bandwidth = float(band_display * max(1, int(self.freq_factor)))

        gaussian_fit = self._last_gaussian_fit or {}
        dm_total = self._dm_total_for_results()

        entry = {
            "file": filename,
            "dm": dm_total,
            "time_downsample": self.time_factor,
            "freq_downsample": self.freq_factor,
            "width_start": self.width_start,
            "width_end": self.width_end,
            "selection_width": width,
            "band_start": self.band_start,
            "band_end": self.band_end,
            "bandwidth": bandwidth,
            "note": self.note_edit.text(),
            "gaussian_fit": gaussian_fit,
            "components": self._components,
        }

        self.results[filename] = entry
        self._all_results[filename] = entry

        self._write_results()

    def _metadata_search_roots(self):
        roots = []
        if self.directory is not None:
            roots.append(Path(self.directory).resolve())
            roots.append(Path(self.directory).resolve().parent)
        roots.append(Path(__file__).resolve().parent)
        seen = set()
        ordered = []
        for root in roots:
            normalized = str(root)
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(root)
        return ordered

    def _read_json_metadata_file(self, filename: str):
        for root in self._metadata_search_roots():
            path = root / filename
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
        return {}

    def _current_file_start_mjd(self) -> float:
        """Return the observation start MJD from obs_dict.json when available."""
        if not self.files or self.index >= len(self.files):
            return 0.0

        obs_dict = self._read_json_metadata_file("obs_dict.json")
        if not isinstance(obs_dict, dict):
            return 0.0

        stem = self.files[self.index].stem
        match = re.search(r"(\d{5,})-(\d+)", stem)
        if match is not None:
            obs_key = match.group(1)
            entry = obs_dict.get(obs_key)
            if isinstance(entry, dict):
                obs_start = entry.get("OBS_START")
                if obs_start is not None:
                    return float(obs_start)

        match = re.search(r"(\d{5,})", stem)
        if match is not None:
            obs_key = match.group(1)
            entry = obs_dict.get(obs_key)
            if isinstance(entry, dict):
                obs_start = entry.get("OBS_START")
                if obs_start is not None:
                    return float(obs_start)

        return 0.0

    def _current_fluxcal_toa(self) -> float:
        """Return the exact fluxcal TOA using obs start MJD plus burst-local TOA in seconds."""
        if not self.files or self.index >= len(self.files):
            return 0.0

        stem = self.files[self.index].stem
        burst_match = re.search(r"(\d{5,})-(\d+)", stem)
        if burst_match is None:
            return 0.0

        obs_key = burst_match.group(1)
        burst_key = f"{obs_key}-{burst_match.group(2)}"

        obs_dict = self._read_json_metadata_file("obs_dict.json")
        burst_dict = self._read_json_metadata_file("burst_dict.json")

        obs_entry = obs_dict.get(obs_key)
        if isinstance(obs_entry, dict):
            obs_start = obs_entry.get("OBS_START")
        else:
            obs_start = None

        burst_entry = burst_dict.get(burst_key)
        if isinstance(burst_entry, dict):
            toa_seconds = burst_entry.get("TOA")
        else:
            toa_seconds = None

        if obs_start is None or toa_seconds is None:
            return 0.0

        return float(obs_start) + float(toa_seconds) / 86400.0

    def _results_path(self) -> Optional[Path]:
        if self.directory is None:
            return None
        return self.directory / "../frb_viewer_results.json"

    def _write_results(self):
        path = self._results_path()
        if path is None:
            return

        payload = self._all_results if self._all_results else self.results
        clean = self._clean_for_json(payload)
        try:
            path.write_text(
                json.dumps(clean, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        except (TypeError, ValueError):
            # JSON serialization can still fail for odd objects; fall back to a
            # fully normalized structure with NaN/Infinity converted away.
            path.write_text(
                json.dumps(self._clean_for_json(clean), indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: FRBViewer._clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [FRBViewer._clean_for_json(v) for v in obj]
        if isinstance(obj, tuple):
            return [FRBViewer._clean_for_json(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [FRBViewer._clean_for_json(v) for v in obj.tolist()]
        if isinstance(obj, np.generic):
            value = obj.item()
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    def run_fluxcal(self):
        if self.current_array is None:
            return

        if self.width_start is None or self.width_end is None:
            QMessageBox.warning(
                self,
                "Fluxcal requires a selected width",
                "Select the burst width before running Fluxcal.",
            )
            return

        width_start = float(self.width_start)
        width_end = float(self.width_end)
        width_seconds = abs(width_end - width_start) * self.time_resolution * max(1, int(self.time_factor))
        if width_seconds <= 0:
            QMessageBox.warning(self, "Fluxcal failed", "Selected width is not positive.")
            return

        on_pulse_start = width_start * self.time_resolution * max(1, int(self.time_factor))
        on_pulse_end = width_end * self.time_resolution * max(1, int(self.time_factor))

        toa = self._current_fluxcal_toa()
        if toa == 0.0:
            start_mjd = self._current_file_start_mjd()
            selection_start_seconds = width_start * self.time_resolution * max(1, int(self.time_factor))
            if self._last_gaussian_fit and self._last_gaussian_fit.get("arrival_time") is not None:
                toa = start_mjd + (float(self._last_gaussian_fit["arrival_time"]) / 86400.0)
            else:
                toa = start_mjd + (selection_start_seconds / 86400.0)
            self.statusBar().showMessage(
                "Fluxcal: no matching obs_dict/burst_dict TOA found; using an approximate "
                "in-file TOA instead of the pipeline single-pulse TOA.",
                8000,
            )

        try:
            result = fluxcal(
                self.current_array,
                toa=toa,
                width=width_seconds,
                num_fwhms=3,
                downsample_factor_time=max(1, int(self.time_factor)),
                show_plot=True,
                on_pulse_start=on_pulse_start,
                on_pulse_end=on_pulse_end,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Fluxcal failed", str(exc))
            return

        self.processing_label.setText(f"Fluxcal result: {result:.6g} Jy·ms")
        self.statusBar().showMessage(f"Fluxcal complete: {result:.6g} Jy·ms")

    def export_fitburst_npz(self):
        if self.current_array is None or self.directory is None:
            return

        self._save_current_state()

        default_name = f"{self.files[self.index].stem}_fitburst_input.npz"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export fitburst-compatible .npz",
            str(self.directory / default_name),
            "NumPy archive (*.npz)",
        )
        if not path:
            return

        # Determine component list: prefer explicit components, then last fit, then selection
        arrival_times: list[float] = []
        widths_seconds: list[float] = []
        amplitudes: list[float] = []

        if self._components:
            for c in self._components:
                arrival_times.append(float(c["arrival_time"]))
                widths_seconds.append(float(c["sigma_seconds"]))
                amplitudes.append(float(c.get("amplitude", c.get("gaussian_fit", {}).get("amplitude", 1.0))))
        elif self._last_gaussian_fit is not None:
            arrival_times = [float(self._last_gaussian_fit.get("arrival_time", 0.0))]
            widths_seconds = [float(self._last_gaussian_fit.get("sigma_seconds", 0.0))]
            amplitudes = [float(self._last_gaussian_fit.get("amplitude", 1.0))]
        else:
            # Fallback to selection-based single component
            if self.width_start is not None and self.width_end is not None:
                widths_seconds = [abs(self.width_end - self.width_start) * self.time_resolution * max(1, int(self.time_factor))]
                arrival_times = [float((self.width_start + self.width_end) / 2.0) * self.time_resolution * max(1, int(self.time_factor))]
                amplitudes = [float(np.nanmax(self.current_array))]
            else:
                # no selection: single default
                widths_seconds = [max(self.current_array.shape[1] * self.time_resolution, self.time_resolution)]
                arrival_times = [0.0]
                amplitudes = [float(np.nanmax(self.current_array))]

        data_full, metadata, burst_parameters = build_fitburst_generic_input(
            self.current_array,
            dm=self.current_dm if self.current_dm is not None else 0.0,
            res_time=self.time_resolution * max(1, int(self.time_factor)),
            res_freq=self.freq_resolution * max(1, int(self.freq_factor)),
            freqs_bin0=400.0,
            bad_chans=[],
            is_dedispersed=self.current_dm is not None,
            arrival_time=arrival_times,
            ref_freq=400.0,
            width_seconds=widths_seconds,
            amplitude=amplitudes,
        )

        np.savez(
            path,
            data_full=data_full,
            metadata=metadata,
            burst_parameters=burst_parameters,
        )

        self.processing_label.setText("fitburst .npz exported.")
        self.statusBar().showMessage(f"fitburst input saved to {path}")

    def export_csv(self):
        if self.directory is None:
            return

        self._save_current_state()

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export results as CSV",
            str(self.directory / "frb_viewer_results.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        fields = [
            "file",
            "dm",
            "time_downsample",
            "freq_downsample",
            "width_start",
            "width_end",
            "width",
            "band_start",
            "band_end",
            "bandwidth",
            "note",
        ]

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for entry in self.results.values():
                writer.writerow({key: entry.get(key) for key in fields})

        self.statusBar().showMessage(f"CSV exported to {path}")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _update_labels(self):
        if not self.files:
            return

        self.burst_label.setText(self.files[self.index].name)
        self.position_label.setText(f"{self.index + 1} / {len(self.files)}")

        if self.current_dm is None:
            self.dm_display.setText("Current DM: —")
        else:
            self.dm_display.setText(f"Current DM: {self.current_dm:.6f} pc cm⁻³")

        if self.width_start is not None and self.width_end is not None:
            width_samples = abs(self.width_end - self.width_start)
            # Each displayed sample corresponds to `time_factor` original samples.
            orig_samples = width_samples * max(1, int(self.time_factor))
            width_ms = orig_samples * self.time_resolution * 1000.0
            self.width_label.setText(
                f"Width: {width_ms:.4g} ms ({width_samples:.0f} displayed, {orig_samples:.0f} original samples)"
            )
        else:
            self.width_label.setText("Width: —")

        if self.band_start is not None and self.band_end is not None:
            band_display = abs(self.band_end - self.band_start)
            orig_band = band_display * max(1, int(self.freq_factor))
            self.band_label.setText(
                f"Bandwidth: {orig_band:.4g} frequency channels ({band_display:.0f} displayed)"
            )
        else:
            self.band_label.setText("Bandwidth: —")

        # Fit label: show TOA and width if available
        fit = self._last_gaussian_fit
        if fit is not None:
            at = fit.get("arrival_time")
            at_err = fit.get("arrival_time_err")
            sig = fit.get("sigma_seconds")
            sig_err = fit.get("sigma_seconds_err")
            if at is not None and sig is not None:
                at_ms = at * 1000.0
                sig_ms = sig * 1000.0
                txt = f"TOA: {at_ms:.4g} ms; Width(σ): {sig_ms:.4g} ms"
                if at_err is not None:
                    txt = f"TOA: {at_ms:.4g} ms ± {at_err*1000.0:.3g} ms; Width(σ): {sig_ms:.4g} ms"
                if sig_err is not None:
                    txt = txt.replace("ms", f"ms ± {sig_err*1000.0:.3g} ms", 1) if at_err is None else txt + f" ± {sig_err*1000.0:.3g} ms"
                # Append component count if present
                if self._components:
                    txt = txt + f"; Components: {len(self._components)}"
                self.fit_label.setText(txt)
            else:
                self.fit_label.setText("Fit: —")
        else:
            self.fit_label.setText("Fit: —")

    def _update_button_states(self):
        have_data = bool(self.files)
        self.previous_button.setEnabled(have_data and self.index > 0)
        self.next_button.setEnabled(have_data and self.index < len(self.files) - 1)
        self.export_fitburst_button.setEnabled(have_data)
        self.fluxcal_button.setEnabled(have_data)

    def closeEvent(self, event):
        self._save_current_state()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Browse FRB dynamic spectra.")
    parser.add_argument("directory", nargs="?", default=None, help="Directory containing .npy burst files")
    parser.add_argument(
        "--no-bad",
        action="store_true",
        help="Load frb_viewer_results.json and exclude events whose note is 'bad'.",
    )
    parser.add_argument(
        "--no-dups",
        action="store_true",
        help="Load frb_viewer_results.json and exclude events whose note is 'duplicate'.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = FRBViewer(args.directory, no_bad=args.no_bad, no_dups=args.no_dups)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
