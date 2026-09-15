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

import csv
import importlib.util
import json
import sys
import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np

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
    """
    Incoherently dedisperse the channelized filterbank data at the specified DM.

    Parameters
    ----------
    dyn_spec: Input dynamic spectrum.
    dm: Dispersion measure to use for dedispersion (pc cm^-3).
    tsamp: Time resolution (s).
    df: Channel width of each filterbank channel (MHz).
    k: Dispersion measure constant (k = 1.0 / 0.000241 is the standard value used in the PSR community).

    Returns
    -------
    dyn_spec_dedispersed: Dedispersed output dynamic spectrum.
    dm: The dispersion measure used for dedispersion.
    """

    freq_hi = 800.0 # MHz
    freq_lo = freq_hi

    dyn_spec_dedispersed = copy.deepcopy(dyn_spec)
    
    nchans, nsamples = dyn_spec.shape

    for chan_idx in np.arange(0, nchans, 1):
        tau = np.multiply(k, np.multiply(dm, np.subtract(np.divide(1.0, np.power(freq_lo, 2.0)),
                                                         np.divide(1.0, np.power(freq_hi, 2.0)))))

        num_samples = int(np.round(np.divide(-tau, tsamp)))
        dyn_spec_dedispersed[chan_idx, :] = np.roll(dyn_spec_dedispersed[chan_idx, :], num_samples, axis=0)

        if (chan_idx < (nchans - 1)):
            freq_lo = np.subtract(freq_lo, np.abs(df))

    return dyn_spec_dedispersed, dm


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
    def __init__(self, directory: Optional[str] = None):
        super().__init__()

        self.setWindowTitle("FRB Viewer")
        self.resize(1400, 1000)

        self.directory: Optional[Path] = None
        self.files: list[Path] = []
        self.index = 0

        self.original_array: Optional[np.ndarray] = None
        self.current_array: Optional[np.ndarray] = None

        # Per-burst metadata.
        self.results: dict[str, dict[str, Any]] = {}

        # Current processing state.
        self.current_dm: Optional[float] = 219.456
        self.time_factor = 1
        self.freq_factor = 1

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

        self.position_label = QLabel("0 / 0")
        self.position_label.setAlignment(Qt.AlignCenter)

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
        self.time_spin.setValue(1)
        processing.addWidget(self.time_spin)

        self.time_button = QPushButton("Downsample Time")
        self.time_button.clicked.connect(self.downsample_time)
        processing.addWidget(self.time_button)

        processing.addWidget(QLabel("Freq. downsample:"))
        self.freq_spin = QSpinBox()
        self.freq_spin.setRange(1, 100000)
        self.freq_spin.setValue(1)
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

        root.addLayout(notes)

        # Measurement/status row.
        status = QHBoxLayout()
        self.width_label = QLabel("Width: —")
        self.band_label = QLabel("Bandwidth: —")
        self.processing_label = QLabel("")

        status.addWidget(self.width_label)
        status.addWidget(self.band_label)
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
        self.index = 0

        self._load_results()
        self.load_burst(0)

        self.dir_label.setText(str(directory))

    def _load_results(self):
        """Load previously saved per-burst metadata from results JSON."""
        self.results = {}
        path = self._results_path()
        if path is None or not path.exists():
            return

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.results = loaded
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
        self.time_factor = int(entry.get("time_downsample", 1) or 1)
        self.freq_factor = int(entry.get("freq_downsample", 1) or 1)

        self.width_start = entry.get("width_start")
        self.width_end = entry.get("width_end")
        self.band_start = entry.get("band_start")
        self.band_end = entry.get("band_end")

        self.time_spin.setValue(self.time_factor)
        self.freq_spin.setValue(self.freq_factor)

        if self.current_dm is not None:
            self.dm_spin.setValue(float(self.current_dm))
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

        self.current_array = arr

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
        self.time_factor = 1
        self.freq_factor = 1

        # Reset UI controls
        self.time_spin.setValue(1)
        self.freq_spin.setValue(1)
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
            arr = downsample_array2d(self.original_array, downsample_factor_col=factor)

            if self.freq_factor > 1:
                arr = downsample_array2d(arr, downsample_factor_row=self.freq_factor)

            self.current_array = arr
            self.time_factor = factor
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
            arr = self.original_array.copy()

            if self.time_factor > 1:
                arr = downsample_array2d(arr, downsample_factor_col=self.time_factor)

            arr = downsample_array2d(arr, downsample_factor_row=factor)

            self.current_array = arr
            self.freq_factor = factor
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
        if self.original_array is None:
            return

        dm = self.dm_spin.value()

        try:
            # Start from original data, then apply DM processing and the
            # selected downsampling factors.
            arr, optimized_dm = dedisperse_incoherent(self.original_array, dm, tsamp=self.time_resolution, df=self.freq_resolution)

            self.current_dm = optimized_dm if optimized_dm is not None else dm

            if self.time_factor > 1:
                arr = downsample_array2d(arr, downsample_factor_col=self.time_factor)
            if self.freq_factor > 1:
                arr = downsample_array2d(arr, downsample_factor_row=self.freq_factor)

            self.current_array = np.asarray(arr)

            self.dm_spin.setValue(float(self.current_dm))
            self._plot()
            self._save_current_state()

            source = "dedisperse_incoherent (built-in)"
            self.processing_label.setText(f"DM adjustment: {self.current_dm:.6f} ({source})")
            self.statusBar().showMessage("DM adjustment complete.")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "DM adjustment failed",
                str(exc),
            )

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

        self.results[filename] = {
            "file": filename,
            "dm": self.current_dm,
            "time_downsample": self.time_factor,
            "freq_downsample": self.freq_factor,
            "width_start": self.width_start,
            "width_end": self.width_end,
            "width": width,
            "band_start": self.band_start,
            "band_end": self.band_end,
            "bandwidth": bandwidth,
            "note": self.note_edit.text(),
        }

        self._write_results()

    def _results_path(self) -> Optional[Path]:
        if self.directory is None:
            return None
        return self.directory / "../frb_viewer_results.json"

    def _write_results(self):
        path = self._results_path()
        if path is None:
            return

        try:
            path.write_text(
                json.dumps(self.results, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        except ValueError:
            # JSON does not permit NaN/Infinity. Convert non-finite floats.
            clean = self._clean_for_json(self.results)
            path.write_text(
                json.dumps(clean, indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: FRBViewer._clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [FRBViewer._clean_for_json(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

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

    def _update_button_states(self):
        have_data = bool(self.files)
        self.previous_button.setEnabled(have_data and self.index > 0)
        self.next_button.setEnabled(have_data and self.index < len(self.files) - 1)

    def closeEvent(self, event):
        self._save_current_state()
        event.accept()


def main():
    app = QApplication(sys.argv)

    directory = sys.argv[1] if len(sys.argv) > 1 else None
    window = FRBViewer(directory)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
