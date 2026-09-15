# FRB Viewer

A small Python/PySide6 GUI for browsing `.npy` fast-radio-burst dynamic spectra.

![FRB Viewer user interface](user_interface.png)

## Install

```bash
python -m pip install numpy matplotlib PySide6
```

## Run

```bash
python frb_viewer.py /path/to/directory
```

Or simply:

```bash
python frb_viewer.py
```

and choose a directory from the GUI.

The directory should contain `.npy` files with shape:

```text
(n_frequency, n_time)
```

Each burst is assumed to be centered in its file.

## Features

- Previous / Next burst navigation
- Dynamic spectrum
- Time series
- Frequency spectrum
- Interactive width selection
- Interactive bandwidth selection
- Time downsampling
- Frequency downsampling
- DM adjustment
- Per-burst notes
- Automatic `frb_viewer_results.json`
- CSV export can be added/used from the application code

### Width selection

Click **Width Selection**, then drag horizontally across the burst in the
time-series panel. The selection is also shown as vertical lines on the
dynamic spectrum.

### Bandwidth selection

Click **Bandwidth Selection**, then drag vertically across the burst in the
frequency-spectrum panel. The selection is also shown as horizontal lines on
the dynamic spectrum.

Click **Accept Selection** when finished.

## Integrating your existing scripts

Put `downsample.py` and `dedisperse.py` beside `frb_viewer.py`.

The application looks for common function names such as:

```python
downsample_time(array, factor)
downsample_freq(array, factor)
dedisperse(array, dm)
```

The adapter functions are near the top of `frb_viewer.py`:

- `run_time_downsample`
- `run_freq_downsample`
- `run_dedisperse`

If your current functions have different signatures, edit those three adapters.

The GUI expects `dedisperse` to return either:

```python
array
```

or preferably:

```python
array, optimized_dm
```

It also accepts a dictionary containing the array and DM.

If no compatible external function is found, time/frequency downsampling uses
simple NumPy block averaging. DM adjustment falls back to leaving the array
unchanged and recording the supplied DM. The status bar indicates whether the
external script or fallback was used.

## Output

The application automatically writes:

```text
frb_viewer_results.json
```

into the selected burst directory.

Example:

```json
{
  "burst_001.npy": {
    "file": "burst_001.npy",
    "dm": 218.184,
    "initial_dm": 219.5,
    "time_downsample": 8,
    "freq_downsample": 4,
    "width_start": 12340.0,
    "width_end": 12355.0,
    "width": 15.0,
    "band_start": 420.0,
    "band_end": 850.0,
    "bandwidth": 430.0,
    "note": "Clean burst"
  }
}
```

The width and bandwidth are currently recorded in **time samples** and
**frequency-channel indices**, because a bare `.npy` array contains no physical
time/frequency metadata. If your project has known `tsamp`, channel width, and
frequency ordering, those can be added easily.
