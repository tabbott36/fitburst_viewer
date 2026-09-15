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

## Output

The application writes everything to a dictionary that used the burst's file name as a key and stores all of the inputted parameters as values. See the ![frb_viewer_results.json](frb_viewer_results.json) for an example. 
