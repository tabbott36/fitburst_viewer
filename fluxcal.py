import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from chimepsr_fluxcal.utils.sefd import SEFD

LOCAL_DIR = Path(__file__).resolve().parent
SOURCE_RA = "23:09:04.9"
SOURCE_DEC = "48:42:25.4"

# np.trapezoid was added in NumPy 2.0; fall back to the legacy alias on older NumPy.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _coerce_dynamic_spectrum(data_or_path, *, flip_for_legacy=False):
    """Accept a raw 2D array or a path to a saved .npy/.npz burst extract."""
    if isinstance(data_or_path, np.ndarray):
        data = np.asarray(data_or_path, dtype=float)
    elif isinstance(data_or_path, dict):
        for key in ("data_full", "data", "dynspec", "dynamic_spectrum"):
            if key in data_or_path:
                data = np.asarray(data_or_path[key], dtype=float)
                break
        else:
            raise KeyError("Could not find a 2D dynamic spectrum in the dictionary input.")
    elif isinstance(data_or_path, (str, os.PathLike)):
        loaded = np.load(data_or_path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            for key in ("data_full", "data", "dynspec", "dynamic_spectrum"):
                if key in loaded:
                    data = np.asarray(loaded[key], dtype=float)
                    break
            else:
                raise KeyError(f"Could not find a 2D dynamic spectrum in {data_or_path!s}.")
        else:
            data = np.asarray(loaded, dtype=float)
    else:
        raise TypeError(
            "Expected a 2D NumPy array, a dict containing the spectrum, or a path to .npy/.npz data."
        )

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D dynamic spectrum, got shape {data.shape!r}")

    if flip_for_legacy:
        data = np.flipud(data)
    return data

def calc_sefd(toa, duration, ra, dec):
        """
        Calculate the System Equivalent Flux Density (SEFD) for a given time of arrival, duration, and position.

        Tsys does not depend on time (only sky position), but the beam gain varies strongly
        with hour angle from transit, so this must be evaluated at (or very near) the burst's
        actual arrival time rather than e.g. the start of a long recording window.

        Parameters:
        toa (float): The burst arrival time in Modified Julian Date (MJD).
        duration (float): The burst/integration duration in seconds, used to nudge the
            evaluation time to the middle of that short window.
        ra (str, optional): The right ascension of the pointing in HH:MM:SS.s format.
        dec (str, optional): The declination of the pointing in DD:MM:SS.s format.

        Returns:
        float: The SEFD value at the frequency channel (512) TODO: see below.
        """

        sefd = SEFD()
        sefd.set_pointing(ra, dec, mjd=float(toa) + (float(duration) / 2.0) / 86400.0)
        sefd.compute_sefd()

        return sefd.sefd[512] # 512 corresponds to beam center
  
    
def radiometer(snr, sefd, tobs, bw=2.62e8): # NOT accounting for 353/1024 channels masked
    """
    Calculate the radiometer equation.
    
    Parameters:
    snr (float): Signal-to-noise ratio.
    sefd (float): System equivalent flux density.
    tobs (float): Observation time in seconds.
    bw (float, optional): Bandwidth in Hz. Default is 2.62e8 Hz.
    
    Returns:
    float: The calculated radiometer value.
    """
    
    return snr * sefd / np.sqrt(2 * tobs * bw) 


def fluxcal(input_file, toa, width, num_fwhms, downsample_factor_time,
            source_ra=SOURCE_RA, source_dec=SOURCE_DEC, output_path=None, show_plot=True,
            on_pulse_start=None, on_pulse_end=None):
    """Compute a fluence estimate for a dynamic spectrum.

    Parameters
    ----------
    input_file : str | Path | np.ndarray | dict
        Either a path to a .npy/.npz file or the dynamic spectrum array itself.
        When passing the array directly, the GUI can supply the currently displayed
        frequency x time matrix without needing to save a temporary file.
    toa : float
        Burst arrival time in MJD (obs_start + singlepulse TOA / 86400). Used both for
        the printed topocentric TOA and as the pointing epoch for the SEFD calculation,
        since gain varies strongly with hour angle from transit while Tsys does not
        depend on time at all.
    width : float
        Burst width (seconds) of the selected on-pulse window. Used to size the
        off-pulse baseline exclusion region, and as the on-pulse window size (centered
        on the array) when on_pulse_start/on_pulse_end are not given.
    num_fwhms : float
        Number of on-pulse widths of margin used when excluding the burst from the
        off-pulse baseline region.
    downsample_factor_time : float
        Time downsampling factor used to map samples to seconds.
    output_path : str | Path | None
        Optional output path for the diagnostic flux plot. If omitted, the plot is
        not saved.
    show_plot : bool
        Whether to display the matplotlib figure.
    on_pulse_start, on_pulse_end : float | None
        Start/end of the on-pulse window (seconds, same time axis as the dynamic
        spectrum) to integrate the fluence over -- i.e. the GUI's "Width Selection".
        If either is omitted, falls back to a `width`-second window centered on the
        array.
    """

    full_spectra = _coerce_dynamic_spectrum(
        input_file,
        flip_for_legacy=isinstance(input_file, (str, os.PathLike, dict)) or not isinstance(input_file, np.ndarray),
    )

    dt = float(downsample_factor_time) * 40.96e-6
    time = np.arange(0, np.shape(full_spectra)[1], 1) * dt
    burst_center = max(time) / 2.0

    if on_pulse_start is not None and on_pulse_end is not None:
        window_start, window_end = sorted((float(on_pulse_start), float(on_pulse_end)))
    else:
        window_start, window_end = burst_center - width / 2.0, burst_center + width / 2.0

    burst_sefd = calc_sefd(toa, duration=0.2, ra=source_ra, dec=source_dec)

    pulse_time_series = full_spectra.mean(axis=0)
    off_pulse_idx = np.where((time <= window_start - num_fwhms * width))[0]
    if off_pulse_idx.size == 0:
        off_pulse_idx = np.where(time <= window_start)[0]
    if off_pulse_idx.size == 0:
        off_pulse_idx = np.arange(time.size)

    off_pulse_spectra = full_spectra[:, off_pulse_idx]
    off_pulse_time_series = off_pulse_spectra.mean(axis=0)
    off_pulse_mean = np.mean(off_pulse_time_series)
    off_pulse_std = np.std(off_pulse_time_series)
    if not np.isfinite(off_pulse_std) or off_pulse_std <= 0:
        off_pulse_std = np.std(pulse_time_series)
    if not np.isfinite(off_pulse_std) or off_pulse_std <= 0:
        off_pulse_std = 1e-12
    snr = (pulse_time_series - off_pulse_mean) / off_pulse_std

    flux = radiometer(snr, sefd=burst_sefd, tobs=dt)

    on_pulse_idx = (time >= window_start) & (time <= window_end)
    fluence = _trapezoid(flux[on_pulse_idx], dx=dt)

    if output_path is not None or show_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, flux, label="Flux")
        ax.axvspan(window_start, window_end, color='blue', alpha=0.2, label="Width Selection")
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title(f"Estimated Fluence = {1000 * np.round(fluence, 4)} Jy⋅ms")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Flux (Jy)")
        ax.grid()
        ax.legend()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)
        if show_plot:
            plt.show()
        plt.close(fig)

    print(
        f"Fitburst Result: Topocentric TOA (referenced to 800MHz): {toa} "
        f"Width Selection: {1000 * (window_end - window_start)}ms, "
        f"Fluence: {1000 * fluence}Jy⋅ms, SEFD: {burst_sefd}Jy"
    )

    return 1000 * fluence


def peak_flux(input_data, toa, width, downsample_factor_time,
              source_ra=SOURCE_RA, source_dec=SOURCE_DEC):
    full_spectra = _coerce_dynamic_spectrum(
        input_data,
        flip_for_legacy=isinstance(input_data, (str, os.PathLike, dict)) or not isinstance(input_data, np.ndarray),
    )

    dt = float(downsample_factor_time) * 40.96e-6
    time = np.arange(0, np.shape(full_spectra)[1], 1) * dt
    burst_center = max(time) / 2.0

    fwhm = 2 * width * (2 * np.sqrt(2 * np.log(2)))
    burst_sefd = calc_sefd(toa, duration=0.2, ra=source_ra, dec=source_dec)

    pulse_time_series = full_spectra.mean(axis=0)
    off_pulse_idx = np.where((time <= burst_center - 5 * fwhm))[0]
    if off_pulse_idx.size == 0:
        off_pulse_idx = np.where(time <= burst_center)[0]
    if off_pulse_idx.size == 0:
        off_pulse_idx = np.arange(time.size)

    off_pulse_spectra = full_spectra[:, off_pulse_idx]
    off_pulse_time_series = off_pulse_spectra.mean(axis=0)
    off_pulse_mean = np.mean(off_pulse_time_series)
    off_pulse_std = np.std(off_pulse_time_series)
    if not np.isfinite(off_pulse_std) or off_pulse_std <= 0:
        off_pulse_std = np.std(pulse_time_series)
    if not np.isfinite(off_pulse_std) or off_pulse_std <= 0:
        off_pulse_std = 1e-12
    snr = (pulse_time_series - off_pulse_mean) / off_pulse_std

    flux = radiometer(snr, sefd=burst_sefd, tobs=dt)
    return float(max(flux))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate burst fluence from a dynamic spectrum.")
    parser.add_argument("input_file", help="Path to a .npy or .npz file, or a raw spectrum file.")
    parser.add_argument("--toa", type=float, default=0.0, help="TOA value passed to the SEFD calculator.")
    parser.add_argument("--width", type=float, default=0.005, help="Burst sigma width in seconds.")
    parser.add_argument("--num-fwhms", type=float, default=3.0, help="Number of FWHMs used for the baseline window.")
    parser.add_argument("--downsample-factor-time", type=float, default=1.0, help="Time downsampling factor.")
    parser.add_argument("--output", type=str, default=None, help="Optional output PNG path for the diagnostic plot.")
    args = parser.parse_args()

    fluxcal(
        args.input_file,
        mjd=args.mjd,
        toa=args.toa,
        width=args.width,
        num_fwhms=args.num_fwhms,
        downsample_factor_time=args.downsample_factor_time,
        output_path=args.output,
    )