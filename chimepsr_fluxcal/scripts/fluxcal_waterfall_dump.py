#!/usr/bin/env python

import os
import argparse
from collections import namedtuple
import numpy as np
from astropy.coordinates import SkyCoord, FK5
from astropy.time import Time
from numpy.ma.extras import count_masked
from scipy import interpolate
from scipy.integrate import simps
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from chimepsr_fluxcal.utils.sefd import SEFD
from chimepsr_fluxcal.utils.baseline import bmask


class DataReader:
    """
    Data reader class to interpret the .npz files output by the waterfaller code.
    """

    def __init__(self, fname, data_location="."):

        # ensure file exists, else raise an AssertionError
        self.file_path = f"{data_location}/{fname}"
        if not os.path.isfile(self.file_path):
            raise IOError(f"Data file not found: {self.file_path}")
        self.data_full = None
        self.data_weights = None
        self.num_freq = None
        self.num_time = None
        self.res_freq = None
        self.res_time = None
        self.times = None
        self.freqs = None
        self.is_dedispersed = None
        self.metadata = None

    def load_data(self):
        """
        Load data from a generic .npz file containing three sub-files:
            spectrum: the raw data in a 2D numpy.ndarray
            metadata: a dictionary containing information required to
                reconstruct data dimensions and physical values, masked
                channels, start MJD, etc.
            burst_parameters: a dictionary containing rough estimates of some
                critical burst parameters that will help the fitters converge
        """
        unpacked_data_set = np.load(self.file_path, allow_pickle=True)

        # ensure required subfiles are present
        expected_subfile_names = ["spectrum", "metadata", "burst_parameters"]
        retrieved_subfile_names = unpacked_data_set.files
        if not all([f in retrieved_subfile_names for f in expected_subfile_names]):
            raise AssertionError(
                f"Data file does not contain one of more of the following keys: "
                f"{expected_subfile_names}"
            )

        self.metadata = unpacked_data_set["metadata"].item()
        print(self.metadata)
        # unpack and derive necessary information
        burst_parameters = unpacked_data_set["burst_parameters"].item()
        print(burst_parameters)
        self.data_full = unpacked_data_set["spectrum"]

        # derive time information from loaded data.
        self.num_freq, self.num_time = self.data_full.shape
        if self.num_freq != self.metadata["num_freq"]:
            raise AssertionError(
                "Data shape does not match recorded number of channels"
                f"({self.num_freq} != {self.metadata['nchan']})"
            )
        if self.num_time != self.metadata["num_time"]:
            raise AssertionError(
                "Data shape does not match recorded number of time samples"
                f"({self.num_time} != {self.metadata['ntime']})"
            )

        # create the weights array, where True = masked
        self.data_weights = np.ones((self.num_freq, self.num_time), dtype=float)
        rfi_mask = self.metadata["bad_chans"]
        self.data_weights[rfi_mask, :] = 0.0

        # create time sample labels from data shape and metadata
        # leave the samples in relative seconds since the beginning of the
        # spectra
        self.res_time = self.metadata["res_time"]
        times = np.arange(self.num_time, dtype=np.float64) * self.res_time
        self.times = times

        # create frequency channel centre labels from data shape and metadata
        self.res_freq = self.metadata["res_freq"]
        freqs = np.arange(self.num_freq, dtype=np.float64) * self.res_freq
        freqs += self.metadata["freqs_bin0"]

        # currently have the leading-edge frequency for each channel, add chan_bw / 2
        freqs += self.res_freq / 2.0
        self.freqs = freqs

        # store boolean that indicates of input data is already dedispersed or not.
        self.is_dedispersed = self.metadata["is_dedispersed"]


def get_sefd(ra, dec, time):
    sefd = SEFD()
    sefd.set_pointing(ra, dec, mjd=time)
    sefd.compute_sefd()

    return sefd


def downsample(arr, new_shape):
    print(f"downsampling array (current shape={arr.shape}), new shape={new_shape}")
    sh = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )
    return arr.reshape(sh).mean(-1).mean(1)


def isPowerOfTwo(n):
    if (n & (n - 1) == 0) and n != 0:
        return True
    else:
        return False


def getClosestPowerOfTwo(n):
    return 1 << (n - 1).bit_length()


def plot_waterfall_profile(spectrum, dt=1, df=1, oname="test.pdf"):

    profile = spectrum.mean(0)

    fig = plt.figure(figsize=plt.figaspect(1.5))
    gs = fig.add_gridspec(4, 1)
    axP = fig.add_subplot(gs[0, :])
    axW = fig.add_subplot(gs[1:, :])

    axins = inset_axes(
        axW,
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.01, 0, 1, 1),
        bbox_transform=axW.transAxes,
        borderpad=0,
    )

    pr = axP.plot(profile, color="k")
    wf = axW.imshow(
        spectrum,
        aspect="auto",
        interpolation="none",
        cmap=plt.get_cmap("gist_gray_r"),
        vmin=np.percentile(spectrum, 5),
        vmax=np.percentile(spectrum, 99),
    )

    axP.set_xlim(axW.get_xlim())

    axW.set_xticks(np.linspace(0, spectrum.shape[1], 5))
    axW.set_xticklabels(np.linspace(0, 1000 * dt * spectrum.shape[1], 5, dtype=int))
    axW.set_yticks(np.linspace(0, spectrum.shape[0], 5))
    axW.set_yticklabels(
        np.linspace(800, 800 - np.abs(df / 1.0e6) * spectrum.shape[0], 5, dtype=int)
    )

    axP.set_xticks(axW.get_xticks())

    axW.set_xlabel("Time (ms)")
    axW.set_ylabel("Frequency (MHz)")
    axP.set_ylabel("Flux density\n(mJy)")

    axP.tick_params(axis="x", which="both", labelbottom=False)

    cbar = fig.colorbar(wf, cax=axins, label="Flux density (mJy)")
    plt.savefig(oname, format="pdf", bbox_inches="tight")
    # dump calibrated waterfall to .npz file
    np.savez(oname.rstrip(".pdf"), data=spectrum, dt=dt, df=df)


if __name__ == "__main__":
    usage = "%(prog)s file position"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument(
        "file", type=str, help=".npz file containing waterfall to calibrate"
    )
    parser.add_argument(
        "position",
        type=str,
        help="RA Dec string (J2000), right ascension in hh:mm:ss, declination in dd:mm:ss",
    )
    parser.add_argument(
        "--mjd",
        dest="base_file_start",
        type=float,
        help="MJD start time of the base filterbank file from which this snapshot was extracted",
    )
    parser.add_argument(
        "--dur",
        dest="duration",
        type=float,
        help="duration in seconds of the base filterbank file from which this snapshot was extracted",
    )
    parser.add_argument(
        "--time_ds", dest="tds", type=int, help="time downsample factor", default=1
    )
    parser.add_argument(
        "--freq_ds", dest="fds", type=int, help="frequency downsample factor", default=1
    )
    parser.add_argument(
        "--at_600",
        action="store_true",
        help="Use SEFD at centre of band for all channels",
    )
    parser.add_argument(
        "--just_print",
        action="store_true",
        help="just print out the computed SEFD and other metadata information",
    )

    args = parser.parse_args()
    oname = args.file.replace(".npz", "_fluxcal.pdf")
    ra, dec = args.position.split(" ")

    # get the SEFD for the middle of the corresponding observation based on
    # provided source position, observation start time, and duration
    time = Time(args.base_file_start + (args.duration / 2) / 86400.0, format="mjd")
    coord_at_time = SkyCoord(
        ra, dec, unit=("hourangle", "deg"), frame=FK5(equinox="J2000"), obstime=time
    )
    d = DataReader(args.file)
    d.load_data()
    s = get_sefd(coord_at_time.ra.deg, coord_at_time.dec.deg, time.mjd)

    # mask obviously bad data
    channel_mask = np.zeros_like(d.data_full, dtype=bool)
    channel_mask[d.metadata["bad_chans"]] = True
    channel_mask[np.where(np.std(d.data_full) == 0)] = True
    masked_data = np.ma.masked_array(d.data_full, mask=channel_mask)

    # extend array in time to nearest factor of two
    nbins_orig = masked_data.shape[1]
    print(f"raw input number of bins = {nbins_orig}")
    if not isPowerOfTwo(nbins_orig):
        n_to_add = getClosestPowerOfTwo(nbins_orig) - nbins_orig
        print(f"needed to pad profile with {n_to_add} data points")
        print("taking up-to 128 samples from start and using those to pad")
        if n_to_add > 128:
            # fill up the arrays by tiling the 128-bin selection
            noise_to_add_mask = np.copy(masked_data.mask[:, :128])
            noise_to_add_data = np.copy(masked_data.data[:, :128])

            # fill in the mask
            _c = 0
            while _c < n_to_add - 128:
                noise_to_add_mask = np.concatenate(
                    (
                        noise_to_add_mask,
                        noise_to_add_mask[:, _c][:, None],
                    ),
                    axis=1,
                )
                noise_to_add_data = np.concatenate(
                    (
                        noise_to_add_data,
                        noise_to_add_data[:, _c][:, None],
                    ),
                    axis=1,
                )
                _c += 1
        else:
            noise_to_add_mask = np.copy(masked_data.mask[:, :n_to_add])
            noise_to_add_data = np.copy(masked_data.data[:, :n_to_add])

        # shuffle the noise data randomly
        for c in range(masked_data.shape[0]):
            np.random.shuffle(noise_to_add_data[c, :])
        noise_to_add = np.ma.masked_array(noise_to_add_data, mask=noise_to_add_mask)

        masked_data = np.ma.append(masked_data, noise_to_add, axis=1)
        print(f"masked data shape = {masked_data.shape}")

    # downsample the data if required
    if args.tds != 1 or args.fds != 1:
        requested_shape = (
            masked_data.shape[0] // args.fds,
            masked_data.shape[1] // args.tds,
        )
        masked_data = downsample(masked_data, requested_shape)

    # get baseline characteristics from mean profile
    offpulse_idx = bmask(masked_data.mean(0).data, dc=0.01)
    offpulse_mean = masked_data[:, offpulse_idx].mean(1).flatten()
    offpulse_rms = masked_data[:, offpulse_idx].std(1).flatten()

    data_sub_mean = masked_data - offpulse_mean[:, np.newaxis]
    data_snr = data_sub_mean / offpulse_rms[:, np.newaxis]

    plt.figure()
    plt.plot(masked_data.mean(0), label="raw")
    _md = masked_data.mean(0)
    _md_mask = np.ones_like(_md, dtype=bool)
    _md_mask[offpulse_idx] = False
    plt.plot(np.ma.masked_array(_md, mask=_md_mask), label="offpulse")
    plt.legend()
    plt.xlabel("sample number")
    plt.ylabel("arb. units")
    plt.savefig(oname.replace(".pdf", "_offpulse.pdf"))
    plt.close()

    plt.figure()
    for c in range(masked_data.shape[0]):
        plt.plot(masked_data[c, :])
    plt.savefig(oname.replace(".pdf", "_perchan_offpulse.pdf"))
    plt.close()

    # convert to flux density spectrum
    npol = 2
    dt = d.res_time * args.tds
    # NOTE: these two frequency quantities are in MHz!
    df = d.res_freq * args.fds
    eff_bw = (masked_data.shape[0] - np.ma.count_masked(masked_data[:, 0])) * np.abs(df)
    # conver frequuncy quantities to Hz
    df_in_Hz = df * 1.0e6
    eff_bw_in_Hz = eff_bw * 1.0e6

    print(f"npol = {npol}")
    print(f"dt = {dt} seconds")
    print(f"df = {df} MHz")
    print(f"eff. bw = {eff_bw} MHz")

    # conversion factor from snr -> mJy
    if d.res_freq < 0:
        # reverse it because data is 800 -> 400 but gain computed for 400 -> 800
        sefd = s.sefd[::-1]
    else:
        sefd = s.sefd

    resampled_sefd = downsample(sefd[:, np.newaxis], (1024 // args.fds, 1)).flatten()
    if args.at_600:
        print(f"sefd at 600 MHz = {resampled_sefd[len(resampled_sefd) // 2]} Jy")

    if not args.just_print:
        rms_to_flux = resampled_sefd / np.sqrt(npol * dt * abs(df_in_Hz))
        rms_to_flux_mJy = 1000 * rms_to_flux

        plt.figure()
        plt.plot(rms_to_flux)
        plt.xlabel("freq. channel")
        plt.ylabel("conversion (Jy/unit)")
        plt.savefig(oname.replace(".pdf", "_convfact.pdf"))
        plt.close()

        if args.at_600:
            data_cal = data_snr * rms_to_flux_mJy[len(rms_to_flux_mJy) // 2]
            print(f"rms in mJy = {rms_to_flux_mJy[len(rms_to_flux_mJy) // 2]}")
        else:
            data_cal = data_snr * rms_to_flux_mJy[:, np.newaxis]

        prof_cal = data_cal.mean(0)

        prof_cal_max = np.max(prof_cal)
        prof_cal_rms = np.std(prof_cal[offpulse_idx])

        plot_waterfall_profile(
            data_cal,
            dt=dt,
            df=df_in_Hz,
            oname=oname,
        )

        # write peak flux density to file
        with open("peak_flux.txt", "a") as f:
            f.write("{0} {1} {2}\n".format(args.file, prof_cal_max, prof_cal_rms))
