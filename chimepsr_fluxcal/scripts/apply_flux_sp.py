#!/usr/bin/env python

import argparse
import numpy as np
from scipy.integrate import simps
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import psrchive


def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A script to convert a single pulse profile (in an archive) from arbitrary intensity to flux density units."
    )
    parser.add_argument(
        "file", type=str, help="Archive file from which the profile will be generated"
    )
    parser.add_argument(
        "-s",
        "--sefd",
        type=float,
        required=True,
        help="The nominal SEFD (in Jy) to apply",
    )
    parser.add_argument(
        "-o",
        "--onpulse",
        nargs=2,
        metavar=("START", "END"),
        type=float,
        help="The on-pulse phase region",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactively decide on-pulse region",
    )
    parser.add_argument(
        "-d",
        "--dutycycle",
        type=float,
        help="Automatically determine baseline by providing an approximate on-pulse duty cycle (as a fraction)",
    )
    parser.add_argument(
        "-b",
        "--binscrunch",
        type=int,
        help="Number of profile bins to scrunch to [Default: 256]",
        default=256,
    )

    args = parser.parse_args()

    # load and prepare archive
    arch = psrchive.Archive_load(args.file)
    arch.dedisperse()
    arch.pscrunch()
    arch.bscrunch_to_nbin(args.binscrunch)
    nbins = args.binscrunch

    # get effective integration time (i.e. time per profile bin)
    nsub = arch.get_nsubint()
    print("nsubint = %d" % nsub)
    total_duration = 0
    for isub in range(nsub):
        sub = arch.get_Integration(isub)
        total_duration += sub.get_duration()
    print("total integration time = %f s" % total_duration)
    dt = total_duration / nbins
    print("integration time (per bin), dt = %f s" % dt)
    arch.tscrunch()

    # get effective bandwidth, accounting for masked channels
    eff_bw = np.abs(arch.get_Integration(0).effective_bandwidth()) * 1.0e6
    print("effective bw = %f MHz" % (eff_bw / 1.0e6))
    arch.fscrunch()

    profile = arch.get_Integration(0)

    npol = 2
    rms_to_flux = args.sefd / np.sqrt(npol * eff_bw * dt)
    print("1-sigma flux density = %f mJy" % (rms_to_flux * 1000))

    prof = np.ma.array(profile.get_Profile(0, 0).get_amps(), mask=False)
    on_s = None
    on_e = None
    idxs = None

    if args.dutycycle is not None:
        # use Joe Taylor's approach at estimating on-pulse region (bmask)
        mask = np.sort(np.copy(prof))
        sort_idx = np.argsort(np.copy(prof))
        i0 = int(0.5 * (1 - args.dutycycle) * nbins + 0.5)
        ave = mask[i0 - 1]
        rms = np.std(mask[:i0] - ave)

        jz = nbins / 128 if (nbins / 128) > 2 else 2
        ntest = jz - 1

        pulse = 0
        for i in range(nbins):
            mask[i] = 1.0
            nbig = 0
            for j in range(-jz, jz + 1):
                k = i + j
                if k < 0:
                    k += nbins
                if k >= nbins:
                    k -= nbins
                if (np.sort(prof)[k] - ave) > 1.5 * rms:
                    nbig += 1

            if nbig > ntest:
                mask[i] = 0.0
                pulse += 1

        if pulse < 2 or pulse > (nbins - 2):
            print("Could not find a pulse with given duty cycle")
            print("Will use profile mean as baseline")

        s = np.sum(mask * np.sort(prof))
        sw = float(np.sum(mask))
        if sw > 0:
            profile_offpulse_mean = s / sw
        else:
            profile_offpulse_mean = 0

        profile_offpulse_rms = np.sqrt(
            np.sum(mask * (np.sort(prof) - profile_offpulse_mean) ** 2) / sw
        )
        print("off-pulse mean = %f" % profile_offpulse_mean)
        print("off-pulse rms = %f" % profile_offpulse_rms)

        idxs = sort_idx[np.where(mask < 1)]

    elif args.onpulse and len(args.onpulse) == 2:
        # get on-pulse from user provided region
        on_s = int(np.floor(args.onpulse[0] * nbins))
        on_e = int(np.ceil(args.onpulse[1] * nbins))

        prof.mask[on_s:on_e] = True

        profile_offpulse_mean = np.ma.mean(prof)
        profile_offpulse_rms = np.ma.std(prof)
    else:
        print("Please use the interactive window to determine the " "on-pulse region")
        print(" - Left click at the start and end of the region")
        print(" - Right click will undo the last click action")
        # bring up an interactive window for the user to click to identify the
        # on-pulse region
        plt.plot(prof)
        plt.xlabel("Bin number")
        plt.ylabel("Intensity")
        clicks = plt.ginput(n=2, timeout=0)
        plt.close()

        on_s = int(np.floor(clicks[0][0]))
        on_e = int(np.ceil(clicks[1][0]))

        prof.mask[on_s:on_e] = True

        profile_offpulse_mean = np.ma.mean(prof)
        profile_offpulse_rms = np.ma.std(prof)

    prof = profile.get_Profile(0, 0).get_amps()
    prof -= profile_offpulse_mean
    prof /= profile_offpulse_rms
    peak_snr = prof.max()
    prof *= rms_to_flux

    peak_flux = prof.max()
    peak_loc = np.argmax(prof)
    print("peak S/N = %.3f" % peak_snr)
    print("peak flux density = %.3f mJy" % (peak_flux * 1000))

    # print("Attempting to fit gaussian to pulse for W50 estimate")
    in_errs = np.ones_like(prof) * rms_to_flux
    popt, pcov = curve_fit(
        gaussian,
        np.arange(nbins),
        prof,
        p0=[0.1, peak_loc, 0.5],
        sigma=in_errs,
        absolute_sigma=True,
        maxfev=1000,
    )
    errs = np.sqrt(np.diag(pcov))
    w50 = 2.355 * popt[-1]
    w50_err = 2.355 * errs[-1]
    # plt.plot(prof)
    # plt.plot(gaussian(np.arange(nbins), *popt))
    # plt.show()
    # plt.close()

    try:
        eph = arch.get_ephemeris()
        f0 = eph.get_value("F0")
        if f0:
            f0 = float(f0)
            p0 = 1 / f0
        else:
            p0 = eph.get_value("P0")
            p0 = float(p0)

        print("P0 = %f s" % p0)
        eff_width = simps(y=prof, dx=p0 / nbins) / peak_flux
        eff_width_bins = int(np.ceil(eff_width / p0))
        print("effective width = %f ms" % (eff_width * 1000))
        print("effective width (bins) = %d" % eff_width_bins)
        eff_dc = eff_width / p0
        print("approx. duty cycle = %f" % eff_dc)
        mean_flux = peak_flux * eff_dc
        print("approx. mean flux density = %.3f mJy" % (mean_flux * 1000))
        w50_ms = (w50 / nbins) * p0 * 1000
        w50_err_ms = (w50_err / nbins) * p0 * 1000
        print(f"W50 (ms) = {w50_ms} +/- {w50_err_ms}")
    except Exception as e:
        print("There was a problem accessing the ephemeris in the given archive")
        print(repr(e))

    plt.plot(prof, label="Profile")
    plt.axhline(0, color="k")
    plt.axhline(-rms_to_flux, ls=":", color="k")
    plt.axhline(rms_to_flux, ls=":", color="k")
    plt.axvline(peak_loc, ls=":", color="C1")
    plt.plot(gaussian(np.arange(nbins), *popt), color="k", ls="--")
    box = np.zeros_like(prof)
    box[peak_loc - eff_width_bins // 2 : peak_loc + eff_width_bins // 2] = peak_flux
    plt.plot(box, color="b", alpha=0.5)
    if on_s and on_e:
        plt.axvline(on_s, ls="--", color="k")
        plt.axvline(on_e, ls="--", color="k")
    elif idxs is not None:
        masked = np.copy(prof)
        masked[idxs] = np.nan
        plt.plot(masked, color="C3", label="used for baseline stats")
    plt.xlabel("Bin number")
    plt.ylabel("Flux density (Jy)")
    plt.title("Peak flux = {0:g} mJy".format(peak_flux * 1000))
    plt.legend()
    plt.savefig("{0}.fluxcal.png".format(args.file))
