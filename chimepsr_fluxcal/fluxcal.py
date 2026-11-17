#!/usr/bin/env python3

import logging
import os
import sys

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import psrchive

from chimepsr_fluxcal.utils.sefd import SEFD
from chimepsr_fluxcal.utils.baseline import bmask

LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(levelname)s::%(module)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
logger = logging.getLogger()


class FluxCal(object):
    def __init__(self):
        """Constructor method"""
        self.archive_path = None
        self.archive_name_base = None
        self.raw = None
        self.original = None
        self.nchan = None
        self.nsubint = None
        self.subint_dur_max = None
        self.pscrunched = None
        self.nbins = None
        self.prof_orig = None
        self.off_start = None
        self.off_end = None
        self.target = None
        self.raj = None
        self.decj = None
        self.p0 = None
        self.f0 = None
        self.rfi_frac = None
        self.cfreq = None
        self.bw = None
        self.start_mjd = None
        self.tobs = None
        self.dspsr_scale = None
        self.lo_freq = None
        self.hi_freq = None
        self.chan_bw = None
        self.freqs = None
        self.subint_sefds = None
        self.smoothed_sefd = None

    def calibrate(self, fname, off_start=None, off_end=None, smooth=False, plot=False):
        """Primary function call for users that does all of the necessary
        preparation work and applies the solution.

        :param fname: The path to the archive file you want to calibrate.
        :type fname: str
        :param smooth: Whether to smooth the gain (and therefore SEFD) curve
            across the band. This reduces the sharp/disjoint features in the
            data-driven primary beam model. This will also make the calibration
            procedure use the smoothed result.
        :type smooth: bool
        :param plot: Whether to produce diagnostic plots that help to confirm
            that the flux calibration is reasonable (or debug).
        :type plot: bool
        """
        self.prepare_archive(fname)
        self.compute_sefd(smooth=smooth)
        if (off_start is not None) and (off_end is not None):
            self.off_start = off_start if isinstance(off_start, list) else [off_start]
            self.off_end = off_end if isinstance(off_end, list) else [off_end]
        self.apply_solutions()
        if plot:
            self.plot_diagnostics()

    def prepare_archive(self, fname):
        """Prepare the given archive file and update class attributes needed
        for the calibration procedure. This entails:
          - dedispersing the data,
          - ensuring the polarisation state is Stokes,
          - calculating and storing observing parameters

        :param fname: The path to the archive file you want to calibrate.
        :type fname: str
        """
        self.archive_path = os.path.abspath(fname)
        self.archive_name_base = os.path.splitext(fname)[0]
        logger.info(f"Loading archive: {self.archive_path}")
        self.raw = psrchive.Archive_load(self.archive_path)
        self.original = self.raw.clone()
        if not self.raw.get_dedispersed():
            logger.info("Dedispersing...")
            self.raw.dedisperse()

        logger.info("Checking polarisation state")
        if self.raw.get_state() == "Stokes":
            logger.info("Data in Stokes state")
            self.pscrunched = False
        elif self.raw.get_state() == "Intensity":
            logger.info("Data in intensity state - Stokes I only")
            self.pscrunched = True
        elif self.raw.get_state() == "Coherence" or self.raw.get_state() == "PPQQ":
            logger.warning(
                f"Data in {self.raw.get_state()} state - " f"converting to Stokes"
            )
            self.raw.convert_state("Stokes")
            self.pscrunched = False
        else:
            logger.warning("Unknown data state - converting to Stokes I only")
            self.raw.pscrunch()  # Total intenisty
            self.pscrunched = True

        self.nbins = self.raw.get_nbin()
        self.nchan = self.raw.get_nchan()
        self.nsubint = self.raw.get_nsubint()

        # logger.info("Inspecting weights and re-normalising if required")
        # weights = self.raw.get_weights()
        # max_weight = np.max(weights)
        # subint_dur = [
        #     self.raw.get_Integration(ii).get_duration()
        #     for ii in range(self.nsubint)
        # ]
        # self.subint_dur_max = np.max(subint_dur)
        #
        # for ii in range(self.nsubint):
        #     isub = self.raw.get_Integration(ii)
        #     dur = isub.get_duration()
        #     for ch in range(self.nchan):
        #         w = isub.get_weight(ch)
        #         if w < max_weight:
        #             w *= (self.subint_dur_max / dur)
        #             if w > max_weight:
        #                 w = max_weight
        #         w /= max_weight
        #         isub.set_weight(ch, float(w))

        # self.raw.unload('reweighted.ar')

        weights = self.raw.get_weights()
        self.rfi_frac = 1 - np.sum(weights) / float(self.nsubint * self.nchan)
        logger.info(f"RFI fraction = {self.rfi_frac}")
        self.prof_orig = self.raw.get_data()[:, 0, :, :].mean(axis=(0, 1))

        # central frequency in MHz
        self.cfreq = self.raw.get_centre_frequency()
        # total bandwidth in MHz
        self.bw = self.raw.get_bandwidth()
        # channel bandwidth in MHz
        self.chan_bw = float(self.bw) / self.nchan
        # high and low frequency boundaries
        self.lo_freq = self.cfreq - np.abs(self.bw) / 2.0
        self.hi_freq = self.cfreq + np.abs(self.bw) / 2.0

        # create a list of frequencies corresponding to the channel ordering
        if self.bw < 0:
            logger.warning(
                "Bandwidth is negative, highest frequency corresponds to "
                "lowest channel number"
            )
            self.freqs = np.arange(self.hi_freq, self.lo_freq, self.chan_bw)
        else:
            self.freqs = np.arange(self.lo_freq, self.hi_freq, self.chan_bw)

        # total observation time
        self.tobs = self.raw.integration_length()
        # target source coordinates (from header)
        coord = self.raw.get_coordinates().getHMSDMS()
        self.raj = coord.split()[0]
        self.decj = coord.split()[-1]
        # observation start time (MJD)
        self.start_mjd = float(self.raw.start_time().strtempo())
        # get the global scale value from dspsr
        s = self.original.execute("e dspsr:scale")
        self.dspsr_scale = int(s.strip().split("=")[1])

        # get some pulsar properties from the installed ephemeris
        eph = self.raw.get_ephemeris()
        f0 = eph.get_value("F0")
        if f0:
            self.f0 = float(f0)
            self.p0 = 1 / self.f0
        else:
            p0 = eph.get_value("P0")
            self.p0 = float(p0)
            self.f0 = 1 / self.p0

        for n in ["PSR", "PSRJ"]:
            target = eph.get_value(n)
            if target:
                self.target = target
                break

        logger.info(
            f"Observation summary:\n"
            f"Archive name = {self.archive_path}\n"
            f"Target name = {self.target}\n"
            f"RA, Dec = {self.raj} {self.decj}\n"
            f"Start MJD = {self.start_mjd}\n"
            f"Observation length = {self.tobs} s\n"
            f"Central frequency = {self.cfreq} MHz\n"
            f"Bandwidth = {self.bw} MHz\n"
            f"State = {self.raw.get_state()}"
        )

    def compute_sefd(self, smooth=False):
        """For all subintegrations in the loaded archive file, compute the SEFD
        as a function of frequency.

        :param smooth: Whether to smooth the gain (and therefore SEFD) curve
            across the band. This reduces the sharp/disjoint features in the
            data-driven primary beam model.
        :type smooth: bool
        """
        self.subint_sefds = []
        logger.info("Computing SEFD over band for all subintegrations")
        for isub in range(self.nsubint):
            # get the middle time of the subintegration
            sub_start = float(
                self.raw.get_Integration(isub).get_start_time().strtempo()
            )
            sub_dur = self.raw.get_Integration(isub).get_duration()

            mid_time = sub_start + 0.5 * sub_dur / 86400.0

            logger.debug(f"Computing SEFD over band for subint = {isub}")
            sefd = SEFD()
            sefd.set_pointing(self.raj, self.decj, mjd=mid_time)
            sefd.compute_sefd(smooth=smooth)
            self.smoothed_sefd = smooth

            self.subint_sefds.append(sefd)

    def print_offpulse_region(self):
        """
        Print the off-pulse region based on either what psrchive provides
        or what the user inputs
        """
        if self.off_start is None or self.off_end is None:
            off_start_bin = int(
                float(str(self.raw.execute("e off:start")).strip().split("=")[-1])
            )
            off_start_phs = float(off_start_bin) / self.nbins
            off_end_bin = int(
                float(str(self.raw.execute("e off:end")).strip().split("=")[-1])
            )
            off_end_phs = float(off_end_bin) / self.nbins
            logger.info(
                "Getting off-pulse statistics from " "psrchive-derived off-pulse region"
            )
            print(
                f"({off_start_phs}, {off_end_phs}) [phase], "
                f"({off_start_bin:d}, {off_end_bin:d}) [bins]"
            )
        else:
            logger.info(
                "Getting off-pulse statistic from user-defined " "off-pulse region"
            )
            for s, e in zip(self.off_start, self.off_end):
                s_bin = int(np.ceil(s * self.nbins))
                e_bin = int(np.floor(e * self.nbins))
                print(f"({s}, {e}) [phase], ({s_bin}, {e_bin}) [bins]")

    def beam_fade_corrected_flux_density(self):
        """
        As the pulsar drifts through the primary beam, subintegrations should
        be weighted when averaging to the mean profile. The profiles from
        subintegrations in the early and late parts of the observation should
        be down-weighted by their gain relative to the gain chosen for the SEFD

        The weights will be time and frequency dependent (since the beam shape
        is frequency dependent). Nominally, dividing the gains for each
        (subint, chan) pair by the mid-point gain (for each channel) should
        normalize things such that we can simply multiple the weights by the
        data when averaging.
        """

        if self.smoothed_sefd:
            peak_subint_gain = self.subint_sefds[self.nsubint // 2].sgain
            weights = (
                np.array([x.sgain for x in self.subint_sefds])
                / peak_subint_gain[None, :]
            )
        else:
            peak_subint_gain = self.subint_sefds[self.nsubint // 2].gain
            weights = (
                np.array([x.gain for x in self.subint_sefds])
                / peak_subint_gain[None, :]
            )

        # times = np.array([x.time.mjd for x in self.subint_sefds])
        # times = (times - times[0]) * 86400
        # plt.plot(times, weights[:, 0], label="400 MHz")
        # plt.plot(times, weights[:, 512], label="600 MHz")
        # plt.plot(times, weights[:, -1], label="800 MHz")
        # plt.axvline(times[self.nsubint // 2], ls=":", color='k')
        # plt.axhline(0, color='k')
        # plt.xlabel("Seconds since obs. start")
        # plt.ylabel("Relative weight")
        # plt.legend()
        # plt.show()

        cut800 = 0.1
        ncut = np.where(weights[:, -1] < cut800)[0].size
        logger.info(f"removed {ncut} subints as their 800 MHz weight is <{cut800}")

        rdata = self.raw.get_data()[:, 0, :, :]

        wstack = np.average(
            rdata, weights=(weights[..., None] * np.ones_like(rdata)), axis=1
        )
        wstack[np.where(weights[:, -1] < cut800)] = np.nan

        data = np.delete(rdata, np.where(weights[:, -1] < cut800), axis=0)
        weights = np.delete(weights, np.where(weights[:, -1] < cut800), axis=0)

        wprof = np.average(
            data, weights=(weights[..., None] * np.ones_like(data)), axis=(0, 1)
        )

        # wwaterfall = np.average(data, weights=(weights[..., None] * np.ones_like(data)), axis=0)
        wstack2 = np.average(
            data, weights=(weights[..., None] * np.ones_like(data)), axis=1
        )
        # wstack2 = np.average(data,
        #                     weights=(weights[..., None] * np.ones_like(data))**2,
        #                     axis=1)

        peak_flux = wprof.max()
        # boxcar_prof = thresholding(wprof, lag=self.nbins // 8, threshold=5, influence=0.2)
        # bins = np.arange(self.nbins)
        # plt.plot(bins, wprof, color='k')
        # plt.plot(bins, boxcar_prof['signals'] * peak_flux, color='r')
        # plt.plot(bins, boxcar_prof['avgFilter'], color='g')
        # plt.fill_between(bins, boxcar_prof['avgFilter']-boxcar_prof['stdFilter'], boxcar_prof['avgFilter']+boxcar_prof['stdFilter'], color='g', alpha=0.4)
        # plt.show()
        # plt.acorr(wprof)
        # plt.show()
        # exit()
        #
        # eff_width_bins = len(np.argwhere(boxcar_prof['signals'] > 0))
        # eff_dc = eff_width_bins / float(self.nbins)
        # eff_width_ms = eff_dc * self.p0 * 1000
        fluence = simps(y=wprof, dx=self.p0 / self.nbins)  # mJy-s
        # print(f"approx. profile fluence = {fluence:.3f} mJy s")
        eff_width_ms = (fluence / peak_flux) * 1000
        eff_width_bins = int((fluence / peak_flux) / (self.p0 / self.nbins))
        eff_dc = eff_width_bins / float(self.nbins)

        print(f"approx. pulse width = {eff_width_ms:.3f} ms")
        print(f"                    = {eff_width_bins:d} bins")
        print(f"approx. duty cycle = {eff_dc:.3f}")
        mean_flux = peak_flux * eff_dc
        print(f"approx. mean flux density = {mean_flux:.3f} mJy")

        boxcar = np.zeros_like(wprof)
        s = int(np.where(wprof == peak_flux)[0] - eff_width_bins // 2)
        e = int(s + eff_width_bins)
        boxcar[s:e] = peak_flux
        plt.plot(wprof, color="C0", lw=3, label="weighted mean profile")
        # plt.plot(boxcar, color='k', alpha=0.3, label=f"boxcar eff. width (dc={eff_dc:g})")
        plt.xlim(0, self.nbins)
        plt.xticks(np.arange(0, self.nbins + 1, self.nbins // 4))
        plt.title(
            fr"{self.target} ($S_{{mean}}={mean_flux:g}\,{{\rm mJy}}$)", fontsize=20
        )
        plt.xlabel("Phase bin", fontsize=18)
        plt.ylabel("Flux density (mJy)", fontsize=18)
        plt.tick_params(labelsize=16)
        # plt.legend()
        plt.savefig(
            "profile.png", format="png", dpi=600, bbox_inches="tight", transparent=True
        )
        plt.close()
        #
        # plt.imshow(wwaterfall, aspect="auto")
        # plt.show()

        min_power = min(rdata.mean(axis=1).min(), wstack.min())
        max_power = max(rdata.mean(axis=1).max(), wstack.max())
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        a = ax1.imshow(
            rdata.mean(axis=1),
            aspect="auto",
            vmin=min_power,
            vmax=max_power,
            origin="lower",
            extent=[0, self.nbins, 0, self.nsubint],
        )
        ax1.set_xticks(np.arange(0, self.nbins + 1, self.nbins // 4))
        ax1.set_title("Full data set", fontsize=20)
        ax1.set_ylabel("Subintegration number", fontsize=18)
        ax1.set_xlabel("Phase bin", fontsize=18)
        ax1.tick_params(labelsize=16)
        # ac = fig.colorbar(a, ax=ax1)
        # ac.set_label("mJy")
        b = ax2.imshow(
            wstack,
            aspect="auto",
            vmin=min_power,
            vmax=max_power,
            origin="lower",
            extent=[0, self.nbins, 0, self.nsubint],
        )
        ax2.set_xticks(np.arange(0, self.nbins + 1, self.nbins // 4))
        ax2.set_title("Beam fade cut", fontsize=20)
        ax2.set_xlabel("Phase bin", fontsize=18)
        ax2.tick_params(labelsize=16, labelleft=False)
        # bc = fig.colorbar(b, ax=ax2)
        # bc.set_label("mJy")
        # plt.show()
        plt.savefig(
            "beam_fade_cut.png",
            format="png",
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )
        plt.close()

    def apply_solutions(self):
        """The primary goal here is to scale the noise in each subintegration
        to be consistent across the observation. i.e., the noise for our system
        doesn't change (by much at least). We DON'T want to scale things to
        try and keep the pulsar signal consistent - that only works when the
        pulsar stays in the same part of the primary beam model.

        Effectively, we want to choose the SEFD around the mid-point of the
        observation and apply those numbers to all subintegrations. We won't
        remove the effect of the pulsar drifting through the beam model, but
        the data will be correctly scaled.
        """
        logger.info("Scaling subint profiles to Jy units")

        # choose the SEFD to apply for all subints such that the consistent
        # noise is converted into Jy units
        # TODO: is just taking the middle subint ideal? We know emperically
        #  that in many cases the middle subint doesn't correspond to "maximum
        #  sensitivity" - it's often offset in time...
        if self.smoothed_sefd:
            subint_sefd_bp = self.subint_sefds[self.nsubint // 2].ssefd
        else:
            subint_sefd_bp = self.subint_sefds[self.nsubint // 2].sefd

        # self.print_offpulse_region()

        for isub in range(self.nsubint):
            logger.debug(f"Applying solution to subint = {isub}")

            subint = self.raw.get_Integration(isub)
            subint_duration = subint.get_duration()

            # get off-pulse statistics - default is to use the psrchive defined
            # values
            # subint_stats = subint.baseline_stats()
            # subint_offpulse_mean = subint_stats[0].astype(float)
            # subint_offpulse_rms = np.sqrt(subint_stats[1].astype(float))
            # PSRCHIVE baseline states don't seem to work once
            # dedisperse() has been called (the off-pulse region limits
            # don't seem to be updated correctly)
            subint_offpulse_mean = np.zeros((self.raw.get_npol(), self.nchan))
            subint_offpulse_rms = np.zeros((self.raw.get_npol(), self.nchan))
            if self.off_start is not None or self.off_end is not None:
                logger.debug("Using provided on/off pulse regions")
                start_end_pairs = [
                    [int(np.ceil(s * self.nbins)), int(np.floor(e * self.nbins))]
                    for s, e in zip(self.off_start, self.off_end)
                ]

                for ipol in range(self.raw.get_npol()):
                    for ichan in range(self.nchan):
                        _prof = subint.get_Profile(ipol, ichan).get_amps()
                        offpulse = np.array([])
                        for se in start_end_pairs:
                            offpulse = np.append(offpulse, _prof[se[0] : se[1]])

                        subint_offpulse_mean[ipol, ichan] = np.mean(offpulse)
                        subint_offpulse_rms[ipol, ichan] = np.std(offpulse)
            else:
                logger.debug(
                    "Attempting to use `bmask` automated " "baseline/rms estimation"
                )
                # get off-pulse indices first from mean profile first
                offidx = bmask(self.prof_orig)
                for ipol in range(self.raw.get_npol()):
                    for ichan in range(self.nchan):
                        _prof = subint.get_Profile(ipol, ichan).get_amps()

                        subint_offpulse_mean[ipol, ichan] = np.mean(_prof[offidx])
                        subint_offpulse_rms[ipol, ichan] = np.std(_prof[offidx])

            for ichan in range(self.nchan):
                dt = subint_duration / self.nbins
                radiometer_advantage = np.sqrt(2 * np.abs(self.chan_bw * 1.0e6) * dt)
                local_conversion = subint_sefd_bp[ichan] / radiometer_advantage

                # get the local rms from Stokes I, that way the S/N for
                # all Stokes will be relative to Stokes I
                local_rms = subint_offpulse_rms[0, ichan]

                # check the local rms make sense (once scaled correctly)
                # compared to the expected radiometer noise
                scaled_local_rms = self.dspsr_scale * local_rms
                noise_is_radiometer_like = np.isclose(
                    scaled_local_rms, radiometer_advantage, atol=0, rtol=0.1
                )
                if not noise_is_radiometer_like:
                    logger.warning(
                        f"Scaled local rms is not within 10% of radiometer "
                        f"expectation for (subint, chan) = ({isub}, {ichan})!"
                    )
                    logger.warning(f"Scaled local rms = {scaled_local_rms:g}")
                    logger.warning(f"Radiometer expectation = {radiometer_advantage:g}")
                    logger.warning(
                        f"Ratio = {scaled_local_rms / radiometer_advantage:g}"
                    )
                else:
                    logger.debug(
                        f"Scaled local rms is within threshold of radiometer "
                        f"expectation for (subint, chan) = ({isub}, {ichan})"
                    )

                npol = self.raw.get_npol()

                for ipol in range(npol):
                    # subtract local mean
                    local_mean = subint_offpulse_mean[ipol, ichan]
                    # local_rms = subint_offpulse_rms[0, ichan]

                    # flag bad profiles
                    if local_rms == 0:
                        logger.debug(
                            f"RMS subint={isub} chan={ichan} "
                            f"pol={ipol} is zero ({local_rms}) - "
                            f"flagging channel"
                        )

                        self.raw.get_Integration(isub).set_weight(ichan, 0)
                        continue

                    # remove the baseline and scale into mJy units
                    self.raw.get_Integration(isub).get_Profile(ipol, ichan).offset(
                        -local_mean
                    )

                    self.raw.get_Integration(isub).get_Profile(ipol, ichan).scale(
                        1000 * local_conversion / local_rms  # units are mJy
                    )

        # get an estimate of the beam-corrected flux density
        self.beam_fade_corrected_flux_density()

        logger.info("Finished scaling, updating metadata and writing file")
        self.raw.set_scale("Jansky")
        self.raw.dededisperse()
        self.raw.unload(f"{self.archive_name_base}.calib")

    def plot_diagnostics(self):
        self.plot_bandpass_before_after()

    def plot_bandpass_before_after(self):
        tmp = self.original.clone()
        tmp.dedisperse()
        tmp.pscrunch()
        tmp.tscrunch()
        before_offpulse_stats = tmp.get_Integration(0).baseline_stats()
        before_offpulse_mean = before_offpulse_stats[0].astype(float)
        before_offpulse_rms = np.sqrt(before_offpulse_stats[1].astype(float))

        before_offpulse_rms[before_offpulse_rms == 0] = None
        before_offpulse_mean[before_offpulse_mean == 0] = None

        tmp = self.raw.clone()
        tmp.dedisperse()
        tmp.pscrunch()
        tmp.tscrunch()
        after_offpulse_stats = tmp.get_Integration(0).baseline_stats()
        after_offpulse_mean = after_offpulse_stats[0].astype(float)
        after_offpulse_rms = np.sqrt(after_offpulse_stats[1].astype(float))

        after_offpulse_rms[after_offpulse_rms == 0] = None
        after_offpulse_mean[after_offpulse_mean == 0] = None

        fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=plt.figaspect(0.4))
        ax1.set_title("Off-pulse mean intensity")
        ax1.plot(self.freqs, before_offpulse_mean[0], label="Before")
        ax1.set_ylabel("Arbitrary units")
        ax2.plot(self.freqs, after_offpulse_mean[0], label="After")
        ax2.set_ylabel("Flux density (Jy)")
        ax2.set_xlabel("Frequency (MHz)")
        ax1.legend()
        ax2.legend()
        plt.subplots_adjust(hspace=0)
        plt.savefig(f"{self.archive_name_base}.bandpass.mean.png", bbox_inches="tight")
        plt.close()

        fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=plt.figaspect(0.4))
        ax1.set_title("Off-pulse rms intensity")
        ax1.plot(self.freqs, before_offpulse_rms[0], label="Before")
        ax1.set_ylabel("Arbitrary units")
        ax2.plot(self.freqs, after_offpulse_rms[0], label="After")
        ax2.set_ylabel("Flux density (Jy)")
        ax2.set_xlabel("Frequency (MHz)")
        ax1.legend()
        ax2.legend()
        plt.subplots_adjust(hspace=0)
        plt.savefig(f"{self.archive_name_base}.bandpass.rms.png", bbox_inches="tight")
        plt.close()
