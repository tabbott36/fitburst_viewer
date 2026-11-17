#!/usr/bin/env python3

import logging
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
import matplotlib.pyplot as plt

from .gain import get_gain_coord
from .tsys import get_tsys_coord

LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(levelname)s::%(module)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
logger = logging.getLogger()


class SEFD(object):
    def __init__(self):
        self.tsys = None  # total system temperature
        self.tsky = None  # temperature contribution from sky
        self.trcvr = None  # temperature contribution from receiver
        self.tstruct = None  # temperature contribution from telescope structure
        self.gain = None  # beam-corrected gain
        self.sgain = None  # "smoothed" gain
        self.mean_gain = None
        self.rms_gain = None
        self.freqs = np.linspace(400.1953125, 800.1953125, 1024)
        self.sefd = None
        self.ssefd = None  # "smoothed" sefd
        self.mean_sefd = None
        self.rms_sefd = None
        self.time = None
        self.pointing = None

    def set_pointing(self, ra, dec, mjd=None, epoch=None):
        if isinstance(ra, str):
            logger.debug("RA passed as string, assuming hh:mm:ss.ss format")
            ra_unit = u.hourangle
        elif isinstance(ra, float):
            logger.debug("RA passed as float, assuming it is in degrees")
            ra_unit = u.deg
        else:
            ra_unit = None
            logger.error("Unrecognized Right Ascension parameter format.")
            logger.error(
                "Please provide the RA in degrees (float) or in "
                "hh:mm:ss.ss format (string)"
            )
            logger.warning(
                "Not setting RA or unit type. "
                "This will cause other functions to fail."
            )

        if isinstance(dec, str):
            logger.debug("Dec passed as string, assuming dd:mm:ss.ss format")
            dec_unit = u.deg
        elif isinstance(dec, float):
            logger.debug("Dec passed as float, assuming it is in degrees")
            dec_unit = u.deg
        else:
            dec_unit = None
            logger.error("Unrecognized Declination parameter format.")
            logger.error(
                "Please provide the Dec in degrees (float) or in "
                "dd:mm:ss.ss format (string)"
            )
            logger.warning(
                "Not setting Dec or unit type. "
                "This will cause other functions to fail."
            )

        if epoch:
            ep = Time(epoch, format="mjd", scale="utc")
        else:
            logger.debug("No epoch provided, assuming J2000")
            ep = Time("J2000", format="jyear_str")

        if mjd:
            time = Time(mjd, format="mjd", scale="utc")
        else:
            logger.warning("No observation time provided, assuming NOW")
            time = Time.now()

        self.time = time
        self.pointing = SkyCoord(
            ra, dec, frame=FK5(equinox=ep), obstime=time, unit=(ra_unit, dec_unit)
        )
        logger.debug(
            f"Created pointing at " f"{self.pointing.to_string('hmsdms', sep=':')}"
        )

    def compute_sefd(self, efficiency=0.5, smooth=False):
        self.tsys, self.tsky, self.trcvr, self.tstruct = get_tsys_coord(self.pointing)
        if smooth:
            logger.debug("Smoothing gain and SEFD curves")
            logger.debug(f"Aperture efficiency set to {efficiency}")
        self.gain, self.sgain = get_gain_coord(
            self.pointing, efficiency=efficiency, smooth=smooth
        )

        if any(self.gain == 0):
            logger.warning("Gain curve contains zeros!")
            logger.warning("Are the coordinates/time set correctly?")
            logger.warning(
                f"{self.pointing.to_string('hmsdms')} @ " f"{self.pointing.equinox.mjd}"
            )

        self.mean_gain = np.mean(self.gain)
        self.rms_gain = np.std(self.gain)
        logger.debug(f"Mean gain = {self.mean_gain} K/Jy")
        logger.debug(f"rms gain = {self.rms_gain} K/Jy")

        self.sefd = self.tsys / self.gain
        if self.sgain is not None:
            self.ssefd = self.tsys / self.sgain

        self.mean_sefd = np.mean(self.sefd)
        self.rms_sefd = np.std(self.sefd)
        logger.debug(f"Mean SEFD = {self.mean_sefd} Jy")
        logger.debug(f"rms SEFD = {self.rms_sefd} Jy")

    def plot_tsys(self):
        if self.tsys is not None:
            logger.info("Plotting system temperature across band")
            plt.figure()
            plt.plot(self.freqs, self.tsky, label=r"$T_{\rm sky}$")
            plt.plot(self.freqs, self.trcvr, label=r"$T_{\rm rcvr}$")
            plt.plot(self.freqs, self.tstruct, label=r"$T_{\rm struct}$")
            plt.plot(self.freqs, self.tsys, color="k", label=r"$T_{\rm total}$")
            plt.xlabel(r"$\rm Frequency\ [MHz]$", fontsize=14)
            plt.ylabel(r"$\rm Temperature\ [K]$", fontsize=14)
            plt.title(
                f"Pointing {self.pointing.to_string('hmsdms', sep=':')}  "
                f"MJD {self.time.mjd}"
            )
            plt.tick_params(labelsize=12)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            logger.error("System temperature has not be computed or failed")

    def plot_gain(self):
        if self.gain is not None:
            logger.info("Plotting gain across band")
            plt.figure()

            if self.sgain is not None:
                plt.plot(self.freqs, self.gain, color="k", ls=":")
                plt.plot(self.freqs, self.sgain, color="k")
            else:
                plt.plot(self.freqs, self.gain, color="k")
            plt.xlabel(r"$\rm Frequency\ [MHz]$", fontsize=14)
            plt.ylabel(r"$\rm Gain\ [K/Jy]$", fontsize=14)
            plt.title(
                f"Pointing {self.pointing.to_string('hmsdms', sep=':')}  "
                f"MJD {self.time.mjd}"
            )
            plt.tick_params(labelsize=12)
            plt.tight_layout()
            plt.show()
        else:
            logger.error("Beam-corrected gain has not be computed or failed")

    def plot_sefd(self):
        if self.sefd is not None:
            logger.info("Plotting SEFD across band")
            plt.figure()
            if self.ssefd is not None:
                plt.plot(self.freqs, self.sefd, color="k", ls=":")
                plt.plot(self.freqs, self.ssefd, color="k")
            else:
                plt.plot(self.freqs, self.sefd, color="k")
            plt.xlabel(r"$\rm Frequency\ [MHz]$", fontsize=14)
            plt.ylabel(r"$\rm SEFD\ [Jy]$", fontsize=14)
            plt.title(
                f"Pointing { self.pointing.to_string('hmsdms', sep=':') }  "
                f"MJD { self.time.mjd }"
            )
            plt.tick_params(labelsize=12)
            plt.tight_layout()
            plt.show()
        else:
            logger.error("SEFD has not be computed or failed")
