#!/usr/bin/env python3

import pkg_resources
import logging
import warnings
from astropy.coordinates import SkyCoord, FK5, Galactic, ICRS
from astropy.time import Time
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from healpy import read_map
from healpy.pixelfunc import get_interp_val
import h5py

LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(levelname)s::%(module)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
logger = logging.getLogger()

SKY_TEMP_MAP_FILE = pkg_resources.resource_filename(
    "chimepsr_fluxcal", "data/haslam408_ds_Remazeilles2014.fits"
)
SKY_TEMP_MAP = read_map(SKY_TEMP_MAP_FILE, dtype=None)

RCVR_TEMP_FILE = pkg_resources.resource_filename(
    "chimepsr_fluxcal", "data/20190530_and_20190614_system_temperature_measurement.h5"
)


def sky_temperature_at_coord(coord, sky_index=-2.55):
    # convert ra, dec to Galactic coordinate
    logger.debug("Converting pointing to Galactic coordinates")
    gal = coord.transform_to(Galactic())
    longitude = gal.l.deg
    latitude = gal.b.deg

    # retrieve sky temperature from sky map
    logger.debug(f"Getting Tsky at (l, b) = ({ longitude }, { latitude }) deg")
    map_tsky = get_interp_val(
        SKY_TEMP_MAP, longitude, latitude, nest=False, lonlat=True
    )

    # scale sky temperature to center frequency of the survey
    logger.debug("Interpolating sky temperature across CHIME band")
    map_freq_mhz = 408.0
    freqs = np.linspace(400.1953125, 800.1953125, 100)
    tsky_sample = map_tsky * (freqs / map_freq_mhz) ** sky_index

    # compute a cubic spline interpolation
    tsky_spline = interp1d(freqs, tsky_sample, kind="cubic")

    return tsky_spline


def receiver_temperature():
    # access the receiver temperature data from disk
    logger.debug(f"Loading receiver measurements from { RCVR_TEMP_FILE }")
    with h5py.File(RCVR_TEMP_FILE, "r") as handler:
        trecv = handler["Trecv"][:]

    # subsititute missing/corrupted data with the median
    logger.debug("Estimating mean and replacing missing values")
    # since there are some channels with no data at all, ignore the raised
    # warnings due to taking means on an empty slice (since they will be
    # replaced in the next step anyway)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        av_trecv = np.nanmean(trecv, axis=1)

    av_trecv[np.isnan(av_trecv)] = np.nanmedian(av_trecv)

    # compute the 2nd-order polynomial coefficients that fit the data
    logger.debug("Interpolating receiver temperature across CHIME band")
    freqs = np.linspace(400.1953125, 800.1953125, 1024)
    polyf = np.polyfit(freqs, av_trecv, 2)
    poly = np.poly1d(polyf)

    return poly


def mesh_balun_temperature(freqs):
    logger.debug("Approximating structure temperature contributions")
    # calculate the approx. temperature contributed by the telescope structure
    tmesh = 1.0 + 3.0 * (freqs - 400.1953125) / 400.1953125
    tbalun = np.repeat(7.0, freqs.size)

    return tmesh + tbalun


def get_tsys(ra, dec, epoch=None):
    """
    Given a Right Ascension (hh:mm:ss) and Declination (dd:mm:ss) and, optionally,
    the epoch of the observation in MJD (i.e., if the coordiantes are not already
    in the J2000 frame), compute the system temperature.
    """
    if epoch:
        ep = Time(epoch, format="mjd", scale="utc")
    else:
        ep = Time("J2000", format="jyear_str")

    coord_fk5 = SkyCoord(ra, dec, frame=FK5(equinox=ep), unit=(u.hourangle, u.degree))
    coord_fk5_j2000 = coord_fk5.transform_to(frame=FK5(equinox="J2000"))

    # get the spline/polyfits from data
    logger.debug("Calculating temperature contributions")
    sspl = sky_temperature_at_coord(coord_fk5_j2000)
    rspl = receiver_temperature()

    # interpolate across the band
    f = np.linspace(400.1953125, 800.1953125, 1024)
    tsky = sspl(f)
    trec = rspl(f)
    tother = mesh_balun_temperature(f)

    tsys = tsky + trec + tother

    return tsys, tsky, trec, tother


def get_tsys_coord(coord):
    logger.debug("Calculating temperature contributions")
    # get the spline/polyfits from data
    sspl = sky_temperature_at_coord(coord.transform_to(FK5(equinox="J2000")))
    rspl = receiver_temperature()

    # interpolate across the band
    f = np.linspace(400.1953125, 800.1953125, 1024)
    tsky = sspl(f)
    trec = rspl(f)
    tother = mesh_balun_temperature(f)

    tsys = tsky + trec + tother

    return tsys, tsky, trec, tother
