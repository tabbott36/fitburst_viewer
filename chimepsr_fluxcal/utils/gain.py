#!/usr/bin/env python3

import logging
import sys
from astropy.coordinates import SkyCoord, FK5
from astropy.time import Time
from astropy.constants import k_B
import astropy.units as u
import numpy as np

import beam_model as bm

LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(levelname)s::%(module)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
logger = logging.getLogger()

config = bm.current_config
# config['datapath_xpol'] = '/home/bmeyers/code/CHIME-Pulsar/beam-model/data/beam_XX_v1.h5'
# config['datapath_ypol'] = '/home/bmeyers/code/CHIME-Pulsar/beam-model/data/beam_YY_v1.h5'

COLLECTING_AREA = 80.0 * 80.0  # units of m^2
MAX_GAIN = 1e-26 * COLLECTING_AREA / (2 * k_B.value)  # units of K/Jy

PRIMARY_BEAM = bm.primary.DataDrivenPrimaryBeamModel(config, True)


def get_primary_beam_response(coord, time, freqs):
    logger.debug("Converting sky coordinates to telescope coordinates")
    above_horizon = PRIMARY_BEAM.is_equatorial_above_horizon(
        coord.ra.deg, coord.dec.deg, time.datetime
    )
    if not above_horizon:
        logger.error(
            "Given RA and Dec are not above the horizon at the " "provided time!"
        )
        logger.error(
            f"RA = {coord.ra.deg}  " f"Dec = {coord.dec.deg}  " f"Time = {time}"
        )
        logger.warning("Setting beam response to 0")

        pb_response = np.zeros_like(freqs)
    else:
        # given a sky position, convert to "telescope" coordinates
        x, y = PRIMARY_BEAM.get_position_from_equatorial(
            coord.ra.deg, coord.dec.deg, time.datetime
        )

        # get the primary beam response at the given position, across the band
        logger.debug(
            f"Calculating primary beam response at:\n"
            f"(RA, Dec) = ({ coord.ra.deg }, { coord.dec.deg }) deg\n"
            f"(x, y) = ({ x }, { y })"
        )
        pb_response = PRIMARY_BEAM.get_sensitivity(np.array([[x, y]]), freqs)[0]

    # re-normalize it to be unity at zenith, since that is where the MAX_GAIN
    # value is valid
    zenith_response = PRIMARY_BEAM.get_sensitivity(np.array([[0, 0]]), freqs)[0]

    renorm_pb_response = np.array(pb_response) / zenith_response

    return renorm_pb_response


def corrected_gain(response, eff=0.5):
    # correct the nominal gain by an efficiency and the PB response
    logger.debug("Calculating beam-corrected gains")
    return MAX_GAIN * eff * response


def get_gain(ra, dec, mjd=None, epoch=None, efficiency=0.5):
    """
    Given a Right Ascension (hh:mm:ss) and Declination (dd:mm:ss) an observation time in MJD,
    and, optionally, the epoch of the observation in MJD (i.e., if the coordinates are not
    already in the J2000 frame), compute the system gain.
    """
    if epoch:
        ep = Time(epoch, format="mjd", scale="utc")
    else:
        ep = Time("J2000", format="jyear_str")

    coord_fk5 = SkyCoord(ra, dec, frame=FK5(equinox=ep), unit=(u.hourangle, u.degree))
    coord_fk5_j2000 = coord_fk5.transform_to(frame=FK5(equinox="J2000"))

    # get the primary beam response and then correct the ideal gain
    if mjd:
        time = Time(mjd, format="mjd", scale="utc")
    else:
        time = Time.now()
    f = np.linspace(400.1953125, 800.1953125, 1024)
    pb_resp = get_primary_beam_response(coord_fk5_j2000, time, f)
    gain = corrected_gain(pb_resp, efficiency)

    return gain


def get_gain_coord(coord, efficiency=0.5, smooth=False):
    # get the primary beam response and then correct the ideal gain
    f = np.linspace(400.1953125, 800.1953125, 1024)
    coord_fk5_j2000 = coord.transform_to(FK5(equinox="J2000"))
    pb_resp = get_primary_beam_response(coord_fk5_j2000, coord.obstime, f)
    gain = corrected_gain(pb_resp, efficiency)

    if smooth:
        # fit a polynomial to the gain curve to capture the general trend
        # removes the small-scale frequency dependent effects
        gain_poly = np.poly1d(np.polyfit(f, gain, 4))
        return gain, gain_poly(f)
    else:
        return gain, None
