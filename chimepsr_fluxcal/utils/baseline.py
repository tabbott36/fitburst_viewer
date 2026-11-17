#!/usr/bin/env python3

import logging
import numpy as np
from scipy.stats import median_abs_deviation
from scipy import sparse
from scipy.sparse.linalg import spsolve

LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(levelname)s::%(module)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
logger = logging.getLogger()


def bmask(prof, dc=0.1):
    """
    Use Joe taylor's approach to determining a reasonably robust baseline and
    rms given a guess of the pulsar duty cycle.

    :param prof: the pulsar profile from which to extract the baseline/rms
    :type prof: np.ndarray

    :param dc: a rough estimate of the pulsar duty cycle (10% is probably ok
    in most cases)
    :type dc: float

    :return: off-pulse mean, off-pulse rms
    :rtype: float, float
    """
    nbins = len(prof)
    sorted_prof_args = np.argsort(prof)
    sorted_prof = prof[sorted_prof_args]
    mask = np.ones_like(sorted_prof)

    i0 = int(0.5 * (1 - dc) * nbins + 0.5)
    avg = sorted_prof[i0 - 1]
    rms = np.std(sorted_prof[:i0] - avg)
    logger.debug(f"First iteration mean = {avg}")
    logger.debug(f"First iteration rms = {rms}")

    jz = nbins // 128 if (nbins // 128) > 2 else 2
    ntest = jz - 1
    logger.debug(f"Number of trials = {ntest}")

    pulse = 0
    for i in range(nbins):
        nbig = 0
        for j in range(-jz, jz + 1):
            # print(i, j)
            k = i + j
            if k < 0:
                k += nbins
            if k >= nbins:
                k -= nbins
            if (sorted_prof[k] - avg) > 1.5 * rms:
                nbig += 1

        if nbig > ntest:
            mask[i] = 0.0
            pulse += 1

    s = np.sum(mask * np.sort(prof))
    sw = float(np.sum(mask))
    if sw > 0:
        profile_offpulse_mean = s / sw
        profile_offpulse_rms = np.sqrt(
            np.sum(mask * (np.sort(prof) - profile_offpulse_mean) ** 2) / sw
        )
    else:
        logger.warning(
            "Weight sum is zero, using simple stats to " "estimate baseline and rms."
        )
        profile_offpulse_mean = np.mean(prof)
        profile_offpulse_rms = median_abs_deviation(prof)

    logger.debug(f"off-pulse mean = {profile_offpulse_mean}")
    logger.debug(f"off-pulse rms = {profile_offpulse_rms}")

    offpulse_indices = sorted_prof_args[np.argwhere(mask > 0)]

    return offpulse_indices


def baseline_als(y, lam=1e5, p=0.1, niter=10):
    """
    Perform Asymmetric Lease Squares fitting to the profile to determine the
    baseline. The down side is that the performance and accuracy of the
    results very much depends on the choice of `p` and `lam` which makes this
    less ideal for pulsar signals.

    Based on this StackOverflow answer: https://stackoverflow.com/a/29185844

    :param y: the 1D data array from which a baseline is to be computed
    :type y: np.ndarray

    :param lam: smoothness parameter (typically between 10^2 and 10^9)
    :type lam: float

    :param p: a weight factor that determines how positive/negative peaks
    affect the baseline (for signals with positive peaks, p << 0.5)
    :type p: float

    :param niter: maximum number of iterations to attempt
    :type niter: int

    :return: the baseline values (same shape as y)
    :rtype: np.ndarray
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    z = np.mean(y)  # if there is no convergence, just return
    converged = False
    for i in range(niter):
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w * y)
        w_new = p * (y > z) + (1 - p) * (y < z)

        if np.allclose(w_new, w):
            # converged, can exit early
            logger.debug(f"ALS baseline estimation stopping after {i} iters")
            converged = True
            break
        else:
            # hasn't yet converged, update w and try again
            w = w_new
            converged = False

    if converged:
        logger.debug("ALS convergence achieved")
    else:
        logger.warning(
            "ALS baseline estimation did not converge! " "Try with larger niter?"
        )

    return z
