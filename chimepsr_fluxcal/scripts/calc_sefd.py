#!/usr/bin/env python3

import sys
import argparse
from numpy import genfromtxt
from chimepsr_fluxcal.utils.sefd import SEFD

if __name__ == "__main__":
    usage = "%(prog)s ra dec [-M | [-t -d]] [--plot]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument(
        "ra",
        metavar="RA",
        type=str,
        help="target Right Ascension, J2000 (hh:mm:ss or decimal)",
    )
    parser.add_argument(
        "dec",
        metavar="DEC",
        type=str,
        help="target Declination, J2000 (dd:mm:ss or decimcal)",
    )
    parser.add_argument(
        "-t",
        "--time",
        metavar="TIME",
        nargs="+",
        type=float,
        help="start of observing epoch (MJD)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        metavar="DURATION",
        nargs="+",
        type=float,
        help="observation duration (seconds)",
    )
    parser.add_argument(
        "-M",
        "--metafile",
        metavar="METAFILE",
        type=str,
        help="a file containing MJDs and durations, one per line (cannot use with -t or -d)",
    )
    parser.add_argument("--plot", action="store_true", help="produce plots")

    args = parser.parse_args()
    if args.metafile:
        if args.time or args.duration:
            print("ERROR: Cannot use -M with -t or -d")
            parser.print_help()
            sys.exit()
        else:
            time, duration = genfromtxt(args.metafile, unpack=True, dtype=float)
    else:
        time = args.time
        duration = args.duration

    print(
        "{0:<20} {1:<9} {2:<12} {3} {4}".format(
            "MJD", "Tsys [K]", "Gain [K/Jy]", "SEFD [Jy]", "(@ 600 MHz)"
        )
    )
    for t, d in zip(time, duration):
        sefd = SEFD()
        sefd.set_pointing(args.ra, args.dec, mjd=t + (d / 2.0) / 86400.0)
        sefd.compute_sefd()
        if args.plot:
            sefd.plot_tsys()
            sefd.plot_gain()
            sefd.plot_sefd()
        print(
            "{0:<20} {1:<9g} {2:<12g} {3:g}".format(
                t, sefd.tsys[512], sefd.gain[512], sefd.sefd[512]
            )
        )
