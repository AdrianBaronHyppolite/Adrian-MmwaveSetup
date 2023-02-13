#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import pandas as pd
from os import listdir
from os.path import isfile, join, basename, normpath

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.ticker as ticker
from cycler import cycler

import numpy as np

plt.rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 36})
plt.rc('text', usetex=True)


directory = ""
packet_len = 200
heatmap = False
plots = [1,2,3,4,5]

def parse_cli_args():

    parser = ArgumentParser()

    # Let us choose which type of plot we want
    group_2 = parser.add_mutually_exclusive_group(required=False)
    group_2 .add_argument(
        '-c', '--calculate',
        help='Calculate Results',
        action='store_true',
        default=False,
        required=False
    )
    group_2 .add_argument(
         '-s', '--show',
         help='Show Results',
         action="store_true",
         default=False,
         required=False
    )

    # Select the input directory
    parser.add_argument(
        '-i', '--input',
        help='Input directory',
        type=str,
        required=True
    )
    # Select the input directory
    parser.add_argument(
        '-a', '--packet',
        help='Packet Length',
        type=int,
        default=1920,
        required=False
    )
    parser.add_argument(
        '-p','--plots',
        nargs='+',
        help="1: IA Accuracy 2: Throughput 3: RX Power 4: IA Power 5: IA Delay",
        required=False,
        default=['1','2','3','4','5','6']
    )

    args = vars(parser.parse_args())

    global directory;  directory = args['input']
    global packet_len;  packet_len = args['packet']
    global plots; plots = [int(x) for x in args['plots']]

    # Convert CLI arguments to dictionary
    return args


def get_files(args):
    # Container to hold files
    files = {}

    prefixes = ("sel_pair", "rate_meas", "sel_kpi")

    # Iterate over all files in the directory that start with a known prefix
    file_list = [ f for f in listdir(args['input']) if \
                 isfile(join(args['input'], f)) and f.startswith(prefixes)]


    # Iterate over log files
    for f in [x for x in file_list if x.startswith(prefixes[0])]:
        angle = f.split('_')[3].split(".log")[0]
        # If the beam sweep period is not in the files
        if angle not in files:
            # Append it to the result dictionary
            files[angle] = {}

    results = pd.DataFrame(
        columns=['angle', 'rad', 'entries', 'accuracy',
                 'elapsed', 'elapsed_std',
                 'rss_boresight', 'rss_boresight_std',
                 'rss_sweep', 'rss_sweep_std',
                 'rss_beam', 'rss_beam_std',
                 'throughput', 'throughput_std',
                 'overhead', 'overhead_std',
                 'rate', 'rate_std']
    )

    # Getting the total size of the nested dictionary
    r_len = len(files)
    # Count size and number of chars
    len_str = len(str(r_len))

    # Iterate over log files
    counter = 0
    for angle in sorted(files):
        # Increment counter
        counter += 1
        # Print progress
        print(f"[{str(counter).zfill(len_str)}/{r_len}] Loading Files", end="\r")

        sel_csv  = pd.read_csv(join(
            args['input'],
            f"sel_pair_angle_{angle}.log"
        ), sep=",")

        rate_csv = pd.read_csv(join(
            args['input'],
            f"rate_meas_angle_{angle}.log"
        ), sep=",")

        kpi_csv = pd.read_csv(join(
            args['input'],
            f"sel_kpi_angle_{angle}.log"
        ), sep=",")


        rate_csv["rate"] =  rate_csv["throughput"] + rate_csv["overhead"]

        static = kpi_csv[(kpi_csv["TX"]==32) & (kpi_csv["RX"]==32)]["KPI"]
        kpi_csv["KPI"] = kpi_csv["KPI"].replace(0, np.NaN)

        angle_to_beam = {
            "315.00": 28,
            "326.25": 29,
            "337.50": 30,
            "348.75": 31,
            "360.00": 32,
            "000.00": 32,
            "011.25": 33,
            "022.50": 34,
            "033.75": 35,
            "045.00": 36,
        }

        if angle in angle_to_beam:
            beam = angle_to_beam[angle]

        elif float(angle) <= 180:
            beam = 36
        elif float(angle) > 180:
            beam = 28

        adjusted = kpi_csv[(kpi_csv["TX"]==32) & (kpi_csv["RX"]==beam)]['KPI']
        rss_beam = adjusted.mean()
        rss_beam_std = adjusted.std()

        global packet_len

        results.loc[len(results)] = {
            "angle": float(angle),
            "rad":  float(angle)*np.pi/180,
            "entries": len(sel_csv.index),
            "accuracy": 0 if not len(sel_csv.index) else len(static)/len(sel_csv.index),
            "elapsed": sel_csv['elapsed'].mean() * 1000,
            "elapsed_std": sel_csv['elapsed'].std() * 1000,
            "rss_boresight": static.mean(),
            "rss_boresight_std": static.std(),
            "rss_sweep": sel_csv["KPI"].mean(),
            "rss_sweep_std": sel_csv["KPI"].std(),
            "rss_beam": rss_beam,
            "rss_beam_std": rss_beam_std,
            "throughput": rate_csv["throughput"].mean()/packet_len,
            "throughput_std": rate_csv["throughput"].std()/packet_len,
            "overhead": rate_csv["overhead"].mean()/packet_len,
            "overhead_std": rate_csv["overhead"].std()/packet_len,
            "rate": rate_csv["rate"].mean()/packet_len,
            "rate_std": rate_csv["rate"].std()/packet_len
        }

    global directory
    # Create file name based on the input directory
    file_name =  basename(normpath(args['input'])) + "_results.csv"
    # Save parsed results into an intermediate CSV
    results.to_csv(file_name, index=False, sep=',')


def plot_results(args):
    # Get the right file name depending on how this was invoked
    file_name =  args['input'] if args['show'] else \
        basename(normpath(args['input'])) + "_results.csv"

    # Check whether the result file exists
    if not isfile(file_name):
        raise FileNotFoundError("Could not find " + str(file_name))

    # Load file
    results = pd.read_csv(file_name, sep=',')

    global plots

    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.set_theta_zero_location("N")
    angles = sorted(results.angle.unique())

    #  print(results)
    #  print("angles", angles)

    new_colors = [plt.get_cmap('viridis')(1. * i/10) for i in range(10)]

    plt.rc('axes', prop_cycle=(cycler('color', new_colors)))
    linewidth = 3
    markersize = 12

    if 1 in plots:

        # Get min max
        vmin = results.min(axis=0)['rss_boresight']
        vmax = results.max(axis=0)['rss_boresight']

        numbers = results.sort_values(by=['rad'])[
            ["rad",
             "rss_boresight", "rss_boresight_std",
             "rss_sweep", "rss_sweep_std",
             "rss_beam", "rss_beam_std"]
        ]

        ax1.plot( #errorbar(
            numbers["rad"],
            numbers["rss_boresight"],
            #  yerr=numbers["rss_boresight_std"],
            label="Static Boresight Beam",
            linestyle="-.",  alpha=0.9,
            linewidth=linewidth , markersize=markersize, marker='o',
            color=new_colors[0]
        )

        ax1.plot( #errorbar(
            numbers["rad"],
            numbers["rss_beam"],
            #  yerr=numbers["rss_beam_std"],
            label="Manual Beam Selection",
            linestyle="-.",  alpha=0.9,
            linewidth=linewidth, markersize=markersize+2, marker='*',
            color=new_colors[8]
        )

        ax1.plot(
            numbers["rad"],
            numbers["rss_sweep"],
            #  yerr=numbers["rss_sweep_std"],
            label="Dynamic Beam Sweep",
            linestyle="-.",  alpha=0.9,
            linewidth=linewidth, markersize=markersize, marker='^',
            color=new_colors[5]
        )

        print("Sweep over Static Gain",
              min(numbers["rss_sweep"] - numbers["rss_boresight"]),
              max(numbers["rss_sweep"] - numbers["rss_boresight"])
        )
        print(
            "Sweep over Manual Gain",
            min(numbers["rss_sweep"] - numbers["rss_beam"]),
            max(numbers["rss_sweep"] - numbers["rss_beam"])
        )
        print("Manual over Static Gain",
              min(numbers["rss_beam"] - numbers["rss_boresight"]),
              max(numbers["rss_beam"] - numbers["rss_boresight"])
        )


        ax1.legend()
        ax1.set_ylim(-60,0)
        ax1.set_title(f"Received Signal Strength [dB]")

    plt.show()

if __name__ == '__main__':
    # Cool pipeline
    args = parse_cli_args()

    if not args["show"]:
        # Get files
        files = get_files(args)

    if not args["calculate"]:
        # Plot results
        plot_results(args)
