#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from math import ceil, floor
from os import listdir
from os.path import basename, isfile, join, normpath

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from cycler import cycler

plt.rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 54})
plt.rc('text', usetex=True)

directory = ""
packet_len = 200
heatmap = False
plots = [1, 2, 3, 4, 5]


def parse_cli_args():

    parser = ArgumentParser()

    # Let us choose which type of plot we want
    group_1 = parser.add_mutually_exclusive_group(required=False)
    group_1.add_argument('-l',
                         '--line',
                         help='Line Plot',
                         action='store_true',
                         default=True,
                         required=False)
    group_1.add_argument('-m',
                         '--heatmap',
                         help='Heatmap Plot',
                         action="store_true",
                         default=False,
                         required=False)
    # Let us choose which type of plot we want
    group_2 = parser.add_mutually_exclusive_group(required=False)
    group_2.add_argument('-c',
                         '--calculate',
                         help='Calculate Results',
                         action='store_true',
                         default=False,
                         required=False)
    group_2.add_argument('-s',
                         '--show',
                         help='Show Results',
                         action="store_true",
                         default=False,
                         required=False)

    # Select the input directory
    parser.add_argument('-i',
                        '--input',
                        help='Input directory',
                        type=str,
                        required=True)
    # Select the input directory
    parser.add_argument('-a',
                        '--packet',
                        help='Packet Length',
                        type=int,
                        default=1920,
                        required=False)
    parser.add_argument(
        '-p',
        '--plots',
        nargs='+',
        help="1: IA Accuracy 2: Throughput 3: RX Power 4: IA Power 5: IA Delay",
        required=False,
        default=['1', '2', '3', '4', '5', '6'])

    args = vars(parser.parse_args())

    global heatmap
    heatmap = args['heatmap']
    global directory
    directory = args['input']
    global packet_len
    packet_len = args['packet']
    global plots
    plots = [int(x) for x in args['plots']]

    # Convert CLI arguments to dictionary
    return args


def get_files(args):
    # Container to hold files
    files = {}

    prefixes = ("sel_pair", "rate_meas", "sel_kpi")

    # Initialize variables
    beam = meas = iadr = 0
    # Iterate over all files in the directory that start with a known prefix
    file_list = [
        f for f in listdir(args['input'])
        if isfile(join(args['input'], f)) and f.startswith(prefixes)
    ]

    # Iterate over log files
    for f in [x for x in file_list if x.startswith(prefixes[0])]:
        beam = f.split('_')[3]
        meas = f.split('_')[5]
        iadr = f.split('_')[7].split(".log")[0]

        # If the beam sweep period is not in the files
        if beam not in files:
            # Append it to the result dictionary
            files[beam] = {}
        # If the measurement period is not in the files
        if meas not in files[beam]:
            # Append it to the result dictionary
            files[beam][meas] = {}
        # If the measurement period is not in the files
        if iadr not in files[beam][meas]:
            # Append it to the result dictionary
            files[beam][meas][iadr] = {}

    results = pd.DataFrame(columns=[
        'beam', 'meas', 'iadr', 'entries', 'accuracy', 'elapsed',
        'elapsed_std', 'rss_reported', 'rss_reported_std', 'rss_decision',
        'rss_decision_std', 'throughput', 'throughput_std', 'overhead',
        'overhead_std', 'rate', 'rate_std', 'update'
    ])

    # Getting the total size of the nested dictionary
    o_f_key = list(files.keys())[0]
    i_f_key = list(files[o_f_key].keys())[0]
    r_len = len(files) * len(files[o_f_key]) * len(files[o_f_key][i_f_key])
    # Count size and number of chars
    len_str = len(str(r_len))

    # Iterate over log files
    counter = 0
    for beam in sorted(files):
        for meas in sorted(files[beam]):
            for iadr in sorted(files[beam][meas]):
                # Increment counter
                counter += 1
                # Print progress
                print(f"[{str(counter).zfill(len_str)}/{r_len}] Loading Files",
                      end="\r")

                sel_csv = pd.read_csv(join(
                    args['input'],
                    f"sel_pair_bp_{beam}_mp_{meas}_id_{iadr}.log"),
                                      sep=",")

                rate_csv = pd.read_csv(join(
                    args['input'],
                    f"rate_meas_bp_{beam}_mp_{meas}_id_{iadr}.log"),
                                       sep=",")

                kpi_csv = pd.read_csv(join(
                    args['input'],
                    f"sel_kpi_bp_{beam}_mp_{meas}_id_{iadr}.log"),
                                      sep=",")

                correct = sel_csv[(sel_csv["TX"] == 32)
                                  & (sel_csv["RX"] == 32)]
                rate_csv[
                    "rate"] = rate_csv["throughput"] + rate_csv["overhead"]
                power = kpi_csv["KPI"].replace(0, np.NaN)

                global packet_len

                results.loc[len(results)] = {
                    "beam":
                    float(beam),
                    "meas":
                    float(meas),
                    "iadr":
                    float(iadr),
                    "entries":
                    len(sel_csv.index),
                    "accuracy":
                    0 if not len(sel_csv.index) else len(correct) /
                    len(sel_csv.index),
                    "elapsed":
                    sel_csv['elapsed'].mean() * 1000,
                    "elapsed_std":
                    sel_csv['elapsed'].std() * 1000,
                    "rss_reported":
                    power.mean(),
                    "rss_reported_std":
                    power.std(),
                    "rss_decision":
                    correct["KPI"].mean(),
                    "rss_decision_std":
                    correct["KPI"].std(),
                    "throughput":
                    rate_csv["throughput"].mean() / packet_len,
                    "throughput_std":
                    rate_csv["throughput"].std() / packet_len,
                    "overhead":
                    rate_csv["overhead"].mean() / packet_len,
                    "overhead_std":
                    rate_csv["overhead"].std() / packet_len,
                    "rate":
                    rate_csv["rate"].mean() / packet_len,
                    "rate_std":
                    rate_csv["rate"].std() / packet_len,
                    "update":
                    9 * 9 * float(beam) + float(iadr) +
                    sel_csv['elapsed'].mean() * 1000
                }

    global directory
    # Create file name based on the input directory
    file_name = basename(normpath(args['input'])) + "_results.csv"
    # Save parsed results into an intermediate CSV
    results.to_csv(file_name, index=False, sep=',')


def get_results(args):
    # Get the right file name depending on how this was invoked
    file_name = args['input'] if args['show'] else \
        basename(normpath(args['input'])) + "_results.csv"

    # Check whether the result file exists
    if not isfile(file_name):
        raise FileNotFoundError("Could not find " + str(file_name))

    # Load file
    results = pd.read_csv(file_name, sep=',')

    return results


def plot_results(results):

    global plots
    global heatmap

    xlim_min = results.min(axis=0)['beam']
    xlim_max = results.max(axis=0)['beam']
    if xlim_min == 0.01:
        xlim_min = 0

    #  stop = None
    stop = 20

    skip = 2
    #  results = results[(results["beam"] >= 1)]

    beams = sorted(results.beam.unique())
    iadrs = sorted(results.iadr.unique())
    meass = sorted(results.meas.unique())

    print("beams", beams)
    print("iadrs", iadrs)
    print("meass", meass)

    new_colors = [
        plt.get_cmap('viridis')(1. * i / len(meass)) for i in range(len(meass))
    ]

    plt.rc('axes', prop_cycle=(cycler('color', new_colors)))
    linewidth = 8

    if 1 in plots:

        if stop is not None:
            size = 1
            fig1, *ax1 = plt.subplots(1, size)
        else:
            size = len(iadrs)
            fig1, ax1 = plt.subplots(1, size)

        im1 = [None] * size

        # Get min max
        vmin = floor(results.min(axis=0)['accuracy'])
        vmax = ceil(results.max(axis=0)['accuracy'])

        if heatmap:
            for index, iadr in enumerate(iadrs):
                outer = []
                for meas in meass:
                    inner = []
                    for beam in beams:
                        inner.append(
                            results.loc[(results['meas'] == meas)
                                        & (results['beam'] == beam) &
                                        (results['iadr']
                                         == iadr)]['accuracy'].values[0])
                    outer.append(inner)

                im1[index] = ax1[index].imshow(outer,
                                               origin='lower',
                                               vmin=vmin,
                                               vmax=vmax)

                ax1[index].set_xticks(range(len(beams)))
                ax1[index].set_yticks(range(len(meass)))

                ax1[index].set_xticklabels(beams)
                ax1[index].set_yticklabels(meass)

                ax1[index].set_xlabel("Beam Duration [ms]")
                ax1[index].set_ylabel("Measurement Cadence [ms]")
                ax1[index].set_title(f"Payload Duration {iadr} [ms]")

            # Create colorbar
            fig1.colorbar(im1[0],
                          ax=ax1.ravel().tolist(),
                          label='Accuracy [\%]',
                          fraction=0.046,
                          pad=0.04)

        else:
            for index, iadr in enumerate(iadrs):
                # If we have a trigger to stop
                if stop is not None:
                    # Skip until we get there
                    if stop != index:
                        continue
                    # Fix index
                    else:
                        index = 0

                for meas in meass:
                    numbers = results[(results['meas'] == meas)
                                      & (results['iadr'] == iadr)].sort_values(
                                          by=['beam'])[["beam", "accuracy"]]

                    ax1[index].plot(numbers["beam"],
                                    numbers["accuracy"],
                                    linestyle="-.",
                                    alpha=0.7,
                                    linewidth=linewidth,
                                    label=f"Meas. Dur. {meas} [ms]")

                    ax1[index].set_ylabel("IA Accucary [\%]")
                    ax1[index].set_ylim([0, 1.005])
                    ax1[index].yaxis.set_major_formatter(
                        ticker.PercentFormatter(xmax=1))

                    ax1[index].set_xlim([xlim_min, xlim_max])
                    #  ax1[index].set_xscale('log')
                    #  ax1[index].set_xticks([1, 10, 100, 1000])
                    ax1[index].get_xaxis().set_major_formatter(
                        ticker.ScalarFormatter())
                    ax1[index].set_title(f"Payload Duration {iadr} [ms]")
                    #  ax1[index].legend()
                    ax1[index].set_xlabel("Beam Duration [ms]")

            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'),
                                       norm=plt.Normalize(vmin=min(meass),
                                                          vmax=max(meass)))
            cb = plt.colorbar(sm,
                              ticks=meass[::skip],
                              label="Measurement Cadence [ms]")
            cb.ax.set_yticklabels([int(x) for x in meass[::skip]])

            fig1.subplots_adjust(top=0.925,
                                 bottom=0.13,
                                 left=0.16,
                                 right=1.0,
                                 hspace=0.2,
                                 wspace=0.2)

    if 2 in plots:

        if stop is not None:
            size = 1
            fig2, *ax2 = plt.subplots(1, size)
        else:
            size = len(iadrs)
            fig2, ax2 = plt.subplots(1, size)

        im2 = [None] * size

        # Get min max
        vmin = floor(results.min(axis=0)['throughput'])
        vmax = ceil(results.max(axis=0)['throughput'])

        if heatmap:
            for index, meas in enumerate(meass):
                outer = []
                for iadr in iadrs:
                    inner = []
                    for beam in beams:
                        inner.append(results.loc[(results['meas'] == meas) &
                                                 (results['beam'] == beam) &
                                                 (results['iadr'] == iadr),
                                                 'throughput'])
                    outer.append(inner)

                im2[index] = ax2[index].imshow(outer,
                                               origin='lower',
                                               vmin=vmin,
                                               vmax=vmax)

                ax2[index].set_xticks(range(len(beams)))
                ax2[index].set_yticks(range(len(iadrs)))

                ax2[index].set_xticklabels(beams)
                ax2[index].set_yticklabels(iadrs)

                ax2[index].set_xlabel("Beam Duration [ms]")
                ax2[index].set_title(f"Measurement Cadence {meas} [ms]")
                ax2[index].set_ylabel(f"Payload Duration {iadr} [ms]")

            fig2.colorbar(im2[0],
                          ax=ax2.ravel().tolist(),
                          label='Packet Rate [pps]',
                          fraction=0.046,
                          pad=0.041)

        else:
            for index, iadr in enumerate(iadrs):
                # If we have a trigger to stop
                if stop is not None:
                    # Skip until we get there
                    if stop != index:
                        continue
                    # Fix index
                    else:
                        index = 0

                for meas in meass:
                    numbers = results[(results['meas'] == meas)
                                      & (results['iadr'] == iadr)].sort_values(
                                          by=['beam'])[[
                                              "beam", "throughput",
                                              "throughput_std"
                                          ]]

                    ax2[index].plot(  # errorbar(
                        numbers["beam"],
                        numbers["throughput"],
                        #  numbers["throughput_std"],
                        linestyle="-.",
                        alpha=0.7,
                        linewidth=linewidth,
                        label=f"Meas. Dur. {meas} [ms]")

                ax2[index].set_ylim([0, 16])
                ax2[index].set_xlim([xlim_min, xlim_max])
                #  ax2[index].set_xscale('log')
                #  ax2[index].set_xticks([1, 10, 100, 1000])
                ax2[index].get_xaxis().set_major_formatter(
                    ticker.ScalarFormatter())

                ax2[index].set_yticks([0, 4, 8, 12, 16])
                ax2[index].set_ylabel("Packet Rate [pps]")
                ax2[index].set_title(f"Payload Duration {iadr} [ms]")
                #  ax2[index].legend()
                ax2[index].set_xlabel("Beam Duration [ms]")

            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'),
                                       norm=plt.Normalize(vmin=min(meass),
                                                          vmax=max(meass)))
            cb = plt.colorbar(sm,
                              ticks=meass[::skip],
                              label="Measurement Cadence [ms]")
            cb.ax.set_yticklabels([int(x) for x in meass[::skip]])

            fig2.subplots_adjust(top=0.925,
                                 bottom=0.13,
                                 left=0.10,
                                 right=1.0,
                                 hspace=0.2,
                                 wspace=0.2)

    if 3 in plots:

        if stop is not None:
            size = 1
            fig3, *ax3 = plt.subplots(1, size)
        else:
            size = len(iadrs)
            fig3, ax3 = plt.subplots(1, size)

        im3 = [None] * size

        # Get min max
        vmin = floor(results.min(axis=0)['rss_reported'])
        vmax = ceil(results.max(axis=0)['rss_reported'])

        if heatmap:
            for index, iadr in enumerate(iadrs):
                outer = []
                for meas in meass:
                    inner = []
                    for beam in beams:
                        inner.append(results.loc[(results['meas'] == meas) &
                                                 (results['beam'] == beam) &
                                                 (results['iadr'] == iadr),
                                                 'rss_reported'])
                    outer.append(inner)

                im3[index] = ax3[index].imshow(outer,
                                               origin='lower',
                                               vmin=vmin,
                                               vmax=vmax)

                ax3[index].set_xticks(range(len(beams)))
                ax3[index].set_yticks(range(len(meass)))

                ax3[index].set_xticklabels(beams)
                ax3[index].set_yticklabels(meass)

                ax3[index].set_xlabel("Beam Duration [ms]")
                ax3[index].set_ylabel(f"Measurement Cadence [ms]")
                ax3[index].set_title(f"Payload Duration {iadr} [ms]")

            # Create colorbar
            fig3.colorbar(im3[1],
                          ax=ax3.ravel().tolist(),
                          label='Received RSS [dB]',
                          fraction=0.046,
                          pad=0.04)

        else:
            for index, iadr in enumerate(iadrs):
                # If we have a trigger to stop
                if stop is not None:
                    # Skip until we get there
                    if stop != index:
                        continue
                    # Fix index
                    else:
                        index = 0

                for meas in meass:
                    numbers = results[(results['meas'] == meas)
                                      & (results['iadr'] == iadr)].sort_values(
                                          by=['beam'])[[
                                              "beam", "rss_reported"
                                          ]]

                    ax3[index].plot(numbers["beam"],
                                    numbers["rss_reported"],
                                    label=f"Meas. Dur. {meas} [ms]",
                                    linewidth=linewidth)

                ax3[index].legend()
                ax3[index].set_ylim([vmin, vmax])
                ax3[index].set_title(f"Payload Duration {iadr} [ms]")
                ax3[index].set_xlim([xlim_min, xlim_max])
                #  ax3[index].set_xscale('log')
                #  ax3[index].set_xticks([1, 10, 100, 1000])
                ax3[index].get_xaxis().set_major_formatter(
                    ticker.ScalarFormatter())

                ax3[index].set_xlabel("Beam Duration [ms]")
                ax3[index].set_ylabel("Received RSS [dB]")

            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'),
                                       norm=plt.Normalize(vmin=min(meass),
                                                          vmax=max(meass)))
            cb = plt.colorbar(sm,
                              ticks=meass[::skip],
                              label="Measurement Cadence [ms]")
            cb.ax.set_yticklabels([int(x) for x in meass[::skip]])

            fig3.subplots_adjust(top=0.925,
                                 bottom=0.13,
                                 left=0.16,
                                 right=1.0,
                                 hspace=0.2,
                                 wspace=0.2)

    if 4 in plots:

        if stop is not None:
            size = 1
            fig4, *ax4 = plt.subplots(1, size)
        else:
            size = len(iadrs)
            fig4, ax4 = plt.subplots(1, size)

        im4 = [None] * size

        # Get min max
        vmin = floor(results.min(axis=0)['rss_decision'])
        vmax = ceil(results.max(axis=0)['rss_decision'])

        if heatmap:
            for index, iadr in enumerate(iadrs):
                outer = []
                for meas in meass:
                    inner = []
                    for beam in beams:
                        inner.append(results.loc[(results['meas'] == meas) &
                                                 (results['beam'] == beam) &
                                                 (results['iadr'] == iadr),
                                                 'rss_decision'])
                    outer.append(inner)

                im4[index] = ax4[index].imshow(outer,
                                               origin='lower',
                                               vmin=vmin,
                                               vmax=vmax)

                ax4[index].set_xticks(range(len(beams)))
                ax4[index].set_yticks(range(len(meass)))

                ax4[index].set_xticklabels(beams)
                ax4[index].set_yticklabels(meass)

                ax4[index].set_xlabel("Beam Duration [ms]")
                ax4[index].set_ylabel(f"Measurement Cadence [ms]")
                ax4[index].set_title(f"Payload Duration {iadr} [ms]")

            # Create colorbar
            fig4.colorbar(im4[1],
                          ax=ax4.ravel().tolist(),
                          label='Decision RSS [dB]',
                          fraction=0.046,
                          pad=0.04)

        else:
            for index, iadr in enumerate(iadrs):
                # If we have a trigger to stop
                if stop is not None:
                    # Skip until we get there
                    if stop != index:
                        continue
                    # Fix index
                    else:
                        index = 0

                for meas in meass:
                    numbers = results[(results['meas'] == meas)
                                      & (results['iadr'] == iadr)].sort_values(
                                          by=['beam'])[[
                                              "beam", "rss_decision"
                                          ]]
                    ax4[index].plot(numbers["beam"],
                                    numbers["rss_decision"],
                                    label=f"Meas. Dur. {meas} [ms]")

                    ax4[index].scatter(numbers["beam"],
                                       numbers["rss_decision"])

                ax4[index].legend()
                ax4[index].set_ylim([vmin, vmax])
                ax4[index].set_title(f"Payload Duration {iadr} [ms]")
                ax4[index].set_xlim([xlim_min, xlim_max])
                #  ax4[index].set_xscale('log')
                #  ax4[index].set_xticks([1, 10, 100, 1000])
                #  ax4[index].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                ax4[index].set_xlabel("Beam Duration [ms]")
                ax4[index].set_ylabel("Decision RSS [dB]")

            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'),
                                       norm=plt.Normalize(vmin=min(meass),
                                                          vmax=max(meass)))
            cb = plt.colorbar(sm,
                              ticks=meass[::skip],
                              label="Measurement Cadence [ms]")
            cb.ax.set_yticklabels([int(x) for x in meass[::skip]])

            fig4.subplots_adjust(top=0.925,
                                 bottom=0.13,
                                 left=0.16,
                                 right=1.0,
                                 hspace=0.2,
                                 wspace=0.2)

    if 5 in plots:

        if stop is not None:
            size = 1
            fig5, *ax5 = plt.subplots(1, size)
        else:
            size = len(iadrs)
            fig5, ax5 = plt.subplots(1, size)

        im5 = [None] * size

        # Get min max
        vmin = floor(results.min(axis=0)['elapsed'])
        vmax = ceil(results.max(axis=0)['elapsed'])

        if heatmap:
            for index, iadr in enumerate(iadrs):
                outer = []
                for meas in meass:
                    inner = []
                    for beam in beams:
                        inner.append(results.loc[(results['meas'] == meas) &
                                                 (results['beam'] == beam) &
                                                 (results['iadr'] == iadr),
                                                 'elapsed'])
                    outer.append(inner)

                im5[index] = ax5[index].imshow(outer,
                                               origin='lower',
                                               vmin=vmin,
                                               vmax=vmax)

                ax5[index].set_xticks(range(len(beams)))
                ax5[index].set_yticks(range(len(meass)))

                ax5[index].set_xticklabels(beams)
                ax5[index].set_yticklabels(meass)

                ax5[index].set_xlabel("Beam Duration [ms]")
                ax5[index].set_ylabel(f"Measurement Cadence [ms]")
                ax5[index].set_title(f"Payload Duration {iadr} [ms]")

            # Create colorbar
            fig5.colorbar(im5[1],
                          ax=ax5.ravel().tolist(),
                          label='Decision Delay [ms]',
                          fraction=0.046,
                          pad=0.04)

        else:
            for index, iadr in enumerate(iadrs):
                # If we have a trigger to stop
                if stop is not None:
                    # Skip until we get there
                    if stop != index:
                        continue
                    # Fix index
                    else:
                        index = 0

                for meas in meass:
                    numbers = results[(results['meas'] == meas)
                                      & (results['iadr'] == iadr)].sort_values(
                                          by=['beam'])[[
                                              "beam", "elapsed", "elapsed_std"
                                          ]]
                    ax5[index].plot(  # errorbar(
                        numbers["beam"],
                        numbers["elapsed"],
                        #  numbers["elapsed_std"],
                        linestyle="-.",
                        alpha=0.7,
                        linewidth=linewidth,
                        label=f"Meas. Dur. {meas} [ms]")

                #  ax5[index].legend()
                ax5[index].set_ylim([0, 10])
                ax5[index].set_title(f"Payload Duration {iadr} [ms]")
                ax5[index].set_xlim([xlim_min, xlim_max])
                #  ax5[index].set_xscale('log')
                #  ax5[index].set_xticks([1, 10, 100, 1000])
                #  ax5[index].get_xaxis().set_major_formatter(ticker.ScalarFormatter())

                ax5[index].set_xlabel("Beam Duration [ms]")
                ax5[index].set_ylabel("Decision Delay [ms]")

            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'),
                                       norm=plt.Normalize(vmin=min(meass),
                                                          vmax=max(meass)))
            cb = plt.colorbar(sm,
                              ticks=meass[::skip],
                              label="Measurement Cadence [ms]")
            cb.ax.set_yticklabels([int(x) for x in meass[::skip]])

            fig5.subplots_adjust(top=0.925,
                                 bottom=0.13,
                                 left=0.10,
                                 right=1.0,
                                 hspace=0.2,
                                 wspace=0.2)

    if 6 in plots:

        stop = 0
        if stop is not None:
            size = 1
            fig6, *ax6 = plt.subplots(1, size)
        else:
            size = len(iadrs)
            fig6, ax6 = plt.subplots(1, size)

        im6 = [None] * size

        # Get min max
        vmin = floor(results.min(axis=0)['update'])
        vmax = ceil(results.max(axis=0)['update'])

        if heatmap:
            for index, meas in enumerate(meass):
                # Skip until we get there
                if stop != index:
                    continue
                # Fix index
                else:
                    index = 0

                outer = []
                for iadr in iadrs:
                    inner = []
                    for beam in beams:
                        inner.append(results.loc[(results['meas'] == meas) &
                                                 (results['beam'] == beam) &
                                                 (results['iadr'] == iadr),
                                                 'update'])
                    outer.append(inner)

                im6[index] = ax6[index].imshow(outer,
                                               origin='lower',
                                               vmin=vmin,
                                               vmax=vmax)

                ax6[index].set_xticks(range(len(beams)))
                ax6[index].set_yticks(range(len(meass)))

                ax6[index].set_xticklabels(beams)
                ax6[index].set_yticklabels(meass)

                ax6[index].set_xlabel("Beam Duration [ms]")
                ax6[index].set_ylabel("Update Delay [ms]")
                ax6[index].set_title(f"Measurement Cadence {meas} [ms]")

            # Create colorbar
            fig6.colorbar(im6[0],
                          ax=ax6,
                          label=' IA Decision [ms]',
                          fraction=0.046,
                          pad=0.04)

        else:
            for index, meas in enumerate(meass):
                # If we have a trigger to stop
                if stop is not None:
                    # Skip until we get there
                    if stop != index:
                        continue
                    # Fix index
                    else:
                        index = 0

                for iadr in iadrs:
                    numbers = results[(results['meas'] == meas)
                                      & (results['iadr'] == iadr)].sort_values(
                                          by=['beam'])[[
                                              "beam", "update", "iadr"
                                          ]]

                    ax6[index].loglog(  # errorbar(
                        numbers["beam"],
                        numbers["update"],
                        linestyle="-.",
                        alpha=0.7,
                        linewidth=linewidth,
                        label=f"Meas. Dur. {meas} [ms]")

                ax6[index].set_xlim([0.01, 10])
                ax6[index].set_ylim([3.8, 1100])
                ax6[index].set_title(f"Payload Duration {iadr} [ms]")
                ax6[index].set_xscale('log')
                ax6[index].set_yscale('log')

                ax6[index].tick_params(axis='x', which='major', pad=15)

                ax6[index].set_xlabel("Beam Duration [ms]")
                ax6[index].set_ylabel("Update Delay [ms]")
                ax6[index].set_title(f"Measurement Cadence {meas} [ms]")

            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'),
                                       norm=plt.Normalize(vmin=min(meass),
                                                          vmax=max(meass)))
            cb = plt.colorbar(sm,
                              ticks=iadrs[::skip],
                              label="Payload Duration [ms]")
            cb.ax.set_yticklabels([int(x) for x in iadrs[::skip]])

            fig6.subplots_adjust(top=0.925,
                                 bottom=0.14,
                                 left=0.12,
                                 right=1.0,
                                 hspace=0.2,
                                 wspace=0.2)

    plt.show()


if __name__ == '__main__':
    # Cool pipeline
    args = parse_cli_args()

    if not args["show"]:
        # Get files
        files = get_files(args)

    if not args["calculate"]:
        results = get_results(args)

        # Plot results
        plot_results(results)
