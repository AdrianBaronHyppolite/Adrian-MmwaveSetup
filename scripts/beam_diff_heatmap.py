#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import sys
from os.path import join
from argparse import ArgumentParser
from math import ceil, floor
from mpl_toolkits.axes_grid1 import ImageGrid

plt.rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 32})
plt.rc('text', usetex=True)


def parse_cli_args():

    parser = ArgumentParser()

    # Select the input directory
    parser.add_argument('-i',
                        '--input',
                        help='Input directory',
                        type=str,
                        required=True)
    # Select the input directory
    parser.add_argument('-a',
                        '--angle',
                        help='Angle range',
                        type=float,
                        default=10,
                        required=False)

    # Convert CLI arguments to dictionary
    args = vars(parser.parse_args())
    return args


def plot(args):

    # Calculate angles
    angles = [
        '{0:06.2f}'.format(x) for x in [360 - args["angle"], 0, args["angle"]]
    ]

    # Join paths
    paths = [
        join(args["input"], f"sel_kpi_angle_{angle}.log") for angle in angles
    ]

    # Load files
    files = [pd.read_csv(path) for path in paths]

    # Get min max
    vmin = 5 * floor(min(file.min(axis=0)['KPI'] for file in files) / 5)
    vmax = 5 * ceil(max(file.max(axis=0)['KPI'] for file in files) / 5)

    # Create figure and grid
    fig = plt.figure()
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 3),
        axes_pad=0.25,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.25,
    )

    # Make tick labels
    tick_labels = [
        '{0:+.1f}°'.format(y) for y in [-45 + 22.5 * x for x in range(0, 5)]
    ]
    tick_labels[2] = tick_labels[2][1:]

    titles = [
        f"-{args['angle']}° Angle", "Boresight", f"{args['angle']}° Angle"
    ]

    for index, ax in enumerate(grid):
        outer = []
        for tx in range(28, 37):
            inner = []
            for rx in range(28, 37):
                inner.append(files[index].loc[(files[index]['TX'] == tx) &
                                              (files[index]['RX'] == rx),
                                              'KPI'].mean())
            outer.append(inner[::-1])

        im = ax.imshow(outer,
                       extent=[28 - 0.5, 36 + 0.5, 28 - 0.5, 36 + 0.5],
                       origin='lower',
                       vmin=vmin,
                       vmax=vmax)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(range(28, 37, 2))
        ax.set_yticks(range(28, 37, 2))

        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel('RX Beam Direction')
        ax.set_ylabel('TX Beam Direction')
        ax.set_title(titles[index])

        print("max", max(max(outer)), "angle", -30 + 30 * index)
        print("min", min(min(outer)), "angle", -30 + 30 * index)

    # Create colorbar
    ax.cax.colorbar(im, label='RSS [dBm]')
    ax.cax.toggle_label(True)

    fig.subplots_adjust(top=1.0,
                        bottom=0.18,
                        left=0.11,
                        right=0.92,
                        hspace=0.2,
                        wspace=0.2)

    plt.show()


if __name__ == '__main__':
    # Cool pipeline
    args = parse_cli_args()
    # Plot results
    plot(args)
