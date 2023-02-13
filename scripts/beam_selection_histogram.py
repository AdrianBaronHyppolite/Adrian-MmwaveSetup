#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import sys
import numpy as np

df = pd.read_csv(sys.argv[1])

tx = df['TX']
rx = df['RX']

# plot 2D histogram using pcolor
img = plt.figure()

plt.hist2d(rx, tx, bins=[9, 9],
           range=[[28,36], [28,36]],
           density=True,  vmin=0, vmax=1)

plt.xlabel('RX Beam Index')
plt.ylabel('TX Beam Index')

cbar = plt.colorbar()
cbar.ax.set_ylabel('Selection [%]')

cbar.formatter = matplotlib.ticker.PercentFormatter(xmax=1)
cbar.update_ticks()

plt.show()
