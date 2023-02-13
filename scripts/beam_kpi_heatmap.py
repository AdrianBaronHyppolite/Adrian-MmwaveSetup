#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import sys


df = pd.read_csv(sys.argv[1])

outer = []
for tx in range(28, 37):
    inner = []
    for rx in range(28, 37):
        inner.append(
            df.loc[(df['TX']  == tx) & (df['RX'] == rx), 'KPI'].mean()
        )
    outer.append(inner)

fig, ax = plt.subplots()
im = ax.imshow(outer, extent=[28-0.5, 36+0.5, 28-0.5, 36+0.5], origin='lower')

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, label='RSS [dB]')

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(28, 37))
ax.set_yticks(range(28, 37))

plt.xlabel('RX Beam Index [#]')
plt.ylabel('TX Beam Index [#]')

fig.tight_layout()

plt.show()
