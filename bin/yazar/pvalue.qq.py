import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from ctmm import draw


out9 = np.load(snakemake.input.out9, allow_pickle=True).item()
out5 = np.load(snakemake.input.out5, allow_pickle=True).item()

V9 = out9['he']['free']['V']
V9_p = (-1) * np.log10(out9['he']['wald']['free']['V'])
V5 = out5['he']['free']['V']
V5_p = (-1) * np.log10(out5['he']['wald']['free']['V'])

# find overlapping genes
genes, comm1, comm2 = np.intersect1d(out9['gene'], out5['gene'], return_indices=True)


fig, ax = plt.subplots(dpi=600)

draw.scatter(V9_p[comm1],  V5_p[comm2], ax=ax, xlab='$-log_{10}$' + f' p(variance differentiation among {V9.shape[1]} cell types)',
             ylab='$-log_{10}$' + f' p(variance differentiation among {V5.shape[1]} cell types)', s=8, xyline=True, linregress=False, heatscatter=True)

# add threshold
threshold = (-1) * np.log10(0.05/len(genes))
ax.axvline(threshold, color='0.8', ls='--', zorder=0)
ax.axhline(threshold, color='0.8', ls='--', zorder=0)

fig.savefig(snakemake.output.png)


# correlation
print(pearsonr(V9_p[comm1], V5_p[comm2]))

print(genes[(V9_p[comm1] < threshold) & (V5_p[comm2] > 30)])
print(V5_p[comm2][(V9_p[comm1] < threshold) & (V5_p[comm2] > 30)])