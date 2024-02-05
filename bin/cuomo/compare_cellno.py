import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from ctmm import draw


def main():
    # read
    cell10 = np.load(snakemake.input.cell10, allow_pickle=True).item()
    cell5 = np.load(snakemake.input.cell5, allow_pickle=True).item()
    cell20 = np.load(snakemake.input.cell20, allow_pickle=True).item()

    # santiy check
    if np.any(cell10['gene'] != cell5['gene']) or np.any(cell10['gene'] != cell20['gene']):
        sys.exit('wrong')


    #
    data = pd.DataFrame({
        'minimum = 10': -1 * np.log10(cell10['reml']['wald']['free']['V']),
        'minimum = 5': -1 * np.log10(cell5['reml']['wald']['free']['V']),
        'minimum = 20': -1 * np.log10(cell20['reml']['wald']['free']['V']),
        })

    p_cut = -1 * np.log10(0.05 / data.shape[0])    

    # 
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=600)

    draw.scatter(data['minimum = 10'], data['minimum = 5'], ax=axes[0], xlab='> 10 cells',
                 ylab='> 5 cells', xyline=True, linregress=False, s=5, heatscatter=True)

    draw.scatter(data['minimum = 10'], data['minimum = 20'], ax=axes[1], xlab='> 10 cells',
                 ylab='> 20 cells', xyline=True, linregress=False, s=5, heatscatter=True)

    axes[0].axhline(y=p_cut, color='0.9', ls='--', zorder=10)
    axes[0].axvline(x=p_cut, color='0.9', ls='--', zorder=10)
    axes[1].axhline(y=p_cut, color='0.9', ls='--', zorder=10)
    axes[1].axvline(x=p_cut, color='0.9', ls='--', zorder=10)

    axes[0].text(0.05, .85, f'{data.loc[(data["minimum = 10"] < p_cut) & (data["minimum = 5"] > p_cut)].shape[0]}', 
                 fontsize=12, transform=axes[0].transAxes) 
    axes[0].text(0.75, .15, f'{data.loc[(data["minimum = 10"] > p_cut) & (data["minimum = 5"] < p_cut)].shape[0]}', 
                 fontsize=12, transform=axes[0].transAxes) 
    axes[0].text(0.75, .85, f'{data.loc[(data["minimum = 10"] > p_cut) & (data["minimum = 5"] > p_cut)].shape[0]}', 
                 fontsize=12, transform=axes[0].transAxes) 
    r, p = pearsonr(data['minimum = 10'], data['minimum = 5'])
    axes[0].text(0.98, 1.01, f'r = {r:.3f}', ha='right', va='bottom',
                 fontsize=12, transform=axes[0].transAxes) 
    axes[0].set_xlabel('> 10 cells', fontsize=16)
    axes[0].set_ylabel('> 5 cells', fontsize=16)
    

    axes[1].text(0.05, .85, f'{data.loc[(data["minimum = 10"] < p_cut) & (data["minimum = 20"] > p_cut)].shape[0]}', 
                 fontsize=12, transform=axes[1].transAxes) 
    axes[1].text(0.75, .15, f'{data.loc[(data["minimum = 10"] > p_cut) & (data["minimum = 20"] < p_cut)].shape[0]}', 
                 fontsize=12, transform=axes[1].transAxes) 
    axes[1].text(0.75, .85, f'{data.loc[(data["minimum = 10"] > p_cut) & (data["minimum = 20"] > p_cut)].shape[0]}', 
                 fontsize=12, transform=axes[1].transAxes) 
    r, p = pearsonr(data['minimum = 10'], data['minimum = 20'])
    axes[1].text(0.98, 1.01, f'r = {r:.3f}', ha='right', va='bottom',
                 fontsize=12, transform=axes[1].transAxes) 
    axes[1].set_xlabel('> 10 cells', fontsize=16)
    axes[1].set_ylabel('> 20 cells', fontsize=16)

    axes[0].text(-0.05, 1.05, '(A)', fontsize=16, transform=axes[0].transAxes)
    axes[1].text(-0.05, 1.05, '(B)', fontsize=16, transform=axes[1].transAxes)
    
    fig.tight_layout()
    fig.savefig(snakemake.output.png)    


if __name__ == '__main__':
    main()