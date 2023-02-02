import numpy as np, pandas as pd
import re, sys, time
from scipy import stats

def check_unique_cell( meta, counts ):
    # check unique cells
    if len(np.unique(meta['cell_name'])) != meta.shape[0]:
        sys.exit('Duplicate cells!\n')

    # check identical cells
    if not np.all(np.unique(meta['cell_name']) == np.unique(counts.index)):
        print(len(np.unique(meta['cell_name'])))
        print(len(np.unique(counts.index)))
        print(meta.loc[~meta['cell_name'].isin(counts.index), 'cell_name'])
        print(counts.loc[~counts.index.isin(meta['cell_name'])])
        sys.exit('Not matching cells!\n')

def main():
    # 
    input = snakemake.input
    output = snakemake.output

    # read
    meta = pd.read_table(input.meta, usecols=['donor', 'day', 'cell_name', 'experiment'])

    counts = pd.read_table(input.counts, index_col=0) # gene * cell
    counts = counts[ np.unique(meta['cell_name']) ] 
    counts = counts.transpose() # cell * gene
   
    # check missing value
    if np.any(pd.isna(counts)):
        print(counts.loc[np.any(pd.isna(counts), axis=1)])
        sys.exit('Missing values in gene expression!\n')

    check_unique_cell( meta, counts )

    # merge
    print( counts.shape[0])
    counts = counts.merge(meta, left_index=True, right_on='cell_name') #  cell * (gene, donor, day)
    if meta.shape[0] != counts.shape[0]:
        print(meta.shape[0], counts.shape[0])
        sys.exit('Missing cells?\n')

    # pseudobulk 
    counts_groupby_ct = counts.groupby(['donor', 'day'])
    ct_counts = counts_groupby_ct.aggregate(np.mean)
    ct_counts.to_csv(output.y, sep='\t')

    # nu
    ct_nu = counts_groupby_ct.aggregate(stats.sem)
    ct_nu = ct_nu**2
    ct_nu.to_csv(output.nu, sep='\t')

if __name__ == '__main__':
    main()
