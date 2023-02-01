import re, sys, time
import numpy as np, pandas as pd
from scipy import stats

def main():
    #
    meta = pd.read_table(sys.argv[1], usecols=['donor', 'day', 'cell_name'])
    #inds = np.unique( meta['donor'] ) # for test
    #meta = meta.loc[meta['donor'].isin(inds[:5])] # for test

    counts = pd.read_table(sys.argv[2], index_col=0) # gene * cell
    counts = counts[ meta['cell_name'] ]
    counts = counts.transpose() # cell * gene
    genes = counts.columns

    # merge
    counts = counts.merge(meta, left_index=True, right_on='cell_name') #  cell * (gene, donor, day)
    #genes = list(genes)
    #genes.remove('ENSG00000001084_GCLC')
    #genes.remove('ENSG00000001167_NFYA')
    counts = counts[['donor','day']+list(genes)]
    print(time.time())

    # nu
    counts_groupby_ct = counts.groupby(['donor', 'day'])

    def my_bootstrap(x):
        if len(np.unique(x)) < 3:
            return 0
        else:
            return stats.bootstrap( (x,), lambda x, axis: stats.sem(x,axis=axis)**2 ).standard_error**2

    var_ct_nu = counts_groupby_ct.agg(lambda x: my_bootstrap(x) )
    var_ct_nu.to_csv(sys.argv[3], sep='\t')
    print(time.time())

if __name__ == '__main__':
    main()
