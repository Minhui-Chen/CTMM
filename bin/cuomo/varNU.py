import re, sys
import numpy as np, pandas as pd
from scipy import stats

def main():
    #
    meta = pd.read_table(sys.argv[1], usecols=['donor', 'day', 'cell_name'])

    counts = pd.read_table(sys.argv[2], index_col=0) # gene * cell
    counts = counts[ meta['cell_name'] ]
    counts = counts.transpose() # cell * gene
    genes = counts.columns

    # merge
    counts = counts.merge(meta, left_index=True, right_on='cell_name') #  cell * (gene, donor, day)
    counts = counts[['donor','day']+list(genes)]

    # nu
    counts_groupby_ct = counts.groupby(['donor', 'day'])

    def my_bootstrap(x):
        if len(np.unique(x)) < 3:
            return 0
        else:
            return stats.bootstrap( (x,), lambda x, axis: stats.sem(x,axis=axis)**2 ).standard_error**2

    var_ct_nu = counts_groupby_ct.agg(lambda x: my_bootstrap(x) )
    var_ct_nu.to_csv(sys.argv[3], sep='\t')

if __name__ == '__main__':
    main()
