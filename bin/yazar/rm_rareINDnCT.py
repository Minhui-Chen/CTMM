import time
import numpy as np, pandas as pd

def count_expressed(x, threshold=0.1):
    x = x.unstack()
    prop = (x>0).sum() / x.count()
    if np.all(prop > threshold):
        return True
    else:
        return False

def main():
    # read
    ctp = pd.read_table(snakemake.input.ctp, index_col=(0,1)).astype('float32')
    ctnu = pd.read_table(snakemake.input.ctnu, index_col=(0,1)).astype('float32')
    n = pd.read_table(snakemake.input.n, index_col=0).astype('int')

    # 
    n_inds = n.shape[0]
    cts = n.columns.to_numpy()
    
    # find common cts with less than 50% missing inds
    common = n.median()[n.median() > int(snakemake.wildcards.ct_min_cellnum)].index.tolist()

    # filter common cts
    ctp = ctp.loc[ctp.index.get_level_values('ct').isin(common)] 
    ctnu = ctnu.loc[ctnu.index.get_level_values('ct').isin(common)] 
    n = n[common]

    # filter rare inds
    n_cells = n.sum(axis=1)
    common_ids = n_cells[n_cells > int(snakemake.wildcards.ind_min_cellnum)].index.tolist()
    ctp = ctp.loc[ctp.index.get_level_values('ind').isin(common_ids)]
    ctnu = ctnu.loc[ctnu.index.get_level_values('ind').isin(common_ids)]
    n = n[n.index.isin(common_ids)]

    # save n and P
    n.to_csv(snakemake.output.n, sep='\t')
    P = n.div(n.sum(axis=1), axis=0)
    P.to_csv(snakemake.output.P, sep='\t')

    # filter rare ind-cts
    print( time.time() )
    n = n.stack()
    common_cts = n[n > int(snakemake.wildcards.ct_min_cellnum)].index
    ctp = ctp.loc[ctp.index.isin( common_cts )]
    ctnu = ctnu.loc[ctnu.index.isin( common_cts )]

    # exclude gene expressed in limited individuals
    print( time.time() )
    selected = ctp.apply( count_expressed )
    genes = ctp.columns[selected]
    print( time.time() )
    ctp = ctp[genes]
    ctnu = ctnu[genes]
    print( time.time() )

    #
    ctp.to_csv(snakemake.output.ctp, sep='\t')
    ctnu.to_csv(snakemake.output.ctnu, sep='\t')

if __name__ == '__main__':
    main()


