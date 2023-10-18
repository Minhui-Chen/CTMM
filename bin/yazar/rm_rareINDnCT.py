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
    ctp = pd.read_table(snakemake.input.ctp, index_col=(0, 1))
    ctnu = pd.read_table(snakemake.input.ctnu, index_col=(0, 1))
    n = pd.read_table(snakemake.input.n, index_col=0).astype('int')

    # 
    n_inds = n.shape[0]
    cts = n.columns.to_numpy()
    
    # find common cts with less than prop missing inds
    ct_min_cellnum = int(snakemake.wildcards.ct_min_cellnum)
    prop = float(snakemake.wildcards.prop)
    ct_nonmissing_props = (n > ct_min_cellnum).sum(axis=0) / n_inds
    common = ct_nonmissing_props[ct_nonmissing_props > prop].index.tolist()

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
    n = n.stack()
    common_cts = n.index[n.to_numpy() > ct_min_cellnum]
    ctp = ctp.loc[ctp.index.isin(common_cts)]
    ctnu = ctnu.loc[ctnu.index.isin(common_cts)]

    # exclude gene expressed in limited individuals
    selected = ctp.apply(count_expressed)
    genes = ctp.columns[selected]
    ctp = ctp[genes]
    ctnu = ctnu[genes]

    #
    ctp.to_csv(snakemake.output.ctp, sep='\t')
    ctnu.to_csv(snakemake.output.ctnu, sep='\t')

if __name__ == '__main__':
    main()


