import numpy as np, pandas as pd

def main():
    # par
    input = snakemake.input
    output = snakemake.output

    # read 
    meta = pd.read_table(input.meta)[['donor', 'day', 'cell_name']]
    y = pd.read_table(input.y, index_col=(0,1)) # donor - day * gene
    nu = pd.read_table(input.nu, index_col=(0,1)) # donor - day * gene

    # collect days and inds and genes
    days = list(np.unique(meta['day']))
    inds = list(np.unique(meta['donor']))

    # cell type proportions (cell numbers)
    P = {}
    for day in days: 
        P[day] = []
        for ind in inds:
            cells = np.array(meta.loc[(meta['donor']==ind) & (meta['day']==day), 'cell_name'])
            P[day].append(len(cells))
    P = pd.DataFrame(P, index=inds) # ind * day
    ## order P
    P.index.name = 'donor'
    P = P.sort_values(by='donor')
    P = P[days]

    # exclude individuals with small number of cells
    P = P.loc[P.sum(axis=1) > int(snakemake.wildcards.ind_min_cellnum)]
    y = y.loc[y.index.get_level_values('donor').isin(P.index)]
    nu = nu.loc[nu.index.get_level_values('donor').isin(P.index)]

    y.to_csv(output.y, sep='\t')
    nu.to_csv(output.nu, sep='\t')

    P.to_csv(output.n, sep='\t')
    P_frac = P.divide(P.sum(axis=1), axis=0)
    P_frac.to_csv(output.P, sep='\t')

if __name__ == '__main__':
    main()
