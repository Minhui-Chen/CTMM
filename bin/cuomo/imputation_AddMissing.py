import numpy as np, pandas as pd

def main():
    # par
    y = pd.read_table( snakemake.input.y, index_col=(0,1) )
    y = y.unstack().stack(dropna=False)
    nu = pd.read_table( snakemake.input.nu, index_col=(0,1) )
    nu = nu.unstack().stack(dropna=False)
    log = open(snakemake.output.log, 'w')

    #
    nmiss = y.isna().sum().sum()
    if nmiss != nu.isna().sum().sum():
        sys.exit('Wrong input!\n')
    ntotal = y.shape[0] * y.shape[1]
    nobs = ntotal - nmiss
    log.write(f'Original missing values: {nmiss} {nmiss/ntotal}\n')
    n_add = int(ntotal * float(snakemake.wildcards.missingness))
    
    rng = np.random.default_rng()
    # y
#    if snakemake.wildcards.random_mask == 'Y':
#        y_ = y.stack(dropna=False)
#        addmiss = np.arange(len(y_)) * 1.0
#        addmiss[np.isnan(y_)] = np.nan
#        #
#        addmiss[np.isin(addmiss, rng.choice(addmiss[~np.isnan(addmiss)], n_add))] = np.nan
#        addmiss = 0 * addmiss
#        log.write(f'Add {n_add} missingness\n')
#        y_ = y_ + addmiss
#        y = y_.unstack()
#    else:
    y_ = y.copy().unstack()
    k = 0
    while k < n_add:
        to = rng.choice(y_.index)
        k1 = np.isnan(y_.loc[to]).sum()
        source = y_.sample().iloc[0]
        y_.loc[to] = np.array(y_.loc[to]) + np.array(source) * 0
        k2 = np.isnan(y_.loc[to]).sum()
        k = k + k2 - k1
        if y_.dropna(how='all').shape[0] != y_.shape[0]:
            y_ = y.copy().unstack()
            k = 0
    log.write(f'Add {k} missingness\n')
    y = y_.stack(dropna=False)
    
    # nu
    nu = nu + 0 * y

    #
    log.close()
    y.to_csv(snakemake.output.y, sep='\t')
    nu.to_csv(snakemake.output.nu, sep='\t')

        
if __name__ == '__main__':
    main()
