import time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    #
    raw_y = pd.read_table(snakemake.input.raw_y, index_col=(0,1)).unstack().stack(dropna=False)
    raw_nu = pd.read_table(snakemake.input.raw_nu, index_col=(0,1)).unstack().stack(dropna=False)

    # make sure raw, masked, imputed all have the same order of donor, day, genes
    donors = raw_y.index.get_level_values('donor')
    days = raw_y.index.get_level_values('day')
    genes = raw_y.columns
    if np.any(donors != raw_nu.index.get_level_values('donor')) or np.any(days != raw_nu.index.get_level_values('day')) or np.any(genes != raw_nu.columns):
        sys.exit('Not matching!\n')

    masked_y = pd.read_table(snakemake.input.masked_y[0], index_col=(0,1))
    masked_nu = pd.read_table(snakemake.input.masked_nu[0], index_col=(0,1))
    print(len(donors), len(days), len(masked_y.index.get_level_values('donor')), len(masked_y.index.get_level_values('day')), len(genes), len(masked_y.columns))
    if np.any(donors != masked_y.index.get_level_values('donor')) or np.any(days != masked_y.index.get_level_values('day')) or np.any(genes != masked_y.columns):
        sys.exit('Not matching!\n')
    if np.any(donors != masked_nu.index.get_level_values('donor')) or np.any(days != masked_nu.index.get_level_values('day')) or np.any(genes != masked_nu.columns):
        sys.exit('Not matching!\n')

    imputed_y = pd.read_table(snakemake.input.imputed_y[0], index_col=(0,1))
    imputed_nu = pd.read_table(snakemake.input.imputed_nu[0], index_col=(0,1))
    if np.any(donors != imputed_y.index.get_level_values('donor')) or np.any(days != imputed_y.index.get_level_values('day')) or np.any(genes != imputed_y.columns):
        sys.exit('Not matching!\n')
    if np.any(donors!=imputed_nu.index.get_level_values('donor')) or np.any(days!=imputed_nu.index.get_level_values('day')) or np.any(genes != imputed_nu.columns):
        sys.exit('Not matching!\n')

    # 
    raw_y = raw_y.unstack()
    raw_nu = raw_nu.unstack()

    raw_y_maskeds = []
    raw_nu_maskeds = []
    imputed_y_maskeds = []
    imputed_nu_maskeds = []
    print( time.time() )
    for i in range(len(snakemake.input.masked_y)):
        masked_y = pd.read_table(snakemake.input.masked_y[i], index_col=(0,1)).unstack()
        masked_nu = pd.read_table(snakemake.input.masked_nu[i], index_col=(0,1)).unstack()
        imputed_y = pd.read_table(snakemake.input.imputed_y[i], index_col=(0,1)).unstack()
        imputed_nu = pd.read_table(snakemake.input.imputed_nu[i], index_col=(0,1)).unstack()
       
        ## only keep entries of added missing, set all others to NA
        raw_y_ = raw_y.mask(~(masked_y.isna() & (~raw_y.isna()))).dropna(how='all') 
        raw_nu_ = raw_nu.mask(~(masked_nu.isna() & (~raw_nu.isna()))).dropna(how='all')
        imputed_y_ = imputed_y.mask(~(masked_y.isna() & (~raw_y.isna()))).dropna(how='all')
        imputed_nu_ = imputed_nu.mask(~(masked_nu.isna() & (~raw_nu.isna()))).dropna(how='all')

        ## make sure index matching
        donor = raw_y_.index
        if np.any(donor != raw_nu_.index):
            sys.exit('Not matching raw_nu_!\n')
        print(donor)
        print(imputed_y_.index)
        if np.any(donor != imputed_y_.index):
            sys.exit('Not matching imputed_y_!\n')
        if np.any(donor != imputed_nu_.index):
            sys.exit('Not matching imptued_nu_!\n')

        raw_y_maskeds.append( raw_y_ )
        raw_nu_maskeds.append( raw_nu_ )
        imputed_y_maskeds.append( imputed_y_ )
        imputed_nu_maskeds.append( imputed_nu_ )

    print( time.time() )
    # merge 
    ## only keep columns that have added missing
    raw_y_masked = pd.concat( raw_y_maskeds, ignore_index=True ).dropna(axis=1, how='all')
    raw_nu_masked = pd.concat( raw_nu_maskeds, ignore_index=True ).dropna(axis=1, how='all')
    imputed_y_masked = pd.concat( imputed_y_maskeds, ignore_index=True ).dropna(axis=1, how='all')
    imputed_nu_masked = pd.concat( imputed_nu_maskeds, ignore_index=True ).dropna(axis=1, how='all')
    ## only keep columns with more than one unique non-nan
    print( time.time() )
    cols = []
    for col in raw_y_masked.columns:
        x = np.array( raw_y_masked[col] )
        x = x[~np.isnan(x)]
        if len(np.unique(x)) > 1:
            cols.append(col)
    print( time.time() )
    raw_y_masked = raw_y_masked[cols]
    raw_nu_masked = raw_nu_masked[cols]
    imputed_y_masked = imputed_y_masked[cols]
    imputed_nu_masked = imputed_nu_masked[cols]
    raw_y = raw_y[cols]
    raw_nu = raw_nu[cols]

    # standize raw accoding to the standardization procejure for imputed data
#    P = pd.read_table(snakemake.input.P, index_col=0) # donor * day
#    P.index.name = 'donor'
#    if snakemake.wildcards.im_mvn == 'D':
#        # delete individuals with missing data (so no imputation)
#        P = P.loc[P.index.isin(y['donor'])]
#    P = P.sort_values(by='donor')[np.unique(days)]
#    print(P.head())
#    if np.any(raw_y.index != P.index):
#        sys.exit('Not matching!\n')
#    for gene in genes:
#        cty = raw_y[gene]
#        y = cty.mul(P).sum(axis=1)
#        y_mean = y.mean()
#        y_std = y.std()
#        y_var = y.var()
#        raw_y_masks[gene] = (raw_y_masks[gene] - y_mean) / y_std
#        raw_nu_masks[gene] = raw_nu_masks[gene] / y_var
#        raw_nu[gene] = raw_nu[gene] / y_var

#    raw_nu.to_csv(snakemake.output.raw_nu_standradized, sep='\t')

    # across replicates
    ## mse
    print( time.time() )
    mse = lambda x, y, z: ( (x - y)**2 ).mean() / z.var()
    #y_mse = ((raw_y_masks-imputed_y_masks)**2).mean() 
    y_mse = mse( raw_y_masked, imputed_y_masked, raw_y )
    if np.any( np.isinf(y_mse) ):
        print( raw_y_masked.var()[np.isinf(y_mse)] )
        tmp = raw_y_masked[raw_y_masked.columns[np.isinf(y_mse)]]
        for col in raw_y_masked.columns[np.isinf(y_mse)]:
            tmp = raw_y_masked[col]
            print( tmp[ ~tmp.isna() ] )
        sys.exit('Error\n')
    #nu_mse = ((raw_nu_masks-imputed_nu_masks)**2).mean() 
    nu_mse = (( raw_nu_masked - imputed_nu_masked )**2 ).mean() / raw_nu.var()
    print( time.time() )

    y_mse.to_csv(snakemake.output.y_mse, header=False)
    nu_mse.to_csv(snakemake.output.nu_mse, header=False)

    ## correlation
    y_cor = raw_y_masked.corrwith(imputed_y_masked)
    nu_cor = raw_nu_masked.corrwith(imputed_nu_masked)
    print( time.time() )
    #fig, ax = plt.subplots()
    #for tmp_column in raw_nu_masked.columns[:10]:
    #    ax.scatter(raw_nu_masked[tmp_column], imputed_nu_masked[tmp_column])
    #fig.savefig(snakemake.output.nu_png)

    y_cor.to_csv(snakemake.output.y_cor, header=False)
    nu_cor.to_csv(snakemake.output.nu_cor, header=False)

    # within replicate
    ## calculate mse and correlation within replicate
    y_mses = []
    nu_mses = []
    y_cors = []
    nu_cors = []
    i = 0
    for raw_y_masked_, imputed_y_masked_, raw_nu_masked_, imputed_nu_masked_ in zip(
            raw_y_maskeds, imputed_y_maskeds, raw_nu_maskeds, imputed_nu_maskeds):
        i += 1
        if i%10 == 0:
            print(i)
        # only column with more than one unique non-nan values
        tmp_cols = []
        for col in cols:
            x = np.array( raw_y_masked_[col] )
            x = x[~np.isnan(x)]
            if len(np.unique(x)) > 1:
                tmp_cols.append( col )

        raw_y_masked_ = raw_y_masked_[tmp_cols]
        imputed_y_masked_ = imputed_y_masked_[tmp_cols]
        raw_nu_masked_ = raw_nu_masked_[tmp_cols]
        imputed_nu_masked_ = imputed_nu_masked_[tmp_cols]
        raw_y_ = raw_y[tmp_cols]
        raw_nu_ = raw_nu[tmp_cols]

        ## mse
        y_mse_ = mse( raw_y_masked_, imputed_y_masked_, raw_y_ )
        nu_mse_ = mse( raw_nu_masked_, imputed_nu_masked_, raw_nu_ )
        y_mses.append( y_mse_ )
        nu_mses.append( nu_mse_ )

        ## cor
        y_cor_ = raw_y_masked_.corrwith( imputed_y_masked_ )
        nu_cor_ = raw_nu_masked_.corrwith( imputed_nu_masked_ )
        y_cors.append( y_cor_ )
        nu_cors.append( nu_cor_ )

    ## merge across replicates and get median
    pd.concat( y_mses, axis=1 ).to_csv(snakemake.output.y_mse_within_tmp, header=False)
    print( pd.concat( y_mses, axis=1 ).shape )
    y_mses = pd.concat( y_mses, axis=1 ).median(axis=1)
    nu_mses = pd.concat( nu_mses, axis=1 ).median(axis=1)
    y_cors = pd.concat( y_cors, axis=1 ).median(axis=1)
    nu_cors = pd.concat( nu_cors, axis=1 ).median(axis=1)

    y_mses.to_csv(snakemake.output.y_mse_within, header=False)
    nu_mses.to_csv(snakemake.output.nu_mse_within, header=False)
    y_cors.to_csv(snakemake.output.y_cor_within, header=False)
    nu_cors.to_csv(snakemake.output.nu_cor_within, header=False)


if __name__ == '__main__':
    main()
