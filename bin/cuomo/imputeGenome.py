import os, shutil, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr, STAP
from rpy2.robjects.conversion import localconverter

def main():
    pandas2ri.activate()
    numpy2ri.activate()
    # par
    input = snakemake.input
    output = snakemake.output

    # read 
    y = pd.read_table(input.y, index_col=(0,1)) # donor - day * gene
    donors = np.unique(y.index.get_level_values('donor'))
    nu = pd.read_table(input.nu, index_col=(0,1)) # donor - day * gene

    if snakemake.wildcards.im_genome == 'Y_Corrected':
        supp = pd.read_table(snakemake.input.supp, usecols=['donor_id_short','sex','donor_disease_status'])
        supp = supp.rename(columns={'donor_id_short':'donor', 'donor_disease_status':'disease'})
        # remove the duplicated individual iisa
        supp = supp.drop_duplicates(subset='donor')
        supp = supp.loc[supp['donor'].isin(donors)]
        # code sex
        supp['code'] = 0
        supp.loc[supp['sex'] == 'male', 'code'] = 1
        supp['sex'] = supp['code']
        # code disease
        supp['code'] = 0
        supp.loc[supp['disease'] == 'normal', 'code'] = 1
        supp['disease'] = supp['code']

        meta = pd.read_table(snakemake.input.meta, usecols=['donor', 'experiment'])
        meta = meta.loc[meta['donor'].isin(donors)]
        meta = meta.drop_duplicates().sort_values(by='donor').reset_index(drop=True)
        if meta.shape[0] != len(np.unique(meta['donor'])):
            print(meta[meta.duplicated(subset='donor',keep=False)])
            sys.exit('More than one experiments for an individual!\n')
        experiments = list( np.unique(meta['experiment']) )
        for experiment in experiments[:-1]:
            meta[experiment] = 0
            meta.loc[meta['experiment']==experiment, experiment] = 1
        meta = meta[['donor']+experiments[:-1]]

        covars = meta.merge(supp, on='donor')
        covars = covars.set_index('donor')
        covars.columns = pd.MultiIndex.from_product([['covar'],covars.columns], names=['first','second'])

    # transform from donor-day * gene to donor * gene-day
    y = y.unstack(level=1)
    y_columns = y.columns
    print(y)
    nu = nu.unstack(level=1)
    nu_columns = nu.columns
    print(nu)

    # impute all genes together
    y_path = os.path.dirname(output.y)
    nu_path = os.path.dirname(output.nu)
    if snakemake.wildcards.im_genome in ['Y', 'Y_Corrected']:
        if snakemake.wildcards.im_genome == 'Y_Corrected':
            y = y.merge(covars, left_index=True, right_index=True)
            nu = nu.merge(covars, left_index=True, right_index=True)
            print('Corrected Y:', y.shape, nu.shape)
        if snakemake.wildcards.im_mvn == 'N':
            softImpute_f = 'bin/my_softImpute.R'
            softImpute_r = STAP( open(softImpute_f).read(), 'softImpute_r' )
            ### y
            if snakemake.wildcards.im_scale in ['Y','Y_Corrected']:
                out = softImpute_r.my_softImpute( r['as.matrix'](y), scale=ro.vectors.BoolVector([True]) )
            elif snakemake.wildcards.im_scale == 'Y2':
                out = softImpute_r.my_softImpute( r['as.matrix'](y), biscale=ro.vectors.BoolVector([True]) )
            else:
                out = softImpute_r.my_softImpute( r['as.matrix'](y) )
            out = dict( zip(out.names, list(out)) )
            y = pd.DataFrame(out['Y'], index=y.index, columns=y.columns)[y_columns]
            ### nu
            if snakemake.wildcards.im_scale in ['Y','Y_Corrected']:
                out = softImpute_r.my_softImpute( r['as.matrix'](nu), scale=ro.vectors.BoolVector([True]) )
            elif snakemake.wildcards.im_scale == 'Y2':
                out = softImpute_r.my_softImpute( r['as.matrix'](nu), biscale=ro.vectors.BoolVector([True]) )
            else:
                out = softImpute_r.my_softImpute( r['as.matrix'](nu) )
            out = dict( zip(out.names, list(out)) )
            nu = pd.DataFrame(out['Y'], index=nu.index, columns=nu.columns)[nu_columns]
        elif snakemake.wildcards.im_mvn == 'D':
            # delete individuals with missing data (no imputation)
            y = y.dropna()
            nu = nu.dropna()
        else:
            sys.exit('Impute Genome only support softImpute!\n')

        # transform back from donor * gene-day to donor-day * gene
        y = y.stack(level=1).sort_values(by=['donor','day'])
        nu = nu.stack(level=1).sort_values(by=['donor','day'])
        y.to_csv(output.y, sep='\t')
        nu.to_csv(output.nu, sep='\t')
    else:
        shutil.copyfile(input.y, output.y)
        shutil.copyfile(input.nu, output.nu)

if __name__ == '__main__':
    main()
