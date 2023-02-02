import os, shutil, sys
import numpy as np, pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr, STAP

def main():
    # par
    input = snakemake.input
    output = snakemake.output

    # read 
    y = pd.read_table(input.y, index_col=(0,1)) # donor - day * gene
    donors = np.unique(y.index.get_level_values('donor'))
    nu = pd.read_table(input.nu, index_col=(0,1)) # donor - day * gene

    # transform from donor-day * gene to donor * gene-day
    y = y.unstack(level=1)
    y_columns = y.columns
    nu = nu.unstack(level=1)
    nu_columns = nu.columns

    # impute all genes together
    y_path = os.path.dirname(output.y)
    nu_path = os.path.dirname(output.nu)
    if snakemake.wildcards.im_genome in ['Y']:
        if snakemake.wildcards.im_mvn == 'N':
            softImpute_f = 'bin/my_softImpute.R'
            softImpute_r = STAP( open(softImpute_f).read(), 'softImpute_r' )
            pandas2ri.activate()
            numpy2ri.activate()
            if 'seed' not in snakemake.params.keys():
                seed = ro.NULL
            else:
                seed = ro.NULL if snakemake.params.seed is None else ro.FloatVector([snakemake.params.seed])
            ### y
            out = softImpute_r.my_softImpute( r['as.matrix'](y), scale=ro.vectors.BoolVector([True]), seed=seed )
            out = dict( zip(out.names, list(out)) )
            y = pd.DataFrame(out['Y'], index=y.index, columns=y.columns)[y_columns]
            ### nu
            out = softImpute_r.my_softImpute( r['as.matrix'](nu), scale=ro.vectors.BoolVector([True]), seed=seed )
            out = dict( zip(out.names, list(out)) )
            nu = pd.DataFrame(out['Y'], index=nu.index, columns=nu.columns)[nu_columns]
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
