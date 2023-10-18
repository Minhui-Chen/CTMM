import os, shutil, sys
import numpy as np, pandas as pd
from ctmm import preprocess

def main():
    # par
    input = snakemake.input
    output = snakemake.output

    # read 
    y = pd.read_table(input.y, index_col=(0,1)) # donor - day * gene
    nu = pd.read_table(input.nu, index_col=(0,1)) # donor - day * gene

    # impute all genes together
    y_path = os.path.dirname(output.y)
    nu_path = os.path.dirname(output.nu)
    if snakemake.wildcards.im_genome in ['Y']:
        if snakemake.wildcards.im_mvn == 'N':
            seed = snakemake.wildcards.get('seed', None)
            if not seed:
                seed = snakemake.params.get('seed', None)
            ### softimpute y
            y = preprocess.softimpute( y, seed=seed, scale=True )
            ### softimpute nu
            nu = preprocess.softimpute( nu, seed=seed, scale=True )
        else:
            sys.exit('Impute Genome only support softImpute!\n')

        y.to_csv(output.y, sep='\t')
        nu.to_csv(output.nu, sep='\t')
    else:
        shutil.copyfile(input.y, output.y)
        shutil.copyfile(input.nu, output.nu)

if __name__ == '__main__':
    main()
