import numpy as np, pandas as pd

nu_fs = []
for list_f in snakemake.input.imputed_ct_nu:
    for line in open(list_f):
        nu_fs.append( line.strip() )

genes = []
for f in nu_fs:
    nu = pd.read_table(f)
    gene = nu.columns[-1]
    nu = nu.pivot(index='donor', columns='day', values=gene)
    if np.any( (nu<1e-12).sum(axis=1) > 1 ):
        pass
    else:
        genes.append( gene )

np.savetxt(snakemake.output.genes, np.random.default_rng().choice(genes, snakemake.params.gene_no, replace=False), 
        fmt='%s')
