import numpy as np, pandas as pd

nu_fs = [line.strip() for line in open(snakemake.input.ctnu)]

genes = []
for f in nu_fs:
    nu = pd.read_table(f)
    gene = nu.columns[-1]
    nu = nu.pivot(index='donor', columns='day', values=gene)
    if np.any( (nu<1e-12).sum(axis=1) > 1 ):
        pass
    else:
        genes.append( gene )

seed = int(snakemake.params.seed)

genes = np.random.default_rng(seed).choice(genes, snakemake.params.gene_no, replace=False)

np.savetxt(snakemake.output.genes, genes, fmt='%s')
