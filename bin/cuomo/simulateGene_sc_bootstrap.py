import os
import re
import sys

import numpy as np
import pandas as pd
from ctmm import util, log


def main():
    rng = np.random.default_rng(int(snakemake.params.seed))

    # fraction of cells to sample
    frac = float(snakemake.wildcards.cell_no) 

    # read depth
    depth = float(snakemake.wildcards.depth)

    # whether to perform resampling individuals with 0 counts
    resample_inds = snakemake.params.get('resample_inds', True)
    
    # read data
    counts = pd.read_table(snakemake.input.counts, index_col=0)
    counts = counts.transpose()
    counts['total_counts'] = counts.sum(axis=1)

    meta = pd.read_table(snakemake.input.meta)
    meta = meta.rename(columns={'cell_name': 'cell'})
    C = len(np.unique(meta['day']))

    # merge
    data = counts.merge(meta, left_index=True, right_on='cell')

    # genes to simulate
    all_genes = np.loadtxt(snakemake.input.genes, dtype='str')

    # ct mean expression
    ct_means = data.groupby('day')[all_genes.tolist()].mean()

    for batch, gene_f, P_f, ctnu_f, cty_f in zip(snakemake.params.batch, snakemake.output.genes, 
                                                 snakemake.output.P, snakemake.output.ctnu, snakemake.output.cty):
        log.logger.info(gene_f)

        # select genes
        genes = all_genes[batch]
        np.savetxt(gene_f, genes, fmt='%s')

        cty_agg = open(cty_f, 'w')
        P_agg = open(P_f, 'w')
        ctnu_agg = open(ctnu_f, 'w')
        for gene in genes:
            main_ct = ct_means[gene].idxmax()
            ind_grouped = data.loc[data['day'] == main_ct].groupby('donor')

            # select inds with > 10 cells
            inds = ind_grouped.size()
            inds = inds[inds > 10].index.to_numpy()

            # exclude inds with >1 read
            cell_y = ind_grouped[gene].max()
            inds2 = cell_y[cell_y >= 1].index.to_numpy()
            inds = inds[np.isin(inds, inds2)]

            # select data
            gene_data = data.loc[data['donor'].isin(inds) & (data['day'] == main_ct), 
                                 ['cell', 'total_counts', gene, 'donor']].copy()
            gene_data = gene_data.rename(columns={gene: 'gene', 'donor': 'ind'})

            # free
            if 'V' in snakemake.wildcards.keys():
                V = float(snakemake.wildcards.V)
            else:
                V = 0

            # simulate
            if gene == 'ENSG00000176896_TCEANC':
                cty, ctnu = util.sim_sc_bootstrap(gene_data, C, frac, depth, V, rng.integers(100000)+1, 
                                                option=int(snakemake.wildcards.option), resample_inds=resample_inds)
            else:
                cty, ctnu = util.sim_sc_bootstrap(gene_data, C, frac, depth, V, rng.integers(100000), 
                                                option=int(snakemake.wildcards.option), resample_inds=resample_inds)

            cty = cty.to_numpy()
            ctnu = pd.DataFrame(ctnu.stack()).reset_index()
            ctnu.columns = ['donor', 'day', gene]

            # save
            gene_cty_f = f'{snakemake.params.cty}/rep{gene}/cty.gz'
            os.makedirs(os.path.dirname(gene_cty_f), exist_ok=True)
            np.savetxt(gene_cty_f, cty)
            cty_agg.write(gene_cty_f + '\n')

            gene_ctnu_f = re.sub('cty', 'ctnu', gene_cty_f)
            ctnu.to_csv(gene_ctnu_f, sep='\t', index=False)
            ctnu_agg.write(gene_ctnu_f + '\n')

            gene_P_f = re.sub('cty', 'P', gene_cty_f)
            P = np.ones_like(cty) / C
            np.savetxt(gene_P_f, P)
            P_agg.write(gene_P_f + '\n')

        cty_agg.close()
        P_agg.close()
        ctnu_agg.close()


if __name__ == '__main__':
    main()

