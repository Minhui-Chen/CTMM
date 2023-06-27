import os, math, re, sys
import numpy as np, pandas as pd
from ctmm import util


def main():
    # select genes
    genes = np.loadtxt(snakemake.input.genes, dtype='str')
    genes = genes[snakemake.params.batch]
    np.savetxt(snakemake.output.genes, genes, fmt='%s')

    # collect P and ctnu files for genes
    P_fs = {}
    P_fs_list = []
    for f in snakemake.input.P:
        for line in open(f):
            P_fs_list.append( line.strip() )
    ctnu_fs = {}
    ctnu_fs_list = []
    for f in snakemake.input.imputed_ct_nu:
        for line in open(f):
            ctnu_fs_list.append( line.strip() )
    for gene in genes:
        for P_f in P_fs_list:
            if f'/rep{gene}/' in P_f:
                P_fs[gene] = P_f
        for ctnu_f in ctnu_fs_list:
            if f'/rep{gene}/' in ctnu_f:
                ctnu_fs[gene] = ctnu_f

    #
    out = np.load(snakemake.input.out, allow_pickle=True).item()
    cty_agg = open(snakemake.output.cty, 'w')
    P_agg = open(snakemake.output.P, 'w')
    ctnu_agg = open(snakemake.output.ctnu, 'w')
    for i, gene in enumerate(genes):
        gene_idx = np.where(out['gene'] == gene)[0][0]
        beta = out['reml']['free']['beta']['ct_beta'][gene_idx]

        P = np.loadtxt(P_fs[gene])
        pi = np.mean(P, axis=0)
        Pd = P-pi
        ss, C = P.shape[0], P.shape[1]
        s = (Pd.T @ Pd) / (ss-1)
        # sanity check
        if not np.isclose(beta @ s @ beta, out['reml']['free']['fixedeffect_vars']['ct_beta'][gene_idx]):
            print( beta @ s @ beta )
            print( out['reml']['free']['fixedeffect_vars']['ct_beta'][gene_idx] )
            sys.exit('Fixed effect variance error!\n')

        # simulate cty
        hom2 = out['reml']['free']['hom2'][gene_idx]
        ctnu = pd.read_table(ctnu_fs[gene]).pivot(index='donor', columns='day', values=gene)

        cty = util.sim_pseudobulk(beta, hom2, ctnu.to_numpy(), ss, C, seed=snakemake.params.seed+i)

        # save
        cty_f = re.sub('/rep/', f'/rep{gene}/', snakemake.params.cty)
        os.makedirs( os.path.dirname(cty_f), exist_ok=True)
        np.savetxt(cty_f, cty)
        cty_agg.write(cty_f+'\n')
        ctnu_agg.write(ctnu_fs[gene]+'\n')
        P_agg.write(P_fs[gene]+'\n')
    
    cty_agg.close()
    P_agg.close()
    ctnu_agg.close()


if __name__ == '__main__':
    main()
