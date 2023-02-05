import os, math, re
import numpy as np, pandas as pd

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
    rng = np.random.default_rng()

    out = np.load(snakemake.input.out, allow_pickle=True).item()
    cty_agg = open(snakemake.output.cty, 'w')
    P_agg = open(snakemake.output.P, 'w')
    ctnu_agg = open(snakemake.output.ctnu, 'w')
    for gene in genes:
        gene_idx = np.where(out['gene'] == gene)[0][0]
        beta = out['reml']['hom']['beta']['ct_beta'][gene_idx]

        P_agg.write(P_fs[gene]+'\n')
        P = np.loadtxt(P_fs[gene])
        pi = np.mean(P, axis=0)
        Pd = P-pi
        ss, C = P.shape[0], P.shape[1]
        s = (Pd.T @ Pd) / ss
        # sanity check
        if not np.isclose(beta @ s @ beta, out['reml']['hom']['fixedeffect_vars']['celltype_main_var'][gene_idx]):
            sys.exit('Fixed effect variance error!\n')

        # homogeneous effect
        hom2 = out['reml']['hom']['hom2'][gene_idx]
        if hom2 < 0:
            hom2 = 0
        alpha = np.outer(rng.normal(scale=math.sqrt(hom2), size=(ss)), np.ones(C))

        # residual effect
        ctnu_agg.write(ctnu_fs[gene]+'\n')
        print(ctnu_fs[gene])
        ctnu = pd.read_table(ctnu_fs[gene]).pivot(index='donor', columns='day', values=gene)
        delta = rng.normal(np.zeros_like(ctnu), np.sqrt(ctnu))

        # pseudobulk
        cty = alpha + np.outer(np.ones(ss), beta) + delta
        cty_f = re.sub('/rep/', f'/rep{gene}/', snakemake.params.cty)
        os.makedirs( os.path.dirname(cty_f), exist_ok=True)
        np.savetxt(cty_f, cty)
        cty_agg.write(cty_f+'\n')
    
    cty_agg.close()
    P_agg.close()
    ctnu_agg.close()

if __name__ == '__main__':
    main()
