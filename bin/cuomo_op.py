import os, sys, re
import helper, mystats 
import scipy
import numpy as np, pandas as pd
import rpy2.robjects as robjects 
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from rpy2.robjects.conversion import localconverter
import wald, util, ong_test, cuomo_ctng_test

def main():
    # par
    params = snakemake.params
    input = snakemake.input
    output = snakemake.output
    wildcards = snakemake.wildcards

    # collect covariates
    fixed_covars_d, random_covars_d = cuomo_ctng_test.collect_covariates(snakemake)

    #
    genes = params.genes
    outs = [re.sub('/rep/', f'/rep{gene}/', params.out) for gene in genes]
    for gene, y_f, P_f, nu_f, out_f, ctnu_f in zip(genes, [line.strip() for line in open(input.y)],
            [line.strip() for line in open(input.P)], [line.strip() for line in open(input.nu)], outs, 
            [line.strip() for line in open(input.imputed_ct_nu)]):
        print(y_f, P_f, nu_f)
        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape # cell type number
        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        # celltype specific mean nu
        ctnu = pd.read_table( ctnu_f )
        cts = np.unique( ctnu['day'] )
        ctnu_grouped = ctnu.groupby('day').mean()

        #X = ong_test.get_X(P, fixed_covars_d)

        ## HE
        # project out fixed effect (including cell type main effect) from y
        #proj = np.eye(len(y)) - X @ np.linalg.inv(X.T @ X) @ X.T
        #y_p = proj @ y
        #Y = y_p**2 - np.diag(proj @ np.diag(vs) @ proj)


        ## HE
        hom_he, hom_he_wald = ong_test.hom_HE(y_f, P_f, nu_f,
                fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d, jack_knife=True)

        iid_he, iid_he_wald = ong_test.iid_HE(y_f, P_f, nu_f, 
                fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d, jack_knife=True)

        free_he, free_he_wald = ong_test.free_HE(y_f, P_f, nu_f, 
                fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d, jack_knife=True)

        full_he = ong_test.full_HE(y_f, P_f, nu_f, 
                fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)


        if 'HE_as_initial' not in snakemake.params.keys():
            snakemake.params.HE_as_initial = False

        ## ML
        #null_ml = null_ML(y_f, P_f, nu_f)
        if not snakemake.params.HE_as_initial:
            hom_ml, hom_ml_wald = ong_test.hom_ML(y_f, P_f, nu_f,  
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)
            iid_ml, iid_ml_wald = ong_test.iid_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)
            free_ml, free_ml_wald = ong_test.free_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)
            full_ml = ong_test.full_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)
        else:
            hom_ml, hom_ml_wald = ong_test.hom_ML(y_f, P_f, nu_f,  
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d,
                    par=util.generate_HE_initial(hom_he, ML=True))
            iid_ml, iid_ml_wald = ong_test.iid_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d,
                    par=util.generate_HE_initial(iid_he, ML=True))
            free_ml, free_ml_wald = ong_test.free_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d,
                    par=util.generate_HE_initial(free_he, ML=True))
            full_ml = ong_test.full_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d,
                    par=util.generate_HE_initial(full_he, ML=True))

        ## REML
        if not snakemake.params.HE_as_initial:
            hom_reml, hom_reml_wald = ong_test.hom_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d)
            iid_reml, iid_reml_wald = ong_test.iid_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d)
            free_reml, free_reml_wald = ong_test.free_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d)
            full_reml = ong_test.full_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d)
        else:
            hom_reml, hom_reml_wald = ong_test.hom_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d,
                    par=util.generate_HE_initial(hom_he, REML=True))
            iid_reml, iid_reml_wald = ong_test.iid_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d,
                    par=util.generate_HE_initial(iid_he, REML=True))
            free_reml, free_reml_wald = ong_test.free_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d,
                    par=util.generate_HE_initial(free_he, REML=True))
            full_reml = ong_test.full_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d,
                    par=util.generate_HE_initial(full_he, REML=True))

        out = {
                'ml': {'hom': hom_ml, 'iid': iid_ml, 'free': free_ml, 'full': full_ml,
                    'wald':{'hom':hom_ml_wald, 'iid':iid_ml_wald, 'free':free_ml_wald} }, 
                'reml':{'hom':hom_reml, 'iid':iid_reml, 'free':free_reml, 'full':full_reml,
                    'wald':{'hom':hom_reml_wald, 'iid':iid_reml_wald, 'free':free_reml_wald} },
                'he': {'hom': hom_he, 'iid': iid_he, 'free': free_he, 'full':full_he,
                    'wald':{'hom':hom_he_wald, 'iid': iid_he_wald, 'free': free_he_wald} },
                'ct_mean_nu': {ct:ctnu_grouped.loc[ct, gene] for ct in cts},
                'gene': gene
                }

        # LRT
        ## ML
        #hom_null_lrt = mystats.lrt(out['ml']['hom']['l'], out['ml']['null']['l'], 1)
        iid_hom_lrt = mystats.lrt(out['ml']['iid']['l'], out['ml']['hom']['l'], 1)
        free_hom_lrt = mystats.lrt(out['ml']['free']['l'], out['ml']['hom']['l'], C)
        free_iid_lrt = mystats.lrt(out['ml']['free']['l'], out['ml']['iid']['l'], C-1)
        full_hom_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['hom']['l'], C*(C+1)//2-1)
        full_iid_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['iid']['l'], C*(C+1)//2-2)
        full_free_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['free']['l'], C*(C+1)//2-C-1)

        #out['ml']['lrt'] = {'hom_null':hom_null_lrt, 'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
        out['ml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
                'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
                'full_free':full_free_lrt}

        ## REML
        iid_hom_lrt = mystats.lrt(out['reml']['iid']['l'], out['reml']['hom']['l'], 1)
        free_hom_lrt = mystats.lrt(out['reml']['free']['l'], out['reml']['hom']['l'], C)
        free_iid_lrt = mystats.lrt(out['reml']['free']['l'], out['reml']['iid']['l'], C-1)
        full_hom_lrt = mystats.lrt(out['reml']['full']['l'], out['reml']['hom']['l'], C*(C+1)//2-1)
        full_iid_lrt = mystats.lrt(out['reml']['full']['l'], out['reml']['iid']['l'], C*(C+1)//2-2)
        full_free_lrt = mystats.lrt(out['reml']['full']['l'], out['reml']['free']['l'], C*(C+1)//2-C-1)

        out['reml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
                'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
                'full_free':full_free_lrt}

        # save
        np.save(out_f, out)

    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))

if __name__ == '__main__':

    main()
