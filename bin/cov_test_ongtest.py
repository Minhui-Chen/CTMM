import os, sys, re
import helper, mystats 
import scipy
import numpy as np, pandas as pd
import rpy2.robjects as robjects 
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from rpy2.robjects.conversion import localconverter
import wald, util, op_R

#def he_randomeffect_vars(x_m, random_covars_d, ests, sig2s):
#    if len(random_covars_d.keys()) > 0:
#        r2_l = ests[x_m.shape[1]:]
#        randomeffect_vars_l, _ = util.RandomeffectVariance( r2_l,
#                [np.loadtxt( random_covars_d[key] ) for key in np.sort(list(random_covars_d.keys()))] )
#        randomeffect_vars_d, r2_d = util.assign_randomeffect_vars(randomeffect_vars_l, 
#                r2_l, random_covars_d)
#        for r2, key in zip(r2_l, np.sort(list(random_covars_d.keys()))):
#            Q = np.loadtxt(random_covars_d[key])
#            sig2s = np.diag(sig2s) + r2 * Q @ Q.T
#    else:
#        randomeffect_vars_d = {}
#        r2_d = {}
#    return( randomeffect_vars_d, r2_d, sig2s)

def main():
    # par
    params = snakemake.params
    input = snakemake.input
    output = snakemake.output
    wildcards = snakemake.wildcards

    # read covariates
    covars_f = helper.generate_tmpfn()
    if open(input.fixed).read().strip() != 'NA':
        fixed = [np.loadtxt(f.strip()) for f in open(input.fixed)]
    else:
        fixed = [None] * len(open(input.y).readlines())
    if open(input.random).read().strip() != 'NA':
        random = [np.loadtxt(f.strip()) for f in open(input.random)]
    else:
        random = [None] * len(open(input.y).readlines())

    #
    genes = params.batch
    outs = [re.sub('/rep/', f'/rep{gene}/', params.out) for gene in genes]
    for gene, y_f, P_f, nu_f, out_f, fixed, random in zip(genes, [line.strip() for line in open(input.y)],
            [line.strip() for line in open(input.P)], [line.strip() for line in open(input.nu)], outs, 
            fixed, random):
        print(y_f, P_f, nu_f)
        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape # cell type number
        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        # celltype specific mean nu
        #ctnu = pd.read_table( ctnu_f )
        #cts = np.unique( ctnu['day'] )
        #ctnu_grouped = ctnu.groupby('day').mean()

        if fixed is not None:
            if len( fixed.shape ) == 1:
                fixed = fixed.reshape(-1,1)
            X = np.concatenate( (P, fixed), axis=1 )
            np.savetxt(covars_f+'.fixed', fixed)
            fixed_covars_d = {'fixed':covars_f+'.fixed'}
        else:
            X = P
            fixed_covars_d = {}

        if random is not None:
            np.savetxt(covars_f+'.random', random)
            random_covars_d = {'random':covars_f+'.random'}
        else:
            random_covars_d = {}

        ## HE
        hom_he, hom_he_wald = op_R.hom_HE(y_f, P_f, nu_f, 
                fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)

        iid_he, iid_he_wald = op_R.iid_HE(y_f, P_f, nu_f, 
                fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)

        free_he, free_he_wald = op_R.free_HE(y_f, P_f, nu_f, 
                fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)

        full_he = op_R.full_HE(y_f, P_f, nu_f, 
                fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)

        if 'HE_as_initial' not in snakemake.params.keys():
            snakemake.params.HE_as_initial = False

        ## ML
        if not snakemake.params.HE_as_initial:
            hom_ml, hom_ml_wald = op_R.hom_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)

            iid_ml, iid_ml_wald = op_R.iid_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)

            free_ml, free_ml_wald = op_R.free_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)

            full_ml = op_R.full_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d)
        else:
            hom_ml, hom_ml_wald = op_R.hom_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d,
                    par=util.generate_HE_initial(hom_he, ML=True) )

            iid_ml, iid_ml_wald = op_R.iid_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d,
                    par=util.generate_HE_initial(iid_he, ML=True))

            free_ml, free_ml_wald = op_R.free_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d,
                    par=util.generate_HE_initial(free_he, ML=True))

            full_ml = op_R.full_ML(y_f, P_f, nu_f, 
                    fixed_covars_d=fixed_covars_d, random_covars_d=random_covars_d,
                    par=util.generate_HE_initial(full_he, ML=True))

        ## REML
        if not snakemake.params.HE_as_initial:
            hom_reml, hom_reml_wald = op_R.hom_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d)

            iid_reml, iid_reml_wald = op_R.iid_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d)

            free_reml, free_reml_wald = op_R.free_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d)

            full_reml = op_R.full_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d)
        else:
            hom_reml, hom_reml_wald = op_R.hom_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d, 
                    par=util.generate_HE_initial(hom_he, REML=True))

            iid_reml, iid_reml_wald = op_R.iid_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d,
                    par=util.generate_HE_initial(iid_he, REML=True))

            free_reml, free_reml_wald = op_R.free_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d,
                    par=util.generate_HE_initial(free_he, REML=True))

            full_reml = op_R.full_REML(y_f, P_f, nu_f, fixed_covars_d, random_covars_d,
                    par=util.generate_HE_initial(full_he, REML=True))

        out = {
                'ml': {'hom': hom_ml, 'iid': iid_ml, 'free': free_ml, 'full': full_ml,
                    'wald':{'hom':hom_ml_wald, 'iid':iid_ml_wald, 'free':free_ml_wald} }, 
                'reml':{'hom':hom_reml, 'iid':iid_reml, 'free':free_reml, 'full':full_reml,
                    'wald':{'hom':hom_reml_wald, 'iid':iid_reml_wald, 'free':free_reml_wald} },
                'he': {'hom': hom_he, 'iid': iid_he, 'free': free_he, 'full':full_he,
                    'wald':{'hom':hom_he_wald, 'iid': iid_he_wald, 'free': free_he_wald} },
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
