import os, sys, re, time
import helper, mystats 
import scipy
import numpy as np, pandas as pd
import rpy2.robjects as robjects 
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from rpy2.robjects.conversion import localconverter
import screml, wald, util, cuomo_ctng_test
import ctp_R

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
    outs = []
    for gene, y_f, P_f, nu_f, ctnu_f, fixed, random in zip(genes, 
            [line.strip() for line in open(input.y)],
            [line.strip() for line in open(input.P)], [line.strip() for line in open(input.nu)],
            [line.strip() for line in open(input.ctnu)], fixed, random):
        print(y_f, P_f, nu_f, ctnu_f)
        out_f = re.sub('/rep/', f'/rep{gene}/', params.out)
        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        P = np.loadtxt(P_f)

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

        out = {'gene':gene}

        ## HE
        if 'HE_as_initial' not in snakemake.params.keys():
            snakemake.params.HE_as_initial = False
        if snakemake.params.HE_as_initial:
            snakemake.params.HE = True

        if snakemake.params.HE:
            hom_he, hom_he_p = cuomo_ctng_test.hom_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
            iid_he, iid_he_p = cuomo_ctng_test.iid_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
            free_he, free_he_p = cuomo_ctng_test.free_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
            full_he = cuomo_ctng_test.full_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)

            out['he'] = {'hom': hom_he, 'iid': iid_he, 'free': free_he, 'full': full_he, 
                    'wald':{'hom':hom_he_p, 'iid':iid_he_p, 'free':free_he_p}}

        ## ML
        #null_ml = null_ML(y_f, P_f, nu_f, fixed_covars_d)
        if snakemake.params.ML:
            if not snakemake.params.HE_as_initial:
                hom_ml, hom_ml_p = ctp_R.hom_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                iid_ml, iid_ml_p = ctp_R.iid_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                free_ml, free_ml_p = ctp_R.free_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                full_ml = ctp_R.full_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
            else:
                hom_ml, hom_ml_p = ctp_R.hom_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(hom_he, ML=True))
                iid_ml, iid_ml_p = ctp_R.iid_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(iid_he, ML=True))
                free_ml, free_ml_p = ctp_R.free_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(free_he, ML=True))
                full_ml = ctp_R.full_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(full_he, ML=True))
            
            out['ml'] = {'hom': hom_ml, 'iid': iid_ml, 'free': free_ml, 'full': full_ml,
                    'wald':{'hom':hom_ml_p, 'iid':iid_ml_p, 'free':free_ml_p}}

            # LRT
            C = np.loadtxt(y_f).shape[1]
            #hom_null_lrt = mystats.lrt(out['ml']['hom']['l'], out['ml']['null']['l'], 1)
            iid_hom_lrt = mystats.lrt(out['ml']['iid']['l'], out['ml']['hom']['l'], 1)
            free_hom_lrt = mystats.lrt(out['ml']['free']['l'], out['ml']['hom']['l'], C)
            free_iid_lrt = mystats.lrt(out['ml']['free']['l'], out['ml']['iid']['l'], C-1)
            full_hom_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['hom']['l'], C*(C+1)//2-1)
            full_iid_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['iid']['l'], C*(C+1)//2-2)
            full_free_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['free']['l'], C*(C+1)//2-C-1)

            out['ml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
                    'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
                    'full_free':full_free_lrt}

        if snakemake.params.REML:
            if not snakemake.params.HE_as_initial:
                hom_reml, hom_reml_p = ctp_R.hom_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                iid_reml, iid_reml_p = ctp_R.iid_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                free_reml, free_reml_p = ctp_R.free_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                full_reml = ctp_R.full_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
            else:
                hom_reml, hom_reml_p = ctp_R.hom_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(hom_he, REML=True))
                iid_reml, iid_reml_p = ctp_R.iid_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(iid_he, REML=True))
                free_reml, free_reml_p = ctp_R.free_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(free_he, REML=True))
                full_reml = ctp_R.full_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(full_he, REML=True))

            out['reml'] = {'hom':hom_reml, 'iid':iid_reml, 'free':free_reml, 'full':full_reml,
                    'wald':{'hom':hom_reml_p, 'iid':iid_reml_p, 'free':free_reml_p}}

            ## LRT
            C = np.loadtxt(y_f).shape[1]
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
        outs.append( out_f )

    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))

if __name__ == '__main__':

    main()

