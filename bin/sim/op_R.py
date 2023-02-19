import os, sys, re, time
import scipy
import numpy as np, pandas as pd
import rpy2.robjects as robjects 
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from rpy2.robjects.conversion import localconverter
from ctmm import wald, util, op

def main():
    # par
    params = snakemake.params
    input = snakemake.input
    output = snakemake.output
    optim_method = 'BFGS' if 'method' not in params.keys() else params.method

    #
    batch = params.batch
    outs = [re.sub('/rep/', f'/rep{i}/', params.out) for i in batch]
    for y_f, P_f, nu_f, out_f in zip([line.strip() for line in open(input.y)],
            [line.strip() for line in open(input.P)], [line.strip() for line in open(input.nu)], outs):
        print(y_f, P_f, nu_f)
        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape

        # project out cell type main effect from y
        #proj = np.eye(len(y)) - P @ np.linalg.inv(P.T @ P) @ P.T
        #y_p = proj @ y
        #Y = y_p**2 - np.diag(proj @ np.diag(vs) @ proj)

        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        out = {}
        ## HE
        if 'HE_as_initial' not in snakemake.params.keys():
            snakemake.params.HE_as_initial = False
        if snakemake.params.HE_as_initial:
            snakemake.params.HE = True

        if snakemake.params.HE:
            hom_he, hom_he_p = op.hom_HE(y_f, P_f, nu_f, jack_knife=True)
            free_he, free_he_p = op.free_HE(y_f, P_f, nu_f, jack_knife=True)
            full_he = op.full_HE(y_f, P_f, nu_f)

            out['he'] = {'hom': hom_he, 'free': free_he, 'full': full_he,
                    'wald':{'hom':hom_he_p, 'free':free_he_p}}

        ## ML
        if snakemake.params.ML:
            if not snakemake.params.HE_as_initial:
                hom_ml, hom_ml_p = op.hom_ML(y_f, P_f, nu_f, method=optim_method, optim_by_R=True)
                free_ml, free_ml_p = op.free_ML(y_f, P_f, nu_f, method=optim_method, optim_by_R=True)
                full_ml = op.full_ML(y_f, P_f, nu_f, method=optim_method, optim_by_R=True)
            else:
                hom_ml, hom_ml_p = op.hom_ML( y_f, P_f, nu_f, par=util.generate_HE_initial(hom_he, ML=True),
                        method=optim_method, optim_by_R=True)
                free_ml, free_ml_p = op.free_ML( y_f, P_f, nu_f, par=util.generate_HE_initial(free_he, ML=True),
                        method=optim_method, optim_by_R=True)
                full_ml = op.full_ML( y_f, P_f, nu_f, par=util.generate_HE_initial(full_he, ML=True),
                        method=optim_method, optim_by_R=True)

            out['ml'] = {'hom': hom_ml, 'free': free_ml, 'full': full_ml,
                    'wald':{'hom':hom_ml_p, 'free':free_ml_p}}

            # LRT
            free_hom_lrt = util.lrt(out['ml']['free']['l'], out['ml']['hom']['l'], C)
            full_hom_lrt = util.lrt(out['ml']['full']['l'], out['ml']['hom']['l'], C*(C+1)//2-1)
            full_free_lrt = util.lrt(out['ml']['full']['l'], out['ml']['free']['l'], C*(C+1)//2-C-1)

            out['ml']['lrt'] = {'free_hom':free_hom_lrt,
                    'full_hom':full_hom_lrt, 'full_free':full_free_lrt}

        ## REML
        if snakemake.params.REML:
            if 'Free_reml_only' not in snakemake.params.keys():
                snakemake.params.Free_reml_only = False

            if not snakemake.params.HE_as_initial:
                if 'Free_reml_jk' in snakemake.params.keys():
                    free_reml, free_reml_p = op.free_REML(y_f, P_f, nu_f, method=optim_method,
                            jack_knife=snakemake.params.Free_reml_jk, optim_by_R=True)
                else:
                    free_reml, free_reml_p = op.free_REML(y_f, P_f, nu_f, method=optim_method, 
                            optim_by_R=True)

                if snakemake.params.Free_reml_only:
                    hom_reml, hom_reml_p = free_reml, free_reml_p
                    full_reml = free_reml
                else:
                    hom_reml, hom_reml_p = op.hom_REML(y_f, P_f, nu_f, method=optim_method, 
                            optim_by_R=True)
                    full_reml = op.full_REML(y_f, P_f, nu_f, method=optim_method, 
                            optim_by_R=True)
            else:
                hom_reml, hom_reml_p = op.hom_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(hom_he, REML=True),
                        method=optim_method, optim_by_R=True)
                free_reml, free_reml_p = op.free_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(free_he, REML=True),
                        method=optim_method, optim_by_R=True)
                full_reml = op.full_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(full_he, REML=True),
                        method=optim_method, optim_by_R=True)

            out['reml'] = {'hom':hom_reml, 'free':free_reml, 'full':full_reml,
                    'wald':{'hom':hom_reml_p, 'free':free_reml_p}}

            ## LRT
            free_hom_lrt = util.lrt(out['reml']['free']['l'], out['reml']['hom']['l'], C)
            full_hom_lrt = util.lrt(out['reml']['full']['l'], out['reml']['hom']['l'], C*(C+1)//2-1)
            full_free_lrt = util.lrt(out['reml']['full']['l'], out['reml']['free']['l'], C*(C+1)//2-C-1)

            out['reml']['lrt'] = {'free_hom':free_hom_lrt,
                    'full_hom':full_hom_lrt, 'full_free':full_free_lrt}

        # save
        np.save(out_f, out)

    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))

if __name__ == '__main__':
    main()
