import os, sys, re
import numpy as np
from ctmm import util, wald, ctp, log


def main():
    # par
    params = snakemake.params
    input = snakemake.input
    output = snakemake.output
    optim_by_R = params.get('optim_by_R', False)
    method = params.get('method', None)

    #
    batch = params.batch
    outs = [re.sub('/rep/', f'/rep{i}/', params.out) for i in batch]

    y_fs = [line.strip() for line in open(input.y)]
    P_fs = [line.strip() for line in open(input.P)]

    if input.nu[-4:] == '.npy':
        nus = np.load(input.nu, allow_pickle=True).item()
        tmpfn = util.generate_tmpfn()
        nu_fs = []
        for i in sorted(nus.keys()):
            f = f'{tmpfn}.ctnu.{i}'
            np.savetxt(f, nus[i])
            nu_fs.append(f)
    else:
        nu_fs = [line.strip() for line in open(input.nu)]
    
    collection = []
    for i in range(len(y_fs)):
        y_f, P_f, nu_f, out_f = y_fs[i], P_fs[i], nu_fs[i], outs[i]
        log.logger.info(f'{y_f}, {P_f}, {nu_f}')

        # cell type number
        C = np.loadtxt(y_f).shape[1]

        os.makedirs(os.path.dirname(out_f), exist_ok=True)
        
        # Dictionary to store output
        out = {}

        # HE
        ## use HE as initial for ML/REML
        HE_as_initial = params.get('HE_as_initial', False)
        if HE_as_initial:
            snakemake.params.HE = True

        if snakemake.params.HE:
            he_free_jk = params.get('he_free_jk', True)
            he_iid_jk = params.get('he_iid_jk', False)
            free_he, free_he_wald = ctp.free_HE(y_f, P_f, nu_f, jack_knife=he_free_jk)
            HE_free_only = params.get('HE_free_only', False)
            if HE_free_only:
                out['he'] = { 'free': free_he, 'wald':{'free': free_he_wald} }
            else:
                hom_he, hom_he_wald = ctp.hom_HE(y_f, P_f, nu_f, jack_knife=False)
                iid_he, iid_he_wald = ctp.iid_HE(y_f, P_f, nu_f, jack_knife=he_iid_jk)
                full_he = ctp.full_HE(y_f, P_f, nu_f)

                out['he'] = {'hom': hom_he, 'iid': iid_he, 'free': free_he, 'full': full_he,
                        'wald':{'hom':hom_he_wald, 'iid': iid_he_wald, 'free': free_he_wald} }

        # ML
        if snakemake.params.ML:
            if not HE_as_initial:
                hom_ml, hom_ml_wald = ctp.hom_ML(y_f, P_f, nu_f, method=method, optim_by_R=optim_by_R)
                iid_ml, iid_ml_wald = ctp.iid_ML(y_f, P_f, nu_f, method=method, optim_by_R=optim_by_R)
                free_ml, free_ml_wald = ctp.free_ML(y_f, P_f, nu_f, method=method, optim_by_R=optim_by_R)
                full_ml = ctp.full_ML(y_f, P_f, nu_f, method=method, optim_by_R=optim_by_R)
            else:
                hom_ml, hom_ml_wald = ctp.hom_ML(y_f, P_f, nu_f, method=method, 
                        par=util.generate_HE_initial(hom_he, ML=True), optim_by_R=optim_by_R )
                iid_ml, iid_ml_wald = ctp.iid_ML(y_f, P_f, nu_f, method=method, 
                        par=util.generate_HE_initial(iid_he, ML=True), optim_by_R=optim_by_R )
                free_ml, free_ml_wald = ctp.free_ML(y_f, P_f, nu_f, method=method, 
                        par=util.generate_HE_initial(free_he, ML=True), optim_by_R=optim_by_R )
                full_ml = ctp.full_ML(y_f, P_f, nu_f, method=method, 
                        par=util.generate_HE_initial(full_he, ML=True), optim_by_R=optim_by_R )

            out['ml'] = {'hom': hom_ml, 'iid': iid_ml, 'free': free_ml, 'full': full_ml,
                    'wald':{'hom':hom_ml_wald, 'iid':iid_ml_wald, 'free':free_ml_wald} }

            # LRT
            iid_hom_lrt = util.lrt(out['ml']['iid']['l'], out['ml']['hom']['l'], 1)
            free_hom_lrt = util.lrt(out['ml']['free']['l'], out['ml']['hom']['l'], C)
            free_iid_lrt = util.lrt(out['ml']['free']['l'], out['ml']['iid']['l'], C-1)
            full_hom_lrt = util.lrt(out['ml']['full']['l'], out['ml']['hom']['l'], C*(C+1)//2-1)
            full_iid_lrt = util.lrt(out['ml']['full']['l'], out['ml']['iid']['l'], C*(C+1)//2-2)
            full_free_lrt = util.lrt(out['ml']['full']['l'], out['ml']['free']['l'], C*(C+1)//2-C-1)

            out['ml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
                    'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
                    'full_free':full_free_lrt}

        ## REML
        if snakemake.params.REML:
            Free_reml_jk = snakemake.params.get('Free_reml_jk', False)

            if not HE_as_initial:
                free_reml, free_reml_wald = ctp.free_REML(y_f, P_f, nu_f, method=method,  
                        jack_knife=Free_reml_jk, optim_by_R=optim_by_R)
                hom_reml, hom_reml_wald = ctp.hom_REML(y_f, P_f, nu_f, method=method, optim_by_R=optim_by_R)
                iid_reml, iid_reml_wald = ctp.iid_REML(y_f, P_f, nu_f, method=method, optim_by_R=optim_by_R)
                full_reml = ctp.full_REML(y_f, P_f, nu_f, method=method, optim_by_R=optim_by_R)
            else:
                hom_reml, hom_reml_wald = ctp.hom_REML(y_f, P_f, nu_f, method=method, 
                        par=util.generate_HE_initial(hom_he, REML=True), optim_by_R=optim_by_R)
                iid_reml, iid_reml_wald = ctp.iid_REML(y_f, P_f, nu_f, method=method, 
                        par=util.generate_HE_initial(iid_he, REML=True), optim_by_R=optim_by_R)
                free_reml, free_reml_wald = ctp.free_REML(y_f, P_f, nu_f, method=method, 
                        par=util.generate_HE_initial(free_he,REML=True), optim_by_R=optim_by_R)
                full_reml = ctp.full_REML(y_f, P_f, nu_f, method=method, 
                        par=util.generate_HE_initial(full_he, REML=True), optim_by_R=optim_by_R)

            out['reml'] = {'hom':hom_reml, 'iid':iid_reml, 'free':free_reml, 'full':full_reml,
                    'wald':{'hom':hom_reml_wald, 'iid':iid_reml_wald, 'free':free_reml_wald} }

            ## REML
            iid_hom_lrt = util.lrt(out['reml']['iid']['l'], out['reml']['hom']['l'], 1)
            free_hom_lrt = util.lrt(out['reml']['free']['l'], out['reml']['hom']['l'], C)
            free_iid_lrt = util.lrt(out['reml']['free']['l'], out['reml']['iid']['l'], C-1)
            full_hom_lrt = util.lrt(out['reml']['full']['l'], out['reml']['hom']['l'], C*(C+1)//2-1)
            full_iid_lrt = util.lrt(out['reml']['full']['l'], out['reml']['iid']['l'], C*(C+1)//2-2)
            full_free_lrt = util.lrt(out['reml']['full']['l'], out['reml']['free']['l'], C*(C+1)//2-C-1)

            out['reml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
                    'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
                    'full_free':full_free_lrt}

        # save
        np.save(out_f, out)
        collection.append(out)

    if output.out[-4:] == '.npy':
        np.save(output.out, collection)
    else:
        with open(output.out, 'w') as f:
            f.write('\n'.join(outs))

if __name__ == '__main__':
    main()

