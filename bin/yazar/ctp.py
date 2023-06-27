import os, sys, re, time
import scipy
import numpy as np, pandas as pd
from ctmm import wald, util, ctp, log


def collect_covariates(snakemake, inds=None):
    '''
    Read covariates
    donors: list of individuals to keep for analysis
    '''

    covars_f = util.generate_tmpfn()
    ## pca
    ### get individuals after filtering from pca result file
    pca = pd.read_table(snakemake.input.pca, index_col=0)
    meta = pd.read_table(snakemake.input.meta, usecols=['individual', 'sex', 'age'])
    meta = meta.drop_duplicates()
    meta = meta.set_index('individual')
    pca_f = covars_f + '.pca'  # TODO: currently don't include genotype PC
    sex_f = covars_f + '.sex'
    age_f = covars_f + '.age'
    np.savetxt(pca_f, util.design(inds, pca=pca, PC=1))
    np.savetxt(sex_f, util.design(inds, cat=meta['sex']))
    np.savetxt(age_f, util.design(inds, cat=util.age_group(meta['age'])))
    fixed_covars = {'pca': pca_f, 'sex': sex_f, 'age': age_f}

    return fixed_covars


def main():
    # par
    params = snakemake.params
    input = snakemake.input
    output = snakemake.output
    wildcards = snakemake.wildcards

    # read
    cty = pd.read_table(input.ctp)
    ctnu = pd.read_table(input.ctnu)
    cts = np.unique(ctnu['ct'])
    # celltype specific mean nu
    ctnu_grouped = ctnu.groupby('ct').mean()

    # P
    P = pd.read_table(input.P, index_col=0)
    inds = P.index.to_numpy()  # order of inds
    tmp_f = util.generate_tmpfn()
    P_f = f'{tmp_f}.P'
    P.to_csv(P_f, sep='\t', index=False, header=False)

    # collect covariates
    fixed_covars_d = collect_covariates(snakemake, inds)

    #
    genes = params.genes
    outs = []
    for gene in genes:
        log.logger.info(gene)
        out_f = re.sub('/rep/', f'/rep{gene}/', params.out)
        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        # transform y and ctnu from vector to matrix
        g_cty = cty.pivot(index='ind', columns='ct', values=gene)
        g_ctnu = ctnu.pivot(index='ind', columns='ct', values=gene)

        y_f = f'{tmp_f}.{gene}.y'
        ctnu_f = f'{tmp_f}.{gene}.ctnu'
        g_cty.to_csv(y_f, sep='\t', index=False, header=False)
        g_ctnu.to_csv(ctnu_f, sep='\t', index=False, header=False)

        # sanity check
        if g_cty.index.equals(P.index) and g_ctnu.index.equals(P.index) and g_cty.columns.equals(
                P.columns) and g_ctnu.columns.equals(P.columns):
            pass
        else:
            sys.exit('Not matching')

        # if there are individuals with more than 1 cts with ctnu =0 , hom and IID is gonna broken
        # so just skip these ~20 genes
        # if np.any( (ctnu < 1e-12).sum(axis=1) > 1 ):
        #    print(gene)
        #    if snakemake.params.Hom or snakemake.params.IID:
        #        continue
        #    else:
        #        pass

        out = {'gene': gene, 'ct_mean_nu': {ct: ctnu_grouped.loc[ct, gene] for ct in cts}}
        # HE
        if params.test == 'he':
            free_he, free_he_p = ctp.free_HE(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
                                             jack_knife=True)
            full_he = ctp.full_HE(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d)
            out['he'] = {'free': free_he, 'full': full_he,
                         'wald': {'free': free_he_p}}

        ## ML
        if params.test == 'ml':
            free_ml, free_ml_p = ctp.free_ML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
                                             optim_by_R=True)
            # full_ml = ctp.full_ML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
            #        optim_by_R=True)
            # hom_ml, hom_ml_p = ctp.hom_ML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
            #        optim_by_R=True)
            # iid_ml, iid_ml_p = ctp.iid_ML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
            #        optim_by_R=True)
            # out['ml'] = {'hom': hom_ml, 'iid': iid_ml, 'free': free_ml, 'full': full_ml,
            #        'wald':{'hom':hom_ml_p, 'iid':iid_ml_p, 'free':free_ml_p}}
            out['ml'] = {'free': free_ml,
                         'wald': {'free': free_ml_p}}

            # LRT
            # C = np.loadtxt(y_f).shape[1]
            # iid_hom_lrt = util.lrt(out['ml']['iid']['l'], out['ml']['hom']['l'], 1)
            # free_hom_lrt = util.lrt(out['ml']['free']['l'], out['ml']['hom']['l'], C)
            # free_iid_lrt = util.lrt(out['ml']['free']['l'], out['ml']['iid']['l'], C-1)
            # full_hom_lrt = util.lrt(out['ml']['full']['l'], out['ml']['hom']['l'], C*(C+1)//2-1)
            # full_iid_lrt = util.lrt(out['ml']['full']['l'], out['ml']['iid']['l'], C*(C+1)//2-2)
            # full_free_lrt = util.lrt(out['ml']['full']['l'], out['ml']['free']['l'], C*(C+1)//2-C-1)

            # out['ml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
            #        'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
            #        'full_free':full_free_lrt}

        # REML
        if params.test == 'reml':
            model = snakemake.params.get('model')
            jk = snakemake.params.get('jk', False)
            
            if model:
                if model == 'free':
                    free_reml, free_reml_p = ctp.free_REML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
                                                        optim_by_R=True, jack_knife=jk)
                    out['reml'] = {'free': free_reml, 'wald': {'free': free_reml_p}}
                elif model == 'full':
                    full_reml = ctp.full_REML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
                                            optim_by_R=True)
                    out['reml'] = {'full': full_reml}

            else:
                free_reml, free_reml_p = ctp.free_REML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
                                                    optim_by_R=True, jack_knife=jk)
                full_reml = ctp.full_REML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
                                          optim_by_R=True)
                out['reml'] = {'free': free_reml, 'full': full_reml,
                            'wald': {'free': free_reml_p}}
            # hom_reml, hom_reml_p = ctp.hom_REML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
            #        optim_by_R=True)
            # iid_reml, iid_reml_p = ctp.iid_REML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
            #        optim_by_R=True)
            # out['reml'] = {'hom':hom_reml, 'iid':iid_reml, 'free':free_reml, 'full':full_reml,
            #        'wald':{'hom':hom_reml_p, 'iid':iid_reml_p, 'free':free_reml_p}}

            ## LRT
            # C = np.loadtxt(y_f).shape[1]
            # iid_hom_lrt = util.lrt(out['reml']['iid']['l'], out['reml']['hom']['l'], 1)
            # free_hom_lrt = util.lrt(out['reml']['free']['l'], out['reml']['hom']['l'], C)
            # free_iid_lrt = util.lrt(out['reml']['free']['l'], out['reml']['iid']['l'], C-1)
            # full_hom_lrt = util.lrt(out['reml']['full']['l'], out['reml']['hom']['l'], C*(C+1)//2-1)
            # full_iid_lrt = util.lrt(out['reml']['full']['l'], out['reml']['iid']['l'], C*(C+1)//2-2)
            # full_free_lrt = util.lrt(out['reml']['full']['l'], out['reml']['free']['l'], C*(C+1)//2-C-1)

            # out['reml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
            #        'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
            #        'full_free':full_free_lrt}

        # save
        np.save(out_f, out)
        outs.append(out_f)

        # TODO
        sys.exit()

    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))


if __name__ == '__main__':
    main()
