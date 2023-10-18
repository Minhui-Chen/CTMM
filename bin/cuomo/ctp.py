import os, sys, re, time
import scipy
import numpy as np, pandas as pd
from ctmm import wald, util, ctp, log


def collect_covariates(snakemake, donors=None):
    """
    Read covariates
    donors: list of individuals to keep for analysis
    """

    fixed_covars_d = {}
    random_covars_d = {}  # random effect only support homogeneous variance i.e. \sigma * I
    covars_f = util.generate_tmpfn()
    ## pca
    ### get individuals after filtering from pca result file
    pca = pd.read_table(snakemake.input.pca).sort_values(by='donor')
    if donors is not None:
        print(donors)
        pca = pca.loc[pca['donor'].isin(donors)]
    else:
        donors = np.array(pca['donor'])
        print(donors)
    if int(snakemake.wildcards.PC) > 0:
        pcs = [f'PC{i + 1}' for i in range(int(snakemake.wildcards.PC))]
        pca = pca[pcs]
        np.savetxt(covars_f + '.pca', np.array(pca))
        fixed_covars_d['pca'] = covars_f + '.pca'
    ## supp: sex disease
    if snakemake.wildcards.sex == 'Y':
        supp = pd.read_table(snakemake.input.supp, usecols=['donor_id_short', 'sex'])
        supp = supp.rename(columns={'donor_id_short': 'donor'})
        # remove the duplicated individual iisa
        supp = supp.drop_duplicates(subset='donor')
        supp = supp.loc[supp['donor'].isin(donors)]
        supp['code'] = 0
        supp.loc[supp['sex'] == 'male', 'code'] = 1 / (supp.loc[supp['sex'] == 'male'].shape[0])
        supp.loc[supp['sex'] == 'female', 'code'] = -1 / (supp.loc[supp['sex'] == 'female'].shape[0])
        np.savetxt(covars_f + '.sex', np.array(supp.sort_values(by='donor')['code']))
        fixed_covars_d['sex'] = covars_f + '.sex'
    if snakemake.wildcards.disease == 'Y':
        supp = pd.read_table(snakemake.input.supp, usecols=['donor_id_short', 'donor_disease_status'])
        supp = supp.rename(columns={'donor_id_short': 'donor', 'donor_disease_status': 'disease'})
        # remove the duplicated individual iisa
        supp = supp.drop_duplicates(subset='donor')
        supp = supp.loc[supp['donor'].isin(donors)]
        if len(np.unique(supp['disease'])) == 1:
            print('No disease')
        else:
            supp['code'] = 0
            supp.loc[supp['disease'] == 'normal', 'code'] = 1 / (supp.loc[supp['disease'] == 'normal'].shape[0])
            supp.loc[supp['disease'] == 'neonatal_diabetes', 'code'] = -1 / (
            supp.loc[supp['disease'] == 'neonatal_diabetes'].shape[0])
            np.savetxt(covars_f + '.disease', np.array(supp.sort_values(by='donor')['code']))
            fixed_covars_d['disease'] = covars_f + '.disease'
    ## meta: experiment
    if snakemake.wildcards.experiment in ['Y', 'R']:
        meta = pd.read_table(snakemake.input.meta, usecols=['donor', 'experiment'])
        meta = meta.loc[meta['donor'].isin(donors)]
        meta = meta.drop_duplicates().sort_values(by='donor').reset_index(drop=True)
        if meta.shape[0] != len(np.unique(meta['donor'])):
            print(meta[meta.duplicated(subset='donor', keep=False)])
            sys.exit('More than one experiments for an individual!\n')
        experiments = list(np.unique(meta['experiment']))
        if snakemake.wildcards.experiment == 'R':
            for experiment in experiments:
                meta[experiment] = 0
                meta.loc[meta['experiment'] == experiment, experiment] = 1
            np.savetxt(covars_f + '.experiment', np.array(meta[experiments]))
            random_covars_d['experiment'] = covars_f + '.experiment'
        else:
            for experiment in experiments[:-1]:
                meta[experiment] = 0
                meta.loc[meta['experiment'] == experiment, experiment] = 1 / (
                meta.loc[meta['experiment'] == experiment].shape[0])
                meta.loc[meta['experiment'] == experiments[-1], experiment] = -1 / (
                meta.loc[meta['experiment'] == experiments[-1]].shape[0])
            np.savetxt(covars_f + '.experiment', np.array(meta[experiments[:-1]]))
            fixed_covars_d['experiment'] = covars_f + '.experiment'

    return fixed_covars_d, random_covars_d


def main():
    # par
    optim_by_R = snakemake.params.get('optim_by_R', False)

    # collect covariates
    fixed_covars_d, random_covars_d = collect_covariates(snakemake)

    #
    genes = snakemake.params.genes
    outs = []
    for gene, y_f, P_f, nu_f, ctnu_f in zip(genes, [line.strip() for line in open(snakemake.input.imputed_ct_y)],
                                            [line.strip() for line in open(snakemake.input.P)],
                                            [line.strip() for line in open(snakemake.input.nu)],
                                            [line.strip() for line in open(snakemake.input.imputed_ct_nu)]):
        # if gene not in ['ENSG00000141448_GATA6', 'ENSG00000141506_PIK3R5']:
        #    continue
        print(y_f, P_f, nu_f, ctnu_f)
        out_f = re.sub('/rep/', f'/rep{gene}/', snakemake.params.out)
        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        # celltype specific mean nu
        ctnu = pd.read_table(ctnu_f)
        cts = np.unique(ctnu['day'])
        ctnu_grouped = ctnu.groupby('day')[gene].mean()

        # transform y and ctnu from vector to matrix
        tmp_f = util.generate_tmpfn()
        y = pd.read_table(y_f)
        y = y.pivot(index='donor', columns='day', values=gene)
        ctnu = ctnu.pivot(index='donor', columns='day', values=gene)
        y_f = tmp_f + '.y'
        ctnu_f = tmp_f + '.ctnu'
        y.to_csv(y_f, sep='\t', index=False, header=False)
        ctnu.to_csv(ctnu_f, sep='\t', index=False, header=False)

        # sanity check
        if y.index.equals(ctnu.index) and y.columns.equals(ctnu.columns):
            pass
        else:
            sys.exit('Not matching')

        # if there are individuals with more than 1 cts with ctnu =0 , hom and IID is gonna broken
        # so just skip these ~20 genes
        if 'Hom' not in snakemake.params.keys():
            snakemake.params.Hom = True
        if 'IID' not in snakemake.params.keys():
            snakemake.params.IID = True
        if np.any((ctnu < 1e-12).sum(axis=1) > 1):
            if snakemake.params.Hom or snakemake.params.IID:
                log.logger.info(gene)
                continue
            else:
                pass

        out = {'gene': gene, 'ct_mean_nu': {ct: ctnu_grouped.loc[ct] for ct in cts}}
        # HE
        HE_as_initial = snakemake.params.get('HE_as_initial', False)

        if HE_as_initial:
            snakemake.params.HE = True

        if snakemake.params.HE:
            jack_knife = snakemake.params.get('jack_knife', False)

            free_he, free_he_p = ctp.free_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                             jack_knife=jack_knife)
            if snakemake.params.Hom:
                hom_he, hom_he_p = ctp.hom_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                              jack_knife=jack_knife)
            else:
                hom_he, hom_he_p = {}, {}
            if snakemake.params.IID:
                iid_he, iid_he_p = ctp.iid_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                              jack_knife=jack_knife)
            else:
                iid_he, iid_he_p = {}, {}
            out['he'] = {'hom': hom_he, 'iid': iid_he, 'free': free_he,
                         'wald': {'hom': hom_he_p, 'iid': iid_he_p, 'free': free_he_p}}
            if 'Full_HE' not in snakemake.params.keys():
                full_he = ctp.full_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                out['he']['full'] = full_he
            else:
                if snakemake.params.Full_HE:
                    full_he = ctp.full_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                    out['he']['full'] = full_he

        ## ML
        # null_ml = null_ML(y_f, P_f, nu_f, fixed_covars_d)
        if snakemake.params.ML:
            if not HE_as_initial:
                free_ml, free_ml_p = ctp.free_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                 optim_by_R=optim_by_R)
                full_ml = ctp.full_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d, optim_by_R=optim_by_R)
                if snakemake.params.Hom:
                    hom_ml, hom_ml_p = ctp.hom_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                  optim_by_R=optim_by_R)
                else:
                    hom_ml, hom_ml_p = {}, {}
                if snakemake.params.IID:
                    iid_ml, iid_ml_p = ctp.iid_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                  optim_by_R=optim_by_R)
                else:
                    iid_ml, iid_ml_p = {}, {}
            else:
                free_ml, free_ml_p = ctp.free_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                 optim_by_R=optim_by_R,
                                                 par=util.generate_HE_initial(free_he, ML=True))
                full_ml = ctp.full_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d, optim_by_R=optim_by_R,
                                      par=util.generate_HE_initial(full_he, ML=True))
                if snakemake.params.Hom:
                    hom_ml, hom_ml_p = ctp.hom_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                  optim_by_R=optim_by_R,
                                                  par=util.generate_HE_initial(hom_he, ML=True))
                else:
                    hom_ml, hom_ml_p = {}, {}
                if snakemake.params.IID:
                    iid_ml, iid_ml_p = ctp.iid_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                  optim_by_R=optim_by_R,
                                                  par=util.generate_HE_initial(iid_he, ML=True))
                else:
                    iid_ml, iid_ml_p = {}, {}

            out['ml'] = {'hom': hom_ml, 'iid': iid_ml, 'free': free_ml, 'full': full_ml,
                         'wald': {'hom': hom_ml_p, 'iid': iid_ml_p, 'free': free_ml_p}}

            # LRT
            C = np.loadtxt(y_f).shape[1]
            # hom_null_lrt = util.lrt(out['ml']['hom']['l'], out['ml']['null']['l'], 1)
            #iid_hom_lrt = util.lrt(out['ml']['iid']['l'], out['ml']['hom']['l'], 1)
            #free_iid_lrt = util.lrt(out['ml']['free']['l'], out['ml']['iid']['l'], C - 1)
            #full_hom_lrt = util.lrt(out['ml']['full']['l'], out['ml']['hom']['l'], C * (C + 1) // 2 - 1)
            #full_iid_lrt = util.lrt(out['ml']['full']['l'], out['ml']['iid']['l'], C * (C + 1) // 2 - 2)
            #full_free_lrt = util.lrt(out['ml']['full']['l'], out['ml']['free']['l'], C * (C + 1) // 2 - C - 1)

            # out['ml']['lrt'] = {'iid_hom': iid_hom_lrt, 'free_hom': free_hom_lrt,
            #                     'free_iid': free_iid_lrt, 'full_hom': full_hom_lrt, 'full_iid': full_iid_lrt,
            #                     'full_free': full_free_lrt}
            if snakemake.params.Hom:
                out['ml']['lrt'] = {'free_hom': util.lrt(out['ml']['free']['l'], out['ml']['hom']['l'], C)}

        # REML
        if snakemake.params.REML:

            Free_reml_jk = snakemake.params.get('Free_reml_jk', False)

            if not HE_as_initial:
                free_reml, free_reml_p = ctp.free_REML(y_f, P_f, ctnu_f, nu_f,
                                                        fixed_covars_d, random_covars_d, optim_by_R=optim_by_R,
                                                        jack_knife=Free_reml_jk)

                full_reml = ctp.full_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                          optim_by_R=optim_by_R)

                if snakemake.params.Hom:
                    hom_reml, hom_reml_p = ctp.hom_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                        optim_by_R=optim_by_R)
                else:
                    hom_reml, hom_reml_p = {}, {}
                if snakemake.params.IID:
                    iid_reml, iid_reml_p = ctp.iid_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                        optim_by_R=optim_by_R)
                else:
                    iid_reml, iid_reml_p = {}, {}
            else:
                free_reml, free_reml_p = ctp.free_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                       optim_by_R=optim_by_R,
                                                       par=util.generate_HE_initial(free_he, REML=True))
                full_reml = ctp.full_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                          optim_by_R=optim_by_R,
                                          par=util.generate_HE_initial(full_he, REML=True))

                if snakemake.params.Hom:
                    hom_reml, hom_reml_p = ctp.hom_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                        optim_by_R=optim_by_R,
                                                        par=util.generate_HE_initial(hom_he, REML=True))
                else:
                    hom_reml, hom_reml_p = {}, {}
                if snakemake.params.IID:
                    iid_reml, iid_reml_p = ctp.iid_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                                                        optim_by_R=optim_by_R,
                                                        par=util.generate_HE_initial(iid_he, REML=True))
                else:
                    iid_reml, iid_reml_p = {}, {}

            out['reml'] = {'hom': hom_reml, 'iid': iid_reml, 'free': free_reml, 'full': full_reml,
                           'wald': {'hom': hom_reml_p, 'iid': iid_reml_p, 'free': free_reml_p}}

            ## LRT
            C = np.loadtxt(y_f).shape[1]
            # iid_hom_lrt = util.lrt(out['reml']['iid']['l'], out['reml']['hom']['l'], 1)
            # free_iid_lrt = util.lrt(out['reml']['free']['l'], out['reml']['iid']['l'], C - 1)
            # full_hom_lrt = util.lrt(out['reml']['full']['l'], out['reml']['hom']['l'], C * (C + 1) // 2 - 1)
            # full_iid_lrt = util.lrt(out['reml']['full']['l'], out['reml']['iid']['l'], C * (C + 1) // 2 - 2)
            # full_free_lrt = util.lrt(out['reml']['full']['l'], out['reml']['free']['l'], C * (C + 1) // 2 - C - 1)

            # out['reml']['lrt'] = {'iid_hom': iid_hom_lrt, 'free_hom': free_hom_lrt,
            #                       'free_iid': free_iid_lrt, 'full_hom': full_hom_lrt, 'full_iid': full_iid_lrt,
                                #   'full_free': full_free_lrt}
            if snakemake.params.Hom:
                out['reml']['lrt'] = {'free_hom': util.lrt(out['reml']['free']['l'], out['reml']['hom']['l'], C)}

        # save
        np.save(out_f, out)
        outs.append(out_f)

    # sys.exit('END')
    with open(snakemake.output.out, 'w') as f:
        f.write('\n'.join(outs))

    log.logger.info('Finished')


if __name__ == '__main__':
    main()
