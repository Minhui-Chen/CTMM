import os, sys, re, time
import scipy
import numpy as np, pandas as pd
from ctmm import wald, util, ctp, log
from gxctmm import util as gxctutil


def collect_covariates(snakemake, inds=None):
    '''
    Read covariates
    donors: list of individuals to keep for analysis
    '''

    fixed_covars_d = {}
    covars_f = util.generate_tmpfn()
    ## pca
    ### get individuals after filtering from pca result file
    pca = pd.read_table(snakemake.input.pca, index_col=0)
    meta = pd.read_table(snakemake.input.meta, usecols=['individual', 'sex', 'age'])
    meta = meta.drop_duplicates()
    meta = meta.set_index('individual')
    pca_f = covars_f + '.pca'
    sex_f = covars_f + '.sex'
    age_f = covars_f + '.age'
    np.savetxt(pca_f, gxctutil.design(inds, pca=pca, PC=1))
    np.savetxt(sex_f, gxctutil.design(inds, cat=meta['sex']))
    np.savetxt(age_f, gxctutil.design(inds, cat=gxctutil.age_group(meta['age'])))
    fixed_covars = {'pca': pca_f, 'sex': sex_f, 'age': age_f}

    return fixed_covars_d


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
        # if gene not in ['ENSG00000141448_GATA6', 'ENSG00000141506_PIK3R5']:
        #    continue
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

        # if there are individuals with more than 1 cts with ctnu =0 , hom and IID is gonna broken
        # so just skip these ~20 genes
        # if np.any( (ctnu < 1e-12).sum(axis=1) > 1 ):
        #    print(gene)
        #    if snakemake.params.Hom or snakemake.params.IID:
        #        continue
        #    else:
        #        pass

        out = {'gene': gene, 'ct_mean_nu': {ct: ctnu_grouped.loc[ct, gene] for ct in cts}}

        # REML
        full_reml = ctp.full_REML(y_f, P_f, ctnu_f, fixed_covars_d=fixed_covars_d,
                                  optim_by_R=True)

        out['reml'] = {'full': full_reml}

        # save
        np.save(out_f, out)
        outs.append(out_f)

    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))


if __name__ == '__main__':
    main()
