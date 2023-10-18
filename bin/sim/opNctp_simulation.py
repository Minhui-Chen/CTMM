import os, re, sys, math
import numpy as np, pandas as pd
from scipy import stats
from ctmm import log


def identifier_outliers(data, lower_percentile=10, upper_percentile=90, threshold=20):
    # Calculate the lower and upper quartiles (percentiles)
    lower_quartile = np.percentile(data, lower_percentile, axis=0)
    upper_quartile = np.percentile(data, upper_percentile, axis=0)

    # Calculate the interquartile range (IQR)
    iqr = upper_quartile - lower_quartile

    # Calculate the lower and upper bounds for outliers
    lower_bound = lower_quartile - (threshold * iqr)
    upper_bound = upper_quartile + (threshold * iqr)

    # Find the indices of the outliers
    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))

    return outlier_indices

def add_fixed(ss):
    ''' Add a test fixed effect'''
    levels = int(snakemake.wildcards.fixed)
    b = np.arange(levels)
    b = b / np.std(b) # to shrink
    X = np.zeros((ss, levels))
    for i, chunk in enumerate(np.array_split(np.arange(ss), levels)):
        X[chunk, i] = 1
    # centralize b
    b = b - np.mean(X @ b)

    return X, b

def add_random(ss):
    ''' Add a test random effect'''
    levels = int(snakemake.wildcards.random)
    rng = np.random.default_rng()
    b = rng.normal(0, 1, levels)
    X = np.zeros((ss, levels))
    for i, chunk in enumerate(np.array_split(np.arange(ss), levels)):
        X[chunk,i] = 1

    return X, b

def main():
    # par
    beta = np.loadtxt(snakemake.input.beta)
    V = np.loadtxt(snakemake.input.V)
    hom2 = float(snakemake.wildcards.vc.split('_')[0]) # variance of individual effect
    mean_nu = float(snakemake.wildcards.vc.split('_')[-1]) # mean variance for residual error acros individuals
    var_nu = float(snakemake.wildcards.var_nu) #variance of variance for residual error across individuals
    a = np.array(snakemake.wildcards.a.split('_')).astype('float')
    ss = int(float(snakemake.wildcards.ss))
    C = len(a)


    P_fs, pi_fs, s_fs, nu_fs, ctnu_fs, y_fs, cty_fs = [],[],[],[],[],[],[]
    fixed_X_fs, random_X_fs = [], []
    for k in snakemake.params.batch:
        log.logger.info(f'{k}')
        seed = snakemake.params.seed + int(k)
        rng = np.random.default_rng(seed)

        P_f = re.sub('repX', 'rep'+str(k), snakemake.params.P)
        P_fs.append( P_f )
        pi_f = re.sub('repX', 'rep'+str(k), snakemake.params.pi)
        pi_fs.append( pi_f )
        s_f = re.sub('repX', 'rep'+str(k), snakemake.params.s)
        s_fs.append( s_f )
        nu_f = re.sub('repX', 'rep'+str(k), snakemake.params.nu)
        nu_fs.append( nu_f )
        ctnu_f = re.sub('repX', 'rep'+str(k), snakemake.params.ctnu)
        ctnu_fs.append( ctnu_f )
        y_f = re.sub('repX', 'rep'+str(k), snakemake.params.y)
        y_fs.append( y_f )
        cty_f = re.sub('repX', 'rep'+str(k), snakemake.params.cty)
        cty_fs.append( cty_f )
        fixed_X_f = re.sub('repX', 'rep'+str(k), snakemake.params.fixed)
        fixed_X_fs.append( fixed_X_f )
        random_X_f = re.sub('repX', 'rep'+str(k), snakemake.params.random)
        random_X_fs.append( random_X_f )
        
        os.makedirs(os.path.dirname(P_f), exist_ok=True)

        j = 0
        while True:
            # simulate cell type proportions
            P = rng.dirichlet(alpha=a, size=ss)
            if 'P_cut' in snakemake.params.keys():
                P_cut = float(snakemake.params.P_cut)
                if np.amin(P) < P_cut:
                    P[P < P_cut] = P_cut
                    P = P / P.sum(axis=1, keepdims=True)

            np.savetxt(P_f, P, delimiter='\t')
            pi = np.mean(P, axis=0)
            np.savetxt(pi_f, pi, delimiter='\t')

            # estimate cov matrix S
            ## demeaning P
            pd = P-pi
            ## covariance
            s = (pd.T @ pd) / ss
            np.savetxt(s_f, s, delimiter='\t')

            # draw alpha: hom random effect
            alpha = rng.normal(loc=0, scale=math.sqrt(hom2), size=ss)

            # draw noise variance
            ## draw variance of noise for each individual from Gamma(k, theta)
            ## with mean = k * theta, var = k * theta^2, so theta = var / mean, k = mean / theta
            ## since mean = 0.25 and assume var = 0.01, we can get k and theta
            if mean_nu != 0:
                theta = var_nu / mean_nu 
                k = mean_nu / theta
                ### variance of residual error for each individual
                nu = rng.gamma(k, scale=theta, size=ss)

                ## if one individual has more than two CTs of low nu,
                ## hom model breaks because of singular variance matrix,
                ## in this case, regenerate nu
                threshold = 1e-10
                i = 1
                while np.any( nu < threshold ):
                    nu = rng.gamma(k, scale=theta, size=ss)
                    i += 1
                    if i > 5:
                        sys.exit('Generate NU failed!\n')
            else:
                nu = np.zeros(ss)
            np.savetxt(nu_f, nu)

            # for ctp
            P_inv = 1 / P
            ctnu = P_inv * nu.reshape(-1,1)

            # draw residual error from normal distribution with variance drawn above
            ct_delta = rng.normal(np.zeros_like(ctnu), np.sqrt(ctnu))

            # generate pseudobulk
            cty = alpha[:, np.newaxis] + beta[np.newaxis, :] + ct_delta

            if snakemake.wildcards.model != 'hom':
                # draw CT-specific random effect
                gamma = rng.multivariate_normal(np.zeros(C), V, size=ss)
                cty += gamma

            break
            # sanity check: crazy outliers
            # outliers = identifier_outliers(cty)
            # if len(outliers[0]) == 0:
            #     break
            # else:
            #     log.logger.info(f'Outlier: {" ".join(np.var(cty, axis=0).astype("str"))}')
            #     j += 1
            #     if j == 3:
            #         sys.exit('Too strict')

        ## add a test fixed effect
        if 'fixed' in snakemake.wildcards.keys():
            if int( snakemake.wildcards.fixed ) > 0:
                X, b = add_fixed(ss)
                cty = cty + np.outer( X @ b, np.ones(C) )
                np.savetxt(fixed_X_f, X[:,:-1])

        ### add a test random effect
        if 'random' in snakemake.wildcards.keys():
            if int( snakemake.wildcards.random ) > 0:
                X, b = add_random(ss)
                cty = cty + np.outer( X @ b, np.ones(C) )
                np.savetxt(random_X_f, X)

        # overall pseudo
        y = (P * cty).sum(axis=1)

        # save
        np.savetxt(ctnu_f, ctnu)
        np.savetxt(y_f, y)
        np.savetxt(cty_f, cty)

    with open(snakemake.output.P, 'w') as f: f.write('\n'.join(P_fs))
    with open(snakemake.output.pi, 'w') as f: f.write('\n'.join(pi_fs))
    with open(snakemake.output.s, 'w') as f: f.write('\n'.join(s_fs))
    with open(snakemake.output.y, 'w') as f: f.write('\n'.join(y_fs))
    with open(snakemake.output.nu, 'w') as f: f.write('\n'.join(nu_fs))
    with open(snakemake.output.cty, 'w') as f: f.write('\n'.join(cty_fs))
    with open(snakemake.output.ctnu, 'w') as f: f.write('\n'.join(ctnu_fs))
    with open(snakemake.output.fixed, 'w') as f: 
        if 'fixed' in snakemake.wildcards.keys():
            if int(snakemake.wildcards.fixed) > 0:
                f.write('\n'.join(fixed_X_fs))
            else:
                f.write('NA')
        else:
            f.write('NA')
    with open(snakemake.output.random, 'w') as f: 
        if 'random' in snakemake.wildcards.keys():
            if int( snakemake.wildcards.random ) > 0:
                f.write('\n'.join(random_X_fs))
            else:
                f.write('NA')
        else:
            f.write('NA')

if __name__ == '__main__':
    main()
