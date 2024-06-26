import helper, os, multiprocessing
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr, STAP
from rpy2.robjects.conversion import localconverter

def softimpute(y, scale, out={}):
    softImpute_f = 'bin/my_softImpute.R'
    softImpute_r = STAP( open(softImpute_f).read(), 'softImpute_r' )
    if scale:
        out_ = softImpute_r.my_softImpute( r['as.matrix'](y), scale=ro.vectors.BoolVector([True]) )
    else:
        out_ = softImpute_r.my_softImpute( r['as.matrix'](y) )
    out_ = dict( zip(out_.names, list(out_)) )
    out['y'] = pd.DataFrame(out_['Y'], index=y.index, columns=y.columns)

def mvn(y, out={}):
    mvn_f = 'bin/mvn_impute.R'
    mvn_r = STAP( open(mvn_f).read(), 'mvn_r' )
    out_ = mvn_r.MVN_impute( r['as.matrix'](y) )
    out_ = dict( zip(out_.names, list(out_)) )
    out['y'] = pd.DataFrame(out_['Y'], index=y.index, columns=y.columns)

def imputeY4eachgene(y_, gene, y_path):
    y_before = y_
    os.makedirs(f'{y_path}/rep{gene}', exist_ok=True)
    y_.to_csv(f'{y_path}/rep{gene}/y.before_imputation.txt', sep='\t')

    manager = multiprocessing.Manager()
    out = manager.dict()
    if snakemake.wildcards.im_mvn == 'N':
        # python doesn't know memory used by rpy2
        # put the r process in a subprocess, so it would free used memory after terminate
        if snakemake.wildcards.im_scale == 'Y':
            p = multiprocessing.Process(target=softimpute, args=(y_, True, out))
        else:
            p = multiprocessing.Process(target=softimpute, args=(y_, False, out))
    else:
        p = multiprocessing.Process(target=mvn, args=(y_, out))

    p.start()
    p.join()

    y_ = out['y']
    y_.to_csv(f'{y_path}/rep{gene}/y.imputed.txt', sep='\t')

    ### make imputation heatmap
#    fig, axes = plt.subplots(ncols=2, figsize=(12,48))
#    vmin = np.nanmin( (y_before, y_) )
#    vmax = np.nanmax( (y_before, y_) )
#    sns.heatmap(y_before, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axes[0])
#    annotation = y_.astype('str')
#    annotation[y_ > 0] = ''
#    sns.heatmap(y_, annot=annotation, fmt='s', cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axes[1])
#    fig.savefig(f'{y_path}/rep{gene}/y.imputation.heatmap.png')
#    plt.close()

    #if snakemake.wildcards.im_miny == 'N':
    #    pass
    #elif snakemake.wildcards.im_miny == 'min':
    #    for day in y_.columns:
    #        min_y = np.nanmin(y_before[day])
    #        y_.loc[y_[day] < min_y, day] = min_y
    #else:
    #    y_[y_ < float(snakemake.wildcards.im_miny)] = float(snakemake.wildcards.im_miny) 
    #y_.to_csv(f'{y_path}/rep{gene}/y.cutted.txt', sep='\t')

    return( y_ )

def imputeNU4eachgene(nu_, nu_path, gene):
    nu_before = nu_

    os.makedirs(f'{nu_path}/rep{gene}', exist_ok=True)
    nu_.to_csv(f'{nu_path}/rep{gene}/nu.before_imputation.txt', sep='\t')

    if snakemake.wildcards.im_mvn == 'N':
        if snakemake.wildcards.im_scale == 'Y':
            os.system(f'Rscript bin/my_softImpute.R {nu_path}/rep{gene}/nu.before_imputation.txt {nu_path}/rep{gene}/nu.imputed.txt scale')
        else:
            os.system(f'Rscript bin/my_softImpute.R {nu_path}/rep{gene}/nu.before_imputation.txt {nu_path}/rep{gene}/nu.imputed.txt')
    else:
        os.system(f'Rscript bin/mvn_impute.R {nu_path}/rep{gene}/nu.before_imputation.txt {nu_path}/rep{gene}/nu.imputed.txt')

    nu_ = pd.read_table(f'{nu_path}/rep{gene}/nu.imputed.txt')
    nu_.index.name = 'donor'

    ### make imputation heatmap
#    fig, axes = plt.subplots(ncols=2, figsize=(12,48))
#    vmin = np.nanmin( (nu_before, nu_) )
#    vmax = np.nanmax( (nu_before, nu_) )
#    sns.heatmap(nu_before, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axes[0])
#    annotation = nu_.astype('str')
#    annotation[nu_ > 0] = ''
#    sns.heatmap(nu_, annot=annotation, fmt='s', cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axes[1])
#    fig.savefig(nu_path+f'/rep{gene}/nu.imputation.heatmap.png')
#    plt.close()

    return( nu_ )

def main():
    pandas2ri.activate()
    numpy2ri.activate()
    # par
    input = snakemake.input
    output = snakemake.output

    # read 
    P = pd.read_table(input.P, index_col=0) # donor * day
    P.index.name = 'donor'
    n = pd.read_table(input.n, index_col=0) # donor * day
    n.index.name = 'donor'
    genes = [line.strip() for line in open(input.y_batch)]
    # collect days and inds and genes

#    if snakemake.wildcards.im_genome == 'Y':
#        y = pd.read_table(input.genome_imputed_y, index_col=(0,1)) # donor - day * gene
#        nu = pd.read_table(input.genome_imputed_nu, index_col=(0,1)) # donor - day * gene
#    else:
    y = pd.read_table(input.y, index_col=(0,1))[genes] # donor - day * gene
    nu = pd.read_table(input.nu, index_col=(0,1))[genes] # donor - day * gene
    y = y.sort_values(by=['donor','day']).reset_index()
    nu = nu.sort_values(by=['donor','day']).reset_index()

    if snakemake.wildcards.im_mvn == 'D':
        # delete individuals with missing data (so no imputation)
        P = P.loc[P.index.isin(y['donor'])]
        n = n.loc[n.index.isin(y['donor'])]

    days = list(np.unique(P.columns))
    inds = list(np.unique(P.index))

    # sort
    P = P.sort_values(by='donor')[days]
    n = n.sort_values(by='donor')[days]

    y_path = os.path.dirname(output.y)
    nu_path = os.path.dirname(output.nu)

    y_imputed_ind = pd.DataFrame(index=inds)
    nu_imputed_ind = pd.DataFrame(index=inds)
    nu_imputed_ind_ctng = pd.DataFrame(index=inds)
    y_imputed_indXct = pd.DataFrame()
    nu_imputed_indXct = pd.DataFrame()
    nu_imputed_indXct_ctng = pd.DataFrame()

    ## for each gene
    tmpfn = helper.generate_tmpfn()
    for i, gene in enumerate(genes):
        print(gene)
        ### for y
        y_ = y.pivot(index='donor', columns='day', values=gene)
        # make sure y_ and P have the same order
        if np.any(np.array(y_.index) != np.array(n.index)) or np.any(np.array(y_.columns) != np.array(n.columns)):
            print(y_)
            print(n)
            sys.exit('Different order!\n')
        if np.any(np.array(y_.index) != np.array(P.index)) or np.any(np.array(y_.columns) != np.array(P.columns)):
            print(y_)
            print(P)
            sys.exit('Different order!\n')

        if snakemake.wildcards.im_genome not in ['Y','Y_Corrected']:
            # impute y for each gene
            y_ = imputeY4eachgene(y_, gene, y_path)

        y_imputed_ind[gene] = y_.mul(P).sum(axis=1)
        y_ = y_.reset_index().melt(id_vars='donor', var_name='day', value_name=gene)
        y_ = y_.sort_values(by=['donor','day']).reset_index(drop=True)
        if 'donor' not in y_imputed_indXct.columns:
            y_imputed_indXct['donor'] = y_['donor']
            y_imputed_indXct['day'] = y_['day']
        else:
            if np.any(y_imputed_indXct['donor'] != y_['donor']) or np.any(y_imputed_indXct['day'] != y_['day']):
                sys.exit('Not matching order!\n')
        y_imputed_indXct[gene] = y_[gene]

        ### for nu
        nu_ = nu.pivot(index='donor', columns='day', values=gene)
        # make sure y_ and P have the same order
        if np.any(np.array(nu_.index) != np.array(n.index)) or np.any(np.array(nu_.columns) != np.array(n.columns)):
            print(nu_)
            print(n)
            sys.exit('Different order!\n')
        if np.any(np.array(nu_.index) != np.array(P.index)) or np.any(np.array(nu_.columns) != np.array(P.columns)):
            print(nu_)
            print(P)
            sys.exit('Different order!\n')

        if snakemake.wildcards.im_genome not in ['Y','Y_Corrected']:
            nu_ = imputeNU4eachgene(nu_, nu_path, gene)

        ### make a copy for nu_: one sets <0 to 0 for ong, the other one sets <0 to max(nu_) for ctng
        nu_ctng_ = nu_.copy()
        if 'missingness' in snakemake.wildcards.keys():
            pass
        else:
            nu_[nu_ < 0] = 0
        nu_ctng_ = nu_ctng_.mask( nu_ctng_ < 0, np.amax(np.array(nu_ctng_)) )

        # calculate NU from CTNU (updated on 4/17)
        #nu_imputed_ind[gene] = nu_.mul(n**2).sum(axis=1) / ((n**2).sum(axis=1)) 
        #nu_imputed_ind_ctng[gene] = nu_ctng_.mul(n).sum(axis=1) / ((n**2).sum(axis=1)) 
        nu_imputed_ind[gene] = nu_.mul(P**2).sum(axis=1)
        nu_imputed_ind_ctng[gene] = nu_ctng_.mul(P**2).sum(axis=1)

        #### for ong
        nu_ = nu_.reset_index().melt(id_vars='donor', var_name='day', value_name=gene)
        nu_ = nu_.sort_values(by=['donor','day']).reset_index(drop=True)
        if 'donor' not in nu_imputed_indXct.columns:
            nu_imputed_indXct['donor'] = nu_['donor']
            nu_imputed_indXct['day'] = nu_['day']
        else:
            if np.any(nu_imputed_indXct['donor'] != nu_['donor']) or np.any(nu_imputed_indXct['day'] != nu_['day']):
                sys.exit('Not matching order!\n')
        nu_imputed_indXct[gene] = nu_[gene]

        #### for ctng
        nu_ctng_ = nu_ctng_.reset_index().melt(id_vars='donor', var_name='day', value_name=gene)
        nu_ctng_ = nu_ctng_.sort_values(by=['donor','day']).reset_index(drop=True)
        if 'donor' not in nu_imputed_indXct_ctng.columns:
            nu_imputed_indXct_ctng['donor'] = nu_ctng_['donor']
            nu_imputed_indXct_ctng['day'] = nu_ctng_['day']
        else:
            if np.any(nu_imputed_indXct_ctng['donor'] != nu_ctng_['donor']) or np.any(nu_imputed_indXct_ctng['day'] != nu_ctng_['day']):
                sys.exit('Not matching order!\n')
        nu_imputed_indXct_ctng[gene] = nu_ctng_[gene]

    # save imputed y and nu of all genes
    #y_imputed_indXct.to_csv(output.y_imputed_allgenes, sep='\t', index=False)
    #nu_imputed_indXct.to_csv(output.nu_imputed_allgenes, sep='\t', index=False)

    # transpose to gene * donor
    y = y_imputed_ind.transpose()
    nu = nu_imputed_ind.transpose()
    nu_ctng = nu_imputed_ind_ctng.transpose()

    # make sure y and nu have the same index and columns
    if np.any(np.array(y.index) != np.array(nu.index)) or np.any(np.array(y.columns) != np.array(nu.columns)):
        sys.exit('Y and NU not matching!\n')

    # standardize y and correspondingly nu
#    y_mean = y.mean(axis=1)
#    y_std = y.std(axis=1)
#    y_var = y.var(axis=1)
#    y = y.sub(y_mean, axis=0).divide(y_std, axis=0)
#    nu = nu.divide(y_var, axis=0)
#    nu_ctng = nu_ctng.divide(y_var, axis=0)

    ## make sure y and y_imputed_indXct have the same order of genes
    y_imputed_indXct = y_imputed_indXct.set_index(keys=['donor','day'])
    if np.any( np.array( y.index ) != np.array( y_imputed_indXct.columns ) ):
        sys.exit('y and y_imputed_indXct not matching!\n')

#    y_imputed_indXct = y_imputed_indXct.sub(y_mean, axis=1).divide(y_std, axis=1)
    y_imputed_indXct = y_imputed_indXct.reset_index()

    ## make sure y and nu_imputed_indXct have the same order of genes
    nu_imputed_indXct = nu_imputed_indXct.set_index(keys=['donor','day'])
    if np.any( np.array( y.index ) != np.array(nu_imputed_indXct.columns) ):
        sys.exit('y and nu_imputed_indXct not matching!\n')

#    nu_imputed_indXct = nu_imputed_indXct.divide( y_var, axis=1 )
    nu_imputed_indXct = nu_imputed_indXct.reset_index()

    ### for ctng
    nu_imputed_indXct_ctng = nu_imputed_indXct_ctng.set_index(keys=['donor','day'])
    if np.any( np.array( y.index ) != np.array(nu_imputed_indXct_ctng.columns) ):
        sys.exit('y and nu_imputed_indXct_ctng not matching!\n')

#    nu_imputed_indXct_ctng = nu_imputed_indXct_ctng.divide( y_var, axis=1 )
    nu_imputed_indXct_ctng = nu_imputed_indXct_ctng.reset_index()

    # y and nu and P
    ## make sure P and inds in the same order
    if np.any(np.array(P.index) != inds) or np.any(np.array(nu.columns) != inds) or np.any(np.array(y.columns)!=inds) or np.any(np.array(nu_ctng.columns) != inds) or np.any(np.unique(y_imputed_indXct['donor'])!=inds) or np.any(np.unique(nu_imputed_indXct['donor'])!=inds):
        sys.exit('Not matching order of individuals!\n')
    ### save inds
    P_path = os.path.dirname(output.P)
    for gene in genes:
        # exclude individuals with nu = 0 (some individuals have enough cells, but all cells have no gene expression)
        # seems we should keep nu = 0
        y_ = np.array(y.loc[gene])
        nu_ = np.array(nu.loc[gene])
        nu_ctng_ = np.array(nu_ctng.loc[gene])
        #in_inds = (nu_ != 0)
        #y_ = y_[in_inds]
        #nu_ = nu_[in_inds]
        os.makedirs(y_path+f'/rep{gene}', exist_ok=True)
        os.makedirs(nu_path+f'/rep{gene}', exist_ok=True)
        os.makedirs(P_path+f'/rep{gene}', exist_ok=True)
        if np.any(np.isnan(y_imputed_indXct[gene])):
            print(y_imputed_indXct[gene])
            sys.exit(gene)
        
        np.savetxt(y_path+f'/rep{gene}/y.gz', y_, delimiter='\t')
        np.savetxt(nu_path+f'/rep{gene}/nu.gz', nu_, delimiter='\t')
        np.savetxt(nu_path+f'/rep{gene}/nu.ctng.gz', nu_ctng_, delimiter='\t')

        y_imputed_indXct[['donor','day',gene]].to_csv(y_path+f'/rep{gene}/ct.y.gz', sep='\t', index=False)

        nu_imputed_indXct[['donor','day',gene]].to_csv(nu_path+f'/rep{gene}/ct.nu.gz', sep='\t', index=False)
        nu_imputed_indXct_ctng[['donor','day',gene]].to_csv(nu_path+f'/rep{gene}/ct.nu.ctng.gz', 
                sep='\t',index=False)

        P.to_csv(P_path+f'/rep{gene}/P.gz', sep='\t', index=False, header=False)

    # collect files
    f1 = open(output.y, 'w')
    f2 = open(output.nu, 'w')
    f3 = open(output.nu_ctng, 'w')
    f4 = open(output.P, 'w')
    f5 = open(output.imputed_ct_y, 'w')
    f6 = open(output.imputed_ct_nu, 'w')
    f7 = open(output.imputed_ct_nu_ctng, 'w')
    for gene in genes:
        f1.write(y_path+f'/rep{gene}/y.gz\n')
        f2.write(nu_path+f'/rep{gene}/nu.gz\n')
        f3.write(nu_path+f'/rep{gene}/nu.ctng.gz\n')
        f4.write(P_path+f'/rep{gene}/P.gz\n')
        f5.write(y_path+f'/rep{gene}/ct.y.gz\n')
        f6.write(nu_path+f'/rep{gene}/ct.nu.gz\n')
        f7.write(nu_path+f'/rep{gene}/ct.nu.ctng.gz\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    f7.close()

if __name__ == '__main__':
    main()
