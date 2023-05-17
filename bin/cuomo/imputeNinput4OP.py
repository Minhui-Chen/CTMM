import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr, STAP
from ctmm import util, preprocess

# def imputeY4eachgene(y, gene, y_path):
#     y_before = y
#     os.makedirs(f'{y_path}/rep{gene}', exist_ok=True)
#     y.to_csv(f'{y_path}/rep{gene}/y.before_imputation.txt', sep='\t')

#     if snakemake.wildcards.im_mvn == 'N':
#         y = preprocess._softimpute(y, True)
#     else:
#         y = preprocess._mvn( y )

#     y.to_csv(f'{y_path}/rep{gene}/y.imputed.txt', sep='\t')

#     return( y )

# def imputeNU4eachgene(nu, nu_path, gene):
#     nu_before = nu

#     os.makedirs(f'{nu_path}/rep{gene}', exist_ok=True)
#     nu.to_csv(f'{nu_path}/rep{gene}/nu.before_imputation.txt', sep='\t')

#     if snakemake.wildcards.im_mvn == 'N':
#         nu = preprocess._softimpute( nu, True )
#     else:
#         nu = preprocess._mvn( nu )

#     nu.to_csv(f'{nu_path}/rep{gene}/nu.imputed.txt', sep='\t')

#     return( nu )

def main():

    # par
    input = snakemake.input
    output = snakemake.output

    # read 
    P = pd.read_table(input.P, index_col=0) # donor * day
    P.index.name = 'donor'
    n = pd.read_table(input.n, index_col=0) # donor * day
    n.index.name = 'donor'
    genes = [line.strip() for line in open(input.y_batch)]

    ctp = pd.read_table(input.y, index_col=(0,1))[genes] # donor - day * gene
    ctnu = pd.read_table(input.nu, index_col=(0,1))[genes] # donor - day * gene
    ctp = ctp.sort_values(by=['donor','day']) #.reset_index()
    ctnu = ctnu.sort_values(by=['donor','day']) #.reset_index()

    days = list(np.unique(P.columns))
    inds = list(np.unique(P.index))

    # sort
    P = P.sort_values(by='donor')[days]
    n = n.sort_values(by='donor')[days]

    y_path = os.path.dirname(output.y)
    nu_path = os.path.dirname(output.nu)

    # impute (if not already imputed transcriptome wide)
    if snakemake.wildcards.im_genome not in ['Y']:
        if snakemake.wildcards.im_mvn == 'N':
            ctp = preprocess.softimpute(ctp, scale=True, per_gene=True)
            ctnu = preprocess.softimpute(ctnu, scale=True, per_gene=True)
        else:
            ctp = preprocess.mvn(ctp)
            ctnu = preprocess.mvn(ctnu)

    # std
    op, nu_op, ctnu_op, ctp, nu_ctp, ctnu_ctp = preprocess.std(ctp, ctnu, P, return_all=True)
    
    # save 
    P_path = os.path.dirname(output.P)
    for gene in genes:
        os.makedirs(os.path.join(y_path,f'rep{gene}'), exist_ok=True)
        os.makedirs(os.path.join(nu_path,f'rep{gene}'), exist_ok=True)
        os.makedirs(os.path.join(P_path,f'rep{gene}'), exist_ok=True)

        np.savetxt(y_path+f'{y_path}/rep{gene}/y.gz', op[gene].to_numpy(), delimiter='\t')
        np.savetxt(f'{nu_path}/rep{gene}/nu.gz', nu_op[gene].to_numpy(), delimiter='\t')
        np.savetxt(f'{nu_path}/rep{gene}/nu.ctng.gz', nu_ctp[gene].to_numpy(), delimiter='\t')

        ctp[[gene]].to_csv(f'{y_path}/rep{gene}/ct.y.gz', sep='\t', index=True)
        ctnu_op[[gene]].to_csv(
                f'{nu_path}/rep{gene}/ct.nu.gz', sep='\t', index=True)
        ctnu_ctp[[gene]].to_csv(
                f'{nu_path}/rep{gene}/ct.nu.ctng.gz', sep='\t',index=True)

        P.to_csv(f'{P_path}/rep{gene}/P.txt', sep='\t', index=False, header=False)


    # op_imputed = pd.DataFrame(index=inds)
    # nu_imputed_ong = pd.DataFrame(index=inds)
    # nu_imputed_ctng = pd.DataFrame(index=inds)
    # ctp_imputed = pd.DataFrame()
    # ctnu_imputed_ong = pd.DataFrame()
    # ctnu_imputed_ctng = pd.DataFrame()

    # ## for each gene
    # for i, gene in enumerate(genes):
    #     print(gene)
    #     ### for y
    #     y_ = ctp.pivot(index='donor', columns='day', values=gene)
    #     # make sure y_ and P have the same order
    #     if not (y_.index.equals(n.index) and y_.columns.equals(n.columns) and 
    #     y_.index.equals(P.index) and y_.columns.equals(P.columns)):
    #         sys.exit('Different order!\n')

    #     if snakemake.wildcards.im_genome not in ['Y']:
    #         # impute y for each gene
    #         y_ = imputeY4eachgene(y_, gene, y_path)

    #     op_imputed[gene] = y_.mul(P).sum(axis=1)
    #     y_ = y_.reset_index().melt(id_vars='donor', var_name='day', value_name=gene)
    #     y_ = y_.sort_values(by=['donor','day']).reset_index(drop=True)
    #     if 'donor' not in ctp_imputed.columns:
    #         ctp_imputed['donor'] = y_['donor']
    #         ctp_imputed['day'] = y_['day']
    #     else:
    #         if not (np.array_equal(ctp_imputed['donor'], y_['donor']) 
    #         and np.array_equal(ctp_imputed['day'], y_['day'])):
    #             sys.exit('Not matching order!\n')
    #     ctp_imputed[gene] = y_[gene]

    #     ### for nu
    #     nu_ = nu.pivot(index='donor', columns='day', values=gene)
    #     # make sure y_ and P have the same order
    #     if not (nu_.index.equal(n.index) and nu_.columns.equal(n.columns)):
    #         sys.exit('Different order!\n')
    #     if not (nu_.index.equal(P.index) and nu_.columns.equal(P.columns)):
    #         sys.exit('Different order!\n')

    #     if snakemake.wildcards.im_genome not in ['Y']:
    #         nu_ = imputeNU4eachgene(nu_, nu_path, gene)

    #     ### make a copy for nu_: one sets <0 to 0 for ong, the other one sets <0 to max(nu_) for ctng
    #     nu_ctng_ = nu_.copy()
    #     nu_[nu_ < 0] = 0
    #     nu_ctng_ = nu_ctng_.mask( nu_ctng_ < 0, np.amax(np.array(nu_ctng_)) )

    #     # calculate NU from CTNU
    #     nu_imputed_ong[gene] = nu_.mul(P**2).sum(axis=1)
    #     nu_imputed_ctng[gene] = nu_ctng_.mul(P**2).sum(axis=1)

    #     #### for ong
    #     nu_ = nu_.reset_index().melt(id_vars='donor', var_name='day', value_name=gene)
    #     nu_ = nu_.sort_values(by=['donor','day']).reset_index(drop=True)
    #     if 'donor' not in ctnu_imputed_ong.columns:
    #         ctnu_imputed_ong['donor'] = nu_['donor']
    #         ctnu_imputed_ong['day'] = nu_['day']
    #     else:
    #         if not (np.array_equal(ctnu_imputed_ong['donor'], nu_['donor']) 
    #         and np.array_equal(ctnu_imputed_ong['day'], nu_['day'])):
    #             sys.exit('Not matching order!\n')
    #     ctnu_imputed_ong[gene] = nu_[gene]

    #     #### for ctng
    #     nu_ctng_ = nu_ctng_.reset_index().melt(id_vars='donor', var_name='day', value_name=gene)
    #     nu_ctng_ = nu_ctng_.sort_values(by=['donor','day']).reset_index(drop=True)
    #     if 'donor' not in ctnu_imputed_ctng.columns:
    #         ctnu_imputed_ctng['donor'] = nu_ctng_['donor']
    #         ctnu_imputed_ctng['day'] = nu_ctng_['day']
    #     else:
    #         if not (np.array_equal(ctnu_imputed_ctng['donor'], nu_ctng_['donor']) 
    #         and np.array_equal(ctnu_imputed_ctng['day'], nu_ctng_['day'])):
    #             sys.exit('Not matching order!\n')
    #     ctnu_imputed_ctng[gene] = nu_ctng_[gene]

    # # transpose to gene * donor
    # op = op_imputed.transpose()
    # nu_ong = nu_imputed_ong.transpose()
    # nu_ctng = nu_imputed_ctng.transpose()

    # # make sure y and nu have the same index and columns
    # if not (op.index.equals(nu_ong.index) and op.columns.equals(nu_ong.columns)):
    #     sys.exit('Y and NU not matching!\n')

    # # standardize y and correspondingly nu
    # op_mean, op_std, op_var = op.mean(axis=1), op.std(axis=1), op.var(axis=1)
    # op = op.sub(op_mean, axis=0).divide(op_std, axis=0)
    # nu_ong = nu_ong.divide(op_var, axis=0)
    # nu_ctng = nu_ctng.divide(op_var, axis=0)

    # ## make sure y and y_imputed_indXct have the same order of genes
    # ctp_imputed = ctp_imputed.set_index(keys=['donor','day'])
    # if not op.index.equals( ctp_imputed.columns ):
    #     sys.exit('op and ctp_imputed not matching!\n')

    # ctp_imputed = ctp_imputed.sub(op_mean, axis=1).divide(op_std, axis=1)
    # ctp_imputed = ctp_imputed.reset_index()

    # ## make sure y and nu_imputed_indXct have the same order of genes
    # ctnu_imputed_ong = ctnu_imputed_ong.set_index(keys=['donor','day'])
    # if not op.index.equals(ctnu_imputed_ong.columns):
    #     sys.exit('op and ctnu_imputed_ong not matching!\n')

    # ctnu_imputed_ong = ctnu_imputed_ong.divide( op_var, axis=1 )
    # ctnu_imputed_ong = ctnu_imputed_ong.reset_index()

    # ### for ctng
    # ctnu_imputed_ctng = ctnu_imputed_ctng.set_index(keys=['donor','day'])
    # if not op.index.equals(ctnu_imputed_ctng.columns):
    #     sys.exit('op and ctnu_imputed_ctng not matching!\n')

    # ctnu_imputed_ctng = ctnu_imputed_ctng.divide( op_var, axis=1 )
    # ctnu_imputed_ctng = ctnu_imputed_ctng.reset_index()

    # # y and nu and P
    # ## make sure P and inds in the same order
    # if not ( P.index.equals(inds) and nu_ong.columns.equals(inds) and 
    #         op.columns.equals(inds) and nu_ctng.columns.equals(inds) and 
    #         np.array_equal(np.unique(ctp_imputed['donor']), inds) and 
    #         np.array_equal(np.unique(ctnu_imputed_ong['donor']), inds) ):
    #     sys.exit('Not matching order of individuals!\n')

    # ### save 
    # P_path = os.path.dirname(output.P)
    # for gene in genes:
    #     y_ = np.array(op.loc[gene])
    #     nu_ = np.array(nu_ong.loc[gene])
    #     nu_ctng_ = np.array(nu_ctng.loc[gene])
    #     os.makedirs(y_path+f'/rep{gene}', exist_ok=True)
    #     os.makedirs(nu_path+f'/rep{gene}', exist_ok=True)
    #     os.makedirs(P_path+f'/rep{gene}', exist_ok=True)
    #     np.savetxt(y_path+f'/rep{gene}/y.gz', y_, delimiter='\t')
    #     np.savetxt(nu_path+f'/rep{gene}/nu.gz', nu_, delimiter='\t')
    #     np.savetxt(nu_path+f'/rep{gene}/nu.ctng.gz', nu_ctng_, delimiter='\t')

    #     ctp_imputed[['donor','day',gene]].to_csv(
    #             y_path+f'/rep{gene}/ct.y.gz', sep='\t', index=False)

    #     ctnu_imputed_ong[['donor','day',gene]].to_csv(
    #             nu_path+f'/rep{gene}/ct.nu.gz', sep='\t', index=False)
    #     ctnu_imputed_ctng[['donor','day',gene]].to_csv(
    #             nu_path+f'/rep{gene}/ct.nu.ctng.gz', sep='\t',index=False)

    #     P.to_csv(P_path+f'/rep{gene}/P.txt', sep='\t', index=False, header=False)

    # collect files
    f1 = open(output.y, 'w')
    f2 = open(output.nu, 'w')
    f3 = open(output.nu_ctp, 'w')
    f4 = open(output.P, 'w')
    f5 = open(output.imputed_cty, 'w')
    f6 = open(output.imputed_ctnu, 'w')
    f7 = open(output.imputed_ctnu_ctp, 'w')
    for gene in genes:
        f1.write(f'{y_path}/rep{gene}/y.gz\n')
        f2.write(f'{nu_path}/rep{gene}/nu.gz\n')
        f3.write(f'{nu_path}/rep{gene}/nu.ctng.gz\n')
        f4.write(f'{P_path}/rep{gene}/P.txt\n')
        f5.write(f'{y_path}/rep{gene}/ct.y.gz\n')
        f6.write(f'{nu_path}/rep{gene}/ct.nu.gz\n')
        f7.write(f'{nu_path}/rep{gene}/ct.nu.ctng.gz\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    f7.close()

if __name__ == '__main__':
    main()
