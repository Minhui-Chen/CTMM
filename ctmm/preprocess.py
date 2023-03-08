from typing import Tuple, Optional, Union

import re, sys
import numpy as np, pandas as pd
from scipy import stats
import pkg_resources
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import STAP

def pseudobulk(counts: pd.DataFrame=None, meta: pd.DataFrame=None, ann: object=None, 
        ind_cut: int=0, ct_cut: int=0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Compute Cell Type-Specific Pseudobulk and Cell Type-specific noise variance
    and remove individuals with # of cells <= ind_cut
    and for paris of individual and cell type with # of cell <= ct_cut, set CTP and CT noise variance to missing

    Parameters:
        counts: contains processed gene expression level across all cells (from all cell types and individuals).
                Each row corresponds to a single genes and each column corresponds to a single cell.
                Use gene names as dataframe row INDEX and cell IDs as dataframe COLUMNS label.
        meta:   contains cell meta informations.
                It contains at least three columns: cell, ind, ct.
                cell contains cell IDs, corresponding to column labels in counts.
                ind contains individual IDs.
                ct contains cell type, indicating the assignment of cells to cell types. 
        ann:    AnnData object (e.g. data from scanpy)
                each row corresponds to a cell with cell id, 
                and each column corresponds to a gene with a gene id.
                each cell have additional metadata, including cell type (column name: ct) 
                and donor information (column name: ind) for each cell
        ind_cut:    exclude individuals with # of cells <= ind_cut
        ct_cut: set pairs of individual and cell type with # of cells <= ct_cut to missing
    Returns:
        A tuple of 
            #. Cell Type-specific Pseudobulk of index: (ind, ct) * columns: genes
            #. Cell Type-specific noise variance of idnex: (ind, ct) * columns: genes
            #. cell type proportion matrix
    '''

    if ann is None:
        # sanity check
        ## missing values
        if np.any(pd.isna(counts)):
            sys.exit('Missing values in gene expression!\n')

        ## identical unique cell ids
        if (len(np.unique(meta['cell'])) != meta.shape[0]):
            sys.exit('Duplicate cells!\n')
        if (meta.shape[0] != counts.shape[1]) or (len(np.setdiff1d(meta['cell'],counts.columns)) != 0):
            sys.exit('Cells not matching in counts and meta!\n')

        # collect genes and cell types
        genes = counts.index.tolist()
        #cts = np.unique(meta['ct'].to_numpy())

        # transfrom gene * cell to cell * gene in counts
        counts = counts.transpose()

        # merge with meta
        data = counts.merge(meta, left_index=True, right_on='cell') # cell * (gene, ind, ct)
    else:
        data = ann.to_df()

        # collect genes 
        genes = data.columns.tolist()

        data = data.reset_index(drop=False, names='cell')
        data['ind'] = ann.obs.ind
        data['ct'] = ann.obs.ct


    # group by ind and ct
    data_grouped = data.groupby(['ind','ct'])

    # compute cell numbers
    P = data_grouped['cell'].count().reset_index(drop=False)
    P = P.pivot(index='ind',columns='ct', values='cell')
    ## fill NA with 0
    P = P.fillna( 0 )

    # compute ctp
    ctp = data_grouped[genes].aggregate(np.mean)
    
    # compute ctnu
    ctnu = data_grouped[genes].aggregate(stats.sem) # need to rm ind-ct pair with only 1 cell
    ctnu = ctnu**2

    # filter individuals
    inds = P.index[P.sum(axis=1) > ind_cut].tolist()
    P = P.loc[P.index.isin(inds)]
    ctp = ctp.loc[ctp.index.get_level_values('ind').isin(inds)]
    ctnu = ctnu.loc[ctnu.index.get_level_values('ind').isin(inds)]

    # filter cts
    P2 = P.stack()
    P2 = P2.loc[P2 > ct_cut]
    ctp = ctp.loc[ctp.index.isin(P2.index)]
    ctnu = ctnu.loc[ctnu.index.isin(P2.index)]

    # conver P cell counts to proportions
    P = P.divide(P.sum(axis=1), axis=0)

    #print(ctp.shape, ctnu.shape, P.shape)

    return( ctp, ctnu, P )

def softimpute(data: pd.DataFrame, seed: int=None) -> pd.DataFrame:
    '''
    Impute missing ctp or ct-specific noise variance (ctnu)

    Parameters:
        data:   ctp or ctnu of shape index: (ind, ct) * columns: genes
        seed:   seed for softImpute, only needed to be replicate imputation
    Results:
        imputed dataset
    '''
    # transform to index: ind * columns: (genes: cts)
    data = data.unstack()

    # load softImpute r package
    rf = pkg_resources.resource_filename(__name__, 'softImpute.R')
    softImpute = STAP( open(rf).read(), 'softImpute' )

    if seed is None:
        seed = ro.NULL

    # Impute
    pandas2ri.activate()
    out = softImpute.my_softImpute( r['as.matrix'](data), scale=ro.vectors.BoolVector([True]), seed=seed )
    out = dict( zip(out.names, list(out)) )
    out = pd.DataFrame(out['Y'], index=data.index, columns=data.columns)
    pandas2ri.deactivate()

    # transform back
    out = out.stack()

    return( out )

def std(ctp: pd.DataFrame, ctnu: pd.DataFrame, P: pd.DataFrame, gene:str) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    '''
    For one Gene, stardardize Overall Pseudobulk (OP) to mean 0 and std 1, and scale ctp, 
    overall noise variance (nu), ct-specific noise variance (ctnu) correspondingly.

    Parameters:
        ctp:    imputed ctp with shape index: (ind, ct) * columns: genes
        ctnu:   imputed ctnu with shape index: (ind, ct) * columns: genes
        P:  cell type proportions matrix of shape ind * ct
        gene:   target gene
    Returns:
        A tuple of 
            #. op
            #. nu
            #. ctp
            #. ctnu
    '''
    # extract gene and transform to ind * ct
    ctp = ctp[gene].unstack()
    ctnu = ctnu[gene].unstack()
    
    # sanity reorder inds and cts in ctp, ctnu, and P
    ctp = ctp.sort_index().sort_index(axis=1)
    ctnu = ctnu.sort_index().sort_index(axis=1)
    P = P.sort_index().sort_index(axis=1)

    # santity check inds and cts matching between ctp, ctnu, and P
    inds = ctp.index.to_numpy()
    cts = ctp.columns.to_numpy()
    if np.any( ctnu.index.to_numpy() != inds ) or np.any( P.index.to_numpy() != inds ):
        sys.exit('Individuals not matching!')
    if np.any( ctnu.columns.to_numpy() != cts ) or np.any( P.columns.to_numpy() != cts ):
        sys.exit('Cell types not matching!')

    # compute op and nu
    op = (ctp * P).sum(axis=1)
    nu = ( ctnu.mask(ctnu<0, 0) * (P**2) ).sum(axis=1) # set negative ctnu to 0 for OP data

    # set negative ctnu to max for CTP data
    ctnu = ctnu.mask(ctnu<0, ctnu.max(), axis=1)

    # standardize op
    mean, std, var = op.mean(), op.std(), op.var()
    op = (op - mean) / std
    nu = nu / var
    ctp = (ctp - mean) / std 
    ctnu = ctnu / var

    return( op, nu, ctp, ctnu )
