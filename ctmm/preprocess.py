from typing import Tuple, Optional, Union

import re, sys
import numpy as np, pandas as pd
from scipy import stats, sparse
import pkg_resources
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import STAP

def pseudobulk(counts: pd.DataFrame=None, meta: pd.DataFrame=None, ann: object=None, 
        X: sparse.csr.csr_matrix=None, obs: pd.DataFrame=None, var: pd.DataFrame=None,
        ind_col: str='ind', ct_col: str='ct', cell_col: str='cell', ind_cut: int=0, ct_cut: int=0
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        X:  ann.X
        obs:    ann.obs 
        var:    ann.var
        ind_col:    column name for individual in meta or ann; it's renamed as ind in the output 
        ct_col: column name for cell type in meta or ann; it's renamed as ct in the output
        cell_col:   column name for cell in meta 
        ind_cut:    exclude individuals with # of cells <= ind_cut
        ct_cut: set pairs of individual and cell type with # of cells <= ct_cut to missing
    Returns:
        A tuple of 
            #. Cell Type-specific Pseudobulk of index: (ind, ct) * columns: genes
            #. Cell Type-specific noise variance of idnex: (ind, ct) * columns: genes
            #. cell type proportion matrix
            #. number of cells in each (ind, ct) before filtering
    '''

    if ann is None and X is None:
        # sanity check
        ## missing values
        if np.any(pd.isna(counts)):
            sys.exit('Missing values in gene expression!\n')

        # change column name
        meta = meta.rename(columns={ind_col:'ind', ct_col:'ct', cell_col:'cell'})

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

        # group by ind and ct
        data_grouped = data.groupby(['ind', 'ct'])
        # exclude groups with only one cell
        b = len( data_grouped )
        data_grouped = data_grouped.filter(lambda x: len(x) > 1).groupby(['ind','ct'])
        a = len( data_grouped )
        if a != b:
            print(f'Exclude {b-a} individual-cell type pairs with only one cell')

        # compute ctp
        ctp = data_grouped[genes].aggregate(np.mean)
        
        # compute ctnu
        ctnu = data_grouped[genes].aggregate(stats.sem) # need to rm ind-ct pair with only 1 cell
        ctnu = ctnu**2

    else:
        if ann is not None:
            obs = ann.obs
            X = ann.X if sparse.issparse(ann.X) else ann.X[:,:] # convert sparse dataset to sparse matrix
            var = ann.var
        obs = obs.rename(columns={ind_col:'ind', ct_col:'ct'})

        # paires of ind and ct
        print('Finding ind-ct pairs')
        sep = '_+_'
        while np.any( obs['ind'].str.contains(sep) ) or np.any( obs['ct'].str.contains(sep) ):
            sep = '_' + sep + '_'
        ind_ct = obs['ind'].astype(str) + sep + obs['ct'].astype(str)

        # indicator matrix
        pairs, index = np.unique( ind_ct, return_index=True )
        raw_order = pairs[np.argsort( index )]
        ind_ct = pd.Categorical(ind_ct, categories=raw_order)
        indicator = pd.get_dummies( ind_ct, sparse=True, dtype='int8' )
        ind_ct = indicator.columns
        cell_num =  indicator.sum().to_numpy()
        print( indicator.shape )
        print( indicator.iloc[:3,:3] )

        ## convert to sparse matrix
        #indicator = sparse.csr_matrix( indicator.to_numpy() )
        indicator = indicator.sparse.to_coo().tocsr()

        # inverse indicator matrix
        indicator_inv = indicator.multiply( 1/cell_num )
        print( indicator_inv.getrow(0)[0,0] )

        # compute ctp
        print('Computing CTP')
        ctp = indicator_inv.T @ X

        # compute ctnu
        print('Computing CTNU')
        ctp2 = indicator_inv.T @ X.power(2)
        ctnu = (ctp2 - ctp.power(2)).multiply( 1 / (cell_num**2).reshape(-1,1) )

        # convert to df
        ctp_index = pd.MultiIndex.from_frame(ind_ct.to_series().str.split(sep, expand=True, regex=False), 
                names=['ind', 'ct'])
        ctp = pd.DataFrame(ctp.toarray(), index=ctp_index, columns=var.index)
        ctnu = pd.DataFrame(ctnu.toarray(), index=ctp_index, columns=var.index)

        # group by ind-ct to compute cell numbers
        data_grouped = obs.reset_index(drop=False,names='cell').groupby(['ind', 'ct'])

    # compute cell numbers
    cell_counts = data_grouped['cell'].count().unstack(fill_value=0).astype('int')

    # filter individuals
    inds = cell_counts.index[cell_counts.sum(axis=1) > ind_cut].tolist()
    P = cell_counts.loc[cell_counts.index.isin(inds)]
    #ctp = ctp.loc[ctp.index.get_level_values('ind').isin(inds)]
    #ctnu = ctnu.loc[ctnu.index.get_level_values('ind').isin(inds)]

    # filter cts
    P2 = P.stack()
    P2 = P2.loc[P2 > ct_cut]
    ctp = ctp.loc[ctp.index.isin(P2.index)]
    ctnu = ctnu.loc[ctnu.index.isin(P2.index)]

    # conver P cell counts to proportions
    P = P.divide(P.sum(axis=1), axis=0)

    #print(ctp.shape, ctnu.shape, P.shape)

    return( ctp, ctnu, P, cell_counts )

def _softimpute(data: pd.DataFrame, seed: int=None, scale: bool->True) -> pd.DataFrame:
    '''
    Impute missing ctp or ct-specific noise variance (ctnu)

    Parameters:
        data:   ctp or ctnu of shape index: (ind, ct) * columns: genes
        seed:   seed for softImpute, only needed to be replicate imputation
        scale:  scale before imputation
    Results:
        imputed dataset
    '''
    # load softImpute r package
    rf = pkg_resources.resource_filename(__name__, 'softImpute.R')
    softImpute = STAP( open(rf).read(), 'softImpute' )

    if seed is None:
        seed = ro.NULL

    # Impute
    pandas2ri.activate()
    if scale:
        out = softImpute.my_softImpute( r['as.matrix'](data), scale=ro.vectors.BoolVector([True]), seed=seed )
    else:
        out = softImpute.my_softImpute( r['as.matrix'](data), seed=seed )
    out = dict( zip(out.names, list(out)) )
    out = pd.DataFrame(out['Y'], index=data.index, columns=data.columns)
    pandas2ri.deactivate()

    return( out )

def softimpute(data: pd.DataFrame, seed: int=None, scale: bool->True) -> pd.DataFrame:
    '''
    Impute missing ctp or ct-specific noise variance (ctnu)

    Parameters:
        data:   ctp or ctnu of shape index: (ind, ct) * columns: genes
        seed:   seed for softImpute, only needed to be replicate imputation
        scale:  scale before imputation
    Results:
        imputed dataset
    '''
    # transform to index: ind * columns: (genes: cts)
    data = data.unstack()

    # impute
    out = _softimpute(data, seed, scale)
    
    # transform back
    out = out.stack()

    return( out )

def mvn(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Impute missing ctp or ctnu

    Parameters:
        data:   ctp or ctnu of shape index: (ind, ct) * columns: genes
    Results:
        imputed dataset
    '''
    
    # load softImpute r package
    rf = pkg_resources.resource_filename(__name__, 'mvn.R')
    mvn_r = STAP( open(rf).read(), 'mvn_r' )
    pandas2ri.activate()

    imputed = []
    for gene in data.columns:
        print(gene)
        # transform to index: ind * columns: (cts)
        Y = data[gene].unstack()

        # Impute
        out = mvn_r.MVN_impute( r['as.matrix'](Y) )
        out = dict( zip(out.names, list(out)) )
        out = pd.DataFrame(out['Y'], index=Y.index, columns=Y.columns)

        # transform back
        out = out.stack()
        out.name = gene 

        imputed.append( out )

    pandas2ri.deactivate()
    imputed = pd.concat(imputed, axis=1)

    return( imputed )

def std(ctp: pd.DataFrame, ctnu: pd.DataFrame, P: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    For each Gene, stardardize Overall Pseudobulk (OP) to mean 0 and std 1, and scale ctp, 
    overall noise variance (nu), ct-specific noise variance (ctnu) correspondingly.

    Parameters:
        ctp:    imputed ctp with shape index: (ind, ct) * columns: genes
        ctnu:   imputed ctnu with shape index: (ind, ct) * columns: genes
        P:  cell type proportions matrix of shape ind * ct
    Returns:
        A tuple of 
            #. op
            #. nu
            #. ctp
            #. ctnu
    '''
    genes = ctp.columns.tolist()

    op_genes = []
    nu_genes = []
    ctp_genes = []
    ctnu_genes = []
    # sanity reorder inds and cts in ctp, ctnu, and P
    P = P.sort_index().sort_index(axis=1)
    inds = P.index
    cts = P.columns
    for gene in genes:
        # extract gene and transform to ind * ct
        gene_ctp = ctp[gene].unstack()
        gene_ctnu = ctnu[gene].unstack()
        
        # sanity reorder inds and cts in ctp, ctnu, and P
        gene_ctp = gene_ctp.sort_index().sort_index(axis=1)
        gene_ctnu = gene_ctnu.sort_index().sort_index(axis=1)

        # santity check inds and cts matching between ctp, ctnu, and P
        if not ( gene_ctnu.index.equals(inds)  or gene_ctp.index.equals(inds) ):
            sys.exit('Individuals not matching!')
        if not ( gene_ctnu.columns.equals(cts) or gene_ctp.columns.equals(cts) ):
            sys.exit('Cell types not matching!')

        # compute op and nu
        gene_op = (gene_ctp * P).sum(axis=1)
        gene_nu = ( gene_ctnu.mask(gene_ctnu<0, 0) * (P**2) ).sum(axis=1) # set negative ctnu to 0 for OP data

        # set negative ctnu to max for CTP data
        gene_ctnu = gene_ctnu.mask(gene_ctnu<0, gene_ctnu.max(), axis=1)

        # standardize op
        mean, std, var = gene_op.mean(), gene_op.std(), gene_op.var()
        gene_op = (gene_op - mean) / std
        gene_nu = gene_nu / var
        gene_ctp = (gene_ctp - mean) / std 
        gene_ctnu = gene_ctnu / var

        # transform back to series
        gene_ctp = gene_ctp.stack()
        gene_ctnu = gene_ctnu.stack()

        # add gene name
        gene_op.name = gene
        gene_nu.name = gene
        gene_ctp.name = gene
        gene_ctnu.name = gene 

        # 
        op_genes.append( gene_op )
        nu_genes.append( gene_nu )
        ctp_genes.append( gene_ctp )
        ctnu_genes.append( gene_ctnu )

    op = pd.concat( op_genes, axis=1 )
    nu = pd.concat( nu_genes, axis=1 )
    ctp = pd.concat( ctp_genes, axis=1 )
    ctnu = pd.concat( ctnu_genes, axis=1 )

    return( op, nu, ctp, ctnu )
