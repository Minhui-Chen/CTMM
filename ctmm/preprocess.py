from typing import Tuple, Optional, Union

import re, sys
import numpy as np, pandas as pd
from pandas import DataFrame
from scipy import stats, sparse
import pkg_resources
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import STAP
from . import log


def normalize(X: sparse.csr.csr_matrix, l=1e6) -> sparse.csr.csr_matrix:
    """
    Normalize read counts to CPM: x/s, where s = total reads / 10^6

    Parameters:
        X:  ann.X (raw read counts)
        l:  normalized total counts

    Returns:
        Normalized X
    """
    s = X.sum(axis=1).A1
    X = X.multiply(l / s[:, np.newaxis])
    
    return X


def transform(X: sparse.csr.csr_matrix, transform: str='raw_counts'):
    """
    Transform read counts

    Parameters:
        X:  ann.X (raw read counts), in the shape of cell x gene
        transform:  transformation methods
                    'raw_counts': no transformation
                    'logp_cpm': ln(CPM + 1)
                    'logp_cp10k': ln(cp10k + 1)

    Returns:
        Transformed X
    """
    if transform == 'raw_counts':
        X = X.copy()
    elif transform == 'logp_cpm':
        l = 1e6
        X = normalize(X, l)
        X = np.log1p(X)
    elif transform == 'logp_cp10k':
        l = 1e4
        X = normalize(X, l)
        X = np.log1p(X)

    return X


def pseudobulk(counts: pd.DataFrame = None, meta: pd.DataFrame = None, ann: object = None,
               X: sparse.csr.csr_matrix = None, obs: pd.DataFrame = None, var: pd.DataFrame = None,
               ind_col: str = 'ind', ct_col: str = 'ct', cell_col: str = 'cell', ind_cut: int = 0, ct_cut: int = 1
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
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
        X:  ann.X (normalized)
        obs:    ann.obs, cell info, with cell id as index
        var:    ann.var, gene info, with gene id as index
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
    """

    if ann is None and X is None:
        # sanity check
        ## missing values
        if np.any(pd.isna(counts)):
            sys.exit('Missing values in gene expression!\n')

        # change column name
        meta = meta.rename(columns={ind_col: 'ind', ct_col: 'ct', cell_col: 'cell'})

        ## identical unique cell ids
        if (len(np.unique(meta['cell'])) != meta.shape[0]):
            sys.exit('Duplicate cells!\n')
        if (meta.shape[0] != counts.shape[1]) or (len(np.setdiff1d(meta['cell'], counts.columns)) != 0):
            sys.exit('Cells not matching in counts and meta!\n')

        # collect genes and cell types
        genes = counts.index.tolist()
        # cts = np.unique(meta['ct'].to_numpy())

        # transfrom gene * cell to cell * gene in counts
        counts = counts.transpose()

        # merge with meta
        data = counts.merge(meta, left_index=True, right_on='cell')  # cell * (gene, ind, ct)

        # group by ind and ct
        data_grouped = data.groupby(['ind', 'ct'])
        # exclude groups with only one cell
        b = len(data_grouped)
        data_grouped = data_grouped.filter(lambda x: len(x) > 1).groupby(['ind', 'ct'])
        a = len(data_grouped)
        if a != b:
            print(f'Exclude {b - a} individual-cell type pairs with only one cell')

        # compute ctp
        ctp = data_grouped[genes].aggregate(np.mean)

        # compute ctnu
        ctnu = data_grouped[genes].aggregate(stats.sem)  # need to rm ind-ct pair with only 1 cell
        ctnu = ctnu ** 2

    else:
        if ann is not None:
            obs = ann.obs
            X = ann.X if sparse.issparse(ann.X) else ann.X[:, :]  # convert sparse dataset to sparse matrix
            var = ann.var
        obs = obs.rename(columns={ind_col: 'ind', ct_col: 'ct'})

        # paires of ind and ct
        print('Finding ind-ct pairs')
        sep = '_+_'
        while np.any(obs['ind'].str.contains(sep)) or np.any(obs['ct'].str.contains(sep)):
            sep = '_' + sep + '_'
        ind_ct = obs['ind'].astype(str) + sep + obs['ct'].astype(str)

        # indicator matrix
        pairs, index = np.unique(ind_ct, return_index=True)
        raw_order = pairs[np.argsort(index)]
        ind_ct = pd.Categorical(ind_ct, categories=raw_order)
        indicator = pd.get_dummies(ind_ct, dtype=float)  # don't use dtype int8
        # indicator = pd.get_dummies(ind_ct, sparse=True, dtype=float)  # don't use dtype int8
        cell_num = indicator.to_numpy().sum(axis=0)[:, np.newaxis]
        ind_ct = indicator.columns

        ## convert to sparse matrix
        indicator = sparse.csr_matrix( indicator.to_numpy() )
        # indicator = indicator.sparse.to_coo().tocsr()

        # inverse indicator matrix
        indicator_inv = indicator.multiply(1 / cell_num.T)
        # print(indicator_inv.getrow(0)[0, 0])

        # compute ctp
        log.logger.info('Computing CTP')
        ctp = indicator_inv.T @ X

        # compute ctnu
        log.logger.info('Computing CTNU')
        adj_cell_num = cell_num - 1
        adj_cell_num[adj_cell_num == 0] = 1e-12  # avoid divide by zero  # array from np matrix doesn't work (bug?)
        # print(adj_cell_num)
        # print(np.any(adj_cell_num == 0))
        # np.savetxt('adj_cell_num.txt', adj_cell_num)

        if sparse.issparse(X): 
            ctp2 = indicator_inv.T @ X.power(2)
            ctnu = (ctp2 - ctp.power(2)).multiply(1 / adj_cell_num)
            # print(ctp[:5, :5].toarray(), ctp2[:5, :5].toarray(), ctnu[:5, :5].toarray(), adj_cell_num[:5, 0])

            ctp = ctp.toarray()
            ctnu = ctnu.toarray()

        else:
            ctp2 = indicator_inv.T @ (X ** 2)
            ctnu = (ctp2 - ctp ** 2) * (1 / adj_cell_num)
            print(ctp[:5, :5], ctp2[:5, :5], ctnu[:5, :5], adj_cell_num[0, :5])

        # convert to df
        ctp_index = pd.MultiIndex.from_frame(ind_ct.to_series().str.split(sep, expand=True, regex=False),
                                            names=['ind', 'ct'])
        ctp = pd.DataFrame(ctp, index=ctp_index, columns=var.index)
        ctnu = pd.DataFrame(ctnu, index=ctp_index, columns=var.index)

        # group by ind-ct to compute cell numbers
        data_grouped = obs.reset_index().groupby(['ind', 'ct'])


    # compute cell numbers
    cell_counts = data_grouped.size().unstack(fill_value=0).astype('int')

    # filter individuals
    inds = cell_counts.index[cell_counts.sum(axis=1) > ind_cut].tolist()
    P = cell_counts.loc[cell_counts.index.isin(inds)]
    # ctp = ctp.loc[ctp.index.get_level_values('ind').isin(inds)]
    # ctnu = ctnu.loc[ctnu.index.get_level_values('ind').isin(inds)]

    # filter cts
    P2 = P.stack()
    P2 = P2.loc[P2 > ct_cut]
    ctp = ctp.loc[ctp.index.isin(P2.index)]
    ctnu = ctnu.loc[ctnu.index.isin(P2.index)]

    # conver P cell counts to proportions
    P = P.divide(P.sum(axis=1), axis=0)

    # print(ctp.shape, ctnu.shape, P.shape)

    return (ctp, ctnu, P, cell_counts)


def _softimpute(data: pd.DataFrame, seed: int = None, scale: bool = True) -> pd.DataFrame:
    '''
    Impute missing ctp or ct-specific noise variance (ctnu)

    Parameters:
        data:   ctp or ctnu of shape index: ind * columns: (genes, cts) (or cts when impute for one gene)
        seed:   seed for softImpute, only needed to be replicate imputation
        scale:  scale before imputation
    Results:
        imputed dataset
    '''
    # load softImpute r package
    is_sourced = r("exists('my_softImpute')")[0]
    if not is_sourced:
        rf = pkg_resources.resource_filename(__name__, 'softImpute.R')
        softImpute = STAP(open(rf).read(), 'softImpute')

    if seed is None:
        seed = ro.NULL

    # Impute
    pandas2ri.activate()
    if scale:
        out = softImpute.my_softImpute(r['as.matrix'](data), scale=ro.vectors.BoolVector([True]), seed=seed)
    else:
        out = softImpute.my_softImpute(r['as.matrix'](data), seed=seed)
    out = dict(zip(out.names, list(out)))
    out = pd.DataFrame(out['Y'], index=data.index, columns=data.columns)
    pandas2ri.deactivate()

    return (out)


def softimpute(data: pd.DataFrame, seed: int = None, scale: bool = True,
               per_gene: bool = False) -> pd.DataFrame:
    '''
    Impute missing ctp or ct-specific noise variance (ctnu)

    Parameters:
        data:   ctp or ctnu of shape index: (ind, ct) * columns: genes
        seed:   seed for softImpute, only needed to be replicate imputation
        scale:  scale before imputation
        per_gene:   perform imputation per gene
    Results:
        imputed dataset
    '''
    if not per_gene:
        # transform to index: ind * columns: (genes: cts)
        data = data.unstack()

        # impute
        imputed = _softimpute(data, seed, scale)

        # transform back
        imputed = imputed.stack()
    else:
        imputed = []
        for i, gene in enumerate(data.columns):
            if i % 100 == 0:
                log.logger.info(f'SoftImpute imputing {gene}')

            # transform to index: ind * columns: (cts)
            Y = data[gene].unstack()

            # Impute
            out = _softimpute(Y)

            # transform back
            out = out.stack()
            out.name = gene

            imputed.append(out)

        imputed = pd.concat(imputed, axis=1)

    return (imputed)


def _mvn(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing ctp or ctnu

    Parameters:
        data:   ctp or ctnu of shape index: ind * columns: ct
    Results:
        imputed dataset
    """

    # load MVN r package
    is_sourced = r("exists('MVN_impute')")[0]
    if not is_sourced:
        rf = pkg_resources.resource_filename(__name__, 'mvn.R')
        mvn_r = STAP(open(rf).read(), 'mvn_r')
    pandas2ri.activate()

    # impute
    out = mvn_r.MVN_impute(r['as.matrix'](data))
    out = dict(zip(out.names, list(out)))
    out = pd.DataFrame(out['Y'], index=data.index, columns=data.columns)

    pandas2ri.deactivate()

    return out


def mvn(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing ctp or ctnu

    Parameters:
        data:   ctp or ctnu of shape index: (ind, ct) * columns: genes
    Results:
        imputed dataset
    """

    imputed = []
    for i, gene in enumerate(data.columns):
        if i % 100 == 0:
            log.logger.info(f'MVN imputing {gene}')

        # transform to index: ind * columns: (cts)
        Y = data[gene].unstack()

        # Impute
        out = _mvn(Y)

        # transform back
        out = out.stack()
        out.name = gene

        imputed.append(out)

    imputed = pd.concat(imputed, axis=1)

    return imputed


def _std(ctp: pd.DataFrame, ctnu: pd.DataFrame, P: pd.DataFrame
         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each Gene, stardardize Overall Pseudobulk (OP) to mean 0 and std 1, and scale ctp,
    overall noise variance (nu), ct-specific noise variance (ctnu) correspondingly.

    Parameters:
        ctp:    imputed ctp for each gene with shape index: ind * ct
        ctnu:   imputed ctnu for each gene with shape index: ind * ct
        P:  cell type proportions matrix of shape ind * ct
    Returns:
        A tuple of
            #. op
            #. nu_op: negative ctnu set to 0
            #. ctnu_op: negative ctnu set to 0
            #. ctp
            #. nu_ctp: negative ctnu set to max
            #. ctnu_ctp: negative ctnu set to max
    """

    # compute op and nu
    op = (ctp * P).sum(axis=1)
    ctnu_op = ctnu.mask(ctnu < 0, 0)  # set negative ctnu to 0 for OP data
    nu_op = (ctnu_op * (P ** 2)).sum(axis=1)

    # set negative ctnu to max for CTP data
    ctnu_ctp = ctnu.mask(ctnu < 0, ctnu.max(), axis=1)
    nu_ctp = (ctnu_ctp * (P ** 2)).sum(axis=1)

    # standardize op
    mean, std, var = op.mean(), op.std(), op.var()

    op = (op - mean) / std
    ctp = (ctp - mean) / std
    nu_op = nu_op / var
    nu_ctp = nu_ctp / var
    ctnu_op = ctnu_op / var
    ctnu_ctp = ctnu_ctp / var

    return op, nu_op, ctnu_op, ctp, nu_ctp, ctnu_ctp


def std(ctp: pd.DataFrame, ctnu: pd.DataFrame, P: pd.DataFrame, return_all: bool = False
        ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], 
                   Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    For each Gene, stardardize Overall Pseudobulk (OP) to mean 0 and std 1, and scale ctp,
    overall noise variance (nu), ct-specific noise variance (ctnu) correspondingly.

    Parameters:
        ctp:    imputed ctp with shape index: (ind, ct) * columns: genes
        ctnu:   imputed ctnu with shape index: (ind, ct) * columns: genes
        P:  cell type proportions matrix of shape ind * ct
        return_all: return ctnu for OP and nu for CTP
    Returns:
        A tuple of
            #. op
            #. nu
            #. ctnu_op (optional)
            #. ctp
            #. nu_ctp (optional)
            #. ctnu
    """
    genes = ctp.columns.tolist()

    op_genes = []
    nu_op_genes = []
    ctnu_op_genes = []
    ctp_genes = []
    nu_ctp_genes = []
    ctnu_ctp_genes = []
    # sanity check order of inds and cts in ctp, ctnu, and P
    inds = P.index
    cts = P.columns

    for i, gene in enumerate(genes):
        # extract gene and transform to ind * ct
        gene_ctp = ctp[gene].unstack()
        gene_ctnu = ctnu[gene].unstack()

        if i == 0:
            # santity check inds and cts matching between ctp, ctnu, and P
            if not (gene_ctnu.index.equals(inds) and gene_ctp.index.equals(inds)):
                sys.exit('Individuals not matching!')
            if not (gene_ctnu.columns.equals(cts) and gene_ctp.columns.equals(cts)):
                sys.exit('Cell types not matching!')

        # standardization
        gene_op, gene_nu_op, gene_ctnu_op, gene_ctp, gene_nu_ctp, gene_ctnu_ctp = _std(
            gene_ctp, gene_ctnu, P)

        # transform back to series
        gene_ctp = gene_ctp.stack()
        gene_ctnu_op = gene_ctnu_op.stack()
        gene_ctnu_ctp = gene_ctnu_ctp.stack()

        # add gene name
        gene_op.name = gene
        gene_nu_op.name = gene
        gene_ctnu_op.name = gene
        gene_ctp.name = gene
        gene_nu_ctp.name = gene
        gene_ctnu_ctp.name = gene

        # 
        op_genes.append(gene_op)
        nu_op_genes.append(gene_nu_op)
        ctnu_op_genes.append(gene_ctnu_op)
        ctp_genes.append(gene_ctp)
        nu_ctp_genes.append(gene_nu_ctp)
        ctnu_ctp_genes.append(gene_ctnu_ctp)

    std_op = pd.concat(op_genes, axis=1)
    nu_op = pd.concat(nu_op_genes, axis=1)
    ctnu_op = pd.concat(ctnu_op_genes, axis=1)
    std_ctp = pd.concat(ctp_genes, axis=1)
    nu_ctp = pd.concat(nu_ctp_genes, axis=1)
    ctnu_ctp = pd.concat(ctnu_ctp_genes, axis=1)

    # santity check order of inds and cts doesn't change after standardization
    if not (std_ctp.index.equals(ctp.index) and std_ctp.columns.equals(ctp.columns)):
        sys.exit('Orders of individuals and cell types changed after standardization!')

    if return_all:
        return std_op, nu_op, ctnu_op, std_ctp, nu_ctp, ctnu_ctp
    else:
        return std_op, nu_op, std_ctp, ctnu_ctp
