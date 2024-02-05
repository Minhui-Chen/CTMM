from typing import Tuple, Optional, Union

import os, tempfile, sys
import numpy as np, pandas as pd
import numpy.typing as npt
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from scipy import stats, linalg, optimize
from numpy.random import default_rng

from . import wald, log

def read_covars(fixed_covars: dict = {}, random_covars: dict = {}, C: Optional[int] = None) -> tuple:
    '''
    Read fixed and random effect design matrices

    Parameters:
        fixed_covars:   files of design matrices for fixed effects,
                        except for cell type-specifc fixed effect, without header
        random_covars:  files of design matrices for random effects,
                        except for cell type-shared and -specifc random effect, without header
        C:  number of cell types
    Returns:
        a tuple of
            #. dict of design matrices for fixed effects
            #. dict of design matrices for random effects
            #. others
    '''
    def read(covars):
        tmp = {}
        for key in covars.keys():
            f = covars[key]
            if isinstance( f, str ):
                tmp[key] = np.loadtxt( f )
            else:
                tmp[key] = f
        return( tmp )

    fixed_covars = read(fixed_covars)
    random_covars = read(random_covars)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )
    random_keys = list( np.sort( list(random_covars.keys()) ) )
    Rs = [random_covars[key] for key in random_keys]
    if C:
        random_MMT = [np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 ) for R in Rs]
    else:
        random_MMT = [R @ R.T for R in Rs]

    return( fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT )

def optim(fun: callable, par: list, args: tuple, method: str) -> Tuple[object, dict]:
    '''
    Optimization use scipy.optimize.minimize

    Parameters:
        fun:    objective function to minimize (e.g. log-likelihood function)
        par:    initial parameters
        args:   extra arguments passed to objective function
        method: optimization method, e.g. BFGS
    Returns:
        a tuple of
            #. OptimizeResult object from optimize.minimize
            #. dict of optimization results
    '''
    if method is None:
        out1 = optimize.minimize( fun, par, args=args, method='BFGS' )
        out = optimize.minimize( fun, out1['x'], args=args, method='Nelder-Mead' )
        opt = {'method1':'BFGS', 'success1':out1['success'], 'status1':out1['status'],
                'message1':out1['message'], 'l1':out1['fun'] * (-1),
                'method':'Nelder-Mead', 'success':out['success'], 'status':out['status'],
                'message':out['message'], 'l':out['fun'] * (-1)}
    else:
        out = optimize.minimize( fun, par, args=args, method=method )
        opt = {'method':method, 'success':out['success'], 'status':out['status'],
                'message':out['message'], 'l':out['fun'] * (-1)}
    return( out, opt )

def check_optim(opt: dict, hom2: float, ct_overall_var: float, fixed_vars: dict, random_vars: dict, cut: float=5) -> bool:
    '''
    Check whether optimization converged successfully

    Parameters:
        opt:    dict of optimization results, e.g. log-likelihood
        hom2:   shared variance of cell type random effect
        ct_ovaerall_var:    overall variance explained by cell type-specific effect
        fixed_var:  dict of variances explained by each fixed effect feature, including cell type-specific fixed effect
        random_var: dict of variances explained by each random effect feature, doesn't include cell type-shared or -specific effect
        cut:    threshold for large variance
    Returns:
        True:   optim failed to converge
        False:  optim successfully to converge
    '''
    if ( (opt['l'] < -1e10) or (not opt['success']) or (hom2 > cut) or (ct_overall_var > cut) or
            np.any(np.array(list(fixed_vars.values())) > cut) or
            np.any(np.array(list(random_vars.values())) > cut) ):
        return True
    else:
        return False

def re_optim(out: object, opt: dict, fun: callable, par: list, args: tuple, method: str, nrep: int=10) -> Tuple[object, dict]:
    '''
    Rerun optimization

    Parameters:
        out:    OptimizeResult object
        opt:    opmization results, e.g. method used, log-likelihood
        fun:    objective function to minimize
        par:    initial parameters used in the first try of optimization
        args:   extra argument passed to the objective function
        method: optimization method, e.g. BFGS
        nrep:   number of optimization repeats
    Returns:
        a tuple of 
            #. OptimizeResult of the best optimization
            #. results of the best optimization
    '''
    rng = default_rng()
    #print( out['fun'] )
    for i in range(nrep):
        par_ = np.array(par) * rng.gamma(2,1/2,len(par))
        out_, opt_ = optim(fun, par_, args=args, method=method)
        print( out_['fun'] )
        if (not out['success']) and out_['success']:
            out, opt = out_, opt_
        elif (out['success'] == out_['success']) and (out['fun'] > out_['fun']):
            out, opt = out_, opt_
    #print( out['fun'] )
    return( out, opt )

def dict2Rlist( X: dict ) -> object:
    '''
    Transform a python dictionary to R list

    Parameters:
        X:  python dictionary
    Returns:
        R list
    '''
    if len( X.keys() ) == 0:
        return( r('NULL') )
    else:
        keys = np.sort( list(X.keys()) )
        rlist = ro.ListVector.from_length( len(keys) )
        for i in range( len(keys) ):
            if isinstance(X[keys[i]], str):
                if os.path.exists( X[keys[i]] ):
                    rlist[i] = r['as.matrix']( r['read.table'](X[keys[i]]) )
                else:
                    try:
                        rlist[i] = np.array( [X[keys[i]]] )
                    except:
                        numpy2ri.activate()
                        rlist[i] = np.array( [X[keys[i]]] )
                        numpy2ri.deactivate() # deactivate would cause numpy2ri deactivated in calling fun
            elif isinstance(X[keys[i]], pd.DataFrame):
                with localconverter(ro.default_converter + pandas2ri.converter):
                    rlist[i] = r['as.matrix']( X[keys[i]] )
            elif isinstance(X[keys[i]], np.ndarray):
                try:
                    rlist[i] = r['as.matrix']( X[keys[i]] )
                except:
                    numpy2ri.activate()
                    rlist[i] = r['as.matrix']( X[keys[i]] )
                    numpy2ri.deactivate()
            elif isinstance(X[keys[i]], int) or isinstance(X[keys[i]], float):
                try:
                    rlist[i] = np.array( [X[keys[i]]] )
                except:
                    numpy2ri.activate()
                    rlist[i] = np.array( [X[keys[i]]] )
                    numpy2ri.deactivate()
        return( rlist )

def generate_HE_initial(he: dict, ML: bool=False, REML: bool=False) -> list:
    '''
    Convert HE estimates to initial parameter for ML / REML

    Parameters:
        he: estiamtes from HE
        ML: generate initial parameters for ML 
        REML:   generate initial parameters for REML
    Returns:
        initial parameters for ML / REML
    '''
    initials_random_effects = []
    # homogeneous effect
    if 'hom2' in he.keys():
        initials_random_effects.append( he['hom2'] )
    # heterogeneous effect
    if 'V' in he.keys():
        C = he['V'].shape[0]
        # determine model based on V
        if np.any( np.diag(np.diag(he['V'])) != he['V'] ):
            # Full model
            initials_random_effects = initials_random_effects + list(he['V'][np.triu_indices(C)])
        elif len( np.unique( np.diag(he['V']) ) ) == 1:
            # IID model
            initials_random_effects.append( he['V'][0,0] )
        else:
            # Free model
            initials_random_effects = initials_random_effects + list(np.diag(he['V']))
    # other random covariates
    if 'r2' in he.keys():
        for key in np.sort( list(he['r2'].keys()) ):
            initials_random_effects.append( he['r2'][key] )

    if REML is True:
        return( initials_random_effects )

    initials_fixed_effects = list(he['beta']['ct_beta'])
    for key in np.sort( list(he['beta'].keys()) ):
        if key != 'ct_beta':
            initials_fixed_effects = initials_fixed_effects + list( he['beta'][key] )

    if ML is True:
        return( initials_fixed_effects + initials_random_effects )

def glse( sig2s: np.ndarray, X: np.ndarray, y: np.ndarray, inverse: bool=False ) -> np.ndarray:
    '''
    Generalized least square estimates

    Parameters:
        sig2s:  covariance matrix of y, pseudobulk
        X:  desing matrix for fixed effects
        y:  pseudobulk
        inverse:    is sig2s inversed
    Returns:
        GLS of fixed effects
    '''
    if not inverse:
        if len( sig2s.shape ) == 1:
            sig2s_inv = 1/sig2s
            A = X.T * sig2s_inv
        else:
            sig2s_inv = np.linalg.inv( sig2s )
            A = X.T @ sig2s_inv
    else:
        sig2s_inv = sig2s
        A = X.T @ sig2s_inv
    B = A @ X
    beta = np.linalg.inv(B) @ A @ y
    return( beta )

def FixedeffectVariance_( beta: np.ndarray, x: np.ndarray ) -> float:
    '''
    Estimate variance explained by fixed effect

    Parameters:
        beta:   fixed effect sizes
        x:  design matrix of fixed effect
    Returns:
        variance explained by fixed effect
    '''
    #xd = x - np.mean(x, axis=0)
    #s = ( xd.T @ xd ) / x.shape[0]
    s = np.cov( x, rowvar=False )
    if len(s.shape) == 0:
        s = s.reshape(1,1)
    return( beta @ s @ beta )

def FixedeffectVariance( beta: np.ndarray, xs: np.ndarray ) -> list:
    '''
    Estimate variance explained by each feature of fixed effect, e.g. cell type, sex

    Parameters:
        beta:   fixed effect sizes
        xs: design matrices for fixed effects
    Returns:
        variances
    '''
    j = 0
    vars = []
    for i,x in enumerate(xs):
        var = FixedeffectVariance_( beta[j:(j+x.shape[1])], x)
        vars.append(var)
        j = j + x.shape[1]
    return( vars )

def fixedeffect_vars(beta: np.ndarray, P: np.ndarray, fixed_covars_d: dict) -> Tuple[dict, dict]:
    '''
    Estimate variance explained by each feature of fixed effect, e.g. cell type, sex

    Parameters:
        beta:   fixed effect sizes
        P:  cell type proportions
        fixed_covars_d: design matrices for fixed effects
    Returns:
        a tuple of 
            #. dict of fixed effects
            #. dict of variances explained
    '''
    # read covars if needed
    fixed_covars_d = read_covars(fixed_covars_d, {})[0]

    beta_d = assign_beta(beta, P, fixed_covars_d)

    fixed_vars_d = {'ct_beta': FixedeffectVariance_( beta_d['ct_beta'], P) }
    for key in fixed_covars_d.keys():
        fixed_vars_d[key] = FixedeffectVariance_( beta_d[key], fixed_covars_d[key] )

#    fixed_covars_l = [P]
#    for key in np.sort(list(fixed_covars_d.keys())):
#        m_ = fixed_covars_d[key]
#        if isinstance( m_, str ):
#            m_ = np.loadtxt( m_ )
#        if len( m_.shape ) == 1:
#            m_ = m_.reshape(-1,1)
#        fixed_covars_l.append( m_ )
#
#    fixedeffect_vars_l = FixedeffectVariance(beta, fixed_covars_l)
#
#    fixedeffect_vars_d = assign_fixedeffect_vars(fixedeffect_vars_l, fixed_covars_d)

    return(beta_d, fixed_vars_d)

def assign_beta(beta_l: list, P: np.ndarray, fixed_covars_d: dict) -> dict:
    '''
    Convert a list of fixed effect to dict for each feature

    Parameters:
        beta_l: fixed effects
        P:  cell type proportions
        fixed_covars_d: design matrices for fixed effects
    Returns:
        dict of fixed effects
    '''
    beta_d = { 'ct_beta': beta_l[:P.shape[1]] }
    beta_l = beta_l[P.shape[1]:]

    for key in np.sort(list(fixed_covars_d.keys())):
        x = fixed_covars_d[key] 
        if len( x.shape ) == 1:
            x = x.reshape(-1,1)
        beta_d[key] = beta_l[:x.shape[1]]
        beta_l = beta_l[x.shape[1]:]

    return(beta_d)

def assign_fixedeffect_vars(fixedeffect_vars_l: list, fixed_covars_d: dict) -> dict:
    '''
    Assign fixed effect variance to each feature

    Parameters:
        fixedeffect_vars_l: fixed effects variances
        fixed_covars_d: design matrices for fixed effects
    Returns:
        fixed effects variances for each feature
    '''
    fixedeffect_vars_d = {'celltype_main_var': fixedeffect_vars_l[0]}
    if len(fixed_covars_d.keys()) > 0:
        for key, value in zip(np.sort(list(fixed_covars_d.keys())), fixedeffect_vars_l[1:]):
            fixedeffect_vars_d[key] = value
    return(fixedeffect_vars_d)

def RandomeffectVariance_( V: np.ndarray, X: np.ndarray ) -> float:
    '''
    Compute variance of random effect

    Parameters:
        V:  covariance matrix of random effect
        X:  design matrix
    Returns:
        variance explained
    '''
    return( np.trace( V @ (X.T @ X) ) / X.shape[0] )

def RandomeffectVariance( Vs: Union[list, dict], Xs: Union[list, dict] ) -> Union[list, dict]:
    if isinstance( Xs, list ):
        if len( np.array( Vs ).shape ) == 1:
            Vs = [V * np.eye(X.shape[1]) for V, X in zip(Vs, Xs)]

        vars = [RandomeffectVariance_(V,X) for V,X in zip(Vs, Xs)]
    elif isinstance( Xs, dict ):
        vars = {}
        for key in Xs.keys():
            V, X = Vs[key], Xs[key]
            if isinstance(V, float):
                V = V  * np.eye(X.shape[1])
            vars[key] = RandomeffectVariance_(V,X)
    return( vars )

def assign_randomeffect_vars(randomeffect_vars_l: list, r2_l: list, random_covars_d: dict) -> Tuple[dict, dict]:
    '''
    Assign variance of random effects
    '''
    randomeffect_vars_d = {}
    r2_d = {}
    keys = np.sort( list(random_covars_d.keys()) )
    if len(keys) != 0:
        for key, v1, v2 in zip( keys, randomeffect_vars_l, r2_l ):
            randomeffect_vars_d[key] = v1
            r2_d[key] = v2

    return( randomeffect_vars_d, r2_d )

def ct_randomeffect_variance( V: np.ndarray, P: np.ndarray ) -> Tuple[float, np.ndarray]:
    '''
    Compute overall and specific variance of each cell type
    
    Parameters:
        V:  cell type-specific random effect covariance matrix
        P:  cell type proportions
    Returns:
        A tuple of
            #. overall variance
            #. cell type-specific variance
    '''
    N, C = P.shape
    ct_overall_var = RandomeffectVariance_(V, P)
    ct_specific_var = np.array([V[i,i] * ((P[:,i]**2).mean()) for i in range(C)])

    return( ct_overall_var, ct_specific_var )

def cal_variance(beta: np.ndarray, P: np.ndarray, fixed_covars: dict, r2: Union[list, np.ndarray, dict], random_covars: dict
        ) -> Tuple[dict, dict, dict, dict]:
    '''
    Compute variance explained by fixed effects and random effects

    Parameters:
        beta:   fixed effects
        P:  cell type propotions
        fixed_covars: design matrices for additional fixed effects
        r2: variances of additional random effects
        random_covars:  design matrices for additional random effects
    '''
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars = fixedeffect_vars( beta, P, fixed_covars ) # fixed effects are always ordered
    if isinstance(r2, list) or isinstance(r2, np.ndarray):
        r2 = dict(zip( np.sort(list(random_covars.keys())), r2 ))
    random_vars = RandomeffectVariance( r2, random_covars )
    return( beta, fixed_vars, r2, random_vars )

#def quantnorm(Y, axis=0):
#    '''
#    # use sklearn.preprocessing.quantile_transform
#    '''
#    pass

def wald_ct_beta(beta: np.ndarray, beta_var: np.ndarray, n: int, P: int) -> float:
    '''
    Wald test on mean expression differentiation

    Parameters:
        beta:   cell type-specific mean expressions
        beta_var:   covariance matrix of cell type-specific mean
        n:  sample size (for Ftest in Wald test)
        P:  number of estimated parameters (for Ftest in Wald test)
    Returns:
        p value for Wald test on mean expression differentiation
    '''
    C = len(beta)
    T = np.concatenate( ( np.eye(C-1), (-1)*np.ones((C-1,1)) ), axis=1 )
    beta = T @ beta
    beta_var = T @ beta_var @ T.T
    return(wald.mvwald_test(beta, np.zeros(C-1), beta_var, n=n, P=P))

def check_R(R: np.ndarray) -> bool:
    '''
    Check R matrix: has to be matrix of 0 and 1
    in the structure of scipy.linalg.block_diag(np.ones((a,1)), np.ones((b,1)), np.ones((c,1))
    '''
    # infer matrix R
    xs = np.sum(R, axis=0).astype('int')
    R_ = np.ones((xs[0],1))
    for i in range(1,len(xs)):
        R_ = linalg.block_diag(R_, np.ones((xs[i],1)))

    if np.any(R != R_):
        print(R[:5,:])
        print(R_[:5,:])
        return( False )
    else:
        return( True )

def order_by_randomcovariate(R: np.ndarray, Xs: list=[], Ys: dict={}
        ) -> Tuple[pd.Index, np.ndarray, list, dict]:
    """
    R is the design matrix of 0 and 1 for a random covriate, which we order along by
    Xs or Ys: a list or dict of matrixs we want to order
    """

    R_df = pd.DataFrame(R)
    index = R_df.sort_values(by=list(R_df.columns), ascending=False).index
    # R2 = np.take_along_axis(R, np.broadcast_to(index, (R.shape[1], R.shape[0])).T, axis=0)
    R = R[index, :]
    # if np.any(R != R2):
        # sys.exit('Not the same')
    if not check_R(R):
        sys.exit('Matrix R is wrong!\n')

    new_Xs = []
    for X in Xs:
        if len(X.shape) > 1:
            # X = np.take_along_axis(X, np.broadcast_to(index, (X.shape[1], X.shape[0])).T, axis=0)
            X = X[index, :]
        else:
            # X = np.take_along_axis(X, index, axis=0)
            X = X[index]
        new_Xs.append(X)

    new_Ys = {}
    for key in Ys.keys():
        Y = Ys[key]
        if len(Y.shape) > 1:
            # Y = np.take_along_axis(Y, np.broadcast_to(index, (Y.shape[1], Y.shape[0])).T, axis=0)
            Y = Y[index, :]
        else:
            # Y = np.take_along_axis(Y, index, axis=0)
            Y = Y[index]
        new_Ys[key] = Y

    return index, R, new_Xs, new_Ys


def jk_rmInd(i: Union[int, np.ndarray], Y: np.ndarray, vs: np.ndarray, fixed_covars: dict={}, random_covars: dict={}, P: Optional[np.ndarray]=None
        ) -> tuple:
    '''
    Remove one individual from the matrices for jackknife
    '''
    Y_ = np.delete(Y, i, axis=0)
    vs_ = np.delete(vs, i, axis=0)
    fixed_covars_ = {}
    for key in fixed_covars.keys():
        fixed_covars_[key] = np.delete(fixed_covars[key], i, axis=0)
    random_covars_ = {}
    for key in random_covars.keys():
        random_covars_[key] = np.delete(random_covars[key], i, axis=0)
    if P is None:
        return(Y_, vs_, fixed_covars_, random_covars_)
    else:
        P_ = np.delete(P, i, axis=0)
        return Y_, vs_, fixed_covars_, random_covars_, P_


def lrt(l: float, l0: float, k: int) -> float:
    '''
    Perfomr Likelihood-ration test (LRT)

    Parameters:
        l, l0:  log likelihood for alternative and null hypothesis models
        k:  number of parameters constrained in null model compared to alternative
    Returns:
        p value for LRT
    '''

    Lambda = 2 * (l-l0)
    p = stats.chi2.sf(Lambda, k)
    return p

def generate_tmpfn():
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    tmpfn = tmpf.name
    tmpf.close()
    log.logger.info(tmpfn)
    return tmpfn


def sim_pseudobulk(beta: np.ndarray, hom2: float, ctnu: np.ndarray, n: int, C: int, V: np.ndarray=None, seed: int=None) -> np.ndarray:
    """
    Simulate CTP from model

    Parameters:
        beta:   cell type fixed effect
        hom2:   variance of homogeneous effect
        ctnu:   variance of residual effect of shape n X C
        n:  number of simulated individuals
        C:  number of simulated CTs
        seed:   seed for generating random effect
    
    Returns:
        CTP
    """

    rng = np.random.default_rng(seed)

    # homogeneous effect
    if hom2 < 0:
        hom2 = 0
    alpha = rng.normal(scale=np.sqrt(hom2), size=n)

    # residual effect
    delta = rng.normal(np.zeros_like(ctnu), np.sqrt(ctnu))

    # pseudobulk
    ctp = beta[np.newaxis, :] + alpha[:, np.newaxis] + delta

    if V:
        ctp += rng.multivariate_normal(np.zeros(C), V, size=(n, C))

    return ctp


def _sim_sc(ss: int, C: int, cell_no: int, cty: np.ndarray, cell_var: np.ndarray, 
            k: float, d: float, depth: float, rng: object) -> np.ndarray:
    """
    ss: sample size
    C:  cell type number
    cell_no:    simulated cells per ind-ct
    cty:    cy pseudobulk
    cell_var:   cell level noise
    k:  CTP was scaled to OP mean 0 var 1, now scale back by sim_CTP * k + d
    d:  CTP was scaled to OP mean 0 var 1, now scale back by sim_CTP * k + d
    depth:  simulate read depth relative to 1M reads. e.g. depth = 0.1 -> 0.1M reads per cell
    rng:    random number generator

    Returns:
        simulated cell expression
    """

    # cell noise
    epsilon = rng.normal(0, np.sqrt(cell_var)[:, :, np.newaxis], (ss, C, cell_no))

    # cell expression
    cell_y = cty[:, :, np.newaxis] + epsilon

    # scale back
    cell_y = cell_y * k + d

    # shift min to 0
    if np.min(cell_y) < 0:
        cell_y -= np.min(cell_y)  

    # trans to count per million
    cpm = 2 ** cell_y - 1

    # read depth
    counts = cpm * depth

    # poisson distribution to simulate counts
    counts = rng.poisson(counts)

    # transform counts to cpm
    cpm = counts / depth

    # gene expression
    cell_y = np.log2(cpm + 1)

    return cell_y


# def sim_sc(beta: np.ndarray, hom2: float, V_hat: np.ndarray, V: np.ndarray, ctnu: np.ndarray, n: np.ndarray, k: float, d: float, depth: float, 
#            cell_no: int, seed: int=None, option: int=0) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Simulate gene expression for single cells

#     Parameters:
#         beta:   cell type fixed effect
#         hom2:   variance of homogeneous effect
#         V_hat:  estimated covariance of cell type specific effect from real data
#         V:  simulated covariance of cell type specific effect
#         ctnu:   variance of residual effect of shape n X C
#         n:  number of cells per individual - ct 
#         k:  CTP was scaled to OP mean 0 var 1, now scale back by sim_CTP * k + d
#         d:  CTP was scaled to OP mean 0 var 1, now scale back by sim_CTP * k + d
#         depth:  simulate read depth relative to 1M reads. e.g. depth = 0.1 -> 0.1M reads per cell
#         cell_no:    number of cells to simulate per individual-cell type pair
#         seed:   seed for generating random effect
    
#     Returns:
#         CTP
#     """

#     rng = np.random.default_rng(seed)
    
#     ss, C = ctnu.shape

#     # find cell type with highest beta
#     c = np.argmax(beta)

#     if option == 1:
#         if hom2 < 0:
#             hom2 = 0
#         beta = np.ones_like(beta) * beta[c]  # beta
#         V += hom2
#         cty = beta[np.newaxis, :] + rng.multivariate_normal(np.zeros(C), V, size=ss)

#     elif option == 2:
#         beta = np.ones_like(beta) * beta[c]  # beta
#         if np.all(V == 0):
#             hom2 += V_hat[c, c]
#             if hom2 < 0:
#                 hom2 = 0
#             alpha = rng.normal(scale=np.sqrt(hom2), size=ss)
#             cty = beta[np.newaxis, :] + alpha[:, np.newaxis]
#         else:
#             if hom2 < 0:
#                 hom2 = 0
#             V += hom2
#             cty = beta[np.newaxis, :] + rng.multivariate_normal(np.zeros(C), V, size=ss)
            
#     elif option == 3:
#         beta = np.zeros_like(beta)  # beta
#         if np.all(V == 0):
#             hom2 += V_hat[c, c]
#             if hom2 < 0:
#                 hom2 = 0
#             alpha = rng.normal(scale=np.sqrt(hom2), size=ss)
#             cty = beta[np.newaxis, :] + alpha[:, np.newaxis]
#         else:
#             if hom2 < 0:
#                 hom2 = 0
#             V += hom2
#             cty = beta[np.newaxis, :] + rng.multivariate_normal(np.zeros(C), V, size=ss)
#         # alpha = rng.normal(scale=np.sqrt(hom2), size=ss)

#     # residual effect
#     cell_var = ctnu * n
#     if option in [1, 2, 3]:
#         # duplicate ctnu from main cell type
#         cell_var = np.ones_like(cell_var) * cell_var[:, [c]]

#     # simulate
#     cell_y = _sim_sc(ss, C, cell_no, cty, cell_var, k, d, depth, rng)
#     sim_cty = np.mean(cell_y, axis=2)

#     # use a large number of cells to compute real ctnu
#     cell_y = _sim_sc(ss, C, 10000, cty, cell_var, k, d, depth, rng)
#     cell_var = np.var(cell_y, axis=2)
#     ctnu = cell_var / cell_no

#     # scale back
#     sim_cty = (sim_cty - d) / k
#     ctnu = ctnu / (k**2)

#     return sim_cty, ctnu


def sim_sc_bootstrap(data: pd.DataFrame, C: int, frac: float, depth: float, prop: float, seed: int, option: int=2,
                     pseudocount: float=1, resample_inds: bool=True, cell_count: bool=False, 
                     mean_difference: Optional[float]=None, log_scale: bool=True, prior: bool=False, ngene: Optional[int]=None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate gene expression for single cells by bootstrap

    Parameters:
        data:   raw counts per cell. with columns: cell, total_counts, gene (for gene counts), ind
        C:  number of cell types
        frac:   fraction/count of cells to sample
        depth:  sequencing depth relative to real data
        pseudocount:    pseudocount for log transformation
        prop:   proportion of individuals to shuffle, to create cell type specific variance
        seed:   seed for random generator
        option: 1. samples of the same cell have different counts; 2. samples of the same cells have the same counts; 
                3. 2 + shuffle cells across individuals
        resample_inds:  resample individuals with 0 counts
        cell_count: whether frac if fraction (False) or count (True)
        mean_difference:    create mean difference = log2(fold change) between cell types
        log_scale:  make fold change at log-transformed expression
        prior:  whether to incorporate prior beta dist (1, # genes) of gene's read proportion
        ngene:  used in prior beta distribution
    
    Returns:
        CTP 
        CTNU
        sim_data: three columns: ind, ct, gene_sim; each row is a simulated cell's gene expression
    """

    rng = np.random.default_rng(seed)
    cts = [f'ct{i+1}' for i in range(C)]

    if mean_difference:
        fold_change = 2 ** mean_difference
    else:
        fold_change = None

    def binomial(row, depth, fold_change=1, prior=False, ngene=None):
        if prior:
            x = rng.binomial(int(row['total_counts'] * depth), (row['gene'] + 1) * fold_change / (row['total_counts'] + ngene))
        else:
            x = rng.binomial(int(row['total_counts'] * depth), row['gene'] * fold_change / row['total_counts'])

        return x

    if option == 1:

        # sample cells
        grouped = data.groupby('ind')
        cells = []
        for i in range(C):
            ct = cts[i]
            if cell_count:
                sampled_cells = grouped.sample(n=int(frac), replace=True, random_state=rng.integers(100000))
            else:
                sampled_cells = grouped.sample(frac=frac, replace=True, random_state=rng.integers(100000))
            sampled_cells['ct'] = ct
            cells.append(sampled_cells)

        sim_data = pd.concat(cells, ignore_index=True)

        # sample counts per cell with binomial
        sim_data['gene_sim'] = sim_data.apply(binomial, axis=1, depth=depth, prior=prior, ngene=ngene)
        if fold_change and (not log_scale):
            sim_data['gene_sim2'] = sim_data.apply(binomial, axis=1, depth=depth, prior=prior, ngene=ngene, fold_change=fold_change)
            sim_data.loc[sim_data['ct'] == cts[-1], 'gene_sim'] = sim_data.loc[sim_data['ct'] == cts[-1], 'gene_sim2']
            sim_data = sim_data.drop(columns='gene_sim2')
        sim_count = sim_data.copy()  # save counts
        
        # log transformation
        sim_data['gene_sim'] = np.log2(sim_data['gene_sim'] * 1e6 / (sim_data['total_counts'] * depth) + pseudocount)
        if fold_change and log_scale:
            sim_data.loc[sim_data['ct'] == cts[-1], 'gene_sim'] = sim_data.loc[sim_data['ct'] == cts[-1], 'gene_sim'] + np.log2(fold_change)

        # calculate ct pseudobulk
        cty = sim_data.groupby(['ind', 'ct'])['gene_sim'].mean().unstack()

        # expected nu
        ## mean and mean of squares per cell
        def mean_f(row, depth, fold_change=1, prior=False, ngene=None, log_scale=None):
            if log_scale:
                fold_change = 1

            if prior:
                counts = rng.binomial(int(row['total_counts'] * depth), (row['gene'] + 1) * fold_change / (row['total_counts'] + ngene), 10000)
            else:
                counts = rng.binomial(int(row['total_counts'] * depth), row['gene'] * fold_change / row['total_counts'], 10000)
                
            return counts.mean(), (counts ** 2).mean()

        data[['mean', 'mean2']] = data.apply(mean_f, axis=1, result_type='expand', depth=depth, prior=prior, ngene=ngene)

        ## mean and mean of squares per ind
        ind_data = data.groupby('ind')[['mean', 'mean2']].mean()

        ## cell var per ind
        ind_data['cell_var'] = ind_data['mean2'] - ind_data['mean'] ** 2

        ## ctnu
        if cell_count:
            cell_no = frac
        else:
            cell_no = (data.groupby('ind').size() * frac).astype('int')
        ctnu = ind_data['cell_var'] / cell_no
        ctnu = ctnu[cty.index]
        ctnu = pd.concat([ctnu] * C, axis=1)
        ctnu.columns = cts   

        # update ctnu if mean difference
        if fold_change and (not log_scale):
            data[['mean', 'mean2']] = data.apply(mean_f, axis=1, result_type='expand', depth=depth, prior=prior, ngene=ngene,
                                                 fold_change=fold_change, log_scale=log_scale)
            ind_data = data.groupby('ind')[['mean', 'mean2']].mean()
            ind_data['cell_var'] = ind_data['mean2'] - ind_data['mean'] ** 2
            ctnu2 = ind_data['cell_var'] / cell_no
            ctnu2.name = cts[-1]
            ctnu = ctnu.drop(columns=cts[-1])
            ctnu = ctnu.merge(ctnu2.to_frame(), left_index=True, right_index=True)

    elif option in [2, 3]:
        if option == 3:
            # shuffle individuals
            data['ind'] = rng.permutation(data['ind'])

        # sample counts per cell with binomial
        data['gene_sim'] = data.apply(binomial, axis=1, depth=depth, 
                                      prior=prior, ngene=ngene)
        if fold_change and (not log_scale):
            data['gene_sim2'] = data.apply(binomial, axis=1, depth=depth, 
                                           fold_change=fold_change, prior=prior,
                                           ngene=ngene)

        # resample inds with 0 counts
        max_counts = data.groupby('ind')['gene_sim'].max()
        mis_inds = max_counts[max_counts == 0].index.to_numpy()

        log.logger.info(f'{len(mis_inds)} have 0 counts')

        while (len(mis_inds) > 0) and resample_inds:
            data.loc[data['ind'].isin(mis_inds), 'gene_sim'] = data.loc[data['ind'].isin(mis_inds)].apply(binomial, axis=1, depth=depth)
            max_counts = data.groupby('ind')['gene_sim'].max()
            mis_inds = max_counts[max_counts == 0].index.to_numpy()


        # sample cells
        grouped = data.groupby('ind')
        cells = []
        x = []
        for i in range(C):
            ct = cts[i]
            random_state = rng.integers(100000)
            while random_state in x:
                random_state = rng.integers(100000)
            x.append(random_state)
            if cell_count:
                sampled_cells = grouped.sample(n=int(frac), replace=True, random_state=random_state)
            else:
                sampled_cells = grouped.sample(frac=frac, replace=True, random_state=random_state)
            sampled_cells['ct'] = ct
            cells.append(sampled_cells)

        sim_count = pd.concat(cells, ignore_index=True)
        sim_data = sim_count.copy()

        # log transformation
        sim_data['gene_sim'] = np.log2(sim_data['gene_sim'] * 1e6 / (sim_data['total_counts'] * depth) + pseudocount)
        if fold_change:
            if log_scale:
                sim_data.loc[sim_data['ct'] == cts[-1], 'gene_sim'] = sim_data.loc[sim_data['ct'] == cts[-1], 'gene_sim'] + np.log2(fold_change)
            else:
                sim_data['gene_sim2'] = np.log2(sim_data['gene_sim2'] * 1e6 / (sim_data['total_counts'] * depth) + pseudocount)
                sim_data.loc[sim_data['ct'] == cts[-1], 'gene_sim'] = sim_data.loc[sim_data['ct'] == cts[-1], 'gene_sim2']

        # calculate ct pseudobulk
        cty = sim_data.groupby(['ind', 'ct'])['gene_sim'].mean().unstack()

        # expected nu
        data['gene_sim'] = np.log2(data['gene_sim'] * 1e6 / (data['total_counts'] * depth) + pseudocount)
        if fold_change and (not log_scale):
            data['gene_sim2'] = np.log2(data['gene_sim2'] * 1e6 / (data['total_counts'] * depth) + pseudocount)
        grouped = data.groupby('ind')
        cell_var = grouped['gene_sim'].var()
        if cell_count:
            ctnu = cell_var / frac
        else:
            cell_no = (grouped.size() * frac).astype('int')
            ctnu = cell_var / cell_no
        ctnu = pd.concat([ctnu] * C, axis=1)
        ctnu.columns = cts

        if fold_change and (not log_scale):
            cell_var2 = grouped['gene_sim2'].var()
            if cell_count:
                ctnu2 = cell_var2 / frac
            else:
                cell_no = (grouped.size() * frac).astype('int')
                ctnu2 = cell_var2 / cell_no
            ctnu2 = pd.concat([ctnu2] * C, axis=1)
            ctnu2.columns = cts

            if ctnu.index.equals(ctnu2.index):
                ctnu[cts[-1]] = ctnu2[cts[-1]]
            else:
                sys.exit('Not matching')

    # santiy check
    if cty.shape[0] != ctnu.shape[0]:
        sys.exit('Missing inds!')

    # make free model
    if prop != 0:
        # find inds to shuffle
        cty_to_shuffle = cty.sample(frac=prop, random_state=rng.integers(100000))
        index_to_shuffle = cty_to_shuffle.index
        index_to_fix = cty.index.difference(index_to_shuffle)
        for ct in cts:
            shuffled_index = cty_to_shuffle.sample(frac=1, random_state=rng.integers(100000)).index
            new_index = shuffled_index.union(index_to_fix, sort=False)
            cty[ct] = cty.loc[new_index, ct].to_numpy()
            ctnu[ct] = ctnu.loc[new_index, ct].to_numpy()

    return cty, ctnu, sim_count[['ind', 'ct', 'gene_sim']]


# def _sim_sc_bootstrap(group, frac, seed):
#     """
#     Define a function to perform random choice and calculate ctp and ctnu for each group
#     """

#     sampled_group = group.sample(n=int(len(group) * frac), replace=True, random_state=seed)  # can add a threshold here. e.g. keep all cells if the ct has < 10 cells
#     ctp = sampled_group.mean()
#     # ctnu = sampled_group.var() / len(group)
#     return ctp


# def sim_sc_bootstrap1(data: pd.DataFrame, C: int, gene: str, frac: float, prop: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Simulate gene expression for single cells by bootstrap

#     Parameters:
#         data:   DF with two columns: donor and gene. donor: individuals. gene: gene expression.
#                 each row is gene expression for a cell
#         C:  number of cell types
#         gene:   column name of the gene
#         frac:   fraction of cells to sample
#         prop:   proportion of individuals to shuffle, to create cell type specific variance
#         seed:   seed for random generator
    
#     Returns:
#         CTP, CTNU
#     """

#     rng = np.random.default_rng(seed)
#     cts = [f'ct{i+1}' for i in range(C)]

#     # simulate
#     grouped = data.groupby('donor')
    
#     sim = [grouped.apply(_sim_sc_bootstrap, frac, rng.integers(100000)) for c in range(C)]

#     cty = pd.concat(sim, axis=1)
#     cty.columns = cts

#     # expected nu
#     cell_var = grouped[gene].agg(np.var)
#     cell_no = (grouped.size() * frac).astype('int')
#     ctnu = cell_var / cell_no
#     ctnu = pd.concat([ctnu] * C, axis=1)
#     ctnu.columns = cts

#     # santiy check
#     if cty.index.equals(ctnu.index):
#         pass
#     else:
#         sys.exit('Wrong order of inds')

#     # make free model
#     if prop != 0:
#         # find inds to shuffle
#         cty_to_shuffle = cty.sample(frac=prop, random_state=rng.integers(100000))
#         index_to_shuffle = cty_to_shuffle.index
#         index_to_fix = cty.index.difference(index_to_shuffle)
#         for ct in cts:
#             shuffled_index = cty_to_shuffle.sample(frac=1, random_state=rng.integers(100000)).index
#             new_index = shuffled_index.union(index_to_fix, sort=False)
#             cty[ct] = cty.loc[new_index, ct].to_numpy()
#             ctnu[ct] = ctnu.loc[new_index, ct].to_numpy()

#     return cty, ctnu


def age_group(age: pd.Series):
    """
    Separate age groups
    """
    bins = np.arange(25, 91, 5)
    new = pd.Series(np.digitize(age, bins), index=age.index)
    if age.name is None:
        return new
    else:
        return new.rename(age.name)


def design(inds: npt.ArrayLike, pca: pd.DataFrame = None, PC: int = None, cat: pd.Series = None,
           con: pd.Series = None, drop_first: bool = True) -> np.ndarray:
    """
    Construct design matrix

    Parameters:
        inds:   order of individuals
        pca:    dataframe of pcs, with index: individuals (sort not required) and columns (PC1-PCx)
        PC: number to PC to adjust
        cat:    series of category elements e.g. sex: male and female
        con:    series of continuous elements e.g. age
        drop_first: drop the first column

    Returns:
        a design matrix
    """

    # pca
    if pca is not None:
        pcs = [f'PC{i}' for i in range(1, int(PC) + 1)]
        return pca.loc[inds, pcs].to_numpy()
    elif cat is not None:
        return pd.get_dummies(cat, drop_first=drop_first, dtype='int').loc[inds, :].to_numpy()
    elif con is not None:
        return con[inds].to_numpy()


def inverse_block_diag_matrix(X: np.ndarray) -> np.ndarray:
    """
    Inverse a block diagonal square matrix
    """

    # find blocks
    blocks = []

    i = 0
    while True:
        index = max(np.nonzero(X[i, :])[0]) + 1
        blocks.append(X[i:index, i:index])
        i = index
        if i == X.shape[1]:
            break

    # sanity check
    if np.any(linalg.block_diag(*blocks) != X):
        sys.exit('Not a block diag matrix!')

    invs = []
    for block in blocks:
        if wald.check_singular(block):
            np.savetxt('block.tmp.txt', block, '%.3f')
            sys.exit('Singular in inverse block diag matrix!')
        else:
            invs.append(np.linalg.inv(block))
    
    return linalg.block_diag(*invs)
