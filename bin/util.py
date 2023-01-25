import os
import numpy as np, pandas as pd
import scipy
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from scipy import stats, linalg, optimize
from numpy.random import default_rng
import wald

def read_covars(fixed_covars={}, random_covars={}):
    covars = [fixed_covars, random_covars]
    for i in range(len(covars)):
        tmp = {}
        for key in covars[i].keys():
            f = covars[i][key]
            if isinstance( f, str ):
                tmp[key] = np.loadtxt( f )
            else:
                tmp[key] = f
        covars[i] = tmp
    return( covars )

def optim(fun, par, args, method):
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

def check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars, cut=5):
    if ( (opt['l'] < -1e10) or (not opt['success']) or (hom2 > cut) or (ct_overall_var > cut) or
            np.any(np.array(list(fixed_vars.values())) > cut) or
            np.any(np.array(list(random_vars.values())) > cut) ):
        return True
    else:
        return False

def re_optim(out, opt, fun, par, args, method, nrep=10):
    rng = default_rng()
    print( out['fun'] )
    for i in range(nrep):
        par_ = np.array(par) * rng.gamma(2,1/2,len(par))
        out_, opt_ = optim(fun, par_, args=args, method=method)
        print( out_['fun'] )
        if (not out['success']) and out_['success']:
            out, opt = out_, opt_
        elif (out['success'] == out_['success']) and (out['fun'] > out_['fun']):
            out, opt = out_, opt_
    return( out, opt )

def dict2Rlist( X, order=True ):
    '''
    X: a python dictionary
    '''
    if len( X.keys() ) == 0:
        return( r('NULL') )
    else:
        keys = np.sort( list(X.keys()) ) if order else list(X.keys())
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

def generate_HE_initial(he, ML=False, REML=False):
    '''
    he: dict
        estiamtes from HE
    ML / REML: boolen
        return initials for ML / REML
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

def glse( sig2s, X, y, inverse=False ):
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

def FixedeffectVariance_( beta, x ):
    #xd = x - np.mean(x, axis=0)
    #s = ( xd.T @ xd ) / x.shape[0]
    s = np.cov( x, rowvar=False )
    return( beta @ s @ beta )

def FixedeffectVariance( beta, xs ):
    j = 0
    vars = []
    for i,x in enumerate(xs):
        var = FixedeffectVariance_( beta[j:(j+x.shape[1])], x)
        vars.append(var)
        j = j + x.shape[1]
    return( vars )

def fixedeffect_vars(beta, P, fixed_covars_d):
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

def assign_beta(beta_l, P, fixed_covars_d):
    beta_d = { 'ct_beta': beta_l[:P.shape[1]] }
    beta_l = beta_l[P.shape[1]:]

    for key in np.sort(list(fixed_covars_d.keys())):
        x = fixed_covars_d[key] 
        if len( x.shape ) == 1:
            x = x.reshape(-1,1)
        beta_d[key] = beta_l[:x.shape[1]]
        beta_l = beta_l[x.shape[1]:]

    return(beta_d)

def assign_fixedeffect_vars(fixedeffect_vars_l, fixed_covars_d):
    fixedeffect_vars_d = {'celltype_main_var': fixedeffect_vars_l[0]}
    if len(fixed_covars_d.keys()) > 0:
        for key, value in zip(np.sort(list(fixed_covars_d.keys())), fixedeffect_vars_l[1:]):
            fixedeffect_vars_d[key] = value
    return(fixedeffect_vars_d)

def RandomeffectVariance_( V, X ):
    return( np.trace( V @ (X.T @ X) ) / X.shape[0] )

def RandomeffectVariance( Vs, Xs ):
    if len( np.array( Vs ).shape ) == 1:
        Vs = [V * np.eye(X.shape[1]) for V, X in zip(Vs, Xs)]

    vars = [RandomeffectVariance_(V,X) for V,X in zip(Vs, Xs)]
    return( vars, Vs )

def assign_randomeffect_vars(randomeffect_vars_l, r2_l, random_covars_d, order=True):
    randomeffect_vars_d = {}
    r2_d = {}
    keys = np.sort( list(random_covars_d.keys()) ) if order else list(random_covars_d.keys())
    if len(keys) != 0:
        for key, v1, v2 in zip( keys, randomeffect_vars_l, r2_l ):
            randomeffect_vars_d[key] = v1
            r2_d[key] = v2

    return( randomeffect_vars_d, r2_d )

def ct_randomeffect_variance( V, P ):
    N, C = P.shape
    ct_overall_var = RandomeffectVariance_(V, P)
    ct_specific_var = np.array([V[i,i] * ((P[:,i]**2).mean()) for i in range(C)])

    return( ct_overall_var, ct_specific_var )

def cal_variance(beta, P, fixed_covars, r2, random_covars, order=False):
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars = fixedeffect_vars( beta, P, fixed_covars )
    random_vars = RandomeffectVariance( r2, list(random_covars.values()) )[0]
    random_vars, r2 = assign_randomeffect_vars(random_vars, r2, random_covars, order=order)
    return( beta, fixed_vars, r2, random_vars )

def quantnorm(Y, axis=0):
    '''
    # use sklearn.preprocessing.quantile_transform
    '''
    pass

def wald_ct_beta(beta, beta_var, n, P):
    '''
    n: scalar (for Ftest in Wald test)
        sample size
    P: scalar (for Ftest in Wald test)
        number of estimated parameters
    '''
    C = len(beta)
    T = np.concatenate( ( np.eye(C-1), (-1)*np.ones((C-1,1)) ), axis=1 )
    beta = T @ beta
    beta_var = T @ beta_var @ T.T
    return(wald.mvwald_test(beta, np.zeros(C-1), beta_var, n=n, P=P))

def check_R(R):
    '''
    Check R matrix: has to be matrix of 0 and 1
    in the structure of scipy.linalg.block_diag(np.ones((a,1)), np.ones((b,1)), np.ones((c,1))
    '''
    # infer matrix R
    xs = np.sum(R, axis=0).astype('int')
    R_ = np.ones((xs[0],1))
    for i in range(1,len(xs)):
        R_ = scipy.linalg.block_diag(R_, np.ones((xs[i],1)))

    if np.any(R != R_):
        print(R[:5,:])
        print(R_[:5,:])
        return( False )
    else:
        return( True )

def order_by_randomcovariate(R, Xs=[], Ys={}):
    '''
    R is the design matrix of 0 and 1 for a random covriate, which we order along by
    Xs or Ys: a list or dict of matrixs we want to order
    '''
    R_df = pd.DataFrame(R)
    index = R_df.sort_values(by=list(R_df.columns), ascending=False).index
    R = np.take_along_axis(R, np.broadcast_to(index, (R.shape[1], R.shape[0])).T, axis=0)
    if not check_R(R):
        sys.exit('Matrix R is wrong!\n')

    new_Xs = []
    for X in Xs:
        if len(X.shape) > 1:
            X = np.take_along_axis(X, np.broadcast_to(index, (X.shape[1], X.shape[0])).T, axis=0)
        else:
            X = np.take_along_axis(X, index, axis=0)
        new_Xs.append(X)

    new_Ys = {}
    for key in Ys.keys():
        Y = Ys[key]
        if len(Y.shape) > 1:
            Y = np.take_along_axis(Y, np.broadcast_to(index, (Y.shape[1], Y.shape[0])).T, axis=0)
        else:
            Y = np.take_along_axis(Y, index, axis=0)
        new_Ys[key] = Y

    return(index, R, new_Xs, new_Ys)

def jk_rmInd(i, Y, vs, fixed_covars={}, random_covars={}, P=None):
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
        return(Y_, vs_, fixed_covars_, random_covars_, P_)

