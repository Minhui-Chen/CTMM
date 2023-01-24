import os, sys, re, multiprocessing, time
import helper, mystats 
import scipy
from scipy import stats, linalg, optimize
import numpy as np, pandas as pd
from numpy.random import default_rng
import wald, util, cuomo_ctng_test

def get_X(P, fixed_covars_d):
    X = P
    for key in np.sort(list(fixed_covars_d.keys())):
        m_ = fixed_covars_d[key]
        if isinstance(m_, str):
            m_ = np.loadtxt(fixed_covars_d[key])
        if len( m_.shape ) == 1:
            m_ = m_.reshape(-1,1)
        X = np.concatenate( (X, m_), axis=1)
    return(X)

def he_ols(y, P, vs, fixed_covars, random_covars, model):
    '''
    Q: design matrix in HE OLS
    '''
    N, C = P.shape
    X = get_X(P, fixed_covars)
    proj = np.eye( N ) - X @ np.linalg.inv(X.T @ X) @ X.T
    
    # project out fixed effects
    y_p = proj @ y
    t = ( np.outer(y_p, y_p) - proj @ np.diag(vs) @ proj ).flatten('F')

    if model == 'hom':
        Q = [ proj.flatten('F') ]
    elif model == 'iid':
        Q = [ proj.flatten('F') ]
        Q.append( (proj @ np.diag(np.diag(P @ P.T)) @ proj).flatten('F') )
    elif model == 'free':
        Q = [ proj.flatten('F') ]
        for i in range(C):
            Q.append( (proj @ np.diag( P[:,i]**2 ) @ proj).flatten('F') )
    elif model == 'full':
        Q = []
        for i in range(C):
            Q.append( (proj @ np.diag( P[:,i]**2 ) @ proj).flatten('F') )
        for i in range(1,C):
            for j in range(i):
                Q.append( 2*(proj @ np.diag( P[:,i] * P[:,j] ) @ proj).flatten('F') )
    
    for R in random_covars.values():
        m = proj @ R
        m = (m @ m.T).flatten('F')
        Q.append( m )

    Q = np.array( Q ).T

    theta = np.linalg.inv( Q.T @ Q ) @ Q.T @ t
    return( theta )

def cal_Vy( P, vs, hom2, V, r2=[], random_MMT=[] ):
    Vy = hom2 + np.diag( P @ V @ P.T ) + vs

    Vy = np.diag( Vy )
    for var, MMT in zip(r2, random_MMT):
        Vy += var * MMT

    return( Vy )

def ML_LL(y, P, X, vs, beta, hom2, V, r2=[], random_MMT=[]):
    Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )

    if np.array_equal( Vy, np.diag( np.diag(Vy) ) ):
        # no covariance between individuals
        Vy = np.diag(Vy)
        if np.any( Vy < 0 ) or ( (np.amax(Vy) / (np.amin(Vy)+1e-99))  > 1e6 ):
            return( 1e12 )

        l = np.sum( np.log(Vy) ) + np.sum( (y - X @ beta)**2 / Vy )
        l = l * 0.5
    else:
        w, v = linalg.eigh(Vy)
        if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ): 
            return(1e12)
        l = stats.multivariate_normal.logpdf(y, mean=X @ beta, cov=Vy) * (-1)
    return( l )

def REML_LL(y, P, X, C, vs, hom2, V, r2=[], random_MMT=[]):
    Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )

    if np.array_equal( Vy, np.diag( np.diag(Vy) ) ):
        # no covariance between individuals
        Vy = np.diag(Vy)
        if np.any( Vy < 0 ) or ( (np.amax(Vy) / (np.amin(Vy)+1e-99))  > 1e6 ): 
            return( 1e12 )
        
        Vy_inv = 1 / Vy
        l = np.sum( np.log(Vy) )

        A = X.T * Vy_inv
        B = A @ X
        w, v = linalg.eigh(B)
        if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ): 
            return(1e12)

        B_inv = v @ np.diag(1/w) @ v.T
        B_det = np.sum( np.log(w) )
        M = np.diag(Vy_inv) - A.T @ B_inv @ A
    else:
        for var, MMT in zip(r2, random_MMT):
            Vy += var * MMT

        w, v = linalg.eigh( Vy )
        if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
            return(1e12)

        Vy_inv = v @ np.diag(1/w) @ v.T
        Vy_det = np.sum( np.log(w) )
        l = Vy_det

        A = X.T @ Vy_inv
        B = A @ X
        w, v = linalg.eigh(B)
        if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
            return(1e12)

        B_inv = v @ np.diag(1/w) @ v.T
        B_det = np.sum( np.log(w) )
        M = Vy_inv - A @ B_inv @ A

    l = 0.5 * (l + B_det + y @ M @y)
    return( l )

def hom_ML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    beta = par[1:(1+X.shape[1])]
    V = np.zeros((C,C))
    r2 = par[(1+X.shape[1]):]

    return( ML_LL(y, P, X, vs, beta, hom2, V, r2, random_MMT) )

def hom_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None):
    print('Hom ML')

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars)
    n_par = 1 + n_random + X.shape[1]

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ y)
        hom2 = np.var(y - X @ beta) / ( n_random + 1 )
        par = [hom2] + list(beta) + [hom2] * n_random

    random_MMT = [R @ R.T for R in random_covars.values()]

    out, opt = util.optim(hom_ML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

    hom2, beta, r2 = out['x'][0], out['x'][1:(1+X.shape[1])], out['x'][(1+X.shape[1]):]
    l = out['fun'] * (-1)
    Vy = cal_Vy( P, vs, hom2, np.zeros((C,C)), r2, random_MMT )
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

    # wald
    Z = [np.eye(N)] + list(random_covars.values())
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
    ml = {'hom2':hom2, 'beta':beta, 'l':l, 'D':D, 'fixedeffect_vars':fixed_vars,
            'randomeffect_vars':random_vars, 'r2':r2, 'nu':np.mean(vs), 'opt':opt}

    p = {}
    p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    #p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
    p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], D[:C,:C], n=N, P=n_par)
    return(ml, p)

def hom_REML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.zeros((C,C))
    r2 = par[1:]
    return( REML_LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )

def hom_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None):
    print('Hom REML')

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars)
    n_par = 1 + n_random

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ y)
        hom2 = np.var(y - X @ beta) / ( n_random + 1 )
        par = [hom2] * (n_random + 1)

    random_MMT = [R @ R.T for R in random_covars.values()]

    out, opt = util.optim(hom_REML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

    hom2, r2 = out['x'][0], out['x'][1:]
    l = out['fun'] * (-1)
    Vy = cal_Vy( P, vs, hom2, np.zeros((C,C)), r2, random_MMT )
    beta = util.glse( Vy, X, y )
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

    # wald
    Z = [np.eye(N)] + list(random_covars.values())
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'hom2':hom2, 'beta':beta, 'l':l, 'D':D, 'fixedeffect_vars':fixed_vars,
            'randomeffect_vars':random_vars, 'r2':r2, 'nu':np.mean(vs),  'opt':opt}
    p = {}
    p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
    # wald test beta1 = beta2 = beta3
    p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], 
            np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
            n=N, P=n_par+X.shape[1])
    return(reml, p)

def hom_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('Hom HE')
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    X = get_X(P, fixed_covars)
    n_par = 1 + n_random

    def he_f(y, P, vs, fixed_covars, random_covars):
        N, C = P.shape
        X = get_X(P, fixed_covars)

        theta = he_ols(y, P, vs, fixed_covars, random_covars, model='hom')
        hom2, r2 = theta[0], theta[1:]

        # 
        random_MMT = [R @ R.T for R in random_covars.values()]
        Vy = cal_Vy( P, vs, hom2, np.zeros((C,C)), r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

        return( hom2, r2, beta, fixed_vars, random_vars )

    #
    hom2, r2, beta, fixed_vars, random_vars = he_f(y, P, vs, fixed_covars, random_covars)
    he = {'hom2':hom2, 'r2':r2, 'beta':beta,
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars, 'nu':np.mean(vs)}

    # jackknife
    p = {}
    if jack_knife:
        jacks = {'ct_beta':[], 'hom2':[]}
        for i in range(N):
            y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, y, vs, fixed_covars, random_covars, P)
            hom2_jk, _, beta_jk, _, _ = he_f(y_jk, P_jk, vs_jk, fixed_covars_jk, random_covars_jk)

            jacks['hom2'].append( hom2_jk )
            jacks['ct_beta'].append( beta_jk['ct_beta'] )

        var_hom2 = (N - 1.0) * np.var(jacks['hom2'])
        var_beta = (N - 1.0) * np.cov(np.array(jacks['ct_beta']).T, bias=True)

        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        p['ct_beta'] = util.wald_ct_beta( beta['ct_beta'], var_beta, n=N, P=n_par+X.shape[1] )

    print( time.time() - start )
    return( he, p )

def iid_ML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.eye(C) * par[1]
    beta = par[2:(2+X.shape[1])]
    r2 = par[(2+X.shape[1]):]

    return( ML_LL(y, P, X, vs, beta, hom2, V, r2, random_MMT) )

def iid_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None):
    print('IID ML')

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars)
    n_par = 2 + n_random + X.shape[1]

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ y)
        hom2 = np.var(y - X @ beta) / ( n_random + 2 )
        par = [hom2, hom2] + list(beta) + [hom2] * n_random

    random_MMT = [R @ R.T for R in random_covars.values()]

    out, opt = util.optim(iid_ML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

    hom2, beta, r2 = out['x'][0], out['x'][2:(2+X.shape[1])], out['x'][(2+X.shape[1]):]
    V = np.eye(C) * out['x'][1]
    l = out['fun'] * (-1)
    Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
    ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

    # wald
    Z = [np.identity(len(y)), scipy.linalg.khatri_rao(np.eye(N), P.T).T]
    Z = Z + list( random_covars.values() )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
    ml = {'hom2':hom2, 'beta':beta, 'V':V, 'l':l, 'D':D,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars, 'r2':r2,
            'nu':np.mean(vs), 'opt':opt}

    p = {}
    p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    p['V'] = wald.wald_test(V[0,0], 0, D[X.shape[1]+1,X.shape[1]+1], N-n_par)
    #p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
    p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], D[:C,:C], n=N, P=n_par)
    return(ml, p)

def iid_REML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.eye(C) * par[1]
    r2 = par[2:]
    return( REML_LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )

def iid_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None):
    print('IID REML')

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars)
    n_par = 1 + 1 + n_random

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ y)
        hom2 = np.var(y - X @ beta) / ( n_random + 2 )
        par = [hom2] * (n_random + 2)

    random_MMT = [R @ R.T for R in random_covars.values()]

    out, opt = util.optim(iid_REML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

    hom2, r2 = out['x'][0], out['x'][2:]
    V = np.eye(C) * out['x'][1]
    l = out['fun'] * (-1)
    Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
    beta = util.glse( Vy, X, y )
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
    ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

    # wald
    Z = [np.eye(N), scipy.linalg.khatri_rao(np.eye(N), P.T).T]
    Z = Z + list( random_covars.values() )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'hom2':hom2, 'V':V, 'beta':beta, 'l':l, 'D':D, 
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars, 'r2':r2,
            'nu':np.mean(vs), 'opt':opt}

    p = {}
    p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
    p['V'] = wald.wald_test(V[0,0], 0, D[1,1], N-n_par)
    # wald test beta1 = beta2 = beta3
    p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], 
            np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C], 
            n=N, P=n_par+X.shape[1])
    return(reml, p)

def iid_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('IID HE')
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    X = get_X(P, fixed_covars)
    n_par = 1 + 1 + n_random

    def he_f(y, P, vs, fixed_covars, random_covars):
        N, C = P.shape
        X = get_X(P, fixed_covars)

        theta = he_ols(y, P, vs, fixed_covars, random_covars, model='iid')
        hom2, het, r2 = theta[0], theta[1], theta[2:]
        V = np.eye(C) * het

        # 
        random_MMT = [R @ R.T for R in random_covars.values()]
        Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return( hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var )

    hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var = he_f(
            y, P, vs, fixed_covars, random_covars)
    he = {'hom2':hom2, 'V':V, 'r2':r2, 'beta':beta, 'nu':np.mean(vs),
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var}

    # jackknife
    p = {}
    if jack_knife:
        jacks = { 'ct_beta':[], 'hom2':[], 'het':[] }
        for i in range(N):
            y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, y, vs, fixed_covars, random_covars, P)
            hom2_jk, V_jk, _, beta_jk, _, _, _, _ = he_f(y_jk, P_jk, vs_jk, fixed_covars_jk, random_covars_jk)

            jacks['hom2'].append( hom2_jk )
            jacks['ct_beta'].append( beta_jk['ct_beta'] )
            jacks['het'].append( V_jk[0,0] )

        var_hom2 = (N - 1.0) * np.var(jacks['hom2'])
        var_het = (N - 1.0) * np.var(jacks['het'])
        var_beta = (N - 1.0) * np.cov(np.array(jacks['ct_beta']).T, bias=True)

        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par) 
        p['V'] = wald.wald_test(V[0,0], 0, var_het, N-n_par)
        p['ct_beta'] = util.wald_ct_beta( beta['ct_beta'], var_beta, n=N, P=n_par+X.shape[1] )

    print( time.time() - start )
    return( he, p )

def free_ML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.diag( par[1:(C+1)] )
    beta = par[(C+1):(C+1+X.shape[1])]
    r2 = par[(C+1+X.shape[1]):]

    return( ML_LL(y, P, X, vs, beta, hom2, V, r2, random_MMT) )

def free_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10):
    print('Free ML')

    def extract( out, C, X, P, fixed_covars, random_covars ):
        hom2, beta, r2 = out['x'][0], out['x'][(C+1):(C+1+X.shape[1])], out['x'][(C+1+X.shape[1]):]
        V = np.diag( out['x'][1:(C+1)] )
        l = out['fun'] * (-1)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        return( hom2, beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars )

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars)
    n_par = 1 + C + X.shape[1] + n_random 

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ y)
        hom2 = np.var(y - X @ beta) / ( n_random + 2 )
        par = [hom2] * (C+1) + list(beta) + [hom2] * n_random

    random_MMT = [R @ R.T for R in random_covars.values()]

    out, opt = util.optim(free_ML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

    hom2, beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract( 
            out, C, X, P, fixed_covars, random_covars )

    if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
        out, opt = util.re_optim(out, opt, free_ML_loglike, par, args=(y, P, X, C, vs, random_MMT), 
                method=method, nrep=nrep)
        hom2, beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract( 
                out, C, X, P, fixed_covars, random_covars )

    # wald
    Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
    eval, evec = np.linalg.eig(Vy)
    print(max(eval), min(eval), max(np.diag(Vy)), min(np.diag(Vy)))

    Z = [np.eye(N)] + [np.diag(P[:,i]) for i in range(C)]
    Z = Z + list( random_covars.values() )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
    ml = {'hom2':hom2, 'beta':beta, 'V':V, 'l':l, 'D':D,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars, 'r2':r2,
            'nu':np.mean(vs), 'opt':opt}

    p = {}
    p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    var_V = D[(X.shape[1]+1):(X.shape[1]+C+1), (X.shape[1]+1):(X.shape[1]+C+1)]
    p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
    #p['Vi'] = [wald.wald_test(V[i,i], 0, D[X.shape[1]+i+1,X.shape[1]+i+1], N-n_par) for i in range(C)]
    #p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
    p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], D[:C,:C], n=N, P=n_par )

    return(ml, p)

def free_REML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.diag( par[1:(C+1)] )
    r2 = par[(C+1):]
    return( REML_LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )

def free_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10, 
        jack_knife=False):
    print('Free REML')

    def reml_f(y, P, vs, fixed_covars, random_covars, par, method):
        N, C = P.shape
        X = get_X(P, fixed_covars)

        random_MMT = [R @ R.T for R in random_covars.values()]

        out, opt = util.optim(free_REML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

        def extract(out, C, X, P, fixed_covars, random_covars):
            hom2, r2 = out['x'][0], out['x'][(C+1):]
            V = np.diag( out['x'][1:(C+1)] )
            l = out['fun'] * (-1)
            Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
            beta = util.glse( Vy, X, y )
            # calcualte variance of fixed and random effects, and convert to dict
            beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
            ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
            return(hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, 
                    ct_overall_var, ct_specific_var)

        hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var = extract(
                out, C, X, P, fixed_covars, random_covars)

        if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, free_REML_loglike, par, 
                    args=(y, P, X, C, vs, random_MMT), method=method, nrep=nrep)
            hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var = extract(
                    out, C, X, P, fixed_covars, random_covars)

        return(hom2, V, r2, beta, l, fixed_vars, random_vars, Vy,
                ct_overall_var, ct_specific_var, opt)

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars)
    n_par = 1 + C + n_random

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ y)
        hom2 = np.var(y - X @ beta) / ( n_random + 2 )
        par = [hom2] * (n_random + C + 1)

    #
    hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var, opt = reml_f(
            y, P, vs, fixed_covars, random_covars, par, method)

    # wald
    Z = [np.eye(N)] + [np.diag(P[:,i]) for i in range(C)]
    Z = Z + list( random_covars.values() )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'hom2':hom2, 'V':V, 'beta':beta, 'l':l, 'D':D, 
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars, 'r2':r2,
            'nu':np.mean(vs), 'opt':opt}

    p = {}
    if not jack_knife:
        p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), D[1:(C+1), 1:(C+1)], n=N, P=n_par)
        #p['Vi'] = [wald.wald_test(V[i,i], 0, D[i+1,i+1], N-n_par) for i in range(C)]
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], 
                np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par+X.shape[1])
    else:
        jacks = {'ct_beta':[], 'hom2':[], 'V':[]}
        for i in range(N):
            y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, y, vs, fixed_covars, random_covars, P)
            hom2_jk, V_jk, _, beta_jk, _, _, _, _, _, _, _ = reml_f(
                    y_jk, P_jk, vs_jk, fixed_covars_jk, random_covars, par, method)

            jacks['ct_beta'].append( beta_jk['ct_beta'] )
            jacks['hom2'].append( hom2_jk )
            jacks['V'].append( np.diag(V_jk) )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_V = (len(jacks['V']) - 1.0) * np.cov( np.array(jacks['V']).T, bias=True )
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )

        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par+X.shape[1])

    return(reml, p)

def free_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('Free HE')
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    X = get_X(P, fixed_covars)
    n_par = 1 + C + n_random

    def he_f(y, P, vs, fixed_covars, random_covars):
        N, C = P.shape
        X = get_X(P, fixed_covars)

        theta = he_ols(y, P, vs, fixed_covars, random_covars, model='free')
        hom2, r2 = theta[0], theta[(1+C):]
        V = np.diag( theta[1:(C+1)] )

        # 
        random_MMT = [R @ R.T for R in random_covars.values()]
        Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return( hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var )

    #
    hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var = he_f(
            y, P, vs, fixed_covars, random_covars)
    he = {'hom2':hom2, 'V':V, 'r2':r2, 'beta':beta, 'nu':np.mean(vs),
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var}

    # jackknife
    p = {}
    if jack_knife:
        jacks = { 'ct_beta':[], 'hom2':[], 'V':[] }
        for i in range(N):
            y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, y, vs, fixed_covars, random_covars, P)
            hom2_jk, V_jk, _, beta_jk, _, _, _, _ = he_f(y_jk, P_jk, vs_jk, fixed_covars_jk, random_covars_jk)

            jacks['hom2'].append( hom2_jk )
            jacks['ct_beta'].append( beta_jk['ct_beta'] )
            jacks['V'].append( np.diag(V_jk) )

        var_hom2 = (N - 1.0) * np.var(jacks['hom2'])
        var_V = (N - 1.0) * np.cov( np.array(jacks['V']).T, bias=True )
        var_beta = (N - 1.0) * np.cov(np.array(jacks['ct_beta']).T, bias=True)

        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par) 
        p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
        p['ct_beta'] = util.wald_ct_beta( beta['ct_beta'], var_beta, n=N, P=n_par+X.shape[1] )

    print( time.time() - start )
    return( he, p )

def full_ML_loglike(par, y, P, X, C, vs, random_MMT):
    ngam = C * (C+1) // 2
    V = np.zeros((C,C))
    V[np.tril_indices(C)] = par[:ngam]
    V = V + V.T
    hom2 = 0
    beta = par[ngam:(ngam+X.shape[1])]
    r2 = par[(ngam+X.shape[1]):]

    return( ML_LL(y, P, X, vs, beta, hom2, V, r2, random_MMT) )

def full_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10):
    print('Full ML')

    def extract( out, C, X, P, fixed_covars, random_covars ):
        ngam = C * (C+1) // 2
        V = np.zeros((C,C))
        V[np.tril_indices(C)] = out['x'][:ngam]
        V = V + V.T
        beta, r2 = out['x'][ngam:(ngam+X.shape[1])], out['x'][(ngam+X.shape[1]):]
        l = out['fun'] * (-1)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        return( beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars )

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ y)
        V = np.eye(C)[np.tril_indices(C)] * np.var(y - X @ beta) / ( n_random + 1 )
        par = list(V) + list(beta) + [V[0]] * n_random

    random_MMT = [R @ R.T for R in random_covars.values()]

    out, opt = util.optim(full_ML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

    beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract( 
            out, C, X, P, fixed_covars, random_covars )

    if util.check_optim(opt, 0, ct_overall_var, fixed_vars, random_vars):
        out, opt = util.re_optim(out, opt, full_ML_loglike, par, args=(y, P, X, C, vs, random_MMT), 
                method=method, nrep=nrep)
        beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract( 
                out, C, X, P, fixed_covars, random_covars )

    ml = {'beta':beta, 'V':V, 'l':l, 
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars, 'r2':r2,
            'r2':r2, 'nu':np.mean(vs), 'opt':opt}
    return(ml)

def full_REML_loglike(par, y, P, X, C, vs, random_MMT):
    ngam = C * (C+1) // 2
    V = np.zeros((C,C))
    V[np.tril_indices(C)] = par[:ngam]
    V = V + V.T
    hom2 = 0
    r2 = par[ngam:]
    return( REML_LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )

def full_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10):
    print('Full REML')

    def extract(out, C, X, P, fixed_covars, random_covars):
        ngam = C * (C+1) // 2
        r2 = out['x'][ngam:]
        V = np.zeros((C,C))
        V[np.tril_indices(C)] = out['x'][:ngam]
        V = V + V.T
        l = out['fun'] * (-1)
        Vy = cal_Vy( P, vs, 0, V, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        return( V, r2, beta, l, fixed_vars, random_vars, Vy,
                ct_overall_var, ct_specific_var )

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    ngam = C * (C+1) // 2
    X = get_X(P, fixed_covars)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ y)
        hom2 = np.var(y - X @ beta) / ( n_random + 1 )
        V = np.eye(C)[np.tril_indices(C)]
        par = list(V * hom2) + [hom2] * n_random

    random_MMT = [R @ R.T for R in random_covars.values()]

    out, opt = util.optim(full_REML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

    V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var = extract(
            out, C, X, P, fixed_covars, random_covars)

    if util.check_optim(opt, 0, ct_overall_var, fixed_vars, random_vars):
        out, opt = util.re_optim(out, opt, full_REML_loglike, par, args=(y, P, X, C, vs, random_MMT), 
                method=method, nrep=nrep)
        V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var = extract(
                out, C, X, P, fixed_covars, random_covars)


    reml = {'V':V, 'beta':beta, 'l':l, 
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars, 'r2':r2,
            'r2':r2, 'nu':np.mean(vs), 'opt':opt}
    return(reml)

def full_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}):
    print('Full HE')
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    X = get_X(P, fixed_covars)

    def he_f(y, P, vs, fixed_covars, random_covars):
        N, C = P.shape
        ngam = C * (C+1) // 2
        X = get_X(P, fixed_covars)

        theta = he_ols(y, P, vs, fixed_covars, random_covars, model='full')
        r2 = theta[ngam:]
        V = np.diag( theta[:C] )
        V[np.tril_indices(C,k=-1)] = theta[C:ngam]
        V = V + V.T - np.diag( np.diag( V ) )

        # 
        random_MMT = [R @ R.T for R in random_covars.values()]
        Vy = cal_Vy( P, vs, 0, V, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return( V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var )

    V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var = he_f(
            y, P, vs, fixed_covars, random_covars)
    he = {'V':V, 'r2':r2, 'beta':beta, 'nu':np.mean(vs),
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var}

    print( time.time() - start )
    return( he )

def main():
    # par
    params = snakemake.params
    input = snakemake.input
    output = snakemake.output

    #
    batch = params.batch
    outs = [re.sub('/rep/', f'/rep{i}/', params.out) for i in batch]
    for y_f, P_f, nu_f, out_f in zip([line.strip() for line in open(input.y)],
            [line.strip() for line in open(input.P)], [line.strip() for line in open(input.nu)], outs):
        print(y_f, P_f, nu_f)
        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape

        # project out cell type main effect from y
        #proj = np.eye(len(y)) - P @ np.linalg.inv(P.T @ P) @ P.T
        #y_p = proj @ y
        #Y = y_p**2 - np.diag(proj @ np.diag(vs) @ proj)

        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        out = {}
        ## HE
        if 'HE_as_initial' not in snakemake.params.keys():
            snakemake.params.HE_as_initial = False
        if snakemake.params.HE_as_initial:
            snakemake.params.HE = True

        if snakemake.params.HE:
            hom_he, hom_he_p = hom_HE(y_f, P_f, nu_f, jack_knife=True)
            free_he, free_he_p = free_HE(y_f, P_f, nu_f, jack_knife=True)
            full_he = full_HE(y_f, P_f, nu_f)

            out['he'] = {'hom': hom_he, 'free': free_he, 'full': full_he,
                    'wald':{'hom':hom_he_p, 'free':free_he_p}}

        ## ML
        if snakemake.params.ML:
            if not snakemake.params.HE_as_initial:
                hom_ml, hom_ml_p = hom_ML(y_f, P_f, nu_f)
                free_ml, free_ml_p = free_ML(y_f, P_f, nu_f)
                full_ml = full_ML(y_f, P_f, nu_f)
            else:
                hom_ml, hom_ml_p = hom_ML( y_f, P_f, nu_f, par=util.generate_HE_initial(hom_he, ML=True) )
                free_ml, free_ml_p = free_ML( y_f, P_f, nu_f, par=util.generate_HE_initial(free_he, ML=True) )
                full_ml = full_ML( y_f, P_f, nu_f, par=util.generate_HE_initial(full_he, ML=True) )

            out['ml'] = {'hom': hom_ml, 'free': free_ml, 'full': full_ml,
                    'wald':{'hom':hom_ml_p, 'free':free_ml_p}}

            # LRT
            free_hom_lrt = mystats.lrt(out['ml']['free']['l'], out['ml']['hom']['l'], C)
            full_hom_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['hom']['l'], C*(C+1)//2-1)
            full_free_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['free']['l'], C*(C+1)//2-C-1)

            out['ml']['lrt'] = {'free_hom':free_hom_lrt,
                    'full_hom':full_hom_lrt, 'full_free':full_free_lrt}

        ## REML
        if snakemake.params.REML:
            if 'Free_reml_only' not in snakemake.params.keys():
                snakemake.params.Free_reml_only = False

            if not snakemake.params.HE_as_initial:
                if 'Free_reml_jk' in snakemake.params.keys():
                    free_reml, free_reml_p = free_REML(y_f, P_f, nu_f, jack_knife=snakemake.params.Free_reml_jk)
                else:
                    free_reml, free_reml_p = free_REML(y_f, P_f, nu_f)

                if snakemake.params.Free_reml_only:
                    hom_reml, hom_reml_p = free_reml, free_reml_p
                    full_reml = free_reml
                else:
                    hom_reml, hom_reml_p = hom_REML(y_f, P_f, nu_f)
                    full_reml = full_REML(y_f, P_f, nu_f)
            else:
                hom_reml, hom_reml_p = hom_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(hom_he, REML=True) )
                free_reml, free_reml_p = free_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(free_he, REML=True))
                full_reml = full_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(full_he, REML=True))

            out['reml'] = {'hom':hom_reml, 'free':free_reml, 'full':full_reml,
                    'wald':{'hom':hom_reml_p, 'free':free_reml_p}}

            ## LRT
            free_hom_lrt = mystats.lrt(out['reml']['free']['l'], out['reml']['hom']['l'], C)
            full_hom_lrt = mystats.lrt(out['reml']['full']['l'], out['reml']['hom']['l'], C*(C+1)//2-1)
            full_free_lrt = mystats.lrt(out['reml']['full']['l'], out['reml']['free']['l'], C*(C+1)//2-C-1)

            out['reml']['lrt'] = {'free_hom':free_hom_lrt,
                    'full_hom':full_hom_lrt, 'full_free':full_free_lrt}

        # save
        np.save(out_f, out)

    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))

if __name__ == '__main__':
    main()
