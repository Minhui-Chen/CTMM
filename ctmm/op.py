import os, sys, re, time
import pkg_resources
from scipy import stats, linalg, optimize
import numpy as np, pandas as pd
from numpy.random import default_rng
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from rpy2.robjects.conversion import localconverter

from . import wald, util

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

def get_MMT(random_covars):
    random_MMT = []
    for key in np.sort( list(random_covars.keys()) ):
        R = random_covars[key]
        random_MMT.append( R @ R.T )
    return( random_MMT )

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
    
    for key in np.sort( list(random_covars.keys()) ):
        R = random_covars[key]
        m = proj @ R
        m = (m @ m.T).flatten('F')
        Q.append( m )

    Q = np.array( Q ).T

    theta = np.linalg.inv( Q.T @ Q ) @ Q.T @ t
    return( theta )

def cal_Vy( P, vs, hom2, V, r2=[], random_MMT=[] ):
    Vy = hom2 + np.diag( P @ V @ P.T ) + vs

    Vy = np.diag( Vy )
    if isinstance(r2, dict):
        r2 = [r2[key] for key in np.sort(list(r2.keys()))]
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
        M = Vy_inv - A.T @ B_inv @ A

    l = 0.5 * (l + B_det + y @ M @y)
    return( l )

def r_optim(y, P, vs, fixed_covars, random_covars, par, nrep, ml, model, method):
    if ml.upper() == 'ML':
        rf = pkg_resources.resource_filename(__name__, 'op.ml.R')
    else:
        rf = pkg_resources.resource_filename(__name__, 'op.reml.R')
    r_optim = STAP( open(rf).read(), 'r_optim' )
    par = robjects.NULL if par is None else robjects.FloatVector(par)
    method = 'BFGS' if method is None else method
    numpy2ri.activate()
    if model == 'hom':
        out_ = r_optim.screml_hom(y=robjects.FloatVector(y), P=r['as.matrix'](P),
                vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars),
                par=par, nrep=nrep, method=method)
    elif model == 'iid':
        out_ = r_optim.screml_iid(y=robjects.FloatVector(y), P=r['as.matrix'](P),
                vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars),
                par=par, nrep=nrep, method=method)
    elif model == 'free':
        out_ = r_optim.screml_free(y=robjects.FloatVector(y), P=r['as.matrix'](P),
                vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars),
                par=par, nrep=nrep, method=method)
    elif model == 'full':
        out_ = r_optim.screml_full(y=robjects.FloatVector(y), P=r['as.matrix'](P),
                vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars),
                par=par, nrep=nrep, method=method)
    numpy2ri.deactivate()
    out = {}
    for key, value in zip(out_.names, list(out_)):
        out[key] = value
    return( out )

def hom_ML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    beta = par[1:(1+X.shape[1])]
    V = np.zeros((C,C))
    r2 = par[(1+X.shape[1]):]

    return( ML_LL(y, P, X, vs, beta, hom2, V, r2, random_MMT) )

def hom_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10, optim_by_R=False):
    print('Hom ML')

    if not optim_by_R:
        def extract( out, C, X, P, fixed_covars, random_covars ):
            hom2, beta, r2 = out['x'][0], out['x'][1:(1+X.shape[1])], out['x'][(1+X.shape[1]):]
            l = out['fun'] * (-1)
            # calcualte variance of fixed and random effects, and convert to dict
            beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
            return( hom2, beta, r2, l, fixed_vars, random_vars )

        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

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

        out, opt = util.optim(hom_ML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

        hom2, beta, r2, l, fixed_vars, random_vars = extract( out, C, X, P, fixed_covars, random_covars )

        if util.check_optim(opt, hom2, 0, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, hom_ML_loglike, par, args=(y, P, X, C, vs, random_MMT),
                    method=method, nrep=nrep)
            hom2, beta, r2, l, fixed_vars, random_vars = extract( out, C, X, P, fixed_covars, random_covars )

        # wald
        Vy = cal_Vy( P, vs, hom2, np.zeros((C,C)), r2, random_MMT )
        Z = [np.eye(N)] + Rs
        D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
        ml = {'hom2':hom2, 'beta':beta, 'l':l, 'D':D, 'fixedeffect_vars':fixed_vars,
                'randomeffect_vars':random_vars, 'r2':r2, 'nu':np.mean(vs), 'opt':opt}

        p = {}
        p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
        #p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], D[:C,:C], n=N, P=n_par)
        return(ml, p)

    else:
        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars)
        n_par = 1 + n_random + X.shape[1]

        out = r_optim(y, P, vs, fixed_covars, random_covars, par, nrep, 'ML', 'hom', method)

        hom2, beta, l, hess = out['hom2'][0], np.array(out['beta']), out['l'][0], np.array(out['hess'])
        convergence = out['convergence'][0]
        fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars)
        beta_d = util.assign_beta(beta, P, fixed_covars)
        randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
                np.array(out['r2']), random_covars )

        # wald
        Vy = np.diag(hom2 + vs)
        Z = [np.identity(len(y))]
        if len(r2_d.keys()) > 0:
            for key in np.sort( list(r2_d.keys()) ):
                M_ = random_covars[key]
                Vy = Vy + r2_d[key] * M_ @ M_.T
                Z.append( M_ )
        D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
        res = {'hom2':hom2, 'beta':beta_d, 'l':l, 'D':D, 'fixedeffect_vars':fixedeffect_vars_d,
                'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d, 'nu':np.mean(vs), 'hess':hess,
                'convergence':convergence}

        wald_p = {}
        wald_p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
        wald_p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
        wald_p['ct_beta'] = util.wald_ct_beta(beta[:C], D[:C,:C], n=N, P=n_par)
        return(res, wald_p)

def hom_REML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.zeros((C,C))
    r2 = par[1:]
    return( REML_LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )

def hom_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10, optim_by_R=False):
    print('Hom REML')
    
    if not optim_by_R:
        def extract(out, C, y, X, P, vs, fixed_covars, random_covars, random_MMT):
            hom2, r2 = out['x'][0], out['x'][1:]
            l = out['fun'] * (-1)
            Vy = cal_Vy( P, vs, hom2, np.zeros((C,C)), r2, random_MMT )
            beta = util.glse( Vy, X, y )
            # calcualte variance of fixed and random effects, and convert to dict
            beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
            return(hom2, r2, beta, l, fixed_vars, random_vars, Vy)

        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

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
            par = [hom2] * (n_random + 1)

        out, opt = util.optim(hom_REML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

        hom2, r2, beta, l, fixed_vars, random_vars, Vy = extract(
                out, C, y, X, P, vs, fixed_covars, random_covars, random_MMT)

        if util.check_optim(opt, hom2, 0, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, hom_REML_loglike, par,
                     args=(y, P, X, C, vs, random_MMT), method=method, nrep=nrep)
            hom2, r2, beta, l, fixed_vars, random_vars, Vy = extract(
                    out, C, y, X, P, vs, fixed_covars, random_covars, random_MMT)

        # wald
        Z = [np.eye(N)] + Rs
        D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

        reml = {'hom2':hom2, 'beta':beta, 'l':l, 'D':D, 'fixedeffect_vars':fixed_vars,
                'randomeffect_vars':random_vars, 'r2':r2, 'nu':np.mean(vs),  'opt':opt}
        p = {}
        p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], 
                np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par)
        return(reml, p)

    else:
        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars)
        n_par = 1 + n_random + X.shape[1]

        out = r_optim(y, P, vs, fixed_covars, random_covars, par, nrep, 'REML', 'hom', method)

        hom2, beta, l, hess = out['hom2'][0], np.array(out['beta']), out['l'][0], np.array(out['hess'])
        convergence = out['convergence'][0]
        fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars)
        beta_d = util.assign_beta(beta, P, fixed_covars)
        randomeffect_vars_d, r2_d  = util.assign_randomeffect_vars(np.array(out['randomeffect_vars']),
                np.array(out['r2']), random_covars)

        # wald
        Vy = np.diag(hom2 + vs)
        Z = [np.identity(len(y))]
        if len(r2_d.keys()) > 0:
            for key in np.sort( list(r2_d.keys()) ):
                M_ = random_covars[key]
                Vy = Vy + r2_d[key] * M_ @ M_.T
                Z.append( M_ )
        D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

        res = {'hom2':hom2, 'beta':beta_d, 'l':l, 'D':D, 'fixedeffect_vars':fixedeffect_vars_d,
                'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d, 'nu':np.mean(vs), 'hess':hess,
                'convergence':convergence}
        wald_p = {}
        wald_p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        # wald test beta1 = beta2 = beta3
        wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par)
        return(res, wald_p)

def hom_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('Hom HE')
    start = time.time()

    def he_f(y, P, vs, fixed_covars, random_covars):
        N, C = P.shape
        X = get_X(P, fixed_covars)

        theta = he_ols(y, P, vs, fixed_covars, random_covars, model='hom')
        hom2, r2 = theta[0], theta[1:]

        # 
        random_MMT = get_MMT( random_covars )
        Vy = cal_Vy( P, vs, hom2, np.zeros((C,C)), r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

        return( hom2, r2, beta, fixed_vars, random_vars )

    # par
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d)

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    X = get_X(P, fixed_covars)
    n_par = 1 + n_random + X.shape[1]

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
        p['ct_beta'] = util.wald_ct_beta( beta['ct_beta'], var_beta, n=N, P=n_par )

    print( time.time() - start )
    return( he, p )

def iid_ML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.eye(C) * par[1]
    beta = par[2:(2+X.shape[1])]
    r2 = par[(2+X.shape[1]):]

    return( ML_LL(y, P, X, vs, beta, hom2, V, r2, random_MMT) )

def iid_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10, optim_by_R=False):
    print('IID ML')

    if not optim_by_R:
        def extract( out, C, X, P, fixed_covars, random_covars ):
            hom2, beta, r2 = out['x'][0], out['x'][2:(2+X.shape[1])], out['x'][(2+X.shape[1]):]
            V = np.eye(C) * out['x'][1]
            l = out['fun'] * (-1)
            ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
            # calcualte variance of fixed and random effects, and convert to dict
            beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
            return( hom2, beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars )

        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

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

        out, opt = util.optim(iid_ML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

        hom2, beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract(
                out, C, X, P, fixed_covars, random_covars )

        if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, iid_ML_loglike, par, args=(y, P, X, C, vs, random_MMT), 
                    method=method, nrep=nrep)
            hom2, beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract( 
                    out, C, X, P, fixed_covars, random_covars )

        # wald
        Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
        Z = [np.identity(len(y)), linalg.khatri_rao(np.eye(N), P.T).T]
        Z = Z + Rs
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
    else:
        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars)
        n_par = 1 + 1 + n_random + X.shape[1]

        out = r_optim(y, P, vs, fixed_covars, random_covars, par, nrep, 'ML', 'iid', method)

        hom2, beta, V, l, hess = out['hom2'][0], np.array(out['beta']), np.array(out['V']), out['l'][0], np.array(out['hess'])
        convergence = out['convergence'][0]
        ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
        fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars)
        beta_d = util.assign_beta(beta, P, fixed_covars)
        randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
                np.array(out['r2']), random_covars)

        # wald
        Vy = np.diag(hom2 + np.diag(P @ V @ P.T) + vs)
        Z = [np.identity(len(y)), scipy.linalg.khatri_rao(np.identity(len(y)), P.T).T]
        if len(r2_d.keys()) > 0:
            for key in np.sort( list(r2_d.keys()) ):
                M_ = random_covars[key]
                Vy = Vy + r2_d[key] * M_ @ M_.T
                Z.append( M_ )
        D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
        res = {'hom2':hom2, 'beta':beta_d, 'V':V, 'l':l, 'D':D,
                'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
                'fixedeffect_vars':fixedeffect_vars_d, 'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
                'nu':np.mean(vs), 'hess':hess, 'convergence':convergence}

        wald_p = {}
        wald_p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
        wald_p['V'] = wald.wald_test(V[0,0], 0, D[X.shape[1]+1,X.shape[1]+1], N-n_par)
        wald_p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
        wald_p['ct_beta'] = util.wald_ct_beta(beta[:C], D[:C,:C], n=N, P=n_par)
        return(res, wald_p)

def iid_REML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.eye(C) * par[1]
    r2 = par[2:]
    return( REML_LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )

def iid_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10, optim_by_R=False):
    print('IID REML')

    if not optim_by_R:
        def extract(out, C, y, X, P, vs, fixed_covars, random_covars, random_MMT):
            hom2, r2 = out['x'][0], out['x'][2:]
            V = np.eye(C) * out['x'][1]
            l = out['fun'] * (-1)
            Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
            beta = util.glse( Vy, X, y )
            # calcualte variance of fixed and random effects, and convert to dict
            beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
            ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
            return(hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, 
                    ct_overall_var, ct_specific_var)

        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars)
        n_par = 1 + 1 + n_random + X.shape[1]

        # optim
        if par is None:
            beta = np.linalg.inv( X.T @ X) @ (X.T @ y)
            hom2 = np.var(y - X @ beta) / ( n_random + 2 )
            par = [hom2] * (n_random + 2)

        out, opt = util.optim(iid_REML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

        hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var = extract(
                out, C, y, X, P, vs, fixed_covars, random_covars, random_MMT)

        if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, iid_REML_loglike, par, 
                    args=(y, P, X, C, vs, random_MMT), method=method, nrep=nrep)
            hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var = extract(
                    out, C, y, X, P, vs, fixed_covars, random_covars, random_MMT)

        # wald
        Z = [np.eye(N), linalg.khatri_rao(np.eye(N), P.T).T]
        Z = Z + Rs
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
                n=N, P=n_par)
        return(reml, p)
    else:
        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars)
        n_par = 1 + 1 + n_random + X.shape[1]

        out = r_optim(y, P, vs, fixed_covars, random_covars, par, nrep, 'REML', 'iid', method)

        beta, hom2, V, l, hess = np.array(out['beta']), out['hom2'][0], np.array(out['V']), out['l'][0], np.array(out['hess'])
        convergence = out['convergence'][0]
        fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars)
        beta_d = util.assign_beta(beta, P, fixed_covars)
        ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
        randomeffect_vars_d, r2_d  = util.assign_randomeffect_vars(np.array(out['randomeffect_vars']),
                np.array(out['r2']), random_covars)

        # wald
        Vy = np.diag(hom2 + np.diag(P @ V @ P.T) + vs)
        Z = [np.identity(len(y)), scipy.linalg.khatri_rao(np.identity(len(y)), P.T).T]
        if len(r2_d.keys()) > 0:
            for key in np.sort( list(r2_d.keys()) ):
                M_ = random_covars[key]
                Vy = Vy + r2_d[key] * M_ @ M_.T
                Z.append( M_ )
        D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

        res = {'hom2':hom2, 'V':V, 'beta':beta_d, 'l':l, 'D':D, 'ct_random_var':ct_random_var,
                'ct_specific_random_var':ct_specific_random_var, 'randomeffect_vars':randomeffect_vars_d,
                'r2':r2_d, 'fixedeffect_vars':fixedeffect_vars_d, 'nu':np.mean(vs), 'hess':hess,
                'convergence':convergence}
        wald_p = {}
        wald_p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        wald_p['V'] = wald.wald_test(V[0,0], 0, D[1,1], N-n_par)
        # wald test beta1 = beta2 = beta3
        wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par)
        return(res, wald_p)

def iid_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('IID HE')
    start = time.time()

    def he_f(y, P, vs, fixed_covars, random_covars):
        N, C = P.shape
        X = get_X(P, fixed_covars)

        theta = he_ols(y, P, vs, fixed_covars, random_covars, model='iid')
        hom2, het, r2 = theta[0], theta[1], theta[2:]
        V = np.eye(C) * het

        # 
        random_MMT = get_MMT( random_covars )
        Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return( hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var )

    # par
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d)

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    X = get_X(P, fixed_covars)
    n_par = 1 + 1 + n_random + X.shape[1]

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
        p['ct_beta'] = util.wald_ct_beta( beta['ct_beta'], var_beta, n=N, P=n_par )

    print( time.time() - start )
    return( he, p )

def free_ML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.diag( par[1:(C+1)] )
    beta = par[(C+1):(C+1+X.shape[1])]
    r2 = par[(C+1+X.shape[1]):]

    return( ML_LL(y, P, X, vs, beta, hom2, V, r2, random_MMT) )

def free_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10, optim_by_R=False):
    print('Free ML')

    if not optim_by_R:
        def extract( out, C, X, P, fixed_covars, random_covars ):
            hom2, beta, r2 = out['x'][0], out['x'][(C+1):(C+1+X.shape[1])], out['x'][(C+1+X.shape[1]):]
            V = np.diag( out['x'][1:(C+1)] )
            l = out['fun'] * (-1)
            ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
            # calcualte variance of fixed and random effects, and convert to dict
            beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
            return( hom2, beta, r2, V, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars )

        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

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
        Z = Z + Rs
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
    else:
        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars)
        n_par = 1 + C + X.shape[1] + n_random

        out = r_optim(y, P, vs, fixed_covars, random_covars, par, nrep, 'ML', 'free', method)

        hom2, beta, V, l, hess = out['hom2'][0], np.array(out['beta']), np.array(out['V']), out['l'][0], np.array(out['hess'])
        convergence = out['convergence'][0]
        fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars)
        beta_d = util.assign_beta(beta, P, fixed_covars)
        ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
        randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
                np.array(out['r2']), random_covars)

        # wald
        Vy = np.diag(hom2 + np.diag(P @ V @ P.T) + vs)
        Z = [np.identity(len(y))] + [np.diag(P[:,i]) for i in range(C)]
        if len(r2_d.keys()) > 0:
            for key in np.sort( list(r2_d.keys()) ):
                M_ = random_covars[key]
                Vy = Vy + r2_d[key] * M_ @ M_.T
                Z.append( M_ )
        D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
        res = {'hom2':hom2, 'beta':beta_d, 'V':V, 'l':l, 'D':D,
                'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
                'fixedeffect_vars':fixedeffect_vars_d, 'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
                'nu':np.mean(vs), 'hess':hess, 'convergence':convergence}
        wald_p = {}
        wald_p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
        wald_p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), D[(X.shape[1]+1):(X.shape[1]+C+1), (X.shape[1]+1):(X.shape[1]+C+1)],
                n=N, P=n_par)
        wald_p['Vi'] = [wald.wald_test(V[i,i], 0, D[X.shape[1]+i+1,X.shape[1]+i+1], N-n_par) for i in range(C)]
        wald_p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
        wald_p['ct_beta'] = util.wald_ct_beta(beta[:C], D[:C,:C], n=N, P=n_par )

        return(res, wald_p)


def free_REML_loglike(par, y, P, X, C, vs, random_MMT):
    hom2 = par[0]
    V = np.diag( par[1:(C+1)] )
    r2 = par[(C+1):]
    return( REML_LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )

def free_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10, 
        jack_knife=False, optim_by_R=False):
    print('Free REML')

    if not optim_by_R:
        def reml_f(y, P, vs, fixed_covars, random_covars, par, method):
            def extract(out, C, y, X, P, vs, fixed_covars, random_covars, random_MMT):
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

            N, C = P.shape
            X = get_X(P, fixed_covars)

            random_MMT = get_MMT( random_covars )

            out, opt = util.optim(free_REML_loglike, par, args=(y, P, X, C, vs, random_MMT), method=method)

            hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var = extract(
                    out, C, y, X, P, vs, fixed_covars, random_covars, random_MMT)

            if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
                out, opt = util.re_optim(out, opt, free_REML_loglike, par, 
                        args=(y, P, X, C, vs, random_MMT), method=method, nrep=nrep)
                hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var = extract(
                        out, C, y, X, P, vs, fixed_covars, random_covars, random_MMT)

            return(hom2, V, r2, beta, l, fixed_vars, random_vars, Vy,
                    ct_overall_var, ct_specific_var, opt)

        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars)
        n_par = 1 + C + n_random + X.shape[1]

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
        Z = Z + Rs
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
                    n=N, P=n_par)
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
            p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par)

        return(reml, p)
    else:
        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars)
        n_par = 1 + C + n_random + X.shape[1]

        out = r_optim(y, P, vs, fixed_covars, random_covars, par, nrep, 'REML', 'free', method)

        hom2, V, beta, l, hess = out['hom2'][0], np.array(out['V']), np.array(out['beta']), out['l'][0], np.array(out['hess'])
        convergence = out['convergence'][0]
        fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars)
        beta_d = util.assign_beta(beta, P, fixed_covars)
        ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
        randomeffect_vars_d, r2_d  = util.assign_randomeffect_vars(np.array(out['randomeffect_vars']),
                np.array(out['r2']), random_covars)

        # wald
        Vy = np.diag(hom2 + np.diag(P @ V @ P.T) + vs)
        Z = [np.identity(len(y))] + [np.diag(P[:,i]) for i in range(C)]
        if len(r2_d.keys()) > 0:
            for key in np.sort( list(r2_d.keys()) ):
                M_ = random_covars[key]
                Vy = Vy + r2_d[key] * M_ @ M_.T
                Z.append( M_ )
        D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

        res = {'hom2':hom2, 'V':V, 'beta':beta_d, 'l':l, 'D':D, 'fixedeffect_vars':fixedeffect_vars_d,
                'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
                'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d, 'nu':np.mean(vs), 'hess':hess,
                'convergence':convergence}

        wald_p = {}
        if not jack_knife:
            wald_p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
            wald_p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), D[1:(C+1), 1:(C+1)], n=N, P=n_par)
            wald_p['Vi'] = [wald.wald_test(V[i,i], 0, D[i+1,i+1], N-n_par) for i in range(C)]
            # wald test beta1 = beta2 = beta3
            X = get_X(P, fixed_covars_d)
            wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                    n=N, P=n_par)
        else:
            jacks = {'ct_beta':[], 'hom2':[], 'V':[]}
            for i in range(N):
                y_tmp, vs_tmp, fixed_covars_d_tmp, random_covars_d_tmp, P_tmp = util.jk_rmInd(
                        i, y, vs, fixed_covars, random_covars, P)

                out_tmp = r_optim(y_tmp, P_tmp, vs_tmp, fixed_covars_d_tmp, random_covars_d_tmp, 
                        par, nrep, 'REML', 'free', method)

                hom2_tmp, V_tmp, beta_tmp = out_tmp['hom2'][0], np.array(out_tmp['V']), np.array(out_tmp['beta'])
                ct_beta_tmp = util.assign_beta(beta_tmp, P_tmp, fixed_covars_d_tmp)['ct_beta']
                jacks['ct_beta'].append( ct_beta_tmp )
                jacks['hom2'].append( hom2_tmp )
                jacks['V'].append( np.diag(V_tmp) )

            var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
            var_V = (len(jacks['V']) - 1.0) * np.cov( np.array(jacks['V']).T, bias=True )
            var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )

            wald_p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
            wald_p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
            wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par)

        return(res, wald_p)

def free_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('Free HE')
    start = time.time()

    def he_f(y, P, vs, fixed_covars, random_covars):
        N, C = P.shape
        X = get_X(P, fixed_covars)

        theta = he_ols(y, P, vs, fixed_covars, random_covars, model='free')
        hom2, r2 = theta[0], theta[(1+C):]
        V = np.diag( theta[1:(C+1)] )

        # 
        random_MMT = get_MMT( random_covars )
        Vy = cal_Vy( P, vs, hom2, V, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return( hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var )

    # par
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d)

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    X = get_X(P, fixed_covars)
    n_par = 1 + C + n_random + X.shape[1]

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
        p['ct_beta'] = util.wald_ct_beta( beta['ct_beta'], var_beta, n=N, P=n_par )

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

def full_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10, optim_by_R=False):
    print('Full ML')

    if not optim_by_R:
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
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

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
    else:
        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars_d)

        out = r_optim(y, P, vs, fixed_covars, random_covars, par, nrep, 'ML', 'full', method)

        beta, V, l, hess = np.array(out['beta']), np.array(out['V']), out['l'][0], np.array(out['hess'])
        convergence = out['convergence'][0]
        ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
        fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars)
        beta_d = util.assign_beta(beta, P, fixed_covars)
        randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
                np.array(out['r2']), random_covars)

        res = {'beta':beta_d, 'V':V, 'l':l, 'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
                'fixedeffect_vars':fixedeffect_vars_d, 'randomeffect_vars':randomeffect_vars_d,
                'r2':r2_d, 'nu':np.mean(vs), 'hess':hess, 'convergence':convergence}
        return(res)

def full_REML_loglike(par, y, P, X, C, vs, random_MMT):
    ngam = C * (C+1) // 2
    V = np.zeros((C,C))
    V[np.tril_indices(C)] = par[:ngam]
    V = V + V.T
    hom2 = 0
    r2 = par[ngam:]
    return( REML_LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )

def full_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, method=None, nrep=10, optim_by_R=False):
    print('Full REML')

    if not optim_by_R:
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
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

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
    else:
        # par
        fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
                fixed_covars_d, random_covars_d)

        y = np.loadtxt(y_f)
        P = np.loadtxt(P_f)
        vs = np.loadtxt(nu_f)
        N, C = P.shape
        X = get_X(P, fixed_covars)

        out = r_optim(y, P, vs, fixed_covars, random_covars, par, nrep, 'REML', 'full', method)

        V, beta, l, hess = np.array(out['V']), np.array(out['beta']), out['l'][0], np.array(out['hess'])
        convergence = out['convergence'][0]
        fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars)
        beta_d = util.assign_beta(beta, P, fixed_covars)
        ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
        randomeffect_vars_d, r2_d  = util.assign_randomeffect_vars(np.array(out['randomeffect_vars']),
                np.array(out['r2']), random_covars)

        res = {'V':V, 'beta':beta_d, 'l':l, 'ct_random_var':ct_random_var,
                'ct_specific_random_var':ct_specific_random_var,
                'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
                'fixedeffect_vars':fixedeffect_vars_d, 'nu':np.mean(vs), 'hess':hess, 'convergence':convergence}
        return(res)

def full_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}):
    print('Full HE')
    start = time.time()

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
        random_MMT = get_MMT( random_covars )
        Vy = cal_Vy( P, vs, 0, V, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return( V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var )

    # par
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d)

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    X = get_X(P, fixed_covars)

    V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var = he_f(
            y, P, vs, fixed_covars, random_covars)
    he = {'V':V, 'r2':r2, 'beta':beta, 'nu':np.mean(vs),
            'fixedeffect_vars':fixed_vars, 'randomeffect_vars':random_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var}

    print( time.time() - start )
    return( he )

