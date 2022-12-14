import os, sys, re, multiprocessing, time
import helper, mystats 
import scipy
import numpy as np, pandas as pd
import rpy2.robjects as robjects 
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from rpy2.robjects.conversion import localconverter
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

def he_fun(M, Y, proj, random_covars_array_d={}):
    M_ = M
    for key in np.sort( list(random_covars_array_d.keys()) ):
        Q = random_covars_array_d[key]
        m = (proj @ Q @ Q.T @ proj).flatten('F').reshape((-1,1))
        M = np.concatenate( ( M, m ), axis=1 )

    theta_ = np.linalg.inv( M.T @ M ) @ M.T @ Y
    theta = {'var':theta_[:M_.shape[1]], 'r2':{}}
    theta_ = theta_[M_.shape[1]:]
    for key in np.sort( list(random_covars_array_d.keys()) ):
        theta['r2'][key] = theta_[0]
        theta_ = theta_[1:]
    #vars = (((Y - M @ ests)**2).sum() / len(Y)) * np.linalg.inv( M.T @ M )
    return( theta )

def hom_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None):
    print('Hom ML')

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars_d)
    n_par = 1 + len(random_covars_d.keys()) + X.shape[1]

    def hom_ml_subprocess(y, P_f, vs, fixed_covars_d, random_covars_d, out, par):
        ong_ml_rf = 'bin/ong.ml.R'
        ong_ml_r = STAP( open(ong_ml_rf).read(), 'ong_ml_r' )
        if par is None:
            out_ = ong_ml_r.screml_hom(y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d) )
        else:
            out_ = ong_ml_r.screml_hom(y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d), par=robjects.FloatVector(par) )
        for key, value in zip(out_.names, list(out_)):
            out[key] = value

    manager = multiprocessing.Manager()
    out = manager.dict()
    p = multiprocessing.Process(target=hom_ml_subprocess,
            args=(y, P_f, vs, fixed_covars_d, random_covars_d, out, par))
    p.start()
    p.join()

    hom2, beta, l, hess = out['hom2'][0], np.array(out['beta']), out['l'][0], np.array(out['hess'])
    convergence = out['convergence'][0]
    fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars_d)
    beta_d = util.assign_beta(beta, P, fixed_covars_d)
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars_d )

    # wald
    Vy = np.diag(hom2 + vs)
    Z = [np.identity(len(y))]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.loadtxt( random_covars_d[key] )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
    ml = {'hom2':hom2, 'beta':beta_d, 'l':l, 'D':D, 'fixedeffect_vars':fixedeffect_vars_d,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d, 'nu':np.mean(vs), 'hess':hess, 
            'convergence':convergence}

    wald_p = {}
    wald_p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    wald_p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
    wald_p['ct_beta'] = util.wald_ct_beta(beta[:C], D[:C,:C], n=N, P=n_par)
    return(ml, wald_p)

def hom_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None):
    print('Hom REML')
    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars_d)
    n_par = 1 + len(random_covars_d.keys())

    def hom_reml_subprocess(y, P_f, vs, fixed_covars_d, random_covars_d, out, par):
        ong_reml_rf = 'bin/ong.reml.R'
        ong_reml_r = STAP( open(ong_reml_rf).read(), 'ong_reml_r' )
        if par is None:
            out_ = ong_reml_r.screml_hom( y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d) )
        else:
            out_ = ong_reml_r.screml_hom( y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d), par=robjects.FloatVector(par) )
        for key, value in zip(out_.names, list(out_)):
            out[key] = value

    manager = multiprocessing.Manager()
    out = manager.dict()
    p = multiprocessing.Process(target=hom_reml_subprocess,
            args=(y, P_f, vs, fixed_covars_d, random_covars_d, out, par))
    p.start()
    p.join()
    
    hom2, beta, l, hess = out['hom2'][0], np.array(out['beta']), out['l'][0], np.array(out['hess'])
    convergence = out['convergence'][0]
    fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars_d)
    beta_d = util.assign_beta(beta, P, fixed_covars_d)
    randomeffect_vars_d, r2_d  = util.assign_randomeffect_vars(np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars_d)

    # wald
    Vy = np.diag(hom2 + vs)
    Z = [np.identity(len(y))]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.loadtxt( random_covars_d[key] )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'hom2':hom2, 'beta':beta_d, 'l':l, 'D':D, 'fixedeffect_vars':fixedeffect_vars_d,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d, 'nu':np.mean(vs), 'hess':hess, 
            'convergence':convergence}
    wald_p = {}
    wald_p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
    # wald test beta1 = beta2 = beta3
    X = get_X(P, fixed_covars_d)
    wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
            n=N, P=n_par+X.shape[1])
    return(reml, wald_p)

def hom_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('Hom HE')
    start = time.time()

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    D = np.diag(vs)
    X = get_X(P, fixed_covars_d)
    n_par = 1 + len(random_covars_d.keys())

    random_covars_array_d = {}
    for key in random_covars_d.keys():
        random_covars_array_d[key] = np.loadtxt( random_covars_d[key] )

    def hom_HE_(X, y, P, D, random_covars_array_d):
        N = len(y)
        proj = np.eye( N ) - X @ np.linalg.inv(X.T @ X) @ X.T

        # project out fixed effects
        y_p = proj @ y
        t = ( np.outer(y_p, y_p) - proj @ D @ proj ).flatten('F')

        M = proj.flatten('F').reshape((-1,1))

        # estimate variances for random effects
        theta = he_fun( M, t, proj, random_covars_array_d )
        hom2 = theta['var'][0]

        # 
        sig2s = hom2 * np.eye(N) + D
        for key in np.sort( list(random_covars_array_d.keys()) ):
            Q = random_covars_array_d[key]
            sig2s += ( Q @ Q.T ) * theta['r2'][key]

        beta = util.glse( sig2s, X, y )
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_d )
        theta['beta'] = beta_d
        return( theta )

    theta = hom_HE_(X, y, P, D, random_covars_array_d)
    hom2 = theta['var'][0]
    ct_beta = theta['beta']['ct_beta']
    he = {'hom2':hom2, 'r2':theta['r2'], 'beta':theta['beta']}

    # jackknife
    if jack_knife:
        jacks = []
        for i in range(N):
            random_covars_array_d_ = {}
            for key in random_covars_array_d.keys():
                random_covars_array_d_[key] = np.delete(random_covars_array_d[key], i, axis=0)
            jacks.append( hom_HE_( np.delete(X,i,axis=0), np.delete(y,i), np.delete(P,i,axis=0),
                np.diag(np.delete(vs,i)), random_covars_array_d_ ) )
        jacks_hom2 = [x['var'][0] for x in jacks]
        var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
        jacks_beta = [x['beta']['ct_beta']  for x in jacks]
        var_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_beta).T, bias=True)

        p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par)}
        p['ct_beta'] = util.wald_ct_beta( ct_beta, var_beta, n=N, P=n_par+X.shape[1] )
    else:
        p = {'hom2':1, 'ct_beta':1}

    print( time.time() - start )
    return( he, p )

def iid_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None):
    print('IID ML')
    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars_d)
    n_par = 1 + 1 + len(random_covars_d.keys()) + X.shape[1]

    def iid_ml_subprocess(y, P_f, vs, fixed_covars_d, random_covars_d, out, par):
        ong_ml_rf = 'bin/ong.ml.R'
        ong_ml_r = STAP( open(ong_ml_rf).read(), 'ong_ml_r' )
        if par is None:
            out_ = ong_ml_r.screml_iid(y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d) )
        else:
            out_ = ong_ml_r.screml_iid(y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d), par=robjects.FloatVector(par) )
        for key, value in zip(out_.names, list(out_)):
            out[key] = value

    manager = multiprocessing.Manager()
    out = manager.dict()
    p = multiprocessing.Process(target=iid_ml_subprocess,
            args=(y, P_f, vs, fixed_covars_d, random_covars_d, out, par))
    p.start()
    p.join()

    hom2, beta, V, l, hess = out['hom2'][0], np.array(out['beta']), np.array(out['V']), out['l'][0], np.array(out['hess'])
    convergence = out['convergence'][0]
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars_d)
    beta_d = util.assign_beta(beta, P, fixed_covars_d)
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars_d)

    # wald
    Vy = np.diag(hom2 + np.diag(P @ V @ P.T) + vs)
    Z = [np.identity(len(y)), scipy.linalg.khatri_rao(np.identity(len(y)), P.T).T]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.loadtxt( random_covars_d[key] )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
    ml = {'hom2':hom2, 'beta':beta_d, 'V':V, 'l':l, 'D':D,
            'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
            'fixedeffect_vars':fixedeffect_vars_d, 'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'nu':np.mean(vs), 'hess':hess, 'convergence':convergence}

    wald_p = {}
    wald_p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    wald_p['V'] = wald.wald_test(V[0,0], 0, D[X.shape[1]+1,X.shape[1]+1], N-n_par)
    wald_p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
    wald_p['ct_beta'] = util.wald_ct_beta(beta[:C], D[:C,:C], n=N, P=n_par)
    return(ml, wald_p)

def iid_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None):
    print('IID REML')
    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars_d)
    n_par = 1 + 1 + len(random_covars_d.keys())

    def iid_reml_subprocess(y, P_f, vs, fixed_covars_d, random_covars_d, out, par):
        ong_reml_rf = 'bin/ong.reml.R'
        ong_reml_r = STAP( open(ong_reml_rf).read(), 'ong_reml_r' )
        if par is None:
            out_ = ong_reml_r.screml_iid( y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d) )
        else:
            out_ = ong_reml_r.screml_iid( y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d), par=robjects.FloatVector(par) )
        for key, value in zip(out_.names, list(out_)):
            out[key] = value

    manager = multiprocessing.Manager()
    out = manager.dict()
    p = multiprocessing.Process(target=iid_reml_subprocess,
            args=(y, P_f, vs, fixed_covars_d, random_covars_d, out, par))
    p.start()
    p.join()
    
    beta, hom2, V, l, hess = np.array(out['beta']), out['hom2'][0], np.array(out['V']), out['l'][0], np.array(out['hess'])
    convergence = out['convergence'][0]
    fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars_d)
    beta_d = util.assign_beta(beta, P, fixed_covars_d)
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    randomeffect_vars_d, r2_d  = util.assign_randomeffect_vars(np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars_d)

    # wald
    Vy = np.diag(hom2 + np.diag(P @ V @ P.T) + vs)
    Z = [np.identity(len(y)), scipy.linalg.khatri_rao(np.identity(len(y)), P.T).T]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.loadtxt( random_covars_d[key] )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'hom2':hom2, 'V':V, 'beta':beta_d, 'l':l, 'D':D, 'ct_random_var':ct_random_var,
            'ct_specific_random_var':ct_specific_random_var, 'randomeffect_vars':randomeffect_vars_d,
            'r2':r2_d, 'fixedeffect_vars':fixedeffect_vars_d, 'nu':np.mean(vs), 'hess':hess, 
            'convergence':convergence}
    wald_p = {}
    wald_p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
    wald_p['V'] = wald.wald_test(V[0,0], 0, D[1,1], N-n_par)
    # wald test beta1 = beta2 = beta3
    X = get_X(P, fixed_covars_d)
    wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C], 
            n=N, P=n_par+X.shape[1])
    return(reml, wald_p)

def iid_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('IID HE')
    start = time.time()

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    D = np.diag(vs)
    X = get_X(P, fixed_covars_d)
    n_par = 1 + 1 + len(random_covars_d.keys())

    random_covars_array_d = {}
    for key in random_covars_d.keys():
        random_covars_array_d[key] = np.loadtxt( random_covars_d[key] )

    def iid_HE_(X, y, P, D, random_covars_array_d):
        N = len(y)
        proj = np.eye( N ) - X @ np.linalg.inv(X.T @ X) @ X.T

        # project out fixed effects
        y_p = proj @ y
        t = ( np.outer(y_p, y_p) - proj @ D @ proj ).flatten('F')

        M = np.array( [proj.flatten('F'),
            ( proj @ np.diag(np.diag(P @ P.T)) @ proj).flatten('F')] ).T

        # estimate variances for random effects
        theta = he_fun( M, t, proj, random_covars_array_d )
        hom2 = theta['var'][0]
        het = theta['var'][1]

        # 
        sig2s = hom2 * np.eye(N) + het * np.diag(np.diag(P @ P.T)) + D
        for key in np.sort( list(random_covars_array_d.keys()) ):
            Q = random_covars_array_d[key]
            sig2s += ( Q @ Q.T ) * theta['r2'][key]

        beta = util.glse( sig2s, X, y )
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_d )
        theta['beta'] = beta_d
        return( theta )

    theta = iid_HE_(X, y, P, D, random_covars_array_d)
    hom2 = theta['var'][0]
    het = theta['var'][1]
    ct_beta = theta['beta']['ct_beta']
    he = {'hom2':hom2, 'V':het * np.eye(C), 'r2':theta['r2'], 'beta':theta['beta']}

    # jackknife
    if jack_knife:
        jacks = []
        for i in range(N):
            random_covars_array_d_ = {}
            for key in random_covars_array_d.keys():
                random_covars_array_d_[key] = np.delete(random_covars_array_d[key], i, axis=0)
            jacks.append( iid_HE_( np.delete(X,i,axis=0), np.delete(y,i), np.delete(P,i,axis=0),
                np.diag(np.delete(vs,i)), random_covars_array_d_ ) )
        jacks_hom2 = [x['var'][0] for x in jacks]
        var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
        jacks_het = [x['var'][1] for x in jacks]
        var_het = (len(jacks) - 1.0) * np.var(jacks_het)
        jacks_beta = [x['beta']['ct_beta']  for x in jacks]
        var_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_beta).T, bias=True)

        p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par), 
                'V': wald.wald_test(het, 0, var_het, N-n_par)}
        p['ct_beta'] = util.wald_ct_beta( ct_beta, var_beta, n=N, P=n_par+X.shape[1] )
    else:
        p = {'hom2':1, 'V':1, 'ct_beta':1}

    print( time.time() - start )
    return( he, p )

def free_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None):
    print('Free ML')
    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars_d)
    n_par = 1 + C + X.shape[1] + len(random_covars_d.keys()) # hom2, V, beta (ct beta & fixed covars), random covars

    def free_ml_subprocess(y, P_f, vs, fixed_covars_d, random_covars_d, out, par):
        ong_ml_rf = 'bin/ong.ml.R'
        ong_ml_r = STAP( open(ong_ml_rf).read(), 'ong_ml_r' )
        if par is None:
            out_ = ong_ml_r.screml_free(y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d) )
        else:
            out_ = ong_ml_r.screml_free(y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d), par=robjects.FloatVector(par) )
        for key, value in zip(out_.names, list(out_)):
            out[key] = value

    manager = multiprocessing.Manager()
    out = manager.dict()
    p = multiprocessing.Process(target=free_ml_subprocess,
            args=(y, P_f, vs, fixed_covars_d, random_covars_d, out, par))
    p.start()
    p.join()

    hom2, beta, V, l, hess = out['hom2'][0], np.array(out['beta']), np.array(out['V']), out['l'][0], np.array(out['hess'])
    convergence = out['convergence'][0]
    fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars_d)
    beta_d = util.assign_beta(beta, P, fixed_covars_d)
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars_d)

    # wald
    Vy = np.diag(hom2 + np.diag(P @ V @ P.T) + vs)
    Z = [np.identity(len(y))] + [np.diag(P[:,i]) for i in range(C)]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.loadtxt( random_covars_d[key] )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)
    ml = {'hom2':hom2, 'beta':beta_d, 'V':V, 'l':l, 'D':D,
            'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
            'fixedeffect_vars':fixedeffect_vars_d, 'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'nu':np.mean(vs), 'hess':hess, 'hess2':np.array(out['hess2']), 'convergence':convergence}

    wald_p = {}
    wald_p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    wald_p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), D[(X.shape[1]+1):(X.shape[1]+C+1), (X.shape[1]+1):(X.shape[1]+C+1)], 
            n=N, P=n_par)
    wald_p['Vi'] = [wald.wald_test(V[i,i], 0, D[X.shape[1]+i+1,X.shape[1]+i+1], N-n_par) for i in range(C)]
    wald_p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], N-n_par, two_sided=True) for i in range(X.shape[1])]
    wald_p['ct_beta'] = util.wald_ct_beta(beta[:C], D[:C,:C], n=N, P=n_par )

    return(ml, wald_p)

def free_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None, jack_knife=False):
    print('Free REML')
    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars_d)
    n_par = 1 + C + len(random_covars_d.keys()) # hom2, V, random covars

    def free_reml_subprocess(y, P, vs, fixed_covars_d, random_covars_d, out, par):
        ong_reml_rf = 'bin/ong.reml.R'
        ong_reml_r = STAP( open(ong_reml_rf).read(), 'ong_reml_r' )
        numpy2ri.activate()
        if par is None:
            out_ = ong_reml_r.screml_free( y=robjects.FloatVector(y), P=r['as.matrix'](P),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d) )
        else:
            out_ = ong_reml_r.screml_free( y=robjects.FloatVector(y), P=r['as.matrix'](P),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d), par=robjects.FloatVector(par) )
        numpy2ri.deactivate()
        for key, value in zip(out_.names, list(out_)):
            out[key] = value

    manager = multiprocessing.Manager()
    out = manager.dict()
    p = multiprocessing.Process(target=free_reml_subprocess,
            args=(y, P, vs, fixed_covars_d, random_covars_d, out, par))
    p.start()
    p.join()
    
    hom2, V, beta, l, hess = out['hom2'][0], np.array(out['V']), np.array(out['beta']), out['l'][0], np.array(out['hess'])
    convergence = out['convergence'][0]
    fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars_d)
    beta_d = util.assign_beta(beta, P, fixed_covars_d)
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    randomeffect_vars_d, r2_d  = util.assign_randomeffect_vars(np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars_d)

    # wald
    Vy = np.diag(hom2 + np.diag(P @ V @ P.T) + vs)
    Z = [np.identity(len(y))] + [np.diag(P[:,i]) for i in range(C)]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.loadtxt( random_covars_d[key] )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'hom2':hom2, 'V':V, 'beta':beta_d, 'l':l, 'D':D, 'fixedeffect_vars':fixedeffect_vars_d, 
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
                n=N, P=n_par+X.shape[1])
    else:
        jacks = {'ct_beta':[], 'hom2':[], 'V':[]}
        for i in range(N):
            y_tmp, vs_tmp, fixed_covars_d_tmp, random_covars_d_tmp, P_tmp = cuomo_ctng_test.he_jackknife_rmInd(
                    i, y, vs, fixed_covars_d, random_covars_d, P)

            manager = multiprocessing.Manager()
            out_tmp = manager.dict()
            p = multiprocessing.Process(target=free_reml_subprocess,
                    args=(y_tmp, P_tmp, vs_tmp, fixed_covars_d_tmp, random_covars_d_tmp, out_tmp, par))
            p.start()
            p.join()
            
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
        wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+X.shape[1])

    return(reml, wald_p)

def free_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('Free HE')
    start = time.time()

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    D = np.diag(vs)
    X = get_X(P, fixed_covars_d)
    n_par = 1 + C + len(random_covars_d.keys()) # hom2, V, random covars

    random_covars_array_d = {}
    for key in random_covars_d.keys():
        random_covars_array_d[key] = np.loadtxt( random_covars_d[key] )

    def free_HE_(X, y, P, D, random_covars_array_d):
        N = len(y)
        proj = np.eye( N ) - X @ np.linalg.inv(X.T @ X) @ X.T

        # project out fixed effects
        y_p = proj @ y
        t = ( np.outer(y_p, y_p) - proj @ D @ proj ).flatten('F')

        M = [proj.flatten('F')]
        for i in range(C):
            M.append( (proj @ np.diag( P[:,i]**2 ) @ proj).flatten('F') )
        M = np.array( M ).T

        # estimate variances for random effects
        theta = he_fun( M, t, proj, random_covars_array_d )
        hom2 = theta['var'][0]
        V = np.diag(theta['var'][1:])

        # 
        sig2s = hom2 * np.eye(N) + np.diag(np.diag(P @ V @ P.T)) + D
        for key in np.sort( list(random_covars_array_d.keys()) ):
            Q = random_covars_array_d[key]
            sig2s += ( Q @ Q.T ) * theta['r2'][key]

        beta = util.glse( sig2s, X, y )
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_d )
        theta['beta'] = beta_d
        return( theta )

    theta = free_HE_(X, y, P, D, random_covars_array_d)
    hom2 = theta['var'][0]
    V = np.diag(theta['var'][1:])
    ct_beta = theta['beta']['ct_beta']
    he = {'hom2': hom2, 'V': V, 'r2':theta['r2'], 'beta': theta['beta']}

    # jackknife
    if jack_knife:
        jacks = []
        for i in range(N):
            random_covars_array_d_ = {}
            for key in random_covars_array_d.keys():
                random_covars_array_d_[key] = np.delete(random_covars_array_d[key], i, axis=0)
            jacks.append( free_HE_( np.delete(X,i,axis=0), np.delete(y,i), np.delete(P,i,axis=0),
                np.diag(np.delete(vs,i)), random_covars_array_d_ ) )
        jacks_hom2 = [x['var'][0] for x in jacks]
        var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
        jacks_V = [x['var'][1:] for x in jacks]
        var_V = (len(jacks) - 1.0) * np.cov(np.array(jacks_V).T, bias=True)
        jacks_beta = [x['beta']['ct_beta']  for x in jacks]
        var_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_beta).T, bias=True)

        p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par), 
                'V': wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par),
                'V_iid': util.wald_ct_beta(np.diag(V), var_V, n=N, P=n_par)}
        p['Vi'] = [wald.wald_test(V[i,i], 0, var_V[i,i], N-n_par) for i in range(C)]
        p['ct_beta'] = util.wald_ct_beta( ct_beta, var_beta, n=N, P=n_par+X.shape[1] )
    else:
        p = {'hom2':1, 'V':1, 'Vi':np.ones(C), 'ct_beta':1}

    print( time.time() - start )
    return( he, p )

def full_ML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None):
    print('Full ML')
    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars_d)

    def full_ml_subprocess(y, P_f, vs, fixed_covars_d, random_covars_d, out, par):
        ong_ml_rf = 'bin/ong.ml.R'
        ong_ml_r = STAP( open(ong_ml_rf).read(), 'ong_ml_r' )
        if par is None:
            out_ = ong_ml_r.screml_full(y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d) )
        else:
            out_ = ong_ml_r.screml_full(y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d), par=robjects.FloatVector(par) )
        for key, value in zip(out_.names, list(out_)):
            out[key] = value

    manager = multiprocessing.Manager()
    out = manager.dict()
    p = multiprocessing.Process(target=full_ml_subprocess,
            args=(y, P_f, vs, fixed_covars_d, random_covars_d, out, par))
    p.start()
    p.join()

    beta, V, l, hess = np.array(out['beta']), np.array(out['V']), out['l'][0], np.array(out['hess'])
    convergence = out['convergence'][0]
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars_d)
    beta_d = util.assign_beta(beta, P, fixed_covars_d)
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars_d)

    ml = {'beta':beta_d, 'V':V, 'l':l, 'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
            'fixedeffect_vars':fixedeffect_vars_d, 'randomeffect_vars':randomeffect_vars_d,
            'r2':r2_d, 'nu':np.mean(vs), 'hess':hess, 'convergence':convergence}
    return(ml)

def full_REML(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}, par=None):
    print('Full REML')
    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(nu_f)
    N, C = P.shape
    X = get_X(P, fixed_covars_d)

    def full_reml_subprocess(y, P_f, vs, fixed_covars_d, random_covars_d, out, par):
        ong_reml_rf = 'bin/ong.reml.R'
        ong_reml_r = STAP( open(ong_reml_rf).read(), 'ong_reml_r' )
        if par is None:
            out_ = ong_reml_r.screml_full( y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d) )
        else:
            out_ = ong_reml_r.screml_full( y=robjects.FloatVector(y), P=r['as.matrix'](r['read.table'](P_f)),
                    vs=robjects.FloatVector(vs), fixed=util.dict2Rlist(fixed_covars_d),
                    random=util.dict2Rlist(random_covars_d), par=robjects.FloatVector(par) )
        for key, value in zip(out_.names, list(out_)):
            out[key] = value

    manager = multiprocessing.Manager()
    out = manager.dict()
    p = multiprocessing.Process(target=full_reml_subprocess,
            args=(y, P_f, vs, fixed_covars_d, random_covars_d, out, par))
    p.start()
    p.join()
    
    V, beta, l, hess = np.array(out['V']), np.array(out['beta']), out['l'][0], np.array(out['hess'])
    convergence = out['convergence'][0]
    fixedeffect_vars_d = util.assign_fixedeffect_vars( np.array(out['fixedeffect_vars']), fixed_covars_d)
    beta_d = util.assign_beta(beta, P, fixed_covars_d)
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    randomeffect_vars_d, r2_d  = util.assign_randomeffect_vars(np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars_d)

    reml = {'V':V, 'beta':beta_d, 'l':l, 'ct_random_var':ct_random_var, 
            'ct_specific_random_var':ct_specific_random_var,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'fixedeffect_vars':fixedeffect_vars_d, 'nu':np.mean(vs), 'hess':hess, 'convergence':convergence}
    return(reml)

def full_HE(y_f, P_f, nu_f, fixed_covars_d={}, random_covars_d={}):
    print('Full HE')
    start = time.time()

    y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    N, C = P.shape
    vs = np.loadtxt(nu_f)
    D = np.diag(vs)
    X = get_X(P, fixed_covars_d)

    random_covars_array_d = {}
    for key in random_covars_d.keys():
        random_covars_array_d[key] = np.loadtxt( random_covars_d[key] )

    def full_HE_(X, y, P, D, random_covars_array_d):
        N = len(y)
        proj = np.eye( N ) - X @ np.linalg.inv(X.T @ X) @ X.T

        # project out fixed effects
        y_p = proj @ y
        t = ( np.outer(y_p, y_p) - proj @ D @ proj ).flatten('F')

        M = []
        for i in range(C):
            M.append( (proj @ np.diag( P[:,i]**2 ) @ proj).flatten('F') )
        for i in range(C-1):
            for j in range(i+1,C):
                M.append( 2*(proj @ np.diag( P[:,i] * P[:,j] ) @ proj).flatten('F') )
        M = np.array( M ).T

        # estimate variances for random effects
        theta = he_fun( M, t, proj, random_covars_array_d )
        V = np.zeros((C,C))
        V[np.triu_indices(C,k=1)] = theta['var'][C:]
        V = V + V.T + np.diag(theta['var'][:C])

        # 
        sig2s = np.diag(np.diag(P @ V @ P.T)) + D
        for key in np.sort( list(random_covars_array_d.keys()) ):
            Q = random_covars_array_d[key]
            sig2s += ( Q @ Q.T ) * theta['r2'][key]

        beta = util.glse( sig2s, X, y )
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_d )
        theta['beta'] = beta_d
        return( theta )

    theta = full_HE_(X, y, P, D, random_covars_array_d)
    V = np.zeros((C,C))
    V[np.triu_indices(C,k=1)] = theta['var'][C:]
    V = V + V.T + np.diag(theta['var'][:C])
    ct_beta = theta['beta']['ct_beta']
    he = {'V': V, 'r2':theta['r2'], 'beta': theta['beta']}

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
