import os, sys, re, time, multiprocessing
import helper, mystats 
import scipy
import numpy as np
import rpy2.robjects as robjects 
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from rpy2.robjects.conversion import localconverter
import util, wald, ctp

def get_X(fixed_covars_d, N, C):
    X = np.kron( np.ones((N,1)), np.eye(C) )
    if len(fixed_covars_d.keys()) != 0:
        for key in np.sort(list(fixed_covars_d.keys())):
            m_ = fixed_covars_d[key]
            if isinstance(fixed_covars_d[key], str):
                m_ = np.loadtxt(fixed_covars_d[key])
            if len( m_.shape ) == 1:
                m_ = m_.reshape(-1,1)
            X = np.concatenate( ( X, np.repeat(m_, C, axis=0)), axis=1 )
    return(X)

def cal_HE_base_vars(Y, vs):
    '''
    With 
    Y = np.loadtxt(y_f)
    vs = np.loadtxt(nu_f)
    '''

    # 
    N, C = Y.shape

    # vectorize Y
    y = Y.flatten()

    # projection matrix
    #proj = np.identity(N * C) - (1 / N) * np.kron(np.ones((N,N)), np.identity(C))

    # projected y
    y_p = y - (1 / N) * (np.transpose(Y) @ np.ones((N,N))).flatten('F')

    # calculate \ve{ \Pi_P^\perp ( I_N \otimes J_C ) \Pi_P^\perp }^T t, where t is vector of dependent variable
    # t = \ve{ y' y'^T - \Pi_P^\perp D \Pi_P^\perp }
    mt1 = (y_p.reshape(N, C).sum(axis=1)**2).sum() - vs.sum() * (N-1) / N

    return( {'y_p': y_p, 'mt1': mt1} )

def hom_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, par=None, nrep=10):
    '''
    par: array of initial parameter for optimization
    '''
    print('Hom ML', flush=True)
    start = time.time()

    def ml(Y, P, ctnu, fixed_covars, random_covars, par, nrep):
        ctng_ml_rf = 'bin/CTP/ml.R'
        ctng_ml_r = STAP( open(ctng_ml_rf).read(), 'ctng_ml_r' )
        par = robjects.NULL if par is None else robjects.FloatVector(par)
        numpy2ri.activate()
        out_ = ctng_ml_r.screml_hom(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), par=par, nrep=nrep)
        numpy2ri.deactivate()
        out = {}
        for key, value in zip(out_.names, list(out_)):
            out[key] = value
        return( out )

    # par 
    fixed_covars, random_covars, n_fixed, n_random = util.read_covars(fixed_covars_d, random_covars_d)[:4]

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = Y.shape
    X = get_X(fixed_covars, N, C)
    n_par = 1 + n_random + X.shape[1]
    
    if len( random_covars_d.keys() ) == 1:
        # order by random covar
        R = random_covars[ list(random_covars.keys())[0] ]
        _, R, tmp, fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        Y, P, vs = tmp
        random_covars[ list(random_covars.keys())[0] ] = R

    out = ml(Y, P, vs, fixed_covars, random_covars, par, nrep)
    
    hom2, beta, hess = out['hom2'][0], np.array(out['beta']), np.array(out['hess'])
    convergence, l = out['convergence'][0], out['l'][0]
    beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars )
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars )

    # wald
    A = np.ones((C,C))*hom2
    Vy = np.kron(np.eye(N), A) + np.diag(vs.flatten())
    Z = [np.kron(np.eye(N), np.ones((C,1)))]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.kron( random_covars[key], np.ones((C,1)) )
            Vy = Vy + r2_d[key] * M_ @ M_.T # only support iid variance
            Z.append( M_ )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)

    ml = {'hom2':hom2, 'beta':beta_d, 'fixedeffect_vars':fixedeffect_vars_d,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d, 
            'convergence':convergence, 'method':out['method'][0], 'l':l, 'D':D, 'hess':hess}
    if nu_f:
        ml['nu'] = np.mean(np.loadtxt(nu_f))

    wald_p = {}
    wald_p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    #wald_p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], two_sided=True) for i in range(X.shape[1])]
    # wald test beta1 = beta2 = beta3
    wald_p['ct_beta'] = util.wald_ct_beta(beta[:C], D[:C,:C], n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(ml, wald_p)

def hom_REML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, par=None, nrep=10, 
        jack_knife=False):
    print('Hom REML', flush=True)
    start = time.time()

    def reml(Y, P, ctnu, fixed_covars, random_covars, par, nrep):
        ctng_reml_rf = 'bin/CTP/reml.R'
        ctng_reml_r = STAP( open(ctng_reml_rf).read(), 'ctng_reml_r' )
        par = robjects.NULL if par is None else robjects.FloatVector(par)
        numpy2ri.activate()
        out_ = ctng_reml_r.screml_hom(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), par=par, nrep=nrep)
        numpy2ri.deactivate()
        out = {}
        for key, value in zip(out_.names, list(out_)):
            out[key] = value
        return( out )

    # par 
    fixed_covars, random_covars, n_fixed, n_random = util.read_covars(fixed_covars_d, random_covars_d)[:4]

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars_d, N, C)
    n_par = 1 + n_random + X.shape[1]

    if len( random_covars_d.keys() ) == 1:
        # order by random covar
        R = random_covars[ list(random_covars.keys())[0] ]
        _, R, tmp, fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        Y, P, vs = tmp
        random_covars[ list(random_covars.keys())[0] ] = R

    # 
    out = reml(Y, P, vs, fixed_covars, random_covars, par, nrep)

    beta, hom2, hess = np.array(out['beta']), out['hom2'][0], np.array(out['hess'])
    convergence, l = out['convergence'][0], out['l'][0]
    beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars )
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars )

    # wald
    A = np.ones((C,C)) * hom2
    Vy = np.kron(np.eye(N), A) + np.diag(vs.flatten())
    Z = [np.kron(np.eye(N), np.ones((C,1)))]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.kron( random_covars[key], np.ones((C,1)) )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'beta':beta_d, 'hom2':hom2, 'fixedeffect_vars':fixedeffect_vars_d,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'convergence':convergence, 'method':out['method'][0], 'l':l, 'D':D, 'hess':hess}
    if nu_f:
        reml['nu'] = np.mean( np.loadtxt(nu_f) )

    wald_p = {}
    if not jack_knife:
        wald_p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        # wald test beta1 = beta2 = beta3
        wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par)
    else:
        jacks = {'ct_beta':[], 'hom2':[]}
        for i in range(N):
            Y_tmp, vs_tmp, fixed_covars_tmp, random_covars_tmp, P_tmp = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            out_tmp = reml(Y_tmp, P_tmp, vs_tmp, fixed_covars_tmp, random_covars_tmp, par, nrep)

            beta_tmp, hom2_tmp = np.array(out_tmp['beta']), out_tmp['hom2'][0]
            ct_beta_tmp = util.fixedeffect_vars( beta_tmp, P_tmp, fixed_covars_tmp )[0]['ct_beta']
            jacks['ct_beta'].append( ct_beta_tmp )
            jacks['hom2'].append( hom2_tmp )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )
        
        wald_p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        # wald test beta1 = beta2 = beta3
        wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(reml, wald_p)

def iid_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, par=None, nrep=10):
    '''
    par: array of initial parameter for optimization
    '''
    print('IID ML', flush=True)
    start = time.time()

    def ml(Y, P, ctnu, fixed_covars, random_covars, par, nrep):
        ctng_ml_rf = 'bin/CTP/ml.R'
        ctng_ml_r = STAP( open(ctng_ml_rf).read(), 'ctng_ml_r' )
        par = robjects.NULL if par is None else robjects.FloatVector(par)
        numpy2ri.activate()
        out_ = ctng_ml_r.screml_iid(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), par=par, nrep=nrep)
        numpy2ri.deactivate()
        out = {}
        for key, value in zip(out_.names, list(out_)):
            out[key] = value
        return( out )

    # par 
    fixed_covars, random_covars, n_fixed, n_random = util.read_covars(fixed_covars_d, random_covars_d)[:4]

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = Y.shape
    X = get_X(fixed_covars, N, C)
    n_par = 1 + n_random + X.shape[1]
    
    if len( random_covars_d.keys() ) == 1:
        # order by random covar
        R = random_covars[ list(random_covars.keys())[0] ]
        _, R, tmp, fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        Y, P, vs = tmp
        random_covars[ list(random_covars.keys())[0] ] = R

    out = ml(Y, P, vs, fixed_covars, random_covars, par, nrep)
    
    hom2, V, beta, hess = out['hom2'][0], np.array(out['V']), np.array(out['beta']), np.array(out['hess'])
    convergence, l = out['convergence'][0], out['l'][0]
    beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars )
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars )

    # wald
    A = np.ones((C,C))*hom2 + V
    Vy = np.kron(np.eye(N), A) + np.diag(vs.flatten())
    Z = [np.repeat(np.eye(N), C, axis=0), np.eye(N*C)]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.repeat( random_covars[key], C, axis=0 )
            Vy = Vy + r2_d[key] * M_ @ M_.T # only support iid variance
            Z.append( M_ )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)

    ml = {'hom2':hom2, 'V':V, 'beta':beta_d, 'fixedeffect_vars':fixedeffect_vars_d,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d, 
            'convergence':convergence, 'method':out['method'][0], 'l':l, 'D':D, 'hess':hess}
    if nu_f:
        ml['nu'] = np.mean(np.loadtxt(nu_f))

    p = {}
    p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    p['V'] = wald.wald_test(V[0,0], 0, D[X.shape[1]+1,X.shape[1]+1], N-n_par)
    #wald_p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], two_sided=True) for i in range(X.shape[1])]
    # wald test beta1 = beta2 = beta3
    p['ct_beta'] = util.wald_ct_beta(beta[:C], D[:C,:C], n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(ml, p)

def iid_REML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, par=None, nrep=10, 
        jack_knife=False):
    print('IID REML', flush=True)
    start = time.time()

    def reml(Y, P, ctnu, fixed_covars, random_covars, par, nrep):
        ctng_reml_rf = 'bin/CTP/reml.R'
        ctng_reml_r = STAP( open(ctng_reml_rf).read(), 'ctng_reml_r' )
        par = robjects.NULL if par is None else robjects.FloatVector(par)
        numpy2ri.activate()
        out_ = ctng_reml_r.screml_iid(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), par=par, nrep=nrep)
        numpy2ri.deactivate()
        out = {}
        for key, value in zip(out_.names, list(out_)):
            out[key] = value
        return( out )

    # par 
    fixed_covars, random_covars, n_fixed, n_random = util.read_covars(fixed_covars_d, random_covars_d)[:4]

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars, N, C)
    n_par = 1 + n_random + X.shape[1]

    if len( random_covars_d.keys() ) == 1:
        # order by random covar
        R = random_covars[ list(random_covars.keys())[0] ]
        _, R, tmp, fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        Y, P, vs = tmp
        random_covars[ list(random_covars.keys())[0] ] = R

    # 
    out = reml(Y, P, vs, fixed_covars, random_covars, par, nrep)

    beta, hom2, V, hess = np.array(out['beta']), out['hom2'][0], np.array(out['V']), np.array(out['hess'])
    convergence, l = out['convergence'][0], out['l'][0]
    beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars )
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars )

    # wald
    A = np.ones((C,C)) * hom2 + V
    Vy = np.kron(np.eye(N), A) + np.diag(vs.flatten())
    Z = [np.repeat(np.eye(N), C, axis=0), np.eye(N*C)]
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.repeat( random_covars[key], C, axis=0 )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'beta':beta_d, 'hom2':hom2, 'fixedeffect_vars':fixedeffect_vars_d,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'convergence':convergence, 'method':out['method'][0], 'l':l, 'D':D, 'hess':hess}
    if nu_f:
        reml['nu'] = np.mean( np.loadtxt(nu_f) )

    wald_p = {}
    if not jack_knife:
        wald_p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        wald_p['V'] = wald.wald_test(V[0,0], 0, D[1,1], N-n_par)
        # wald test beta1 = beta2 = beta3
        wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par)
    else:
        jacks = {'ct_beta':[], 'hom2':[], 'het':[]}
        for i in range(N):
            Y_tmp, vs_tmp, fixed_covars_tmp, random_covars_tmp, P_tmp = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            out_tmp = reml(Y_tmp, P_tmp, vs_tmp, fixed_covars_tmp, random_covars_tmp, par, nrep)

            beta_tmp, hom2_tmp, V_tmp = np.array(out_tmp['beta']), out_tmp['hom2'][0], np.array(out_tmp['V'])
            ct_beta_tmp = util.fixedeffect_vars( beta_tmp, P_tmp, fixed_covars_tmp )[0]['ct_beta']
            jacks['ct_beta'].append( ct_beta_tmp )
            jacks['hom2'].append( hom2_tmp )
            jacks['het'].append( V_tmp[0,0] )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_het = (len(jacks['het']) - 1.0) * np.var(jacks['het'])
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )
        
        wald_p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        wald_p['V'] = wald.wald_test(V[0,0], 0, var_het, N-n_par)
        # wald test beta1 = beta2 = beta3
        wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(reml, wald_p)

def free_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, par=None, nrep=10):
    print('Free ML', flush=True)
    start = time.time()

    def ml(Y, P, ctnu, fixed_covars, random_covars, par, nrep):
        ctng_ml_rf = 'bin/CTP/ml.R'
        ctng_ml_r = STAP( open(ctng_ml_rf).read(), 'ctng_ml_r' )
        par = robjects.NULL if par is None else robjects.FloatVector(par)
        numpy2ri.activate()
        out_ = ctng_ml_r.screml_free(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), par=par, nrep=nrep )
        numpy2ri.deactivate()
        out = {}
        for key, value in zip(out_.names, list(out_)):
            out[key] = value
        return( out )

    # par 
    fixed_covars, random_covars, n_fixed, n_random = util.read_covars(fixed_covars_d, random_covars_d)[:4]

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars, N, C)
    n_par = 1 + C + n_random + X.shape[1]

    if len( random_covars.keys() ) == 1:
        # order by random covar
        R = random_covars[ list(random_covars.keys())[0] ]
        _, R, tmp, fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        Y, P, vs = tmp
        random_covars[ list(random_covars.keys())[0] ] = R

    out = ml(Y, P, vs, fixed_covars, random_covars, par, nrep)

    hom2, beta, V, hess = out['hom2'][0], np.array(out['beta']), np.array(out['V']), np.array(out['hess'])
    convergence, l = out['convergence'][0], out['l'][0]
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars )
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars)

    # wald
    A = np.ones((C,C))*hom2+V
    Vy = np.kron(np.eye(N), A) + np.diag(vs.flatten())
    Z = [np.kron(np.eye(N), np.ones((C,1)))]
    for i in range(C):
        m = np.zeros(C)
        m[i] = 1
        Z.append(np.kron(np.identity(N), m.reshape(-1,1)))
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.kron( random_covars[key], np.ones((C,1)) )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)

    ml = {'hom2':hom2, 'beta':beta_d, 'V':V, 'fixedeffect_vars':fixedeffect_vars_d,
            'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'convergence':convergence, 'method':out['method'][0], 'l':l, 'D':D, 'hess':hess}
    if nu_f:
        ml['nu'] = np.mean(np.loadtxt(nu_f))

    wald_p = {}
    wald_p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    var_V = D[(X.shape[1]+1):(X.shape[1]+C+1), (X.shape[1]+1):(X.shape[1]+C+1)]
    wald_p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
    wald_p['Vi'] = [wald.wald_test(V[i,i], 0, D[X.shape[1]+i+1,X.shape[1]+i+1], N-n_par) for i in range(C)]
    #wald_p['beta'] = [wald.wald_test(beta[i], 0, D[i,i], two_sided=True) for i in range(X.shape[1])]
    wald_p['ct_beta'] = util.wald_ct_beta(beta[:C], D[:C,:C], n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(ml, wald_p)

def free_REML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, par=None, nrep=10,
        jack_knife=False):
    print('Free REML', flush=True)
    start = time.time()

    def reml(Y, P, ctnu, fixed_covars, random_covars, par, nrep):
        ctng_reml_rf = 'bin/CTP/reml.R'
        ctng_reml_r = STAP( open(ctng_reml_rf).read(), 'ctng_reml_r' )
        par = robjects.NULL if par is None else robjects.FloatVector(par)
        numpy2ri.activate()
        out_ = ctng_reml_r.screml_free(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), par=par, nrep=nrep)
        numpy2ri.deactivate()
        out = {}
        for key, value in zip(out_.names, list(out_)):
            out[key] = value
        return out

    # par 
    fixed_covars, random_covars, n_fixed, n_random = util.read_covars(fixed_covars_d, random_covars_d)[:4]

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars, N, C)
    n_par = 1 + C + n_random + X.shape[1]

    if len( random_covars_d.keys() ) == 1:
        # order by random covar
        R = random_covars[ list(random_covars.keys())[0] ]
        _, R, tmp, fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        Y, P, vs = tmp
        random_covars[ list(random_covars.keys())[0] ] = R

    out = reml(Y, P, vs, fixed_covars, random_covars, par, nrep)

    hom2, beta, V, hess = out['hom2'][0], np.array(out['beta']), np.array(out['V']), np.array(out['hess'])
    convergence, l = out['convergence'][0], out['l'][0]
    beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars )
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars)

    # wald
    A = np.ones((C,C))*hom2+V
    Vy = np.kron(np.eye(N), A) + np.diag(vs.flatten())
    X = get_X(fixed_covars, N, C)
    Z = [np.kron(np.eye(N), np.ones((C,1)))]
    for i in range(C):
        m = np.zeros(C)
        m[i] = 1
        Z.append(np.kron(np.identity(N), m.reshape(-1,1)))
    if len(r2_d.keys()) > 0:
        for key in np.sort( list(r2_d.keys()) ):
            M_ = np.kron( random_covars[key], np.ones((C,1)) )
            Vy = Vy + r2_d[key] * M_ @ M_.T
            Z.append( M_ )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)
    reml = {'beta':beta_d, 'hom2':hom2, 'V':V, 'fixedeffect_vars':fixedeffect_vars_d,
            'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'convergence':convergence, 'method':out['method'][0], 'l':l, 'D':D, 'hess':hess}
    if nu_f:
        reml['nu'] = np.mean( np.loadtxt(nu_f) )

    wald_p = {}
    if not jack_knife:
        wald_p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        var_V = D[1:(C+1), 1:(C+1)]
        wald_p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
        wald_p['Vi'] = [wald.wald_test(V[i,i], 0, D[i+1,i+1], N-n_par) for i in range(C)]
        # wald test beta1 = beta2 = beta3
        wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par)
    else:
        jacks = { 'ct_beta':[], 'hom2':[], 'V':[] }
        for i in range(N):
            Y_tmp, vs_tmp, fixed_covars_tmp, random_covars_tmp, P_tmp = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            out_tmp = reml(Y_tmp, P_tmp, vs_tmp, fixed_covars_tmp, 
                    random_covars_tmp, par, nrep)
            beta_tmp, hom2_tmp, V_tmp = np.array(out_tmp['beta']), out_tmp['hom2'][0], np.array(out_tmp['V'])
            ct_beta_tmp = util.fixedeffect_vars( beta_tmp, P_tmp, fixed_covars_tmp )[0]['ct_beta']
            jacks['ct_beta'].append( ct_beta_tmp )
            jacks['hom2'].append( hom2_tmp )
            jacks['V'].append( np.diag(V_tmp) )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_V = (len(jacks['V']) - 1.0) * np.cov( np.array(jacks['V']).T, bias=True )
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )

        wald_p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        wald_p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
        # wald test beta1 = beta2 = beta3
        wald_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(reml, wald_p)

def full_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, par=None, nrep=10):
    print('Full ML', flush=True)
    start = time.time()

    def ml(Y, P, ctnu, fixed_covars, random_covars, par, nrep):
        ctng_ml_rf = 'bin/CTP/ml.R'
        ctng_ml_r = STAP( open(ctng_ml_rf).read(), 'ctng_ml_r' )
        par = robjects.NULL if par is None else robjects.FloatVector(par)
        numpy2ri.activate()
        out_ = ctng_ml_r.screml_full(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), par=par, nrep=nrep)
        numpy2ri.deactivate()
        out = {}
        for key, value in zip(out_.names, list(out_)):
            out[key] = value
        return( out )

    # par 
    fixed_covars, random_covars, n_fixed, n_random = util.read_covars(fixed_covars_d, random_covars_d)[:4]

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape

    if len( random_covars_d.keys() ) == 1:
        # order by random covar
        R = random_covars[ list(random_covars.keys())[0] ]
        _, R, tmp, fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        Y, P, vs = tmp
        random_covars[ list(random_covars.keys())[0] ] = R

    out = ml(Y, P, vs, fixed_covars, random_covars, par, nrep)

    beta, V, hess = np.array(out['beta']), np.array(out['V']), np.array(out['hess'])
    convergence, l = out['convergence'][0], out['l'][0]
    beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars )
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars)

    ml = {'beta':beta_d, 'V':V, 'fixedeffect_vars':fixedeffect_vars_d,
            'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'convergence':convergence, 'method':out['method'][0], 'l':l, 'hess':hess}
    if nu_f:
        ml['nu'] = np.mean(np.loadtxt(nu_f))

    print( time.time() - start, flush=True )
    return(ml)

def full_REML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, par=None, nrep=10):
    print('Full REML', flush=True)
    start = time.time()

    def reml(Y, P, ctnu, fixed_covars, random_covars, par, nrep):
        ctng_reml_rf = 'bin/CTP/reml.R'
        ctng_reml_r = STAP( open(ctng_reml_rf).read(), 'ctng_reml_r' )
        par = robjects.NULL if par is None else robjects.FloatVector(par)
        numpy2ri.activate()
        out_ = ctng_reml_r.screml_full(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), par=par, nrep=nrep)
        numpy2ri.deactivate()
        out = {}
        for key, value in zip(out_.names, list(out_)):
            out[key] = value
        return( out )
    
    # par 
    fixed_covars, random_covars, n_fixed, n_random = util.read_covars(fixed_covars_d, random_covars_d)[:4]

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape

    if len( random_covars_d.keys() ) == 1:
        # order by random covar
        R = random_covars[ list(random_covars.keys())[0] ]
        _, R, tmp, fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        Y, P, vs = tmp
        random_covars[ list(random_covars.keys())[0] ] = R

    out = reml(Y, P, vs, fixed_covars, random_covars, par, nrep)

    beta, V, hess = np.array(out['beta']), np.array(out['V']), np.array(out['hess'])
    convergence, l = out['convergence'][0], out['l'][0]
    beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars )
    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )
    randomeffect_vars_d, r2_d = util.assign_randomeffect_vars( np.array(out['randomeffect_vars']),
            np.array(out['r2']), random_covars)
    
    reml = {'beta':beta_d, 'V':V, 'fixedeffect_vars':fixedeffect_vars_d,
            'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'convergence':convergence, 'method':out['method'][0], 'l':l, 'hess':hess}
    if nu_f:
        reml['nu'] = np.mean( np.loadtxt(nu_f) )
    
    print( time.time() - start, flush=True )
    return(reml)

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

        C = np.loadtxt(y_f).shape[1]

        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        out = {}
        ## HE
        if 'HE_as_initial' not in snakemake.params.keys():
            snakemake.params.HE_as_initial = False
        if snakemake.params.HE_as_initial:
            snakemake.params.HE = True

        if snakemake.params.HE:
            if 'n_equation' not in snakemake.wildcards.keys():
                # number of equations to decide degree of freedom in Wald test
                n_equation = None
            else:
                n_equation = snakemake.wildcards.n_equation
            free_he, free_he_wald = ctp.free_HE(y_f, P_f, nu_f, jack_knife=True, n_equation=n_equation)
            HE_free_only = False
            if 'HE_free_only' in snakemake.params.keys():
                HE_free_only = snakemake.params.HE_free_only
            if HE_free_only:
                out['he'] = { 'free': free_he, 'wald':{'free': free_he_wald} }
            else:
                hom_he, hom_he_wald = ctp.hom_HE(y_f, P_f, nu_f, jack_knife=True)
                full_he, full_he_wald = ctp.full_HE(y_f, nu_f)

                out['he'] = {'hom': hom_he, 'free': free_he, 'full': full_he,
                        'wald':{'hom':hom_he_wald, 'free': free_he_wald, 'full':full_he_wald} }

        ## ML
        if snakemake.params.ML:
            if not snakemake.params.HE_as_initial:
                hom_ml, hom_ml_wald = hom_ML(y_f, P_f, nu_f)
                free_ml, free_ml_wald = free_ML(y_f, P_f, nu_f)
                full_ml = full_ML(y_f, P_f, nu_f)
            else:
                hom_ml, hom_ml_wald = hom_ML(y_f, P_f, nu_f, par=util.generate_HE_initial(hom_he, ML=True) )
                free_ml, free_ml_wald = free_ML(y_f, P_f, nu_f, par=util.generate_HE_initial(free_he, ML=True) )
                full_ml = full_ML(y_f, P_f, nu_f, par=util.generate_HE_initial(full_he, ML=True) )

            out['ml'] = {'hom': hom_ml, 'free': free_ml, 'full': full_ml,
                    'wald':{'hom':hom_ml_wald, 'free':free_ml_wald} }

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
                if snakemake.params.Free_reml_only:
                    if 'Free_reml_jk' in snakemake.params.keys():
                        free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f,  
                                jack_knife=snakemake.params.Free_reml_jk)
                    else:
                        free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f)
                else:
                    hom_reml, hom_reml_wald = hom_REML(y_f, P_f, nu_f)
                    if 'Free_reml_jk' in snakemake.params.keys():
                        free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f,  
                                jack_knife=snakemake.params.Free_reml_jk)
                    else:
                        free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f)
                    full_reml = full_REML(y_f, P_f, nu_f)
            else:
                hom_reml, hom_reml_wald = hom_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(hom_he, REML=True))
                free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(free_he,REML=True))
                full_reml = full_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(full_he, REML=True))

            if snakemake.params.Free_reml_only:
                out['reml'] = {'free':free_reml, 'wald':{'free':free_reml_wald} }
            else:
                out['reml'] = {'hom':hom_reml, 'free':free_reml, 'full':full_reml,
                        'wald':{'hom':hom_reml_wald, 'free':free_reml_wald} }

                ## REML
                free_hom_lrt = mystats.lrt(out['reml']['free']['l'], 
                                            out['reml']['hom']['l'], C)
                full_hom_lrt = mystats.lrt(out['reml']['full']['l'], 
                                            out['reml']['hom']['l'], C*(C+1)//2-1)
                full_free_lrt = mystats.lrt(out['reml']['full']['l'], 
                                            out['reml']['free']['l'], C*(C+1)//2-C-1)

                out['reml']['lrt'] = {
                                        'free_hom':free_hom_lrt,
                                        'full_hom':full_hom_lrt, 
                                        'full_free':full_free_lrt
                                        }

        # save
        np.save(out_f, out)

    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))

if __name__ == '__main__':

    # load R functions
    ctng_ml_rf = 'bin/CTP/ml.R'
    ctng_reml_rf = 'bin/CTP/reml.R'

    ctng_ml_r = STAP( open(ctng_ml_rf).read(), 'ctng_ml_r' )
    ctng_reml_r = STAP( open(ctng_reml_rf).read(), 'ctng_reml_r' )

    main()

