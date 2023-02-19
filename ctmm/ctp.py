import os, sys, re, time
import numpy as np
from scipy import linalg, optimize, stats
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import STAP
from . import util, wald

def get_X(fixed_covars_d, N, C):
    X = np.kron( np.ones((N,1)), np.eye(C) )
    fixed_covars = util.read_covars(fixed_covars_d)[0]
    for key in np.sort(list(fixed_covars.keys())):
        m = fixed_covars[key]
        if len( m.shape ) == 1:
            m = m.reshape(-1,1)
        X = np.concatenate( ( X, np.repeat(m, C, axis=0)), axis=1 )
    return(X)

def get_MMT(random_covars, C):
    random_MMT = []
    for key in np.sort( list(random_covars.keys()) ):
        R = random_covars[key]
        m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
        random_MMT.append( m )
    return( random_MMT )

def cal_Vy(A, vs, r2=[], random_MMT=[]):
    if isinstance(r2, dict):
        r2 = list(r2.values())
    N, C = vs.shape
    Vy = np.kron(np.eye(N), A) + np.diag( vs.flatten() )
    for var, MMT in zip(r2, random_MMT):
        Vy += var * MMT
    return( Vy )

#def HE_stats(Y, vs, fixed_covars=[], random_covars=[], reorder_random=True):
#    # 
#    N, C = Y.shape
#    X = get_X(fixed_covars, N, C)
#
#    #
#    R = None
#    if len( random_covars.keys() ) == 1:
#        # order by random covar
#        R = list( random_covars.values() )[0]
#        if reorder_random:
#            _, R, [Y, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, vs], fixed_covars)
#            random_covars[ list(random_covars.keys())[0] ] = R
#            X = get_X(fixed_covars, N, C)
#    else:
#        R = []
#        for key in np.sort( list(random_covars.keys()) ):
#            R.append( random_covars[key] )
#        R = np.concatenate( R, axis=1 )
#
#
#    # projection matrix
#    proj = np.eye(N * C) - X @ np.linalg.inv(X.T @ X) @ X.T
#    #proj = np.identity(N * C) - (1 / N) * np.kron(np.ones((N,N)), np.identity(C))
#
#    # vectorize Y
#    y = Y.flatten()
#
#    # projected y
#    y_p = proj @ y
#    stats = {'R':R, 'X':X, 'Y':Y, 'vs':vs, 'proj':proj, 'y_p':y_p}
#    #y_p = y - (1 / N) * (np.transpose(Y) @ np.ones((N,N))).flatten('F')
#
#    # calculate \ve{ \Pi_P^\perp ( I_N \otimes J_C ) \Pi_P^\perp }^T t, where t is vector of dependent variable
#    # t = \ve{ y' y'^T - \Pi_P^\perp D \Pi_P^\perp }
#    #mt1 = (y_p.reshape(N, C).sum(axis=1)**2).sum() - vs.sum() * (N-1) / N
#
#    return( stats )
#
def he_ols(Y, X, vs, random_covars, model):
    '''
    Q: design matrix in HE OLS
    '''
    
    N, C = Y.shape
    y = Y.flatten()
    proj = np.eye(N * C) - X @ np.linalg.inv(X.T @ X) @ X.T

    # vec(M @ A @ M)^T @ vec(M @ B M) = vec(M @ A)^T @ vec((M @ B)^T), when A, B, and M are symmetric #
    t = np.outer( proj @ y, y ) - proj * vs.flatten()  # proj @ y @ y^T @ proj - proj @ D @ proj

    if model == 'hom':
        Q = [ np.hstack( np.hsplit(proj, N) @ np.ones((C,C)) ) ] # M (I_N \otimes J_C)
    elif model == 'iid':
        Q = [ np.hstack( np.hsplit(proj, N) @ np.ones((C,C)) ) ] # M (I_N \otimes J_C)
        Q.append( proj ) 
    elif model == 'free':
        Q = [ np.hstack( np.hsplit(proj, N) @ np.ones((C,C)) ) ] # M (I_N \otimes J_C)
        for i in range(C):
            L = np.zeros((C,C))
            L[i,i] = 1
            Q.append( np.hstack( np.hsplit(proj, N) @ L ) )
    elif model == 'full':
        Q = []
        for i in range(C):
            L = np.zeros((C,C))
            L[i,i] = 1
            Q.append( np.hstack( np.hsplit(proj, N) @ L ) )
        for i in range(1,C):
            for j in range(i):
                L = np.zeros((C,C))
                L[i,j] = 1
                Q.append( np.hstack( np.hsplit(proj, N) @ (L+L.T) ) )

    random_MMT = []
    for key in np.sort( list(random_covars.keys()) ):
        R = random_covars[key]
        m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
        print( proj )
        random_MMT.append( m )
        Q.append( proj @ m )

    QTQ = np.array([m.flatten('F') for m in Q]) @ np.array([m.flatten() for m in Q]).T
    QTQt = np.array([m.flatten('F') for m in Q]) @ t.flatten()
    theta = np.linalg.inv(QTQ) @ QTQt

    #if return_QTQQT:
        #return( theta, sigma_y2, random_vars, QTQ, QTQQT )
        #return( theta, random_MMT, QTQQT )
    #else:
        #return( theta, sigma_y2, random_vars, QTQ )
    return( theta, random_MMT )

def ML_LL(Y, X, N, C, vs, hom2, beta, V, r2=[], random_MMT=[]):
    y = Y.flatten()
    A = np.ones((C,C)) * hom2 + V

    if len(random_MMT) == 0:
        yd = y - X @ beta
        Yd = yd.reshape( (N,C) )

        l = 0
        for i in range(N):
            D_i = np.diag( vs[i,] )
            AD = A + D_i

            w, v = linalg.eigh( AD )
            if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
                return( 1e12 )

            det_AD = np.sum( np.log(w) )
            inv_AD = v @ np.diag(1/w) @ v.T
            l = l + det_AD + Yd[i,] @ inv_AD @ Yd[i,]

        l = 0.5 * l
    elif len(random_MMT) == 1:
        # assmue M is sorted with structure 1_a, 1_b, 1_c, so is MMT
        yd = y - X @ beta
        Yd = yd.reshape( (N,C) )

        Vy = cal_Vy(A, vs, r2, random_MMT)

        l = 0 
        i = 0
        while i < (N*C):
            j = i + np.sum( random_MMT[0][i,]==1 )
            Vy_k = Vy[i:j,i:j]

            w, v = linalg.eigh(Vy_k)
            if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
                return( 1e12 )

            det_Vy_k = np.sum( np.log(w) )
            inv_Vy_k = v @ np.diag(1/w) @ v.T
            l = l + det_Vy_k + yd[i:j] @ inv_Vy_k @ yd[i:j]

            i = j

        l = l * 0.5
    else:
        Vy = cal_Vy(A, vs, r2, random_MMT)
        l = stats.multivariate_normal.logpdf(y, mean=X @ beta, cov=Vy) * (-1)

    return(l)

def REML_LL(Y, X, N, C, vs, hom2, V, r2=[], random_MMT=[]):
    A = np.ones((C,C)) * hom2 + V

    if X.shape[1] == C and len(random_MMT) == 0:
        AD_inv_sum = np.zeros((C,C))
        AD_det_sum = 0
        ADy_sum = np.zeros(C)
        yADy_sum = 0

        for i in range(N):
            AD = A + np.diag(vs[i,])

            w, v = linalg.eigh(AD)
            if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
                return( 1e12 )

            AD_inv = v @ np.diag(1/w) @ v.T
            AD_det = np.sum( np.log(w) )
            ADy = AD_inv @ Y[i,]
            yADy = Y[i,] @ ADy

            AD_inv_sum += AD_inv
            AD_det_sum += AD_det
            ADy_sum += ADy
            yADy_sum += yADy

        w, v = linalg.eigh( AD_inv_sum )
        if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
            return( 1e12 )
        AD_inv_sum_det = np.sum( np.log(w) )
        AD_inv_sum_inv = v @ np.diag(1/w) @ v.T

        #
        L = AD_det_sum + AD_inv_sum_det + yADy_sum - ADy_sum @ AD_inv_sum_inv @ ADy_sum
        L = 0.5 * L
    elif X.shape[1] != C and len(random_MMT) == 0:
        AD_det_sum = 0
        yADy_sum = 0
        XADy_sum = 0
        XADX_sum = np.zeros((X.shape[1], X.shape[1]))

        for i in range(N):
            AD = A + np.diag( vs[i,] )

            w, v = linalg.eigh( AD )
            if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
                return( 1e12 )
            AD_inv = v @ np.diag(1/w) @ v.T
            AD_det =  np.sum( np.log(w) )

            Xi = X[(i * C): ((i+1) * C),]
            XADX = Xi.T @ AD_inv @ Xi
            yADy = Y[i,] @ AD_inv @ Y[i,]
            XADy = Xi.T @ AD_inv @ Y[i,]

            XADX_sum += XADX
            AD_det_sum += AD_det
            yADy_sum += yADy
            XADy_sum += XADy

        w, v = linalg.eigh( XADX_sum )
        if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
            return( 1e12 )
        XADX_sum_inv = v @ np.diag(1/w) @ v.T
        XADX_sum_det = np.sum( np.log(w) )

        L = AD_det_sum + XADX_sum_det + yADy_sum - XADy_sum @ XADX_sum_inv @ XADy_sum
        L = L * 0.5
    elif len(random_MMT) > 0:
        Vy = cal_Vy( A, vs, r2, random_MMT )

        if len(random_MMT) == 1:
            MMT = random_MMT[0]
            Vy_inv = []
            Vy_det = 0
            i = 0
            while i < (N*C):
                j = i + (MMT[i,]==1).sum()
                Vy_k = Vy[i:j, i:j]

                w, v = linalg.eigh( Vy_k )
                if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
                    return( 1e12 )
                Vy_inv.append( v @ np.diag(1/w) @ v.T )
                Vy_det += np.sum( np.log(w) )

                i = j

            Vy_inv = linalg.block_diag( *Vy_inv )
        else:
            w, v = linalg.eigh( Vy )
            if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
                return( 1e12 )
            Vy_det = np.sum( np.log(w) )
            Vy_inv = v @ np.diag(1/w) @ v.T

        F = X.T @ Vy_inv
        B = F @ X
        w, v = linalg.eigh( B )
        if np.any(w < 0) or ( (np.amax(w) / (np.amin(w)+1e-99)) > 1e6 ):
            return( 1e12 )
        B_inv = v @ np.diag(1/w) @ v.T
        B_det = np.sum( np.log(w) )
        M = Vy_inv - F.T @ B_inv @ F
        L = Vy_det + B_det + Y.flatten() @ M @ Y.flatten()
        L = 0.5 * L
        
    return( L )

def r_optim(Y, P, ctnu, fixed_covars, random_covars, par, nrep, ml, model, method):
    rf = 'bin/CTP/ml.R' if ml.upper() == 'ML' else 'bin/CTP/reml.R'
    r_ml = STAP( open(rf).read(), 'r_ml' )
    par = robjects.NULL if par is None else robjects.FloatVector(par)
    method = 'BFGS' if method is None else method
    numpy2ri.activate()
    if model == 'hom':
        out_ = r_ml.screml_hom(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), 
                par=par, nrep=nrep, method=method)
    elif model == 'iid':
        out_ = r_ml.screml_iid(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), 
                par=par, nrep=nrep, method=method)
    elif model == 'free':
        out_ = r_ml.screml_free(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), 
                par=par, nrep=nrep, method=method)
    elif model == 'full':
        out_ = r_ml.screml_full(Y=r['as.matrix'](Y), P=r['as.matrix'](P),
                vs=r['as.matrix'](ctnu), fixed=util.dict2Rlist(fixed_covars),
                random=util.dict2Rlist(random_covars), 
                par=par, nrep=nrep, method=method)
    numpy2ri.deactivate()
    out = {}
    for key, value in zip(out_.names, list(out_)):
        out[key] = value
    return( out )

def hom_ML_loglike(par, Y, X, N, C, vs, random_MMT):
    hom2 = par[0]
    beta = par[1:(1+X.shape[1])]
    V = np.zeros((C,C))
    r2 = par[(1+X.shape[1]):]

    return( ML_LL(Y, X, N, C, vs, hom2, beta, V, r2, random_MMT) )

def hom_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10, optim_by_r=False):
    '''
    par: np.array of initial parameter for optimization
    '''
    print('Hom ML', flush=True)
    start = time.time()

    def extract(out, C, X, P, fixed_covars, random_covars):
        hom2, beta, r2 = out['x'][0], out['x'][1:(1+X.shape[1])], out['x'][(1+X.shape[1]):]
        l = out['fun'] * (-1)
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        return( hom2, beta, r2, l, fixed_vars, random_vars )

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = Y.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars, N, C)
    n_par = 1 + n_random + X.shape[1]

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        random_MMT = [np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1)]
        X = get_X(fixed_covars, N, C)

    # optim
    if optim_by_r:
        out = r_optim(Y, P, vs, fixed_covars, random_covars, par, nrep, 'ml', 'hom', method)

        hom2, beta = out['hom2'][0], np.array(out['beta'])
        l = out['l'][0]
        opt = {'convergence':out['convergence'][0], 'method':out['method'][0]}
        random_vars, r2 = np.array(out['randomeffect_vars']), np.array(out['r2'])
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
    else:
        if par is None:
            beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
            hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 1 )
            par = [hom2] + list(beta) + [hom2] * n_random

        out, opt = util.optim(hom_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)

        hom2, beta, r2, l, fixed_vars, random_vars = extract( out, C, X, P, fixed_covars, random_covars )

        if util.check_optim(opt, hom2, 0, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, hom_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT),
                    method=method, nrep=nrep)
            hom2, beta, r2, l, fixed_vars, random_vars = extract( out, C, X, P, fixed_covars, random_covars )
        
    # wald
    A = np.ones((C,C)) * hom2
    Vy = cal_Vy( A, vs, r2, random_MMT )
    Z = [np.repeat(np.eye(N), C, axis=0)]
    for key in random_covars.keys():
        m = np.repeat( random_covars[key], C, axis=0 )
        Z.append( m )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)

    ml = {'hom2':hom2, 'beta':beta, 'fixedeffect_vars':fixed_vars,
            'randomeffect_vars':random_vars, 'r2':r2, 'l':l, 'D':D, 'opt':opt}
    if nu_f:
        ml['nu'] = np.mean(np.loadtxt(nu_f))

    p = {}
    p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    # wald test beta1 = beta2 = beta3
    p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], D[:C,:C], n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(ml, p)

def hom_REML_loglike(par, Y, X, N, C, vs, random_MMT):
    hom2 = par[0]
    V = np.zeros((C,C))
    r2 = par[1:]

    return( REML_LL(Y, X, N, C, vs, hom2, V, r2, random_MMT) )

def hom_REML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10, jack_knife=False, optim_by_r=False):
    print('Hom REML', flush=True)
    start = time.time()
    
    def extract(out, Y, X, P, vs, fixed_covars, random_covars, random_MMT):
        N, C = P.shape
        hom2, r2 = out['x'][0], out['x'][1:]
        l = out['fun'] * (-1)
        A = np.ones((C,C)) * hom2
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, Y.flatten() )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        return(hom2, r2, beta, l, fixed_vars, random_vars, Vy)

    def reml_f(Y, vs, P, fixed_covars, random_covars, method):
        ''' wrapper for hom reml '''
        N, C = Y.shape
        X = get_X(fixed_covars, N, C)

        random_MMT = get_MMT(random_covars, C)
           
        out, opt = util.optim(hom_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
       
        hom2, r2, beta, l, fixed_vars, random_vars, Vy = extract(
                out, Y, X, P, vs, fixed_covars, random_covars, random_MMT)

        if util.check_optim(opt, hom2, 0, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, hom_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT),
                    method=method, nrep=nrep)
            hom2, r2, beta, l, fixed_vars, random_vars, Vy = extract(
                    out, Y, X, P, vs, fixed_covars, random_covars, random_MMT)

        return(hom2, r2, beta, l, fixed_vars, random_vars, Vy, opt)

    def extract_R(out):
        hom2, beta = out['hom2'][0], np.array(out['beta'])
        l = out['l'][0]
        opt = {'convergence':out['convergence'][0], 'method':out['method'][0]}
        random_vars, r2 = np.array(out['randomeffect_vars']), np.array(out['r2'])
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        return( hom2, beta, r2, fixed_vars, random_vars, l, opt)

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars, N, C)
    n_par = 1 + n_random + X.shape[1]

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        random_MMT = [np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1)]
        X = get_X(fixed_covars, N, C)

    # optim
    if optim_by_r:
        out = r_optim(Y, P, vs, fixed_covars, random_covars, par, nrep, 'reml', 'hom', method)
        hom2, beta, r2, fixed_vars, random_vars, l, opt = extract_R( out )

        A = np.ones((C,C)) * hom2
        Vy = cal_Vy( A, vs, r2, random_MMT )
    else:
        if par is None:
            beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
            hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 1 )
            par = [hom2] * (n_random + 1)

        #
        hom2, r2, beta, l, fixed_vars, random_vars, Vy, opt = reml_f(
                Y, vs, P, fixed_covars, random_covars, method)

    # wald
    Z = [np.repeat(np.eye(N), C, axis=0)]
    for key in random_covars.keys():
        m = np.repeat( random_covars[key], C, axis=0 )
        Z.append( m )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'beta':beta, 'hom2':hom2, 'fixedeffect_vars':fixed_vars,
            'randomeffect_vars':random_vars, 'r2':r2, 'l':l, 'D':D, 'opt':opt}
    if nu_f:
        reml['nu'] = np.mean( np.loadtxt(nu_f) )

    p = {}
    if not jack_knife:
        p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], 
                np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par)
    else:
        jacks = {'ct_beta':[], 'hom2':[]}
        for i in range(N):
            Y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            if optim_by_r:
                out_jk = r_optim(Y_jk, P_jk, vs_jk, fixed_covars_jk, random_covars_jk, par, nrep, 
                        'reml', 'hom', method)
                hom2_jk, beta_jk = extract_R( out_jk )[:2]
            else:
                hom2_jk, _, beta_jk, _, _, _, _ = reml_f(
                        Y_jk, vs_jk, P_jk, fixed_covars_jk, random_covars_jk, method)

            jacks['hom2'].append( hom2_jk )
            jacks['ct_beta'].append( beta_jk['ct_beta'] )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )
        
        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(reml, p)

def hom_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, jack_knife=True):
    print('Hom HE', flush=True )
    start = time.time()

    def he_f(Y, vs, P, fixed_covars, random_covars):
        N, C = Y.shape
        y = Y.flatten()
        X = get_X(fixed_covars, N, C)

        theta, random_MMT = he_ols(Y, X, vs, random_covars, model='hom')
        hom2, r2 = theta[0], theta[1:]

        # beta
        A = np.ones((C,C)) * hom2
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

        return(hom2, r2, beta, fixed_vars, random_vars)

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars, N, C)
    n_par = 1 + n_random + X.shape[1]

    #
    hom2, r2, beta, fixed_vars, random_vars = he_f(Y, vs, P, fixed_covars, random_covars)

    he = {'hom2': hom2, 'beta':beta, 'fixedeffect_vars':fixed_vars,
            'randomeffect_vars':random_vars, 'r2':r2 }
    if nu_f:
        he['nu'] = np.mean( np.loadtxt(nu_f) )

    # p values
    p = {}
    if jack_knife:
        jacks = {'ct_beta':[], 'hom2':[]}
        for i in range(N):
            Y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            hom2_jk, _, beta_jk, _, _ = he_f(Y_jk, vs_jk, P_jk, fixed_covars_jk, random_covars_jk)

            jacks['hom2'].append( hom2_jk )
            jacks['ct_beta'].append( beta_jk['ct_beta'] )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )
        
        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par)


    print( time.time() - start )
    return(he, p)

def iid_ML_loglike(par, Y, X, N, C, vs, random_MMT):
    hom2 = par[0]
    V = np.eye(C) * par[1]
    beta = par[2:(2+X.shape[1])]
    r2 = par[(2+X.shape[1]):]

    return( ML_LL(Y, X, N, C, vs, hom2, beta, V, r2, random_MMT) )

def iid_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10, optim_by_r=False):
    print('IID ML', flush=True) 
    start = time.time()

    def extract(out, X, P, fixed_covars, random_covars):
        N, C = P.shape
        hom2, beta, r2 = out['x'][0], out['x'][2:(2+X.shape[1])], out['x'][(2+X.shape[1]):]
        V = np.eye(C) * out['x'][1]
        l = out['fun'] * (-1)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        return( hom2, V, beta, r2, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars )

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars_d, N, C)
    n_par = 2 + n_random + X.shape[1]

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        random_MMT = [np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1)]
        X = get_X(fixed_covars, N, C)

    # optim
    if optim_by_r:
        out = r_optim(Y, P, vs, fixed_covars, random_covars, par, nrep, 'ml', 'iid', method)

        hom2, V, beta = out['hom2'][0], np.array(out['V']), np.array(out['beta'])
        l = out['l'][0]
        opt = {'convergence':out['convergence'][0], 'method':out['method'][0]}
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        random_vars, r2 = np.array(out['randomeffect_vars']), np.array(out['r2'])
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
    else:
        if par is None:
            beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
            hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 2 )
            par = [hom2, hom2] + list(beta) + [hom2] * n_random

        out, opt = util.optim(iid_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        hom2, V, beta, r2, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract(
                out, X, P, fixed_covars, random_covars)

        if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, iid_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT),
                    method=method, nrep=nrep)
            hom2, V, beta, r2, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract(
                    out, X, P, fixed_covars, random_covars)

    # wald
    A = np.ones((C,C))*hom2+V
    Vy = cal_Vy( A, vs, r2, random_MMT )
    Z = [np.repeat(np.eye(N), C, axis=0), np.eye(N*C)]
    for key in random_covars.keys():
        m = np.repeat( random_covars[key], C, axis=0 )
        Z.append( m )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)

    ml = {'hom2':hom2, 'beta':beta, 'V':V, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2, 'l':l, 'D':D, 'opt':opt}
    if nu_f:
        ml['nu'] = np.mean(np.loadtxt(nu_f))

    p = {}
    p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    p['V'] = wald.wald_test(V[0,0], 0, D[X.shape[1]+1,X.shape[1]+1], N-n_par)
    p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], D[:C,:C], n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(ml, p)

def iid_REML_loglike(par, Y, X, N, C, vs, random_MMT):
    hom2 = par[0]
    V = np.eye(C) * par[1]
    r2 = par[2:]

    return( REML_LL(Y, X, N, C, vs, hom2, V, r2, random_MMT) )

def iid_REML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10, jack_knife=False, optim_by_r=False):
    print('IID REML', flush=True)
    start = time.time()

    def extract(out, Y, X, P, vs, fixed_covars, random_covars, random_MMT):
        N, C = P.shape
        hom2, r2 = out['x'][0], out['x'][2:]
        V = np.eye(C) * out['x'][1]
        l = out['fun'] * (-1)
        A = np.ones((C,C)) * hom2 + V
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, Y.flatten() )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        return(hom2, V, r2, beta, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars, Vy)

    def reml_f(Y, X, N, C, vs, P, fixed_covars, random_covars, method):
        ''' wrapper for iid reml '''
        random_MMT = get_MMT(random_covars, C)
           
        out, opt = util.optim(iid_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        hom2, V, r2, beta, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars, Vy = extract(
                out, Y, X, P, vs, fixed_covars, random_covars, random_MMT)

        if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, iid_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT),
                    method=method, nrep=nrep)
            hom2, V, r2, beta, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars, Vy = extract(
                    out, Y, X, P, vs, fixed_covars, random_covars, random_MMT)

        return(hom2, V, r2, beta, l, fixed_vars, random_vars, 
                Vy, ct_overall_var, ct_specific_var, opt)

    def extract_R(out):
        hom2, V, beta = out['hom2'][0], np.array(out['V']), np.array(out['beta'])
        l = out['l'][0]
        opt = {'convergence':out['convergence'][0], 'method':out['method'][0]}
        random_vars, r2 = np.array(out['randomeffect_vars']), np.array(out['r2'])
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return( hom2, V, beta, r2, ct_overall_var, ct_specific_var, fixed_vars, random_vars, l, opt)

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars_d, N, C)
    n_par = 2 + n_random + X.shape[1]

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        random_MMT = [np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1)]
        X = get_X(fixed_covars, N, C)

    # optim
    if optim_by_r:
        out = r_optim(Y, P, vs, fixed_covars, random_covars, par, nrep, 'reml', 'iid', method)
        hom2, V, beta, r2, ct_overall_var, ct_specific_var, fixed_vars, random_vars, l, opt = extract_R(out)

        A = np.ones((C,C)) * hom2 + V
        Vy = cal_Vy( A, vs, r2, random_MMT )
    else:
        if par is None:
            beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
            hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 2 )
            par = [hom2] * (n_random + 2)

        hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var, opt = reml_f(
                Y, X, N, C, vs, P, fixed_covars, random_covars, method)

    # wald
    Z = [np.repeat(np.eye(N), C, axis=0), np.eye(N*C)]
    for key in random_covars.keys():
        m = np.repeat( random_covars[key], C, axis=0 )
        Z.append( m )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'beta': beta, 'hom2':hom2, 'V':V, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2, 'l':l, 'D':D, 'opt':opt}
    if nu_f:
        reml['nu'] = np.mean( np.loadtxt(nu_f) )

    p = {}
    if not jack_knife:
        p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        p['V'] = wald.wald_test(V[0,0], 0, D[1,1], N-n_par)
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], 
                np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par)
    else:
        jacks = { 'ct_beta':[], 'hom2':[], 'het':[] }
        for i in range(N):
            Y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            if optim_by_r:
                out_jk = r_optim(Y_jk, P_jk, vs_jk, fixed_covars_jk, random_covars_jk, par, nrep, 
                        'reml', 'iid', method)
                hom2_jk, V_jk, beta_jk = extract_R(out_jk)[:3]
            else:
                X_jk = get_X(fixed_covars_jk, N-1, C)
                hom2_jk, V_jk, _, beta_jk, _, _, _, _, _, _ = reml_f(
                        Y_jk, X_jk, N-1, C, vs_jk, P_jk, fixed_covars_jk, random_covars_jk, method)

            jacks['ct_beta'].append( beta_jk['ct_beta'] )
            jacks['hom2'].append( hom2_jk )
            jacks['het'].append( V_jk[0,0] )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_het = (len(jacks['het']) - 1.0) * np.var(jacks['het'])
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )

        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        p['V'] = wald.wald_test(V[0,0], 0, var_het, N-n_par)
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(reml, p)

def iid_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, jack_knife=True):
    print('IID HE', flush=True )
    start = time.time()

    def he_f(Y, vs, P, fixed_covars, random_covars):
        N, C = Y.shape
        y = Y.flatten()
        X = get_X(fixed_covars, N, C)

        theta, random_MMT = he_ols(Y, X, vs, random_covars, model='iid')
        hom2, het, r2 = theta[0], theta[1], theta[2:]
        V = np.eye(C) * het

        # beta
        A = np.ones((C,C)) * hom2 + V
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return(hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var)

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars, N, C)
    n_par = 2 + n_random + X.shape[1]

    hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var = he_f(
            Y, vs, P, fixed_covars, random_covars)

    he = {'hom2': hom2, 'V': V, 'beta':beta, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2 }
    if nu_f:
        he['nu'] = np.mean( np.loadtxt(nu_f) )

    # p values
    p = {}
    if jack_knife:
        jacks = { 'ct_beta':[], 'hom2':[], 'het':[] }
        for i in range(N):
            Y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            hom2_jk, V_jk, _, beta_jk, _, _, _, _ = he_f(
                    Y_jk, vs_jk, P_jk, fixed_covars_jk, random_covars_jk)

            jacks['hom2'].append( hom2_jk )
            jacks['ct_beta'].append( beta_jk['ct_beta'] )
            jacks['het'].append( V_jk[0,0] )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_het = (len(jacks['het']) - 1.0) * np.var(jacks['het'])
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )

        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        p['V'] = wald.wald_test(V[0,0], 0, var_het, N-n_par)
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par)

    print( time.time() - start )
    return(he, p)

def free_ML_loglike(par, Y, X, N, C, vs, random_MMT):
    hom2 = par[0]
    V = np.diag( par[1:(C+1)] )
    beta = par[(C+1):(C+1+X.shape[1])]
    r2 = par[(C+1+X.shape[1]):]

    return( ML_LL(Y, X, N, C, vs, hom2, beta, V, r2, random_MMT) )

def free_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10, optim_by_r=False):
    print('Free ML', flush=True)
    start = time.time()

    def extract(out, X, P, fixed_covars, random_covars):
        N, C = P.shape
        hom2, beta, r2 = out['x'][0], out['x'][(C+1):(C+1+X.shape[1])], out['x'][(C+1+X.shape[1]):]
        V = np.diag( out['x'][1:(C+1)] )
        l = out['fun'] * (-1)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        return( hom2, V, beta, r2, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars )

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars_d, N, C)
    n_par = 1 + C + n_random + X.shape[1]

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        random_MMT = [np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1)]
        X = get_X(fixed_covars, N, C)

    # optim
    if optim_by_r:
        out = r_optim(Y, P, vs, fixed_covars, random_covars, par, nrep, 'ml', 'free', method)

        hom2, V, beta = out['hom2'][0], np.array(out['V']), np.array(out['beta'])
        l = out['l'][0]
        opt = {'convergence':out['convergence'][0], 'method':out['method'][0]}
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        random_vars, r2 = np.array(out['randomeffect_vars']), np.array(out['r2'])
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
    else:
        if par is None:
            beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
            hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 2 )
            par = [hom2]*(C+1) + list(beta) + [hom2] * n_random

        out, opt = util.optim(free_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        hom2, V, beta, r2, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract(
                out, X, P, fixed_covars, random_covars)
        
        if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, free_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT),
                    method=method, nrep=nrep)
            hom2, V, beta, r2, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract(
                    out, X, P, fixed_covars, random_covars)

    # wald
    A = np.ones((C,C)) * hom2 + V
    Vy = cal_Vy( A, vs, r2, random_MMT )
    Z = [np.repeat(np.eye(N), C, axis=0)]
    for i in range(C):
        m = np.zeros(C)
        m[i] = 1
        Z.append(np.kron(np.identity(N), m.reshape(-1,1)))
    for key in random_covars.keys():
        m = np.repeat( random_covars[key], C, axis=0 )
        Z.append( m )
    D = wald.asymptotic_dispersion_matrix(X, Z, Vy)

    ml = {'hom2':hom2, 'beta':beta, 'V':V, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2, 'l':l, 'D':D, 'opt':opt}
    if nu_f:
        ml['nu'] = np.mean(np.loadtxt(nu_f))

    p = {}
    p['hom2'] = wald.wald_test(hom2, 0, D[X.shape[1],X.shape[1]], N-n_par)
    var_V = D[(X.shape[1]+1):(X.shape[1]+C+1), (X.shape[1]+1):(X.shape[1]+C+1)]
    p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
    #p['V_iid'] = util.wald_ct_beta(np.diag(V), var_V, n=N, P=n_par)
    #p['Vi'] = [wald.wald_test(V[i,i], 0, D[X.shape[1]+i+1,X.shape[1]+i+1], N-n_par) for i in range(C)]
    p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], D[:C,:C], n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(ml, p)

def free_REML_loglike(par, Y, X, N, C, vs, random_MMT):
    hom2 = par[0]
    V = np.diag( par[1:(C+1)] )
    r2 = par[(C+1):]

    return( REML_LL(Y, X, N, C, vs, hom2, V, r2, random_MMT) )

def free_REML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10, jack_knife=False, optim_by_r=False):
    print('Free REML', flush=True)
    start = time.time()

    def extract(out, Y, X, P, vs, fixed_covars, random_covars, random_MMT):
        N, C = P.shape
        hom2, r2 = out['x'][0], out['x'][(C+1):]
        V = np.diag( out['x'][1:(C+1)] )
        l = out['fun'] * (-1)
        A = np.ones((C,C)) * hom2 + V
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, Y.flatten() )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        return(hom2, V, r2, beta, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars, Vy)

    def reml_f(Y, X, N, C, vs, P, fixed_covars, random_covars, method):
        ''' wrapper for iid reml '''
        random_MMT = get_MMT(random_covars, C)
           
        out, opt = util.optim(free_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        hom2, V, r2, beta, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars, Vy = extract(
                out, Y, X, P, vs, fixed_covars, random_covars, random_MMT)

        if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, free_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT),
                    method=method, nrep=nrep)
            hom2, V, r2, beta, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars, Vy = extract(
                    out, Y, X, P, vs, fixed_covars, random_covars, random_MMT)

        return(hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, 
                ct_overall_var, ct_specific_var, opt)

    def extract_R(out):
        hom2, V, beta = out['hom2'][0], np.array(out['V']), np.array(out['beta'])
        l = out['l'][0]
        opt = {'convergence':out['convergence'][0], 'method':out['method'][0]}
        random_vars, r2 = np.array(out['randomeffect_vars']), np.array(out['r2'])
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return( hom2, V, beta, r2, ct_overall_var, ct_specific_var, fixed_vars, random_vars, l, opt)

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars_d, N, C)
    n_par = 1 + C + n_random + X.shape[1]

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        random_MMT = [np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1)]
        X = get_X(fixed_covars, N, C)

    # optim
    if optim_by_r:
        out = r_optim(Y, P, vs, fixed_covars, random_covars, par, nrep, 'reml', 'free', method)
        hom2, V, beta, r2, ct_overall_var, ct_specific_var, fixed_vars, random_vars, l, opt = extract_R(out)

        A = np.ones((C,C)) * hom2 + V
        Vy = cal_Vy( A, vs, r2, random_MMT )
    else:
        if par is None:
            beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
            hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 2 )
            par = [hom2] * (n_random + C + 1)

        #
        hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var, opt = reml_f(
                Y, X, N, C, vs, P, fixed_covars, random_covars, method)

    # wald
    Z = [np.repeat(np.eye(N), C, axis=0)]
    for i in range(C):
        m = np.zeros(C)
        m[i] = 1
        Z.append(np.kron(np.eye(N), m.reshape(-1,1)))
    for key in random_covars.keys():
        m = np.repeat( random_covars[key], C, axis=0 )
        Z.append( m )
    D = wald.reml_asymptotic_dispersion_matrix(X, Z, Vy)

    reml = {'beta':beta, 'hom2':hom2, 'V':V, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2, 'l':l, 'D':D, 'opt':opt}
    if nu_f:
        reml['nu'] = np.mean( np.loadtxt(nu_f) )

    p = {}
    if not jack_knife:
        p['hom2'] = wald.wald_test(hom2, 0, D[0,0], N-n_par)
        var_V = D[1:(C+1), 1:(C+1)]
        p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
        #p['V_iid'] = util.wald_ct_beta(np.diag(V), var_V, n=N, P=n_par)
        #p['Vi'] = [wald.wald_test(V[i,i], 0, D[i+1,i+1], N-n_par) for i in range(C)]
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], 
                np.linalg.inv(X.T @ np.linalg.inv(Vy) @ X)[:C,:C],
                n=N, P=n_par)
    else:
        jacks = { 'ct_beta':[], 'hom2':[], 'V':[] }
        for i in range(N):
            Y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            if optim_by_r:
                out_jk = r_optim(Y_jk, P_jk, vs_jk, fixed_covars_jk, random_covars_jk, par, nrep, 
                        'reml', 'free', method)
                hom2_jk, V_jk, beta_jk = extract_R(out_jk)[:3]
            else:
                X_jk = get_X(fixed_covars_jk, N-1, C)
                hom2_jk, V_jk, _, beta_jk, _, _, _, _, _, _ = reml_f(
                        Y_jk, X_jk, N-1, C, vs_jk, P_jk, fixed_covars_jk, random_covars_jk, method)

            jacks['ct_beta'].append( beta_jk['ct_beta'] )
            jacks['hom2'].append( hom2_jk )
            jacks['V'].append( np.diag(V_jk) )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_V = (len(jacks['V']) - 1.0) * np.cov( np.array(jacks['V']).T, bias=True )
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )

        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
        #p['V_iid'] = util.wald_ct_beta(np.diag(V), var_V, n=N, P=n_par)
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par)

    print( time.time() - start, flush=True )
    return(reml, p)

def free_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('Free HE', flush=True )
    start = time.time()

    def he_f(Y, vs, P, fixed_covars, random_covars):
        N, C = Y.shape
        y = Y.flatten()
        X = get_X(fixed_covars, N, C)

        theta, random_MMT = he_ols(Y, X, vs, random_covars, model='free')
        hom2, r2 = theta[0], theta[(1+C):]
        V = np.diag( theta[1:(C+1)] )

        # beta
        A = np.ones((C,C)) * hom2 + V
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return(hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var)

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars, N, C)
    n_par = 1 + C + n_random + X.shape[1]

    hom2, V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var = he_f(
            Y, vs, P, fixed_covars, random_covars)

    he = {'hom2': hom2, 'V': V, 'beta':beta, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2 }
    if nu_f:
        he['nu'] = np.mean( np.loadtxt(nu_f) )

    # p values
    p = {}
    if jack_knife:
        jacks = { 'ct_beta':[], 'hom2':[], 'V':[] }
        for i in range(N):
            Y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            hom2_jk, V_jk, _, beta_jk, _, _, _, _ = he_f(
                    Y_jk, vs_jk, P_jk, fixed_covars_jk, random_covars_jk)

            jacks['ct_beta'].append( beta_jk['ct_beta'] )
            jacks['hom2'].append( hom2_jk )
            jacks['V'].append( np.diag(V_jk) )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_V = (len(jacks['V']) - 1.0) * np.cov( np.array(jacks['V']).T, bias=True )
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )

        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=N, P=n_par)
        p['ct_beta'] = util.wald_ct_beta( beta['ct_beta'], var_ct_beta, n=N, P=n_par )

    print( time.time() - start , flush=True )
    return(he, p)

def full_ML_loglike(par, Y, X, N, C, vs, random_MMT):
    ngam = C * (C+1) // 2
    V = np.zeros((C,C))
    V[np.tril_indices(C)] = par[:ngam]
    V = V + V.T
    hom2 = 0
    beta = par[ngam:(ngam+X.shape[1])]
    r2 = par[(ngam+X.shape[1]):]

    return( ML_LL(Y, X, N, C, vs, hom2, beta, V, r2, random_MMT) )

def full_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10, optim_by_r=False):
    print('Full ML', flush=True)
    start = time.time()

    def extract(out, X, P, fixed_covars, random_covars):
        N, C = P.shape
        ngam = C * (C+1) // 2
        V = np.zeros((C,C))
        V[np.tril_indices(C)] = out['x'][:ngam]
        V = V + V.T
        beta, r2 = out['x'][ngam:(ngam+X.shape[1])], out['x'][(ngam+X.shape[1]):]
        l = out['fun'] * (-1)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        return( V, beta, r2, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars )

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    X = get_X(fixed_covars_d, N, C)

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        random_MMT = [np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1)]
        X = get_X(fixed_covars, N, C)

    # optim
    if optim_by_r:
        out = r_optim(Y, P, vs, fixed_covars, random_covars, par, nrep, 'ml', 'full', method)

        V, beta = np.array(out['V']), np.array(out['beta'])
        l = out['l'][0]
        opt = {'convergence':out['convergence'][0], 'method':out['method'][0]}
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        random_vars, r2 = np.array(out['randomeffect_vars']), np.array(out['r2'])
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
    else:
        if par is None:
            beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
            V = np.eye(C)[np.tril_indices(C)] * np.var(Y.flatten() - X @ beta) / ( n_random + 1 )
            par = list(V) + list(beta) + [V[0]] * n_random

        out, opt = util.optim(full_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        V, beta, r2, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract(
                out, X, P, fixed_covars, random_covars)
        
        if util.check_optim(opt, 0, ct_overall_var, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, full_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT),
                    method=method, nrep=nrep)
            V, beta, r2, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars = extract(
                    out, X, P, fixed_covars, random_covars)

    ml = {'beta':beta, 'V':V, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2, 'l':l, 'opt':opt}
    if nu_f:
        ml['nu'] = np.mean(np.loadtxt(nu_f))

    print( time.time() - start, flush=True )
    return(ml)

def full_REML_loglike(par, Y, X, N, C, vs, random_MMT):
    ngam = C * (C+1) // 2
    V = np.zeros((C,C))
    V[np.tril_indices(C)] = par[:ngam]
    V = V + V.T
    hom2 = 0
    r2 = par[ngam:]

    return( REML_LL(Y, X, N, C, vs, hom2, V, r2, random_MMT) )

def full_REML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10, optim_by_r=False):
    print('Full REML', flush=True)
    start = time.time()

    def extract(out, X, P, fixed_covars, random_covars, random_MMT):
        N, C = P.shape
        ngam = C*(C+1)//2
        r2 = out['x'][ngam:]
        V = np.zeros((C,C))
        V[np.tril_indices(C)] = out['x'][:ngam]
        V = V + V.T
        l = out['fun'] * (-1)
        A = V
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, Y.flatten() )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
        return(V, r2, beta, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars, Vy)

    def reml_f(Y, X, N, C, vs, P, fixed_covars, random_covars, method):
        ''' wrapper for iid reml '''
        random_MMT = get_MMT(random_covars, C)
           
        out, opt = util.optim(full_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        V, r2, beta, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars, Vy = extract(
                out, X, P, fixed_covars, random_covars, random_MMT)

        if util.check_optim(opt, hom2, ct_overall_var, fixed_vars, random_vars):
            out, opt = util.re_optim(out, opt, full_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT),
                    method=method, nrep=nrep)
            V, r2, beta, l, ct_overall_var, ct_specific_var, fixed_vars, random_vars, Vy = extract(
                    out, X, P, fixed_covars, random_covars, random_MMT)

        return(V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var, opt)

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)
    ngam = C*(C+1)//2
    X = get_X(fixed_covars_d, N, C)

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        random_MMT = [np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1)]
        X = get_X(fixed_covars, N, C)

    # optim
    if optim_by_r:
        out = r_optim(Y, P, vs, fixed_covars, random_covars, par, nrep, 'reml', 'full', method)

        V, beta = np.array(out['V']), np.array(out['beta'])
        l = out['l'][0]
        opt = {'convergence':out['convergence'][0], 'method':out['method'][0]}
        random_vars, r2 = np.array(out['randomeffect_vars']), np.array(out['r2'])
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        A = V
        Vy = cal_Vy( A, vs, r2, random_MMT )
    else:
        if par is None:
            beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
            V = np.eye(C)[np.tril_indices(C)]
            hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 1 )
            par = list(V * hom2) + [hom2] * n_random

        V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var, opt = reml_f(
                Y, X, N, C, vs, P, fixed_covars, random_covars, method)
        
    reml = {'beta':beta, 'V':V, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2, 'l':l, 'opt':opt}
    if nu_f:
        reml['nu'] = np.mean( np.loadtxt(nu_f) )
    
    print( time.time() - start, flush=True )
    return(reml)

def full_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}):
    print('Full HE', flush=True )
    start = time.time()

    def he_f(Y, vs, P, fixed_covars, random_covars):
        N, C = Y.shape
        ngam = C * (C+1) // 2
        y = Y.flatten()
        X = get_X(fixed_covars, N, C)

        theta, random_MMT = he_ols(Y, X, vs, random_covars, model='full')
        r2 = theta[ngam:]
        V = np.diag( theta[:C] )
        V[np.tril_indices(C,k=-1)] = theta[C:ngam]
        V = V + V.T - np.diag( np.diag( V ) )

        # beta
        A = V
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, y )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return(V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var)

    # par
    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    fixed_covars, random_covars, n_fixed, n_random, random_keys, Rs, random_MMT = util.read_covars(
            fixed_covars_d, random_covars_d, C)

    V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var = he_f(
            Y, vs, P, fixed_covars, random_covars)

    he = {'V': V, 'beta':beta, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2 }
    if nu_f:
        he['nu'] = np.mean( np.loadtxt(nu_f) )

    print( time.time() - start )

    return( he )
