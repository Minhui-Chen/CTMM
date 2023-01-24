import os, sys, re, time
import helper, mystats 
import scipy
import numpy as np
from scipy import linalg, optimize, stats
import util, wald

def get_X(fixed_covars_d, N, C):
    X = np.kron( np.ones((N,1)), np.eye(C) )
    fixed_covars = util.read_covars(fixed_covars_d)[0]
    for key in np.sort(list(fixed_covars.keys())):
        m = fixed_covars[key]
        if len( m.shape ) == 1:
            m = m.reshape(-1,1)
        X = np.concatenate( ( X, np.repeat(m, C, axis=0)), axis=1 )
    return(X)

def cal_Vy(A, vs, r2=[], random_MMT=[]):
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
    for R in random_covars.values():
        m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
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

def optim(fun, par, args, method):
    if method is None:
        out1 = optimize.minimize( fun, par, args=args, method='BFGS' )
        out = optimize.minimize( fun, out1['x'], args=args, method='Nelder-Mead' )
        opt = {'method1':'BFGS', 'success1':out1['success'], 'status1':out1['status'], 'message1':out1['message'],
                'method':'Nelder-Mead', 'success':out['success'], 'status':out['status'], 'message':out['message']}
    else:
        out = optimize.minimize( fun, par, args=args, method=method )
        opt = {'method':method, 'success':out['success'], 'status':out['status'], 'message':out['message']}
    return( out, opt )

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

def hom_ML_loglike(par, Y, X, N, C, vs, random_MMT):
    hom2 = par[0]
    beta = par[1:(1+X.shape[1])]
    V = np.zeros((C,C))
    r2 = par[(1+X.shape[1]):]

    return( ML_LL(Y, X, N, C, vs, hom2, beta, V, r2, random_MMT) )

def hom_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10):
    '''
    par: np.array of initial parameter for optimization
    '''
    print('Hom ML', flush=True)
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = Y.shape
    X = get_X(fixed_covars, N, C)
    n_par = 1 + n_random + X.shape[1]

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        X = get_X(fixed_covars, N, C)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
        hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 1 )
        par = [hom2] + list(beta) + [hom2] * n_random

    random_MMT = []
    for R in random_covars.values():
        m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
        random_MMT.append( m )
       
    out, opt = optim(hom_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
    
    hom2, beta, r2 = out['x'][0], out['x'][1:(1+X.shape[1])], out['x'][(1+X.shape[1]):]
    l = out['fun'] * (-1)
    A = np.ones((C,C)) * hom2
    Vy = cal_Vy( A, vs, r2, random_MMT )
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

    # wald
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
        par=None, method=None, nrep=10, jack_knife=False):
    print('Hom REML', flush=True)
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars, N, C)
    n_par = 1 + n_random

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        X = get_X(fixed_covars, N, C)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
        hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 1 )
        par = [hom2] * (n_random + 1)

    def reml_f(Y, vs, P, fixed_covars, random_covars, method):
        ''' wrapper for hom reml '''
        N, C = Y.shape
        X = get_X(fixed_covars, N, C)

        random_MMT = []
        for R in random_covars.values():
            m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
            random_MMT.append( m )
           
        out, opt = optim(hom_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        
        hom2, r2 = out['x'][0], out['x'][1:]
        l = out['fun'] * (-1)
        A = np.ones((C,C)) * hom2
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, Y.flatten() )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

        return(hom2, r2, beta, l, fixed_vars, random_vars, Vy, opt)

    #
    hom2, r2, beta, l, fixed_vars, random_vars, Vy, opt = reml_f(
            Y, vs, P, fixed_covars, random_covars, method)

    # wald
    Z = [np.repeat(np.eye(N), C, axis=0)]
    for key in r2.keys():
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
                n=N, P=n_par+X.shape[1])
    else:
        jacks = {'ct_beta':[], 'hom2':[]}
        for i in range(N):
            Y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
            hom2_jk, _, beta_jk, _, _, _, _ = reml_f(
                    Y_jk, vs_jk, P_jk, fixed_covars_jk, random_covars_jk, method)

            jacks['hom2'].append( hom2_jk )
            jacks['ct_beta'].append( beta_jk['ct_beta'] )

        var_hom2 = (len(jacks['hom2']) - 1.0) * np.var(jacks['hom2'])
        var_ct_beta = (len(jacks['ct_beta']) - 1.0) * np.cov( np.array(jacks['ct_beta']).T, bias=True )
        
        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, N-n_par)
        # wald test beta1 = beta2 = beta3
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par+X.shape[1])

    print( time.time() - start, flush=True )
    return(reml, p)

def hom_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, jack_knife=True):
    print('Hom HE', flush=True )
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    X = get_X(fixed_covars, N, C)
    n_par = 1 + n_random

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
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par+X.shape[1])


    print( time.time() - start )
    return(he, p)

def iid_ML_loglike(par, Y, X, N, C, vs, random_MMT):
    hom2 = par[0]
    V = np.eye(C) * par[1]
    beta = par[2:(2+X.shape[1])]
    r2 = par[(2+X.shape[1]):]

    return( ML_LL(Y, X, N, C, vs, hom2, beta, V, r2, random_MMT) )

def iid_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10):
    print('IID ML', flush=True) 
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars_d, N, C)
    n_par = 2 + n_random + X.shape[1]

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        X = get_X(fixed_covars, N, C)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
        hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 2 )
        par = [hom2, hom2] + list(beta) + [hom2] * n_random

    random_MMT = []
    for R in random_covars.values():
        m = m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
        random_MMT.append( m )
       
    out, opt = optim(iid_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
    
    hom2, beta, r2 = out['x'][0], out['x'][2:(2+X.shape[1])], out['x'][(2+X.shape[1]):]
    V = np.eye(C) * out['x'][1]
    l = out['fun'] * (-1)
    A = np.ones((C,C))*hom2+V
    Vy = cal_Vy( A, vs, r2, random_MMT )
    ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

    # wald
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
        par=None, method=None, nrep=10, jack_knife=False):
    print('IID REML', flush=True)
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars_d, N, C)
    n_par = 2 + n_random

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        X = get_X(fixed_covars, N, C)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
        hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 2 )
        par = [hom2] * (n_random + 2)

    def reml_f(Y, X, N, C, vs, P, fixed_covars, random_covars, method):
        ''' wrapper for iid reml '''
        random_MMT = []
        for R in random_covars.values():
            m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
            random_MMT.append( m )
           
        out, opt = optim(iid_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        
        hom2, r2 = out['x'][0], out['x'][2:]
        V = np.eye(C) * out['x'][1]
        l = out['fun'] * (-1)
        A = np.ones((C,C)) * hom2 + V
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, Y.flatten() )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return(hom2, V, r2, beta, l, fixed_vars, random_vars, 
                Vy, ct_overall_var, ct_specific_var, opt)

    #
    hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var, opt = reml_f(
            Y, X, N, C, vs, P, fixed_covars, random_covars, method)

    # wald
    Z = [np.repeat(np.eye(N), C, axis=0), np.eye(N*C)]
    for key in r2.keys():
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
                n=N, P=n_par+X.shape[1])
    else:
        jacks = { 'ct_beta':[], 'hom2':[], 'het':[] }
        for i in range(N):
            Y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
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
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par+X.shape[1])

    print( time.time() - start, flush=True )
    return(reml, p)

def iid_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, jack_knife=True):
    print('IID HE', flush=True )
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    X = get_X(fixed_covars, N, C)
    n_par = 2 + n_random

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
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par+X.shape[1])

    print( time.time() - start )
    return(he, p)

def free_ML_loglike(par, Y, X, N, C, vs, random_MMT):
    hom2 = par[0]
    V = np.diag( par[1:(C+1)] )
    beta = par[(C+1):(C+1+X.shape[1])]
    r2 = par[(C+1+X.shape[1]):]

    return( ML_LL(Y, X, N, C, vs, hom2, beta, V, r2, random_MMT) )

def free_ML(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, 
        par=None, method=None, nrep=10):
    print('Free ML', flush=True)
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars_d, N, C)
    n_par = 1 + C + n_random + X.shape[1]

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        X = get_X(fixed_covars, N, C)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
        hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 2 )
        par = [hom2]*(C+1) + list(beta) + [hom2] * n_random

    random_MMT = []
    for R in random_covars.values():
        m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
        random_MMT.append( m )
       
    out, opt = optim(free_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
    
    hom2, beta, r2 = out['x'][0], out['x'][(C+1):(C+1+X.shape[1])], out['x'][(C+1+X.shape[1]):]
    V = np.diag( out['x'][1:(C+1)] )
    l = out['fun'] * (-1)
    A = np.ones((C,C))*hom2+V
    Vy = cal_Vy( A, vs, r2, random_MMT )
    ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

    # wald
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
        par=None, method=None, nrep=10, jack_knife=False):
    print('Free REML', flush=True)
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars_d, N, C)
    n_par = 1 + C + n_random

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        X = get_X(fixed_covars, N, C)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
        hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 2 )
        par = [hom2] * (n_random + C + 1)

    def reml_f(Y, X, N, C, vs, P, fixed_covars, random_covars, method):
        ''' wrapper for iid reml '''
        random_MMT = []
        for R in random_covars.values():
            m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
            random_MMT.append( m )
           
        out, opt = optim(free_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        
        hom2, r2 = out['x'][0], out['x'][(C+1):]
        V = np.diag( out['x'][1:(C+1)] )
        l = out['fun'] * (-1)
        A = np.ones((C,C)) * hom2 + V
        Vy = cal_Vy( A, vs, r2, random_MMT )
        beta = util.glse( Vy, X, Y.flatten() )
        # calcualte variance of fixed and random effects, and convert to dict
        beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)
        ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )

        return(hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, 
                ct_overall_var, ct_specific_var, opt)

    #
    hom2, V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var, opt = reml_f(
            Y, X, N, C, vs, P, fixed_covars, random_covars, method)

    # wald
    Z = [np.repeat(np.eye(N), C, axis=0)]
    for i in range(C):
        m = np.zeros(C)
        m[i] = 1
        Z.append(np.kron(np.eye(N), m.reshape(-1,1)))
    for key in r2.keys():
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
                n=N, P=n_par+X.shape[1])
    else:
        jacks = { 'ct_beta':[], 'hom2':[], 'V':[] }
        for i in range(N):
            Y_jk, vs_jk, fixed_covars_jk, random_covars_jk, P_jk = util.jk_rmInd(
                    i, Y, vs, fixed_covars, random_covars, P)
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
        p['ct_beta'] = util.wald_ct_beta(beta['ct_beta'], var_ct_beta, n=N, P=n_par+X.shape[1])

    print( time.time() - start, flush=True )
    return(reml, p)

def free_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, jack_knife=False, n_equation=None):
    print('Free HE', flush=True )
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    X = get_X(fixed_covars, N, C)
    if not n_equation:
        n_equation = N
    elif n_equation == 'ind':
        n_equation = N
    elif n_equation == 'indXct':
        n_equation = N*C

    n_par = 1 + C + n_random

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

        p['hom2'] = wald.wald_test(hom2, 0, var_hom2, n_equation-n_par)
        p['V'] = wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=n_equation, P=n_par)
        #p['V_iid'] = util.wald_ct_beta(np.diag(V), var_V, n=n_equation, P=n_par)
        p['ct_beta'] = util.wald_ct_beta( beta['ct_beta'], var_ct_beta, n=n_equation, P=n_par+X.shape[1] )

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
        par=None, method=None, nrep=10):
    print('Full ML', flush=True)
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    X = get_X(fixed_covars_d, N, C)

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        X = get_X(fixed_covars, N, C)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
        V = np.eye(C)[np.tril_indices(C)] * np.var(Y.flatten() - X @ beta) / ( n_random + 1 )
        par = list(V) + list(beta) + [V[0]] * n_random

    random_MMT = []
    for R in random_covars.values():
        m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
        random_MMT.append( m )
       
    out, opt = optim(full_ML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
    
    ngam = C * (C+1) // 2
    V = np.zeros((C,C))
    V[np.tril_indices(C)] = out['x'][:ngam]
    V = V + V.T
    beta, r2 = out['x'][ngam:(ngam+X.shape[1])], out['x'][(ngam+X.shape[1]):]
    l = out['fun'] * (-1)
    ct_overall_var, ct_specific_var = util.ct_randomeffect_variance( V, P )
    # calcualte variance of fixed and random effects, and convert to dict
    beta, fixed_vars, r2, random_vars = util.cal_variance(beta, P, fixed_covars, r2, random_covars)

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
        par=None, method=None, nrep=10):
    print('Full REML', flush=True)
    start = time.time()

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    N, C = P.shape
    ngam = C*(C+1)//2
    X = get_X(fixed_covars_d, N, C)

    if n_random == 1:
        # order by random covar
        R = list( random_covars.values() )[0]
        _, R, [Y, P, vs], fixed_covars = util.order_by_randomcovariate(R, [Y, P, vs], fixed_covars)
        random_covars[ list(random_covars.keys())[0] ] = R
        X = get_X(fixed_covars, N, C)

    # optim
    if par is None:
        beta = np.linalg.inv( X.T @ X) @ (X.T @ Y.flatten())
        V = np.eye(C)[np.tril_indices(C)]
        hom2 = np.var(Y.flatten() - X @ beta) / ( n_random + 1 )
        par = list(V * hom2) + [hom2] * n_random

    def reml_f(Y, X, N, C, vs, P, fixed_covars, random_covars, method):
        ''' wrapper for iid reml '''
        random_MMT = []
        for R in random_covars.values():
            m = np.repeat( np.repeat(R @ R.T, C, axis=0), C, axis=1 )
            random_MMT.append( m )
           
        out, opt = optim(full_REML_loglike, par, args=(Y, X, N, C, vs, random_MMT), method=method)
        
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

        return(V, r2, beta, l, fixed_vars, random_vars, Vy, ct_overall_var, ct_specific_var, opt)

    #
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

    # par
    fixed_covars, random_covars = util.read_covars(fixed_covars_d, random_covars_d)
    n_fixed, n_random = len( fixed_covars.keys() ), len( random_covars.keys() )

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape

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

    V, r2, beta, fixed_vars, random_vars, ct_overall_var, ct_specific_var = he_f(
            Y, vs, P, fixed_covars, random_covars)

    he = {'V': V, 'beta':beta, 'fixedeffect_vars':fixed_vars,
            'ct_random_var':ct_overall_var, 'ct_specific_random_var':ct_specific_var,
            'randomeffect_vars':random_vars, 'r2':r2 }
    if nu_f:
        he['nu'] = np.mean( np.loadtxt(nu_f) )

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
            free_he, free_he_wald = free_HE(y_f, P_f, nu_f, jack_knife=True, n_equation=n_equation)
            HE_free_only = False
            if 'HE_free_only' in snakemake.params.keys():
                HE_free_only = snakemake.params.HE_free_only
            if HE_free_only:
                out['he'] = { 'free': free_he, 'wald':{'free': free_he_wald} }
            else:
                hom_he, hom_he_wald = hom_HE(y_f, P_f, nu_f, jack_knife=True)
                iid_he, iid_he_wald = iid_HE(y_f, P_f, nu_f, jack_knife=True)
                full_he = full_HE(y_f, P_f, nu_f)

                out['he'] = {'hom': hom_he, 'iid': iid_he, 'free': free_he, 'full': full_he,
                        'wald':{'hom':hom_he_wald, 'iid': iid_he_wald, 'free': free_he_wald} }

        ## ML
        if snakemake.params.ML:
            if not snakemake.params.HE_as_initial:
                hom_ml, hom_ml_wald = hom_ML(y_f, P_f, nu_f)
                iid_ml, iid_ml_wald = iid_ML(y_f, P_f, nu_f)
                free_ml, free_ml_wald = free_ML(y_f, P_f, nu_f)
                full_ml = full_ML(y_f, P_f, nu_f)
            else:
                hom_ml, hom_ml_wald = hom_ML(y_f, P_f, nu_f, par=util.generate_HE_initial(hom_he, ML=True) )
                iid_ml, iid_ml_wald = iid_ML(y_f, P_f, nu_f, par=util.generate_HE_initial(iid_he, ML=True) )
                free_ml, free_ml_wald = free_ML(y_f, P_f, nu_f, par=util.generate_HE_initial(free_he, ML=True) )
                full_ml = full_ML(y_f, P_f, nu_f, par=util.generate_HE_initial(full_he, ML=True) )

            out['ml'] = {'hom': hom_ml, 'iid': iid_ml, 'free': free_ml, 'full': full_ml,
                    'wald':{'hom':hom_ml_wald, 'iid':iid_ml_wald, 'free':free_ml_wald} }

            # LRT
            iid_hom_lrt = mystats.lrt(out['ml']['iid']['l'], out['ml']['hom']['l'], 1)
            free_hom_lrt = mystats.lrt(out['ml']['free']['l'], out['ml']['hom']['l'], C)
            free_iid_lrt = mystats.lrt(out['ml']['free']['l'], out['ml']['iid']['l'], C-1)
            full_hom_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['hom']['l'], C*(C+1)//2-1)
            full_iid_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['iid']['l'], C*(C+1)//2-2)
            full_free_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['free']['l'], C*(C+1)//2-C-1)

            out['ml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
                    'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
                    'full_free':full_free_lrt}

        ## REML
        if snakemake.params.REML:
            if 'Free_reml_only' not in snakemake.params.keys():
                snakemake.params.Free_reml_only = False

            if not snakemake.params.HE_as_initial:
                if snakemake.params.Free_reml_only:
                    if 'Free_reml_jk' in snakemake.params.keys():
                        free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f, nrep=5, 
                                jack_knife=snakemake.params.Free_reml_jk)
                    else:
                        free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f)
                else:
                    hom_reml, hom_reml_wald = hom_REML(y_f, P_f, nu_f)
                    iid_reml, iid_reml_wald = iid_REML(y_f, P_f, nu_f)
                    if 'Free_reml_jk' in snakemake.params.keys():
                        free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f, nrep=5, 
                                jack_knife=snakemake.params.Free_reml_jk)
                    else:
                        free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f)
                    full_reml = full_REML(y_f, P_f, nu_f)
            else:
                hom_reml, hom_reml_wald = hom_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(hom_he, REML=True))
                iid_reml, iid_reml_wald = iid_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(iid_he, REML=True))
                free_reml, free_reml_wald = free_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(free_he,REML=True))
                full_reml = full_REML(y_f, P_f, nu_f, par=util.generate_HE_initial(full_he, REML=True))

            if snakemake.params.Free_reml_only:
                out['reml'] = {'free':free_reml, 'wald':{'free':free_reml_wald} }
            else:
                out['reml'] = {'hom':hom_reml, 'iid':iid_reml, 'free':free_reml, 'full':full_reml,
                        'wald':{'hom':hom_reml_wald, 'iid':iid_reml_wald, 'free':free_reml_wald} }

                ## REML
                iid_hom_lrt = mystats.lrt(out['reml']['iid']['l'], out['reml']['hom']['l'], 1)
                free_hom_lrt = mystats.lrt(out['reml']['free']['l'], out['reml']['hom']['l'], C)
                free_iid_lrt = mystats.lrt(out['reml']['free']['l'], out['reml']['iid']['l'], C-1)
                full_hom_lrt = mystats.lrt(out['reml']['full']['l'], out['reml']['hom']['l'], C*(C+1)//2-1)
                full_iid_lrt = mystats.lrt(out['reml']['full']['l'], out['reml']['iid']['l'], C*(C+1)//2-2)
                full_free_lrt = mystats.lrt(out['reml']['full']['l'], out['reml']['free']['l'], C*(C+1)//2-C-1)

                out['reml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
                        'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
                        'full_free':full_free_lrt}

        # save
        np.save(out_f, out)

    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))

if __name__ == '__main__':
    main()

