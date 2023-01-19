import os, sys, re, time
import helper, mystats 
import scipy
import numpy as np, pandas as pd
import rpy2.robjects as robjects 
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from rpy2.robjects.conversion import localconverter
import wald, util, ctp_test

def inverse_sig2s(R, A, vs, sigma_r2):
    '''
    Inverse V(y) or sig2s, for model with one randam covariate with design matrix R
    R has to be 0 and 1, and in the structure of 1_a 1_b 1_c
    '''

    if not util.check_R(R):
        sys.exit('Wrong matrix R!\n')
    sig2s_inv_ = []
    C = A.shape[0]
    for i in range(R.shape[1]):
        n = R[:,i].sum()
        x = np.kron( np.eye(n), A )
        x += np.diag( vs[R[:,i]==1].flatten() )
        x += np.ones((n*C, n*C)) * sigma_r2
        inv = np.linalg.inv( x )
        sig2s_inv_.append( inv )
    sig2s_inv = scipy.linalg.block_diag( *sig2s_inv_ )

    return( sig2s_inv )

#def RRT(R, C):
#    '''
#    Calculate np.kron(R @ R.T, np.ones((C,C))).
#    R: has to be matrix of 0 and 1, 
#    in the structure of scipy.linalg.block_diag(np.ones((a,1)), np.ones((b,1)), np.ones((c,1)))
#    '''
#    if not util.check_R(R):
#        sys.exit('Wrong matrix R!\n')
#
#    # 
#    xs = np.sum(R, axis=1)
#    RRT = np.ones((xs[0]*C, xs[0]*C))
#    for i in range(1, len(xs)):
#        RRT = scipy.linalg.block_diag( RRT, np.ones((xs[i]*C, xs[i]*C)) )
#
#    return(RRT)

def HE_MTt_hom2(y_p, proj, D, N, C):
    Y_p = y_p.reshape(N,C)
    MTt_hom2 = y_p @ ( np.ones((C,C)) @ Y_p.T ).flatten('F')
    for i in range(N):
        for j in range(N):
            proj_ = proj[(i*C):((i+1)*C), (j*C):((j+1)*C)]
            D_ = D[(j*C):((j+1)*C), (j*C):((j+1)*C)]
            MTt_hom2 -= (proj_ @ D_ @ proj_.T).sum()
    return(MTt_hom2)

def HE_MTt_het(y_p, proj, D):
    return( y_p @ y_p - np.diag(D) @ np.diag(proj) )

def HE_MTt_free(y_p, proj, D, N, C):
    Y_p = y_p.reshape(N,C)
    xs = []
    for c in range(C):
        I_c = np.zeros((C,C))
        I_c[c,c] = 1
        xs.append( y_p @ (I_c @ Y_p.T).flatten('F') )

    #for c in range(C):
    #    y_p_c = y_p[ np.array([i*C+c for i in range(N)]) ]
    #    x = y_p_c @ y_p_c
    #    xs.append(x)

    for i in range(N):
        for j in range(N):
            proj_ = proj[(i*C):((i+1)*C), (j*C):((j+1)*C)]
            D_ = D[(j*C):((j+1)*C), (j*C):((j+1)*C)]
            m = proj_ @ D_ @ proj_.T
            for c in range(C):
                xs[c] = xs[c] - m[c,c]
    return(xs)
        
def HE_MTt_random_covar(y_p, proj, D, R, N, C):
    '''
    R : ordered matrix for random covar
    '''
    m_ = [np.ones((R[:,i].sum()*C, R[:,i].sum()*C)) for i in range(R.shape[1])]
    m_ = scipy.linalg.block_diag( *m_ )
    MTt_random_covar = y_p @ m_ @ y_p
    for i in range(R.shape[1]):
        index_i = (np.repeat(R[:,i], C) != 0)
        for j in range(R.shape[1]):
            index_j = (np.repeat(R[:,j], C) != 0)
            proj_ = proj[np.ix_(index_i, index_j)]
            D_ = D[np.ix_(index_j, index_j)]
            MTt_random_covar -= (proj_ @ D_ @ proj_.T).sum()
    return( MTt_random_covar )

def HE_MTM_hom2(proj, N, C):
    MTM_hom2 = 0
    for i in range(N):
        for j in range(N):
            proj_ = proj[(i*C):((i+1)*C), (j*C):((j+1)*C)]
            MTM_hom2 += (proj_.sum()) ** 2
    return(MTM_hom2)

def HE_MTM_hom2Nfree(proj, N, C):
    xs = np.zeros(C)
    for i in range(N):
        for j in range(N):
            proj_ = proj[(i*C):((i+1)*C), (j*C):((j+1)*C)]
            for c in range(C):
                xs[c] = xs[c] + ( (proj_[:,c]).sum() ) ** 2
    return( np.array(xs) )

def HE_MTM_freeNfree(proj, c1, c2, N, C):
    x = 0
    for i in range(N):
        for j in range(N):
            proj_ = proj[(i*C):((i+1)*C), (j*C):((j+1)*C)]
            x += proj_[c1,c2] ** 2
    return(x)

def HE_MTM_hom2Nrandom_covar(proj, R, N, C):
    MTM_hom2Nrandom_covar = 0
    for i in range(R.shape[1]):
        index_i = (np.repeat(R[:,i], C) != 0)
        for j in range(N):
            proj_ = proj[index_i, (j*C):((j+1)*C)]
            MTM_hom2Nrandom_covar += (proj_ @ np.ones((C,C)) @ proj_.T).sum() 
    return(MTM_hom2Nrandom_covar)

def HE_MTM_random_covar(proj, R, C):
    MTM_random_covar = 0
    for i in range(R.shape[1]):
        index_i = (np.repeat(R[:,i], C) != 0)
        for j in range(R.shape[1]):
            index_j = (np.repeat(R[:,j], C) != 0)
            proj_ = proj[np.ix_(index_i, index_j)]
            MTM_random_covar += (proj_ @ np.ones((index_j.sum(), index_j.sum())) @ proj_.T).sum()
    return(MTM_random_covar)

def HE_MTM_het(proj):
    return( np.trace(proj) )

def HE_MTM_hom2Nhet(proj, N, C):
    x = 0
    for i in range(N):
        proj_ = proj[(i*C):((i+1)*C), (i*C):((i+1)*C)]
        x += proj_.sum()
    return(x)

def HE_MTM_hetNrandom_covar(proj, R, C):
    x = 0
    for i in range(R.shape[1]):
        index_i = (np.repeat(R[:,i], C) != 0)
        proj_ = proj[np.ix_(index_i, index_i)]
        x += proj_.sum()
    return(x)

def HE_MTM_freeNrandom_covar(proj, R, c, N, C):
    x = 0
    I_c = np.zeros((C,C))
    I_c[c,c] = 1
    for i in range(R.shape[1]):
        index_i = (np.repeat(R[:,i], C) != 0)
        for j in range(N):
            proj_ = proj[index_i, (j*C):((j+1)*C)]
            x += (proj_ @ I_c @ proj_.T).sum()
    return(x)

def cal_HE_base_vars(Y, vs, fixed_covars_array_d={}, random_covars_array_d={}, reorder_R=True):
    '''
    With 
    Y = np.loadtxt(y_f)
    vs = np.loadtxt(nu_f)
    '''
    
    stats = {}
    # 
    N, C = Y.shape

    # 
    X = ctp_test.get_X(fixed_covars_array_d, N, C) 

    if len(random_covars_array_d.keys()) == 0:
        R = None
    elif len(random_covars_array_d.keys()) == 1:
        # order individuals by random effect matrix
        R = random_covars_array_d[list(random_covars_array_d.keys())[0]]
        if isinstance(R, str):
            R = np.loadtxt( R )
        ## convert to int
        R_ = R.astype(int)
        if np.any( R_ != R):
            print(R)
            sys.exit('Wrong matrix R!\n')
        else:
            R = R_
        
        if reorder_R:
            _, R, tmp, fixed_covars_array_d_ = util.order_by_randomcovariate(R, Xs=[Y, vs], Ys=fixed_covars_array_d)
            Y, vs = tmp
            X = ctp_test.get_X(fixed_covars_array_d_, N, C)

    # proj matrix
    proj = np.eye(N * C) - X @ np.linalg.inv(X.T @ X) @ X.T
    stats['proj'] = proj

    if len(random_covars_array_d.keys()) > 1:
        R = []
        for key in np.sort( list(random_covars_array_d.keys()) ):
            R.append( random_covars_array_d[key] )
        R = np.concatenate( R, axis=1 )

    stats['R'] = R
    stats['X'] = X
    stats['Y'] = Y
    stats['vs'] = vs

    # vectorize Y
    y = Y.flatten()

    # projected y
    y_p = proj @ y
    stats['y_p'] = y_p

#    if len(random_covars_array_d.keys()) > 1:
#        # projected nu
#        stats['vs_p'] =  proj @ np.diag(vs.flatten()) @ proj
#
#        # t
#        stats['t'] = (np.outer(y_p, y_p) - stats['vs_p']).flatten('F')

    # 
    #stats['a'] = y_p @ np.kron(np.eye(N), np.ones((C,C))) @ y_p

    return( stats )

def he_randomeffect_vars(C, stats, random_covars_array_d, M_T, return_MTMMT=False):
    M_T = list(M_T)
    k = len(M_T)
   
    Qs = []
    QQTs = []
    for key in np.sort( list(random_covars_array_d.keys()) ):
        Q = random_covars_array_d[key]
        if isinstance(Q, str):
            Q = np.loadtxt( Q )
        Qs.append( Q )
        QQT = np.repeat( np.repeat(Q @ Q.T, C, axis=0), C, axis=1 )
        QQTs.append( QQT )
        M_ = (stats['proj'] @ QQT @ stats['proj']).flatten('F')
        M_T.append(M_)
    M = np.array( M_T ).T
    #print( M.T @ M )
    #print( M.T @ stats['t'] )
    MTM = M.T @ M
    MTMMT = np.linalg.inv(MTM) @ M.T
    theta = MTMMT @ stats['t']

    sig2s = 0
    randomeffect_vars_d = {}
    r2_d = {}
    for Q, QQT, key, variance in zip(Qs, QQTs, np.sort( list(random_covars_array_d.keys()) ), theta[k:]):
        r2_d[key] = variance 
        V = variance * np.eye( Q.shape[1] )
        randomeffect_vars_d[key] = util.RandomeffectVariance_(V, Q)
        sig2s = sig2s + variance * QQT

    if return_MTMMT:
        return( theta, sig2s, randomeffect_vars_d, r2_d, MTM, MTMMT )
    else:
        return( theta, sig2s, randomeffect_vars_d, r2_d, MTM )

def hom_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('Hom HE', flush=True )
    start = time.time()

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    n_par = 1 + len(random_covars_d.keys())

    fixed_covars_array_d = {}
    for key in fixed_covars_d.keys():
        fixed_covars_array_d[key] = np.loadtxt( fixed_covars_d[key] )
    random_covars_array_d = {}
    for key in random_covars_d.keys():
        random_covars_array_d[key] = np.loadtxt( random_covars_d[key] )

    if N < 200:
        def hom_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d):
            N, C = Y.shape
            D = np.diag(vs.flatten())

            stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d, reorder_R=False)
            y_p2 = np.outer(stats['y_p'], stats['y_p'])
            #print(y_p2.shape, stats['proj'].shape, D.shape)
            stats['t'] = (y_p2 - stats['proj'] @ D @ stats['proj']).flatten('F')

            # np.linalg.inv(M.T @ M) @ M.T @ t
            M_T = [(stats['proj'] @ np.kron(np.eye(N), np.ones((C,C))) @ stats['proj']).flatten('F')]
            theta, sig2s, randomeffect_vars_d, r2_d, MTM = he_randomeffect_vars(C, stats, random_covars_array_d, M_T)
            hom2 = theta[0]

            # beta
            A = np.ones((C,C)) * hom2
            sig2s = np.kron(np.eye(N), A) + D + sig2s
            sig2s_inv = np.linalg.inv( sig2s )
            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            return(hom2, r2_d, beta, randomeffect_vars_d, sig2s, sig2s_inv, stats, y_p2, MTM)

        hom2, r2_d, beta, randomeffect_vars_d, sig2s, sig2s_inv, stats, y_p2, MTM = hom_HE_(
                Y, vs, fixed_covars_array_d, random_covars_array_d)
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_array_d )
        #print(hom2, r2_d, beta)

        # p values
        if jack_knife:
            print('jack_knife', flush=True)
            jacks = []
            for i in range(N):
                Y_, vs_, fixed_covars_array_d_, random_covars_array_d_ = util.jk_rmInd(
                        i, Y, vs, fixed_covars_array_d, random_covars_array_d)
                jacks.append( hom_HE_(Y_, vs_, fixed_covars_array_d_, random_covars_array_d_) )

            jacks_hom2 = [x[0] for x in jacks]
            var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
            
            jacks_ct_beta = [util.fixedeffect_vars( x[2], P, fixed_covars_array_d )[0]['ct_beta'] for x in jacks]
            var_ct_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_ct_beta).T, bias=True)

            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par),
                    'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta)) }

            he_p['r2'] = {}
            for i, key in enumerate( np.sort( list(random_covars_array_d.keys()) ) ):
                jacks_r2 = [x[1][key] for x in jacks]
                var_r2 = (len(jacks) - 1.0) * np.var(jacks_r2)
                he_p['r2'][key] = wald.wald_test(r2_d[key], 0, var_r2, N-n_par)

        else:
            sigma_e2 = np.var( y_p2 - stats['proj'] @ sig2s @ stats['proj'] )

            var = sigma_e2 * np.linalg.inv(MTM)
            var_hom2 = var[0,0]
            var_r2 = var[1:(1+len(random_covars_d.keys())),1:(1+len(random_covars_d.keys()))]
            var_ct_beta = np.linalg.inv( stats['X'][:,:C].T @ sig2s_inv @ stats['X'][:,:C] )
            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par) }
            if wald.check_singular( sig2s ):
                he_p['ct_beta'] = np.nan
            else:
                he_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta)) 
            he_p['r2'] = {}
            for i, key in enumerate( np.sort( list(random_covars_array_d.keys()) ) ):
                he_p['r2'][key] = wald.wald_test(r2_d[key], 0, var_r2[i,i], N-n_par)

    elif N > 200 and len(random_covars_d.keys()) == 0:
        def hom_HE_nocovars(Y, vs):
            stats = ctp_test.cal_HE_base_vars(Y, vs)
            N, C = Y.shape

            hom2 = stats['mt1'] / ( (N-1) * C * C )

            # beta
            A = np.ones((C,C)) * hom2
            sig2s_inv = [np.linalg.inv( A + np.diag(vs[i]) ) for i in range(N)]
            sig2s_inv = scipy.linalg.block_diag( *sig2s_inv )
            beta = util.glse( sig2s_inv, np.kron(np.ones((N,1)), np.eye(C)), Y.flatten(), inverse=True )

            return( hom2, beta )


        def hom_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d):
            stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
            N, C = Y.shape
            D = np.diag(vs.flatten())
            y_p, proj = stats['y_p'], stats['proj']

            # M^T t
            MTt_hom2 = HE_MTt_hom2(y_p, proj, D, N, C)

            # M^T M
            MTM_hom2 = HE_MTM_hom2(proj, N, C)

            hom2 = MTt_hom2 / MTM_hom2

            # beta
            A = np.ones((C,C)) * hom2
            sig2s_inv = [np.linalg.inv( A + np.diag(vs[i]) ) for i in range(N)]
            sig2s_inv = scipy.linalg.block_diag( *sig2s_inv )

            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            return( hom2, beta )
       
        if len(fixed_covars_array_d.keys()) == 0:
            hom2, beta = hom_HE_nocovars(Y, vs)
        else:
            hom2, beta = hom_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d)
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_array_d )

        # jackknife
        jacks = []
        for i in range(N):
            Y_, vs_, fixed_covars_array_d_, random_covars_array_d_ = util.jk_rmInd(
                    i, Y, vs, fixed_covars_array_d, random_covars_array_d)
            if len(fixed_covars_array_d.keys()) == 0:
                jacks.append( hom_HE_nocovars(Y_, vs_) )
            else:
                jacks.append(hom_HE_(Y_, vs_, fixed_covars_array_d_, random_covars_array_d_))
        jacks_hom2 = [x[0] for x in jacks]
        var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
        jacks_ct_beta = [util.fixedeffect_vars( x[1], P, fixed_covars_array_d )[0]['ct_beta'] for x in jacks]
        var_ct_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_ct_beta).T, bias=True)

        he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par),
                'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta)) }

        #stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
        randomeffect_vars_d = {}
        r2_d = {}

    elif N > 200 and len(random_covars_d.keys()) == 1:
        ### not used, but reserve it for future
        # possibly because of small sample size, this computationally optimized version for analysis with 
        # one extra random covariate is slower than the naive version
        def hom_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d):
            stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
            Y, vs, proj, y_p, R = stats['Y'], stats['vs'], stats['proj'], stats['y_p'], stats['R']
            N, C = Y.shape
            D = np.diag(vs.flatten())

            # M^T t
            MTt_hom2 = HE_MTt_hom2(y_p, proj, D, N, C)
            MTt_random_covar = HE_MTt_random_covar(y_p, proj, D, R, N, C)
            MTt = np.array( [MTt_hom2, MTt_random_covar] )

            # M^T M
            MTM_hom2 = HE_MTM_hom2(proj, N, C)
            MTM_hom2Nrandom_covar = HE_MTM_hom2Nrandom_covar(proj, R, N, C)
            MTM_random_covar = HE_MTM_random_covar(proj, R, C)
            MTM = np.array( [[MTM_hom2, MTM_hom2Nrandom_covar], [MTM_hom2Nrandom_covar, MTM_random_covar]] )

            hom2, sigma_r2 = np.linalg.inv(MTM) @ MTt

            # beta
            A = np.ones((C,C)) * hom2
            sig2s_inv = inverse_sig2s( R, A, vs, sigma_r2 )
            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            return( hom2, sigma_r2, beta, sig2s_inv, stats, MTM )
        
        hom2, sigma_r2, beta, sig2s_inv, stats, MTM = hom_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d)
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_array_d )

        randomeffect_vars_d = {}
        r2_d = {}
        for key in random_covars_array_d.keys():
            r2_d[key] = sigma_r2 
            V_r2 = sigma_r2 * np.eye( stats['R'].shape[1] )
            randomeffect_vars_d[key] = util.RandomeffectVariance_(V_r2, stats['R'])
        #print(hom2, r2_d, beta)

        # jackknife
        if jack_knife:
            jacks = []
            for i in range(N):
                Y_, vs_, fixed_covars_array_d_, random_covars_array_d_ = util.jk_rmInd(
                        i, Y, vs, fixed_covars_array_d, random_covars_array_d)
                jacks.append( hom_HE_(Y_, vs_, fixed_covars_array_d_, random_covars_array_d_) )
            jacks_hom2 = [x[0] for x in jacks]
            var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
            
            jacks_r2 = [x[1] for x in jacks]
            var_r2 = (len(jacks) - 1.0) * np.var(jacks_r2)

            jacks_ct_beta = [util.fixedeffect_vars( x[2], P, fixed_covars_array_d )[0]['ct_beta'] for x in jacks]
            var_ct_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_ct_beta).T, bias=True)

            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par),
                    'r2': {list(random_covars_d.keys())[0]: wald.wald_test(sigma_r2, 0, var_r2, N-n_par)},
                    'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta)) }
        else:
            m = 0
            for i in range(stats['R'].shape[1]):
                index_i = (stats['R'][:,i] != 0)
                m_ = stats['vs'][index_i].flatten() 
                m_ = m_ + hom2 * np.kron( np.eye(index_i.sum()), np.ones((C,C)) )
                m_ = m_ + sigma_r2 * np.ones( index_i.sum()*C, index_i.sum()*C )
                proj_ = stats['proj'][:, np.repeat(index_i, C)]
                m = m + proj_ @ m_ @ proj_.T
            sigma_e2 = np.var(np.outer(stats['y_p'], stats['y_p']) - m)
            var_hom2 = sigma_e2 * np.linalg.inv(MTM)[0,0]
            var_r2 = sigma_e2 * np.linalg.inv(MTM)[1,1]
            var_ct_beta = np.linalg.inv( stats['X'][:,:C].T @ sig2s_inv @ stats['X'][:,:C] )
            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par), 
                    'r2': {list(random_covars_d.keys())[0]: wald.wald_test(sigma_r2, 0, var_r2, N-n_par)}, 
                    'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta)) }


    he = {'hom2': hom2, 'beta':beta_d, 'fixedeffect_vars':fixedeffect_vars_d,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d }
    if nu_f:
        he['nu'] = np.mean( np.loadtxt(nu_f) ) 

    print( time.time() - start )
    return(he, he_p)

def iid_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, jack_knife=False):
    print('IID HE', flush=True )
    start = time.time()

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    n_par = 1 + 1 + len(random_covars_d.keys())

    fixed_covars_array_d = {}
    for key in fixed_covars_d.keys():
        fixed_covars_array_d[key] = np.loadtxt( fixed_covars_d[key] )
    random_covars_array_d = {}
    for key in random_covars_d.keys():
        random_covars_array_d[key] = np.loadtxt( random_covars_d[key] )


    if N <= 200:
        def iid_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d):
            N, C = Y.shape
            D = np.diag(vs.flatten())

            stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d, reorder_R=False)
            y_p2 = np.outer(stats['y_p'], stats['y_p'])
            stats['t'] = (y_p2 - stats['proj'] @ D @ stats['proj']).flatten('F')

            # np.linalg.inv(M.T @ M) @ M.T @ t
            M_T = [(stats['proj'] @ np.kron(np.eye(N), np.ones((C,C))) @ stats['proj']).flatten('F'),
                    stats['proj'].flatten('F')]
            theta, sig2s, randomeffect_vars_d, r2_d, MTM = he_randomeffect_vars(C, stats, random_covars_array_d, M_T)
            #print( MTM )
            hom2, het = theta[0], theta[1]
            V = np.eye(C) * het

            # beta
            A = np.ones((C,C)) * hom2 + V
            sig2s = np.kron(np.eye(N), A) + D + sig2s
            sig2s_inv = np.linalg.inv( sig2s )
            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            return(hom2, het, r2_d, beta, randomeffect_vars_d, sig2s, sig2s_inv, stats, y_p2, MTM)

        hom2, het, r2_d, beta, randomeffect_vars_d, sig2s, sig2s_inv, stats, y_p2, MTM = iid_HE_(
                Y, vs, fixed_covars_array_d, random_covars_array_d)
        V = np.eye(C) * het
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_d )
        #print(hom2, het, beta_d, r2_d)

        # p values
        if jack_knife:
            jacks = []
            for i in range(N):
                Y_, vs_, fixed_covars_array_d_, random_covars_array_d_ = util.jk_rmInd(
                        i, Y, vs, fixed_covars_array_d, random_covars_array_d)
                jacks.append( iid_HE_(Y_, vs_, fixed_covars_array_d_, random_covars_array_d_) )

            jacks_hom2 = [x[0] for x in jacks]
            var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
            
            jacks_het = [x[1] for x in jacks]
            var_het = (len(jacks) - 1.0) * np.var(jacks_hom2)

            jacks_ct_beta = [util.fixedeffect_vars( x[3], P, fixed_covars_array_d )[0]['ct_beta'] for x in jacks]
            var_ct_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_ct_beta).T, bias=True)

            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par),
                    'V': wald.wald_test(het, 0, var_het, N-n_par),
                    'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta)) }

            he_p['r2'] = {}
            for i, key in enumerate( np.sort( list(random_covars_array_d.keys()) ) ):
                jacks_r2 = [x[2][key] for x in jacks]
                var_r2 = (len(jacks) - 1.0) * np.var(jacks_r2)
                he_p['r2'][key] = wald.wald_test(r2_d[key], 0, var_r2, N-n_par)

        else:
            sigma_e2 = np.var( y_p2 - stats['proj'] @ sig2s @ stats['proj'] )

            var = sigma_e2 * np.linalg.inv(MTM)
            var_hom2 = var[0,0]
            var_het = var[1,1]
            var_r2 = var[2:(2+len(random_covars_d.keys())),2:(2+len(random_covars_d.keys()))]
            var_ct_beta = np.linalg.inv( stats['X'][:,:C].T @ sig2s_inv @ stats['X'][:,:C] )
            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par), 
                    'V': wald.wald_test(het, 0, var_het, N-n_par)}
            if wald.check_singular( sig2s ):
                he_p['ct_beta'] = np.nan
            else:
                he_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta))
            he_p['r2'] = {}
            for i, key in enumerate( np.sort( list(random_covars_array_d.keys()) ) ):
                he_p['r2'][key] = wald.wald_test(r2_d[key], 0, var_r2[i,i], N-n_par)

    elif N > 200 and len(random_covars_d.keys()) == 0:
        def iid_HE_nocovars(Y, vs):
            stats = ctp_test.cal_HE_base_vars(Y, vs)
            N, C = Y.shape
            D = np.diag( vs.flatten() )

            # calculate \ve{ \Pi_P^\perp (I_N \otimes I_C) \Pi_P^\perp }^T t, where t is vector of dependent variable
            # t = \ve{ y' y'^T - \Pi_P^\perp D \Pi_P^\perp }
            mt2 = (stats['y_p']**2).sum() - vs.sum() * (N-1) / N

            #
            mt = np.array([stats['mt1'], mt2])

            # make matrix M^T M, where is M is design matrix of HE regression
            mm = np.array([[(N-1)*C*C, (N-1)*C], [(N-1)*C, (N-1)*C]])

            #
            theta = np.linalg.inv(mm) @ mt
            hom2, het = theta

            # beta
            A = np.ones((C,C)) * hom2 + np.eye(C) * het
            sig2s_inv = [np.linalg.inv( A + np.diag(vs[i]) ) for i in range(N)]
            sig2s_inv = scipy.linalg.block_diag( *sig2s_inv )
            beta = util.glse( sig2s_inv, np.kron(np.ones((N,1)), np.eye(C)), Y.flatten(), inverse=True )

            return(hom2, het, beta)

        def iid_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d):
            stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
            y_p, proj = stats['y_p'], stats['proj']
            N, C = Y.shape
            D = np.diag(vs.flatten())

            # M^T t
            MTt_hom2 = HE_MTt_hom2(y_p, proj, D, N, C)
            MTt_het = HE_MTt_het(y_p, proj, D)
            MTt = np.array([MTt_hom2, MTt_het])

            # M^T M
            MTM_hom2 = HE_MTM_hom2(proj, N, C)
            MTM_het = HE_MTM_het(proj)
            MTM_hom2Nhet = HE_MTM_hom2Nhet(proj, N, C)
            MTM = np.array( [[MTM_hom2, MTM_hom2Nhet], [MTM_hom2Nhet, MTM_het]] )

            hom2, het = np.linalg.inv(MTM) @ MTt

            # beta
            A = np.ones((C,C)) * hom2 + np.eye(C) * het
            sig2s_inv = [np.linalg.inv( A + np.diag(vs[i]) ) for i in range(N)]
            sig2s_inv = scipy.linalg.block_diag( *sig2s_inv )

            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            sig2s_inv = np.linalg.inv( A + np.diag(vs[0]) )
            for i in range(1,N):
                sig2s_inv = scipy.linalg.block_diag( sig2s_inv, np.linalg.inv( A + np.diag(vs[i]) ) )

            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            return( hom2, het, beta )

        if len(fixed_covars_array_d.keys()) == 0:
            hom2, het, beta = iid_HE_nocovars(Y, vs)
        else:
            hom2, het, beta = iid_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d)
        V = np.eye(C) * het
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_array_d )

        # jackknife
        jacks = []
        for i in range(N):
            Y_, vs_, fixed_covars_array_d_, random_covars_array_d_ = util.jk_rmInd(
                    i, Y, vs, fixed_covars_array_d, random_covars_array_d)
            if len(fixed_covars_array_d.keys()) == 0:
                jacks.append( iid_HE_nocovars(Y_, vs_) )
            else:
                jacks.append( iid_HE_(Y_, vs_, fixed_covars_array_d_, random_covars_array_d_) )
        jacks_hom2 = [x[0] for x in jacks]
        var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
        jacks_het = [x[1] for x in jacks]
        var_het = (len(jacks) - 1.0) * np.var(jacks_het)
        jacks_ct_beta = [util.fixedeffect_vars( x[2], P, fixed_covars_array_d )[0]['ct_beta'] for x in jacks]
        var_ct_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_ct_beta).T, bias=True)

        he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par),
                'V': wald.wald_test(het, 0, var_het, N-n_par), 
                'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta)) }

        #stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
        randomeffect_vars_d = {}
        r2_d = {}

    elif N > 200 and len(random_covars_d.keys()) == 1:
        ### not used, but reserve it for future
        # possibly because of small sample size, this computationally optimized version for analysis with 
        # one extra random covariate is slower than the naive version
        def iid_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d):
            stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
            Y, vs, proj, y_p, R = stats['Y'], stats['vs'], stats['proj'], stats['y_p'], stats['R']
            N, C = Y.shape
            D = np.diag(vs.flatten())

            # M^T t
            MTt_hom2 = HE_MTt_hom2(y_p, proj, D, N, C)
            MTt_het = HE_MTt_het(y_p, proj, D)
            MTt_random_covar = HE_MTt_random_covar(y_p, proj, D, R, N, C)
            MTt = np.array( [MTt_hom2, MTt_het, MTt_random_covar] )

            # M^T M
            MTM_hom2 = HE_MTM_hom2(proj, N, C)
            MTM_het = HE_MTM_het(proj)
            MTM_random_covar = HE_MTM_random_covar(proj, R, C)
            MTM_hom2Nhet = HE_MTM_hom2Nhet(proj, N, C)
            MTM_hom2Nrandom_covar = HE_MTM_hom2Nrandom_covar(proj, R, N, C)
            MTM_hetNrandom_covar = HE_MTM_hetNrandom_covar(proj, R, C)
            MTM = np.array( [[MTM_hom2, MTM_hom2Nhet, MTM_hom2Nrandom_covar], 
                [0, MTM_het, MTM_hetNrandom_covar],
                [0, 0, MTM_random_covar]] )
            MTM = MTM + MTM.T - np.diag(np.diag(MTM))

            hom2, het, sigma_r2 = np.linalg.inv(MTM) @ MTt

            # beta
            A = np.ones((C,C)) * hom2 + np.eye(C) * het
            sig2s_inv = inverse_sig2s( R, A, vs, sigma_r2 )

            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            return( hom2, het, sigma_r2, beta, sig2s_inv, stats, MTM )

        hom2, het, sigma_r2, beta, sig2s_inv, stats, MTM = iid_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d)
        #print( MTM )
        V = np.eye(C) * het
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_array_d )

        randomeffect_vars_d = {}
        r2_d = {}
        for key in random_covars_array_d.keys():
            r2_d[key] = sigma_r2 
            V_r2 = sigma_r2 * np.eye( stats['R'].shape[1] )
            randomeffect_vars_d[key] = util.RandomeffectVariance_(V_r2, stats['R'])
        #print( hom2, het, beta_d, r2_d)

        # jackknife
        if jack_knife:
            jacks = []
            for i in range(N):
                Y_, vs_, fixed_covars_array_d_, random_covars_array_d_ = util.jk_rmInd(
                        i, Y, vs, fixed_covars_array_d, random_covars_array_d)
                jacks.append( iid_HE_(Y_, vs_, fixed_covars_array_d_, random_covars_array_d_) )

            jacks_hom2 = [x[0] for x in jacks]
            var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
            
            jacks_het = [x[1] for x in jacks]
            var_het = (len(jacks) - 1.0) * np.var(jacks_het)

            jacks_r2 = [x[2] for x in jacks]
            var_r2 = (len(jacks) - 1.0) * np.var(jacks_r2)

            jacks_ct_beta = [util.fixedeffect_vars( x[3], P, fixed_covars_array_d )[0]['ct_beta'] for x in jacks]
            var_ct_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_ct_beta).T, bias=True)

            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par),
                    'V': wald.wald_test(het, 0, var_het, N-n_par),
                    'r2': {list(random_covars_d.keys())[0]: wald.wald_test(sigma_r2, 0, var_r2, N-n_par)},
                    'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta)) }
        else:
            m = 0
            for i in range(stats['R'].shape[1]):
                index_i = (stats['R'][:,i] != 0)
                m_ = stats['vs'][index_i].flatten()
                m_ = m_ + np.kron( np.eye(index_i.sum()), np.ones((C,C))*hom2 + V )
                m_ = m_ + sigma_r2 * np.ones( index_i.sum()*C, index_i.sum()*C )
                proj_ = stats['proj'][:, np.repeat(index_i, C)]
                m = m + proj_ @ m_ @ proj_.T
            sigma_e2 = np.var(np.outer(stats['y_p'], stats['y_p']) - m)
            var_hom2 = sigma_e2 * np.linalg.inv(MTM)[0,0]
            var_het = sigma_e2 * np.linalg.inv(MTM)[1,1]
            var_r2 = sigma_e2 * np.linalg.inv(MTM)[2,2]
            var_ct_beta = np.linalg.inv( stats['X'][:,:C].T @ sig2s_inv @ stats['X'][:,:C] )
            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, N-n_par), 
                    'V': wald.wald_test(het, 0, var_het, N-n_par),
                    'r2': {list(random_covars_d.keys())[0]: wald.wald_test(sigma_r2, 0, var_r2, N-n_par)}, 
                    'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=N, P=n_par+len(beta)) }

    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )

    he = {'hom2': hom2, 'V': V, 'beta':beta_d, 'fixedeffect_vars':fixedeffect_vars_d,
            'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d }
    if nu_f:
        he['nu'] = np.mean( np.loadtxt(nu_f) ) 

    print( time.time() - start )
    return(he, he_p)

def free_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}, jack_knife=False, n_equation=None):
    print('Free HE', flush=True )
    start = time.time()

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape
    if not n_equation:
        n_equation = N
    elif n_equation == 'ind':
        n_equation = N
    elif n_equation == 'indXct':
        n_equation = N*C
        
    n_par = 1 + C + len(random_covars_d.keys())

    fixed_covars_array_d = {}
    for key in fixed_covars_d.keys():
        fixed_covars_array_d[key] = np.loadtxt( fixed_covars_d[key] )
    random_covars_array_d = {}
    for key in random_covars_d.keys():
        random_covars_array_d[key] = np.loadtxt( random_covars_d[key] )


    if N <= 200:
        def free_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d):
            N, C = Y.shape
            D = np.diag(vs.flatten())

            stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d, reorder_R=False)
            y_p2 = np.outer(stats['y_p'], stats['y_p'])
            stats['t'] = (y_p2 - stats['proj'] @ D @ stats['proj']).flatten('F')

            # np.linalg.inv(M.T @ M) @ M.T @ t
            M_T = [(stats['proj'] @ np.kron(np.eye(N), np.ones((C,C))) @ stats['proj']).flatten('F')]
            for i in range(C):
                m = np.zeros((C,C))
                m[i,i] = 1
                M_T.append( (stats['proj'] @ np.kron(np.eye(N), m) @ stats['proj']).flatten('F') )
            theta, sig2s, randomeffect_vars_d, r2_d, MTM = he_randomeffect_vars(C,stats, random_covars_array_d,M_T)
            hom2, V = theta[0], np.diag(theta[1:(C+1)])

            # beta
            A = np.ones((C,C)) * hom2 + V
            sig2s = np.kron(np.eye(N), A) + D + sig2s
            sig2s_inv = np.linalg.inv( sig2s )
            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            return(hom2, V, r2_d, beta, randomeffect_vars_d, sig2s, sig2s_inv, stats, y_p2, MTM)

        hom2, V, r2_d, beta, randomeffect_vars_d, sig2s, sig2s_inv, stats, y_p2, MTM = free_HE_(
                Y, vs, fixed_covars_array_d, random_covars_array_d)
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_d )

        # p values
        if jack_knife:
            print( 'HE (N <=200 ) Jackknife' )
            jacks = []
            for i in range(N):
                Y_, vs_, fixed_covars_array_d_, random_covars_array_d_ = util.jk_rmInd(
                        i, Y, vs, fixed_covars_array_d, random_covars_array_d)
                jacks.append( free_HE_(Y_, vs_, fixed_covars_array_d_, random_covars_array_d_) )

            jacks_hom2 = [x[0] for x in jacks]
            var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
            
            jacks_V = [np.diag(x[1]) for x in jacks]
            var_V = (len(jacks) - 1.0) * np.cov(np.array(jacks_V).T, bias=True)

            jacks_ct_beta = [util.fixedeffect_vars( x[3], P, fixed_covars_array_d )[0]['ct_beta'] for x in jacks]
            var_ct_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_ct_beta).T, bias=True)

            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, n_equation-n_par),
                    'V': wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=n_equation, P=n_par),
                    'V_iid': util.wald_ct_beta(np.diag(V), var_V, n=n_equation, P=n_par),
                    'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=n_equation, P=n_par+len(beta)) }

            #he_p['var_V'] = var_V # tmp
            #he_p['jacks_V'] = jacks_V # tmp
            #he_p['var_ct_beta'] = var_ct_beta # tmp
            #he_p['jacks_ct_beta'] = jacks_ct_beta # tmp

            he_p['r2'] = {}
            for i, key in enumerate( np.sort( list(random_covars_array_d.keys()) ) ):
                jacks_r2 = [x[2][key] for x in jacks]
                var_r2 = (len(jacks) - 1.0) * np.var(jacks_r2)
                he_p['r2'][key] = wald.wald_test(r2_d[key], 0, var_r2, n_equation-n_par)

        else:
            sigma_e2 = np.var( y_p2 - stats['proj'] @ sig2s @ stats['proj'] )

            var = sigma_e2 * np.linalg.inv(MTM)
            var_hom2 = var[0,0]
            var_V = var[1:(C+1),1:(C+1)]
            var_r2 = var[(C+1):(C+1+len(random_covars_d.keys())),(C+1):(C+1+len(random_covars_d.keys()))]
            var_ct_beta = np.linalg.inv( stats['X'][:,:C].T @ sig2s_inv @ stats['X'][:,:C] )
            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, n_equation-n_par), 
                    'V': wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=n_equation, P=n_par)}
            if wald.check_singular( sig2s ):
                he_p['ct_beta'] = np.nan
            else:
                he_p['ct_beta'] = util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=n_equation, P=n_par+len(beta)) 
            he_p['r2'] = {}
            for i, key in enumerate( np.sort( list(random_covars_array_d.keys()) ) ):
                he_p['r2'][key] = wald.wald_test(r2_d[key], 0, var_r2[i,i], n_equation-n_par)

    elif N > 200 and len(random_covars_d.keys()) == 0:
        def free_HE_nocovars(Y, vs):
            stats = ctp_test.cal_HE_base_vars(Y, vs)
            N, C = Y.shape

            mt = [stats['mt1']]
            # calculate \ve{  \Pi_P^\perp (I_N \otimes { m_{k,k}=1 }  \Pi_P^\perp } t, where t is vector of dependent variable
            # t = \ve{ y' y'^T - \Pi_P^\perp D \Pi_P^\perp }
            for k in range(C):
                mt.append((stats['y_p'].reshape(N,C)[:,k]**2).sum() - vs[:,k].sum() * (N-1) / N)

            # make matrix M^T M, where is M is design matrix of HE regression
            mm = np.identity(C+1)
            mm[0,:] = 1
            mm[:,0] = 1
            mm = (N-1) * mm
            mm[0,0] = (N-1) * C * C

            #
            theta = np.linalg.inv(mm) @ mt
            hom2 = theta[0]
            V = np.diag(theta[1:])

            # beta
            A = np.ones((C,C)) * hom2 + V
            sig2s_inv = [np.linalg.inv( A + np.diag(vs[i]) ) for i in range(N)]
            sig2s_inv = scipy.linalg.block_diag( *sig2s_inv )
            beta = util.glse( sig2s_inv, np.kron(np.ones((N,1)), np.eye(C)), Y.flatten(), inverse=True )

            return( theta, beta )

        def free_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d):
            stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
            y_p, proj = stats['y_p'], stats['proj']
            N, C = Y.shape
            D = np.diag(vs.flatten())

            # M^T t
            MTt_hom2 = HE_MTt_hom2(y_p, proj, D, N, C)
            MTt_free = HE_MTt_free(y_p, proj, D, N, C)
            MTt = np.array( [MTt_hom2] +  MTt_free )

            # M^T M
            MTM_hom2 = np.array([HE_MTM_hom2(proj, N, C)])
            MTM_hom2Nfree = HE_MTM_hom2Nfree(proj, N, C)

            MTM_freeNfree = []
            for c1 in range(C):
                MTM_freeNfree_ = []
                for c2 in range(C):
                    if c1 <= c2:
                        MTM_freeNfree_.append( HE_MTM_freeNfree(proj, c1, c2, N, C) )
                    else:
                        MTM_freeNfree_.append( 0 )
                MTM_freeNfree.append( MTM_freeNfree_ )
            MTM_freeNfree = np.array(MTM_freeNfree)

            MTM = np.block( [[MTM_hom2, MTM_hom2Nfree], [np.zeros((C,1)), MTM_freeNfree]] )
            MTM = MTM + MTM.T - np.diag(np.diag(MTM))

            theta = np.linalg.inv(MTM) @ MTt
            hom2, V = theta[0], np.diag(theta[1:])

            # beta
            A = np.ones((C,C)) * hom2 + V
            sig2s_inv = np.linalg.inv( A + np.diag(vs[0]) )
            for i in range(1,N):
                sig2s_inv = scipy.linalg.block_diag( sig2s_inv, np.linalg.inv( A + np.diag(vs[i]) ) )

            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            return( theta, beta )

        if len(fixed_covars_array_d.keys()) == 0:
            theta, beta = free_HE_nocovars(Y, vs)
        else:
            theta, beta = free_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d)
        hom2, V = theta[0], np.diag(theta[1:])
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_array_d )

        # jackknife
        jacks = []
        for i in range(N):
            Y_, vs_, fixed_covars_array_d_, random_covars_array_d_ = util.jk_rmInd(
                    i, Y, vs, fixed_covars_array_d, random_covars_array_d)
            if len(fixed_covars_array_d.keys()) == 0:
                jacks.append( free_HE_nocovars(Y_, vs_) )
            else:
                jacks.append( free_HE_(Y_, vs_, fixed_covars_array_d_, random_covars_array_d_) )
        jacks_hom2 = [x[0][0] for x in jacks]
        var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
        jacks_V = [x[0][1:] for x in jacks]
        var_V = (len(jacks) - 1.0) * np.cov(np.array(jacks_V).T, bias=True)
        jacks_ct_beta = [util.fixedeffect_vars( x[1], P, fixed_covars_array_d )[0]['ct_beta'] for x in jacks]
        var_ct_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_ct_beta).T, bias=True)

        he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, n_equation-n_par),
                'V': wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=n_equation, P=n_par),
                'V_iid': util.wald_ct_beta(np.diag(V), var_V, n=n_equation, P=n_par),
                'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=n_equation, P=n_par+len(beta)) }

        #stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
        randomeffect_vars_d = {}
        r2_d = {}
    elif N > 200 and len(random_covars_d.keys()) == 1:
        ### not used, but reserve it for future
        # possibly because of small sample size, this computationally optimized version for analysis with 
        # one extra random covariate is slower than the naive version
        def free_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d):
            stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
            Y, vs, proj, y_p, R = stats['Y'], stats['vs'], stats['proj'], stats['y_p'], stats['R']
            N, C = Y.shape
            D = np.diag(vs.flatten())

            # M^T t
            MTt_hom2 = HE_MTt_hom2(y_p, proj, D, N, C)
            MTt_free = HE_MTt_free(y_p, proj, D, N, C) 
            MTt_random_covar = HE_MTt_random_covar(y_p, proj, D, R, N, C)
            MTt = np.array( [MTt_hom2] + MTt_free + [MTt_random_covar] )

            # M^T M
            MTM_hom2 = HE_MTM_hom2(proj, N, C)
            MTM_hom2Nfree = HE_MTM_hom2Nfree(proj, N, C)
            MTM_hom2Nrandom_covar = HE_MTM_hom2Nrandom_covar(proj, R, N, C)

            MTM_freeNfree = []
            for c1 in range(C):
                MTM_freeNfree_ = []
                for c2 in range(C):
                    if c1 <= c2:
                        MTM_freeNfree_.append( HE_MTM_freeNfree(proj, c1, c2, N, C) )
                    else:
                        MTM_freeNfree_.append( 0 )
                MTM_freeNfree.append( MTM_freeNfree_ )
            MTM_freeNfree = np.array(MTM_freeNfree)

            MTM_freeNrandom_covar = np.array([HE_MTM_freeNrandom_covar(proj, R, c, N, C) for c in range(C)])

            MTM_random_covar = HE_MTM_random_covar(proj, R, C)

            MTM = np.block( [[MTM_hom2, MTM_hom2Nfree, MTM_hom2Nrandom_covar],
                [np.zeros((C,1)), MTM_freeNfree, MTM_freeNrandom_covar.reshape(-1,1)],
                [0, np.zeros(C), MTM_random_covar]] )
            MTM = MTM + MTM.T - np.diag(np.diag(MTM))

            theta = np.linalg.inv(MTM) @ MTt
            #print( MTM )
            #print(MTt)

            # beta
            hom2, V, sigma_r2 = theta[0], np.diag(theta[1:-1]), theta[-1]
            A = np.ones((C,C)) * hom2 + V
            sig2s_inv = inverse_sig2s( R, A, vs, sigma_r2 )

            beta = util.glse( sig2s_inv, stats['X'], Y.flatten(), inverse=True )

            return( theta, beta, sig2s_inv, stats, MTM )
        
        theta, beta, sig2s_inv, stats, MTM = free_HE_(Y, vs, fixed_covars_array_d, random_covars_array_d)
        hom2, V, sigma_r2 = theta[0], np.diag(theta[1:-1]), theta[-1]
        beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_array_d )
        #print( hom2, V, sigma_r2, beta_d )

        randomeffect_vars_d = {}
        r2_d = {}
        for key in random_covars_array_d.keys():
            r2_d[key] = sigma_r2 
            V_r2 = sigma_r2 * np.eye( stats['R'].shape[1] )
            randomeffect_vars_d[key] = util.RandomeffectVariance_(V_r2, stats['R'])

        # jackknife
        if jack_knife:
            jacks = []
            for i in range(N):
                Y_, vs_, fixed_covars_array_d_, random_covars_array_d_ = util.jk_rmInd(
                        i, Y, vs, fixed_covars_array_d, random_covars_array_d)
                jacks.append( free_HE_(Y_, vs_, fixed_covars_array_d_, random_covars_array_d_) )

            jacks_hom2 = [x[0][0] for x in jacks]
            var_hom2 = (len(jacks) - 1.0) * np.var(jacks_hom2)
            
            jacks_V = [x[0][1:-1] for x in jacks]
            var_V = (len(jacks) - 1.0) * np.cov(np.array(jacks_V).T, bias=True)

            jacks_r2 = [x[0][-1] for x in jacks]
            var_r2 = (len(jacks) - 1.0) * np.var(jacks_r2)

            jacks_ct_beta = [util.fixedeffect_vars( x[1], P, fixed_covars_array_d )[0]['ct_beta'] for x in jacks]
            var_ct_beta = (len(jacks) - 1.0) * np.cov(np.array(jacks_ct_beta).T, bias=True)

            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, n_equation-n_par),
                    'V': wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=n_equation, P=n_par),
                    'r2': {list(random_covars_d.keys())[0]: wald.wald_test(sigma_r2, 0, var_r2, n_equation-n_par)}, 
                    'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=n_equation, P=n_par+len(beta)) }
        else:
            m = 0
            for i in range(stats['R'].shape[1]):
                index_i = (stats['R'][:,i] != 0)
                m_ = stats['vs'][index_i].flatten()
                m_ = m_ + np.kron( np.eye(index_i.sum()), np.ones((C,C))*hom2 + V )
                m_ = m_ + sigma_r2 * np.ones( index_i.sum()*C, index_i.sum()*C )
                proj_ = stats['proj'][:, np.repeat(index_i, C)]
                m = m + proj_ @ m_ @ proj_.T
            sigma_e2 = np.var(np.outer(stats['y_p'], stats['y_p']) - m)
            var_hom2 = sigma_e2 * np.linalg.inv(MTM)[0,0]
            var_V = sigma_e2 * np.linalg.inv(MTM)[1:(C+1),1:(C+1)]
            var_r2 = sigma_e2 * np.linalg.inv(MTM)[C+1,C+1]
            var_ct_beta = np.linalg.inv( stats['X'][:,:C].T @ sig2s_inv @ stats['X'][:,:C] )
            he_p = {'hom2': wald.wald_test(hom2, 0, var_hom2, n_equation-n_par), 
                    'V': wald.mvwald_test(np.diag(V), np.zeros(C), var_V, n=n_equation, P=n_par),
                    'r2': {list(random_covars_d.keys())[0]: wald.wald_test(sigma_r2, 0, var_r2, n_equation-n_par)}, 
                    'ct_beta': util.wald_ct_beta(beta_d['ct_beta'], var_ct_beta, n=n_equation, P=n_par+len(beta)) }

    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )

    he = {'hom2': hom2, 'V': V, 'beta':beta_d, 'fixedeffect_vars':fixedeffect_vars_d,
            'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d }
    if nu_f:
        he['nu'] = np.mean( np.loadtxt(nu_f) ) 
    
    print( time.time() - start , flush=True )
    return(he, he_p)

def full_HE(y_f, P_f, ctnu_f, nu_f=None, fixed_covars_d={}, random_covars_d={}):
    print('Full HE', flush=True )
    start = time.time()

    Y = np.loadtxt(y_f)
    P = np.loadtxt(P_f)
    vs = np.loadtxt(ctnu_f)
    D = np.diag(vs.flatten())
    N, C = Y.shape

    fixed_covars_array_d = {}
    for key in fixed_covars_d.keys():
        fixed_covars_array_d[key] = np.loadtxt( fixed_covars_d[key] )
    random_covars_array_d = {}
    for key in random_covars_d.keys():
        random_covars_array_d[key] = np.loadtxt( random_covars_d[key] )

    stats = cal_HE_base_vars(Y, vs, fixed_covars_array_d, random_covars_array_d)
    t = (np.outer(stats['y_p'], stats['y_p']) - stats['proj'] @ D @ stats['proj']).flatten('F')
    stats['t'] = t

    M = []

    for i in range(C):
        m = np.zeros((C,C))
        m[i,i] = 1
        M.append((stats['proj'] @ np.kron(np.eye(N), m) @ stats['proj']).flatten('F') )
    for i in range(1,C):
        for j in range(i):
            m = np.zeros((C,C))
            m[i,j] = 1
            m[j,i] = 1
            M.append((stats['proj'] @ np.kron(np.eye(N), m) @ stats['proj']).flatten('F') )

    M = np.array(M).T

    if len(random_covars_d.keys()) == 0:
        theta = np.linalg.inv(M.T @ M) @ M.T @ t
        V = np.zeros((C,C))
        V[np.tril_indices(C,k=-1)] = theta[C:]
        V = V + V.T
        V = V + np.diag(theta[:C])
        sig2s = np.kron(np.eye(N), V) + D
        randomeffect_vars_d = {}
        r2_d = {}
    else:
        theta, sig2s, randomeffect_vars_d, r2_d,MTM = he_randomeffect_vars(C, stats, random_covars_array_d,list(M.T))
        V = np.zeros((C,C))
        ngam = C * (C+1) // 2
        V[np.tril_indices(C,k=-1)] = theta[C:ngam]
        V = V + V.T
        V = V + np.diag(theta[:C])
        sig2s = np.kron(np.eye(N), V) + D + sig2s

    beta = util.glse( sig2s, stats['X'], Y.flatten() )

    beta_d, fixedeffect_vars_d = util.fixedeffect_vars( beta, P, fixed_covars_d )

    ct_random_var, ct_specific_random_var = util.ct_randomeffect_variance( V, P )

    he = {'V': V, 'beta':beta_d, 'fixedeffect_vars':fixedeffect_vars_d,
            'randomeffect_vars':randomeffect_vars_d, 'r2':r2_d,
            'ct_random_var':ct_random_var, 'ct_specific_random_var':ct_specific_random_var }
    if nu_f:
        he['nu'] = np.mean( np.loadtxt(nu_f) )

    print( time.time() - start )

    return( he )

def collect_covariates(snakemake, donors=None):
    '''
    Read covariates
    donors: list of individuals to keep for analysis
    '''

    fixed_covars_d = {}
    random_covars_d = {}  # random effect only support homogeneous variance i.e. \sigma * I
    covars_f = helper.generate_tmpfn()
    ## pca
    ### get individuals after filtering from pca result file
    pca = pd.read_table(snakemake.input.pca).sort_values(by='donor')
    if donors is not None:
        print( donors )
        pca = pca.loc[pca['donor'].isin(donors)]
    else:
        donors = np.array(pca['donor'])
        print( donors )
    if int(snakemake.wildcards.PC) > 0:
        pcs = [f'PC{i+1}' for i in range(int(snakemake.wildcards.PC))]
        pca = pca[pcs]
        np.savetxt(covars_f+'.pca', np.array(pca))
        fixed_covars_d['pca'] = covars_f+'.pca'
    ## supp: sex disease
    if snakemake.wildcards.sex == 'Y':
        supp = pd.read_table(snakemake.input.supp, usecols=['donor_id_short','sex'])
        supp = supp.rename(columns={'donor_id_short':'donor'})
        # remove the duplicated individual iisa
        supp = supp.drop_duplicates(subset='donor')
        supp = supp.loc[supp['donor'].isin(donors)]
        supp['code'] = 0
        supp.loc[supp['sex'] == 'male', 'code'] = 1 / (supp.loc[supp['sex'] == 'male'].shape[0])
        supp.loc[supp['sex'] == 'female', 'code'] = -1 / (supp.loc[supp['sex'] == 'female'].shape[0])
        np.savetxt(covars_f+'.sex', np.array(supp.sort_values(by='donor')['code']))
        fixed_covars_d['sex'] = covars_f+'.sex'
    if snakemake.wildcards.disease == 'Y':
        supp = pd.read_table(snakemake.input.supp, usecols=['donor_id_short','donor_disease_status'])
        supp = supp.rename(columns={'donor_id_short':'donor', 'donor_disease_status':'disease'})
        # remove the duplicated individual iisa
        supp = supp.drop_duplicates(subset='donor')
        supp = supp.loc[supp['donor'].isin(donors)]
        if len(np.unique(supp['disease'])) == 1:
            print('No disease')
        else:
            supp['code'] = 0
            supp.loc[supp['disease'] == 'normal', 'code'] = 1 / (supp.loc[supp['disease'] == 'normal'].shape[0])
            supp.loc[supp['disease'] == 'neonatal_diabetes', 'code'] = -1 / (supp.loc[supp['disease'] == 'neonatal_diabetes'].shape[0])
            np.savetxt(covars_f+'.disease', np.array(supp.sort_values(by='donor')['code']))
            fixed_covars_d['disease'] = covars_f+'.disease'
    ## meta: experiment
    if snakemake.wildcards.experiment in ['Y', 'R']:
        meta = pd.read_table(snakemake.input.meta, usecols=['donor', 'experiment'])
        meta = meta.loc[meta['donor'].isin(donors)]
        meta = meta.drop_duplicates().sort_values(by='donor').reset_index(drop=True)
        if meta.shape[0] != len(np.unique(meta['donor'])):
            print(meta[meta.duplicated(subset='donor',keep=False)])
            sys.exit('More than one experiments for an individual!\n')
        experiments = list( np.unique(meta['experiment']) )
        if snakemake.wildcards.experiment == 'R':
            for experiment in experiments:
                meta[experiment] = 0
                meta.loc[meta['experiment']==experiment, experiment] = 1
            np.savetxt(covars_f+'.experiment', np.array(meta[experiments]))
            random_covars_d['experiment'] = covars_f+'.experiment'
        else:
            for experiment in experiments[:-1]:
                meta[experiment] = 0
                meta.loc[meta['experiment'] == experiment, experiment] = 1 / (meta.loc[meta['experiment'] == experiment].shape[0])
                meta.loc[meta['experiment'] == experiments[-1], experiment] = -1 / (meta.loc[meta['experiment'] == experiments[-1]].shape[0])
            np.savetxt(covars_f+'.experiment', np.array(meta[experiments[:-1]]))
            fixed_covars_d['experiment'] = covars_f+'.experiment'

    return(fixed_covars_d, random_covars_d)

def main():
    # par
    params = snakemake.params
    input = snakemake.input
    output = snakemake.output
    wildcards = snakemake.wildcards

    # collect covariates
    fixed_covars_d, random_covars_d = collect_covariates(snakemake)

    #
    genes = params.genes
    outs = []
    for gene, y_f, P_f, nu_f, ctnu_f in zip(genes, [line.strip() for line in open(input.imputed_ct_y)],
            [line.strip() for line in open(input.P)], [line.strip() for line in open(input.nu)],
            [line.strip() for line in open(input.imputed_ct_nu)]):
        #if gene not in ['ENSG00000141448_GATA6', 'ENSG00000141506_PIK3R5']:
        #    continue
        print(y_f, P_f, nu_f, ctnu_f)
        out_f = re.sub('/rep/', f'/rep{gene}/', params.out)
        os.makedirs(os.path.dirname(out_f), exist_ok=True)

        # celltype specific mean nu
        ctnu = pd.read_table( ctnu_f )
        cts = np.unique( ctnu['day'] )
        ctnu_grouped = ctnu.groupby('day').mean()

        # transform y and ctnu from vector to matrix
        tmp_f = helper.generate_tmpfn()
        y = pd.read_table(y_f)
        y = y.pivot(index='donor', columns='day', values=gene)
        ctnu = ctnu.pivot(index='donor', columns='day', values=gene)
        y_f = tmp_f+'.y'
        ctnu_f = tmp_f+'.ctnu'
        y.to_csv(y_f, sep='\t', index=False, header=False)
        ctnu.to_csv(ctnu_f, sep='\t', index=False, header=False)

        if snakemake.wildcards.im_miny.split('-')[0] == 'std':
            threshold = float(snakemake.wildcards.im_miny.split('-')[1])
            tmp = y.copy()
            for ct in cts:
                mean, std = np.mean(y[ct]), np.std(y[ct])
                tmp.loc[(tmp[ct] > (mean + threshold * std)) | (tmp[ct] < (mean - threshold * std)), ct] = np.nan
            tmp = tmp.dropna()
            remain_inds = np.array(tmp.index)
            #print(y.index)
            #print(tmp.index)
            #print( y.shape, tmp.shape)

            nu = np.loadtxt(nu_f)
            nu_f = tmp_f+'.nu'
            np.savetxt(nu_f, nu[np.isin(np.unique(y.index),remain_inds)], delimiter='\t')
            P = pd.read_table(P_f, header=None)
            P = P.loc[y.index.isin(remain_inds)]
            P_f = tmp_f+'.P'
            P.to_csv(P_f, sep='\t', index=False, header=False)
            y = y.loc[y.index.isin(remain_inds)]
            y.to_csv(y_f, sep='\t', index=False, header=False)
            ctnu = ctnu.loc[ctnu.index.isin(remain_inds)]
            ctnu.to_csv(ctnu_f, sep='\t', index=False, header=False)

            # collect covariates
            #print( remain_inds )
            fixed_covars_d, random_covars_d = collect_covariates(snakemake, remain_inds)


        # if there are individuals with more than 1 cts with ctnu =0 , hom and IID is gonna broken
        # so just skip these ~20 genes
        if 'Hom' not in snakemake.params.keys():
            snakemake.params.Hom = True
        if 'IID' not in snakemake.params.keys():
            snakemake.params.IID = True
        if np.any( (ctnu < 1e-12).sum(axis=1) > 1 ):
            print(gene)
            if snakemake.params.Hom or snakemake.params.IID:
                continue
            else:
                pass

        out = {'gene': gene, 'ct_mean_nu': {ct:ctnu_grouped.loc[ct, gene] for ct in cts}}
        # HE
        if 'HE_as_initial' not in snakemake.params.keys():
            snakemake.params.HE_as_initial = False
        if snakemake.params.HE_as_initial:
            snakemake.params.HE = True

        if snakemake.params.HE:
            jack_knife = False
            if 'jack_knife' in snakemake.params.keys():
                jack_knife = snakemake.params.jack_knife
            free_he, free_he_p = free_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                    jack_knife=jack_knife)
            if snakemake.params.Hom:
                hom_he, hom_he_p = hom_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d, 
                        jack_knife=jack_knife)
            else:
                hom_he, hom_he_p = free_he, free_he_p
            if snakemake.params.IID:
                iid_he, iid_he_p = iid_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d, 
                        jack_knife=jack_knife)
            else:
                iid_he, iid_he_p = free_he, free_he_p
            out['he'] = {'hom': hom_he, 'iid': iid_he, 'free': free_he, 
                    'wald':{'hom':hom_he_p, 'iid':iid_he_p, 'free':free_he_p}}
            if 'Full_HE' not in snakemake.params.keys():
                full_he = full_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                out['he']['full'] = full_he
            else:
                if snakemake.params.Full_HE:
                    full_he = full_HE(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                    out['he']['full'] = full_he

        ## ML
        #null_ml = null_ML(y_f, P_f, nu_f, fixed_covars_d)
        if snakemake.params.ML:
            if not snakemake.params.HE_as_initial:
                free_ml, free_ml_p = ctp_test.free_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                full_ml = ctp_test.full_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                if snakemake.params.Hom:
                    hom_ml, hom_ml_p = ctp_test.hom_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                else:
                    hom_ml, hom_ml_p = free_ml, free_ml_p
                if snakemake.params.IID:
                    iid_ml, iid_ml_p = ctp_test.iid_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                else:
                    iid_ml, iid_ml_p = free_ml, free_ml_p
            else:
                free_ml, free_ml_p = ctp_test.free_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(free_he, ML=True))
                full_ml = ctp_test.full_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(full_he, ML=True))
                if snakemake.params.Hom:
                    hom_ml, hom_ml_p = ctp_test.hom_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                            par=util.generate_HE_initial(hom_he, ML=True))
                else:
                    hom_ml, hom_ml_p = free_ml, free_ml_p
                if snakemake.params.IID:
                    iid_ml, iid_ml_p = ctp_test.iid_ML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                            par=util.generate_HE_initial(iid_he, ML=True))
                else:
                    iid_ml, iid_ml_p = free_ml, free_ml_p

            out['ml'] = {'hom': hom_ml, 'iid': iid_ml, 'free': free_ml, 'full': full_ml, 
                    'wald':{'hom':hom_ml_p, 'iid':iid_ml_p, 'free':free_ml_p}}

            # LRT
            C = np.loadtxt(y_f).shape[1]
            #hom_null_lrt = mystats.lrt(out['ml']['hom']['l'], out['ml']['null']['l'], 1)
            iid_hom_lrt = mystats.lrt(out['ml']['iid']['l'], out['ml']['hom']['l'], 1)
            free_hom_lrt = mystats.lrt(out['ml']['free']['l'], out['ml']['hom']['l'], C)
            free_iid_lrt = mystats.lrt(out['ml']['free']['l'], out['ml']['iid']['l'], C-1)
            full_hom_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['hom']['l'], C*(C+1)//2-1)
            full_iid_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['iid']['l'], C*(C+1)//2-2)
            full_free_lrt = mystats.lrt(out['ml']['full']['l'], out['ml']['free']['l'], C*(C+1)//2-C-1)

            out['ml']['lrt'] = {'iid_hom':iid_hom_lrt, 'free_hom':free_hom_lrt,
                    'free_iid':free_iid_lrt, 'full_hom':full_hom_lrt, 'full_iid':full_iid_lrt,
                    'full_free':full_free_lrt}

        # REML
        if snakemake.params.REML:
            if not snakemake.params.HE_as_initial:
                if 'Free_reml_jk' in snakemake.params.keys():
                    free_reml, free_reml_p = ctp_test.free_REML(y_f, P_f, ctnu_f, nu_f, 
                            fixed_covars_d, random_covars_d, nrep=5, jack_knife=snakemake.params.Free_reml_jk)
                else:
                    free_reml, free_reml_p = ctp_test.free_REML(y_f, P_f, ctnu_f, nu_f, 
                            fixed_covars_d, random_covars_d, nrep=5)
                full_reml = ctp_test.full_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)

                if snakemake.params.Hom:
                    hom_reml, hom_reml_p = ctp_test.hom_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                else:
                    hom_reml, hom_reml_p = free_reml, free_reml_p
                if snakemake.params.IID:
                    iid_reml, iid_reml_p = ctp_test.iid_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d)
                else:
                    iid_reml, iid_reml_p = free_reml, free_reml_p
            else:
                free_reml, free_reml_p = ctp_test.free_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(free_he, REML=True))
                full_reml = ctp_test.full_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                        par=util.generate_HE_initial(full_he, REML=True))

                if snakemake.params.Hom:
                    hom_reml, hom_reml_p = ctp_test.hom_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                            par=util.generate_HE_initial(hom_he, REML=True))
                else:
                    hom_reml, hom_reml_p = free_reml, free_reml_p
                if snakemake.params.IID:
                    iid_reml, iid_reml_p = ctp_test.iid_REML(y_f, P_f, ctnu_f, nu_f, fixed_covars_d, random_covars_d,
                            par=util.generate_HE_initial(iid_he, REML=True))
                else:
                    iid_reml, iid_reml_p = free_reml, free_reml_p

            out['reml'] = {'hom':hom_reml, 'iid':iid_reml, 'free':free_reml, 'full':full_reml, 
                    'wald':{'hom':hom_reml_p, 'iid':iid_reml_p, 'free':free_reml_p}}

            ## LRT
            C = np.loadtxt(y_f).shape[1]
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
        outs.append( out_f )
    
    #sys.exit('END')
    with open(output.out, 'w') as f:
        f.write('\n'.join(outs))

    print('Finished')

if __name__ == '__main__':
    main()

