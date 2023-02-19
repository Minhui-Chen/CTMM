import sys, math, time
import numpy as np
import pandas as pd
from scipy import stats

def check_singular(X, eigtol=1e16):
    eval, evec = np.linalg.eig(X)
    #if ( ( max(eval) / min(eval)+1e-99 ) > eigtol ) or ( min(eval) < 0 ):
    if min(eval) < 0:
        print( eval )
        #print( evec[:, np.argmin(eval)] )
        return True # singular
    else:
        return False

def asymptotic_dispersion_matrix(X, Z, V):
    '''
    Asymptotic dispersion matrix from expected Information matrix (i.e Fisher information)
    Refer to 6.3 Asymptotic dispersion matrix in Variance Components
    X : ndarray
        design matrix for fixed effects
    Z : list
        list of design matrixs for random effects
    V : ndarray
        covariance matrix of y
    D : ndarray
        dispersion matrix
    '''
    if check_singular(V):
        #print(V)
        sys.exit('Singular V in asymptotic dispersion matrix!\n')
    #eval, evec = np.linalg.eig(V)
    #if min(eval) < 0 or (max(eval) / (min(eval) + 1e-99)) > 1e8:
        #sys.exit(f'Singular V in asymptotic_dispersion_matrix: {max(eval)} {min(eval)}\n')
    #V_inv = evec @ np.diag(1/eval) @ evec.T
    V_inv = np.linalg.inv(V)
    # information matrix elements
    minus_E_l_beta_beta = X.T @ V_inv @ X
    #print(V_inv[V_inv < 0 ])
    #print(X.T @ V_inv)
    #print(minus_E_l_beta_beta)
    minus_E_l_sigma_sigma = []
    for i in range(len(Z)):
        tmp = []
        for j in range(len(Z)):
            tmp.append(np.trace(V_inv @ Z[i] @ Z[i].T @ V_inv @ Z[j] @ Z[j].T) / 2)
        minus_E_l_sigma_sigma.append(tmp)
    minus_E_l_sigma_sigma = np.array(minus_E_l_sigma_sigma)

    # asymptotic dispersion matrix
    if check_singular(minus_E_l_beta_beta):
        print(minus_E_l_beta_beta)
        sys.exit('Singular minus_E_l_beta_beta!\n')
    minus_E_l_beta_beta_inv = np.linalg.inv(minus_E_l_beta_beta)
    #print(minus_E_l_beta_beta)
    #print(minus_E_l_beta_beta_inv)
    #if not np.allclose( minus_E_l_beta_beta @ minus_E_l_beta_beta_inv, np.identity(minus_E_l_beta_beta.shape[0]) ):
    #    print( minus_E_l_beta_beta @ minus_E_l_beta_beta_inv )
    #    sys.exit('Singular matrix?\n')
    #print( np.linalg.eigvals(minus_E_l_beta_beta) )
    #print(minus_E_l_beta_beta @ minus_E_l_beta_beta_inv @ minus_E_l_beta_beta)
    #print(minus_E_l_beta_beta @ np.linalg.pinv(minus_E_l_beta_beta) @ minus_E_l_beta_beta)
    if check_singular( minus_E_l_sigma_sigma ):
        print(minus_E_l_sigma_sigma)
        sys.exit('Singular minus_E_l_sigma_sigma!\n')
    minus_E_l_sigma_sigma_inv = np.linalg.inv(minus_E_l_sigma_sigma)
    D = np.block([
        [minus_E_l_beta_beta_inv, np.zeros((minus_E_l_beta_beta_inv.shape[0], minus_E_l_sigma_sigma_inv.shape[1]))],
        [np.zeros((minus_E_l_sigma_sigma_inv.shape[0], minus_E_l_beta_beta_inv.shape[1])), minus_E_l_sigma_sigma_inv]
        ])

    #print('Asymptotic dispersion matrix:\n', D)
    return D

def reml_asymptotic_dispersion_matrix(X, Z, V):
    '''
    Asymptotic dispersion matrix for REML from expected Information matrix (i.e Fisher information)
    Refer to 6.6 Restricted maximum likelihood
    X : ndarray
        design matrix for fixed effects
    Z : list
        list of design matrixs for random effects
    V : ndarray
        covariance matrix of y
    D : ndarray
        dispersion matrix
    '''
    if check_singular(V):
        print(V)
        sys.exit('Singular V in REML asymptotic dispersion matrix!\n')
    V_inv = np.linalg.inv(V)

    A = X.T @ V_inv @ X
    if check_singular(A):
        print(A)
        sys.exit('Singular A!\n')

    P = V_inv - V_inv @ X @ np.linalg.inv(A) @ X.T @ V_inv 
    # information matrix elements
    I = []
    for i in range(len(Z)):
        tmp = []
        for j in range(len(Z)):
            sesq = np.sum( (Z[i].T @  P @ Z[j])**2 )
            tmp.append(sesq)
        I.append(tmp)
    I = np.array(I)

    # asymptotic dispersion matrix
    if check_singular(I):
        print(I)
        sys.exit('Singular I!\n')
    D = 2 * np.linalg.inv(I)

    return D

def wald_test(estimate, expectation, var, df, two_sided=False):
    '''
    estimate : scalar
        ML estimate of parameter
    expectation : scalar
        expectation of parameter
    var : scalar
        variance of parameter normal distribution
    df : scalar
        degree of freedom
    two_sided : boolen
        two-sided or one-sided test (estiamte > expectation)
    '''
    if var <= 0:
        sys.exit(f'Non-positive variance: {var}\n')
    W = (estimate-expectation) / math.sqrt(var)

    if two_sided:
        return 2 * stats.t.cdf(-abs(W), df=df)
    else:
        return stats.t.sf(W, df=df)

def mvwald_test(estimate, expectation, var, eigtol=1e8, Ftest=True, n=None, P=None, df=None):
    '''
    estimate : array of scalar
        ML estimates of parameters
    expectation : array of scalar
        expectation of parameters
    var : ndarray of scalar
        variance-covariance matrix of parameter normal distribution
    Ftest: bool
        by default False, chi squire test is used (refer to Wiki Wald test - test on multiple parameters)
    n: scalar (for Ftest)
        sample size
    P: scalar (for Ftest)
        number of estimated parameters
    '''
    # chi square test
    if check_singular(var, eigtol):
        print(var)
        sys.exit('Singular matrix in mvwald!\n')
        #return float('nan')
    W2 = (estimate-expectation) @ np.linalg.inv(var) @ (estimate-expectation)
    p = stats.chi2.sf(W2, df=len(estimate))
    
    # F test
    if Ftest:
        F = W2/len(estimate)
        if not df:
            df = n-P
        p = stats.f.sf(F, len(estimate), df)

    return p

if __name__ == '__main__':
    pass
