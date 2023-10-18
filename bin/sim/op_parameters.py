import math
import numpy as np
import pandas as pd

def main():

    # cell type proportion P
    alpha = np.array(snakemake.wildcards.a.split('_')).astype('float')
    C = len(alpha)
    # dirichlet distribution's moments
    a0 = np.sum(alpha)
    pi = alpha / a0
    print(pi)
    np.savetxt(snakemake.output.pi, pi)

    var = pi * (1 - pi) / (a0 + 1)
    s_fun = lambda alpha, a0, i, j: -1 * pi[i] * pi[j] / (a0 + 1)
    s = []
    for i in range(C):
        s_e = []
        for j in range(C):
            if i == j:
                s_e.append(var[i])
            else:
                s_e.append(s_fun(alpha, a0, i, j))
        s.append(s_e)
    s = np.array(s)
    np.savetxt(snakemake.output.s, s)

    # beta
    beta = np.array(snakemake.wildcards.beta.split('_')).astype('float')
    vc1 = float(snakemake.wildcards.vc.split('_')[1]) # variance explained by \beta^T S \beta

    ## calculate beta based on \beta^T S \beta = 0.25
    ## \beta^2 [1 1/2 1/4 1/8] cov [1 1/2 1/4 1/8]^T = 0.25
    ## \beta^2 = 0.25 / ([1 1/2 1/4 1/8] cov [1 1/2 1/4 1/8]^T)
    ##x = np.array([1*(params.ratio**(i)) for i in range(celltype_no)])
    print(vc1)
    print(beta @ s @ beta)
    scale = math.sqrt(vc1 / (beta @ s @ beta))
    beta = beta * scale

    np.savetxt(snakemake.output.beta, beta, delimiter='\t')

    # V
    vc2 = float( snakemake.wildcards.vc.split('_')[2]) #variance explained by interaction
    V_diag = snakemake.wildcards.V_diag
    if V_diag == '0':
        V_diag = np.zeros(C)
    else:
        V_diag = np.array(V_diag.split('_')).astype('float')
    V_tril = snakemake.wildcards.V_tril
    if V_tril == '0':
        V_tril = np.zeros(C * (C - 1) // 2)
    else:
        V_tril = np.array(V_tril.split('_')).astype('float')

    ## calculate V
    V = np.diag(V_diag)
    V[np.tril_indices(C, k=-1)] = V_tril
    for i in range(1, C):
        for j in range(i):
            V[i,j] = V[i,j] * math.sqrt(V[i,i] * V[j,j])
    V = V + np.tril(V, k=-1).T

    ## \tr{V S} + pi^T V pi = 0.25
    if np.all(V == 0):
        scale = 0
    else:
        scale = vc2 / (np.trace(V @ s) + pi @ V @ pi)
    V = V * scale

    np.savetxt(snakemake.output.V, V, delimiter='\t')

if __name__ == '__main__':
    main()
