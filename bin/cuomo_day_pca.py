import math
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # par

    # read
    y = np.loadtxt(snakemake.input.y) # ind * gene
    ## scale
    y = (y-np.mean(y,axis=0))/np.std(y,axis=0)
    #P = pd.read_table(snakemake.input.P) # donor * day
    #P = P.sort_values(by='donor').reset_index(drop=True)
    cty = pd.read_table(open(snakemake.input.imputed_ct_y).readline().strip())

    # svd
    u, s, vh = np.linalg.svd(y)
    s2 = s * s
    s2 = s2 / s2.sum()
    ## pca eval plot
    fig, ax = plt.subplots()
    ax.scatter(range(1,21), s2[:20], ls='-', c='0.8')
    ax.scatter(range(1,21), s2[:20])
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlabel('PC')
    ax.set_ylabel('Percentage of variance explained')
    fig.savefig(snakemake.output.png)

    # save
    np.savetxt(snakemake.output.evec, vh.transpose())
    np.savetxt(snakemake.output.eval, s2)
    pca = y @ (vh.transpose())
    pca = pd.DataFrame(pca)
    pca.columns = ['PC'+str(i+1) for i in range(pca.shape[1])]
    #if pca.shape[0] != P.shape[0]:
    if pca.shape[0] != len(np.unique(cty['donor'])):
        sys.exit('Not matching!\n')
    #pca['donor'] = np.array(P['donor'])
    pca['donor'] = np.unique(cty['donor'])

    pca.to_csv(snakemake.output.pca, sep='\t', index=False)

if __name__ == '__main__':
    main()
