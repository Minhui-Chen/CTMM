import math
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # par

    # read
    op = pd.read_table(snakemake.input.op, index_col=0)
    y = op.to_numpy()
    ## scale
    y = (y-np.mean(y,axis=0))/np.std(y,axis=0)

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
    pca = pd.DataFrame(u, index=op.index)
    pca.columns = ['PC'+str(i+1) for i in range(pca.shape[1])]

    pca.to_csv(snakemake.output.pca, sep='\t')

if __name__ == '__main__':
    main()
