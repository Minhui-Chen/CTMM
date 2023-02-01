import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    #

    # read
    meta = pd.read_table(snakemake.input.meta, usecols=['donor', 'day', 'cell_name', 'experiment'])
    cts = list(np.unique(meta['day']))
    meta = meta.sort_values(by='donor')
    ## for donor with more than one experiments, pick the one with largest number of cells
    meta_ = meta.groupby(['donor', 'experiment']).count().reset_index().sort_values(by=['donor','day'])
    meta_ = meta_.drop_duplicates('donor', keep='last')
    meta = meta.merge(meta_, on=['donor', 'experiment'], suffixes=('', '_y'))
    meta = meta[['donor', 'day', 'cell_name', 'experiment']]

    meta.to_csv(snakemake.output.meta, sep='\t', index=False)

    # count number of cells for each donor and ct
    meta = meta.groupby(['donor', 'day']).count().reset_index()
    meta = meta.pivot(index='donor', columns='day', values='cell_name')
    meta['total'] = meta.sum(axis=1)
    for ct in cts:
        meta[ct] = meta[ct] / meta['total']

    # plot
    fig = plt.figure( figsize=(18,24) )
    
    meta = meta.sort_values(by='donor')
    plt.subplot2grid( (6,11), (0,0), colspan=5)
    plt.hist( meta['total'], bins=np.arange(0,meta['total'].max()+10,10) )
    plt.xticks(np.arange(0, meta['total'].max(), 100))
    plt.xlabel('cell number per donor')
    plt.ylabel('number of donor')

    plt.subplot2grid( (6,11), (0,6), colspan=5)
    sns.boxplot( data=meta[cts] )

    plt.subplot2grid( (6,11), (1,0), rowspan=5, colspan=1)
    sns.heatmap(data=meta[['total']].astype('int'), annot=True, cmap='YlGnBu', fmt='d')
    plt.xlabel('')

    plt.subplot2grid( (6,11), (1,2), rowspan=5, colspan=3)
    sns.heatmap(data=meta[cts], annot=True, cmap='YlGnBu', 
            xticklabels=[f'{ct}\naver:{meta[ct].mean():.2f}' for ct in cts])
    plt.xlabel('Cell type proportions')

    meta = meta.sort_values(by='total')
    plt.subplot2grid( (6,11), (1,6), rowspan=5, colspan=1)
    sns.heatmap(data=meta[['total']].astype('int'), annot=True, cmap='YlGnBu', fmt='d')
    plt.xlabel('')

    plt.subplot2grid( (6,11), (1,8), rowspan=5, colspan=3)
    sns.heatmap(data=meta[cts], annot=True, cmap='YlGnBu',
            xticklabels=[f'{ct}\naver:{meta[ct].mean():.2f}' for ct in cts])
    plt.xlabel('Cell type proportions')

    fig.savefig( snakemake.output.png )

if __name__ == '__main__':
    main()
