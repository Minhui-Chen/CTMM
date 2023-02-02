import os, math, re
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # read
    ong = np.load(snakemake.input.ong, allow_pickle=True).item()
    ctng = np.load(snakemake.input.ctng, allow_pickle=True).item()

    #
    methods = ['ml', 'reml', 'he']
    V = ong['reml']['free']['V']
    n = V.shape[0]
    C = V.shape[1]
    CTs = [f'CT{i}' for i in range(1,C+1)]
    CT_pairs = [f'CT{i}-CT{j}' for i in range(1,C) for j in range(i+1,C+1)]

    ## Free model
    def cut_data(data, l, r):
        #mean = np.mean(np.array(data))
        #std = np.std(np.array(data))
        #l = mean - 2.5*std
        #r = mean + 2.5*std
        #r = 1.5
        data[data > r] = r
        data[data < l] = l
        return data
    
    ong_free_datas = []
    ctng_free_datas = []
    ong_full_datas = []
    ctng_full_datas = []
    for method in methods:
        ong_free_data = pd.DataFrame()
        ctng_free_data = pd.DataFrame()
        # Free model
        ## hom2
        ong_free_data['hom2'] = ong[method]['free']['hom2']
        ctng_free_data['hom2'] = ctng[method]['free']['hom2']
        ## V
        V = [np.diag(x) for x in ong[method]['free']['V']]
        V = pd.DataFrame(V, columns=[f'V-{ct}' for ct in CTs])
        ong_free_data = pd.concat( (ong_free_data, V), axis=1)
        V = [np.diag(x) for x in ctng[method]['free']['V']]
        V = pd.DataFrame(V, columns=[f'V-{ct}' for ct in CTs])
        ctng_free_data = pd.concat( (ctng_free_data, V), axis=1)

        for ct in CTs:
            ong_free_data[f'V-{ct}-hom'] = ong_free_data[f'V-{ct}']+ong_free_data['hom2']
            ctng_free_data[f'V-{ct}-hom'] = ctng_free_data[f'V-{ct}']+ctng_free_data['hom2']

        ong_free_datas.append( cut_data(ong_free_data[['hom2']+[f'V-{ct}' for ct in CTs]].copy(), -2, 2) )
        ctng_free_datas.append( cut_data(ctng_free_data[['hom2']+[f'V-{ct}' for ct in CTs]].copy(), -0.5, 1) )

        #data = cut_data(ong_data[[f'V-{ct}' for ct in CTs]].copy())
        #sns.violinplot(data=data, ax=axes[0,0], cut=0, color=color, orient='h')
        #data = cut_data(ctng_data[[f'V-{ct}' for ct in CTs]].copy())
        #sns.violinplot(data=data, ax=axes[1,0], cut=0, color=color, orient='h')

        #data = cut_data(ong_data[[f'V-{ct}-hom' for ct in CTs]+['hom2']].copy())
        #sns.violinplot(data=data, ax=axes[0], cut=0, color=color, orient='h')
        #axes[0].set_ylabel('ONG')
        #ctng_free_data = cut_data(ctng_free_data[['hom2']+[f'V-{ct}' for ct in CTs]].copy(), -10, 1.5)

        ## Full model
        V = ong[method]['full']['V']
        ong_cov = [x[np.triu_indices(C,1)] for x in V]
        ong_cov = pd.DataFrame(ong_cov, columns=CT_pairs)
        ong_full_data = pd.melt(ong_cov)
        #threshold = [np.mean(data['value']) - 3*np.std(data['value']), np.mean(data['value']) + 3*np.std(data['value'])]
        if method != 'he':
            threshold = [-2, 2]
        else:
            threshold = [-5, 5]
        ong_full_data.loc[ong_full_data['value'] > threshold[1], 'value'] = threshold[1]
        ong_full_data.loc[ong_full_data['value'] < threshold[0], 'value'] = threshold[0]
        ong_full_datas.append( ong_full_data )

        V = ctng[method]['full']['V']
        ctng_cov = [x[np.triu_indices(C,1)] for x in V]
        ctng_cov = pd.DataFrame(ctng_cov, columns=CT_pairs)
        ctng_full_data = pd.melt(ctng_cov)
        #threshold = [np.mean(data['value']) - 3*np.std(data['value']), np.mean(data['value']) + 3*np.std(data['value'])]
        threshold = [-0.5, 1]
        ctng_full_data.loc[ctng_full_data['value'] > threshold[1], 'value'] = threshold[1]
        ctng_full_data.loc[ctng_full_data['value'] < threshold[0], 'value'] = threshold[0]
        ctng_full_datas.append( ctng_full_data )

    # plot
    my_pal = {}
    for pair in CT_pairs:
        ct1, ct2 = int(pair.split('-')[0][2]), int(pair.split('-')[1][2])
        if ct2 - ct1 != 1:
            my_pal[pair] = 'lightblue'
        else:
            my_pal[pair] = sns.color_palette('muted')[0]

    mpl.rcParams.update({'font.size': 11})
    color = sns.color_palette()[0]

    for free_datas, full_datas, png in zip([ong_free_datas, ctng_free_datas],
            [ong_full_datas, ctng_full_datas],
            [snakemake.output.ong, snakemake.output.ctng]):
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), dpi=600, sharex='col', sharey=True)
        
        for i, method, free_data, full_data in zip(range(len(methods)), methods, free_datas, full_datas):
            sns.violinplot(data=free_data, ax=axes[i,0], cut=0, color=color)
            axes[i,0].axhline(0, ls='--', color='0.8', zorder=0)
            axes[i,0].set_xlabel('')
            axes[i,0].set_ylabel('Variance in Free model', fontsize=12)
            axes[i,0].set_title(method.upper())

            sns.violinplot( x='variable', y='value', data=full_data, ax=axes[i,1], cut=0, palette=my_pal )
            axes[i,1].axhline(0, ls='--', color='0.8', zorder=0)
            axes[i,1].set_xlabel('')
            axes[i,1].set_ylabel( 'Covariance between CTs in Full model', fontsize=12, labelpad=8 )
            axes[i,1].set_title(method.upper())

        # change ticks in Free model
        fig.canvas.draw_idle()
        plt.sca(axes[2,0])
        locs, labels = plt.xticks()
        print(locs)
        for i in range(len(labels)):
            label = labels[i].get_text()
            print(label)
            if '-' in label:
                ct = label.split('-')[1]
                labels[i] = r'$V_{%s}$'%(ct)
                #labels[i] = r'$\sigma_\alpha^2 + V_{%s}$'%(ct)
            elif label == 'hom2':
                labels[i] = r'$\sigma_\alpha^2$'
        plt.xticks(locs, labels)

        # change ticks in Full model
        fig.canvas.draw_idle()
        plt.sca(axes[2,1])
        locs, labels = plt.xticks()
        print(locs)
        for i in range(len(labels)):
            label = labels[i].get_text()
            print(label)
            ct1, ct2 = label.split('-')
            labels[i] = r'$V_{({%s},{%s})}$'%(ct1, ct2)
        plt.xticks(locs, labels)

       
        #axes[0].text(-0.05, 1.05, '(A)', fontsize=16, transform=axes[0].transAxes)
        #axes[1].text(-0.05, 1.05, '(B)', fontsize=16, transform=axes[1].transAxes)
        fig.tight_layout(pad=2, w_pad=3)
        fig.savefig(png)

if __name__ == '__main__':
    main()
