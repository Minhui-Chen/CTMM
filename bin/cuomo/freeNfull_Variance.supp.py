import os, math, re
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def cut_data(data, l, r):
    data[data > r] = r
    data[data < l] = l
    return data

def ct_cor(V):
    C = V.shape[0]
    cor = []
    for i in range(C-1):
        for j in range(i+1, C):
            cor.append( V[i,j]/math.sqrt(V[i,i] * V[j,j]) )
    return( cor )
    
def main():
    # read
    op = np.load(snakemake.input.op, allow_pickle=True).item()
    ctp = np.load(snakemake.input.ctp, allow_pickle=True).item()

    #
    methods = ['ml', 'reml', 'he']
    V = op['reml']['free']['V']
    n = V.shape[0]
    C = V.shape[1]
    CTs = [f'CT{i}' for i in range(1,C+1)]
    CT_pairs = [f'CT{i}-CT{j}' for i in range(1,C) for j in range(i+1,C+1)]

    ## Free model
    op_free_datas = []
    ctp_free_datas = []
    op_full_datas = []
    ctp_full_datas = []
    for method in methods:
        op_free_data = pd.DataFrame()
        ctp_free_data = pd.DataFrame()
        # Free model
        ## hom2
        op_free_data['hom2'] = op[method]['free']['hom2']
        ctp_free_data['hom2'] = ctp[method]['free']['hom2']
        ## V
        V = [np.diag(x) for x in op[method]['free']['V']]
        V = pd.DataFrame(V, columns=[f'V-{ct}' for ct in CTs])
        op_free_data = pd.concat( (op_free_data, V), axis=1)
        V = [np.diag(x) for x in ctp[method]['free']['V']]
        V = pd.DataFrame(V, columns=[f'V-{ct}' for ct in CTs])
        ctp_free_data = pd.concat( (ctp_free_data, V), axis=1)

        for ct in CTs:
            op_free_data[f'V-{ct}-hom'] = op_free_data[f'V-{ct}']+op_free_data['hom2']
            ctp_free_data[f'V-{ct}-hom'] = ctp_free_data[f'V-{ct}']+ctp_free_data['hom2']

        op_free_datas.append( cut_data(op_free_data[['hom2']+[f'V-{ct}' for ct in CTs]].copy(), -2, 2) )
        ctp_free_datas.append( cut_data(ctp_free_data[['hom2']+[f'V-{ct}' for ct in CTs]].copy(), -0.5, 2) )

        #data = cut_data(op_data[[f'V-{ct}' for ct in CTs]].copy())
        #sns.violinplot(data=data, ax=axes[0,0], cut=0, color=color, orient='h')
        #data = cut_data(ctp_data[[f'V-{ct}' for ct in CTs]].copy())
        #sns.violinplot(data=data, ax=axes[1,0], cut=0, color=color, orient='h')

        #data = cut_data(op_data[[f'V-{ct}-hom' for ct in CTs]+['hom2']].copy())
        #sns.violinplot(data=data, ax=axes[0], cut=0, color=color, orient='h')
        #axes[0].set_ylabel('op')
        #ctp_free_data = cut_data(ctp_free_data[['hom2']+[f'V-{ct}' for ct in CTs]].copy(), -10, 1.5)

        ## Full model
        V = op[method]['full']['V']
        op_cor = [ct_cor(x) for x in V if np.all(np.diag(x) > 0)]
        print( len(V) - len(op_cor) )
        op_cor = pd.DataFrame(op_cor, columns=CT_pairs)
        op_cor = cut_data(op_cor, -2, 2) if method != 'he' else cut_data(op_cor, -5, 5)
        op_full_data = pd.melt(op_cor)
        op_full_datas.append( op_full_data )

        V = ctp[method]['full']['V']
        ctp_cor = [ct_cor(x) for x in V if np.all(np.diag(x) > 0)]
        print( len(V) - len(ctp_cor) )
        ctp_cor = pd.DataFrame(ctp_cor, columns=CT_pairs)
        ctp_cor = cut_data(ctp_cor, -1.5, 1.5) 
        ctp_full_data = pd.melt(ctp_cor)
        ctp_full_datas.append( ctp_full_data )

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

    for free_datas, full_datas, png in zip([op_free_datas, ctp_free_datas],
            [op_full_datas, ctp_full_datas],
            [snakemake.output.op, snakemake.output.ctp]):
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), dpi=600, sharex='col')
        #fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), dpi=600, sharex='col', sharey=True)
        
        for i, method, free_data, full_data in zip(range(len(methods)), methods, free_datas, full_datas):
            sns.violinplot(data=free_data, ax=axes[i,0], cut=0, color=color)
            axes[i,0].axhline(0, ls='--', color='0.8', zorder=0)
            axes[i,0].set_xlabel('')
            axes[i,0].set_ylabel('Variance in Free model', fontsize=12)
            axes[i,0].set_title(method.upper())

            sns.violinplot( x='variable', y='value', data=full_data, ax=axes[i,1], cut=0, palette=my_pal )
            axes[i,1].axhline(0, ls='--', color='0.8', zorder=0)
            axes[i,1].set_xlabel('')
            axes[i,1].set_ylabel( 'Correlation between CTs in Full model', fontsize=12, labelpad=8 )
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

        fig.tight_layout(pad=2, w_pad=3)
        fig.savefig(png)

if __name__ == '__main__':
    main()
