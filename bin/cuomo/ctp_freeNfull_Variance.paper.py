import os, math, re, sys
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # read
    op = np.load(snakemake.input.op, allow_pickle=True).item()
    ctp = np.load(snakemake.input.ctp, allow_pickle=True).item()

    #
    method = 'reml'
    V = op['reml']['free']['V']
    n = V.shape[0]
    C = V.shape[1]
    CTs = [f'CT{i}' for i in range(1,C+1)]

    op_data = pd.DataFrame()
    ctp_data = pd.DataFrame()
    # Free model
    ## hom2
    op_data['hom2'] = op[method]['free']['hom2']
    ctp_data['hom2'] = ctp[method]['free']['hom2']
    ## V
    V = [np.diag(x) for x in op[method]['free']['V']]
    V = pd.DataFrame(V, columns=[f'V-{ct}' for ct in CTs])
    op_data = pd.concat( (op_data, V), axis=1)
    V = [np.diag(x) for x in ctp[method]['free']['V']]
    V = pd.DataFrame(V, columns=[f'V-{ct}' for ct in CTs])
    ctp_data = pd.concat( (ctp_data, V), axis=1)

    for ct in CTs:
        op_data[f'V-{ct}-hom'] = op_data[f'V-{ct}']+op_data['hom2']
        ctp_data[f'V-{ct}-hom'] = ctp_data[f'V-{ct}']+ctp_data['hom2']

    # plot
    mpl.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), dpi=600)
    color = sns.color_palette()[0]
    
    ## Free model
    data = ctp_data[['hom2']+[f'V-{ct}' for ct in CTs]]
    data.to_csv(snakemake.output.dataA, sep='\t', index=False)
    data = data.clip(upper=2)
    sns.violinplot(data=data, ax=axes[0], cut=0, color=color)
    axes[0].axhline(0, ls='--', color='0.8', zorder=0)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Variance in Free model', fontsize=12)

    # change ticks in Free model
    fig.canvas.draw_idle()
    plt.sca(axes[0])
    locs, labels = plt.xticks()
    print(locs)
    for i in range(len(labels)):
        label = labels[i].get_text()
        print(label)
        if '-' in label:
            ct = label.split('-')[1]
            labels[i] = r'$V_{%s}$'%(ct)
        elif label == 'hom2':
            labels[i] = r'$\sigma_\alpha^2$'
    plt.xticks(locs, labels)

    ## Full model
    m = 'reml'
    V = ctp[m]['full']['V']
    n, C = V.shape[:2]
    CT_pairs = [f'CT{i}-CT{j}' for i in range(1,C) for j in range(i+1,C+1)]
    print(CT_pairs)
    # plot cov or cor
    show = 'cor'
    if show == 'cov':
        ctp_cov = [x[np.triu_indices(C,1)] for x in V]
        ctp_cov = pd.DataFrame(ctp_cov, columns=CT_pairs)
        data = pd.melt( ctp_cov )

        threshold = [-1.0, 1.5]
        data.loc[data['value'] > threshold[1], 'value'] = threshold[1]
        data.loc[data['value'] < threshold[0], 'value'] = threshold[0]

        ylab = 'Covariance between CTs in Full model'
    else:
        ctp_cor = []
        k = 0
        for x in V:
            if np.any(np.diag(x) < 0):
                k += 1
                continue
            cor = []
            for i in range(C-1):
                for j in range(i+1,C):
                    try:
                        cor.append( x[i,j]/math.sqrt(x[i,i] * x[j,j]) )
                    except:
                        print( x )
                        sys.exit()
            ctp_cor.append( cor )
        print( k )
        ctp_cor = pd.DataFrame(ctp_cor, columns=CT_pairs)
        print(ctp_cor.median(axis=0))
        data = pd.melt( ctp_cor )

        threshold = [-1.5, 1.5]
        data.loc[data['value'] > threshold[1], 'value'] = threshold[1]
        data.loc[data['value'] < threshold[0], 'value'] = threshold[0]

        ylab = 'Correlation between CTs in Full model'
    #threshold = [np.mean(data['value']) - 3*np.std(data['value']), np.mean(data['value']) + 3*np.std(data['value'])]
    my_pal = {}
    for pair in CT_pairs:
        ct1, ct2 = int(pair.split('-')[0][2]), int(pair.split('-')[1][2])
        if ct2 - ct1 != 1:
            my_pal[pair] = 'lightblue'
        else:
            my_pal[pair] = sns.color_palette('muted')[0]
    data.to_csv(snakemake.output.dataB, sep='\t', index=False)
    sns.violinplot( x='variable', y='value', data=data, ax=axes[1], cut=0, palette=my_pal )
    axes[1].axhline(0, ls='--', color='0.9', zorder=0)
    axes[1].set_ylabel( ylab, fontsize=12 )
    axes[1].set_xlabel('')
    # change ticks in Full model
    fig.canvas.draw_idle()
    plt.sca(axes[1])
    locs, labels = plt.xticks()
    print(locs)
    for i in range(len(labels)):
        label = labels[i].get_text()
        print(label)
        ct1, ct2 = label.split('-')
        labels[i] = r'$V_{({%s},{%s})}$'%(ct1, ct2)
    plt.xticks(locs, labels)

   
    axes[0].text(-0.05, 1.05, '(A)', fontsize=16, transform=axes[0].transAxes)
    axes[1].text(-0.05, 1.05, '(B)', fontsize=16, transform=axes[1].transAxes)
    fig.tight_layout(pad=2, w_pad=3)
    fig.savefig(snakemake.output.png)

if __name__ == '__main__':
    main()
