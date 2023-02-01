import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plot, plot_help

def free_cor( out, hom_max=None, hom_min=None, with_hom2=False):
    cor_dict = {}
    for m in ['ml', 'reml', 'he']:
        if m not in out.keys():
            continue
        hom2 = out[m]['free']['hom2']
        V = out[m]['free']['V']
        n = V.shape[0]
        C = V.shape[1]
        CTs = ['CT'+str(i+1) for i in range(C)]

        # format data
        data = []
        for i in range(len(V)):
            if with_hom2:
                data.append( [hom2[i]] + list(hom2[i] + np.diag(V[i])) )
            else:
                data.append( [hom2[i]] + list(np.diag(V[i])) )
        data = pd.DataFrame(data, columns=['hom2']+CTs)

        #
        if hom_max:
            data = data.loc[data['hom2'] < hom_max]
        if hom_min:
            data = data.loc[data['hom2'] > hom_min]
        data = data[CTs]

        # cal correlation
        cors = []
        ps = []
        anns = []
        for i in range(C):
            cor = []
            p = []
            ann = []
            for j in range(C):
                if j < i:
                    cor_, p_ = stats.pearsonr(data['CT'+str(i+1)], data['CT'+str(j+1)])
                    cor.append(cor_)
                    p.append(p_)
                    ann.append(f'r={cor_:.2f}\np={p_:.2g}')
                else:
                    cor.append(np.nan)
                    p.append(np.nan)
                    ann.append('')
            cors.append(cor)
            ps.append(p)
            anns.append(ann)
        cors = np.array(cors)
        ps = np.array(ps)
        anns = np.array(anns)
        #print(cors)
        #print(ps)
        #cor_dict[m] = [cors, ps]
        cor_dict[m] = [cors, ps, anns]

    return( cor_dict )

def full_cor( out ):
    cor_dict = {}
    for m in ['ml', 'reml', 'he']:
        V = out[m]['full']['V']
        nu = out[m]['full']['nu']
        n, C = V.shape[:2]
        CTs = ['CT'+str(i+1) for i in range(C)]

        # cal correlation
        cor = []
        for V_, nu_ in zip(V, nu):
            if np.any( (np.diag(V_) + nu_) <= 0 ):
                cor_ = np.zeros_like( V_ ) * np.nan
            else:
                cor_ = V_ / (np.sqrt( np.diag(V_) + nu_ )).reshape(-1,1)
                cor_ = cor_ / np.sqrt( np.diag(V_) + nu_ )
                if np.any( np.isnan( cor_ ) ):
                    print( cor_ )
                    sys.exit('nan in cor!\n')
            cor.append( cor_ )
        cor_dict[m] = cor

    return( cor_dict )

def main():
    # par
    
    # read
    out = np.load(snakemake.input.out, allow_pickle=True).item()
    #ms = [x for x in ['ml', 'reml', 'he'] if x in out.keys()]

    fig, ax = plt.subplots(figsize=(6,4),dpi=600)

#    # Free model
#    ## correlation
#    out_cor = free_cor( out, hom_max=5 )
#    base_cor = free_cor( base, hom_max=5 )
#
#    out_cor_withhom2 = free_cor( out, hom_max=5, with_hom2=True )
#    base_cor_withhom2 = free_cor( base, hom_max=5, with_hom2=True )
#
#    ## colorbar min max
#    vmin = 0
#    vmax = 1
#    #cors = [[out_cor[m][0], base_cor[m][0]] for m in ['ml', 'reml', 'he']]
#    #if  np.nanmin( cors ) < -0.5:
#    #    vmin = -1
#    #elif np.nanmin( cors ) < 0:
#    #    vmin = -0.5
#
#    for k in range(len(ms)):
#        m = ms[k]
#        if m not in out.keys():
#            continue
#        V = out[m]['free']['V']
#        n, C = V.shape[:2]
#        CTs = ['CT'+str(i+1) for i in range(C)]
#
#        if snakemake.input.out == snakemake.input.base[0]:
#            #sns.heatmap(out_cor[m][0].T, annot=out_cor[m][2].T, xticklabels=CTs, yticklabels=CTs, 
#            #        cmap="YlGnBu", ax=axes[0,k], vmin=vmin, vmax=vmax, fmt='s')
#            sns.heatmap(out_cor_withhom2[m][0].T, annot=out_cor_withhom2[m][2].T, 
#                    xticklabels=CTs, yticklabels=CTs, cmap="YlGnBu", ax=axes[0,k], 
#                    vmin=vmin, vmax=vmax, fmt='s')
#        else:
#            #data = np.where( ~np.isnan(out_cor[m][0]), out_cor[m][0], base_cor[m][0].T )
#            #ann = np.where( ~np.isnan(out_cor[m][0]), out_cor[m][2], base_cor[m][2].T )
#            #sns.heatmap(data, annot=ann, xticklabels=CTs, yticklabels=CTs, cmap="YlGnBu", 
#            #        ax=axes[0,k], vmin=vmin, vmax=vmax, fmt='s')
#            data_withhom2 = np.where( ~np.isnan(out_cor_withhom2[m][0]), out_cor_withhom2[m][0], 
#                    base_cor_withhom2[m][0].T )
#            ann_withhom2 = np.where( ~np.isnan(out_cor_withhom2[m][0]), out_cor_withhom2[m][2], 
#                    base_cor_withhom2[m][2].T )
#            sns.heatmap(data_withhom2, annot=ann_withhom2, xticklabels=CTs, yticklabels=CTs, 
#                    cmap="YlGnBu", ax=axes[0,k], vmin=vmin, vmax=vmax, fmt='s')
#
#        axes[0,k].set_title( m.upper() )
#    #axes[0,0].set_ylabel( 'Cor between CTs in Free model \n(without hom2)' )
#    axes[0,0].set_ylabel( 'Cor between CTs in Free model' )

    # Full model
    ## correlation
    #out_cor = full_cor( out )
    #base_cor = full_cor( base )

    m = 'reml'
    V = out[m]['full']['V']
    #nu = out[m]['full']['nu']
    n, C = V.shape[:2]
    CT_pairs = [f'CT{i}-CT{j}' for i in range(1,C) for j in range(i+1,C+1)]
    print(CT_pairs)

    # covariance
    ## format data
    print(out[m]['full']['V'][0])
    print(out[m]['full']['V'][0][np.triu_indices(C,1)])
    out_cov = [x[np.triu_indices(C,1)] for x in out[m]['full']['V']]
    out_cov = pd.DataFrame(out_cov, columns=CT_pairs)

    data = pd.melt(out_cov)
    threshold = [np.mean(data['value']) - 3 * np.std(data['value']), np.mean(data['value']) + 3 * np.std(data['value'])]
    data.loc[data['value'] > threshold[1], 'value'] = threshold[1]
    data.loc[data['value'] < threshold[0], 'value'] = threshold[0]
    my_pal = {}
    for pair in CT_pairs:
        ct1, ct2 = int(pair.split('-')[0][2]), int(pair.split('-')[1][2])
        if ct2 - ct1 != 1:
            my_pal[pair] = 'lightblue'
        else:
            my_pal[pair] = plot_help.mycolors()[0]
    sns.violinplot( x='variable', y='value', data=data, ax=ax, cut=0, palette=my_pal )

    ax.set_ylabel( 'Covariance between CTs in Full model' )
    ax.set_xlabel('')

    fig.savefig(snakemake.output.png)

if __name__ == '__main__':
    main()
