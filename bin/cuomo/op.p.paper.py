import math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import draw

def main():
    # par

    #
    out = np.load(snakemake.input.out, allow_pickle=True).item()
    marker = ['NANOG','T','GATA6']

    fig, axes = plt.subplots(nrows=2, figsize=(6,8), dpi=600)
    for i, m in enumerate(['reml', 'he']):
        if m not in out.keys():
            continue
        print('\n'+m)
        genes = np.array([x.split('_')[1] for x in out['gene']])
        # free model 
        beta_ps = out[m]['wald']['free']['ct_beta']
        if m == 'reml':
            V_ps = out[m]['lrt']['free_hom']
        else:
            V_ps = out[m]['wald']['free']['V']

        # set missing p values in ct beta in HE to min
        #if m == 'he':
            #beta_ps[np.isnan(beta_ps)] = np.nanmin( beta_ps )

        # remove p = 0 
        V_ps = np.array(V_ps)
        beta_ps = np.array(beta_ps)
        if np.any(V_ps == 0):
            print('V')
            #print(V_ps[V_ps == 0].sum())
            print((V_ps == 0).sum())
        if np.any(beta_ps == 0):
            print('beta')
            #print(beta_ps[beta_ps <= 0].sum())
            print((beta_ps == 0).sum())


        print(len(V_ps))
        #if np.any(V_ps == 0) or np.any(beta_ps == 0):
        #    keep = ~((V_ps == 0) | (beta_ps == 0))
        #    beta_ps = beta_ps[keep]
         #   V_ps = V_ps[keep]
         #   genes = genes[keep]
        #print(len(V_ps))

        beta_ps = np.log10(beta_ps) * (-1)
        V_ps = np.log10(V_ps) * (-1)
        #if np.any(np.isnan(beta_ps_)):
            #print('beta')
            #print(beta_ps[np.isnan(beta_ps_)])
        #if np.any(np.isnan(V_ps)):
            #print('V')

        # check inf
        if np.any(np.isinf(V_ps)) or np.any(np.isinf(beta_ps)):
            sys.exit('still inf')
            #noninf = ~(np.isinf(V_ps) | np.isinf(beta_ps))
            #beta_ps = beta_ps[noninf]
            #V_ps = V_ps[noninf]
            #V2_ps = V2_ps[noninf]
            #genes = genes[noninf]

        data = pd.DataFrame({'beta':beta_ps, 'V':V_ps, 'gene':genes})

        ## free 
        draw.scatter(beta_ps, V_ps, s=5, heatscatter=True, linregress=False, ax=axes[i])

        axes[i].scatter(data.loc[data['gene'].isin(marker),'beta'], 
                data.loc[data['gene'].isin(marker),'V'], s=10, c='m')
        axes[i].axhline((-1)*math.log10(0.05/len(beta_ps)), color='0.8', ls='--', zorder=0)
        axes[i].axvline((-1)*math.log10(0.05/len(beta_ps)), color='0.8', ls='--', zorder=0)
        ## add arrow to three markers
        xlim = axes[i].get_xlim()
        ylim = axes[i].get_ylim()
        for index, row in data.loc[data['gene'].isin(marker)].iterrows():
            #print(row)
            if m == 'reml':
                if row['gene'] == 'GATA6':
                    shiftx = (xlim[1]-xlim[0])/5
                    shifty = (ylim[1]-ylim[0])/20
                elif row['gene'] == 'T':
                    shiftx = (-1)*(xlim[1]-xlim[0])/10
                    shifty = (ylim[1]-ylim[0])/20
                else:
                    shiftx = (xlim[1]-xlim[0])/5
                    shifty = (ylim[1]-ylim[0])/5
                axes[i].annotate(row['gene'], xy=(row['beta'],row['V']),
                        xytext=(row['beta']+shiftx, row['V']+shifty), 
                        arrowprops={'arrowstyle':'->'})
            elif m == 'he':
                if row['gene'] == 'GATA6':
                    shiftx = (xlim[1]-xlim[0])/20
                    shifty = (ylim[1]-ylim[0])/10
                elif row['gene'] == 'T':
                    shiftx = (xlim[1]-xlim[0])/20
                    shifty = (ylim[1]-ylim[0])/10
                else:
                    shiftx = (xlim[1]-xlim[0])/5
                    shifty = (ylim[1]-ylim[0])/20
                axes[i].annotate(row['gene'], xy=(row['beta'],row['V']),
                        xytext=(row['beta']+shiftx, row['V']+shifty), 
                        arrowprops={'arrowstyle':'->'})

    for ax in axes.flatten():
        ax.set_xlabel('$-log_{10}$ p(mean differentiation)', fontsize=14)
        ax.set_ylabel('$-log_{10}$ p(variance differentiation)', fontsize=14)
    axes[0].set_title('REML (LRT)')
    axes[1].set_title('HE')
    
    fig.tight_layout(h_pad=3)
    fig.savefig(snakemake.output.png)

if __name__ == '__main__':
    main()
