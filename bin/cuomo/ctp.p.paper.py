import math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import draw

out = np.load(snakemake.input.out, allow_pickle=True).item()
remlJK = np.load(snakemake.input.remlJK, allow_pickle=True).item()
marker = ['NANOG','T','GATA6']
candidate = ['POU5F1']

# p
reml_beta_ps = remlJK['reml']['wald']['free']['ct_beta']
reml_V_ps = remlJK['reml']['wald']['free']['V']
he_beta_ps = out['he']['wald']['free']['ct_beta']
he_V_ps = out['he']['wald']['free']['V']

reml_beta_ps = np.log10(reml_beta_ps) * (-1)
reml_V_ps = np.log10(reml_V_ps) * (-1)
he_beta_ps = np.log10(he_beta_ps) * (-1)
he_V_ps = np.log10(he_V_ps) * (-1)
genes = np.array([x.split('_')[1] for x in out['gene']])
remlJK_genes = np.array([x.split('_')[1] for x in remlJK['gene']])
print(len(genes), len(remlJK_genes))

remlJK_data = pd.DataFrame({'reml_beta':reml_beta_ps, 'reml_V':reml_V_ps, 'gene': remlJK_genes})
data = pd.DataFrame({'he_beta':he_beta_ps, 'he_V':he_V_ps,'gene':genes})
remlJK_threshold = (-1)*math.log10(0.05/remlJK_data.shape[0])
threshold = (-1)*math.log10(0.05/data.shape[0])

for method, tmp_data, tmp_threshold, png_f in zip(['reml','he'], [remlJK_data, data], 
        [remlJK_threshold, threshold], [snakemake.output.reml_p, snakemake.output.he_p]):

    # plot
    fig, ax = plt.subplots(dpi=600)

    ## heatscatter
    draw.scatter(tmp_data[method+'_beta'], tmp_data[method+'_V'], s=5, heatscatter=True, linregress=False, ax=ax)

    # add nonsignificant V in HE
    #tmp = data.loc[data['he_V'] < threshold]
    #ax.scatter(tmp['reml_beta'], tmp['reml_V'], s=5, c='0.8')

    # add three markers
    ax.scatter(tmp_data.loc[tmp_data['gene'].isin(marker), method+'_beta'],
            tmp_data.loc[tmp_data['gene'].isin(marker), method+'_V'], s=10, c='m')
    # add candidate genes
    ax.scatter(tmp_data.loc[tmp_data['gene'].isin(candidate), method+'_beta'],
            tmp_data.loc[tmp_data['gene'].isin(candidate), method+'_V'], s=10, c='r')


    ## add arrow to three markers
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    k = 0
    for index, row in tmp_data.loc[tmp_data['gene'].isin(marker+candidate)].iterrows():
        if row[method+'_beta'] < ((xlim[1]-xlim[0])*0.9+xlim[0]):
            k += 1
            if k % 2 == 0:
                shiftx = (xlim[1]-xlim[0])/10
                shifty = (ylim[1]-ylim[0])/20
            else:
                shiftx = (xlim[1]-xlim[0])/5
                shifty = (ylim[1]-ylim[0])/10
            if row[method+'_V'] > ((ylim[1]-ylim[0])*0.8+ylim[0]):
                shifty = shifty * (-1)
            ax.annotate(row['gene'], xy=(row[method+'_beta'],row[method+'_V']),
                    xytext=(row[method+'_beta']+shiftx, row[method+'_V']+shifty),
                    arrowprops={'arrowstyle':'->'})
        else:
            ax.annotate(row['gene'], xy=(row[method+'_beta'],row[method+'_V']),
                    xytext=(row[method+'_beta']-(xlim[1]-xlim[0])/10, row[method+'_V']+(ylim[1]-ylim[0])/10),
                    arrowprops={'arrowstyle':'->'})

    ax.axhline(tmp_threshold, color='0.8', ls='--', zorder=0)

    ax.axvline(tmp_threshold, color='0.8', ls='--', zorder=0)

    ax.set_xlabel('$-log_{10}$ p(mean differentiation)', fontsize=14)
    ax.set_ylabel('$-log_{10}$ p(variance differentiation)', fontsize=14)

    fig.tight_layout(pad=1)
    fig.savefig(png_f)

# QQ plot
reml_beta_ps = np.sort( remlJK['reml']['wald']['free']['ct_beta'] )
reml_V_ps = np.sort( remlJK['reml']['wald']['free']['V'] )
he_beta_ps = np.sort( out['he']['wald']['free']['ct_beta'] )
he_V_ps = np.sort( out['he']['wald']['free']['V'] )

fig, ax = plt.subplots(dpi=600)

osm, osr = stats.probplot(reml_V_ps, dist='uniform', fit=False)
ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Variance differentiation in REML' )

osm, osr = stats.probplot(he_V_ps, dist='uniform', fit=False)
ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Variance differentiation in HE' )

osm, osr = stats.probplot(reml_beta_ps, dist='uniform', fit=False)
ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Mean differentiation in REML' )

osm, osr = stats.probplot(he_beta_ps, dist='uniform', fit=False)
ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Mean differentiation in HE' )

plt.legend()

lims = [
        np.max([ax.get_xlim()[0], ax.get_ylim()[0]]),  # min of both axes
        np.min([ax.get_xlim()[1], ax.get_ylim()[1]]),  # max of both axes
        ]

ax.plot(lims, lims, '--', color='0.8', zorder=0)

ax.set_xlabel('Expected $-log_{10}P$')
ax.set_ylabel('Observed $-log_{10}P$')

fig.savefig(snakemake.output.qq)
