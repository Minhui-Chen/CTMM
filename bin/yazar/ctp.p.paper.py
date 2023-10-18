import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from ctmm import draw

out = np.load(snakemake.input.out, allow_pickle=True).item()

method = snakemake.params.get('method', 'he')

# p
beta_ps = out[method]['wald']['free']['ct_beta']
beta_ps_zeros = beta_ps == 0
V_ps = out[method]['wald']['free']['V']

beta_ps = np.log10(beta_ps) * (-1)
V_ps = np.log10(V_ps) * (-1)
genes = out['gene']

data = pd.DataFrame({'beta': beta_ps[~ beta_ps_zeros], 
                     'V': V_ps[~ beta_ps_zeros], 
                     'gene': genes[~ beta_ps_zeros]})
# data = pd.DataFrame({'beta': beta_ps, 
#                      'V': V_ps, 
#                      'gene': genes})
threshold = (-1) * np.log10(0.05/len(genes))

# plot
fig, ax = plt.subplots(dpi=600)


axins0 = ax.inset_axes([0.65, 0.68, 0.3, 0.3]) # [x0, y0, width, height]
axins0.scatter(np.arange(beta_ps_zeros.sum()), np.sort(V_ps[beta_ps_zeros]), s=0.2, c='0.7')
axins0.tick_params(axis='x', labelsize=6)  # Change the fontsize value as needed
axins0.tick_params(axis='y', labelsize=6)  # Change the fontsize value as needed

axins0.set_xlabel('gene number', fontsize=7)
axins0.set_ylabel('$-log_{10}$ p', fontsize=7)


## heatscatter
draw.scatter(data['beta'], data['V'], s=5, heatscatter=True, linregress=False, ax=ax)


ax.axhline(threshold, color='0.8', ls='--', zorder=0)

ax.axvline(threshold, color='0.8', ls='--', zorder=0)

ax.set_xlabel('$-log_{10}$ p(mean differentiation)', fontsize=14)
ax.set_ylabel('$-log_{10}$ p(variance differentiation)', fontsize=14)


ylim1 = ax.get_ylim()
ylim2 = axins0.get_ylim()

axins0.set_ylim(np.amin([ylim1, ylim2]), np.amax([ylim1, ylim2]))
ax.set_ylim(np.amin([ylim1, ylim2]), np.amax([ylim1, ylim2]))


fig.tight_layout(pad=1)
fig.savefig(snakemake.output.png)

# QQ plot
# reml_beta_ps = np.sort( remlJK['reml']['wald']['free']['ct_beta'] )
# reml_V_ps = np.sort( remlJK['reml']['wald']['free']['V'] )
# he_beta_ps = np.sort( out['he']['wald']['free']['ct_beta'] )
# he_V_ps = np.sort( out['he']['wald']['free']['V'] )

# fig, ax = plt.subplots(dpi=600)

# osm, osr = stats.probplot(reml_V_ps, dist='uniform', fit=False)
# ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Variance differentiation in REML' )

# osm, osr = stats.probplot(he_V_ps, dist='uniform', fit=False)
# ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Variance differentiation in HE' )

# osm, osr = stats.probplot(reml_beta_ps, dist='uniform', fit=False)
# ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Mean differentiation in REML' )

# osm, osr = stats.probplot(he_beta_ps, dist='uniform', fit=False)
# ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Mean differentiation in HE' )

# plt.legend()

# lims = [
#         np.max([ax.get_xlim()[0], ax.get_ylim()[0]]),  # min of both axes
#         np.min([ax.get_xlim()[1], ax.get_ylim()[1]]),  # max of both axes
#         ]

# ax.plot(lims, lims, '--', color='0.8', zorder=0)

# ax.set_xlabel('Expected $-log_{10}P$')
# ax.set_ylabel('Observed $-log_{10}P$')

# fig.savefig(snakemake.output.qq)
