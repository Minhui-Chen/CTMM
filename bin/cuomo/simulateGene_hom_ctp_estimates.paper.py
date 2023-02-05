import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#
nu_noises = snakemake.params.nu_noises
for i in range(len(nu_noises)):
    nu_noise = nu_noises[i].split('_')
    if float(nu_noise[1]) == 0 and float(nu_noise[2]) == 0:
        nu_noises[i] = 'No noise'
    else:
        nu_noises[i] = f'Beta({nu_noise[1]}, {nu_noise[2]})'

outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.outs]

C = outs[0]['reml']['free']['V'].shape[1]
CTs = ['CT'+str(i) for i in range(C)]
reml = []
#remlJK = []
he = []
for nu_noise, out in zip(nu_noises, outs):
    reml_V = pd.DataFrame( np.array([np.diag(x) for x in out['reml']['free']['V']]), columns=CTs )
    reml_V[reml_V > 0.2] = 0.2
    reml_V[reml_V < -0.2] = -0.2
    reml_V['noise'] = nu_noise
    reml_V = pd.melt(reml_V, id_vars=['noise'], var_name='CT', value_name='CT specific variance')
    reml.append( reml_V )

    he_V = pd.DataFrame( np.array([np.diag(x) for x in out['he']['free']['V']]), columns=CTs )
    he_V[he_V > 0.2] = 0.2
    he_V[he_V < -0.2] = -0.2
    he_V['noise'] = nu_noise
    he_V = pd.melt(he_V, id_vars=['noise'], var_name='CT', value_name='CT specific variance')
    he.append( he_V )

reml = pd.concat( reml, ignore_index=True)
he = pd.concat( he, ignore_index=True)

#
fig, axes = plt.subplots(nrows=2, figsize=(6,8), sharex=True, sharey=True)
sns.violinplot(x='noise', y='CT specific variance', hue='CT', data=reml, ax=axes[0], cut=0)
axes[0].set_xlabel('')
axes[0].set_ylabel('CT specific variance', fontsize=14)
axes[0].set_title('REML', fontsize=14)
sns.violinplot(x='noise', y='CT specific variance', hue='CT', data=he, cut=0, ax=axes[1])
axes[1].set_xlabel('noise', fontsize=14)
axes[1].set_ylabel('CT specific variance', fontsize=14)
axes[1].set_title('HE', fontsize=14)
axes[1].legend().set_visible(False)
for ax in axes:
    ax.axhline(y=0.00, color='0.7', ls='--', lw=2, zorder=0)

fig.tight_layout(pad=2, h_pad=3)
fig.savefig(snakemake.output.png)
