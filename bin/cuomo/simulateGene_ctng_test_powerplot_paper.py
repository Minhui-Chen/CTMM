import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plot_help

# coeffcient of variation
var_nu = pd.read_table(snakemake.input.var_nu, index_col=(0,1))
nu = pd.read_table(snakemake.input.nu, index_col=(0,1))
cv = np.sqrt( var_nu ) / np.array(nu)

# Hom simulation with noise
nu_noises = snakemake.params.nu_noises
for i in range(len(nu_noises)):
    nu_noise = nu_noises[i].split('_')
    if float(nu_noise[1]) == 0 and float(nu_noise[2]) == 0:
        nu_noises[i] = 'No noise'
    else:
        nu_noises[i] = f'Beta({nu_noise[1]},{nu_noise[2]})'

outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.outs]
remlJK_outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.remlJK_outs]

#
data = {'noise':[], 'REML (Wald)':[], 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}
gene_no = len(outs[0]['ml']['wald']['free']['V'])
for noise, out, remlJK_out in zip(nu_noises, outs, remlJK_outs):
    data['noise'].append( noise )
    data['REML (Wald)'].append( (out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no )
    data['REML (LRT)'].append( (out['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no )
    data['REML (JK)'].append( (remlJK_out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no )
    data['HE'].append( (out['he']['wald']['free']['V'] < 0.05).sum() / gene_no )

data = pd.DataFrame(data)
data = data.melt(id_vars='noise', var_name='method', value_name='power')

# Free simulation with noise
V3 = []
for x in snakemake.params.V3:
    x = x.split('_')[0]
    if x == '0':
        V3.append('Hom')
    else:
        V3.append(x)
#V3 = [x.split('_')[0] for x in snakemake.params.V3]
outs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.outs3]
remlJKs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.remlJKs3]

# 
data3 = {'V':V3, 'REML (Wald)':[], 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}

for out3, remlJK3 in zip(outs3, remlJKs3):
    data3['REML (Wald)'].append( (out3['reml']['wald']['free']['V'] < 0.05).sum() / gene_no )
    data3['REML (LRT)'].append( (out3['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no )
    data3['REML (JK)'].append( (remlJK3['reml']['wald']['free']['V'] < 0.05).sum() / gene_no )
    data3['HE'].append( (out3['he']['wald']['free']['V'] < 0.05).sum() / gene_no )

data3 = pd.DataFrame(data3)
data3 = data3.melt(id_vars='V', var_name='method', value_name='power')

#
#plt.rcParams.update({'font.size' : 6})
fig, axes = plt.subplots(nrows=3, figsize=(6,12), dpi=600)

axes[0].hist(np.array(cv).flatten(), density=True, bins=20)
axes[0].set_xlabel(r'coefficient of variation($\nu$)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].text(-0.05, 1.05, '(A)', fontsize=14, transform=axes[0].transAxes)
# add coefficient of beta distribution
rng = np.random.default_rng()
mycolors = plot_help.mycolors()
axes[0].axvline(x=np.std(rng.choice([-1,1], 10000) * rng.beta(2,20,10000)),
        color='0.9', ls='--', zorder=10, label='Beta(2,20)')
axes[0].axvline(x=np.std(rng.choice([-1,1], 10000) * rng.beta(2,10,10000)),
        color='0.7', ls='--', zorder=10, label='Beta(2,10)')
axes[0].axvline(x=np.std(rng.choice([-1,1], 10000) * rng.beta(2,5,10000)),
        color='0.5', ls='--', zorder=10, label='Beta(2,5)')
axes[0].axvline(x=np.std(rng.choice([-1,1], 10000) * rng.beta(2,3,10000)),
        color='0.3', ls='--', zorder=10, label='Beta(2,3)')
axes[0].axvline(x=np.std(rng.choice([-1,1], 10000) * rng.beta(2,2,10000)),
        color='0.1', ls='--', zorder=10, label='Beta(2,2)')
axes[0].legend()

#sns.lineplot(x='noise', y='power', hue='method', data=data, ax=axes[1], markers=True)
sns.barplot(x='noise', y='power', hue='method', data=data, ax=axes[1])
axes[1].set_xlabel('Noise', fontsize=12)
axes[1].set_ylabel('False positive rate', fontsize=12)
axes[1].text(-0.05, 1.05, '(B)', fontsize=14, transform=axes[1].transAxes)
axes[1].axhline(y=0.05, color='0.6', ls='--', zorder=0)

sns.barplot(x='V', y='power', hue='method', data=data3, ax=axes[2])
axes[2].set_ylabel('True positive rate', fontsize=12)
axes[2].axhline(y=0.05, color='0.9', ls='--', zorder=0)
axes[2].text(-0.05, 1.05, '(C)', fontsize=14, transform=axes[2].transAxes)
#axes[2].set_title('Celltype specific variance V (x,0.1,0.1,0.1)')

fig.tight_layout(pad=2, h_pad=3)
fig.savefig(snakemake.output.png)
