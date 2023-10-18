import math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# coeffcient of variation
var_nu = pd.read_table(snakemake.input.var_nu, index_col=(0,1))
nu = pd.read_table(snakemake.input.nu, index_col=(0,1))
cv = np.array( np.sqrt( var_nu ) / nu )

# Hom simulation with noise
rng = np.random.default_rng()
nu_noises = snakemake.params.nu_noises
for i in range(len(nu_noises)):
    nu_noise = nu_noises[i].split('_')
    a, b = float(nu_noise[1]), float(nu_noise[2])
    if a == 0 and b == 0:
        nu_noises[i] = 0
    else:
        nu_noises[i] = np.std(rng.beta(a,b,100000) * rng.choice([-1,1],100000))

outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.outs]
remlJK_outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.remlJK_outs]

#
data = {'noise':[], 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}
gene_no = len(outs[0]['reml']['wald']['free']['V'])
for noise, out, remlJK_out in zip(nu_noises, outs, remlJK_outs):
    data['noise'].append( noise )
    data['REML (LRT)'].append( (out['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no )
    data['REML (JK)'].append( (remlJK_out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no )
    data['HE'].append( (out['he']['wald']['free']['V'] < 0.05).sum() / gene_no )

data = pd.DataFrame(data)
data = data.melt(id_vars='noise', var_name='Method', value_name='power')

# Free simulation with noise
V3 = []
for x in snakemake.params.V3:
    x = x.split('_')[0]
    # if x == '0':
        # V3.append('Hom')
    # else:
    V3.append(float(x))
#V3 = [x.split('_')[0] for x in snakemake.params.V3]
outs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.outs3]
remlJKs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.remlJKs3]

# 
data3 = {'V':V3, 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}

for out3, remlJK3 in zip(outs3, remlJKs3):
    data3['REML (LRT)'].append( (out3['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no )
    data3['REML (JK)'].append( (remlJK3['reml']['wald']['free']['V'] < 0.05).sum() / gene_no )
    data3['HE'].append( (out3['he']['wald']['free']['V'] < 0.05).sum() / gene_no )

data3 = pd.DataFrame(data3)
data3 = data3.melt(id_vars='V', var_name='Method', value_name='power')

#
plt.rcParams.update({'font.size' : 7})

fig, ax = plt.subplots(figsize=(4.45,4), dpi=600)
ax.hist(np.array(cv).flatten(), density=True, bins=30)
ax.set_xlabel(r'Coefficient of variation($\nu$)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)

cv = cv.flatten()
cv = cv[~np.isnan(cv)]
ax.axvline(x=np.percentile(cv, 10), color='0.7', ls='--', zorder=10)
ax.axvline(x=np.percentile(cv, 50), color='0.7', ls='--', zorder=10)
ax.axvline(x=np.percentile(cv, 90), color='0.7', ls='--', zorder=10)

fig.savefig( snakemake.output.png1 )


# variance
colors = sns.color_palette()
lw = 1.0
fig, axes = plt.subplots(ncols=2, figsize=(6.85,2.8), dpi=600)

sns.lineplot(x='noise', y='power', hue='Method', palette=colors[1:4], style='Method', 
        dashes=False, markers=True, data=data, ax=axes[0], lw=lw)
axes[0].set_xlabel(r'Coefficient of variation($\nu$)', fontsize=10)
axes[0].set_ylabel('False positive rate', fontsize=10)
axes[0].text(-0.05, 1.05, '(A)', fontsize=10, transform=axes[0].transAxes)
axes[0].axhline(y=0.05, color='0.8', ls='--', zorder=0)
axes[0].set_ylim([-0.02,1.02])
axes[0].legend(loc='upper left')

axes[0].axvline(x=np.percentile(cv, 10), color='0.6', ls='--', zorder=0, lw=0.8 * lw)
axes[0].axvline(x=np.percentile(cv, 50), color='0.6', ls='--', zorder=0, lw=0.8 * lw)
axes[0].axvline(x=np.percentile(cv, 90), color='0.6', ls='--', zorder=0, lw=0.8 * lw)

sns.lineplot(x='V', y='power', hue='Method', palette=colors[1:4], style='Method', 
        dashes=False, markers=True, data=data3, ax=axes[1], lw=lw)
axes[1].set_ylabel('True positive rate', fontsize=10)
axes[1].axhline(y=0.05, color='0.8', ls='--', zorder=0)
axes[1].text(-0.05, 1.05, '(B)', fontsize=10, transform=axes[1].transAxes)
axes[1].legend().set_visible(False)
axes[1].set_ylim([-0.02, 1.02])

# add lines of quantile v_i from Cuomo
real = np.load(snakemake.input.real, allow_pickle=True).item()
real_V = np.diagonal(real['reml']['free']['V'], axis1=1, axis2=2)
axes[1].axvline(x=np.percentile(real_V, 10), color='0.7', ls='--', zorder=0, lw=0.8 * lw)
axes[1].axvline(x=np.percentile(real_V, 50), color='0.7', ls='--', zorder=0, lw=0.8 * lw)
# axes[1].axvline(x=np.percentile(real_V, 60), color='0.7', ls='--', zorder=10)

# Get current tick positions and labels
xticks, labels = plt.xticks()
print(xticks)
print(labels)

# Customize x tick labels
labels = np.array(labels)
labels[xticks == 0] = 'Hom'
print(labels)

plt.xticks(xticks, labels)
axes[1].set_xlim([-0.02, 0.52])


fig.tight_layout(pad=2, h_pad=5)
fig.savefig(snakemake.output.png2)
