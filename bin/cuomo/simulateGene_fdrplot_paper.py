import math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# coeffcient of variation
var_nu = pd.read_table(snakemake.input.var_nu, index_col=(0,1))
nu = pd.read_table(snakemake.input.nu, index_col=(0,1))
cv = np.array( np.sqrt( var_nu ) / nu )
cv = cv.flatten()
cv = cv[~np.isnan(cv)]

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
data = {'noise': nu_noises, 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}
gene_no = len(outs[0]['reml']['wald']['free']['V'])
for out, remlJK_out in zip(outs, remlJK_outs):
    data['REML (LRT)'].append( (out['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no )
    data['REML (JK)'].append( (remlJK_out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no )
    data['HE'].append( (out['he']['wald']['free']['V'] < 0.05).sum() / gene_no )

data = pd.DataFrame(data)
data = data.melt(id_vars='noise', var_name='Method', value_name='fp')

# Free simulation with noise
outs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.outs3]
remlJKs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.remlJKs3]

# 
data3 = {'noise': nu_noises, 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}

for out3, remlJK3 in zip(outs3, remlJKs3):
    data3['REML (LRT)'].append( (out3['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no )
    data3['REML (JK)'].append( (remlJK3['reml']['wald']['free']['V'] < 0.05).sum() / gene_no )
    data3['HE'].append( (out3['he']['wald']['free']['V'] < 0.05).sum() / gene_no )

data3 = pd.DataFrame(data3)
data3 = data3.melt(id_vars='noise', var_name='Method', value_name='tp')

# FDR
data = data.merge(data3)
print(data.head())
data['FDR'] = data['fp'] / (data['fp'] + data['tp'])
data['PPV'] = 1 - data['FDR']
data['TPR'] = data['tp'] / (data['fp'] +data['tp'])
data.to_csv('fdr.txt', sep='\t', index=False)

#
plt.rcParams.update({'font.size' : 7})

colors = sns.color_palette()
lw = 1.0

fig, ax = plt.subplots(dpi=600)

tmp_data = pd.melt(data, id_vars=['noise', 'Method'], value_vars=['FDR', 'PPV'], var_name='Measure')
sns.lineplot(x='noise', y='value', hue='Method', palette=colors[1:4], style='Measure', 
        dashes=True, markers=True, style_order=['FDR', 'PPV'], data=tmp_data, ax=ax, lw=lw)

ax.set_xlabel(r'Coefficient of variation($\nu$)', fontsize=10)
ax.set_ylabel('Rate', fontsize=10)

ax.axvline(x=np.percentile(cv, 10), color='0.6', ls='--', zorder=0, lw=0.8 * lw)
ax.axvline(x=np.percentile(cv, 50), color='0.6', ls='--', zorder=0, lw=0.8 * lw)
ax.axvline(x=np.percentile(cv, 90), color='0.6', ls='--', zorder=0, lw=0.8 * lw)
ax.axhline(y=0.5, color='0.8', ls='--', zorder=0)

ax.legend()

fig.savefig(snakemake.output.png)


# PPV vs FDR
fig, ax = plt.subplots(dpi=600)

sns.lineplot(x='PPV', y='FDR', hue='Method', palette=colors[1:4],
             dashes=True, markers=True, data=data, lw=lw)

ax.set_xlabel('PPV', fontsize=10)
ax.set_ylabel('FDR', fontsize=10)

fig.savefig(snakemake.output.png2)
