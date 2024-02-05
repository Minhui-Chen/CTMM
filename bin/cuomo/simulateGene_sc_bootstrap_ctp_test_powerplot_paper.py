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
# transform beta dist to cv
rng = np.random.default_rng(11111)
nu_noises = snakemake.params.nu_noises
for i in range(len(nu_noises)):
    nu_noise = nu_noises[i].split('_')
    a, b = float(nu_noise[1]), float(nu_noise[2])
    if a == 0 and b == 0:
        nu_noises[i] = 0
    else:
        nu_noises[i] = np.std(rng.beta(a,b,100000) * rng.choice([-1,1],100000))

# read hom results
cellno_outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.cellno_outs]
depth_outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.depth_outs]
remlJK_cellno_outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.remlJK_cellno_outs]
remlJK_depth_outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.remlJK_depth_outs]


# collect results
gene_no = len(cellno_outs[0]['he']['wald']['free']['V'])
# cellno_data = {'Cell':[], 'Depth': 1, 'noise':[], 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}
cellno_data = {'Cell':[], 'Depth': 1, 'noise':[], 'REML (JK)':[], 'HE':[]}
i = 0
for cell_no in snakemake.params.cell_nos:
    for noise in nu_noises:
        print(cell_no, noise, snakemake.input.cellno_outs[i])
        out = cellno_outs[i]
        remlJK_out = remlJK_cellno_outs[i]

        cellno_data['Cell'].append(cell_no)
        cellno_data['noise'].append(noise)
        # cellno_data['REML (LRT)'].append((out['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no)
        cellno_data['REML (JK)'].append((remlJK_out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no)
        cellno_data['HE'].append((out['he']['wald']['free']['V'] < 0.05).sum() / gene_no)

        i += 1

cellno_data = pd.DataFrame(cellno_data)
cellno_data = cellno_data.melt(id_vars=['Cell', 'Depth', 'noise'], var_name='Method', value_name='power')

depth_data = {'Cell': 1, 'Depth': [], 'noise':[], 'REML (JK)':[], 'HE':[]}
# depth_data = {'Cell': 1, 'Depth': [], 'noise':[], 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}
i = 0
for depth in snakemake.params.depths:
    for noise in nu_noises:
        print(depth, noise, snakemake.input.depth_outs[i])
        out = depth_outs[i]
        remlJK_out = remlJK_depth_outs[i]

        depth_data['Depth'].append(depth)
        depth_data['noise'].append(noise)
        # depth_data['REML (LRT)'].append((out['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no)
        depth_data['REML (JK)'].append((remlJK_out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no)
        depth_data['HE'].append((out['he']['wald']['free']['V'] < 0.05).sum() / gene_no)

        i += 1

depth_data = pd.DataFrame(depth_data)
depth_data = depth_data.melt(id_vars=['Cell', 'Depth', 'noise'], var_name='Method', value_name='power')

# Free simulation with noise
cellno_outs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.cellno_outs3]
depth_outs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.depth_outs3]
cellno_remlJKs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.cellno_remlJKs3]
depth_remlJKs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.depth_remlJKs3]


#
cellno_data3 = {'Cell':[], 'Depth': 1, 'V':[], 'REML (JK)':[], 'HE':[]}
# cellno_data3 = {'Cell':[], 'Depth': 1, 'V':[], 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}
i = 0
for cell_no in snakemake.params.cell_nos:
    for V in snakemake.params.Vs:
        # if V == '0':
            # V = 'Hom'

        out = cellno_outs3[i]
        remlJK_out = cellno_remlJKs3[i]

        cellno_data3['Cell'].append(cell_no)
        cellno_data3['V'].append(V)
        # cellno_data3['REML (LRT)'].append((out['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no)
        cellno_data3['REML (JK)'].append((remlJK_out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no)
        cellno_data3['HE'].append((out['he']['wald']['free']['V'] < 0.05).sum() / gene_no)

        i += 1

cellno_data3 = pd.DataFrame(cellno_data3)
cellno_data3 = cellno_data3.melt(id_vars=['Cell', 'Depth', 'V'], var_name='Method', value_name='power')

depth_data3 = {'Cell': 1, 'Depth': [], 'V':[], 'REML (JK)':[], 'HE':[]}
# depth_data3 = {'Cell': 1, 'Depth': [], 'V':[], 'REML (LRT)':[], 'REML (JK)':[], 'HE':[]}
i = 0
for depth in snakemake.params.depths:
    for V in snakemake.params.Vs:
        # if V == '0':
            # V = 'Hom'

        out = depth_outs3[i]
        remlJK_out = depth_remlJKs3[i]

        depth_data3['Depth'].append(depth)
        depth_data3['V'].append(V)
        # depth_data3['REML (LRT)'].append((out['reml']['lrt']['free_hom'] < 0.05).sum() / gene_no)
        depth_data3['REML (JK)'].append((remlJK_out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no)
        depth_data3['HE'].append((out['he']['wald']['free']['V'] < 0.05).sum() / gene_no)

        i += 1

depth_data3 = pd.DataFrame(depth_data3)
depth_data3 = depth_data3.melt(id_vars=['Cell', 'Depth', 'V'], var_name='Method', value_name='power')

#
plt.rcParams.update({'font.size' : 7})
colors = sns.color_palette()[1:4]
markers = ['o', 'X', 's', 'P']
lw = 0.6
markersize = 5
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.85, 6), dpi=600)





sns.lineplot(x='noise', y='power', hue='Method', palette=colors[1:], style='Depth', 
        dashes=False, markers=markers, data=depth_data, ax=axes[0, 0], lw=lw, markersize=markersize, markeredgecolor='None')
axes[0, 0].set_xlabel(r'Coefficient of variation($\nu$)', fontsize=10)
axes[0, 0].set_ylabel('False positive rate', fontsize=10)
axes[0, 0].text(-0.05, 1.05, '(A)', fontsize=10, transform=axes[0, 0].transAxes)
axes[0, 0].axhline(y=0.05, color='0.8', ls='--', zorder=0)
axes[0, 0].set_ylim([-0.02, 1.02])
# axes[1, 0].legend(loc='upper left', ncols=2)

axes[0, 0].axvline(x=np.percentile(cv, 10), color='0.6', ls='--', zorder=0, lw=lw)
axes[0, 0].axvline(x=np.percentile(cv, 50), color='0.6', ls='--', zorder=0, lw=lw)
axes[0, 0].axvline(x=np.percentile(cv, 90), color='0.6', ls='--', zorder=0, lw=lw)

# make two legends
# line2, = axes[1, 0].plot([], [], color=colors[0], label='REML (LRT)')
line3, = axes[0, 0].plot([], [], color=colors[1], label='REML (JK)')
line4, = axes[0, 0].plot([], [], color=colors[2], label='HE')
first_legend = axes[0, 0].legend(handles=[line3, line4], loc='upper left', title='Method')
# first_legend = axes[1, 0].legend(handles=[line2, line3, line4], loc='upper left', title='Method')
axes[0, 0].add_artist(first_legend)

lines = []
for i, cell_no in enumerate(np.unique(depth_data['Depth'])):
    lines.append(axes[0, 0].plot([], [], color='k', marker=markers[i], label=cell_no)[0])
axes[0, 0].legend(handles=lines, loc='upper center', title='Depth')


sns.lineplot(x='V', y='power', hue='Method', palette=colors[1:], style='Depth', 
        dashes=False, markers=markers, data=depth_data3, ax=axes[0, 1], lw=lw, markersize=markersize, markeredgecolor='None')
axes[0, 1].set_ylabel('True positive rate', fontsize=10)
axes[0, 1].axhline(y=0.05, color='0.8', ls='--', zorder=0)
axes[0, 1].text(-0.05, 1.05, '(B)', fontsize=10, transform=axes[0, 1].transAxes)
axes[0, 1].legend().set_visible(False)
axes[0, 1].set_ylim([-0.02, 1.02])

# add lines of quantile v_i from Cuomo
# axes[1, 1].axvline(x=np.percentile(real_V, 10), color='0.6', ls='--', zorder=0, lw=lw)
# axes[1, 1].axvline(x=np.percentile(real_V, 50), color='0.6', ls='--', zorder=0, lw=lw)
# axes[1, 1].axvline(x=np.percentile(real_V, 90), color='0.6', ls='--', zorder=0, lw=lw)

# Get current tick positions and labels
xticks, labels = axes[0, 1].get_xticks(), axes[0, 1].get_xticklabels()
print(xticks)
print(labels)

# Customize x tick labels
labels = np.array(labels)
labels[xticks == 0] = 'Hom'
print(labels)

axes[0, 1].set_xticks(xticks, labels)
axes[0, 1].set_xlim([-0.04, 1.04])


sns.lineplot(x='noise', y='power', hue='Method', palette=colors[1:], style='Cell', 
        dashes=False, markers=markers, data=cellno_data, ax=axes[1, 0], lw=lw, markersize=markersize, markeredgecolor='None')
axes[1, 0].set_xlabel(r'Coefficient of variation($\nu$)', fontsize=10)
axes[1, 0].set_ylabel('False positive rate', fontsize=10)
axes[1, 0].text(-0.05, 1.05, '(C)', fontsize=10, transform=axes[1, 0].transAxes)
axes[1, 0].axhline(y=0.05, color='0.8', ls='--', zorder=0)
axes[1, 0].set_ylim([-0.02, 1.02])

axes[1, 0].axvline(x=np.percentile(cv, 10), color='0.6', ls='--', zorder=0, lw=lw)
axes[1, 0].axvline(x=np.percentile(cv, 50), color='0.6', ls='--', zorder=0, lw=lw)
axes[1, 0].axvline(x=np.percentile(cv, 90), color='0.6', ls='--', zorder=0, lw=lw)


sns.lineplot(x='V', y='power', hue='Method', palette=colors[1:], style='Cell', 
        dashes=False, markers=markers, data=cellno_data3, ax=axes[1, 1], lw=lw, 
        markersize=markersize, markeredgecolor='None')

axes[1, 1].set_ylabel('True positive rate', fontsize=10)
axes[1, 1].axhline(y=0.05, color='0.8', ls='--', zorder=0)
axes[1, 1].text(-0.05, 1.05, '(D)', fontsize=10, transform=axes[1, 1].transAxes)
axes[1, 1].set_ylim([-0.02, 1.02])

# Get current tick positions and labels
xticks, labels = axes[1, 1].get_xticks(), axes[1, 1].get_xticklabels()
print(xticks)
print(labels)

# Customize x tick labels
labels = np.array(labels)
labels[xticks == 0] = 'Hom'
print(labels)

axes[1, 1].set_xticks(xticks, labels)
axes[1, 1].set_xlim([-0.04, 1.04])


# add cell count = 10 lines
if 'cellcount_outs' in snakemake.input.keys():
    cellcount_outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.cellcount_outs]
    remlJK_cellcount_outs = [np.load(f, allow_pickle=True).item() for f in snakemake.input.remlJK_cellcount_outs]

    cellcount_data = {'Cell': 10, 'Depth': 1, 'noise':[], 'REML (JK)':[], 'HE':[]}
    for i, noise in enumerate(nu_noises):
        print(noise, snakemake.input.cellcount_outs[i])
        out = cellcount_outs[i]
        remlJK_out = remlJK_cellcount_outs[i]

        cellcount_data['noise'].append(noise)
        cellcount_data['REML (JK)'].append((remlJK_out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no)
        cellcount_data['HE'].append((out['he']['wald']['free']['V'] < 0.05).sum() / gene_no)

    cellcount_data = pd.DataFrame(cellcount_data)
    cellcount_data = cellcount_data.melt(id_vars=['Cell', 'Depth', 'noise'], var_name='Method', value_name='power')

    cellcount_outs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.cellcount_outs3]
    remlJK_cellcount_outs3 = [np.load(f, allow_pickle=True).item() for f in snakemake.input.cellcount_remlJKs3]

    cellcount_data3 = {'Cell': 10, 'Depth': 1, 'V':[], 'REML (JK)':[], 'HE':[]}
    for i, V in enumerate(snakemake.params.Vs):
        print(V, snakemake.input.cellcount_outs3[i])
        out = cellcount_outs3[i]
        remlJK_out = remlJK_cellcount_outs3[i]

        cellcount_data3['V'].append(V)
        cellcount_data3['REML (JK)'].append((remlJK_out['reml']['wald']['free']['V'] < 0.05).sum() / gene_no)
        cellcount_data3['HE'].append((out['he']['wald']['free']['V'] < 0.05).sum() / gene_no)

    cellcount_data3 = pd.DataFrame(cellcount_data3)
    cellcount_data3 = cellcount_data3.melt(id_vars=['Cell', 'Depth', 'V'], var_name='Method', value_name='power')

    # plot
    sns.lineplot(x='noise', y='power', hue='Method', palette=colors[1:],  
            linestyle='--', marker='d', data=cellcount_data, ax=axes[1, 0], lw=lw, markersize=markersize, markeredgecolor='None')
    sns.lineplot(x='V', y='power', hue='Method', palette=colors[1:],  
            linestyle='--', marker='d', data=cellcount_data3, ax=axes[1, 1], lw=lw, 
            markersize=markersize, markeredgecolor='None')


# make two legends
line3, = axes[1, 0].plot([], [], color=colors[1], label='REML (JK)')
line4, = axes[1, 0].plot([], [], color=colors[2], label='HE')
first_legend = axes[1, 0].legend(handles=[line3, line4], loc='upper left', title='Method')
axes[1, 0].add_artist(first_legend)

lines = []
for i, cell_no in enumerate(np.unique(cellno_data['Cell'])):
    lines.append(axes[1, 0].plot([], [], color='k', marker=markers[i], label=cell_no)[0])
axes[1, 0].legend(handles=lines, loc='upper center', title='Cell')


axes[1, 1].legend().set_visible(False)

# add lines of quantile v_i from Cuomo
# real = np.load(snakemake.input.real, allow_pickle=True).item()
# real_V = np.diagonal(real['reml']['free']['V'], axis1=1, axis2=2)
# axes[0, 1].axvline(x=np.percentile(real_V, 10), color='0.6', ls='--', zorder=0, lw=lw)
# axes[0, 1].axvline(x=np.percentile(real_V, 50), color='0.6', ls='--', zorder=0, lw=lw)
# axes[0, 1].axvline(x=np.percentile(real_V, 90), color='0.6', ls='--', zorder=0, lw=lw)


fig.tight_layout(pad=2, h_pad=5)
fig.savefig(snakemake.output.png)

# save source data
if 'data' in snakemake.output.keys():
    if 'cellcount_outs' in snakemake.input.keys():
        data = pd.concat([cellno_data, depth_data, cellno_data3, depth_data3, cellcount_data, cellcount_data3], ignore_index=True)
    else:
        data = pd.concat([cellno_data, depth_data, cellno_data3, depth_data3], ignore_index=True)

    data.to_csv(snakemake.output.data, sep='\t', index=False)