import os, sys, tempfile
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ctmm import draw


def error1(ax, V, beta, CTs, label=None, shift=0):
    # plot with V only, no hom2
    error = V[CTs].iloc[0]
    error = np.array( [error * (-1), error] )
    error[error<0] = 0
    ax.errorbar( np.arange(len(CTs))+shift, beta[CTs].iloc[0], error, 
            marker='o', linestyle='', label=label, zorder=10, alpha=0.8 )


def error2(ax, V, beta, CTs, label=None, shift=0, color=None):
    # plot with V + hom2, and V
    error = V[CTs].iloc[0]
    hom2 = V.iloc[0]['hom2']
    add = error+hom2
    if np.any( (add) < 0):
        print(V, beta, label)
        add[add<0] = 0
        ax.text(-0.05, 1.05, 'Negative variance!', transform=ax.transAxes)
    eb = ax.errorbar( np.arange(len(CTs))+shift, beta[CTs].iloc[0], error+hom2, capsize=3, capthick=1.5,
            marker='o', linestyle='', label=label, zorder=10, alpha=0.6, color=color, fillstyle='none' )
    eb[-1][0].set_linestyle('--')
    # for i in range(len(CTs)):
    #     ax.arrow( i+shift, beta[CTs].iloc[0][i], 0, error[i], head_width=0.05,
    #             length_includes_head=True, color=color, zorder=10)


def format_e(x):
    if x > 1e-3:
        x = '%.3f'%(x)
    else:
        x = draw.format_e('%.2e'%(x))
    return( x )


# 
meta = pd.read_table(snakemake.input.meta)
out = np.load(snakemake.input.out, allow_pickle=True).item()
remlJK = np.load(snakemake.input.remlJK, allow_pickle=True).item()
reml_V = [np.diag(x) for x in remlJK['reml']['free']['V']]
reml_V_p = dict( zip(remlJK['gene'], remlJK['reml']['wald']['free']['V']) )
C = len( reml_V[0] )
CTs = ['day0', 'day1', 'day2', 'day3']
reml_V = pd.DataFrame(reml_V, columns=CTs)

reml_V['gene'] = remlJK['gene']
reml_V['hom2'] = remlJK['reml']['free']['hom2']

reml_beta = pd.DataFrame(remlJK['reml']['free']['beta']['ct_beta'], columns=CTs)
reml_beta['gene'] = remlJK['gene']
reml_beta_p = dict( zip(remlJK['gene'], remlJK['reml']['wald']['free']['ct_beta']) )

he_V = [np.diag(x) for x in out['he']['free']['V']]
he_V = pd.DataFrame(he_V, columns=CTs)
he_V['gene'] = out['gene']
he_V['hom2'] = out['he']['free']['hom2']

he_beta = pd.DataFrame(out['he']['free']['beta']['ct_beta'], columns=CTs)
he_beta['gene'] = out['gene']

genes = snakemake.params.genes

plt.rcParams.update( {'font.size' : 10} )
fig, axes = plt.subplots( nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10,6), dpi=600 )

ct_ys = None
for gene, ax in zip(genes, axes.flatten()):
    if '_' not in gene:
        tmp = reml_V.loc[reml_V['gene'].str.contains('_%s$'%(gene))]
        if tmp.shape[0] != 1:
            print( gene )
            print( tmp )
            for g in reml_V['gene']:
                if 'ENSG00000065518_NDUFB4' in g:
                    print(g)
            sys.exit('Missing gene!\n')
        gene = np.array(tmp['gene'])[0]

    gene_reml_V = reml_V.loc[reml_V['gene'] == gene]
    gene_reml_beta = reml_beta.loc[reml_beta['gene'] == gene]

    gene_he_V = he_V.loc[he_V['gene'] == gene]
    gene_he_beta = he_beta.loc[he_beta['gene'] == gene]

    # tmpf = tempfile.NamedTemporaryFile(delete=False)
    # tmpfn = tmpf.name
    # tmpf.close()
    # os.system(f'zcat {snakemake.input.counts} | head -n1 > {tmpfn}')
    # os.system(f'zcat {snakemake.input.counts} | grep {gene} >> {tmpfn}')
    # counts = pd.read_table(tmpfn, index_col=0)
    # counts = counts.loc[counts.index==gene]
    # counts = counts.transpose()

    ct_y_f = None
    for f in snakemake.input.imputed_ct_y:
        for line in open(f):
            if gene in line:
                ct_y_f = line.strip()
                break
        if ct_y_f:
            break
    ct_y = pd.read_table(ct_y_f)
    if ct_ys is None:
        ct_ys = ct_y
    else:
        ct_ys = ct_ys.merge(ct_y)

    #
    # data = counts.merge(meta, left_index=True, right_on='cell_name')
    # print(data.shape[0], meta.shape[0])

    # plot
    sns.violinplot(data=ct_y, x='day', y=gene, order=CTs, inner=None, color=".8", ax=ax)
    #sns.swarmplot(data=ct_y, x='day', y=gene, order=CTs, color="white", edgecolor="gray", size=3, ax=ax)

    error2(ax, gene_reml_V, gene_reml_beta, CTs, label='REML', color=snakemake.params.mycolors[0])
    #error2(ax, gene_he_V, gene_he_beta, CTs, label='HE', shift=0.03, color=snakemake.params.mycolors[1])

    ax.text(0.96, 0.02, 
            f'p(variance)={format_e(reml_V_p[gene])}\np(mean)={format_e(reml_beta_p[gene])}',
            horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(gene.split('_')[1])


axes[0,0].set_ylabel('CT-specific pseudo-bulk', fontsize=12)
axes[1,0].set_ylabel('CT-specific pseudo-bulk', fontsize=12)

fig.tight_layout(w_pad=3)

fig.savefig(snakemake.output.png)


ct_ys.to_csv(snakemake.output.data, sep='\t', index=False)
