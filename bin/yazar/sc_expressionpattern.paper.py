import os, re, sys
import numpy as np, pandas as pd
import matplotlib as mpl
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


def error2(ax, V, beta, CTs, label=None, shift=0, color=None, plot_negative=False, negative_variance_warn=True):
    # plot with V + hom2, and V
    error = V[CTs].iloc[0]
    hom2 = V.iloc[0]['hom2']
    add = error+hom2
    positive_variance = True
    if np.any(add < 0):
        print(V, beta, label)
        add[add<0] = 0
        if negative_variance_warn:
            ax.text(-0.05, 1.05, 'Negative variance!', transform=ax.transAxes)
        positive_variance = False

    if positive_variance or plot_negative:
        eb = ax.errorbar( np.arange(len(CTs))+shift, beta[CTs].iloc[0], add, capsize=3, capthick=1.5,
                marker='o', linestyle='', label=label, zorder=10, alpha=0.6, color=color, fillstyle='none' )
        eb[-1][0].set_linestyle('--')
        # set head leangth for arrow. trans from Axes coord to data coord
        head_length = 0.01
        head_length = ax.transLimits.inverted().transform((0, head_length))[1] - ax.transLimits.inverted().transform((0, 0))[1]

        # for i in range(len(CTs)):
        #     ax.arrow(i+shift, beta[CTs].iloc[0][i], 0, error[i], head_width=0.1, head_length=head_length,
        #             length_includes_head=True, color=color, zorder=10)
        

def format_e(x):
    if x > 1e-3:
        x = '%.3f'%(x)
    else:
        x = draw.format_e('%.2e'%(x))
    return x 

# 
P = pd.read_table(snakemake.input.P, sep='\t', index_col=0)
out = np.load(snakemake.input.out, allow_pickle=True).item()
Vs = np.diagonal(out['he']['free']['V'], axis1=1, axis2=2)
C = len(Vs[0])
CTs = P.columns.tolist()
CTs = np.array([re.sub(' ', '_', ct) for ct in CTs])
ct_order = ['CD4_NC', 'CD8_ET', 'NK', 'CD8_NC', 'B_IN', 'CD4_ET', 'B_Mem', 'Mono_C', 'CD8_S100B', 'Mono_NC', 'NK_R', 'DC', 'CD4_SOX4', 'Plasma']
ordered_CTs = [ct for ct in ct_order if ct in CTs]

Vs = pd.DataFrame(Vs, columns=CTs)[ordered_CTs]
Vs['gene'] = out['gene']
Vs['hom2'] = out['he']['free']['hom2']

genes = snakemake.params.genes

betas = pd.DataFrame(out['he']['free']['beta']['ct_beta'], columns=CTs)[ordered_CTs]
betas['gene'] = out['gene']

ctp = pd.read_table(snakemake.input.ctp)
ctp['ct'] = ctp['ct'].str.replace(' ', '_')
ctp[['ct'] + genes].to_csv(snakemake.output.data, sep='\t', index=False)

var = pd.read_table(snakemake.input.var, index_col=0)

# plot
mpl.rcParams['font.size'] = 9

fig, axes = plt.subplots(ncols=len(genes), figsize=(10, 4), dpi=600)


for gene, ax in zip(genes, axes):
    gene_Vs = Vs.loc[Vs['gene'] == gene]
    print(gene_Vs)
    if gene_Vs.shape[0] != 1:
        sys.exit('Missing gene!\n')

    gene_betas = betas.loc[betas['gene'] == gene]

    sns.violinplot(data=ctp, x='ct', y=gene, order=ordered_CTs, inner=None, color=".8", scale='width', ax=ax)

    error2(ax, gene_Vs, gene_betas, ordered_CTs, label='HE', shift=0, color=snakemake.params.mycolors[0], plot_negative=True, negative_variance_warn=False)

    ax.set_xlabel('Cell types')
    ax.set_ylabel('Cell type-specific pseudobulk', fontsize=11)
    all_genes = out['gene']
    p_var = out['he']['wald']['free']['V'][all_genes == gene][0]
    p_mean = out['he']['wald']['free']['ct_beta'][all_genes == gene][0]
    if gene == genes[0]:
        ax.text(0.02, 0.02, 
                f'p(variance differentiation)={format_e(p_var)}\np(mean differentiation)={format_e(p_mean)}',
                fontsize=9, transform=ax.transAxes)
    else:
        ax.text(0.02, 0.9, 
                f'p(variance differentiation)={format_e(p_var)}\np(mean differentiation)={format_e(p_mean)}',
                fontsize=9, transform=ax.transAxes)

    gene_var = var.loc[var.index == gene]
    if gene_var.shape[0] == 1:
        gene = gene_var['GeneSymbol'].tolist()[0]
    ax.set_title(gene)
    # ax.legend()

fig.tight_layout(w_pad=3)

fig.savefig(snakemake.output.png)
