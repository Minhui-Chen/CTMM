import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu


def compute_mean(cty, P, gene):
    cty = cty[gene].unstack()
    cty = cty.sort_index().sort_index(axis=1)

    # sanity check
    if cty.index.equals(P.index) and cty.columns.equals(P.columns):
        pass
    else:
        sys.exit("Not mathcing order!")

    op = (cty * P).sum(axis=1)

    return op.mean(), op.std(), cty.mean()


def main():
    # 
    top = np.loadtxt(snakemake.input.top, dtype='str')
    bot = np.loadtxt(snakemake.input.bot, dtype='str')
    cty = pd.read_table(snakemake.input.cty, index_col=(0, 1))
    P = pd.read_table(snakemake.input.P, index_col=0)
    remlJK = np.load(snakemake.input.remlJK, allow_pickle=True).item()
    genes = remlJK['gene']
    beta_ps = remlJK['reml']['wald']['free']['ct_beta']
    cts = P.columns
    C = P.shape[1]

    # sort P
    P = P.sort_index().sort_index(axis=1)

    # statistics
    top_op_means = [compute_mean(cty, P, gene)[0] for gene in top]
    top_op_vars = [compute_mean(cty, P, gene)[1] for gene in top]
    top_ct_means = pd.concat([compute_mean(cty, P, gene)[2] for gene in top], axis=1).T
    bot_op_means = [compute_mean(cty, P, gene)[0] for gene in bot]
    bot_op_vars = [compute_mean(cty, P, gene)[1] for gene in bot]
    bot_ct_means = pd.concat([compute_mean(cty, P, gene)[2] for gene in bot], axis=1).T
    top_beta_ps = (-1) * np.log10(beta_ps[np.isin(genes, top)])
    bot_beta_ps = (-1) * np.log10(beta_ps[np.isin(genes, bot)])
    t_means, p_means = ttest_ind(top_op_means, bot_op_means, equal_var=False)
    t_vars, p_vars = ttest_ind(top_op_vars, bot_op_vars, equal_var=False)
    p_ct_means = [ttest_ind(top_ct_means[ct], bot_ct_means[ct], equal_var=False)[1] 
                            for ct in cts]
    u_beta_ps, p_beta_ps = mannwhitneyu(top_beta_ps, bot_beta_ps)

    # 
    op_means = pd.DataFrame({'top': top_op_means, 'bottom': bot_op_means})
    op_vars = pd.DataFrame({'top': top_op_vars, 'bottom': bot_op_vars})
    beta_ps = pd.DataFrame({'top': top_beta_ps, 'bottom': bot_beta_ps})
    top_ct_means = top_ct_means.melt(var_name='cell type', value_name='ct mean')
    top_ct_means['gene'] = 'top'
    bot_ct_means = bot_ct_means.melt(var_name='cell type', value_name='ct mean')
    bot_ct_means['gene'] = 'bottom'
    ct_means = pd.concat([top_ct_means, bot_ct_means])

    # plot
    # overall means
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=600)

    sns.violinplot(op_means, ax=axes[0, 0])
    axes[0, 0].text(0.02, .95, f'p = {p_means:.2e}', fontsize=9, transform=axes[0, 0].transAxes)
    axes[0, 0].text(-0.05, 1.05, '(A)', fontsize=12, transform=axes[0, 0].transAxes)
    axes[0, 0].set_ylabel('Average OP across individuals', fontsize=10)
    axes[0, 0].set_xlabel('Genes', fontsize=10)


    # cell type specific cty
    p_ct_means = ['%.2e'%(p) for p in p_ct_means]
    sns.violinplot(data=ct_means, x='cell type', y='ct mean', hue='gene', ax=axes[0, 1])
    axes[0, 1].text(0.02, .95, f'p = {", ".join(p_ct_means)}', fontsize=9, transform=axes[0, 1].transAxes)
    axes[0, 1].text(-0.05, 1.05, '(B)', fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].set_ylabel('Average CTP across individuals', fontsize=10)
    axes[0, 1].set_xlabel('Cell types', fontsize=10)


    # variance of OP
    sns.violinplot(op_vars, ax=axes[1, 0])
    axes[1, 0].text(0.02, 0.95, f'p = {p_vars:.2e}', fontsize=9, transform=axes[1, 0].transAxes)
    axes[1, 0].text(-0.05, 1.05, '(C)', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].set_xlabel('Genes', fontsize=10)
    axes[1, 0].set_ylabel('Standard deviation of OP across individuals', fontsize=10)


    # p for cell type specific beta
    sns.violinplot(beta_ps, ax=axes[1, 1])
    axes[1, 1].text(0.02, .95, f'p = {p_beta_ps:.2e}', fontsize=9, transform=axes[1, 1].transAxes)
    axes[1, 1].text(-0.05, 1.05, '(D)', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlabel('Genes', fontsize=10)
    axes[1, 1].set_ylabel('$-log_{10}$p(mean differentiation)', fontsize=10)

    plt.tight_layout()
    fig.savefig(snakemake.output.png)


if __name__ == '__main__':
    main()
    