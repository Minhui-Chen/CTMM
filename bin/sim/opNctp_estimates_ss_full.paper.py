import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ctmm import draw


def get_V(out, label, method, ):
    model = 'full'
    V_diag = [np.diag(V) for V in out[method][model]['V']]
    C = len(V_diag[0])
    V_triu = [V[np.triu_indices(C,k=1)] for V in out[method][model]['V']]
    CTs = [f'CT{i}' for i in range(1,C+1)]
    CT_pairs = []
    for i in range(C-1):
        for j in range(i+1,C):
            CT_pairs.append(f'CT{i+1} - CT{j+1}')
    V_diag = pd.DataFrame(V_diag, columns=CTs)
    V_diag = V_diag.melt(var_name='CT', value_name='CT-specific variance')
    V_diag['label'] = str(int(float(label)))
    V_triu = pd.DataFrame(V_triu, columns=CT_pairs)
    V_triu = V_triu.melt(var_name='CT pair', value_name='covariance')
    V_triu['label'] = str(int(float(label)))
    return( V_diag, V_triu )

def main():
    #

    # collect data
    op_full = [np.load(f, allow_pickle=True).item() for f in snakemake.input.op_full]
    ctp_full = [np.load(f, allow_pickle=True).item() for f in snakemake.input.ctp_full]
    full_labels = snakemake.params.full
    arg = 'ss'
    plot_order = np.array(snakemake.params.plot_order['full'][arg])
    plot_order = np.array( [str(int(float(x))) for x in plot_order] )
    trueV_diag = [np.diag(np.loadtxt(V_f)) for V_f in snakemake.input.V]
    C = len(trueV_diag[0])
    trueV_triu = [np.loadtxt(V_f)[np.triu_indices(C,k=1)] for V_f in snakemake.input.V]
    

    op_full_Vtriu_ml = pd.concat([get_V(out, label, method='ml')[1] for out, label in zip(op_full, full_labels)], ignore_index=True)
    plot_order_tmp = plot_order[np.isin(plot_order, op_full_Vtriu_ml['label'])]
    op_full_Vtriu_ml['label'] = pd.Categorical(op_full_Vtriu_ml['label'], plot_order_tmp)
    # cap at -30 to 30
    op_full_Vtriu_ml.loc[op_full_Vtriu_ml['covariance'] > 10, 'covariance'] = 10
    op_full_Vtriu_ml.loc[op_full_Vtriu_ml['covariance'] < -10, 'covariance'] = -10

    op_full_Vtriu_reml = pd.concat([get_V(out, label, method='reml')[1] for out, label in zip(op_full, full_labels)], ignore_index=True)
    op_full_Vtriu_reml['label'] = pd.Categorical(op_full_Vtriu_reml['label'], plot_order_tmp)
    # cap at -30 to 30
    op_full_Vtriu_reml.loc[op_full_Vtriu_reml['covariance'] > 10, 'covariance'] = 10
    op_full_Vtriu_reml.loc[op_full_Vtriu_reml['covariance'] < -10, 'covariance'] = -10

    op_full_Vtriu_he = pd.concat([get_V(out, label, method='he')[1] for out, label in zip(op_full, full_labels)], ignore_index=True)
    op_full_Vtriu_he['label'] = pd.Categorical(op_full_Vtriu_he['label'], plot_order_tmp)
    # cap at -30 to 30
    op_full_Vtriu_he.loc[op_full_Vtriu_he['covariance'] > 10, 'covariance'] = 10
    op_full_Vtriu_he.loc[op_full_Vtriu_he['covariance'] < -10, 'covariance'] = -10

    ctp_full_Vtriu_ml = pd.concat([get_V(out, label, method='ml')[1] for out, label in zip(ctp_full, full_labels)], ignore_index=True)
    ctp_full_Vtriu_ml['label'] = pd.Categorical(ctp_full_Vtriu_ml['label'], plot_order_tmp)
    # cap at -10 to 10
    #ctp_full_Vtriu_ml.loc[ctp_full_Vtriu_ml['covariance'] > 2, 'covariance'] = 2
    #ctp_full_Vtriu_ml.loc[ctp_full_Vtriu_ml['covariance'] < -1, 'covariance'] = -1

    ctp_full_Vtriu_reml = pd.concat([get_V(out, label, method='reml')[1] for out, label in zip(ctp_full, full_labels)], ignore_index=True)
    ctp_full_Vtriu_reml['label'] = pd.Categorical(ctp_full_Vtriu_reml['label'], plot_order_tmp)
    # cap at -10 to 10
    #ctp_full_Vtriu_reml.loc[ctp_full_Vtriu_reml['covariance'] > 2, 'covariance'] = 2
    #ctp_full_Vtriu_reml.loc[ctp_full_Vtriu_reml['covariance'] < -1, 'covariance'] = -1

    ctp_full_Vtriu_he = pd.concat([get_V(out, label, method='he')[1] for out, label in zip(ctp_full, full_labels)], ignore_index=True)
    ctp_full_Vtriu_he['label'] = pd.Categorical(ctp_full_Vtriu_he['label'], plot_order_tmp)
    # cap at -10 to 10
    #ctp_full_Vtriu_he.loc[ctp_full_Vtriu_he['covariance'] > 2, 'covariance'] = 2
    #ctp_full_Vtriu_he.loc[ctp_full_Vtriu_he['covariance'] < -1, 'covariance'] = -1

    # plot
    mpl.rcParams.update({'font.size': 8, 'lines.markersize': mpl.rcParams['lines.markersize']*0.3})
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(6.85,6), dpi=600)

    sns.boxplot(data=op_full_Vtriu_ml, x='label', y='covariance', hue='CT pair', ax=axes[0,0], 
            linewidth=0.6, fliersize=1)
    sns.boxplot(data=ctp_full_Vtriu_ml, x='label', y='covariance', hue='CT pair', ax=axes[0,1],
            linewidth=0.6, fliersize=1)
    axes[0,0].set_title('OP')
    axes[0,1].set_title('CTP')

    sns.boxplot(data=op_full_Vtriu_reml, x='label', y='covariance', hue='CT pair', ax=axes[1,0],
            linewidth=0.6, fliersize=1)
    sns.boxplot(data=ctp_full_Vtriu_reml, x='label', y='covariance', hue='CT pair', ax=axes[1,1],
            linewidth=0.6, fliersize=1)

    sns.boxplot(data=op_full_Vtriu_he, x='label', y='covariance', hue='CT pair', ax=axes[2,0],
            linewidth=0.6, fliersize=1)
    sns.boxplot(data=ctp_full_Vtriu_he, x='label', y='covariance', hue='CT pair', ax=axes[2,1],
            linewidth=0.6, fliersize=1)

    ## add true V
    trueV_triu_ = list( trueV_triu[0] + float(snakemake.params.vc.split('_')[2]) ) * len(np.unique(op_full_Vtriu_ml['label'])) 
    xs = draw.snsbox_get_x(len(np.unique(op_full_Vtriu_ml['label'])), (C-1)*C/2)
    for ax in axes.flatten():
        ax.scatter(xs, trueV_triu_, color=snakemake.params.pointcolor, zorder=10, )

    #axes[0,0].legend().set_visible(False)
    axes[0,1].legend(fontsize=5, ncol=2)
    axes[0,0].legend().set_visible(False)
    for ax in axes.flatten()[2:]:
        ax.legend().set_visible(False)

    for ax in axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')
    axes[-1,0].set_xlabel('sample size')
    axes[-1,1].set_xlabel('sample size')
    axes[0,0].set_ylabel('ML')
    axes[1,0].set_ylabel('REML')
    axes[2,0].set_ylabel('HE')
    plt.figtext(0.02, 0.5, 'Covariance between CTs', horizontalalignment='center', 
            verticalalignment='center', rotation='vertical', fontsize=12)

    #fig.tight_layout()
    fig.savefig(snakemake.output.png)

if __name__ == '__main__':
    main()
