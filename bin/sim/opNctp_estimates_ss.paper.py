import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ctmm import draw 


def get_V(out, label, method, ):
    model = 'free'
    data = [np.diag(V) for V in out[method][model]['V']]
    C = len(data[0])
    CTs = [f'CT{i}' for i in range(1,C+1)]
    data = pd.DataFrame(data, columns=CTs)
    data = data.melt(var_name='CT', value_name='CT-specific variance')
    data['label'] = str(int(float(label)))
    return( data )

def main():
    #

    # collect data
    op_free = [np.load(f, allow_pickle=True).item() for f in snakemake.input.op_free]
    ctp_free = [np.load(f, allow_pickle=True).item() for f in snakemake.input.ctp_free]
    free_labels = snakemake.params.free
    arg = 'ss'
    plot_order = np.array(snakemake.params.plot_order['free'][arg])
    plot_order = np.array( [str(int(float(x))) for x in plot_order] )
    trueV = [np.diag(np.loadtxt(V_f)) for V_f in snakemake.input.V]
    

    op_free_ml = pd.concat([get_V(out, label, method='ml') for out, label in zip(op_free, free_labels)], ignore_index=True)
    plot_order_tmp = plot_order[np.isin(plot_order, op_free_ml['label'])]
    op_free_ml['label'] = pd.Categorical(op_free_ml['label'], plot_order_tmp)

    op_free_reml = pd.concat([get_V(out, label, method='reml') for out, label in zip(op_free, free_labels)], ignore_index=True)
    op_free_reml['label'] = pd.Categorical(op_free_reml['label'], plot_order_tmp)

    op_free_he = pd.concat([get_V(out, label, method='he') for out, label in zip(op_free, free_labels)], ignore_index=True)
    op_free_he['label'] = pd.Categorical(op_free_he['label'], plot_order_tmp)
    # cap at -30 to 30
    op_free_reml['CT-specific variance'] = op_free_reml['CT-specific variance'].clip(-30, 30)
    op_free_he['CT-specific variance'] = op_free_he['CT-specific variance'].clip(-30, 30)

    ctp_free_ml = pd.concat([get_V(out, label, method='ml') for out, label in zip(ctp_free, free_labels)], ignore_index=True)
    ctp_free_ml['label'] = pd.Categorical(ctp_free_ml['label'], plot_order_tmp)

    ctp_free_reml = pd.concat([get_V(out, label, method='reml') for out, label in zip(ctp_free, free_labels)], ignore_index=True)
    ctp_free_reml['label'] = pd.Categorical(ctp_free_reml['label'], plot_order_tmp)

    ctp_free_he = pd.concat([get_V(out, label, method='he') for out, label in zip(ctp_free, free_labels)], ignore_index=True)
    ctp_free_he['label'] = pd.Categorical(ctp_free_he['label'], plot_order_tmp)
    # cap at -10 to 10
    ctp_free_he['CT-specific variance'] = ctp_free_he['CT-specific variance'].clip(-5, 10)

    # plot
    mpl.rcParams.update({'font.size': 8, 'lines.markersize': mpl.rcParams['lines.markersize']*0.3})
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(6.85,6), dpi=600)

    sns.boxplot(data=op_free_ml, x='label', y='CT-specific variance', hue='CT', ax=axes[0,0], 
            linewidth=0.6, fliersize=1)
    sns.boxplot(data=ctp_free_ml, x='label', y='CT-specific variance', hue='CT', ax=axes[0,1],
            linewidth=0.6, fliersize=1)
    axes[0,0].set_title('OP')
    axes[0,1].set_title('CTP')

    sns.boxplot(data=op_free_reml, x='label', y='CT-specific variance', hue='CT', ax=axes[1,0],
            linewidth=0.6, fliersize=1)
    sns.boxplot(data=ctp_free_reml, x='label', y='CT-specific variance', hue='CT', ax=axes[1,1],
            linewidth=0.6, fliersize=1)

    sns.boxplot(data=op_free_he, x='label', y='CT-specific variance', hue='CT', ax=axes[2,0],
            linewidth=0.6, fliersize=1)
    sns.boxplot(data=ctp_free_he, x='label', y='CT-specific variance', hue='CT', ax=axes[2,1],
            linewidth=0.6, fliersize=1)

    ## add true V
    trueV_ = list( trueV[0] ) * len(np.unique(op_free_ml['label'])) 
    C = len(np.unique(op_free_ml['CT']))
    xs = draw.snsbox_get_x(len(np.unique(op_free_ml['label'])), C)
    for ax in axes.flatten():
        ax.scatter(xs, trueV_, color=snakemake.params.pointcolor, zorder=10, )

    #axes[0,0].legend().set_visible(False)
    axes[0,0].legend(fontsize=5)
    for ax in axes.flatten()[1:]:
        ax.legend().set_visible(False)

    for ax in axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')
    axes[-1,0].set_xlabel('sample size')
    axes[-1,1].set_xlabel('sample size')
    axes[0,0].set_ylabel('ML')
    axes[1,0].set_ylabel('REML')
    axes[2,0].set_ylabel('HE')
    plt.figtext(0.02, 0.5, 'CT-specific variance', horizontalalignment='center', 
            verticalalignment='center', rotation='vertical', fontsize=12)

    #fig.tight_layout()
    fig.savefig(snakemake.output.png)

if __name__ == '__main__':
    main()
