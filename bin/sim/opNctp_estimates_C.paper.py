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
    data['label'] = label
    return( data )

def main():
    #

    # collect data
    op_free = [np.load(f, allow_pickle=True).item() for f in snakemake.input.op_free]
    ctp_free = [np.load(f, allow_pickle=True).item() for f in snakemake.input.ctp_free]
#     arg = 'C'

#     label_f = lambda x: '%.2f'%(float(x.split('_')[0])/(np.array([float(x_) for x_ in x.split('_')]).sum()))
#     free_labels = [label_f(x) for x in snakemake.params.free]
    free_labels = snakemake.params.free

    plot_order = snakemake.params.plot_order
#     plot_order = np.array( [label_f(x) for x in plot_order] )
    plot_order = np.array(plot_order)
    plot_order_tmp = plot_order[np.isin(plot_order, free_labels)]

    trueV_d = {label:np.diag(np.loadtxt(V_f)) for label, V_f in zip(free_labels, snakemake.input.V)}
    trueV = [trueV_d[label] for label in plot_order_tmp]
    

    op_free_ml = pd.concat([get_V(out, label, method='ml') for out, label in zip(op_free, free_labels)], 
            ignore_index=True)
    op_free_ml['label'] = pd.Categorical(op_free_ml['label'], plot_order_tmp)

    op_free_reml = pd.concat([get_V(out, label, method='reml') for out, label in zip(op_free, free_labels)], ignore_index=True)
    op_free_reml['label'] = pd.Categorical(op_free_reml['label'], plot_order_tmp)

    op_free_he = pd.concat([get_V(out, label, method='he') for out, label in zip(op_free, free_labels)], ignore_index=True)
    op_free_he['label'] = pd.Categorical(op_free_he['label'], plot_order_tmp)
    # cap at -30 to 30
    op_free_he['CT-specific variance'] = op_free_he['CT-specific variance'].clip(-50, 50)

    ctp_free_ml = pd.concat([get_V(out, label, method='ml') for out, label in zip(ctp_free, free_labels)], ignore_index=True)
    ctp_free_ml['label'] = pd.Categorical(ctp_free_ml['label'], plot_order_tmp)
    # cap at -10 to 10
#     ctp_free_ml['CT-specific variance'] = ctp_free_ml['CT-specific variance'].clip(upper=5)

    ctp_free_reml = pd.concat([get_V(out, label, method='reml') for out, label in zip(ctp_free, free_labels)], ignore_index=True)
    ctp_free_reml['label'] = pd.Categorical(ctp_free_reml['label'], plot_order_tmp)
    # cap at -10 to 10
#     ctp_free_reml['CT-specific variance'] = ctp_free_reml['CT-specific variance'].clip(-1, 5)

    ctp_free_he = pd.concat([get_V(out, label, method='he') for out, label in zip(ctp_free, free_labels)], ignore_index=True)
    ctp_free_he['label'] = pd.Categorical(ctp_free_he['label'], plot_order_tmp)
    # cap at -10 to 10
    ctp_free_he['CT-specific variance'] = ctp_free_he['CT-specific variance'].clip(-10, 20)

    # plot
    mpl.rcParams.update({'font.size': 8, 'lines.markersize': mpl.rcParams['lines.markersize']*0.3})
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(6.85,6), dpi=600)
    #fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(12,12), dpi=600)

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
    #ctp_free_he_tmp = ctp_free_he.copy()
    #ctp_free_he_tmp.loc[ctp_free_he_tmp['CT-specific variance'] > 20, 'CT-specific variance'] = 20
    #ctp_free_he_tmp.loc[ctp_free_he_tmp['CT-specific variance'] < -20, 'CT-specific variance'] = -20
    sns.boxplot(data=ctp_free_he, x='label', y='CT-specific variance', hue='CT', ax=axes[2,1],
            linewidth=0.6, fliersize=1)
    #axins = axes[2,1].inset_axes([0.6, 0.6, 0.37, 0.37])
    #ctp_free_he_tmp2 = ctp_free_he.loc[ctp_free_he['label']=='0.10']
    #ctp_free_he_tmp2['label'] = ctp_free_he_tmp2['label'].cat.remove_unused_categories()
    #sns.boxplot(data=ctp_free_he_tmp2, x='label', y='CT-specific variance', hue='CT', ax=axins,
    #        linewidth=0.6, fliersize=1)
    #axins.set_xlabel('')
    #axins.set_ylabel('')
    #axins.legend().set_visible(False)

    ## add true V
    #trueV_ = list( trueV[0] ) * len(np.unique(op_free_ml['label'])) 
    C = len(np.unique(op_free_ml['CT']))
    trueV_ = []
    for i in range(len(trueV)):
        c = len(trueV[i])
        V = trueV[i].tolist() + [0] * (C - c)
        trueV_.append(V)
    trueV_ = np.array(trueV_).flatten()

    #print(op_free_ml)
    #print(op_free_ml['label'][:10])
    #print(np.unique(op_free_ml['label']))
    xs = draw.snsbox_get_x(len(np.unique(op_free_ml['label'])), C)
    for ax in axes.flatten():
        ax.scatter(xs[trueV_ != 0], trueV_[trueV_ != 0], color=snakemake.params.pointcolor, zorder=10, )

    #axes[0,0].legend().set_visible(False)
    axes[0,0].legend(fontsize=5)
    for ax in axes.flatten():
        ax.legend().set_visible(False)

    for ax in axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')
    axes[-1,0].set_xlabel('Number of cell types')
    axes[-1,1].set_xlabel('Number of cell types')
    axes[0,0].set_ylabel('ML')
    axes[1,0].set_ylabel('REML')
    axes[2,0].set_ylabel('HE')
    plt.figtext(0.02, 0.5, 'CT-specific variance', horizontalalignment='center', 
            verticalalignment='center', rotation='vertical', fontsize=12)

    #fig.tight_layout()
    fig.savefig(snakemake.output.png)

if __name__ == '__main__':
    main()
