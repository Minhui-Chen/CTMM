import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plot_help

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
    op_hom = np.load(snakemake.input.op_hom[0], allow_pickle=True).item()
    ctp_hom = np.load(snakemake.input.ctp_hom[0], allow_pickle=True).item()
    arg = snakemake.params.arg

    label_f = lambda x: x.split('_')[2]
    free_labels = ['Hom']+[label_f(x) for x in snakemake.params.free]

    plot_order = np.array(snakemake.params.plot_order['free'][arg])
    plot_order = np.array( ['Hom']+[label_f(x) for x in plot_order] )
    plot_order_tmp = plot_order[np.isin(plot_order, free_labels)]

    trueV_d = {label:np.diag(np.loadtxt(V_f)) for label, V_f in zip(free_labels, snakemake.input.V_hom+snakemake.input.V)}
    trueV = [trueV_d[label] for label in plot_order_tmp]

    op_free_ml = pd.concat([get_V(out, label, method='ml') for out, label in zip([op_hom]+op_free, free_labels)], 
            ignore_index=True)
    op_free_ml['label'] = pd.Categorical(op_free_ml['label'], plot_order_tmp)

    op_free_reml = pd.concat([get_V(out, label, method='reml') for out, label in zip([op_hom]+op_free, free_labels)], 
            ignore_index=True)
    op_free_reml['label'] = pd.Categorical(op_free_reml['label'], plot_order_tmp)

    op_free_he = pd.concat([get_V(out, label, method='he') for out, label in zip([op_hom]+op_free, free_labels)], 
            ignore_index=True)
    op_free_he['label'] = pd.Categorical(op_free_he['label'], plot_order_tmp)
    # cap at -30 to 30
    op_free_he.loc[op_free_he['CT-specific variance'] > 20, 'CT-specific variance'] = 20
    #op_free_he.loc[op_free_he['CT-specific variance'] < -30, 'CT-specific variance'] = -30

    ctp_free_ml = pd.concat([get_V(out,label,method='ml') for out, label in zip([ctp_hom]+ctp_free, free_labels)], 
            ignore_index=True)
    ctp_free_ml['label'] = pd.Categorical(ctp_free_ml['label'], plot_order_tmp)
    # cap at -10 to 10
    #ctp_free_ml.loc[ctp_free_ml['CT-specific variance'] > 15, 'CT-specific variance'] = 15
    #ctp_free_ml.loc[ctp_free_ml['CT-specific variance'] < -3, 'CT-specific variance'] = -3

    ctp_free_reml = pd.concat([get_V(out, label, method='reml') for out, label in zip([ctp_hom]+ctp_free, free_labels)], 
            ignore_index=True)
    ctp_free_reml['label'] = pd.Categorical(ctp_free_reml['label'], plot_order_tmp)
    # cap at -10 to 10
    #ctp_free_reml.loc[ctp_free_reml['CT-specific variance'] > 15, 'CT-specific variance'] = 15
    #ctp_free_reml.loc[ctp_free_reml['CT-specific variance'] < -3, 'CT-specific variance'] = -3

    ctp_free_he = pd.concat([get_V(out, label, method='he') for out,label in zip([ctp_hom]+ctp_free,free_labels)], 
            ignore_index=True)
    ctp_free_he['label'] = pd.Categorical(ctp_free_he['label'], plot_order_tmp)
    # cap at -10 to 10
    ctp_free_he.loc[ctp_free_he['CT-specific variance'] > 10, 'CT-specific variance'] = 10
    ctp_free_he.loc[ctp_free_he['CT-specific variance'] < -5, 'CT-specific variance'] = -5

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
    #trueV_ = list( trueV[0] ) * len(np.unique(op_free_ml['label'])) 
    trueV_ = np.array(trueV).flatten()
    C = len(np.unique(op_free_ml['CT']))
    #print(op_free_ml)
    xs = plot_help.snsbox_get_x(len(np.unique(op_free_ml['label'])), C)
    print(trueV_)
    for ax in axes.flatten():
        ax.scatter(xs, trueV_, color=snakemake.params.pointcolor, zorder=10, )

    #axes[0,0].legend().set_visible(False)
    axes[0,0].legend(fontsize=5)
    for ax in axes.flatten()[1:]:
        ax.legend().set_visible(False)

    for ax in axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')
    axes[-1,0].set_xlabel(r'$\bar{V}$')
    axes[-1,1].set_xlabel(r'$\bar{V}$')
    axes[0,0].set_ylabel('ML')
    axes[1,0].set_ylabel('REML')
    axes[2,0].set_ylabel('HE')
    plt.figtext(0.02, 0.5, 'CT-specific variance', horizontalalignment='center', 
            verticalalignment='center', rotation='vertical', fontsize=12)

    #fig.tight_layout()
    fig.savefig(snakemake.output.png)

if __name__ == '__main__':
    main()
