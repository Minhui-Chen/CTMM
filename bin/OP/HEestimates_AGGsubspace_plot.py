import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plot_help

def main():
    # par
    input = snakemake.input
    output = snakemake.output
    params = snakemake.params
    wildcards = snakemake.wildcards 

    # get cell type number
    hom = np.load(input.out[0], allow_pickle=True).item()['ml']['hom']
    C = len(hom['beta']['ct_beta'][0])

    ## collect subspace
    subspace = params.subspace 

    # collect estimates from Hom IID Free
    summaries = {'hom':[], 'iid':[], 'free':[]}
    for arg_, out_f in zip(np.array(subspace[wildcards.arg]), input.out):
        npy = np.load(out_f, allow_pickle=True).item()
        estimates = npy['he']
        # read Hom model estimates
        hom = estimates['hom']
        hom2 = hom['hom2']
        summary = pd.DataFrame({'subject':hom2})
        summary['arg'] = arg_
        summaries['hom'].append(summary)

        # read IID model estimates
        iid = estimates['iid']
        hom2, V = iid['hom2'], iid['V']
        V = np.array([x[0,0] for x in V]) # extract the first diagonal of V
        summary = pd.DataFrame({'subject':hom2})
        summary['V'] = V
        summary['arg'] = arg_
        summaries['iid'].append(summary)

        # read Free model estimates
        free = estimates['free']
        hom2, V = free['hom2'], free['V']
        V = np.array([np.diag(x) for x in V]) # extract out diagonal of V
        V = V.T
        summary = pd.DataFrame({'subject':hom2})
        for i in range(C):
            summary['V_'+str(i+1)] = V[i]
        summary['arg'] = arg_
        summaries['free'].append(summary)

    # concat and sort
    for model in summaries.keys():
        summaries_ = pd.concat(summaries[model], ignore_index=True)
        plot_order_ = np.array(params.plot_order[wildcards.model][wildcards.arg])
        plot_order_ = plot_order_[np.isin(plot_order_, summaries_['arg'])]
        summaries_['arg'] = pd.Categorical(summaries_['arg'], plot_order_)
        summaries_ = summaries_.sort_values('arg').reset_index(drop=True)
        summaries_.to_csv(sys.stdout, sep='\t')
        summaries[model] = summaries_

    # collect true V
    trueV = {}
    for arg_, trueV_f in zip(np.array(subspace[wildcards.arg]), input.V):
        # read true V values
        trueV[arg_] = np.loadtxt(trueV_f)

    # collect True variances explained
    vcs = []
    subspace_ = subspace.copy()  # may don't need to copy
    plot_order_ = np.array(params.plot_order[wildcards.model][wildcards.arg])
    plot_order_ = plot_order_[np.isin(plot_order_, subspace_[wildcards.arg])]
    subspace_[wildcards.arg] = pd.Categorical(subspace_[wildcards.arg], plot_order_)
    subspace_ = subspace_.sort_values(wildcards.arg).reset_index(drop=True)
    for vc in np.array(subspace_['vc']):
        vcs.append(float(vc.split('_')[0]))

    # plot
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)
    ## Hom model
    ### variance components
    sns.boxplot(x='arg', y='subject', data=summaries['hom'], ax=axes[0,0], color=params.mycolors[0])

    ### add True variances to plot
    print(summaries['hom'].head())
    print(summaries['hom']['arg'])
    xs = plot_help.snsbox_get_x(len(np.unique(summaries['hom']['arg'])), 1)
    axes[0,0].scatter(xs, vcs, color=params.pointcolor, zorder=10)

    axes[0,0].xaxis.label.set_visible(False)
    axes[0,0].set_ylabel('$\sigma_{hom}^2$')
    axes[0,0].text(-0.29, 0.95, 'Hom', fontsize=16, transform=axes[0,0].transAxes)
    axes[0,0].text(-0.05, 1.05, '(A)', fontsize=12, transform=axes[0,0].transAxes)
    #handles, labels = axes[0,0].get_legend_handles_labels()
    #axes[0,0].legend(handles=handles, labels=labels)

    axes[0,1].axis('off')

    ## IID model
    ### variance components
    sns.boxplot(x='arg', y='subject', data=summaries['iid'], ax=axes[1,0], color=params.mycolors[0])

    ### add True variances to plot
    xs = plot_help.snsbox_get_x(len(np.unique(summaries['iid']['arg'])), 1)
    axes[1,0].scatter(xs, vcs, color=params.pointcolor, zorder=10)

    axes[1,0].xaxis.label.set_visible(False)
    axes[1,0].set_ylabel('$\sigma_{hom}^2$')
    axes[1,0].text(-0.29, 0.95, 'IID', fontsize=16, transform=axes[1,0].transAxes)
    axes[1,0].text(-0.05, 1.05, '(B)', fontsize=12, transform=axes[1,0].transAxes)
    #handles, labels = axes[1,0].get_legend_handles_labels()
    #axes[1,0].legend(handles=handles, labels=labels)

    ### V
    sns.boxplot(x='arg', y='V', data=summaries['iid'], ax=axes[1,1], color=params.mycolors[0])
    #### add true V
    trueV_ = np.array([np.diag(trueV[x]) for x in pd.unique(summaries['iid']['arg'])]).T.flatten()
    xs = plot_help.snsbox_get_x(len(np.unique(summaries['iid']['arg'])), 1)
    xs = list(xs) * C
    axes[1,1].scatter(xs, trueV_, color=params.pointcolor, zorder=10)
    axes[1,1].xaxis.label.set_visible(False)
    axes[1,1].set_ylabel('V_diag (cell type-specific genetic variance)')
    axes[1,1].text(-0.05, 1.05, '(C)', fontsize=12, transform=axes[1,1].transAxes)

    ## Free model
    ### variance components
    sns.boxplot(x='arg', y='subject', data=summaries['free'], ax=axes[2,0], color=params.mycolors[0])

    #### add True variances
    xs = plot_help.snsbox_get_x(len(np.unique(summaries['free']['arg'])), 1)
    axes[2,0].scatter(xs, vcs, color=params.pointcolor, zorder=10)

    axes[2,0].xaxis.label.set_visible(False)
    axes[2,0].set_ylabel('$\sigma_{hom}^2$')
    axes[2,0].text(-0.29, 0.95, 'Free', fontsize=16, transform=axes[2,0].transAxes)
    axes[2,0].text(-0.05, 1.05, '(D)', fontsize=12, transform=axes[2,0].transAxes)
    #handles, labels = axes[2,0].get_legend_handles_labels()
    #axes[2,0].legend(handles=handles, labels=labels)

    ### V
    V_df =  pd.melt(summaries['free'], id_vars=['arg'], value_vars=['V_'+str(i+1) for i in range(C)])
    sns.boxplot(x='arg', y='value', hue='variable', data=V_df, ax=axes[2,1], palette=params.colorpalette)
    #### add true sig gam
    trueV_ = np.array([np.diag(trueV[x]) for x in pd.unique(V_df['arg'])]).flatten()
    xs = plot_help.snsbox_get_x(len(np.unique(V_df['arg'])), len(np.unique(V_df['variable'])))
    axes[2,1].scatter(xs, trueV_, color=params.pointcolor, zorder=10)
    axes[2,1].xaxis.label.set_visible(False)
    axes[2,1].set_ylabel('V_diag (cell type-specific genetic variance')
    axes[2,1].text(-0.05, 1.05, '(E)', fontsize=12, transform=axes[2,1].transAxes)
    handles, labels = axes[2,1].get_legend_handles_labels()
    axes[2,1].legend(handles=handles, labels=[r'$%s$'%(x) for x in labels])

    #### tweak x labels
    if len(summaries['free']['arg'].values[0]) > 15:
        for ax in axes.flatten():
            ax.tick_params(axis='x', labelsize='small', labelrotation=15)
    fig.tight_layout()
    fig.savefig(output.png)

if __name__ == '__main__':
    main()
