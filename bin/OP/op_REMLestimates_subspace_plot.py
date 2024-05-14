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
    subspace = params.subspace

    #
    # get cell type number
    hom = np.load(input.out[0], allow_pickle=True).item()['ml']['hom']
    C = len(hom['beta']['ct_beta'][0])

    # plot order
    plot_order = np.array(params.plot_order[wildcards.model][wildcards.arg])

    # collect estimates from Hom IID Free Full
    summaries = {'hom':[], 'iid':[], 'free':[], 'full':[]}
    for arg_, out_f in zip(np.array(subspace[wildcards.arg]), input.out):
        npy = np.load(out_f, allow_pickle=True).item()
        estimates = npy['reml']
        # read Hom model estimates
        hom = estimates['hom']
        hom2, cellspecific_var = hom['hom2'], hom['cellspecific_var']
        summary = pd.DataFrame({'subject':hom2, 
            'celltype-genetic':np.zeros(len(hom2)), 'cell-specific':cellspecific_var})
        summary['arg'] = arg_
        summaries['hom'].append(summary)

        # read IID model estimates
        iid = estimates['iid']
        hom2, V, interxn_var, cellspecific_var = iid['hom2'], iid['V'], iid['interxn_var'], iid['cellspecific_var']
        V = np.array([x[0,0] for x in V]) # extract the first diagonal of V
        summary = pd.DataFrame({'subject':hom2, 
            'celltype-genetic':interxn_var, 'cell-specific':cellspecific_var})
        summary['V'] = V
        summary['arg'] = arg_
        summaries['iid'].append(summary)

        # read Free model estimates
        free = estimates['free']
        hom2, V, interxn_var, cellspecific_var = free['hom2'], free['V'],free['interxn_var'], free['cellspecific_var']
        V = np.array([np.diag(x) for x in V]) # extract out diagonal of V
        V = V.T
        summary = pd.DataFrame({'subject':hom2, 
            'celltype-genetic':interxn_var, 'cell-specific':cellspecific_var})
        for i in range(C):
            summary['V_'+str(i+1)] = V[i]
        summary['arg'] = arg_
        summaries['free'].append(summary)

        # read Full model estimates
        full = estimates['full']
        V, interxn_var, cellspecific_var = full['V'], full['interxn_var'], full['cellspecific_var']
        V_diag = np.array([np.diag(x) for x in V])
        V_tril = np.array([x[np.tril_indices(C, k=-1)].flatten() for x in V])
        V_diag, V_tril = V_diag.T, V_tril.T
        summary = pd.DataFrame({'subject':np.zeros(len(interxn_var)), 
            'celltype-genetic':interxn_var, 'cell-specific':cellspecific_var})
        for i in range(C):
            summary['V_'+str(i+1)] = V_diag[i]
        for j in range(V_tril.shape[0]):
            summary['Vlow_'+str(j+1)] = V_tril[j]
        summary['arg'] = arg_
        summaries['full'].append(summary)

    # concat and sort
    for model in ['hom', 'iid', 'free', 'full']:
        summaries_ = pd.concat(summaries[model], ignore_index=True)
        plot_order_ = plot_order[np.isin(plot_order, summaries_['arg'])]
        summaries_ = summaries_.loc[summaries_['arg'].isin(plot_order_)]
        summaries_['arg'] = pd.Categorical(summaries_['arg'], plot_order_)
        summaries_ = summaries_.sort_values('arg').reset_index(drop=True)
        summaries_.to_csv(sys.stdout, sep='\t')
        summaries[model] = summaries_

    # collect true beta and V
    trueV = {}
    for arg_, trueV_f in zip(np.array(subspace[wildcards.arg]), input.V):
        # read true V values
        trueV[arg_] = np.loadtxt(trueV_f)

    # collect True variances explained
    vcs = []
    plot_order_ = plot_order[np.isin(plot_order, subspace[wildcards.arg])]
    subspace = subspace.loc[subspace[wildcards.arg].isin(plot_order_)]
    subspace[wildcards.arg] = pd.Categorical(subspace[wildcards.arg], plot_order_)
    subspace = subspace.sort_values(wildcards.arg).reset_index(drop=True)
    for vc in np.array(subspace['vc']):
        vcs = vcs+ [float(vc.split('_')[0])] + [float(x) for x in vc.split('_')[2:]] # get rid of cell type main effect

    # plot
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), sharex=True)
    ## Hom model
    ### variance components
    var = pd.melt(summaries['hom'], id_vars=['arg'],
            value_vars=['subject', 'celltype-genetic', 'cell-specific'])
    var.to_csv(sys.stdout, sep='\t')
    sns.boxplot(x='arg', y='value', hue='variable', data=var, ax=axes[0,0], palette=params.colorpalette)

    ### add True variances to plot
    xs = plot_help.snsbox_get_x(len(np.unique(var['arg'])), len(np.unique(var['variable'])))
    axes[0,0].scatter(xs, vcs, color=params.pointcolor, zorder=10)

    axes[0,0].xaxis.label.set_visible(False)
    axes[0,0].set_ylabel('Proportion of variance')
    axes[0,0].text(-0.29, 0.95, 'Hom', fontsize=20, transform=axes[0,0].transAxes)
    axes[0,0].text(-0.05, 1.05, '(A)', fontsize=16, transform=axes[0,0].transAxes)
    handles, labels = axes[0,0].get_legend_handles_labels()
    axes[0,0].legend(handles=handles, labels=labels)

    axes[0,1].axis('off')
    axes[0,2].axis('off')

    ## IID model
    ### variance components
    var = pd.melt(summaries['iid'], id_vars=['arg'],
            value_vars=['subject', 'celltype-genetic', 'cell-specific'])
    var.to_csv(sys.stdout, sep='\t')
    sns.boxplot(x='arg', y='value', hue='variable', data=var, ax=axes[1,0], palette=params.colorpalette)

    ### add True variances to plot
    xs = plot_help.snsbox_get_x(len(np.unique(var['arg'])), len(np.unique(var['variable'])))
    axes[1,0].scatter(xs, vcs, color=params.pointcolor, zorder=10)

    axes[1,0].xaxis.label.set_visible(False)
    axes[1,0].set_ylabel('Proportion of variance')
    axes[1,0].text(-0.29, 0.95, 'IID', fontsize=20, transform=axes[1,0].transAxes)
    axes[1,0].text(-0.05, 1.05, '(B)', fontsize=16, transform=axes[1,0].transAxes)
    handles, labels = axes[1,0].get_legend_handles_labels()
    axes[1,0].legend(handles=handles, labels=labels)

    ### V
    sns.boxplot(x='arg', y='V', data=summaries['iid'], ax=axes[1,1], color=params.mycolors[0])
    #### add true V
    trueV_ = np.array([np.diag(trueV[x]) for x in pd.unique(summaries['iid']['arg'])]).T.flatten()
    xs = plot_help.snsbox_get_x(len(np.unique(summaries['iid']['arg'])), 1)
    xs = list(xs) * C
    axes[1,1].scatter(xs, trueV_, color=params.pointcolor, zorder=10)
    axes[1,1].xaxis.label.set_visible(False)
    axes[1,1].set_ylabel('V_diag (cell type-specific genetic variance)')
    axes[1,1].text(-0.05, 1.05, '(C)', fontsize=16, transform=axes[1,1].transAxes)
    axes[1,2].axis('off')

    ## Free model
    ### variance components
    var = pd.melt(summaries['free'], id_vars=['arg'],
            value_vars=['subject', 'celltype-genetic', 'cell-specific'])
    var.to_csv(sys.stdout, sep='\t')
    sns.boxplot(x='arg', y='value', hue='variable', data=var, ax=axes[2,0], palette=params.colorpalette)

    #### add True variances
    xs = plot_help.snsbox_get_x(len(np.unique(var['arg'])), len(np.unique(var['variable'])))
    axes[2,0].scatter(xs, vcs, color=params.pointcolor, zorder=10)

    axes[2,0].xaxis.label.set_visible(False)
    axes[2,0].set_ylabel('Proportion of variance')
    axes[2,0].text(-0.29, 0.95, 'Free', fontsize=20, transform=axes[2,0].transAxes)
    axes[2,0].text(-0.05, 1.05, '(D)', fontsize=16, transform=axes[2,0].transAxes)
    handles, labels = axes[2,0].get_legend_handles_labels()
    axes[2,0].legend(handles=handles, labels=labels)

    ### V
    V_df =  pd.melt(summaries['free'], id_vars=['arg'], value_vars=['V_'+str(i+1) for i in range(C)])
    sns.boxplot(x='arg', y='value', hue='variable', data=V_df, ax=axes[2,1], palette=params.colorpalette)
    #### add true sig gam
    trueV_ = np.array([np.diag(trueV[x]) for x in pd.unique(V_df['arg'])]).flatten()
    xs = plot_help.snsbox_get_x(len(np.unique(V_df['arg'])), len(np.unique(V_df['variable'])))
    axes[2,1].scatter(xs, trueV_, color=params.pointcolor, zorder=10)
    axes[2,1].xaxis.label.set_visible(False)
    axes[2,1].set_ylabel('V_diag (cell type-specific genetic variance')
    axes[2,1].text(-0.05, 1.05, '(E)', fontsize=16, transform=axes[2,1].transAxes)
    handles, labels = axes[2,1].get_legend_handles_labels()
    axes[2,1].legend(handles=handles, labels=[r'$%s$'%(x) for x in labels])
    axes[2,2].axis('off')

    ## Full model
    ### variance components
    var = pd.melt(summaries['full'], id_vars=['arg'],
            value_vars=['subject', 'celltype-genetic', 'cell-specific'])
    var.to_csv(sys.stdout, sep='\t')
    sns.boxplot(x='arg', y='value', hue='variable', data=var, ax=axes[3,0], palette=params.colorpalette)

    ### add True variances to plot
    xs = plot_help.snsbox_get_x(len(np.unique(var['arg'])), len(np.unique(var['variable'])))
    axes[3,0].scatter(xs, vcs, color=params.pointcolor, zorder=10)

    axes[3,0].set_xlabel(wildcards.arg)
    axes[3,0].set_ylabel('Proportion of variance')
    axes[3,0].text(-0.29, 0.95, 'Full', fontsize=20, transform=axes[3,0].transAxes)
    axes[3,0].text(-0.05, 1.05, '(F)', fontsize=16, transform=axes[3,0].transAxes)
    handles, labels = axes[3,0].get_legend_handles_labels()
    axes[3,0].legend(handles=handles, labels=labels)

    ### V
    #### diag
    V_df =  pd.melt(summaries['full'], id_vars=['arg'], value_vars=['V_'+str(i+1) for i in range(C)])
    sns.boxplot(x='arg', y='value', hue='variable', data=V_df, ax=axes[3,1], palette=params.colorpalette)
    #### add true V
    trueV_ = np.array([np.diag(trueV[x]) for x in pd.unique(V_df['arg'])]).flatten()
    xs = plot_help.snsbox_get_x(len(np.unique(V_df['arg'])), len(np.unique(V_df['variable'])))
    axes[3,1].scatter(xs, trueV_, color=params.pointcolor, zorder=10)
    axes[3,1].set_xlabel(wildcards.arg)
    axes[3,1].set_ylabel('V_diag (cell type-specific genetic variance')
    axes[3,1].text(-0.05, 1.05, '(G)', fontsize=16, transform=axes[3,1].transAxes)
    handles, labels = axes[3,1].get_legend_handles_labels()
    axes[3,1].legend(handles=handles, labels=[r'$%s$'%(x) for x in labels])

    #### non-diag
    Vlow_df = pd.melt(summaries['full'], id_vars=['arg'],value_vars=['Vlow_'+str(i+1) for i in range(C*(C-1)//2)])
    sns.boxplot(x='arg', y='value', hue='variable', data=Vlow_df, ax=axes[3,2], palette=params.colorpalette)
    #### add true V
    trueVlow_ = np.array([trueV[x][np.tril_indices(C,k=-1)].flatten() for x in pd.unique(V_df['arg'])]).flatten()
    xs = plot_help.snsbox_get_x(len(np.unique(Vlow_df['arg'])), len(np.unique(Vlow_df['variable'])))
    axes[3,2].scatter(xs, trueVlow_, color=params.pointcolor, zorder=10)
    axes[3,2].set_xlabel(wildcards.arg)
    axes[3,2].set_ylabel('V_lowtri (cell type-specific genetic covariance')
    axes[3,2].text(-0.05, 1.05, '(H)', fontsize=16, transform=axes[3,2].transAxes)
    handles, labels = axes[3,2].get_legend_handles_labels()
    axes[3,2].legend(handles=handles, labels=labels)
    #### add dash lines
    for ax in [axes[0,0], axes[1,0], axes[1,1], axes[2,0], axes[2,1], 
            axes[3,0], axes[3,1], axes[3,2]]:
        ax.axhline(0, c='0.8', ls='--', zorder=0)
    for i in range(4):
        axes[i,0].axhline(0.25, c='0.8', ls='--', zorder=0)

    #### tweak x labels
    if len(summaries['full']['arg'].values[0]) > 15:
        for ax in axes.flatten():
            ax.tick_params(axis='x', labelsize='small', labelrotation=15)
    fig.tight_layout()
    fig.savefig(output.png)

if __name__ == '__main__':
    main()
