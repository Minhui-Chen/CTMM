import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plot_help

def main():
    # par
    input = snakemake.input
    output = snakemake.output 
    params = snakemake.params 
    mycolors = plot_help.mycolors(n=10, palette='muted')

    # collect data
    hom_ss = [np.load(f, allow_pickle=True).item()['ml']['lrt'] for f in input.hom_ss]
    hom_ss_labels = params.hom_ss
    iid_ss = [np.load(f, allow_pickle=True).item()['ml']['lrt'] for f in input.iid_ss]
    iid_ss_labels = params.iid_ss
    free_ss = [np.load(f, allow_pickle=True).item()['ml']['lrt'] for f in input.free_ss]
    free_ss_labels = params.free_ss
    hom_a = [np.load(f, allow_pickle=True).item()['ml']['lrt'] for f in input.hom_a]
    hom_a_labels = params.hom_a
    iid_a = [np.load(f, allow_pickle=True).item()['ml']['lrt'] for f in input.iid_a]
    iid_a_labels = params.iid_a
    free_a = [np.load(f, allow_pickle=True).item()['ml']['lrt'] for f in input.free_a]
    free_a_labels = params.free_a
    iid_vc = [np.load(f, allow_pickle=True).item()['ml']['lrt'] for f in input.iid_vc]
    iid_vc_labels = params.iid_vc
    free_vc = [np.load(f, allow_pickle=True).item()['ml']['lrt'] for f in input.free_vc]
    free_vc_labels = params.free_vc
    free_V_diag = [np.load(f, allow_pickle=True).item()['ml']['lrt'] for f in input.free_V_diag]
    free_V_diag_labels = params.free_V_diag

#    ## combine labels
#    ss_labels = np.unique(np.append(hom_ss_labels, iid_ss_labels, free_ss_labels))
#    a_labels = np.unique(np.append(hom_a_labels, iid_a_labels, free_a_labels))
#    vc_labels = np.unique(np.append(hom_vc_labels, iid_vc_labels, free_vc_labels))
#    ### order labels
#    try:
#        ss_labels = np.take_along_axis(np.argsort(ss_labels.astype('float')))
#    except:
#        pass

    # compute LRT power
    def lrt_power(outs, labels, arg):
        print(labels)
        power = {'Hom vs Null':[], 'IID vs Hom':[], 'Free vs Hom':[], 'Free vs IID':[], 'label': labels}
        for out in outs:
            power['Hom vs Null'].append(np.sum(out['hom_null'] < 0.05) / out['hom_null'].shape[0])
            power['IID vs Hom'].append(np.sum(out['iid_hom'] < 0.05) / out['iid_hom'].shape[0])
            power['Free vs Hom'].append(np.sum(out['free_hom'] < 0.05) / out['free_hom'].shape[0])
            power['Free vs IID'].append(np.sum(out['free_iid'] < 0.05) / out['free_iid'].shape[0])
        power = pd.DataFrame(power)
        print(power)
        
        # shift overlapping lines
        columns = power.columns.to_list()
        columns.remove('label')
        for i in range(len(columns)-1):
            for j in range(i+1, len(columns)):
                if np.all(power[columns[i]] == power[columns[j]]):
                    power[columns[i]] = power[columns[i]] + 0.01
                    power[columns[j]] = power[columns[j]] - 0.01
        
        plot_order = np.array(params.plot_order['free'][arg])
        plot_order = plot_order[np.isin(plot_order, labels)]
        power['label'] = pd.Categorical(power['label'], plot_order)

        power = pd.melt(power, id_vars=['label'], value_vars=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], 
                var_name='LRT', value_name='Positive rate')
        return power

    hom_ss_power = lrt_power(hom_ss, hom_ss_labels, 'ss')
    iid_ss_power = lrt_power(iid_ss, iid_ss_labels, 'ss')
    free_ss_power = lrt_power(free_ss, free_ss_labels, 'ss')
    hom_a_power = lrt_power(hom_a, hom_a_labels, 'a')
    iid_a_power = lrt_power(iid_a, iid_a_labels, 'a')
    free_a_power = lrt_power(free_a, free_a_labels, 'a')
    #hom_vc_power = lrt_power(hom_vc, hom_vc_labels, 'vc')
    iid_vc_power = lrt_power(iid_vc, iid_vc_labels, 'vc')
    free_vc_power = lrt_power(free_vc, free_vc_labels, 'vc')
    free_V_diag_power = lrt_power(free_V_diag, free_V_diag_labels, 'V_diag')

    # plot
    fig, axes = plt.subplots(nrows=4, ncols=3, sharex='row', figsize=(15,20))
    alpha = 0.7
    lw = 3
    ms = 10

    ## 
    sns.lineplot(data=hom_ss_power, x='label', y='Positive rate', hue='LRT', alpha=alpha, lw=lw, ms=ms,
            hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[0,0])
    sns.lineplot(data=iid_ss_power, x='label', y='Positive rate', hue='LRT', alpha=alpha, lw=lw, ms=ms,
            hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[0,1])
    sns.lineplot(data=free_ss_power, x='label', y='Positive rate', hue='LRT', alpha=alpha, lw=lw, ms=ms,
            hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[0,2])
    for ax in axes[0,:]:
        ax.set_xlabel('sample size')
        ax.axhline(y=0.05, color='0.6', ls='--', zorder=0)
    sns.lineplot(data=hom_a_power, x='label', y='Positive rate', hue='LRT', alpha=alpha, lw=lw, ms=ms,
            hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[1,0])
    sns.lineplot(data=iid_a_power, x='label', y='Positive rate', hue='LRT', alpha=alpha, lw=lw, ms=ms,
            hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[1,1])
    sns.lineplot(data=free_a_power, x='label', y='Positive rate', hue='LRT', alpha=alpha, lw=lw, ms=ms,
            hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[1,2])
    for ax in axes[1,:]:
        ax.set_xlabel('Cell type proportions')
        ax.axhline(y=0.05, color='0.6', ls='--', zorder=0)
    #sns.lineplot(data=hom_vc_power, x='label', y='Positive rate', hue='LRT', 
    #        hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[2,0])
    sns.lineplot(data=iid_vc_power, x='label', y='Positive rate', hue='LRT', alpha=alpha, lw=lw, ms=ms,
            hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[2,1])
    sns.lineplot(data=free_vc_power, x='label', y='Positive rate', hue='LRT', alpha=alpha, lw=lw, ms=ms,
            hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[2,2])
    for ax in axes[2,1:]:
        ax.set_xlabel('Variance proportions')
        ax.tick_params(axis='x', labelsize='small', labelrotation=10)
        ax.axhline(y=0.05, color='0.6', ls='--', zorder=0)
    sns.lineplot(data=free_V_diag_power, x='label', y='Positive rate', hue='LRT', alpha=alpha, lw=lw, ms=ms,
            hue_order=['IID vs Hom', 'Free vs Hom', 'Free vs IID'], marker='o', ax=axes[3,2])
    axes[3,2].set_xlabel('Cell type specific genetic variance')
    axes[3,2].axhline(y=0.05, color='0.6', ls='--', zorder=0)

    # hide legend
    for ax in axes.flatten()[1:]:
        ax.legend().set_visible(False)

    # set y limit
    for ax in axes.flatten():
        ax.set_ylim([0-0.02, 1+0.05])

    axes[2,0].axis('off')
    #axes[2,1].yaxis.set_visible(True)
    #axes[2,1].set_ylabel('Positive rate')
    axes[3,0].axis('off')
    axes[3,1].axis('off')
    #axes[3,2].set_ylabel('Positive rate')

    # add Title for each column
    axes[0,0].set_title('Hom simulation', fontsize=16, pad=3*plt.rcParams['axes.titlepad'])
    axes[0,1].set_title('IID simulation', fontsize=16, pad=3*plt.rcParams['axes.titlepad'])
    axes[0,2].set_title('Free simulation', fontsize=16, pad=3*plt.rcParams['axes.titlepad'])

    #
    fig.savefig(output.png)

if __name__ == '__main__':
    main()
