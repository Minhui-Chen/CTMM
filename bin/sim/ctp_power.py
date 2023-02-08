import sys
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # par

    # collect data
    hom_ss = [np.load(f, allow_pickle=True).item() for f in snakemake.input.hom_ss]
    hom_ss_remlJK = [np.load(f, allow_pickle=True).item() for f in snakemake.input.hom_ss_remlJK]
    free_ss = [np.load(f, allow_pickle=True).item() for f in snakemake.input.free_ss]
    free_ss_remlJK = [np.load(f, allow_pickle=True).item() for f in snakemake.input.free_ss_remlJK]
    hom_ss_labels = snakemake.params.hom_ss
    hom_ss_remlJK_labels = snakemake.params.hom_ss_remlJK
    free_ss_labels = snakemake.params.free_ss
    free_ss_remlJK_labels = snakemake.params.free_ss_remlJK

    hom_a = [np.load(f, allow_pickle=True).item() for f in snakemake.input.hom_a]
    hom_a_remlJK = [np.load(f, allow_pickle=True).item() for f in snakemake.input.hom_a_remlJK]
    free_a = [np.load(f, allow_pickle=True).item() for f in snakemake.input.free_a]
    free_a_remlJK = [np.load(f, allow_pickle=True).item() for f in snakemake.input.free_a_remlJK]
    hom_a_labels = snakemake.params.hom_a
    hom_a_remlJK_labels = snakemake.params.hom_a_remlJK
    free_a_labels = snakemake.params.free_a
    free_a_remlJK_labels = snakemake.params.free_a_remlJK

    free_vc = [np.load(f, allow_pickle=True).item() for f in snakemake.input.free_vc]
    free_vc_remlJK = [np.load(f, allow_pickle=True).item() for f in snakemake.input.free_vc_remlJK]
    free_vc_labels = snakemake.params.free_vc
    free_vc_remlJK_labels = snakemake.params.free_vc_remlJK

    def get_power(outs, labels, arg, method, test):
        power = {'Method':[method.upper()] * len(labels), 'Positive rate':[], 'label': labels}
        if method in ['ml', 'reml']:
            for out in outs:
                out = out[method][test]
                if test == 'lrt':
                    power['Positive rate'].append(np.sum(out['free_hom'] < 0.05) / out['free_hom'].shape[0])
                else:
                    power['Positive rate'].append(np.sum(out['free']['V'] < 0.05) / out['free']['V'].shape[0])
        elif method == 'he':
            for out in outs:
                out = out['he']['wald']
                power['Positive rate'].append(np.sum(out['free']['V'] < 0.05) / out['free']['V'].shape[0])
        power = pd.DataFrame(power)

        # shift overlapping lines
        #columns = power.columns.to_list()
        #columns.remove('label')
        #for i in range(len(columns)-1):
        #    for j in range(i+1, len(columns)):
        #        if np.all(power[columns[i]] == power[columns[j]]):
        #            power[columns[i]] = power[columns[i]] + 0.01
        #            power[columns[j]] = power[columns[j]] - 0.01

        plot_order = np.array(snakemake.params.plot_order['free'][arg])
        if arg == 'ss':
            power['label'] = power.apply(lambda x: str(int(float(x['label']))), axis=1)
            plot_order = np.array( [str(int(float(x))) for x in plot_order] )
        elif arg == 'a':
            def transform_a(a):
                a = np.array( [float(x) for x in a.split('_')] )
                p = a[0] / np.sum(a)
                return '%.2f'%p

            power['label'] = power.apply(lambda x: transform_a(x['label']), axis=1)
            plot_order = np.array( [transform_a(x) for x in plot_order] )
        elif arg == 'vc':
            power['label'] = power.apply(lambda x: x['label'].split('_')[2], axis=1)
            plot_order = np.array( [x.split('_')[2] for x in plot_order] )
        plot_order = plot_order[np.isin(plot_order, power['label'])]
        print(plot_order)
        power['label'] = pd.Categorical(power['label'], plot_order)

        #power = pd.melt(power, id_vars=['label'], value_vars=['IID vs Hom', 'Free vs Hom', 'Free vs IID'],
        #        var_name='model', value_name='Positive rate')

        print(power)
        return power

    def collect_power(outs, labels, arg, method):
        wald_power = get_power(outs, labels, arg, method, 'wald')
        wald_power['Test'] = 'Wald'
        lrt_power = get_power(outs, labels, arg, method, 'lrt')
        lrt_power['Test'] = 'LRT'
        power = pd.concat([wald_power, lrt_power], ignore_index=True)

        return power

    # sample size
    ml_hom_ss_power = collect_power(hom_ss, hom_ss_labels, snakemake.params.arg_ss, 'ml')
    ml_free_ss_power = collect_power(free_ss, free_ss_labels, snakemake.params.arg_ss, 'ml')
    reml_hom_ss_power = collect_power(hom_ss, hom_ss_labels, snakemake.params.arg_ss, 'reml')
    reml_free_ss_power = collect_power(free_ss, free_ss_labels, snakemake.params.arg_ss, 'reml')
    remlJK_hom_ss_power = get_power(hom_ss_remlJK, hom_ss_remlJK_labels, snakemake.params.arg_ss, 'reml', 'wald')
    remlJK_hom_ss_power['Method'] = 'REML (JK)'
    remlJK_hom_ss_power['Test'] = 'Wald'
    remlJK_free_ss_power = get_power(free_ss_remlJK, free_ss_remlJK_labels, snakemake.params.arg_ss, 'reml', 'wald')
    remlJK_free_ss_power['Method'] = 'REML (JK)'
    remlJK_free_ss_power['Test'] = 'Wald'
    he_hom_ss_power = get_power(hom_ss, hom_ss_labels, snakemake.params.arg_ss, 'he', 'wald')
    he_free_ss_power = get_power(free_ss, free_ss_labels, snakemake.params.arg_ss, 'he', 'wald')
    he_hom_ss_power['Test'] = 'Wald'
    he_free_ss_power['Test'] = 'Wald'

    hom_ss_power = pd.concat([ml_hom_ss_power, reml_hom_ss_power, remlJK_hom_ss_power, he_hom_ss_power], 
            ignore_index=True)
    plot_order = np.array(snakemake.params.plot_order['free']['ss'])
    plot_order = np.array( [str(int(float(x))) for x in plot_order] )
    plot_order = plot_order[np.isin(plot_order, hom_ss_power['label'])]
    hom_ss_power['label'] = pd.Categorical(hom_ss_power['label'], plot_order)

    free_ss_power = pd.concat([ml_free_ss_power, reml_free_ss_power, remlJK_free_ss_power, he_free_ss_power], 
            ignore_index=True)
    free_ss_power['label'] = pd.Categorical(free_ss_power['label'], plot_order)

    # cell type proportion
    ml_hom_a_power = collect_power(hom_a, hom_a_labels, snakemake.params.arg_a, 'ml')
    ml_free_a_power = collect_power(free_a, free_a_labels, snakemake.params.arg_a, 'ml')
    reml_hom_a_power = collect_power(hom_a, hom_a_labels, snakemake.params.arg_a, 'reml')
    reml_free_a_power = collect_power(free_a, free_a_labels, snakemake.params.arg_a, 'reml')
    remlJK_hom_a_power = get_power(hom_a_remlJK, hom_a_remlJK_labels, snakemake.params.arg_a, 'reml', 'wald')
    remlJK_hom_a_power['Method'] = 'REML (JK)'
    remlJK_hom_a_power['Test'] = 'Wald'
    remlJK_free_a_power = get_power(free_a_remlJK, free_a_remlJK_labels, snakemake.params.arg_a, 'reml', 'wald')
    remlJK_free_a_power['Method'] = 'REML (JK)'
    remlJK_free_a_power['Test'] = 'Wald'
    he_hom_a_power = get_power(hom_a, hom_a_labels, snakemake.params.arg_a, 'he', 'wald')
    he_free_a_power = get_power(free_a, free_a_labels, snakemake.params.arg_a, 'he', 'wald')
    he_hom_a_power['Test'] = 'Wald'
    he_free_a_power['Test'] = 'Wald'

    hom_a_power = pd.concat([ml_hom_a_power, reml_hom_a_power, remlJK_hom_a_power, he_hom_a_power], 
            ignore_index=True)
    free_a_power = pd.concat([ml_free_a_power, reml_free_a_power, remlJK_free_a_power, he_free_a_power], 
            ignore_index=True)

    # vc
    ml_free_vc_power = collect_power(free_vc, free_vc_labels, snakemake.params.arg_vc, 'ml')
    reml_free_vc_power = collect_power(free_vc, free_vc_labels, snakemake.params.arg_vc, 'reml')
    remlJK_free_vc_power = get_power(free_vc_remlJK, free_vc_remlJK_labels, snakemake.params.arg_vc, 'reml', 'wald')
    remlJK_free_vc_power['Method'] = 'REML (JK)'
    remlJK_free_vc_power['Test'] = 'Wald'
    he_free_vc_power = get_power(free_vc, free_vc_labels, snakemake.params.arg_vc, 'he', 'wald')
    he_free_vc_power['Test'] = 'Wald'

    free_vc_power = pd.concat([ml_free_vc_power, reml_free_vc_power, remlJK_free_vc_power, he_free_vc_power], 
            ignore_index=True)

    # plot
    mpl.rcParams.update({'font.size': 8, 'lines.markersize': mpl.rcParams['lines.markersize']*0.8})
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex='row', sharey=False, figsize=(6.85,9), dpi=600)
    alpha = 1
    lw = 1
    #ms = 10

    ## 
    sns.lineplot(data=hom_ss_power, x='label', y='Positive rate', hue='Method', 
            style='Test', dashes={'Wald':'','LRT':(2.5,2)}, ax=axes[0,0], markers=True, alpha=alpha, lw=lw)
    sns.lineplot(data=free_ss_power, x='label', y='Positive rate', hue='Method', 
            style='Test', dashes={'Wald':'','LRT':(2.5,2)}, ax=axes[0,1], markers=True, alpha=alpha, lw=lw)
    axes[0,0].set_xlabel('Sample size', fontsize=10)
    axes[0,1].set_xlabel('Sample size', fontsize=10)
    axes[0,0].axvline(x=2, color='0.8', ls=':', zorder=0)
    axes[0,1].axvline(x=2, color='0.8', ls=':', zorder=0)

    sns.lineplot(data=hom_a_power, x='label', y='Positive rate', hue='Method', 
            style='Test', dashes={'Wald':'','LRT':(2.5,2)}, ax=axes[1,0], markers=True, alpha=alpha, lw=lw)
    sns.lineplot(data=free_a_power, x='label', y='Positive rate', hue='Method', 
            style='Test', dashes={'Wald':'','LRT':(2.5,2)}, ax=axes[1,1], markers=True, alpha=alpha, lw=lw)
    axes[1,0].set_xlabel('Main cell type proportion', fontsize=10)
    axes[1,1].set_xlabel('Main cell type proportion', fontsize=10)
    axes[1,0].axvline(x=2, color='0.8', ls=':', zorder=0)
    axes[1,1].axvline(x=2, color='0.8', ls=':', zorder=0)

    sns.lineplot(data=free_vc_power, x='label', y='Positive rate', hue='Method', 
            style='Test', dashes={'Wald':'','LRT':(2.5,2)}, ax=axes[2,1], markers=True, alpha=alpha, lw=lw)
    axes[2,1].set_xlabel('$\sigma_{het}^2$', fontsize=10)
    axes[2,1].axvline(x=2, color='0.8', ls=':', zorder=0)

    axes[0,0].set_title('Hom simulation', fontsize=12)
    axes[0,1].set_title('Free simulation', fontsize=12)

    for ax in axes[:,0]:
        ax.set_ylabel('False positive rate', fontsize=10)
    for ax in axes[:,1]:
        ax.set_ylabel('True positive rate', fontsize=10)
    #ylims = [ax.get_ylim() for ax in axes.flatten()]
    for ax in axes.flatten():
        if ax == axes[2,0]:
            continue
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.05, color='0.8', ls='--', zorder=0)
    for ax in axes.flatten()[1:]:
        if ax == axes[2,0]:
            ax.legend(handlelength=3.3)
        ax.legend().set_visible(False)
    axes[2,0].axis('off')
    #axes[0,1].set_yticks([])
    #axes[0,1].set_ylabel('')
    #axes[1,1].set_yticks([])
    #axes[1,1].set_ylabel('')
    
    plt.tight_layout(w_pad=2, h_pad=2)
    fig.savefig( snakemake.output.png )

if __name__ == '__main__':
    main()


