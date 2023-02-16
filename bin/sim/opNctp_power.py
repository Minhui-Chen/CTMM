import sys
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # par

    # collect data
    op_hom = [np.load(f, allow_pickle=True).item() for f in snakemake.input.op_hom]
    op_free = [np.load(f, allow_pickle=True).item() for f in snakemake.input.op_free]
    op_hom_remlJK = [np.load(f, allow_pickle=True).item() for f in snakemake.input.op_hom_remlJK]
    op_free_remlJK = [np.load(f, allow_pickle=True).item() for f in snakemake.input.op_free_remlJK]
    ctp_hom = [np.load(f, allow_pickle=True).item() for f in snakemake.input.ctp_hom]
    ctp_free = [np.load(f, allow_pickle=True).item() for f in snakemake.input.ctp_free]
    ctp_hom_remlJK = [np.load(f, allow_pickle=True).item() for f in snakemake.input.ctp_hom_remlJK]
    ctp_free_remlJK = [np.load(f, allow_pickle=True).item() for f in snakemake.input.ctp_free_remlJK]
    hom_labels = snakemake.params.hom
    free_labels = snakemake.params.free
    hom_remlJK_labels = snakemake.params.hom_remlJK
    free_remlJK_labels = snakemake.params.free_remlJK

    def get_power(outs, labels, arg, method):
        power = {'Method':[method] * len(labels), 'Positive rate':[], 'label': labels}
        if method == 'REML (LRT)':
            for out in outs:
                out = out['reml']['lrt'] 
                power['Positive rate'].append(np.sum(out['free_hom'] < 0.05) / out['free_hom'].shape[0])
        elif method in ['REML (Wald)', 'REML (JK)']:
            for out in outs:
                out = out['reml']['wald']
                power['Positive rate'].append(np.sum(out['free']['V'] < 0.05) / out['free']['V'].shape[0])
        elif method == 'HE':
            for out in outs:
                out = out['he']['wald']
                power['Positive rate'].append(np.sum(out['free']['V'] < 0.05) / out['free']['V'].shape[0])
        power = pd.DataFrame(power)

        plot_order = np.array(snakemake.params.plot_order['free'][arg])
        try:
            power['label'] = power.apply(lambda x: str(int(float(x['label']))), axis=1)
            plot_order = np.array( [str(int(float(x))) for x in plot_order] )
            print('transform labels')
        except:
            pass
        plot_order = plot_order[np.isin(plot_order, power['label'])]
        power['label'] = pd.Categorical(power['label'], plot_order)

        print(power)
        return power

    reml_wald_op_hom_power = get_power(op_hom, hom_labels, 'ss', 'REML (Wald)')
    reml_wald_op_hom_power['Framework'] = 'OP'
    reml_lrt_op_hom_power = get_power(op_hom, hom_labels, 'ss', 'REML (LRT)')
    reml_lrt_op_hom_power['Framework'] = 'OP'
    reml_jk_op_hom_power = get_power(op_hom_remlJK, hom_remlJK_labels, 'ss', 'REML (JK)')
    reml_jk_op_hom_power['Framework'] = 'OP'

    reml_wald_ctp_hom_power = get_power(ctp_hom, hom_labels, 'ss', 'REML (Wald)')
    reml_wald_ctp_hom_power['Framework'] = 'CTP'
    reml_lrt_ctp_hom_power = get_power(ctp_hom, hom_labels, 'ss', 'REML (LRT)')
    reml_lrt_ctp_hom_power['Framework'] = 'CTP'
    reml_jk_ctp_hom_power = get_power(ctp_hom_remlJK, hom_remlJK_labels, 'ss', 'REML (JK)')
    reml_jk_ctp_hom_power['Framework'] = 'CTP'

    he_op_hom_power = get_power(op_hom, hom_labels, 'ss', 'HE')
    he_op_hom_power['Framework'] = 'OP'
    he_ctp_hom_power = get_power(ctp_hom, hom_labels, 'ss', 'HE')
    he_ctp_hom_power['Framework'] = 'CTP'
    #he_hom_power = pd.concat( [he_op_hom_power, he_ctp_hom_power], ignore_index=True )
    #he_hom_power = he_hom_power.rename( columns={'label':'sample size'} )
    reml_hom_power = pd.concat( [reml_wald_op_hom_power, reml_lrt_op_hom_power, reml_jk_op_hom_power,
        reml_wald_ctp_hom_power, reml_lrt_ctp_hom_power, reml_jk_ctp_hom_power, he_op_hom_power, 
        he_ctp_hom_power], 
        ignore_index=True )

    plot_order = np.array(snakemake.params.plot_order['free']['ss'])
    try:
        plot_order = np.array( [str(int(float(x))) for x in plot_order] )
    except:
        pass
    plot_order = plot_order[np.isin(plot_order, reml_hom_power['label'])]
    reml_hom_power['label'] = pd.Categorical(reml_hom_power['label'], plot_order)

    reml_hom_power = reml_hom_power.rename( columns={'label':'sample size'} )

    reml_wald_op_free_power = get_power(op_free, free_labels, 'ss', 'REML (Wald)')
    reml_wald_op_free_power['Framework'] = 'OP'
    reml_lrt_op_free_power = get_power(op_free, free_labels, 'ss', 'REML (LRT)')
    reml_lrt_op_free_power['Framework'] = 'OP'
    reml_jk_op_free_power = get_power(op_free_remlJK, free_remlJK_labels, 'ss', 'REML (JK)')
    reml_jk_op_free_power['Framework'] = 'OP'

    reml_wald_ctp_free_power = get_power(ctp_free, free_labels, 'ss', 'REML (Wald)')
    reml_wald_ctp_free_power['Framework'] = 'CTP'
    reml_lrt_ctp_free_power = get_power(ctp_free, free_labels, 'ss', 'REML (LRT)')
    reml_lrt_ctp_free_power['Framework'] = 'CTP'
    reml_jk_ctp_free_power = get_power(ctp_free_remlJK, free_remlJK_labels, 'ss', 'REML (JK)')
    reml_jk_ctp_free_power['Framework'] = 'CTP'

    he_op_free_power = get_power(op_free, free_labels, 'ss', 'HE')
    he_op_free_power['Framework'] = 'OP'
    he_ctp_free_power = get_power(ctp_free, free_labels, 'ss', 'HE')
    he_ctp_free_power['Framework'] = 'CTP'
    #he_free_power = pd.concat( [he_op_free_power, he_ctp_free_power], ignore_index=True )
    #he_free_power = he_free_power.rename( columns={'label':'sample size'} )
    reml_free_power = pd.concat( [reml_wald_op_free_power, reml_lrt_op_free_power, 
        reml_jk_op_free_power, reml_wald_ctp_free_power, reml_lrt_ctp_free_power, reml_jk_ctp_free_power,
        he_op_free_power, he_ctp_free_power], 
        ignore_index=True )

    reml_free_power['label'] = pd.Categorical(reml_free_power['label'], plot_order)

    reml_free_power = reml_free_power.rename( columns={'label':'sample size'} )

    # plot
    mpl.rcParams.update({'font.size': 7, 'lines.markersize': mpl.rcParams['lines.markersize']*1})
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(6.85,2.8), dpi=600)
    alpha = 1.0
    lw = 1.0

    ## 
    sns.lineplot(data=reml_hom_power, x='sample size', y='Positive rate', hue='Method', 
            style='Framework', dashes={'OP':(3,2), 'CTP':''}, ax=axes[0], markers={'OP':'X', 'CTP':'o'},
            alpha=alpha, lw=lw, legend=False)
    axes[0].set_ylabel('False positive rate', fontsize=9)
    axes[0].set_xlabel('Sample size', fontsize=9)
    axes[0].text(-0.10, 1.05, '(A)', fontsize=10, transform=axes[0].transAxes)
    # make two legends
    colors = sns.color_palette()
    markers = sns.color_palette()
    line1, = axes[0].plot([], [], color=colors[0], label='REML (Wald)')
    line2, = axes[0].plot([], [], color=colors[1], label='REML (LRT)')
    line3, = axes[0].plot([], [], color=colors[2], label='REML (JK)')
    line4, = axes[0].plot([], [], color=colors[3], label='HE')
    first_legend = axes[0].legend(handles=[line1, line2, line3, line4], loc='upper right', title='Method')
    axes[0].add_artist(first_legend)
    line1, = axes[0].plot([], [], color='k', marker='o', label='CTP')
    line2, = axes[0].plot([], [], color='k', linestyle=(0,(2.2,1.5)), marker='x', label='OP')
    axes[0].legend(handles=[line1, line2], loc='upper left', title='Data')

    sns.lineplot(data=reml_free_power, x='sample size', y='Positive rate', hue='Method', 
            style='Framework', dashes={'OP':(3,2), 'CTP':''}, ax=axes[1], markers={'OP':'X', 'CTP':'o'}, 
            alpha=alpha, lw=lw)
    axes[1].set_ylabel('True positive rate', fontsize=9)
    axes[1].set_xlabel('Sample size', fontsize=9)
    axes[1].text(-0.10, 1.05, '(B)', fontsize=10, transform=axes[1].transAxes)

    ylim_min = np.amin([axes[0].get_ylim(), axes[1].get_ylim()])
    ylim_max = np.amax([axes[0].get_ylim(), axes[1].get_ylim()])
    for ax in axes.flatten():
        ax.axhline(y=0.05, color='0.9', ls='--', zorder=0, lw=lw)
        ax.set_ylim(ylim_min, ylim_max)
    for ax in axes.flatten()[1:]:
        ax.legend().set_visible(False)

    plt.tight_layout(pad=2, w_pad=3)
    fig.savefig( snakemake.output.png )

if __name__ == '__main__':
    main()


