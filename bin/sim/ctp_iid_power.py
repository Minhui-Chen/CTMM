import sys
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def get_power(outs, labels, arg, method):
    power = {'Method':[method] * len(labels), 'Positive rate':[], 'label': labels}
    if method == 'REML (LRT)':
        for out in outs:
            out = out['reml']['lrt'] 
            power['Positive rate'].append(np.sum(out['iid_hom'] < 0.05) / out['iid_hom'].shape[0])
    elif method in ['REML (Wald)', 'REML (JK)']:
        for out in outs:
            out = out['reml']['wald']
            power['Positive rate'].append(np.sum(out['iid']['V'] < 0.05) / out['iid']['V'].shape[0])
    elif method == 'HE':
        for out in outs:
            out = out['he']['wald']
            power['Positive rate'].append(np.sum(out['iid']['V'] < 0.05) / out['iid']['V'].shape[0])
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

def main():
    # par

    # collect data
    ctp_hom = [np.load(f, allow_pickle=True).item() for f in snakemake.input.ctp_hom]
    ctp_free = [np.load(f, allow_pickle=True).item() for f in snakemake.input.ctp_free]
    hom_labels = snakemake.params.hom
    free_labels = snakemake.params.free

    reml_wald_ctp_hom_power = get_power(ctp_hom, hom_labels, 'ss', 'REML (Wald)')
    reml_wald_ctp_hom_power['Framework'] = 'Wald'
    reml_lrt_ctp_hom_power = get_power(ctp_hom, hom_labels, 'ss', 'REML (LRT)')
    reml_lrt_ctp_hom_power['Framework'] = 'LRT'

    he_ctp_hom_power = get_power(ctp_hom, hom_labels, 'ss', 'HE')
    he_ctp_hom_power['Framework'] = 'Wald'
    #he_hom_power = pd.concat( [he_op_hom_power, he_ctp_hom_power], ignore_index=True )
    #he_hom_power = he_hom_power.rename( columns={'label':'sample size'} )
    reml_hom_power = pd.concat( [
        reml_wald_ctp_hom_power, reml_lrt_ctp_hom_power, 
        he_ctp_hom_power], 
        ignore_index=True )

    plot_order = np.array(snakemake.params.plot_order['free']['ss'])
    try:
        plot_order = np.array( [str(int(float(x))) for x in plot_order] )
    except:
        pass
    plot_order = plot_order[np.isin(plot_order, reml_hom_power['label'])]
    #reml_hom_power['label'] = pd.Categorical(reml_hom_power['label'], plot_order)
    reml_hom_power['label'] = reml_hom_power['label'].astype('int')

    reml_hom_power = reml_hom_power.rename( columns={'label':'sample size'} )

    reml_wald_ctp_free_power = get_power(ctp_free, free_labels, 'ss', 'REML (Wald)')
    reml_wald_ctp_free_power['Framework'] = 'Wald'
    reml_lrt_ctp_free_power = get_power(ctp_free, free_labels, 'ss', 'REML (LRT)')
    reml_lrt_ctp_free_power['Framework'] = 'LRT'

    he_ctp_free_power = get_power(ctp_free, free_labels, 'ss', 'HE')
    he_ctp_free_power['Framework'] = 'Wald'
    #he_free_power = pd.concat( [he_op_free_power, he_ctp_free_power], ignore_index=True )
    #he_free_power = he_free_power.rename( columns={'label':'sample size'} )
    reml_free_power = pd.concat( [
        reml_wald_ctp_free_power, reml_lrt_ctp_free_power, he_ctp_free_power], 
        ignore_index=True )

    #reml_free_power['label'] = pd.Categorical(reml_free_power['label'], plot_order)
    reml_free_power['label'] = reml_free_power['label'].astype('int')

    reml_free_power = reml_free_power.rename( columns={'label':'sample size'} )

    # plot
    mpl.rcParams.update({'font.size': 7, 'lines.markersize': mpl.rcParams['lines.markersize']*1})
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(6.85,2.8), dpi=600)
    alpha = 1.0
    lw = 1.0

    ## 
    colors = sns.color_palette()
    sns.lineplot(data=reml_hom_power, x='sample size', y='Positive rate', hue='Method', 
            alpha=alpha, lw=lw, legend=False, ax=axes[0], palette=[colors[0], colors[1], colors[3]], marker='o',
            style='Framework', dashes={'LRT':(3,2), 'Wald':''})
            # style='Framework', dashes={'OP':(3,2), 'CTP':''}, ax=axes[0], markers={'OP':'X', 'CTP':'o'},
    print(reml_hom_power)
    axes[0].set_ylabel('False positive rate', fontsize=9)
    axes[0].set_xlabel('Sample size', fontsize=9)
    axes[0].text(-0.10, 1.05, '(A)', fontsize=10, transform=axes[0].transAxes)
    # make two legends
    markers = sns.color_palette()
    line1, = axes[0].plot([], [], color=colors[0], label='REML (Wald)')
    line2, = axes[0].plot([], [], color=colors[1], label='REML (LRT)')
    # line3, = axes[0].plot([], [], color=colors[2], label='REML (JK)')
    line4, = axes[0].plot([], [], color=colors[3], label='HE')
    first_legend = axes[0].legend(handles=[line1, line2, line4], loc='upper right', title='Method')
    axes[0].add_artist(first_legend)
    # line1, = axes[0].plot([], [], color='k', marker='o', label='CTP')
    # line2, = axes[0].plot([], [], color='k', linestyle=(0,(2.2,1.5)), marker='x', label='OP')
    # axes[0].legend(handles=[line1, line2], loc='upper left', title='Data')

    sns.lineplot(data=reml_free_power, x='sample size', y='Positive rate', hue='Method', 
            alpha=alpha, lw=lw, ax=axes[1], palette=[colors[0], colors[1], colors[3]], marker='o',
            style='Framework', dashes={'LRT':(3,2), 'Wald':''})
            # style='Framework', dashes={'OP':(3,2), 'CTP':''}, ax=axes[1], markers={'OP':'X', 'CTP':'o'}, 
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


