import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_help

def main():
    # par
    input = snakemake.input
    output = snakemake.output
    params = snakemake.params
    wildcards = snakemake.wildcards 
    models = ['hom', 'iid', 'free']

    # select and order data
    data = pd.DataFrame({'arg':np.array(params.subspace[wildcards.arg]), 'out':input.out})
    plot_order_ = np.array(params.plot_order[wildcards.model][wildcards.arg])
    plot_order_ = plot_order_[np.isin(plot_order_, data['arg'])]
    data['arg'] = pd.Categorical(data['arg'], plot_order_)
    data = data.sort_values('arg').reset_index(drop=True)
    data.to_csv(sys.stdout, sep='\t')

    he_power = {}
    for f in data['out']:
        out = np.load(f, allow_pickle=True).item()
        # wald
        he = out['he']['wald'] # structure: he_p - model (e.g. hom_p) - statistics (e.g. hom2, V)
        for m in models:
            if m not in he.keys():
                continue
            he_m = he[m]
            if m not in he_power.keys():
                he_power[m] = {}
            if 'hom2' in he_m.keys():
                key = 'hom2'
                value = he_m[key]
                power_ = np.sum(value < 0.05, axis=0) / value.shape[0]
                if key not in he_power[m].keys():
                    he_power[m][key] = [power_]
                else:
                    he_power[m][key].append(power_)
            if 'V' in he_m.keys():
                key = 'V'
                value = he_m[key]
                power_ = np.sum(value < 0.05, axis=0) / value.shape[0]
                if key not in he_power[m].keys():
                    he_power[m][key] = [power_]
                else:
                    he_power[m][key].append(power_)

    for m in he_power.keys():
        for key in he_power[m].keys():
            he_power[m][key] = np.array(he_power[m][key])
            if he_power[m][key].ndim > 1:
                he_power[m][key] = np.array(he_power[m][key]).T

    # plot
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=params.mycolors)
    markers = plot_help.mymarkers()
    fig, axes = plt.subplots(ncols=len(he_power.keys()), sharex=True, sharey=True, figsize=(15, 4))
    # wald
    i = 0
    ## hom
    if 'hom' in he_power.keys():
        axes[i].plot(data['arg'], he_power['hom']['hom2'], marker=markers[0], label='$\sigma_{hom}^2$')
        axes[i].set_title('Hom')
        axes[i].set_xlabel(wildcards.arg)
        axes[i].set_ylabel('Positive rate')
        axes[i].legend()
        i += 1

    ## iid
    if 'iid' in he_power.keys():
        axes[i].plot(data['arg'], he_power['iid']['hom2'], marker=markers[0], label='$\sigma_{hom}^2$')
        axes[i].plot(data['arg'], he_power['iid']['V'], marker=markers[0], label='$\sigma_{het}^2$')
        axes[i].set_title('IID')
        axes[i].set_xlabel(wildcards.arg)
        axes[i].legend()
        i += 1

    ## free
    if 'free' in he_power.keys():
        axes[i].plot(data['arg'], he_power['free']['hom2'], marker=markers[0], label='$\sigma_{hom}^2$')
        axes[i].plot(data['arg'], he_power['free']['V'], marker=markers[0], label='$\sigma_{het}^2$')
        if 'Vi' in he_power['free'].keys():
            for i in range(len(he_power['free']['Vi'])):
                axes[i].plot(data['arg'], he_power['free']['Vi'][i], marker='.', label=f'$V_{i+1}$', ls='--',
                        color=params.mycolors[3+i], alpha=0.5)

        axes[i].set_title('Free')
        axes[i].set_xlabel(wildcards.arg)
        axes[i].legend()
        i += 1

    axes[0].set_ylim((0-0.02,1+0.02))
    for ax in axes:
        ax.axhline(y=0.05, color='0.6', ls='--', zorder=0)
        if len(data['arg'].values[0]) > 15:
            ax.tick_params(axis='x', labelsize='small', labelrotation=15)
    fig.tight_layout()
    fig.savefig(output.waldNlrt)

if __name__ == '__main__':
    main()
