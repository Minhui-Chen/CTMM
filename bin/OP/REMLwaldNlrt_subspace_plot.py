import sys, re
import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_help

def main():
    # par
    input = snakemake.input
    output = snakemake.output
    wildcards = snakemake.wildcards
    params = snakemake.params

    # select and order data
    data = pd.DataFrame({'arg':np.array(params.subspace[wildcards.arg]), 'out':input.out})
    plot_order_ = np.array(params.plot_order[wildcards.model][wildcards.arg])
    plot_order_ = plot_order_[np.isin(plot_order_, data['arg'])]
    data = data.loc[data['arg'].isin(plot_order_)]
    data['arg'] = pd.Categorical(data['arg'], plot_order_)
    data = data.sort_values('arg').reset_index(drop=True)
    data.to_csv(sys.stdout, sep='\t')
    
    wald_power = {}
    lrt_power = {}
    for f in data['out']:
        out = np.load(f, allow_pickle=True).item()
        # wald
        wald = out['reml']['wald'] # structure: wald - model (e.g. hom_p) - statistics (e.g. hom2, beta)
        for m in ['hom', 'iid', 'free']:
            wald_m = wald[m]
            if m not in wald_power.keys():
                wald_power[m] = {}
            for key, value in wald_m.items():
                power_ = np.sum(value < 0.05, axis=0) / value.shape[0]
                if key not in wald_power[m].keys():
                    wald_power[m][key] = [power_]
                else:
                    wald_power[m][key].append(power_)
        # lrt
        lrt = out['reml']['lrt'] # structure: lrt - two comparing model (e.g. full_free)
        for key, value in lrt.items():
            power = np.sum(value < 0.05) / value.shape[0]
            if key not in lrt_power.keys():
                lrt_power[key] = [power]
            else:
                lrt_power[key].append(power)

    for m in ['hom', 'iid', 'free']:
        for key in wald_power[m].keys():
            wald_power[m][key] = np.array(wald_power[m][key])
            if wald_power[m][key].ndim > 1:
                wald_power[m][key] = np.array(wald_power[m][key]).T

    # plot
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=params.mycolors)
    markers = plot_help.mymarkers()
    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(18, 4))
    # wald
    ## hom
    axes[0].plot(data['arg'], wald_power['hom']['hom2'], marker=markers[0], label='$\sigma_{hom}^2$')
    axes[0].set_title('Hom')
    axes[0].set_xlabel(wildcards.arg)
    axes[0].set_ylabel('Positive rate')
    axes[0].legend()
    ## iid
    axes[1].plot(data['arg'], wald_power['iid']['hom2'], marker=markers[0], label='$\sigma_{hom}^2$')
    axes[1].plot(data['arg'], wald_power['iid']['V'], marker=markers[0], label='$\sigma_{het}^2$')
    axes[1].set_title('IID')
    axes[1].set_xlabel(wildcards.arg)
    axes[1].legend()
    ## free
    axes[2].plot(data['arg'], wald_power['free']['hom2'], marker=markers[0], label='$\sigma_{hom}^2$')
    axes[2].plot(data['arg'], wald_power['free']['V'], marker=markers[0], label='$\sigma_{het}^2$')
    for i in range(len(wald_power['free']['Vi'])):
        axes[2].plot(data['arg'], wald_power['free']['Vi'][i], marker='.', label=f'$V_{i+1}$', ls='--',
                color=params.mycolors[3+i], alpha=0.5)

    axes[2].set_title('Free')
    axes[2].set_xlabel(wildcards.arg)
    axes[2].legend()

    # lrt
    def format_name(x):
        x = re.sub('full','Full',re.sub('free','Free',re.sub('iid','IID',re.sub('hom','Hom',re.sub('null','Null',x)))))
        return(' vs '.join(x.split('_')[:2]))

    a, b, c, d = 0, 0, 0, 0
    for x in lrt_power.keys():
        if re.search('^hom', x):
            axes[3].plot(data['arg'], lrt_power[x], marker=markers[a], label=format_name(x), color=params.mycolors[0])
            a += 1
        if re.search('^iid', x):
            axes[3].plot(data['arg'], lrt_power[x], marker=markers[b], label=format_name(x), color=params.mycolors[1])
            b += 1
        if re.search('^free', x):
            axes[3].plot(data['arg'], lrt_power[x], marker=markers[c], label=format_name(x), color=params.mycolors[2])
            c += 1
        if re.search('^full', x):
            axes[3].plot(data['arg'], lrt_power[x], marker=markers[d], label=format_name(x), color=params.mycolors[3])
            d += 1
    axes[3].legend()
    axes[3].set_title('LRT')
    axes[3].set_xlabel(wildcards.arg)

    axes[0].set_ylim((0-0.02,1+0.02))
    for ax in axes:
        ax.axhline(y=0.05, color='0.6', ls='--', zorder=0)
        if len(data['arg'].values[0]) > 15:
            ax.tick_params(axis='x', labelsize='small', labelrotation=15)
    fig.tight_layout()
    fig.savefig(output.waldNlrt)

if __name__ == '__main__':
    main()
