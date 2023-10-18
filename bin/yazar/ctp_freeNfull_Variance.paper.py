import os, math, re, sys
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # read
    P = pd.read_table(snakemake.input.P, index_col=0)
    ctp = np.load(snakemake.input.ctp, allow_pickle=True).item()

    #
    method = 'he'
    V = ctp['he']['free']['V']
    n = V.shape[0]
    C = V.shape[1]
    CTs = P.columns.to_numpy()
    CTs = np.array([re.sub(' ', '_', ct) for ct in CTs])
    ct_order = ['CD4_NC', 'CD8_ET', 'NK', 'CD8_NC', 'B_IN', 'CD4_ET', 'B_Mem', 'Mono_C', 'CD8_S100B', 'Mono_NC', 'NK_R', 'DC', 'CD4_SOX4', 'Plasma']
    # CTs = [ct for ct in ct_order if ct in CTs]

    # Create a dictionary to map each element to its corresponding index in the reference order array
    element_to_index = {element: index for index, element in enumerate(ct_order)}

    # Use the dictionary to get the sorting indices for the original_array
    sorting_indices = [element_to_index[element] for element in CTs]

    # Reorder the original_array based on the sorting indices
    ordered_CTs = CTs[np.argsort(sorting_indices)]

    ctp_data = pd.DataFrame()
    # Free model
    ## hom2
    ctp_data['hom2'] = ctp[method]['free']['hom2']
    ## V
    V = [np.diag(x) for x in ctp[method]['free']['V']]
    V = pd.DataFrame(V, columns=[f'V-{ct}' for ct in CTs])
    ctp_data = pd.concat( (ctp_data, V), axis=1)

    for ct in CTs:
        ctp_data[f'V-{ct}-hom'] = ctp_data[f'V-{ct}'] + ctp_data['hom2']

    # plot
    if C < 8:
        mpl.rcParams.update({'font.size': 11})
        fs = 10
    else:
        mpl.rcParams.update({'font.size': 8})
        fs = 7
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
    color = sns.color_palette()[0]
    
    ## Free model
    data = ctp_data[['hom2'] + [f'V-{ct}' for ct in ordered_CTs]].clip(lower=-2, upper=2)
    print(data.min().min())
    ax = sns.violinplot(data=data, cut=0, palette=['0.8'] + [color] * C, scale='width')

    # Add median values as text annotations
    medians = data.median(axis=0)
    y_loc = -0.15
    ax.text(-0.05, y_loc, "median:", ha='center', va='center', fontsize=fs, transform=ax.transAxes)
    for xtick, median in zip(ax.get_xticks(), medians):
        # ax.text(xtick, median, f"{median:.2f}", ha='center', va='bottom', fontsize=10)
        x = ax.transLimits.transform((xtick, median))[0]
        ax.text(x, y_loc, f"{median:.2f}", ha='center', va='center', fontsize=fs, transform=ax.transAxes)

    ax.axhline(0, ls='--', color='0.8', zorder=0)
    ax.set_xlabel('')
    ax.set_ylabel('Variance in Free model')

    # change ticks in Free model
    fig.canvas.draw_idle()
    # plt.sca(ax)
    locs, labels = plt.xticks()
    print(locs)
    for i in range(len(labels)):
        label = labels[i].get_text()
        print(label)
        if '-' in label:
            ct = label.split('-')[1]
            ct = re.sub('_', '\\_', ct)
            labels[i] = r'$V_{%s}$'%(ct)
        elif label == 'hom2':
            labels[i] = r'$\sigma_\alpha^2$'
    plt.xticks(locs, labels)

    plt.tight_layout()
    fig.savefig(snakemake.output.free)

    # Full model
    m = 'he'
    V = ctp['he']['full']['V']
    # CT_pairs = [f'{CTs[i]}-{CTs[j]}' for i in range(C-1) for j in range(i+1,C)]
    # print(CT_pairs)
    # plot cov or cor
    show = 'cor'
    if show == 'cov':
        # ctp_cov = [x[np.triu_indices(C,1)] for x in V]
        # ctp_cov = pd.DataFrame(ctp_cov, columns=CT_pairs)
        # data = pd.melt( ctp_cov )
        # data = data.clip(lower=-1.0, upper=1.5)
        data = np.median(V, axis=0)
        ylab = 'Covariance between CTs in Full model'
    else:
        data = []
        k = 0
        for x in V:
            if np.any(np.diag(x) < 0):
                k += 1
                continue
            cor = x / np.sqrt(np.outer(np.diag(x), np.diag(x)))
            data.append(cor)
        print(len(V), k)
        data = np.array(data)
        # data = data.clip(lower=-1.5, upper=1.5)
        data = np.median(data, axis=0)
        ylab = 'Correlation between CTs in Full model'

    # order cts
    print(CTs)
    print(data)
    data = data[np.argsort(sorting_indices), :]
    data = data[:, np.argsort(sorting_indices)]

    #threshold = [np.mean(data['value']) - 3*np.std(data['value']), np.mean(data['value']) + 3*np.std(data['value'])]
    # my_pal = {}
    # for pair in CT_pairs:
    #     ct1, ct2 = int(pair.split('-')[0][2]), int(pair.split('-')[1][2])
    #     if ct2 - ct1 != 1:
    #         my_pal[pair] = 'lightblue'
    #     else:
    #         my_pal[pair] = sns.color_palette('muted')[0]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    print(ordered_CTs)
    print(data)
    data = data[1:, :-1]
    sns.heatmap(data, square=True, mask=np.triu(np.ones_like(data), k=1), xticklabels=ordered_CTs[:-1], 
                yticklabels=ordered_CTs[1:], cmap="Blues", vmin=0, vmax=1)
                # yticklabels=ordered_CTs, cmap=sns.color_palette("Blues", as_cmap=True), vmin=0, vmax=1)

    
    # Rotate the y tick labels to be horizontal
    plt.yticks(rotation=0)

    fig.savefig(snakemake.output.full)

if __name__ == '__main__':
    main()
