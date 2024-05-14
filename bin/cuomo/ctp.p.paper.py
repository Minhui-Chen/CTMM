import math
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from ctmm import draw


def meta_regression(grouped_data, column, stat):
    if stat == 'mean':
        meta = grouped_data[column].mean()
    elif stat == 'median':
        meta = grouped_data[column].median()
    print(meta)    

    X = sm.add_constant(pd.Series(meta.index.codes, index=meta.index, name=column))

    # fit
    model = sm.OLS(meta.values, X)
    results = model.fit()

    # Access the coefficients
    slope = results.params[column]
    intercept = results.params['const']

    # Print the equation of the line
    line = f"y = {slope:.4f}x + {intercept:.4f}"
    p = results.pvalues[column]

    return line, p


def matrix_plot(data, features, propertys, num_bins, png, gs):
    # divide EDS into bins
    for property in propertys:
        if len(data[property].unique()) > num_bins:
            bins, bin_labels = pd.qcut(data[property], q=num_bins, retbins=True)
            data[property + '_bin'] = bins
            grouped = data.groupby(by=property + '_bin')
            medians = grouped[property].median()
            # format
            if property in ['gene_length (kb)', 'ActivityLinking_EnhancerNumber']:
                custom_format = lambda x: format(x, '.0f')
            else:
                custom_format = lambda x: format(x, f'.{2}f') if x >= 0.01 else format(x, f'.{1}e')
            medians = medians.apply(custom_format)

            bins = bins.cat.rename_categories(medians.to_dict())
            data[property + '_bin'] = bins

            print(bin_labels)
        else:
            data[property + '_bin'] = data[property]


    # plot
    xlabs = { 
             'gene_length (kb)': 'Gene length (kb) bins',
             'ActivityLinking_EnhancerNumber': 'Enhancer number bins',
             'EDS': 'EDS bins',
             'LOEUF': 'LOEUF bins',
             'pLI': 'pLI bins',
             }

    ylabs = { 
             'Mean expression': 'Mean expression',
             'g': 'Total interindividual variance:\n' + r'$\sigma_\alpha^2 + \tilde{V}$',
             'hom2': 'Shared variance (hom)',
             'V': 'Mean ct-specific variance (V) across cts',
             'V_prop': 'Variance differentiation:\n' + r'$\tilde{V}\ /\ (\sigma_\alpha^2 + \tilde{V})$',
             'pV': 'Proportion of genes with significant \nvariance differentiation',
             'std(mean)': 'Mean differentiation:\n' + r'sd($\beta$)',
             'var(mean)': 'Mean differentiation:\n' + r'var($\beta$)',
             'pMean': 'Proportion of genes with significant \nmean differentiation',
             }

    mainfig_ylabs = { 
             'V_prop': 'Variance differentiation:\n' + r'$\tilde{V}\ /\ (\sigma_\alpha^2 + \tilde{V})$',
             'var(mean)': 'Mean differentiation: ' + r'var($\beta$)',
             }

    p_format = lambda x: format(x, f'.{3}f') if x >= 0.01 else format(x, f'.{2}e')
    
    # main figure
    fs = 6
    property = 'LOEUF'
    grouped = data.groupby(by=property + '_bin')
    tmp_data = data[[property + '_bin', 'V_prop', 'var(mean)', 'pV', 'pMean']].groupby(property + '_bin').mean().reset_index()

    ax1 = plt.subplot(gs[1, 1])

    line, p = meta_regression(grouped, 'V_prop', 'mean')
    ls = ':'
    ms = 40
    p = p_format(p)
    sns.pointplot(data=data, x=property + '_bin', y='V_prop', estimator='mean',
                errorbar='se', linestyles=ls, ax=ax1, 
                color=sns.color_palette()[0])
           
    plt.scatter(x=tmp_data[property + '_bin'], y=tmp_data['V_prop'],
                color=sns.color_palette()[0], s=ms)

    ax1.text(0.08, .10, f'regression p-value = {p}', color=sns.color_palette()[0], fontsize=fs, transform=ax1.transAxes)
    ax1.text(-0.15, 1.05, '(c)', fontsize=7, transform=ax1.transAxes)
    # ax1.set_xlabel('')
    ax1.set_xlabel(xlabs[property], fontsize=7)
    ax1.set_ylabel(mainfig_ylabs['V_prop'], fontsize=7, color=sns.color_palette()[0])
    # plt.xticks(visible=False)
    plt.xticks(fontsize=fs)
    plt.yticks([0.90, 0.95, 1.00, 1.05], color=sns.color_palette()[0])

    # add arrow
    arrow_position = (0.7, 0.9)
    plt.annotate('', xy=(arrow_position[0] - 0.42, arrow_position[1]), xytext=arrow_position,
             arrowprops=dict(arrowstyle='->', lw=1.5),
             fontsize=fs, ha='center', va='center', xycoords=ax1.transAxes)
    plt.text(0.5, 0.92, 'more constrained', ha='center', va='bottom', fontsize=fs, transform=ax1.transAxes)

    # 2nd y-axis
    ax1_2 = ax1.twinx()

    line, p = meta_regression(grouped, 'var(mean)', 'mean')
    p = p_format(p)
    sns.pointplot(data=data, x=property + '_bin', y='var(mean)', estimator='mean',
                errorbar='se', linestyles=ls, ax=ax1_2, 
                color=sns.color_palette()[1])
           
    plt.scatter(x=tmp_data[property + '_bin'], y=tmp_data['var(mean)'],
                color=sns.color_palette()[1], s=ms)

    ax1_2.text(0.08, .02, f'regression p-value = {p}', color=sns.color_palette()[1], fontsize=fs, transform=ax1_2.transAxes)
    ax1_2.set_ylabel(mainfig_ylabs['var(mean)'], fontsize=7, color=sns.color_palette()[1])
    plt.yticks(color=sns.color_palette()[1])

    #
    property = 'EDS'
    grouped = data.groupby(by=property + '_bin')
    tmp_data = data[[property + '_bin', 'V_prop', 'var(mean)', 'pV', 'pMean']].groupby(property + '_bin').mean().reset_index()

    ax2 = plt.subplot(gs[0, 1])
    ax2.text(-0.15, 1.05, '(b)', fontsize=7, transform=ax2.transAxes)

    line, p = meta_regression(grouped, 'V_prop', 'mean')
    # line, p = meta_regression(grouped, 'pV', 'mean')
    p = p_format(p)
    # sns.pointplot(data=data, x=property + '_bin', y='pV', estimator='mean',
    sns.pointplot(data=data, x=property + '_bin', y='V_prop', estimator='mean',
                errorbar='se', linestyles=ls, ax=ax2, color=sns.color_palette()[0])
           
    # plt.scatter(x=tmp_data[property + '_bin'], y=tmp_data['pV'],
    plt.scatter(x=tmp_data[property + '_bin'], y=tmp_data['V_prop'],
                color=sns.color_palette()[0], s=ms)

    ax2.text(0.96, .10, f'regression p-value = {p}', fontsize=fs, ha='right', color=sns.color_palette()[0], transform=ax2.transAxes)
    ax2.set_xlabel(xlabs[property], fontsize=7)
    ax2.set_ylabel(mainfig_ylabs['V_prop'], fontsize=7, color=sns.color_palette()[0])
    # ax2.set_ylabel(ylabs['pV'], fontsize=12, color=sns.color_palette()[0])
    plt.xticks(fontsize=fs)
    plt.yticks(color=sns.color_palette()[0])

    # add arrow
    arrow_position = (0.05, 0.9)
    plt.annotate('', xy=(arrow_position[0] + 0.52, arrow_position[1]), xytext=arrow_position,
             arrowprops=dict(arrowstyle='->', lw=1.5),
             fontsize=fs, ha='center', va='center', xycoords=ax2.transAxes)
    plt.text(0.305, 0.92, 'larger enhancer domains', ha='center', va='bottom', transform=ax2.transAxes)

    # 2nd y-axis
    ax2_2 = ax2.twinx()

    line, p = meta_regression(grouped, 'var(mean)', 'mean')
    # line, p = meta_regression(grouped, 'pMean', 'mean')
    p = p_format(p)
    # sns.pointplot(data=data, x=property + '_bin', y='pMean', estimator='mean',
    sns.pointplot(data=data, x=property + '_bin', y='var(mean)', estimator='mean',
                errorbar='se', linestyles=ls, ax=ax2_2, color=sns.color_palette()[1])
           
    # plt.scatter(x=tmp_data[property + '_bin'], y=tmp_data['pMean'],
    plt.scatter(x=tmp_data[property + '_bin'], y=tmp_data['var(mean)'],
                color=sns.color_palette()[1], s=ms)

    ax2_2.text(0.96, .02, f'regression p-value = {p}', fontsize=fs, ha='right', color=sns.color_palette()[1], transform=ax2_2.transAxes)
    ax2_2.set_ylabel(mainfig_ylabs['var(mean)'], fontsize=fs, color=sns.color_palette()[1])
    # ax2_2.set_ylabel(ylabs['pMean'], fontsize=12, color=sns.color_palette()[1])
    plt.yticks(color=sns.color_palette()[1])

    # supp figure 
    fs = 18
    mpl.rcParams["lines.linewidth"] = 1.2
    mpl.rcParams["font.size"] = 10
    median_marker = 's'
    var_median_color = '#2EA4DB'
    mean_median_color = '#FF950A'
    ms = 170
    fig, axes = plt.subplots(nrows=len(features), ncols=len(propertys), 
                             sharex='col', sharey='row', dpi=300,
                             figsize=(6 * len(propertys), 5 * len(features)))


    for i, feature in enumerate(features):
        for j, property in enumerate(propertys):
            ax = axes[i, j]
            
            # meta regression
            grouped = data.groupby(property + '_bin')
            if len(data[property].unique()) > num_bins:
                line, p = meta_regression(grouped, feature, 'mean')
                ls = '-' if p < 0.05 else '--'                
                p = p_format(p)
            else:
                p = np.nan
                ls = '--'

            if feature not in ['pV', 'pMean']:
                mean_color = sns.color_palette()[0] if feature != 'var(mean)' else sns.color_palette()[1]
                sns.pointplot(data=data, x=property + '_bin', y=feature, estimator='mean',
                            errorbar='se', color=mean_color, linestyles=ls, label='mean', ax=ax)
                 
                tmp_data = grouped[feature].mean().reset_index()
                ax.scatter(x=tmp_data[property + '_bin'], y=tmp_data[feature],
                            color=mean_color, s=ms)

                if len(data[property].unique()) > num_bins:
                    line2, p2 = meta_regression(grouped, feature, 'median')
                    ls2 = '-' if p2 < 0.05 else '--'                
                    p2 = p_format(p2)
                else:
                    p2 = np.nan
                    ls2 = '--'

                ax.text(0.15, .95, f'regression p-values:\n  p(mean) = {p}\n  p(median) = {p2}', 
                                va='top', fontsize=fs, transform=ax.transAxes)

                median_color = var_median_color if feature != 'var(mean)' else mean_median_color
                sns.pointplot(data=data, x=property + '_bin', y=feature, estimator='median',
                            errorbar='se', linestyles=ls2, label='median', 
                            markers=median_marker, color=median_color, ax=ax)
                
                tmp_data = grouped[feature].median().reset_index()
                ax.scatter(x=tmp_data[property + '_bin'], y=tmp_data[feature],
                            color=median_color, marker=median_marker, s=ms)

            else:           
                p_color = '#3995DB' if feature != 'pMean' else '#FF8A0E'
                sns.pointplot(data=data, x=property + '_bin', y=feature, estimator='mean',
                            errorbar='se', linestyles=ls, ax=ax, color=p_color)

                tmp_data = grouped[feature].mean().reset_index()
                ax.scatter(x=tmp_data[property + '_bin'], y=tmp_data[feature],
                            color=p_color, s=ms)

                ax.text(0.15, .02, f'regression p-value = {p}', 
                                va='bottom', fontsize=fs, transform=ax.transAxes)

            if property == 'pLI':
                ax.axvline(x=7.5, color='0.6', ls='--', zorder=10)
            
            if i == len(features) - 1:
                ax.tick_params(axis='x', rotation=0)
                ax.set_xlabel(xlabs[property], fontsize=24)
            else:
                ax.set_xlabel('')
            ax.set_ylabel('')

        axes[i, 0].set_ylabel(ylabs[feature], fontsize=20)
    axes[0, 0].legend(loc=4, fontsize=20, markerscale=3)
    
    fig.tight_layout()
    fig.savefig(png)


    # main figure
    # property = 'LOEUF'
    # grouped = data.groupby(property + '_bin')



def main():
    # par 
    out = np.load(snakemake.input.out, allow_pickle=True).item()
    remlJK = np.load(snakemake.input.remlJK, allow_pickle=True).item()
    marker = ['NANOG','T','GATA6']
    candidate = ['POU5F1', 'NDUFB4']

    eds = pd.read_table(snakemake.input.eds)
    eds['gene_length (kb)'] = eds['gene_length'] / 1e3

    # p
    reml_beta_ps = remlJK['reml']['wald']['free']['ct_beta']
    reml_V_ps = remlJK['reml']['wald']['free']['V']
    he_beta_ps = out['he']['wald']['free']['ct_beta']
    he_V_ps = out['he']['wald']['free']['V']

    reml_beta_ps = np.log10(reml_beta_ps) * (-1)
    reml_V_ps = np.log10(reml_V_ps) * (-1)
    he_beta_ps = np.log10(he_beta_ps) * (-1)
    he_V_ps = np.log10(he_V_ps) * (-1)

    genes = np.array([x.split('_')[1] for x in out['gene']])
    remlJK_genes = np.array([x.split('_')[1] for x in remlJK['gene']])
    # print(len(genes), len(remlJK_genes))

    # make subplots for reml p value plot and main matrix plot
    # mpl.rcParams["lines.linewidth"] = 0.7
    mpl.rcParams.update({'font.size': 5, 'font.family': 'sans-serif', 'lines.linewidth': 0.7})
    main_fig = plt.figure(figsize=(7.08, 4.0), dpi=600)
    grids = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=(1.7, 1))

    # p values scatter plot
    remlJK_data = pd.DataFrame({'reml_beta':reml_beta_ps, 'reml_V':reml_V_ps, 'gene': remlJK_genes})
    remlJK_data.to_csv(snakemake.output.reml_data, sep='\t', index=False)
    data = pd.DataFrame({'he_beta':he_beta_ps, 'he_V':he_V_ps,'gene':genes})
    data.to_csv(snakemake.output.he_data, sep='\t', index=False)
    remlJK_threshold = (-1) * math.log10(0.05 / remlJK_data.shape[0])
    threshold = (-1) * math.log10(0.05 / data.shape[0])

    for method, tmp_data, tmp_threshold, png_f in zip(['reml','he'], [remlJK_data, data], 
            [remlJK_threshold, threshold], [snakemake.output.reml_p, snakemake.output.he_p]):

        # plot
        if method == 'he':
            fig, ax = plt.subplots(dpi=600)
        elif method == 'reml':
            ax = plt.subplot(grids[:, 0])
            ax.text(-0.05, 1.02, '(a)', fontsize=7, transform=ax.transAxes)

        ## heatscatter
        draw.scatter(tmp_data[method+'_beta'], tmp_data[method+'_V'], s=5, heatscatter=True, linregress=False, ax=ax)

        # add nonsignificant V in HE
        #tmp = data.loc[data['he_V'] < threshold]
        #ax.scatter(tmp['reml_beta'], tmp['reml_V'], s=5, c='0.8')

        # add three markers
        ax.scatter(tmp_data.loc[tmp_data['gene'].isin(marker), method+'_beta'],
                tmp_data.loc[tmp_data['gene'].isin(marker), method+'_V'], s=10, c='m')
        # add candidate genes
        ax.scatter(tmp_data.loc[tmp_data['gene'].isin(candidate), method+'_beta'],
                tmp_data.loc[tmp_data['gene'].isin(candidate), method+'_V'], s=10, c='r')


        ## add arrow to three markers
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for index, row in tmp_data.loc[tmp_data['gene'].isin(marker+candidate)].iterrows():
            print(row['gene'])
            if row[method+'_beta'] < ((xlim[1]-xlim[0])*0.9+xlim[0]):
                if method == 'reml':
                    if row['gene'] == 'T':
                        shiftx = (xlim[1]-xlim[0])/20
                        shifty = (ylim[1]-ylim[0])/10
                    elif row['gene'] == 'GATA6':
                        shiftx = (xlim[1]-xlim[0])/15
                        shifty = (ylim[1]-ylim[0])/10
                    else:
                        shiftx = (xlim[1]-xlim[0])/15
                        shifty = (ylim[1]-ylim[0])/8
                    if row[method+'_V'] > ((ylim[1]-ylim[0])*0.8+ylim[0]):
                        shifty = shifty * (-1)
                    ax.annotate(row['gene'], xy=(row[method + '_beta'], row[method + '_V']),
                            xytext=(row[method + '_beta'] + shiftx, row[method + '_V'] + shifty),
                            arrowprops={'arrowstyle':'->'})
                elif method == 'he':
                    shiftx = (xlim[1]-xlim[0])/10
                    shifty = (ylim[1]-ylim[0])/10
                    if row[method+'_V'] > ((ylim[1]-ylim[0])*0.8+ylim[0]):
                        shifty = shifty * (-1)
                    ax.annotate(row['gene'], xy=(row[method + '_beta'], row[method + '_V']),
                            xytext=(row[method + '_beta'] + shiftx, row[method + '_V'] + shifty),
                            arrowprops={'arrowstyle':'->'})
            else:
                ax.annotate(row['gene'], xy=(row[method + '_beta'], row[method + '_V']),
                        xytext=(row[method + '_beta'] + (xlim[1] - xlim[0]) / 40, row[method + '_V'] + (ylim[1] - ylim[0]) / 15),
                        arrowprops={'arrowstyle':'->'})

        ax.axhline(tmp_threshold, color='0.8', ls='--', zorder=0)

        ax.axvline(tmp_threshold, color='0.8', ls='--', zorder=0)

        ax.set_xlabel('$-log_{10}$ p(mean differentiation)', fontsize=7)
        ax.set_ylabel('$-log_{10}$ p(variance differentiation)', fontsize=7)

        if method == 'he':
            fig.tight_layout(pad=1)
            fig.savefig(png_f)
        elif method == 'reml':
            # matrix plots
            gene_ids = [gene.split('_')[0] for gene in out['gene']]
            Vs = np.diagonal(remlJK['reml']['free']['V'], axis1=1, axis2=2)
            gs = Vs + remlJK['reml']['free']['hom2'][:, np.newaxis]
            # beta_p = out[method]['wald']['free']['ct_beta']
            # beta_p[beta_p == 0] = np.amin(beta_p[beta_p != 0])
            data = pd.DataFrame({
                                'gene': gene_ids, 
                                'V': np.mean(Vs, axis=1), 
                                'hom2': remlJK['reml']['free']['hom2'], 
                                'g': np.mean(gs, axis=1),
                                'pV': reml_V_ps > (-1 * np.log10(0.05 / len(gene_ids))),
                                'var(mean)': np.var(remlJK['reml']['free']['beta']['ct_beta'], axis=1, ddof=1),
                                'pMean': reml_beta_ps > (-1 * np.log10(0.05 / len(gene_ids))),
                                })
            data['V_prop'] = data['V'] / data['g']


            data = data.merge(eds, left_on='gene', right_on='GeneSymbol')
            data.to_csv(snakemake.output.matrix_data, sep='\t', index=False)

            matrix_plot(data, ['g', 'V_prop', 'pV', 'var(mean)', 'pMean'], 
                        ['LOEUF', 'pLI', 'ActivityLinking_EnhancerNumber', 'EDS'], 
                        10, snakemake.output.matrix, grids)

            main_fig.tight_layout()
            main_fig.savefig(snakemake.output.reml_p)



    # QQ plot
    reml_beta_ps = np.sort( remlJK['reml']['wald']['free']['ct_beta'] )
    reml_V_ps = np.sort( remlJK['reml']['wald']['free']['V'] )
    he_beta_ps = np.sort( out['he']['wald']['free']['ct_beta'] )
    he_V_ps = np.sort( out['he']['wald']['free']['V'] )

    fig, ax = plt.subplots(dpi=600)

    osm, osr = stats.probplot(reml_V_ps, dist='uniform', fit=False)
    ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Variance differentiation in REML' )

    osm, osr = stats.probplot(he_V_ps, dist='uniform', fit=False)
    ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Variance differentiation in HE' )

    osm, osr = stats.probplot(reml_beta_ps, dist='uniform', fit=False)
    ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Mean differentiation in REML' )

    osm, osr = stats.probplot(he_beta_ps, dist='uniform', fit=False)
    ax.scatter( (-1) * np.log10(osm), (-1) * np.log10(osr), marker='.', label='Mean differentiation in HE' )

    plt.legend()

    lims = [
            np.max([ax.get_xlim()[0], ax.get_ylim()[0]]),  # min of both axes
            np.min([ax.get_xlim()[1], ax.get_ylim()[1]]),  # max of both axes
            ]

    ax.plot(lims, lims, '--', color='0.8', zorder=0)

    ax.set_xlabel('Expected $-log_{10}P$')
    ax.set_ylabel('Observed $-log_{10}P$')

    fig.savefig(snakemake.output.qq)


if __name__ == '__main__':
    main()

