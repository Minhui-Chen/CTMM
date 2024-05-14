import numpy as np
import pandas as pd


def main():
    # 
    out = np.load(snakemake.input.out, allow_pickle=True).item()
    remlJK = np.load(snakemake.input.remlJK, allow_pickle=True).item()

    #
    he = {'gene': out['gene'], 'he': out['he']['wald']['free']['V']}
    reml = {'gene': remlJK['gene'], 'reml': remlJK['reml']['wald']['free']['V']}

    he = pd.DataFrame(he)
    reml = pd.DataFrame(reml)

    # 
    data = he.merge(reml, on='gene')

    # sort by reml p
    data = data.sort_values('reml')

    # top genes 
    p_cut = 0.05 / data.shape[0]
    tmp = data.loc[(data['he'] < p_cut) & (data['reml'] < p_cut)]
    top = tmp.head(snakemake.params.n)['gene'].to_numpy()
    np.savetxt(snakemake.output.top, top, fmt='%s')

    # bot genes
    bot = data.tail(snakemake.params.n)['gene'].to_numpy()
    np.savetxt(snakemake.output.bot, bot, fmt='%s')


if __name__ == '__main__':
    main()
    