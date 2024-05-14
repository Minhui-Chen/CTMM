import sys
import numpy as np, pandas as pd


def main():
    raw = pd.read_table(snakemake.input.raw)
    sim = pd.read_table(snakemake.input.sim)

    # keep the first cell type
    raw = raw.loc[raw['ct'] == 'day0']
    raw = raw.drop(columns='ct')
    sim = sim.loc[sim['ct'] == 'ct1']
    sim = sim.drop(columns='ct')

    # remove genes not in the first cell type
    sim = sim.loc[sim['gene'].isin(raw['gene'])]

    # find common inds across genes
    data = raw.drop_duplicates(subset=['ind', 'gene'])
    data = data.pivot(index='ind', columns='gene')
    print(data.shape)
    data = data.dropna()
    print(data.shape)
    inds = data.index.to_list()

    # keep common inds
    raw = raw.loc[raw['ind'].isin(inds)]
    sim = sim.loc[sim['ind'].isin(inds)]

    # sort by ind and gene
    raw = raw.sort_values(by=['ind', 'gene']).reset_index(drop=True)
    sim = sim.sort_values(by=['ind', 'gene']).reset_index(drop=True)

    # sanity check ind gene matching
    if raw[['ind', 'gene']].equals(sim[['ind', 'gene']]):
        pass
    else:
        raw = raw[['ind', 'gene']].reset_index(drop=True)
        sim = sim[['ind', 'gene']].reset_index(drop=True)
        print(raw.iloc[0, 0] == sim.iloc[0, 0])
        print(raw.iloc[0, 1] == sim.iloc[0, 1])
        idx = (raw == sim).all(axis=1)
        print(raw[~idx])
        print(sim[~idx])
        print(raw.shape, sim.shape)
        print(raw.head())
        print(sim.head())
        sys.exit('Wrong')

    # index cells
    raw['cell'] = raw.groupby(['ind', 'gene']).cumcount() + 1
    raw['cell'] = 'raw_' + raw['ind'] + '_' + raw['cell'].astype('str')
    sim['cell'] = sim.groupby(['ind', 'gene']).cumcount() + 1
    sim['cell'] = 'sim_' + sim['ind'] + '_' + sim['cell'].astype('str')

    # random assign case / control
    raw['treat'] = 'control'
    sim['treat'] = 'control'
    raw.loc[raw['ind'].isin(inds[:int(len(inds) / 2)]), 'treat'] = 'case'
    sim.loc[sim['ind'].isin(inds[:int(len(inds) / 2)]), 'treat'] = 'case'

    # save meta
    raw_meta = raw[['cell', 'ind', 'treat']].drop_duplicates().reset_index(drop=True)
    sim_meta = sim[['cell', 'ind', 'treat']].drop_duplicates().reset_index(drop=True)
    raw_meta.to_csv(snakemake.output.raw_meta, sep='\t', index=False)
    sim_meta.to_csv(snakemake.output.sim_meta, sep='\t', index=False)
     
    # transpose
    raw = raw.pivot(index='gene', columns='cell', values='count')
    sim = sim.pivot(index='gene', columns='cell', values='count')
    raw[raw_meta['cell']].to_csv(snakemake.output.raw, sep='\t', na_rep='NA')
    sim[sim_meta['cell']].to_csv(snakemake.output.sim, sep='\t', na_rep='NA')
    

if __name__ == '__main__':
    main()