import numpy as np, pandas as pd

def main():
    # par
    y_fs = []
    nu_fs = []
    for f in snakemake.input.imputed_ct_y:
        for line in open(f):
            y_fs.append(line.strip())
    for f in snakemake.input.imputed_ct_nu:
        for line in open(f):
            nu_fs.append(line.strip())

    # confirm donor and day order
    y = pd.read_table(y_fs[0])
    donors = np.array(y['donor'])
    days = np.array(y['day'])
    for i in range(1, len(y_fs), 50):
        y_ = pd.read_table( y_fs[i] )
        y_donors_ = np.array( y_['donor'] )
        y_days_ = np.array( y_['day'] )
        nu_ = pd.read_table( nu_fs[i] )
        nu_donors_ = np.array( nu_['donor'] )
        nu_days_ = np.array( nu_['day'] )
        if np.any(donors != y_donors_) or np.any(days != y_days_) or np.any(donors != nu_donors_) or np.any(days != nu_days_):
            print(y_fs[i])
            print(nu_fs[i])
            print(donors)
            print(donors_)
            print(days)
            print(days_)
            sys.exit('Not machting!\n')

    # collect genes
    ys = {'donor':donors, 'day':days}
    for f in y_fs:
        y = pd.read_table(f)
        gene = y.columns[-1]
        ys[gene] = y[gene]
    ys = pd.DataFrame(ys)

    nus = {'donor':donors, 'day':days}
    for f in nu_fs:
        nu = pd.read_table(f)
        gene = nu.columns[-1]
        nus[gene] = nu[gene]
    nus = pd.DataFrame(nus)

    #
    ys.to_csv(snakemake.output.merged_ct_y, sep='\t', index=False)
    nus.to_csv(snakemake.output.merged_ct_nu, sep='\t', index=False)

if __name__ == '__main__':
    main()
