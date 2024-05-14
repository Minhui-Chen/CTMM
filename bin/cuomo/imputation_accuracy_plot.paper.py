import re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    #
    labels = []
    for index, row in snakemake.params.labels.iterrows():
        label = []
        if row['im_mvn'] == 'Y':
            label.append('MVN')
        else:
            if row['im_scale'] == 'Y':
                label.append('softImpute_Scale')
            else:
                label.append('softImpute_noScale')
        if row['im_genome'] == 'Y':
            label.append('(transcriptome)')
        else:
            label.append('(gene)')
#        if row['random_mask'] == 'Y':
#            label.append('random')
#        else:
#            label.append('struct')
        #label.append(row['missingness'])
        label = '\n'.join(label)
        labels.append(label)
    print(labels)
    y_mses = [pd.read_csv(f,index_col=(0,1),names=[label]) 
            for f,label in zip(snakemake.input.y_mse,labels)]
    y_mses = pd.concat(y_mses, axis=1)
    nu_mses = [pd.read_csv(f,index_col=(0,1),names=[label]) 
            for f,label in zip(snakemake.input.nu_mse,labels)]
    nu_mses = pd.concat(nu_mses, axis=1)
    y_cors = [pd.read_csv(f,index_col=(0,1),names=[label]) for f,label in zip(snakemake.input.y_cor,labels)]
    y_cors = pd.concat(y_cors, axis=1)
    nu_cors = [pd.read_csv(f,index_col=(0,1),names=[label]) for f,label in zip(snakemake.input.nu_cor,labels)]
    nu_cors = pd.concat(nu_cors, axis=1)

    # remove no scale 
    labels = [label for label in labels if 'noScale' not in label]
    y_mses = y_mses[labels]
    nu_mses = nu_mses[labels]
    y_cors = y_cors[labels]
    nu_cors = nu_cors[labels]
    for label in labels:
        if '_Scale' in label:
            y_mses = y_mses.rename(columns={label:re.sub('_Scale','',label)})
            nu_mses = nu_mses.rename(columns={label:re.sub('_Scale','',label)})
            y_cors = y_cors.rename(columns={label:re.sub('_Scale','',label)})
            nu_cors = nu_cors.rename(columns={label:re.sub('_Scale','',label)})

    # plot
    plt.rcParams.update({'font.size' : 14})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8), dpi=600)
    #y_mses
    y_mses_ = y_mses.loc[~np.any(y_mses.isna(), axis=1)]
    tmp = np.array(y_mses_).flatten()
    quans = np.quantile(tmp, [0.1,0.9])
    tmp = tmp[tmp<quans[1]]
    mean_ = np.mean(tmp)
    std_ = np.std(tmp)
    #y_mses_[y_mses_ > (mean_ + 3*std_)] = mean_ + 3*std_
    y_mses_[y_mses_ > 1.5] = 1.5
    sns.violinplot(data=y_mses_, ax=axes[0,0], orient='h', cut=0)
    axes[0,0].text(-0.05, 1.05, '(A)', transform=axes[0,0].transAxes, fontsize=16)

    # nu_mses
    nu_mses_ = nu_mses.loc[~np.any(nu_mses.isna(), axis=1)]
    tmp = np.array(nu_mses_).flatten()
    quans = np.quantile(tmp, [0.1,0.9])
    tmp = tmp[tmp<quans[1]]
    mean_ = np.mean(tmp)
    std_ = np.std(tmp)
    #nu_mses_[nu_mses_ > (mean_ + 3*std_)] = mean_ + 3*std_
    nu_mses_[nu_mses_ > 2] = 2
    sns.violinplot(data=nu_mses_, ax=axes[0,1], orient='h', cut=0)
    axes[0,1].text(-0.05, 1.05, '(B)', transform=axes[0,1].transAxes, fontsize=16)

    # y cors
    y_cors_ = y_cors.loc[~np.any(y_cors.isna(), axis=1)]
    #mean_ = np.mean(np.array(y_cors_))
    #std_ = np.std(np.array(y_cors_))
    #y_cors_[y_cors_ > (mean_ + 3*std_)] = mean_ + 3*std_
    #y_cors_[y_cors_ < (mean_ - 3*std_)] = mean_ - 3*std_
    #y_cors_[y_cors_ < 0] = 0
    sns.violinplot(data=y_cors_, ax=axes[1,0], orient='h', cut=0)
    axes[1,0].text(-0.05, 1.05, '(C)', transform=axes[1,0].transAxes, fontsize=16)

    # nu cors
    nu_cors_ = nu_cors.loc[~np.any(nu_cors.isna(), axis=1)]
    #mean_ = np.mean(np.array(nu_cors_))
    #std_ = np.std(np.array(nu_cors_))
    #nu_cors_[nu_cors_ > (mean_ + 3*std_)] = mean_ + 3*std_
    #nu_cors_[nu_cors_ < (mean_ - 3*std_)] = mean_ - 3*std_
    #nu_cors_[nu_cors_ < 0] = 0
    sns.violinplot(data=nu_cors_, ax=axes[1,1], orient='h', cut=0)
    axes[1,1].text(-0.05, 1.05, '(D)', transform=axes[1,1].transAxes, fontsize=16)

    axes[0,0].set_xlabel('MSE(y)')
    axes[0,1].set_xlabel('MSE($\\nu$)')
    axes[1,0].set_xlabel('cor(y)')
    axes[1,1].set_xlabel('cor($\\nu$)')

    fig.tight_layout()

    fig.savefig(snakemake.output.png)

if __name__ == '__main__':
    main()
