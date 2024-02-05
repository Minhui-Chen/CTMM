import shutil, re, sys, os, gzip, math, scipy, copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from snakemake.utils import Paramspace


# make a logs folders to save log files
os.makedirs('./logs/', exist_ok=True)

mycolors = sns.color_palette()
pointcolor = 'red' # color for expected values  in estimates plots

def get_subspace(arg, model_params):
    ''' model_params df include or not include the column of model, but the first row is the basemodel'''
    if 'model' in model_params.columns:
        model_params = model_params.drop('model', axis=1)
    model_params = model_params.reset_index(drop=True)
    basemodel = model_params.iloc[0].to_dict()
    columns = model_params.columns.to_list()
    columns.remove(arg)
    evaluate = ' & '.join([f"{_} == '{basemodel[_]}'" for _ in columns])
    subspace = model_params[model_params.eval(evaluate)]
    return Paramspace(subspace, filename_params="*")

def get_effective_args(model_params):
    ''' model_params df include or not include the column of model, but the first row is the basemodel'''
    #print(model_params)
    if 'model' in model_params.columns:
        model_params = model_params.drop('model', axis=1)
    effective_args = [] # args with multiple parameters. e.g. ss has '1e2', '2e2', '5e2'
    for arg_ in np.array(model_params.columns):
        subspace = get_subspace(arg_, model_params)
        if subspace.shape[0] > 1:
            effective_args.append(arg_)
    return effective_args

# wildcard constraints
wildcard_constraints: i='[0-9]+' 
wildcard_constraints: l='[^/]+' 
wildcard_constraints: group='[^/]+' 
wildcard_constraints: m='[0-9]+' 
wildcard_constraints: model='\w+'
wildcard_constraints: cut_off = '[\d\.]+'
wildcard_constraints: prop = '[\d\.]+'
wildcard_constraints: pseudocount='[^/]+' 
wildcard_constraints: meandifference='[^/]+' 

#########################################################################################
# OP Simulation
#########################################################################################
# par
op_replicates = 1000
op_batch_no = 100
op_batches = np.array_split(range(op_replicates), op_batch_no)

## paramspace
op_params = pd.read_table("op.params.txt", dtype="str", comment='#', na_filter=False)
if op_params.shape[0] != op_params.drop_duplicates().shape[0]:
    sys.exit('Duplicated parameters!\n')
op_paramspace = Paramspace(op_params.drop('model', axis=1), filename_params="*")

op_plot_order = {
        'hom':{
            'ss':['2e1', '5e1', '1e2', '2e2', '3e2', '5e2', '1e3'], 
            'a':['0.5_2_2_2', '1_2_2_2', '2_2_2_2', '4_2_2_2'],
            'c': [2, 4, 6, 8, 10, 12]
            },
        'iid':{
            'ss':['2e1', '5e1', '1e2', '3e2', '5e2', '1e3'], 
            'a':['0.5_2_2_2', '1_2_2_2', '2_2_2_2', '4_2_2_2'],
            'vc':['0.25_0.40_0.10_0.25', '0.25_0.30_0.20_0.25', '0.25_0.25_0.25_0.25', 
                '0.25_0.20_0.30_0.25', '0.25_0.10_0.40_0.25']
            }, 
        'free': {
            'ss':['2e1', '5e1', '1e2', '2e2', '3e2', '5e2', '1e3'], 
            'a':['0.32_2_2_2','0.5_2_2_2', '0.666_2_2_2', '0.77_2_2_2', 
                '0.88_2_2_2', '1_2_2_2', '1.05_2_2_2', '2_2_2_2', '4_2_2_2'], 
            'vc':['0.25_0.40_0.10_0.25', '0.25_0.30_0.20_0.25', '0.25_0.25_0.25_0.25', 
                '0.25_0.20_0.30_0.25', '0.25_0.10_0.40_0.25'],
            'V_diag':['1_1_1_1', '8_4_2_1', '27_9_3_1', '64_16_4_1'],
            'c': [2, 4, 6, 8, 10, 12]
            },
        'full':{
            'ss':['2e1', '5e1', '1e2', '3e2', '5e2', '1e3'], 
            'a':['0.5_2_2_2', '1_2_2_2', '2_2_2_2', '4_2_2_2'],
            'vc':['0.25_0.40_0.10_0.25', '0.25_0.30_0.20_0.25', '0.25_0.25_0.25_0.25', 
                '0.25_0.20_0.30_0.25', '0.25_0.10_0.40_0.25'],
            'V_diag':['1_1_1_1', '8_4_2_1', '27_9_3_1', '64_16_4_1', '64_64_1_1'],
            'V_tril':['0.25_0.25_0_-0.25_0_0', '0.5_0.5_0_-0.5_0_0', '0.75_0.75_0_-0.75_0_0', 
                '0.95_0.95_0.95_-0.95_-0.95_-0.95']
            },
        }


rule op_parameters:
    output:
        pi = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/PI.txt',
        s = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/S.txt',
        beta = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/V.txt',
    priority: 100
    script: 'bin/sim/op_parameters.py'


rule op_simulation:
    input:
        beta = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/V.txt',
    output:
        P = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        pi = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/estPI.batch{{i}}.txt',
        s = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/estS.batch{{i}}.txt',
        nu = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        y = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
        ctnu = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
        cty = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        # add a test fixed effect. if it's not needed, this file is 'NA'
        fixed = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/fixed.X.batch{{i}}.txt',
        # add a test random effect. if it's not needed, this file is 'NA'
        random = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/random.X.batch{{i}}.txt',
    params:
        batch = lambda wildcards: op_batches[int(wildcards.i)],
        P = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/repX/P.txt',
        pi = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/repX/estPI.txt',
        s = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/repX/estS.txt',
        nu = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/repX/nu.txt',
        y = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/repX/y.txt',
        ctnu = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/repX/ctnu.txt',
        cty = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/repX/cty.txt',
        fixed = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/repX/fixed.X.txt',
        random = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/repX/random.X.txt',
        seed = 235435,
    priority: 99
    script: 'bin/sim/opNctp_simulation.py'


rule op_test:
    input:
        y = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
        P = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.batch{{i}}',
    params:
        out = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/rep/out.npy',
        batch = lambda wildcards: op_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        method = 'BFGS',
    resources:
        mem_mb = '5G',
        time = '50:00:00',
    priority: 98
    script: 'bin/sim/op_R.py'


rule op_aggReplications:
    input:
        out = [f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(op_batches))],
    output:
        out = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.npy',
    priority: 97
    script: 'bin/mergeBatches.py'


use rule op_test as op_test_remlJK with:
    input:
        y = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
        P = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.remlJK.batch{{i}}',
    params:
        out = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/rep/out.remlJK.npy',
        batch = lambda wildcards: op_batches[int(wildcards.i)],
        ML = False,
        REML = True,
        Free_reml_only = True,
        Free_reml_jk = True,
        HE = False,
    resources:
        mem_mb = '10gb',
        time = '50:00:00',
    priority: 98


use rule op_aggReplications as op_remlJK_aggReplications with:
    input:
        out = [f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.remlJK.batch{i}' 
                for i in range(len(op_batches))],
    output:
        out = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.remlJK.npy',
    priority: 100



#########################################################################################
# CTP
#########################################################################################
# par
ctp_replicates = 1000
ctp_batch_no = 250
ctp_batches = np.array_split(range(ctp_replicates), ctp_batch_no)

use rule op_simulation as ctp_simulation with:
    input:
        beta = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/V.txt',
    output:
        P = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        pi = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/estPI.batch{{i}}.txt',
        s = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/estS.batch{{i}}.txt',
        nu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        y = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
        ctnu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
        cty = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        fixed = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/fixed.X.batch{{i}}.txt',
        random = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/random.X.batch{{i}}.txt',
    params:
        batch = lambda wildcards: ctp_batches[int(wildcards.i)],
        P = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/P.txt',
        pi = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/estPI.txt',
        s = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/estS.txt',
        nu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/nu.txt',
        y = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/y.txt',
        ctnu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/ctnu.txt',
        cty = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/cty.txt',
        fixed = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/fixed.X.txt',
        random = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/random.X.txt',
        seed = 376487,

rule ctp_test:
    input:
        y = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        P = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.batch{{i}}',
    params:
        out = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/rep/out.npy',
        batch = lambda wildcards: ctp_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        optim_by_R = True,
    resources:
        mem_mb = '5gb',
        time = '48:00:00',
    priority: 97
    script: 'bin/sim/ctp.py'


use rule op_aggReplications as ctp_aggReplications with:
    input:
        out = [f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(ctp_batches))],
    output:
        out = f'analysis/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.npy',


rule ctp_test_remlJK:
    input:
        y = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        P = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.remlJK.batch{{i}}',
    params:
        out = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/rep/out.remlJK.npy',
        batch = lambda wildcards: ctp_batches[int(wildcards.i)],
        optim_by_R = True,
    resources:
        mem_mb = '10gb',
        time = '120:00:00',
    priority: 98
    script: 'bin/sim/ctp.remlJK.py'


use rule op_aggReplications as ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.remlJK.batch{i}' 
                for i in range(len(ctp_batches))],
    output:
        out = f'analysis/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.remlJK.npy',


rule paper_opNctp_power:
    input:
        op_hom = expand('analysis/op/hom/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom']).instance_patterns),
        op_free = expand('analysis/op/free/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
        op_hom_remlJK = expand('analysis/op/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom']).instance_patterns),
        op_free_remlJK = expand('analysis/op/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
        ctp_hom = expand('analysis/ctp/hom/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom']).instance_patterns),
        ctp_free = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
        ctp_hom_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom']).instance_patterns),
        ctp_free_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
    output:
        png = 'results/ctp/opNctp.power.paper.png',
    params: 
        hom = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom'])['ss']),
        free = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free'])['ss']),
        hom_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom'])['ss']),
        free_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free'])['ss']),
        plot_order = op_plot_order,
    script: 'bin/sim/opNctp_power.py'


#########################################################################################
# Cuomo et al 2020 Nature Communications
#########################################################################################
cuomo_ind_col = 'donor'
cuomo_ct_col = 'day'
cuomo_cell_col = 'cell_name'

####################### data ###########################
rule cuomo_data_download:
    output:
        raw = 'data/cuomo2020natcommun/raw_counts.csv.zip',
        log = 'data/cuomo2020natcommun/log_normalised_counts.csv.zip',
        meta = 'data/cuomo2020natcommun/cell_metadata_cols.tsv',
        supp2 = 'data/cuomo2020natcommun/suppdata2.txt',
    shell:
        '''
        mkdir -p $(dirname {output.log})
        cd $(dirname {output.log})
        wget https://zenodo.org/record/3625024/files/raw_counts.csv.zip
        wget https://zenodo.org/record/3625024/files/log_normalised_counts.csv.zip
        wget https://zenodo.org/record/3625024/files/cell_metadata_cols.tsv
        wget --no-check-certificate https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-14457-z/MediaObjects/41467_2020_14457_MOESM4_ESM.txt
        mv 41467_2020_14457_MOESM4_ESM.txt suppdata2.txt
        '''

rule cuomo_data_format:
    input:
        log = 'data/cuomo2020natcommun/log_normalised_counts.csv.zip',
    output:
        log = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    params:
        log = lambda wildcards, input: re.sub('\.zip$', '', input.log),
    shell:
        '''
        unzip -p {input.log} | tr ',' '\t' | sed 's/"//g' | gzip > {output.log}
        '''

rule cuomo_extractLargeExperiments:
    # extract the largest experiment for each individual
    input:
        meta = 'data/cuomo2020natcommun/cell_metadata_cols.tsv',
    output:
        meta = 'analysis/cuomo/data/meta.txt',
        png = 'analysis/cuomo/data/meta.png',
    script: 'bin/cuomo/extractLargeExperiments.py'


rule cuomo_cellno:
    input:
        meta = 'data/cuomo2020natcommun/cell_metadata_cols.tsv',
    output: 
        png = 'analysis/cuomo/data/cellno.png',
    run:
        meta = pd.read_table(input.meta, index_col=0)[['donor', 'day', 'cell_name']]
        grouped = meta.groupby(['donor', 'day'])
        cellno = grouped.size()
        cellno = cellno.unstack().stack(dropna=False).fillna(0)
        cellno = cellno.clip(upper=200)
        fig, ax = plt.subplots()
        bins = np.arange(0.1, 201, 10)
        bins[0] = -0.1
        plt.hist(cellno, bins=bins)
        ax.set_xlabel('Number of cells per individual-cell type pair')
        ax.set_ylabel('Number of individual-cell type pairs')
        ax.axvline(x=10, ls='--', color='0.7', zorder=10)
        fig.savefig(output.png)


rule cuomo_pseudobulk:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        y = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
    resources: 
        mem_mb = '10gb',
        time = '24:00:00',
    script: 'bin/cuomo/pseudobulk.py'

rule cuomo_day_pseudobulk_log_splitCounts:
    input:
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        counts = expand('staging/cuomo/bootstrapedNU/data/counts{i}.txt.gz', i=range(100)),
    resources:
        mem_mb = '10gb',
    script: 'bin/cuomo/day_pseudobulk_log_splitCounts.py'

rule cuomo_varNU:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'staging/cuomo/bootstrapedNU/data/counts{i}.txt.gz',
    output:
        var_nu = 'staging/cuomo/bootstrapedNU/data/counts{i}.var_nu.gz',
    resources: 
        mem_mb = '10gb',
        time = '48:00:00',
    script: 'bin/cuomo/varNU.py'

rule cuomo_varNU_merge:
    input:
        var_nu = expand('staging/cuomo/bootstrapedNU/data/counts{i}.var_nu.gz', i=range(100)),
    output:
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
    run:
        nus = [pd.read_table(f, index_col=(0,1)) for f in input.var_nu]
        # check donor day
        index = nus[0].index
        for data in nus[1:]:
            if np.any( index != data.index ):
                sys.exit('Wrop order!\n')
        # merge
        data = pd.concat( nus, axis=1 )
        data.to_csv( output.var_nu, sep='\t')

########### analysis ###############
cuomo_batch_no = 1000
## read parameters
cuomo_params = pd.read_table('cuomo.params.txt', dtype="str", comment='#')
cuomo_paramspace = Paramspace(cuomo_params, filename_params="*")


rule cuomo_filterInds:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        y = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
    output:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
    script: 'bin/cuomo/filterInds.py'


rule cuomo_filterInds_cellno:
    input:
        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/cellno.png',
    run:
        n = pd.read_table(input.n, index_col=0)
        cellno = n.stack()
        cellno = cellno.clip(upper=200)
        fig, ax = plt.subplots()
        bins = np.arange(0.1, 201, 10)
        bins[0] = -0.1
        plt.hist(cellno, bins=bins)
        ax.set_xlabel('Number of cells per individual-cell type pair')
        ax.set_ylabel('Number of individual-cell type pairs')
        ax.axvline(x=10, ls='--', color='0.7', zorder=10)
        fig.savefig(output.png)


rule cuomo_filterCTs:
    input:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
    output:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
    run:
        y = pd.read_table(input.y, index_col=(0,1))  # donor - day * gene
        nu = pd.read_table(input.nu, index_col=(0,1)) # donor - day * gene
        n = pd.read_table(input.n, index_col=0) # donor * day
        # set ct with less cells to missing
        ## find low ct
        low_cts = (n <= int(wildcards.ct_min_cellnum))
        low_cts_index = n[low_cts].stack().index
        ## set to NA
        y.loc[y.index.isin(low_cts_index)] = np.nan
        nu.loc[nu.index.isin(low_cts_index)] = np.nan
        y.to_csv(output.y, sep='\t')
        nu.to_csv(output.nu, sep='\t')


rule cuomo_split2batches:
    input:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz',
    output:
        y_batch = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{i}.txt' 
                for i in range(cuomo_batch_no)],
    run:
        y = pd.read_table(input.y, index_col=(0,1))
        # create batches
        genes = list(y.columns)
        genes = np.array_split(genes, len(output.y_batch))
        # output
        for genes_, y_batch_f in zip(genes, output.y_batch):
            with open(y_batch_f, 'w') as f:
                f.write( '\n'.join(genes_) )


rule cuomo_imputeGenome:
    input:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
        supp = 'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment 
    output:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene
    params:
        seed = 123453, # seed for softimpute
    resources: 
        mem_mb = '20gb',
        time = '20:00:00',
    script: 'bin/cuomo/imputeGenome.py'


rule cuomo_imputeNinput4OP:
    # also exclude individuals with nu = 0 which cause null model fail (some individuals have enough cells, but all cells have no expression of specific gene)
    # seems we should keep nu = 0
    input:
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene
        y_batch = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
    output:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # y for each gene is sorted by ind order
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt',
        nu_ctp = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctp.txt', 
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', 
        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/n.txt', 
        imputed_cty = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        imputed_ctnu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        imputed_ctnu_ctp = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
    resources: mem_mb = '10gb',
    script: 'bin/cuomo/imputeNinput4OP.py'


rule cuomo_y_collect:
    input:
        y = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/y.txt' 
                for i in range(cuomo_batch_no)],
    output:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',
    run:
        y_fs = []
        for f in input.y: y_fs = y_fs + [line.strip() for line in open(f)]
        y = []
        for f in y_fs: y.append(np.loadtxt(f))
        y = np.array(y) # gene * ind
        y = y.transpose() # ind * gene
        np.savetxt(output.y, y)


rule cuomo_pca:
    input:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',
        imputed_ct_y = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch0/ct.y.txt', # donor - day * gene
    output:
        evec = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/evec.txt',
        eval = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/eval.txt',
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.png',
    script: 'bin/cuomo/pca.py'


rule cuomo_op_test:
    input:
        y_batch = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt',
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt',
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt',
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = 'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment 
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', # donor - day * gene
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/op.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/op.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
    resources:
        time = '48:00:00',
        mem_mb = '8gb',
    priority: -1
    script: 'bin/cuomo/op.py'


use rule op_aggReplications as cuomo_op_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/op.txt' 
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',


rule cuomo_ctp_test:
    input:
        y_batch = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctp.txt',
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt', #donor-day * gene 
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt',
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = 'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,  
        REML = True,
        HE = True, 
        jack_knife = True,
        IID = False,
        Hom = False,
        optim_by_R = True, 
    resources: 
        mem_mb = '10gb',
        time = '48:00:00',
    script: 'bin/cuomo/ctp.py'


use rule op_aggReplications as cuomo_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',


use rule cuomo_ctp_test as cuomo_ctp_test2 with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp2.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp2.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,  
        REML = True,
        HE = True, 
        jack_knife = True,
        IID = True,
        Hom = True,
        optim_by_R = True, 


use rule op_aggReplications as cuomo_ctp_test2_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp2.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp2.npy',


use rule cuomo_ctp_test as cuomo_ctp_test_remlJK with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.remlJK.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.remlJK.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,  
        REML = True,
        Free_reml_jk = True,
        HE = False, 
        Hom = False,
        IID = False,
        optim_by_R = True, 
    resources: 
        mem_mb = '16gb',
        time = '48:00:00',


# TODO: tmp
use rule cuomo_ctp_test_remlJK as cuomo_ctp_test_remlJK_tmp with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.tmp.remlJK.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.tmp.remlJK.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,  
        REML = True,
        Free_reml_jk = True,
        HE = False, 
        Hom = False,
        IID = False,
        optim_by_R = True, 


use rule op_aggReplications as cuomo_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.remlJK.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',


# paper plot
rule cuomo_sc_expressionpattern_paper:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
        imputed_ct_y = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_batch_no)], # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/paper.ctp.png', 
        data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/paper.ctp.sourcedata.txt', 
    params:
        mycolors = mycolors,
        genes = ['ENSG00000204531_POU5F1', 'NDUFB4', 'ENSG00000185155_MIXL1', 'ENSG00000163508_EOMES'],
    script: 'bin/cuomo/sc_expressionpattern.paper.py'


rule cuomo_ctp_pvalue_paper:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
        eds = 'analysis/cuomo/eds.paper.txt',
    output:
        reml_p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.REMLpvalue.paper.png',
        reml_data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.REMLpvalue.paper.sourcedata.txt',
        he_p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.HEpvalue.paper.png',
        he_data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.HEpvalue.paper.sourcedata.txt',
        qq = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.qq.supp.png',
        matrix = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/matrix.cor.supp.png',
        matrix_data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/matrix.cor.data.txt',
    script: 'bin/cuomo/ctp.p.paper.py'


rule cuomo_ctp_freeNfull_Variance_paper:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
        ctp = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.paper.png',
        dataA = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.paper.sroucedataA.txt',
        dataB = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.paper.sroucedataB.txt',
    script: 'bin/cuomo/ctp_freeNfull_Variance.paper.py'


rule cuomo_all:
    input:
        ctp_p = expand('results/cuomo/{params}/ctp.REMLpvalue.paper.png',
                params=cuomo_paramspace.instance_patterns),
        ctp_v = expand('results/cuomo/{params}/ctp.freeNfull.Variance.paper.png',
                params=cuomo_paramspace.instance_patterns),
        gene = expand('results/cuomo/{params}/genes/paper.ctp.png', 
                params=cuomo_paramspace.instance_patterns),


rule cuomo_compare_cellno:
    input:
        cell10 = f'analysis/cuomo/ind_min_cellnum~100_ct_min_cellnum~10_im_genome~Y_im_mvn~N_sex~Y_PC~6_experiment~R_disease~Y/ctp.remlJK.npy',
        cell5 = f'analysis/cuomo/ind_min_cellnum~100_ct_min_cellnum~5_im_genome~Y_im_mvn~N_sex~Y_PC~6_experiment~R_disease~Y/ctp.remlJK.npy',
        cell20 = f'analysis/cuomo/ind_min_cellnum~100_ct_min_cellnum~20_im_genome~Y_im_mvn~N_sex~Y_PC~6_experiment~R_disease~Y/ctp.remlJK.npy',
    output:
        png = 'results/cuomo/cellno.png',
    script: 'bin/cuomo/compare_cellno.py' 



##########################################################################
#  correlation with gene features
##########################################################################
rule cuomo_eds_paper:
    input:
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz', 
        eds = 'data/Wang_Goldstein.tableS1.txt',
        gnomad = 'data/gnomad.v2.1.1.lof_metrics.by_gene.txt.gz',
    output:
        eds = 'analysis/cuomo/eds.paper.txt',
    run:
        counts = pd.read_table(input.counts, index_col=0)
        eds = pd.read_table(input.eds)

        print(counts.shape, eds.shape)
        print(len(np.unique(counts.index)))
        print(len(np.unique(eds['GeneSymbol'])))

        # drop dozens of duplicated genes
        eds = eds.drop_duplicates(subset=['GeneSymbol'], keep=False, ignore_index=True)

        eds = eds.loc[eds['GeneSymbol'].isin(counts.index.str.split('_').str.get(0))]

        # drop pLI in eds, instead using pLI from gnomad
        eds = eds.drop('pLI', axis=1)

        # read gene length from gnomad
        gnomad = pd.read_table(input.gnomad, usecols=['gene', 'gene_id', 'gene_length', 'pLI', 'oe_lof_upper'])
        gnomad = gnomad.rename(columns={'gene_id': 'GeneSymbol', 'oe_lof_upper': 'LOEUF'})
        # lose dozens of genes that are not exist in gnomad
        eds = eds.merge(gnomad[['GeneSymbol', 'gene_length', 'pLI', 'LOEUF']])

        print(eds.shape)

        eds.to_csv(output.eds, sep='\t', index=False)




###########################################################################################
# simulate Cuomo genes: a random gene's hom2, ct main variance, nu
###########################################################################################
wildcard_constraints: nu_noise='[\d\._]+' 
wildcard_constraints: V='[\d\._]+' 

cuomo_simulateGene_gene_no = 1000
cuomo_simulateGene_batch_no = 250
cuomo_simulateGene_batches = np.array_split(range(cuomo_simulateGene_gene_no), cuomo_simulateGene_batch_no)


nu_noises = ['1_0_0', '1_2_20', '1_2_10', '1_2_5', '1_2_3', '1_2_2']
V3 = ['0_0_0_0', '0.05_0.1_0.1_0.1', '0.1_0.1_0.1_0.1', '0.2_0.1_0.1_0.1', '0.5_0.1_0.1_0.1']


rule cuomo_collect_std_cty:
    input:
        data = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_batch_no)], # cty after std
    output:
        data = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ct.y.merge.txt',
    shell:
        'cat {input.data} > {output.data}'


use rule cuomo_collect_std_cty as cuomo_collect_std_ctnu with:
    input:
        data = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.nu.ctp.txt'
                for i in range(cuomo_batch_no)], #donor-day * gene 
    output:
        data = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ct.nu.ctp.merge.txt',


use rule cuomo_collect_std_cty as cuomo_collect_P with:
    input:
        data = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/P.txt'
                for i in range(cuomo_batch_no)], # list
    output:
        data = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/P.merge.txt',


use rule cuomo_collect_std_cty as cuomo_collect_n with:
    input:
        data = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/n.txt' 
                for i in range(cuomo_batch_no)], # list
    output:
        data = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/n.merge.txt',


rule cuomo_simulateGene_randompickgene:
    input:
        ctnu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ct.nu.ctp.merge.txt',
    output:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
    params: 
        gene_no = cuomo_simulateGene_gene_no,
        seed = 23456,
    script: 'bin/cuomo/simulateGene_randompickgene.py'


#########################################################
# simulate pseudobulk 
#########################################################

rule cuomo_simulateGene_hom:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
        ctnu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ct.nu.ctp.merge.txt',
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/P.merge.txt',
    output:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/genes.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/P.batch{{i}}.txt',
        ctnu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/ctnu.batch{{i}}.txt',
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/cty.batch{{i}}.txt',
    params:
        batch = lambda wildcards: cuomo_simulateGene_batches[int(wildcards.i)],
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/rep/cty.txt',
        seed = 2468,
    priority: 90
    script: 'bin/cuomo/simulateGene_hom.py'


rule cuomo_simulateGene_hom_addUncertainty:
    input:
        nu = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/ctnu.batch{i}.txt' 
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        nu = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctnu.batch{i}.npy'
                for i in range(cuomo_simulateGene_batch_no)],
    params:
        seed = 23763,
    priority: 89
    run:
        rng = np.random.default_rng(seed=params.seed)
        for f1, f2 in zip(input.nu, output.nu):
            output_nus = {}
            # output_nus = open(f2, 'w')
            i = 0
            for line in open(f1):
                nu_f = line.strip()
                # uncertain_nu_f = f'{nu_f}.{wildcards.nu_noise}.uncertain'
                nu = pd.read_table(nu_f)
                gene = nu.columns[-1]
                if 'donor' in nu.columns:
                    nu = nu.pivot(index='donor', columns='day', values=gene)
                elif 'ind' in nu.columns:
                    nu = nu.pivot(index='ind', columns='ct', values=gene)
                nu = np.array(nu)
                # add uncertainty
                prop = float(wildcards.nu_noise.split('_')[0])
                noise = np.zeros(len(nu.flatten()))
                if float(wildcards.nu_noise.split('_')[1]) != 0 and float(wildcards.nu_noise.split('_')[2]) != 0:
                    noise[rng.choice(len(noise), int(len(noise)*prop), replace=False)] = rng.beta( 
                            a=float(wildcards.nu_noise.split('_')[1]), b=float(wildcards.nu_noise.split('_')[2]), 
                            size=int(len(noise)*prop))
                noise = noise * rng.choice([-1,1], len(noise))
                nu_withnoise = nu * (1 + noise.reshape(nu.shape[0], nu.shape[1]))

                # np.savetxt(uncertain_nu_f, nu_withnoise)
                # output_nus.write(uncertain_nu_f+'\n')
                output_nus[i] = nu_withnoise
                i += 1
            # output_nus.close()
            np.save(f2, output_nus)


use rule ctp_test as cuomo_simulateGene_hom_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctp.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = True,
        REML = True,
        HE = True,
        optim_by_R = True,
    priority: 88


use rule op_aggReplications as cuomo_simulateGene_hom_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctp.batch{i}.npy'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctp.npy',
    priority: 100


use rule ctp_test as cuomo_simulateGene_hom_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctp.remlJK.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = False,
        REML = True,
        Free_reml_jk = True,
        HE = False,
        optim_by_R = True,
    resources:
        mem_per_cpu = '12gb',
        time = '48:00:00',
    priority: 87


use rule op_aggReplications as cuomo_simulateGene_hom_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctp.remlJK.batch{i}.npy'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctp.remlJK.npy',
    priority: 100


rule cuomo_simulateGene_Free_addV:
    input:
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/cty.batch{{i}}.txt',
    output:
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/cty.batch{{i}}.txt',
    params:
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/rep/cty.txt',
        seed = 987,
    resources: burden=20,
    priority: 86
    run:
        free_fs = open(output.cty, 'w')

        V = np.diag([float(x) for x in wildcards.V.split('_')])
        for line in open(input.cty):
            hom_f = line.strip()
            rep = re.findall('/rep[^/]+/', hom_f)[0]
            free_f = re.sub('/rep/', rep, params.cty)

            cty = np.loadtxt( hom_f )
            
            rep_no = np.array([ord(x) for x in rep]).sum()
            seed = rep_no + params.seed
            rng = np.random.default_rng(seed)
            gamma_b = rng.multivariate_normal(np.zeros(cty.shape[1]), V, size=cty.shape[0])
            cty = cty + gamma_b

            os.makedirs( os.path.dirname(free_f), exist_ok=True)
            np.savetxt(free_f, cty)
            free_fs.write(free_f + '\n')
        free_fs.close()


use rule ctp_test as cuomo_simulateGene_Free_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/{{nu_noise}}/ctp.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/{{nu_noise}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = True,
        REML = True,
        HE = True,
        optim_by_R = True,
    priority: 85


use rule op_aggReplications as cuomo_simulateGene_Free_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/{{nu_noise}}/ctp.batch{i}.npy'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/{{nu_noise}}/ctp.npy',
    priority: 100


use rule ctp_test as cuomo_simulateGene_Free_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{{nu_noise}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/{{nu_noise}}/ctp.remlJK.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/{{nu_noise}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = False,
        REML = True,
        Free_reml_jk = True,
        HE = False,
        optim_by_R = True,
    resources:
        mem_per_cpu = '12gb',
        time = '48:00:00',
    priority: 84


use rule op_aggReplications as cuomo_simulateGene_Free_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/{{nu_noise}}/ctp.remlJK.batch{i}.npy'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{{V}}/{{nu_noise}}/ctp.remlJK.npy',
    priority: 100


rule cuomo_simulateGene_ctp_test_powerplot_paper:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{nu_noise}/ctp.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{nu_noise}/ctp.remlJK.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{V}/1_2_5/ctp.npy'
                for V in V3],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{V}/1_2_5/ctp.remlJK.npy'
                for V in V3],
        real = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png1 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.pseudo.power.paper.supp.png',
        png2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.pseudo.power.paper.png',
    params:
        nu_noises = nu_noises,
        V3 = V3,
    script: 'bin/cuomo/simulateGene_ctp_test_powerplot_paper.py'


rule cuomo_simulateGene_pseudo_all:
    input:
        png2 = expand('results/cuomo/{params}/simulateGene/ctp.pseudo.power.paper.png',
                params=cuomo_paramspace.instance_patterns)


#######################################################################
# FDR 
#######################################################################
rule cuomo_simulateGene_FDR:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{nu_noise}/ctp.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{nu_noise}/ctp.remlJK.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_0.1_0.1_0.1_0.1/{nu_noise}/ctp.npy'
                for nu_noise in nu_noises],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_0.1_0.1_0.1_0.1/{nu_noise}/ctp.remlJK.npy'
                for nu_noise in nu_noises],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.pseudo.fdr.paper.supp.png',
        png2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.pseudo.fdr2.paper.supp.png',
    params:
        nu_noises = nu_noises,
    script: 'bin/cuomo/simulateGene_fdrplot_paper.py'
    






###########################################################################################
# simulate Cuomo genes: single cell simulation
###########################################################################################
wildcard_constraints: cell_no='[\d\.]+' 
wildcard_constraints: depth='[\d\.]+' 
wildcard_constraints: option='\d+' 

cuomo_sc_batch_no = 20
cuomo_sc_batches = np.array_split(range(cuomo_simulateGene_gene_no), cuomo_sc_batch_no)


cell_nos = [0.5, 1, 2]
depths = [0.01, 0.1, 1, 2]
V4 = [0, 0.2, 0.4, 0.6, 0.8, 1]  # proportion of shuffled individual to simulate cell type specific variance 

rule cuomo_simulateGene_sc_bootstrap_hom:
    input:
        counts = 'data/cuomo2020natcommun/raw_counts.csv.gz',
        meta = 'analysis/cuomo/data/meta.txt',
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
    output:
        genes = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        P = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        ctnu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        cty = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/cty.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.gz',
        sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.gz',
    params:
        batch = cuomo_sc_batches,
        cty = lambda wildcards, output: os.path.dirname(output.cty[0]),
        seed = 145342,
        resample_inds = False,
    resources:
        mem_mb = '15gb',
    priority: 100
    script: 'bin/cuomo/simulateGene_sc_bootstrap.py'


use rule cuomo_simulateGene_hom_addUncertainty as cuomo_simulateGene_sc_bootstrap_hom_addUncertainty with:
    input:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    output:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctnu.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    params:
        seed = 237632,
    priority: 99


use rule ctp_test as cuomo_simulateGene_sc_bootstrap_hom_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = False,
        REML = False,
        HE = True,
        HE_free_only = True,
        optim_by_R = True,
    priority: 98


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_hom_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.npy',
    priority: 100


use rule ctp_test_remlJK as cuomo_simulateGene_sc_bootstrap_hom_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        optim_by_R = True,
    resources:
        mem_mb = '12gb',
        time = '48:00:00',
    priority: 97


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_hom_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.remlJK.npy',
    priority: 100


use rule cuomo_simulateGene_sc_bootstrap_hom as cuomo_simulateGene_sc_bootstrap_free with:
    output:
        genes = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/genes.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        P = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/P.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        ctnu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        cty = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/cty.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    params:
        batch = cuomo_sc_batches,
        cty = lambda wildcards, output: os.path.dirname(output.cty[0]), 
        seed = 143352,
        resample_inds = False,
    resources:
        mem_mb = '30gb',
        partition = 'tier2q',
    priority: 100


use rule cuomo_simulateGene_hom_addUncertainty as cuomo_simulateGene_sc_bootstrap_free_addUncertainty with:
    input:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    output:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctnu.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    params:
        seed = 211313,
    priority: 99


use rule ctp_test as cuomo_simulateGene_sc_bootstrap_Free_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = False,
        REML = False,
        HE = True,
        HE_free_only = True,
        optim_by_R = True,
    priority: 98


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_Free_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.npy',
    priority: 100


use rule ctp_test_remlJK as cuomo_simulateGene_sc_bootstrap_Free_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        optim_by_R = True,
        method = 'BFGS-Nelder',
    resources:
        mem_mb = '12gb',
        time = '68:00:00',
    priority: 97


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_Free_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.remlJK.npy',
    priority: 100



rule cuomo_simulateGene_sc_bootstrap_ctp_test_powerplot_merge_paper:
    input:
        cellno_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{cell_no}/1/{nu_noise}/{{option}}/ctp.npy'
                for cell_no in cell_nos for nu_noise in nu_noises],
        depth_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/1/{depth}/{nu_noise}/{{option}}/ctp.npy'
                for depth in depths for nu_noise in nu_noises],
        remlJK_cellno_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{cell_no}/1/{nu_noise}/{{option}}/ctp.remlJK.npy'
                for cell_no in cell_nos for nu_noise in nu_noises],
        remlJK_depth_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/1/{depth}/{nu_noise}/{{option}}/ctp.remlJK.npy'
                for depth in depths for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        cellno_outs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{cell_no}/1/V_{V}/1_2_5/{{option}}/ctp.npy'
                for cell_no in cell_nos for V in V4],
        depth_outs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/1/{depth}/V_{V}/1_2_5/{{option}}/ctp.npy'
                for depth in depths for V in V4],
        cellno_remlJKs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{cell_no}/1/V_{V}/1_2_5/{{option}}/ctp.remlJK.npy'
                for cell_no in cell_nos for V in V4],
        depth_remlJKs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/1/{depth}/V_{V}/1_2_5/{{option}}/ctp.remlJK.npy'
                for depth in depths for V in V4],
        real = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/ctp.sc.power.{{option}}.paper.png',
        data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/ctp.sc.power.{{option}}.paper.txt',
    params:
        nu_noises = nu_noises,
        Vs = V4,
        cell_nos = cell_nos,
        depths = depths,
    script: 'bin/cuomo/simulateGene_sc_bootstrap_ctp_test_powerplot_paper.py'


rule cuomo_simulateGene_all:
    input:
        pseudo = expand('results/cuomo/{params}/simulateGene/ctp.pseudo.power.paper.png',
                params=cuomo_paramspace.instance_patterns),
        boot = expand('results/cuomo/{params}/simulateGene/bootstrap/ctp.sc.power.2.paper.png', 
                    params=cuomo_paramspace.instance_patterns),




#########################################################################################
# Yazar et al 2022 Science
#########################################################################################
yazar_ind_col = 'individual'
yazar_ct_col = 'cell_label'
yazar_cell_col = 'cell'
yazar_batch_no = 2000

# read parameters
yazar_params = pd.read_table('yazar.params.txt', dtype="str", comment='#')
yazar_paramspace = Paramspace(yazar_params, filename_params="*")

# yazar preprocessing
rule yazar_extract_meta:
    input:
        h5ad = 'data/Yazar2022Science/OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad.gz',
    output:
        obs = 'data/Yazar2022Science/obs.txt', # cells
        var = 'data/Yazar2022Science/var.txt', # genes
    run:
        import scanpy as sc

        data = sc.read_h5ad(input.h5ad, backed='r')
        obs = data.obs.reset_index(drop=False, names='cell')
        obs.to_csv(output.obs, sep='\t', index=False)

        var = data.var.reset_index(drop=False, names='feature')
        var.to_csv(output.var, sep='\t', index=False)


rule yazar_exclude_repeatedpool:
    input:
        obs = 'data/Yazar2022Science/obs.txt',
    output:
        obs = 'analysis/yazar/exclude_repeatedpool.obs.txt',
        dup_inds = 'analysis/yazar/duplicated_inds.txt',
    params:
        ind_col = yazar_ind_col,
    run:
        obs = pd.read_table(input.obs)
        # id repeated pool: the same individual seqed in more than one pool
        data = obs[['pool',params.ind_col]].drop_duplicates()
        inds, counts = np.unique(data[params.ind_col], return_counts=True)
        inds = inds[counts > 1]
        np.savetxt(output.dup_inds, inds, fmt='%s')

        # for each ind find the largest pool
        for ind in inds:
            pools, counts = np.unique(obs.loc[obs[params.ind_col]==ind,'pool'], return_counts=True)
            excluded_pools = pools[counts < np.amax(counts)]
            obs = obs.loc[~((obs[params.ind_col]==ind) & (obs['pool'].isin(excluded_pools)))]

        obs.to_csv(output.obs, sep='\t', index=False)


rule yazar_ctp_extractX:
    input:
        h5ad = 'data/Yazar2022Science/OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad.gz',
        var = 'data/Yazar2022Science/var.txt',
        obs = 'analysis/yazar/exclude_repeatedpool.obs.txt',
    output:
        X = 'staging/data/yazar/X.npz',
        obs = 'staging/data/yazar/obs.gz',
        var = 'staging/data/yazar/var.gz',
    params:
        ind_col = yazar_ind_col,
        ct_col = yazar_ct_col,
    resources:
        mem_mb = '40G',
    run:
        import scanpy as sc
        from scipy import sparse
        from ctmm import preprocess

        var = pd.read_table(input.var, index_col=0)
        if 'feature_is_filtered' in var.columns:
            genes = var.loc[~var['feature_is_filtered']].index.to_numpy()
        else:
            genes = var.index.to_numpy()

        if 'subset_gene' in params.keys():
            # random select genes
            rng = np.random.default_rng(seed=params.seed)
            genes = rng.choice(genes, params.subset_gene, replace=False)

        obs = pd.read_table(input.obs, index_col=0)
        ind_pool = np.unique(obs[params.ind_col].astype('str')+'+'+obs['pool'].astype('str'))

        ann = sc.read_h5ad(input.h5ad, backed='r')
        data = ann[(~ann.obs[params.ind_col].isna())
                & (~ann.obs[params.ct_col].isna())
                & (ann.obs[params.ind_col].astype('str')+'+'+ann.obs['pool'].astype('str')).isin(ind_pool), genes]
        # normalize and natural logarithm of one plus the input array
        X = preprocess.normalize(data.X, 1e4).log1p()
        sparse.save_npz(output.X, X)

        data.obs.rename_axis('cell').to_csv(output.obs, sep='\t')
        data.var.rename_axis('feature').to_csv(output.var, sep='\t')


rule yazar_ctp:
    input:
        X = 'staging/data/yazar/X.npz',
        obs = 'staging/data/yazar/obs.gz',
        var = 'staging/data/yazar/var.gz',
    output:
        ctp = 'data/Yazar2022Science/ctp.gz',
        ctnu = 'data/Yazar2022Science/ctnu.gz',
        P = 'data/Yazar2022Science/P.gz',
        n = 'data/Yazar2022Science/n.gz',
    params:
        ind_col = yazar_ind_col,
        ct_col = yazar_ct_col,
    resources:
        mem_mb = '40G',
    run:
        from scipy import sparse
        from ctmm import preprocess

        X = sparse.load_npz(input.X)
        obs = pd.read_table(input.obs, index_col=0)
        var = pd.read_table(input.var, index_col=0)
        ctp, ctnu, P, n = preprocess.pseudobulk(X=X, obs=obs, var=var, ind_cut=100, ct_cut=10,
                ind_col=params.ind_col, ct_col=params.ct_col)

        # save
        ctp.to_csv(output.ctp, sep='\t')
        ctnu.to_csv(output.ctnu, sep='\t')
        P.to_csv(output.P, sep='\t')
        n.to_csv(output.n, sep='\t')


rule yazar_P_plot:
    input:
        P = 'data/Yazar2022Science/P.gz',
    output:
        png = 'results/yazar/P.png',
    run:
        P = pd.read_table(input.P, index_col=0)
        P = P.drop(['Erythrocytes', 'Platelets'], axis=1)
        P = P.div(P.sum(axis=1), axis=0)
        P = P[P.mean().sort_values(ascending=False).index]
        P.columns = P.columns.str.replace(' ', '_')

        plt.rcParams.update({'font.size' : 6})
        fig, ax = plt.subplots(figsize=(8,4), dpi=600)
        sns.violinplot(data=P, scale='width', cut=0)
        ax.axhline(y=0, color='0.9', ls='--', zorder=0)
        ax.set_xlabel('Cell type', fontsize=10)
        ax.set_ylabel('Cell type proportion', fontsize=10)
        plt.tight_layout()
        fig.savefig(output.png)


# CTMM
rule yazar_rm_rareINDnCT_filterGenes:
    input:
        ctp = 'data/Yazar2022Science/ctp.gz',
        ctnu = 'data/Yazar2022Science/ctnu.gz',
        n = 'data/Yazar2022Science/n.gz',
    output:
        ctp = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctp.gz',
        ctnu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.gz',
        P = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/P.gz',
        n = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/n.gz',
    resources:
        mem_mb = '10G',
    script: 'bin/yazar/rm_rareINDnCT.py'


rule yazar_rmIND:
    input:
        ctp = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctp.gz',
        ctnu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.gz',
        P = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/P.gz',
        n = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/n.gz',
    output:
        ctp = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.gz',
        ctnu = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctnu.gz',
        P = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/P.gz',
        n = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/n.gz',
    run:
        ctp = pd.read_table(input.ctp, index_col=(0, 1)).sort_index()
        ctnu = pd.read_table(input.ctnu, index_col=(0, 1)).sort_index()
        P = pd.read_table(input.P, index_col=0)
        n = pd.read_table(input.n, index_col=0)

        # select ids
        ids = n.index.to_numpy()[~(n <= int(wildcards.ct_min_cellnum)).any(axis='columns')]

        # 
        ctp.loc[ctp.index.get_level_values('ind').isin(ids)].to_csv(output.ctp, sep='\t')
        ctnu.loc[ctnu.index.get_level_values('ind').isin(ids)].to_csv(output.ctnu, sep='\t')
        P.loc[P.index.isin(ids)].to_csv(output.P, sep='\t')
        n.loc[n.index.isin(ids)].to_csv(output.n, sep='\t')


rule yazar_std_op:
    input:
        ctp = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.gz',
        ctnu = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctnu.gz',
        P = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/P.gz',
    output:
        op = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/op.std.gz',
        nu = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/nu.std.gz',
        ctp = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
        ctnu = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctnu.std.gz',
    resources:
        mem_mb = '10G',
    run:
        from ctmm import preprocess
        ctp = pd.read_table(input.ctp, index_col=(0,1)).astype('float32')
        ctnu = pd.read_table(input.ctnu, index_col=(0,1)).astype('float32')
        P = pd.read_table(input.P, index_col=0)

        op, nu, ctp, ctnu = preprocess.std(ctp, ctnu, P)

        op.to_csv(output.op, sep='\t')
        nu.to_csv(output.nu, sep='\t')
        ctp.to_csv(output.ctp, sep='\t')
        ctnu.to_csv(output.ctnu, sep='\t')
        

rule yazar_op_pca:
    input:
        op = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/op.std.gz',
    output:
        evec = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/evec.txt',
        eval = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/eval.txt',
        pca = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/pca.txt',
        png = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/pca.png',
    resources:
        mem_mb = '4G',
    script: 'bin/yazar/pca.py'


use rule cuomo_split2batches as yazar_split2batches with:
    input:
        y = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
    output:
        y_batch = [f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/y/batch{i}.txt' 
                for i in range(yazar_batch_no)],


rule yazar_split_ctp:
    input:
        ctp = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
        ctnu = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctnu.std.gz',
        y_batch = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt',
    output:
        ctp = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.std.gz', 
        ctnu = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.std.gz', 
    resources:
        burden = 20,
    run:
        ctp = pd.read_table(input.ctp, index_col=(0,1))
        ctnu = pd.read_table(input.ctnu, index_col=(0,1))
        genes = np.loadtxt(input.y_batch, dtype='str')
        ctp[genes].to_csv(output.ctp, sep='\t')
        ctnu[genes].to_csv(output.ctnu, sep='\t')


rule yazar_ctp_HE_free:
    input:
        y_batch = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        ctp = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.std.gz', 
        ctnu = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.std.gz', 
        P = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/P.gz',
        pca = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/pca.txt',
        meta = 'staging/data/yazar/obs.gz',
    output:
        out = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'he',
        model = 'free',
        he_jk = True,
        he_free_ols = True,
    resources:
        mem_mb = '30gb', 
        time = '100:00:00',
    script: 'bin/yazar/ctp.py'


rule yazar_ctp_HE_free_tmp:
    input:
        y_batch = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        ctp = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.std.gz', 
        ctnu = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.std.gz', 
        P = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/P.gz',
        pca = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/pca.txt',
        meta = 'staging/data/yazar/obs.gz',
    output:
        out = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.tmp.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'he',
        model = 'free',
        he_jk = True,
        he_free_ols = True,
    resources:
        mem_mb = '30gb', 
        time = '100:00:00',
    script: 'scripts/yazar/ctp.tmp.py'


use rule yazar_ctp_HE_free as yazar_ctp_HE_full with:
    output:
        out = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'he',
        model = 'full',
    resources:
        mem_mb = '60gb',


rule yazar_ctp_HE:
    input:
        free = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
        full = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    output:
        out = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.npy',
    run:
        free = np.load(input.free, allow_pickle=True)
        full = np.load(input.full, allow_pickle=True)
        for i in range(len(free)):
            free[i]['he']['full'] = full[i]['he']['full']
        np.save(output.out, free)


use rule op_aggReplications as yazar_ctp_HE_aggReplications with:
    input:
        out = [f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.npy'
                for i in range(yazar_batch_no)],
    output:
        out = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    

rule yazar_ctp_freeNfull_Variance_paper:
    input:
        P = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/P.gz',
        ctp = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        free = f'results/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.free.Variance.paper.png',
        full = f'results/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.full.Variance.paper.png',
    script: 'bin/yazar/ctp_freeNfull_Variance.paper.py'


rule yazar_ctp_pvalue_paper:
    input:
        out = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        png = f'results/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.he.pvalue.png',
    script: 'bin/yazar/ctp.p.paper.py'


rule yazar_all:
    input:
        ctp = expand('results/yazar/nomissing/{params}/ctp.free.Variance.paper.png', 
                params=yazar_paramspace.instance_patterns),
        p = expand('results/yazar/nomissing/{params}/ctp.he.pvalue.png',
                params=yazar_paramspace.instance_patterns),







###########
# OTHERS
###########
if os.path.exists('CTMM.smk'):
    include: 'CTMM.smk'
