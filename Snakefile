from snakemake.utils import Paramspace
import re, sys, os, gzip, math, time, scipy, tempfile, copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# make a logs folders to save log files
os.makedirs('./logs/', exist_ok=True)

mycolors = sns.color_palette()
pointcolor = 'red' # color for expected values  in estimates plots

def get_subspace(arg, model_params):
    ''' model_params df include or not include the column of model, but the first row is the basemodel'''
    sim_args = list(model_params.columns)
    if 'model' in sim_args:
        sim_args.remove('model')
    model_params = model_params[sim_args].reset_index(drop=True)
    #print(model_params)
    basemodel = model_params.iloc[0].to_dict()
    sim_args.remove(arg)
    evaluate = ' & '.join([f"{_} == '{basemodel[_]}'" for _ in sim_args])
    subspace = model_params[model_params.eval(evaluate)]
    return Paramspace(subspace, filename_params="*")

def get_effective_args(model_params):
    ''' model_params df include or not include the column of model, but the first row is the basemodel'''
    #print(model_params)
    sim_args = list(model_params.columns)
    if 'model' in sim_args:
        sim_args.remove('model')
    model_params = model_params[sim_args]
    effective_args = [] # args with multiple parameters. e.g. ss has '1e2', '2e2', '5e2'
    for arg_ in np.array(model_params.columns):
        subspace = get_subspace(arg_, model_params)
        if subspace.shape[0] > 1:
            effective_args.append(arg_)
    return effective_args

# wildcard constraints
wildcard_constraints: i='[0-9]+' 
wildcard_constraints: m='[0-9]+' 
wildcard_constraints: model='\w+'
wildcard_constraints: cut_off = '[\d\.]+'
wildcard_constraints: prop = '[\d\.]+'

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
par_columns = list(op_params.columns)
par_columns.remove('model') # columns after removing 'model'
op_paramspace = Paramspace(op_params[par_columns], filename_params="*")

op_plot_order = {
        'hom':{
            'ss':['2e1', '5e1', '1e2', '2e2', '3e2', '5e2', '1e3'], 
            'a':['0.5_2_2_2', '1_2_2_2', '2_2_2_2', '4_2_2_2']
            },
        'iid':{
            'ss':['2e1', '5e1', '1e2', '3e2', '5e2', '1e3'], 
            'a':['0.5_2_2_2', '1_2_2_2', '2_2_2_2', '4_2_2_2'],
            'vc':['0.25_0.40_0.10_0.25', '0.25_0.30_0.20_0.25', '0.25_0.25_0.25_0.25', 
                '0.25_0.20_0.30_0.25', '0.25_0.10_0.40_0.25']
            }, 
        'free': {
            'ss':['2e1', '5e1', '1e2', '2e2', '3e2', '5e2', '1e3'], 
            'a':['0.32_2_2_2','0.5_2_2_2', '0.66_2_2_2', '0.77_2_2_2', 
                '0.88_2_2_2', '1_2_2_2', '1.05_2_2_2', '2_2_2_2', '4_2_2_2'], 
            'vc':['0.25_0.40_0.10_0.25', '0.25_0.30_0.20_0.25', '0.25_0.25_0.25_0.25', 
                '0.25_0.20_0.30_0.25', '0.25_0.10_0.40_0.25'],
            'V_diag':['1_1_1_1', '8_4_2_1', '27_9_3_1', '64_16_4_1']
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

op_excluderareCT_plot_order = copy.deepcopy(op_plot_order)
for model in op_excluderareCT_plot_order.keys():
    op_excluderareCT_plot_order[model]['a'].remove('0.5_2_2_2')

rule op_parameters:
    output:
        pi = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/PI.txt',
        s = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/S.txt',
        beta = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/V.txt',
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
        mem_per_cpu = '5000',
    script: 'bin/sim/op_R.py'

rule op_aggReplications:
    input:
        out = [f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(op_batches))],
    output:
        out = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.npy',
    script: 'bin/mergeBatches.py'

rule op_test_remlJK:
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
        mem_per_cpu = '10gb',
        time = '18:00:00',
    script: 'bin/sim/op_R.py'

use rule op_aggReplications as op_remlJK_aggReplications with:
    input:
        out = [f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.remlJK.batch{i}' 
                for i in range(len(op_batches))],
    output:
        out = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.remlJK.npy',

#########################################################################################
# CTP
#########################################################################################
# par
ctp_replicates = 1000
ctp_batch_no = 100
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
        mem_per_cpu = '5gb',
        time = '48:00:00',
    priority: 1
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
        ML = False,
        REML = True,
        Free_reml_only = True,
        Free_reml_jk = True,
        HE = False,
        optim_by_R = True,
    resources:
        mem_per_cpu = '10gb',
        time = '200:00:00',
    priority: 1
    script: 'bin/sim/ctp.py'

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

rule cuomo_pseudobulk:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        y = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
    resources: 
        mem_per_cpu = '10gb',
        time = '24:00:00',
    script: 'bin/cuomo/pseudobulk.py'

rule cuomo_varNU:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'staging/cuomo/bootstrapedNU/data/counts{i}.txt.gz',
    output:
        var_nu = 'staging/cuomo/bootstrapedNU/data/counts{i}.var_nu.gz',
    resources: 
        mem_per_cpu = '10gb',
        time = '48:00:00',
    shell: 
        '''
        module load python/3.8.1
        python3 bin/cuomo/varNU.py {input.meta} {input.counts} {output.var_nu}
        '''

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

cuomo_batch_no = 1000
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
        seed = 123450, # seed for softimpute
    resources: 
        mem = '20gb',
        time = '20:00:00',
    script: 'bin/cuomo/imputeGenome.py'

#rule impute_tst:
#    input:
#        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz',
#    output:
#        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.tmp.gz',
#    params:
#        seed = 123450,
#    resources:
#        mem = '20gb',
#        time = '20:00:00',
#    script: 'bin/impute_tst.py'

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
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
        nu_ctp = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctp.txt', # list
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_cty = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        imputed_ctnu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        imputed_ctnu_ctp = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
    resources: mem = '10gb',
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
        mem = '8gb',
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
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctp.txt', # list
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt', #donor-day * gene 
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
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
        mem = '10gb',
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
        mem = '16gb',
        time = '48:00:00',

use rule op_aggReplications as cuomo_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.remlJK.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',

# paper plot
rule cuomo_sc_expressionpattern_paper:
    # single cell expression pattern plot
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
        imputed_ct_y = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_batch_no)], # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/paper.ctp.png', 
    params:
        mycolors = mycolors,
        genes = ['ENSG00000204531_POU5F1', 'NDUFB4', 'ENSG00000185155_MIXL1', 'ENSG00000163508_EOMES'],
    script: 'bin/cuomo/sc_expressionpattern.paper.py'

rule cuomo_ctp_pvalue_paper:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
    output:
        reml_p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.REMLpvalue.paper.png',
        he_p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.HEpvalue.paper.png',
        qq = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.qq.supp.png',
    script: 'bin/cuomo/ctp.p.paper.py'

rule cuomo_ctp_freeNfull_Variance_paper:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
        ctp = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.paper.png',
    script: 'bin/cuomo/ctp_freeNfull_Variance.paper.py'

rule cuomo_all:
    input:
        ctp_p = expand('results/cuomo/{params}/ctp.REMLpvalue.paper.png',
                params=cuomo_paramspace.instance_patterns),
        ctp_v = expand('results/cuomo/{params}/ctp.freeNfull.Variance.paper.png',
                params=cuomo_paramspace.instance_patterns),
###########################################################################################
# simulate Cuomo genes: a random gene's hom2, ct main variance, nu
###########################################################################################
cuomo_simulateGene_gene_no = 1000
cuomo_simulateGene_batch_no = 100
cuomo_simulateGene_batches = np.array_split(range(cuomo_simulateGene_gene_no), cuomo_simulateGene_batch_no)
nu_noises = ['1_0_0', '1_2_20', '1_2_10', '1_2_5', '1_2_3', '1_2_2']

rule cuomo_simulateGene_randompickgene:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.nu.ctp.txt'
                for i in range(cuomo_batch_no)], #donor-day * gene 
    output:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
    params: gene_no = cuomo_simulateGene_gene_no,
    script: 'bin/cuomo/simulateGene_randompickgene.py'

rule cuomo_simulateGene_hom:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
        imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.nu.ctp.txt'
                for i in range(cuomo_batch_no)], #donor-day * gene 
        P = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/P.txt'
                for i in range(cuomo_batch_no)], # list
    output:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        ctnu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/ctnu.batch{{i}}.txt',
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/cty.batch{{i}}.txt',
    params:
        batch = lambda wildcards: cuomo_simulateGene_batches[int(wildcards.i)],
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/rep/cty.txt' 
    script: 'bin/cuomo/simulateGene_hom.py'

rule cuomo_simulateGene_hom_addUncertainty:
    input:
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/ctnu.batch{{i}}.txt',
    output:
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    run:
        output_nus = open(output.nu, 'w')
        rng = np.random.default_rng()
        for line in open(input.nu):
            nu_f = line.strip()
            uncertain_nu_f = f'{nu_f}.{wildcards.nu_noise}.uncertain'
            nu = pd.read_table(nu_f)
            gene = nu.columns[-1]
            nu = nu.pivot(index='donor', columns='day', values=gene)
            nu = np.array( nu )
            # add uncertainty
            prop = float(wildcards.nu_noise.split('_')[0])
            noise = np.zeros(len(nu.flatten()))
            if float(wildcards.nu_noise.split('_')[1]) != 0 and float(wildcards.nu_noise.split('_')[2]) != 0:
                noise[rng.choice(len(noise), int(len(noise)*prop), replace=False)] = rng.beta( 
                        a=float(wildcards.nu_noise.split('_')[1]), b=float(wildcards.nu_noise.split('_')[2]), 
                        size=int(len(noise)*prop) )
            noise = noise * rng.choice([-1,1], len(noise))
            nu_withnoise = nu * (1 + noise.reshape(nu.shape[0], nu.shape[1]))

            np.savetxt(uncertain_nu_f, nu_withnoise)
            output_nus.write(uncertain_nu_f+'\n')
        output_nus.close()

use rule ctp_test as cuomo_simulateGene_hom_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = True,
        REML = True,
        HE = True,
        optim_by_R = True,

use rule op_aggReplications as cuomo_simulateGene_hom_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.npy',

use rule ctp_test as cuomo_simulateGene_hom_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.remlJK.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = False,
        REML = True,
        Free_reml_jk = True,
        HE = False,
        optim_by_R = True,
    resources:
        mem_per_cpu = '12gb',
        time = '48:00:00',

use rule op_aggReplications as cuomo_simulateGene_hom_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.remlJK.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.remlJK.npy',

rule cuomo_simulateGene_Free_addV:
    input:
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/cty.batch{{i}}.txt',
    output:
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/cty.batch{{i}}.txt',
    params:
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/rep/cty.txt' 
    run:
        free_fs = open(output.cty, 'w')
        for line in open(input.cty):
            hom_f = line.strip()
            rep = re.findall('/rep[^/]+/', hom_f)[0]
            free_f = re.sub('/rep/', rep, params.cty)

            cty = np.loadtxt( hom_f )
            
            rng = np.random.default_rng()
            V = np.diag([float(x) for x in wildcards.V.split('_')])
            gamma_b = rng.multivariate_normal(np.zeros(cty.shape[1]), V, size=cty.shape[0])
            cty = cty + gamma_b

            os.makedirs( os.path.dirname(free_f), exist_ok=True)
            np.savetxt(free_f, cty)
            free_fs.write(free_f + '\n')
        free_fs.close()

use rule ctp_test as cuomo_simulateGene_Free_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = True,
        REML = True,
        HE = True,
        optim_by_R = True,

use rule op_aggReplications as cuomo_simulateGene_Free_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.npy',

use rule ctp_test as cuomo_simulateGene_Free_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.remlJK.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = False,
        REML = True,
        Free_reml_jk = True,
        HE = False,
        optim_by_R = True,
    resources:
        mem_per_cpu = '12gb',
        time = '48:00:00',

use rule op_aggReplications as cuomo_simulateGene_Free_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.remlJK.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.remlJK.npy',

V1 = ['0_0_0_0', '0.05_0_0_0','0.1_0_0_0', '0.2_0_0_0', '0.5_0_0_0']
V2 = ['0.05_0.05_0.05_0.05', '0.1_0.1_0.1_0.1', '0.2_0.2_0.2_0.2', '0.5_0.5_0.5_0.5']
V3 = ['0_0_0_0', '0.05_0.1_0.1_0.1', '0.1_0.1_0.1_0.1', '0.2_0.1_0.1_0.1', '0.5_0.1_0.1_0.1']
rule cuomo_simulateGene_ctp_test_powerplot_paper:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.remlJK.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.npy'
                for V in V3],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.npy'
                for V in V3],
    output:
        png1 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.power.paper.supp.png',
        png2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.power.paper.png',
    params:
        nu_noises = nu_noises,
        V3 = V3,
    script: 'bin/cuomo/simulateGene_ctp_test_powerplot_paper.py'

###########
# OTHERS
###########
if os.path.exists('CTMM.snake'):
    include: 'CTMM.snake'
