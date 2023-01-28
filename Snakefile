from snakemake.utils import Paramspace
import re, sys, os, gzip, math, time, scipy, tempfile, copy
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mystats, plot_help
sys.path.insert(0, 'bin')
import screml, wald


colorpalette='muted'
mycolors = plot_help.mycolors(n=10, palette=colorpalette)
pointcolor = 'red' # color for expected values  in estimates plots
def generate_tmpfn():
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    tmpfn = tmpf.name
    tmpf.close()
    print(tmpfn)
    return tmpfn

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
    resources:
        mem_per_cpu = '5000',
    script: 'bin/op_R.py'

rule op_aggReplications:
    input:
        out = [f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(op_batches))],
    output:
        out = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.npy',
    script: 'bin/mergeBatches.py'

rule op_test_scipy:
    input:
        y = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
        P = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.scipy.batch{{i}}',
    params:
        out = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/rep/out.scipy.npy',
        batch = lambda wildcards: op_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
    script: 'bin/op.py'

use rule op_aggReplications as op_scipy_aggReplications with:
    input:
        out = [f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.scipy.batch{i}' 
                for i in range(len(op_batches))],
    output:
        out = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.scipy.npy',

rule op_compare_optim_RvsPython:
    input:
        r = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.npy',
        p = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.scipy.npy',
    output:
        png = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/optim.RvsPython.png',
    script: 'bin/compare_optim_RvsPython.py'

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
    script: 'bin/op_R.py'

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
        optim_by_r = True,
    resources:
        mem_per_cpu = '5gb',
        time = '48:00:00',
    priority: 1
    script: 'bin/ctp.py'
    #script: 'bin/ctp_R.py'

use rule op_aggReplications as ctp_aggReplications with:
    input:
        out = [f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(ctp_batches))],
    output:
        out = f'analysis/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.npy',

rule ctp_test_scipy:
    input:
        y = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        P = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.scipy.batch{{i}}',
    params:
        out = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/rep/out.scipy.npy',
        batch = lambda wildcards: ctp_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        optim_by_r = False,
    resources:
        mem_per_cpu = '5gb',
        time = '48:00:00',
    priority: 1
    script: 'bin/ctp.py'

use rule op_aggReplications as ctp_scipy_aggReplications with:
    input:
        out = [f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.scipy.batch{i}' 
                for i in range(len(ctp_batches))],
    output:
        out = f'analysis/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.scipy.npy',

use rule op_compare_optim_RvsPython as ctp_compare_optim_RvsPython with:
    input:
        r = f'analysis/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.npy',
        p = f'analysis/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.scipy.npy',
    output:
        png = f'analysis/ctp/{{model}}/{op_paramspace.wildcard_pattern}/optim.RvsPython.png',

rule ctp_test_remlJK:
    input:
        cty = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        P = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        ctnu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
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
    resources:
        mem_per_cpu = '10gb',
        time = '200:00:00',
    priority: 1
    script: 'bin/ctp_test.py'

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
    script: 'bin/paper_opNctp_power.py'

rule paper_ctp_power:
    input:
        hom_ss = expand('analysis/ctp/hom/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom']).instance_patterns),
        hom_ss_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[(op_params['model']=='hom') & (op_params['ss'].astype('float')<=100)]).instance_patterns),
        free_ss = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
        free_ss_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[(op_params['model']=='free') & (op_params['ss'].astype('float')<=100)]).instance_patterns),
        hom_a = expand('analysis/ctp/hom/{params}/out.npy',
                params=get_subspace('a', op_params.loc[op_params['model']=='hom']).instance_patterns),
        hom_a_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
                params=get_subspace('a', op_params.loc[(op_params['model']=='hom') & (op_params['ss'].astype('float')<=100)]).instance_patterns),
        free_a = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('a', op_params.loc[op_params['model']=='free']).instance_patterns),
        free_a_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
                params=get_subspace('a', op_params.loc[(op_params['model']=='free') & (op_params['ss'].astype('float')<=100)]).instance_patterns),
        free_vc = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('vc', op_params.loc[op_params['model']=='free']).instance_patterns),
        free_vc_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
                params=get_subspace('vc', op_params.loc[(op_params['model']=='free') & (op_params['ss'].astype('float')<=100)]).instance_patterns),
    output:
        png = 'results/ctp/power.paper.supp.png',
    params: 
        arg_ss = 'ss',
        hom_ss = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom'])['ss']),
        free_ss = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free'])['ss']),
        hom_ss_remlJK = np.array(get_subspace('ss', op_params.loc[(op_params['model']=='hom') & (op_params['ss'].astype('float')<=100)])['ss']),
        free_ss_remlJK = np.array(get_subspace('ss', op_params.loc[(op_params['model']=='free') & (op_params['ss'].astype('float')<=100)])['ss']),
        arg_a = 'a',
        hom_a = np.array(get_subspace('a', op_params.loc[op_params['model']=='hom'])['a']),
        free_a = np.array(get_subspace('a', op_params.loc[op_params['model']=='free'])['a']),
        hom_a_remlJK = np.array(get_subspace('a', op_params.loc[(op_params['model']=='hom') & (op_params['ss'].astype('float')<=100)])['a']),
        free_a_remlJK = np.array(get_subspace('a', op_params.loc[(op_params['model']=='free') & (op_params['ss'].astype('float')<=100)])['a']),
        arg_vc = 'vc',
        free_vc = np.array(get_subspace('vc', op_params.loc[op_params['model']=='free'])['vc']),
        free_vc_remlJK = np.array(get_subspace('vc', op_params.loc[(op_params['model']=='free') & (op_params['ss'].astype('float')<=100)])['vc']),
        plot_order = op_plot_order,
    script: 'bin/paper_ctp_power.py'

rule paper_opNctp_estimates_ss:
    input:
        op_free = expand('analysis/op/free/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
        ctp_free = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
        V = expand('analysis/op/free/{params}/V.txt',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
    output:
        png = 'results/ctp/opNctp.estimate.ss.paper.supp.png',
    params:
        free = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free'])['ss']),
        plot_order = op_plot_order,
        #subspace = lambda wildcards: get_subspace(wildcards.arg,
        #        op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        #colorpalette = colorpalette,
        pointcolor = pointcolor,
        #mycolors = mycolors,
    script: 'bin/paper_opNctp_estimates_ss.py'

rule paper_opNctp_estimates_a:
    input:
        op_free = expand('analysis/op/free/{params}/out.npy',
                params=get_subspace('a', op_params.loc[op_params['model']=='free']).instance_patterns),
        ctp_free = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('a', op_params.loc[op_params['model']=='free']).instance_patterns),
        V = expand('analysis/op/free/{params}/V.txt',
                params=get_subspace('a', op_params.loc[op_params['model']=='free']).instance_patterns),
    output:
        png = 'results/ctp/opNctp.estimate.a.paper.supp.png',
    params:
        free = np.array(get_subspace('a', op_params.loc[op_params['model']=='free'])['a']),
        plot_order = op_plot_order,
        #subspace = lambda wildcards: get_subspace(wildcards.arg,
        #        op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        #colorpalette = colorpalette,
        pointcolor = pointcolor,
        #mycolors = mycolors,
    script: 'bin/paper_opNctp_estimates_a.py'

rule paper_opNctp_estimates_vc:
    input:
        op_free = expand('analysis/op/free/{params}/out.npy',
                params=get_subspace('vc', op_params.loc[op_params['model']=='free']).instance_patterns),
        ctp_free = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('vc', op_params.loc[op_params['model']=='free']).instance_patterns),
        V = expand('analysis/op/free/{params}/V.txt',
                params=get_subspace('vc', op_params.loc[op_params['model']=='free']).instance_patterns),
        op_hom = expand('analysis/op/hom/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom'].iloc[[0]]).instance_patterns),
        ctp_hom = expand('analysis/ctp/hom/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom'].iloc[[0]]).instance_patterns),
        V_hom = expand('analysis/op/hom/{params}/V.txt',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom'].iloc[[0]]).instance_patterns),
    output:
        png = 'results/ctp/opNctp.estimate.vc.paper.supp.png',
    params:
        arg = 'vc',
        free = np.array(get_subspace('vc', op_params.loc[op_params['model']=='free'])['vc']),
        plot_order = op_plot_order,
        #subspace = lambda wildcards: get_subspace(wildcards.arg,
        #        op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        #colorpalette = colorpalette,
        pointcolor = pointcolor,
        #mycolors = mycolors,
    script: 'bin/paper_opNctp_estimates_vc.py'

rule paper_opNctp_estimates_ss_full:
    input:
        op_full = expand('analysis/op/full/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='full']).instance_patterns),
        ctp_full = expand('analysis/ctp/full/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='full']).instance_patterns),
        V = expand('analysis/op/full/{params}/V.txt',
                params=get_subspace('ss', op_params.loc[op_params['model']=='full']).instance_patterns),
    output:
        png = 'results/ctp/opNctp.estimate.ss.full.paper.supp.png',
    params:
        full = np.array(get_subspace('ss', op_params.loc[op_params['model']=='full'])['ss']),
        plot_order = op_plot_order,
        #subspace = lambda wildcards: get_subspace(wildcards.arg,
        #        op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        #colorpalette = colorpalette,
        pointcolor = pointcolor,
        #mycolors = mycolors,
        vc = get_subspace('ss', op_params.loc[op_params['model']=='full'])['vc'][0],
    script: 'bin/paper_opNctp_estimates_ss_full.py'


#########################################################################################
# Cuomo et al 2020 Nature Communications
#########################################################################################
############################################ data ###########################
rule cuomo_data_format:
    input:
        raw = 'data/cuomo2020natcommun/raw_counts.csv.zip',
        log = 'data/cuomo2020natcommun/log_normalised_counts.csv.zip',
    output:
        raw = 'data/cuomo2020natcommun/raw_counts.csv.gz',
        log = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    params:
        raw = lambda wildcards, input: re.sub('\.zip$', '', input.raw),
        log = lambda wildcards, input: re.sub('\.zip$', '', input.log),
    shell:
        '''
        unzip -p {input.raw} | tr ',' '\t' | sed 's/"//g' | gzip > {output.raw}
        unzip -p {input.log} | tr ',' '\t' | sed 's/"//g' | gzip > {output.log}
        '''

rule cuomo_metadata_columns:
    input:
        meta = 'data/cuomo2020natcommun/cell_metadata_cols.tsv',
    output:
        info = 'data/cuomo2020natcommun/data.info',
    script: "bin/cuomo_data_explore.py"

#rule cuomo_cellnum_summary:
#    input:
#        meta = 'data/cuomo2020natcommun/cell_metadata_cols.tsv',
#    output:
#        summary = 'analysis/cuomo/data/cellnum.txt',
#        png = 'analysis/cuomo/data/cellnum.png',
#        png2 = 'analysis/cuomo/data/cellnum2.png',
#    script: "bin/cuomo_cellnum_summary.py"

rule cuomo_day_meta_extractLargeExperiments:
    # extract the largest experiment for each individual
    input:
        meta = 'data/cuomo2020natcommun/cell_metadata_cols.tsv',
    output:
        meta = 'analysis/cuomo/data/meta.txt',
        png = 'analysis/cuomo/data/meta.png',
    script: 'bin/cuomo_day_meta_extractLargeExperiments.py'

rule cuomo_day_pseudobulk_log:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        y = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
    resources: 
        mem_per_cpu = '10gb',
        time = '24:00:00',
    script: 'bin/cuomo_day_pseudobulk.py'

# coefficient of variation for NU
rule cuomo_day_pseudobulk_log_varNU:
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
        python3 bin/cuomo_day_pseudobulk_log_varNU.py {input.meta} {input.counts} {output.var_nu}
        '''

rule cuomo_day_pseudobulk_log_varNU_merge:
    input:
        var_nu = expand('staging/cuomo/bootstrapedNU/data/counts{i}.var_nu.gz', i=range(100)),
    output:
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
    run:
        nus = [pd.read_table(f, index_col=(0,1)) for f in input.var_nu]
        # check donor day
        index = nus[0].index
        #donors = nus[0]['donor']
        #days = nus[0]['day']
        for data in nus[1:]:
            #if np.any( donors != data['donor'] ) or np.any( days != data['day'] ):
            if np.any( index != data.index ):
                sys.exit('Wrop order!\n')
        # merge
        data = pd.concat( nus, axis=1 )
        data.to_csv( output.var_nu, sep='\t')

rule cuomo_day_pseudobulk_log_varNU_dist:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
    output:
        png = 'results/cuomo/data/log/bootstrapedNU/day.raw.var_nu.png',
    script: 'bin/cuomo_day_pseudobulk_log_varNU_dist.py'

#################################### analysis ###############
## read parameters
cuomo_params = pd.read_table('cuomo.params.txt', dtype="str", comment='#')
if cuomo_params.shape[0] != cuomo_params.drop_duplicates().shape[0]:
    sys.exit('Duplicated parameters!\n')
cuomo_paramspace = Paramspace(cuomo_params, filename_params="*")

rule cuomo_day_filterInds:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        y = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
    output:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
    script: 'bin/cuomo_day_filterInds.py'

rule cuomo_day_filterCTs:
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
    script: "bin/cuomo_split2batches.py"

rule cuomo_day_imputeGenome:
    input:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment 
    output:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene
    resources: 
        mem = '20gb',
        time = '20:00:00',
    script: 'bin/cuomo_day_imputeGenome.py'

rule cuomo_day_imputeNinputForop:
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
        imputed_ct_y = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        imputed_ct_nu_ctp = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
    resources: mem = '10gb',
    script: 'bin/cuomo_day_imputeNinputForop.py'

rule cuomo_day_summary_imputation:
    input:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
        imputed_ct_y = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_batch_no)], # donor - day * gene
        imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.nu.txt'
                for i in range(cuomo_batch_no)], #donor-day * gene # negative ct_nu set to 0
        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
    output: png = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/imputation.png',
    script: 'bin/cuomo_day_summary_imputation.py'

rule cuomo_day_y_collect:
    input:
        y = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/y.txt' 
                for i in range(cuomo_batch_no)],
    output:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',
    script: 'bin/cuomo_day_y_merge.py'

rule cuomo_day_pca:
    input:
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',
        imputed_ct_y = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch0/ct.y.txt', # donor - day * gene
        #P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
    output:
        evec = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/evec.txt',
        eval = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/eval.txt',
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.png',
    script: 'bin/cuomo_day_pca.py'

rule cuomo_day_PCassociatedVar:
    input:
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment 
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.associatedVar.png',
    script: 'bin/cuomo_day_PCassociatedVar.py'

rule cuomo_op_test:
    input:
        y_batch = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt',
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt',
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt',
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment 
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', # donor - day * gene
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/out.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        HE_as_initial = False,
    resources:
        time = '48:00:00',
        mem = '8gb',
    priority: -1
    script: "bin/cuomo/op_test.py"

#def cuomo_op_test_agg(wildcards):
#    checkpoint_output = checkpoints.cuomo_split2batches.get(**wildcards).output[0]
#    # snakemake bug
#    par = ''
#    for column in cuomo_params.columns:
#        par = par + f'{column}={wildcards[column]}/'
#    #print(par)
#    return expand(f"staging/cuomo/{par[:-1]}/batch{{i}}/out.txt", 
#            i=glob_wildcards(os.path.join(checkpoint_output, "batch{i}.txt")).i)

use rule op_aggReplications as cuomo_op_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/out.txt' 
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',

rule cuomo_op_corr_plot:
    input:
        base = expand('analysis/cuomo/{params}/out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/CTcorr.png',
    script: 'bin/cuomo_op_corr_plot.py'

rule cuomo_op_rVariance_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/rVariance.png',
    script: 'bin/cuomo_op_rVariance_plot.py'

rule cuomo_op_variance_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/vc.png',
    params:
        cut_off = {'free':[-1.5,2], 'full':[-3,3]},
    script: 'bin/cuomo_op_variance_plot.py'

rule cuomo_op_waldNlrt_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/waldNlrt.png',
    script: 'bin/cuomo_op_waldNlrt_plot.py'

rule cuomo_op_experimentR_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy', 
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/hom.png',
    script: 'bin/cuomo_op_experimentR_plot.py'

rule cuomo_op_experimentR_all:
    input:
        png = expand('results/cuomo/{params}/hom.png',
                params=Paramspace(cuomo_params.loc[cuomo_params['experiment']=='R'], filename_params="*").instance_patterns),

rule cuomo_ctp_test:
    input:
        y_batch = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctp.txt', # list
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt', #donor-day * gene 
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.out.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,  
        REML = True,
        HE = True, 
        jack_knife = True,
        IID = False,
        Hom = False,
    resources: 
        mem = '10gb',
        time = '48:00:00',
    script: 'bin/cuomo_ctp_test.py'

use rule op_aggReplications as cuomo_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',

rule cuomo_ctp_test2:
    input:
        y_batch = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctp.txt', # list
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt', #donor-day * gene 
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.out2.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp/rep/out2.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,  
        REML = True,
        HE = True, 
        jack_knife = True,
        IID = True,
        Hom = True,
    resources: 
        mem = '10gb',
        time = '48:00:00',
    script: 'bin/cuomo_ctp_test.py'

use rule op_aggReplications as cuomo_ctp_test2_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.out2.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out2.npy',

use rule cuomo_ctp_test as cuomo_ctp_test_remlJK with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.remlJK.out.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp/rep/remlJK.out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,  
        REML = True,
        Free_reml_jk = True,
        HE = False, 
        Hom = False,
        IID = False,
    resources: 
        mem = '16gb',
        time = '48:00:00',

use rule op_aggReplications as cuomo_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.remlJK.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.out.npy',

use rule cuomo_op_waldNlrt_plot as  cuomo_ctp_waldNlrt_plot with:
    # when using LRT test p value in Free REML
#rule cuomo_ctp_waldNlrt_plot:
    # when using Wald test p value in Free REML
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.waldNlrt.png',
    #script: 'bin/cuomo_ctp_waldNlrt_plot.py'

use rule cuomo_op_corr_plot as cuomo_ctp_corr_plot with:
    input:
        base = expand('analysis/cuomo/{params}/ctp.out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.CTcorr.png',

use rule cuomo_op_rVariance_plot as cuomo_ctp_rVariance_plot with:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.rVariance.png',

rule cuomo_ctp_variance_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        nu_ctp = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/nu.ctp.txt'
                for i in range(cuomo_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.vc.png',
    params:
        free = ['hom', 'CT_main', 'ct_random_var', 'nu'],
        cut_off = {'free':[-0.5,0.5], 'full':[-3,3]},
    script: 'bin/cuomo_ctp_variance_plot.py'

rule cuomo_ctp_Vplot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.V.png',
    script: 'bin/cuomo_ctp_Vplot.py'

use rule cuomo_ctp_test as cuomo_ctp_test_miny with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.miny.out.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp/rep/miny.out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,  
        REML = True,
        HE = True, 
        jack_knife = True,
    resources: 
        mem = '10gb',
        time = '48:00:00',

use rule op_aggReplications as cuomo_ctp_miny_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.miny.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.out.npy',

use rule cuomo_ctp_waldNlrt_plot as cuomo_ctp_waldNlrt_plot_miny with:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.waldNlrt.png',

use rule cuomo_op_rVariance_plot as cuomo_ctp_rVariance_plot_miny with:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.rVariance.png',

rule cuomo_ctp_HEpvalue_acrossmodel_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.HEpvalue.png',
    script: 'bin/cuomo_ctp_HEpvalue_acrossmodel_plot.py'

###### p values in REML vs HE free
rule cuomo_ctp_pvalue_REMLvsHE:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.free.REMLvsHE.inflated_zeros_{{prop}}.png',
    script: 'bin/cuomo_ctp_pvalue_REMLvsHE.py'

rule cuomo_ctp_pvalue_REMLvsHE_addrVariance:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.free.REMLvsHE.rVariance.png',
    script: 'bin/cuomo_ctp_pvalue_REMLvsHE_addrVariance.py'

rule cuomo_ng_all:
    input:
        imputation = expand('analysis/cuomo/{params}/imputation.png', 
                params=cuomo_paramspace.instance_patterns),
        op_CTcorr = expand('results/cuomo/{params}/CTcorr.png', 
                params=cuomo_paramspace.instance_patterns),
        #op_rVar = expand('results/cuomo/{params}/rVariance.png', 
        #        params=cuomo_paramspace.instance_patterns),
        #op_varcomponent = expand('results/cuomo/{params}/vc.png', 
        #        params=cuomo_paramspace.instance_patterns),
        ctp_CTcorr = expand('results/cuomo/{params}/ctp.CTcorr.png', 
                params=cuomo_paramspace.instance_patterns),
        ctp_rVar = expand('results/cuomo/{params}/ctp.rVariance.png', 
                params=cuomo_paramspace.instance_patterns),
        #ctp_varcomponent = expand('results/cuomo/{params}/ctp.vc.png', 
        #        params=cuomo_paramspace.instance_patterns),
        #opVSctp = expand('results/cuomo/{params}/opVSctp.hom.png',
        #        params=cuomo_paramspace.instance_patterns),
        #ctp_bugs = expand('analysis/cuomo/{params}/ctp_bugs.png',
        #        params=cuomo_paramspace.instance_patterns),
        pca = expand('results/cuomo/{params}/pca.associatedVar.png',
                params=cuomo_paramspace.instance_patterns),
        op_wald = expand('results/cuomo/{params}/waldNlrt.png',
                params=cuomo_paramspace.instance_patterns),
        ctp_wald = expand('results/cuomo/{params}/ctp.waldNlrt.png',
                params=cuomo_paramspace.instance_patterns),
        #ctp_HE = expand('results/cuomo/{params}/ctp.HEpvalue.png',
        #        params=cuomo_paramspace.instance_patterns),
        ctp_REMLvsHE = expand('results/cuomo/{params}/ctp.free.REMLvsHE.inflated_zeros_1.png',
                params=cuomo_paramspace.instance_patterns),
        #ctp_enrichment = expand('results/cuomo/{params}/enrichment/reml.V_bon.beta_bon.enrich.txt',
        #        params=cuomo_paramspace.instance_patterns),
        #ctp_remlJK = expand('analysis/cuomo/{params}/ctp.remlJK.out.npy',
        #        params=cuomo_paramspace.instance_patterns),

# single cell expression pattern plot
rule cuomo_sc_expressionpattern:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.out.npy',
        imputed_ct_y = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_batch_no)], # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/ctp.{{gene}}.png', 
    params:
        mycolors = mycolors,
        paper = True,
    script: 'bin/cuomo_sc_expressionpattern.py'

rule cuomo_sc_expressionpattern_paper:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.out.npy',
        imputed_ct_y = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_batch_no)], # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/paper.ctp.png', 
    params:
        mycolors = mycolors,
        genes = ['ENSG00000204531_POU5F1', 'NDUFB4', 'ENSG00000185155_MIXL1', 'ENSG00000163508_EOMES'],
    script: 'bin/cuomo_sc_expressionpattern.paper.py'

rule cuomo_sc_expressionpattern_collect:
    input:
        png = [f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/ctp.{gene}.png'
                for gene in ['ENSG00000111704_NANOG', 'ENSG00000141448_GATA6', 'ENSG00000204531_POU5F1',
                    'ENSG00000181449_SOX2', 'ENSG00000065518_NDUFB4', 'ENSG00000074047_GLI2', 'ENSG00000136997_MYC',
                    'ENSG00000125845_BMP2', 'ENSG00000107984_DKK1', 'ENSG00000234964_FABP5P7', 
                    'ENSG00000166105_GLB1L3', 'ENSG00000237550_UBE2Q2P6', 'ENSG00000230903_RPL9P8']],
    output:
        touch(f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.sc_expressionpattern.flag'),

# likelihood
rule cuomo_likelihood_plot:
    input:
        imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt'
                for i in range(cuomo_batch_no)], #donor-day * gene 
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.likelihood.png',
    script: 'bin/cuomo_likelihood_plot.py'

# p value across test methods: HE, ML, REML, Wald, JK
rule cuomo_ctp_p:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        out2 = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out2.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.out.npy',
    output:
        hom2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.hom2.png',
        p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.p.png',
    script: 'bin/cuomo_ctp_p.py'

rule cuomo_opVSctp_p:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.out.npy',
    output:
        p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/opVSctp.p.png',
    script: 'bin/cuomo_opVSctp_p.py'

# find top genes
rule cuomo_geneP:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/ctp.{{gene}}.P.txt',
    script: 'bin/cuomo_geneP.py'

rule cuomo_topgenes:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        ctp = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.out.npy',
    output:
        topgenes = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/opNctp.topgenes.txt',
    params:
        op = ['reml', 'he'],
        ctp = ['remlJK', 'he'],
    script: 'bin/cuomo_topgenes.py'

# paper plot
rule paper_cuomo_op_pvalue_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/waldNlrt.supp.png',
    script: 'bin/cuomo_op_waldNlrt_plot_paper.py'

rule paper_cuomo_ctp_pvalue_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.out.npy',
    output:
        reml_p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.REMLpvalue.paper.png',
        he_p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.HEpvalue.paper.png',
        qq = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.qq.supp.png',
    script: 'bin/paper_cuomo_ctp_pvalue_plot.py'

rule paper_cuomo_ctp_pvalue_plot_ASHG:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.REMLpvalue.paper.ASHG.png',
        png2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.HEpvalue.paper.ASHG.png',
    script: 'bin/paper_cuomo_ctp_pvalue_plot.ASHG.py'

rule paper_cuomo_freeNfull_Variance_plot:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        ctp = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.paper.png',
    script: 'bin/cuomo/paper_cuomo_freeNfull_Variance_plot.py'

rule paper_cuomo_freeNfull_Variance_plot_ASHG:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        ctp = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.paper.ASHG.png',
    script: 'bin/paper_cuomo_freeNfull_Variance_plot.ASHG.py'

rule paper_cuomo_freeNfull_Variance_plot_supp:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        ctp = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        op = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.freeNfull.Variance.supp.png',
        ctp = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.supp.png',
    script: 'bin/paper_cuomo_freeNfull_Variance_plot_supp.py'

rule paper_cuomo_ctp_corr_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.CTcorr.paper.png',
    script: 'bin/paper_cuomo_ctp_corr_plot.py'


###########################################################################################
# simulate Cuomo genes: a random gene's hom2, ct main variance, nu
###########################################################################################
cuomo_simulateGene_gene_no = 1000
cuomo_simulateGene_batch_no = 100
cuomo_simulateGene_batches = np.array_split(range(cuomo_simulateGene_gene_no), cuomo_simulateGene_batch_no)
rule cuomo_simulateGene_randompickgene:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.nu.ctp.txt'
                for i in range(cuomo_batch_no)], #donor-day * gene 
    output:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
    params: gene_no = cuomo_simulateGene_gene_no,
    script: 'bin/cuomo_simulateGene_randompickgene.py'

localrules: cuomo_simulateGene_hom_batch
rule cuomo_simulateGene_hom_batch:
    output: touch(f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/generatedata.batch')

for _, batch in enumerate(cuomo_simulateGene_batches):
    rule:
        name: f'cuomo_simulateGene_hom_batch{_}'
        input:
            flag = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/generatedata.batch',
            out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
            genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
            imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.nu.ctp.txt'
                    for i in range(cuomo_batch_no)], #donor-day * gene 
            P = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/P.txt'
                    for i in range(cuomo_batch_no)], # list
        output:
            genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{_}.txt',
            P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{_}.txt',
            ctnu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/ctnu.batch{_}.txt',
            cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/cty.batch{_}.txt',
        params:
            batch = batch,
            cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/rep/cty.txt' 
        resources: burden = 105,
        script: 'bin/cuomo_simulateGene_hom_batchx.py'

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
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/rep/ctp.out.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = True,
        REML = True,
        HE = True,

use rule op_aggReplications as cuomo_simulateGene_hom_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.out.npy',

use rule ctp_test as cuomo_simulateGene_hom_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.remlJK.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/rep/ctp.remlJK.out.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = False,
        REML = True,
        Free_reml_jk = True,
        HE = False,
    resources:
        mem_per_cpu = '12gb',
        time = '48:00:00',

use rule op_aggReplications as cuomo_simulateGene_hom_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.remlJK.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctp.remlJK.out.npy',

nu_noises = ['1_0_0', '1_2_20', '1_2_10', '1_2_5', '1_2_3', '1_2_2']
rule cuomo_simulateGene_hom_ctp_test_powerplot:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.remlJK.out.npy'
                for nu_noise in nu_noises],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.hom.power.png',
    params:
        nu_noises = nu_noises,
    script: 'bin/cuomo_simulateGene_hom_ctp_test_powerplot.py'

rule cuomo_simulateGene_hom_ctp_test_estimates:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.remlJK.out.npy'
                for nu_noise in nu_noises],
        real_out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        genes = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{i}.txt' 
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.hom.estimates.png',
    params:
        nu_noises = nu_noises,
    script: 'bin/cuomo_simulateGene_hom_ctp_test_estimates.py'

rule cuomo_simulateGene_hom_ctp_test_estimates_paper:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.out.npy'
                for nu_noise in nu_noises],
        #remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.remlJK.out.npy'
        #        for nu_noise in nu_noises],
        #real_out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        #genes = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{i}.txt' 
        #        for i in range(cuomo_simulateGene_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.hom.estimates.paper.png',
    params:
        nu_noises = nu_noises,
    script: 'bin/cuomo_simulateGene_hom_ctp_test_estimates_paper.py'

rule cuomo_simulateGene_Free_addV:
    input:
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/cty.batch{{i}}.txt',
    output:
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/cty.batch{{i}}.txt',
    params:
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/rep/cty.txt' 
    resources: burden = 105,
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
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/rep/ctp.out.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = True,
        REML = True,
        HE = True,

use rule op_aggReplications as cuomo_simulateGene_Free_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.out.npy',

use rule ctp_test as cuomo_simulateGene_Free_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.remlJK.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/rep/ctp.remlJK.out.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = False,
        REML = True,
        Free_reml_jk = True,
        HE = False,
    resources:
        mem_per_cpu = '12gb',
        time = '48:00:00',

use rule op_aggReplications as cuomo_simulateGene_Free_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.remlJK.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctp.remlJK.out.npy',

V1 = ['0_0_0_0', '0.05_0_0_0','0.1_0_0_0', '0.2_0_0_0', '0.5_0_0_0']
V2 = ['0.05_0.05_0.05_0.05', '0.1_0.1_0.1_0.1', '0.2_0.2_0.2_0.2', '0.5_0.5_0.5_0.5']
V3 = ['0_0_0_0', '0.05_0.1_0.1_0.1', '0.1_0.1_0.1_0.1', '0.2_0.1_0.1_0.1', '0.5_0.1_0.1_0.1']
rule cuomo_simulateGene_Free_ctp_test_powerplot:
    input:
        outs1 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.out.npy'
                for V in V1],
        outs2 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.out.npy'
                for V in V2],
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.out.npy'
                for V in V3],
        remlJKs1 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.out.npy'
                for V in V1],
        remlJKs2 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.out.npy'
                for V in V2],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.out.npy'
                for V in V3],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.free.1_2_5.power.png',
    params:
        V1 = V1,
        V2 = V2,
        V3 = V3,
    script: 'bin/cuomo_simulateGene_Free_ctp_test_powerplot.py'

rule cuomo_simulateGene_ctp_test_powerplot_paper:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.remlJK.out.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.out.npy'
                for V in V3],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.out.npy'
                for V in V3],
    output:
        png1 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.power.paper.supp.png',
        png2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.power.paper.png',
    params:
        nu_noises = nu_noises,
        V3 = V3,
    script: 'bin/cuomo/simulateGene_ctp_test_powerplot_paper.py'

rule cuomo_simulateGene_ctp_test_powerplot_paper_ASHG:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.remlJK.out.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.out.npy'
                for V in V3],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.out.npy'
                for V in V3],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.power.paper.ASHG.png',
    params:
        nu_noises = nu_noises,
        V3 = V3,
    script: 'bin/cuomo_simulateGene_ctp_test_powerplot_paper.ASHG.py'

rule cuomo_simulateGene_ctp_test_powerplot_paper_ZJU:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.remlJK.out.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.out.npy'
                for V in V3],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.out.npy'
                for V in V3],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.power.paper.ZJU.png',
    params:
        nu_noises = nu_noises,
        V3 = V3,
    script: 'bin/cuomo_simulateGene_ctp_test_powerplot_paper.ZJU.py'

rule cuomo_simulateGene_free_ctp_test_estimates:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.out.npy'
                for V in V3],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.out.npy'
                for V in V3],
        #real_out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.out.npy',
        #genes = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{i}.txt' 
        #        for i in range(cuomo_simulateGene_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.free.estimates.png',
    params:
        V3 = V3,
    script: 'bin/cuomo_simulateGene_free_ctp_test_estimates.py'


#########################################################################
################ imputation accuracy ###########################
#########################################################################
cuomo_imput_params = pd.read_table('cuomo.imput.params.txt', dtype="str", comment='#')
cuomo_imput_paramspace = Paramspace(cuomo_imput_params, filename_params="*")
#cuomo_imput_paramspace2 = Paramspace(cuomo_imput_params.loc[cuomo_imput_params['im_genome']=='Y'], filename_params="*")
cuomo_imput_batch_no = 10
cuomo_imputataion_reps = range(10)
#cuomo_imputataion_reps = range(50)

wildcard_constraints:
    random_mask = "\w+"
wildcard_constraints:
    ind_min_cellnum = "\d+"
wildcard_constraints:
    ct_min_cellnum = "\d+"
wildcard_constraints:
    k = "\d+"
wildcard_constraints:
    missingness = "[\d\.]+"

use rule cuomo_day_filterInds as cuomo_imputation_day_filterInds with:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        y = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
    output:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        #P = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        #n = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
        y = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.nu.gz', # donor - day * gene
        P = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.prop.gz', # donor * day
        n = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.cellnum.gz', # donor * day

use rule cuomo_day_filterCTs as cuomo_imputation_day_filterCTs with:
    input:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        #n = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
        y = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.nu.gz', # donor - day * gene
        n = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.cellnum.gz', # donor * day
    output:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
        y = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/day.filterCTs.nu.gz', # donor - day * gene

rule cuomo_imputation_AddMissing:
    input:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
        y = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/day.filterCTs.nu.gz', # donor - day * gene
    output:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.masked.pseudobulk.gz', # donor - day * gene
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.masked.nu.gz', # donor - day * gene
        #log = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/addmissing.log',
        y = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/rep{k}/day.masked.pseudobulk.gz', # donor - day * gene
        nu = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/rep{k}/day.masked.nu.gz', # donor - day * gene
        log = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/rep{k}/addmissing.log',
    script: 'bin/cuomo_imputation_AddMissing.py'

use rule cuomo_split2batches as cuomo_imputation_split2batches with:
    input:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.masked.pseudobulk.gz', # donor - day * gene
        y = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/rep{k}/day.masked.pseudobulk.gz', # donor - day * gene
    output:
        y_batch = expand('staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/y/batch{i}.txt', 
                i=range(cuomo_imput_batch_no)),

use rule cuomo_day_imputeGenome as cuomo_imputation_day_imputeGenome with:
    input:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.masked.pseudobulk.gz', # donor - day * gene
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.masked.nu.gz', # donor - day * gene
        y = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/rep{k}/day.masked.pseudobulk.gz', # donor - day * gene
        nu = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/rep{k}/day.masked.nu.gz', # donor - day * gene
    output:
        #y = temp(f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.Gimputed.pseudobulk.gz'), # donor - day * gene
        #nu = temp(f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.Gimputed.nu.gz'), #donor - day * gene
        y = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', #donor - day * gene
        #y = temp(f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz'), # donor - day * gene
        #nu = temp(f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/day.Gimputed.nu.gz'), #donor - day * gene

#use rule cuomo_day_imputeNinputForop as cuomo_imputation_day_imputeNinputForop with:
# no standardization of y and nu
rule cuomo_imputation_day_imputeNinputForop:
    input:
        #P = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        #n = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.Gimputed.nu.gz', #donor - day * gene
        #y_batch = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/y/batch{{i}}.txt', # genes
        P = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.prop.gz', # donor * day
        n = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.cellnum.gz', # donor * day
        y = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', #donor - day * gene
        y_batch = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/rep{k}/y/batch{i}.txt', # genes
    output:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/nu.txt', # list
        #nu_ctp = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/nu.ctp.txt', # list
        #P = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/P.txt', # list
        #imputed_ct_y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/ct.y.txt', # donor - day * gene
        #imputed_ct_nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        #imputed_ct_nu_ctp = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/ct.nu.ctp.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
        y = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        nu = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
        nu_ctp = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/nu.ctp.txt', # list
        P = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_ct_y = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # list 
        imputed_ct_nu = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', # list # negative ct_nu set to 0
        imputed_ct_nu_ctp = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt', # list  # negative ct_nu set to max(ct_nu)
    resources: 
        time= '200:00:00',
        mem = lambda wildcards: '15gb' if wildcards.im_mvn == 'N' else '5gb',
    script: 'bin/cuomo_imputation_day_imputeNinputForop.py'

rule cuomo_imputation_day_cleanfile:
    input:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/nu.txt', # list
        #nu_ctp = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/nu.ctp.txt', # list
        #P = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/P.txt', # list
        #imputed_ct_nu_ctp = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/ct.nu.ctp.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
        y = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        nu = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
        nu_ctp = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/nu.ctp.txt', # list
        P = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_ct_nu_ctp = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt', # list * gene # negative ct_nu set to max(ct_nu)
    output:
        #touch(f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/clean.txt'),
        touch(f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/clean.txt'),
    run:
        for fs in input:
            for f in open(fs):
                os.system(f'rm {f.strip()}')

rule cuomo_imputation_merge:
    # and initiate file cleaning
    input:
        #clean = [f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{i}/clean.txt'
        #        for i in range(cuomo_imput_batch_no)],
        #imputed_ct_y = [f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{i}/ct.y.txt'
        #        for i in range(cuomo_imput_batch_no)], # donor - day * gene
        #imputed_ct_nu = [f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{i}/ct.nu.txt'
        #        for i in range(cuomo_imput_batch_no)], #donor-day * gene # negative ct_nu set to 0
        clean = [f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{i}/clean.txt'
                for i in range(cuomo_imput_batch_no)],
        imputed_ct_y = [f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_imput_batch_no)], # donor - day * gene
        imputed_ct_nu = [f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{i}/ct.nu.txt'
                for i in range(cuomo_imput_batch_no)], #donor-day * gene # negative ct_nu set to 0
    output:
        #merged_ct_y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/merged.ct.y.txt', # donor - day * gene
        #merged_ct_nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/merged.ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        merged_ct_y = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/merged.ct.y.gz', # donor - day * gene
        merged_ct_nu = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/merged.ct.nu.gz', #donor-day * gene # negative ct_nu set to 0
    script: 'bin/cuomo_imputation_merge.py'

rule cuomo_imputation_accuracy:
    input:
        #P = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        #raw_y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        #raw_nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
        #masked_y = [f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{k}/day.masked.pseudobulk.gz'
        #        for k in cuomo_imputataion_reps], # donor - day * gene
        #masked_nu = [f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{k}/day.masked.nu.gz'
        #        for k in cuomo_imputataion_reps], # donor - day * gene
        #imputed_y = [f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{k}/merged.ct.y.txt'
        #        for k in cuomo_imputataion_reps], # donor - day * gene
        #imputed_nu = [f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{k}/merged.ct.nu.txt'
        #        for k in cuomo_imputataion_reps], #donor-day * gene # negative ct_nu kept
        P = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/day.filterInds.prop.gz', # donor * day
        raw_y = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        raw_nu = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/day.filterCTs.nu.gz', # donor - day * gene
        masked_y = expand('staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{k}/day.masked.pseudobulk.gz',
            k=cuomo_imputataion_reps), # donor - day * gene
        masked_nu = expand('staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{k}/day.masked.nu.gz',
            k=cuomo_imputataion_reps), # donor - day * gene
        imputed_y = [f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{k}/{cuomo_imput_paramspace.wildcard_pattern}/merged.ct.y.gz'
            for k in cuomo_imputataion_reps], # donor - day * gene
        imputed_nu = [f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{k}/{cuomo_imput_paramspace.wildcard_pattern}/merged.ct.nu.gz'
            for k in cuomo_imputataion_reps], #donor-day * gene # negative ct_nu kept
    output:
        #y_mse = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/y.mse', # series for gene-ct
        #nu_mse = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/nu.mse', # series for gene-ct
        #y_cor = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/y.cor', # series for gene-ct
        #nu_cor = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/nu.cor', # series for gene-ct
        #nu_png = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/nu.png', # series for gene-ct
        #raw_nu_standradized = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.nu.std.gz', # donor - day * gene
        y_mse = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.mse', # series for gene-ct
        nu_mse = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.mse', # series for gene-ct
        y_cor = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.cor', # series for gene-ct
        nu_cor = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.cor', # series for gene-ct
        y_mse_within = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.withinrep.mse', # series for gene-ct
        y_mse_within_tmp = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.withinrep.tmp.mse', # series for gene-ct
        nu_mse_within = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.withinrep.mse', # series for gene-ct
        y_cor_within = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.withinrep.cor', # series for gene-ct
        nu_cor_within = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.withinrep.cor', # series for gene-ct
        #nu_png = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.png', # series for gene-ct
        #raw_nu_standradized = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.nu.std.gz', # donor - day * gene
    resources: 
        time = '48:00:00',
        mem = '10gb',
    script: 'bin/cuomo_imputation_accuracy.py'

rule cuomo_imputation_accuracy_plot:
    input:
        #y_mse = expand('analysis/imp/{params}/y.mse', params=cuomo_imput_paramspace.instance_patterns),
        #nu_mse = expand('analysis/imp/{params}/nu.mse', params=cuomo_imput_paramspace.instance_patterns),
        #y_cor = expand('analysis/imp/{params}/y.cor', params=cuomo_imput_paramspace.instance_patterns),
        #nu_cor = expand('analysis/imp/{params}/nu.cor', params=cuomo_imput_paramspace.instance_patterns),
        y_mse = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/y.mse', params=cuomo_imput_paramspace.instance_patterns),
        nu_mse = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/nu.mse', params=cuomo_imput_paramspace.instance_patterns),
        y_cor = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/y.cor', params=cuomo_imput_paramspace.instance_patterns),
        nu_cor = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/nu.cor', params=cuomo_imput_paramspace.instance_patterns),
    output:
        png = 'results/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/imputation.accuracy.png',
    params:
        labels = cuomo_imput_paramspace.loc[:],
    script: 'bin/cuomo_imputation_accuracy_plot.py'

use rule cuomo_imputation_accuracy_plot as cuomo_imputation_accuracy_plot2 with:
    input:
        y_mse = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/y.withinrep.mse', 
                params=cuomo_imput_paramspace.instance_patterns),
        nu_mse = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/nu.withinrep.mse', 
                params=cuomo_imput_paramspace.instance_patterns),
        y_cor = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/y.withinrep.cor', 
                params=cuomo_imput_paramspace.instance_patterns),
        nu_cor = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/nu.withinrep.cor', 
                params=cuomo_imput_paramspace.instance_patterns),
    output:
        png = 'results/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/imputation.accuracy.withinrep.png',

#use rule cuomo_imputation_accuracy_plot as cuomo_imputation_accuracy_plot_tmp with:
#    input:
#        y_mse = expand('analysis/imp/{params}/y.mse', params=cuomo_imput_paramspace2.instance_patterns),
#        nu_mse = expand('analysis/imp/{params}/nu.mse', params=cuomo_imput_paramspace2.instance_patterns),
#        y_cor = expand('analysis/imp/{params}/y.cor', params=cuomo_imput_paramspace2.instance_patterns),
#        nu_cor = expand('analysis/imp/{params}/nu.cor', params=cuomo_imput_paramspace2.instance_patterns),
#    output:
#        png = 'results/imp/imputation.accuracy.tmp.png',
#    params:
#        labels = cuomo_imput_paramspace2.loc[:],

rule cuomo_imputation_all:
    input:
        png = expand('results/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/imputation.accuracy.png',
                ind_min_cellnum=100, ct_min_cellnum=10, missingness=[0.1]),
        png2 = expand('results/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/imputation.accuracy.withinrep.png',
                ind_min_cellnum=100, ct_min_cellnum=10, missingness=[0.1]),


cuomo_imput_paper_params = pd.read_table('cuomo.imput.params.txt', dtype="str", comment='#')
cuomo_imput_paper_params = cuomo_imput_paper_params.loc[
        (cuomo_imput_paper_params['im_mvn'] == 'Y') | (cuomo_imput_paper_params['im_scale'] == 'Y')]
cuomo_imput_paper_paramspace = Paramspace(cuomo_imput_paper_params, filename_params="*")

rule paper_cuomo_imputation_accuracy_plot:
    input:
        y_mse = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/y.withinrep.mse', 
                params=cuomo_imput_paper_paramspace.instance_patterns),
        nu_mse = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/nu.withinrep.mse', 
                params=cuomo_imput_paper_paramspace.instance_patterns),
        y_cor = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/y.withinrep.cor', 
                params=cuomo_imput_paper_paramspace.instance_patterns),
        nu_cor = expand('analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{params}/nu.withinrep.cor', 
                params=cuomo_imput_paper_paramspace.instance_patterns),
    output:
        png = 'results/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/imputation.accuracy.withinrep.paper.png',
    params:
        labels = cuomo_imput_paper_paramspace.loc[:],
    script: 'bin/paper_cuomo_imputation_accuracy_plot.py'

rule paper_cuomo_imputation_all:
    input:
        png = expand('results/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/imputation.accuracy.withinrep.paper.png',
                ind_min_cellnum=100, ct_min_cellnum=10, missingness=[0.1]),


###########
# NEW
###########
include: 'CTMM.snake'
include: 'xCTMM.snake'
