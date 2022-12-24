from snakemake.utils import Paramspace
import re, sys, os, gzip, math, time, scipy, tempfile, copy
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
#import rpy2.robjects as robjects
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

#def bmatrix(a):
#    """Returns a LaTeX bmatrix
#
#    :a: numpy array
#    :returns: LaTeX bmatrix as a string
#    """
#    if len(a.shape) > 2:
#        raise ValueError('bmatrix can at most display two dimensions')
#    lines = str(a).replace('[', '').replace(']', '').splitlines()
#    rv = [r'\begin{bmatrix}']
#    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
#    rv +=  [r'\end{bmatrix}']
#    return '\n'.join(rv)

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
ong_replicates = 1000
ong_batch_no = 100
ong_batches = np.array_split(range(ong_replicates), ong_batch_no)

## paramspace
ong_params = pd.read_table("ong.params.txt", dtype="str", comment='#', na_filter=False)
if ong_params.shape[0] != ong_params.drop_duplicates().shape[0]:
    sys.exit('Duplicated parameters!\n')
par_columns = list(ong_params.columns)
par_columns.remove('model') # columns after removing 'model'
ong_paramspace = Paramspace(ong_params[par_columns], filename_params="*")

ong_plot_order = {
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

ong_excluderareCT_plot_order = copy.deepcopy(ong_plot_order)
for model in ong_excluderareCT_plot_order.keys():
    ong_excluderareCT_plot_order[model]['a'].remove('0.5_2_2_2')

rule ong_celltype_expectedPInSnBETAnV:
    output:
        pi = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/PI.txt',
        s = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/S.txt',
        beta = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/V.txt',
    params:
        simulation=ong_paramspace.instance,
    script: 'bin/ong_celltype_expectedPInSnBETAnV.py'

localrules: ong_generatedata_batch
rule ong_generatedata_batch:
    output: touch(f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/generatedata.batch')

for _, batch in enumerate(ong_batches):
    rule: # generate simulation data for each batch
        name: f'ong_generatedata_batch{_}'
        input:
            flag = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/generatedata.batch',
            beta = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/celltypebeta.txt',
            V = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/V.txt',
        output:
            P = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/P.batch{_}.txt',
            pi = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/estPI.batch{_}.txt',
            s = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/estS.batch{_}.txt',
            nu = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/nu.batch{_}.txt',
            y = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/pseudobulk.batch{_}.txt',
        params:
            P = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/P.txt' for i in batch],
            pi = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/estPI.txt' 
                    for i in batch],
            s = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/estS.txt' 
                    for i in batch],# sample prop cov matrix
            nu = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/nu.txt' for i in batch],
            y = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/pseudobulk.txt' for i in batch],
            sim = ong_paramspace.instance,
        run:
            # par
            beta = np.loadtxt(input.beta)
            V = np.loadtxt(input.V)
            try:
                hom2 = float(params.sim['vc'].values[0].split('_')[0]) # variance of individual effect
                mean_nu = float(params.sim['vc'].values[0].split('_')[-1]) # mean variance for residual error acros individuals
                var_nu = float(params.sim['var_nu'].values[0]) #variance of variance for residual error across individuals
                a = np.array([float(x) for x in params.sim['a'].values[0].split('_')])
                ss = int(float(params.sim['ss'].values[0]))
                C = len(params.sim['a'].values[0].split('_'))
            except:
                hom2 = float(params.sim['vc'].split('_')[0]) # variance of individual effect
                mean_nu = float(params.sim['vc'].split('_')[-1]) # mean variance for residual error acros individuals
                var_nu = float(params.sim['var_nu']) #variance of variance for residual error across individuals
                a = np.array([float(x) for x in params.sim['a'].split('_')])
                ss = int(float(params.sim['ss']))
                C = len(params.sim['a'].split('_'))
            rng = np.random.default_rng()

            for P_f, pi_f, s_f, y_f, nu_f in zip(params.P, params.pi, params.s, params.y, params.nu):
                os.makedirs(os.path.dirname(P_f), exist_ok=True)
                # simulate cell type proportions
                P = rng.dirichlet(alpha=a, size=ss)
                np.savetxt(P_f, P, delimiter='\t')
                pi = np.mean(P, axis=0)
                np.savetxt(pi_f, pi, delimiter='\t')

                # estimate cov matrix S
                ## demeaning P 
                pd = P-pi
                ## covariance
                s = (pd.T @ pd)/ss
                #print(bmatrix(s))
                np.savetxt(s_f, s, delimiter='\t')

                # draw alpha / hom effect
                alpha = rng.normal(loc=0, scale=math.sqrt(hom2), size=ss)

                # draw gamma (interaction)
                if wildcards.model != 'hom':
                    gamma = rng.multivariate_normal(np.zeros(C), V, size=ss)
                    interaction = np.sum(P * gamma, axis=1)
                    #interaction = scipy.linalg.khatri_rao(np.eye(ss), P.T).T @ gamma.flatten()

                # draw residual error
                ## draw variance of residual error for each individual from gamma distribution \Gamma(k, theta)
                ## with mean = k * theta, var = k * theta^2, so theta = var / mean, k = mean / theta
                ## since mean = 0.25 and assume var = 0.01, we can get k and theta
                theta = var_nu / mean_nu 
                k = mean_nu / theta 
                ### variance of residual error for each individual
                nu = rng.gamma(k, scale=theta, size=ss)
                np.savetxt(nu_f, nu, delimiter='\t')

                ## draw residual error from normal distribution with variance drawn above
                delta = rng.normal(np.zeros_like(nu), np.sqrt(nu))

                # generate pseudobulk
                if wildcards.model == 'hom':
                    y = alpha + P @ beta + delta
                else:
                    y = alpha + P @ beta + interaction + delta 
                
                # save
                np.savetxt(y_f, y, delimiter='\t')

            with open(output.P, 'w') as f: f.write('\n'.join(params.P))
            with open(output.pi, 'w') as f: f.write('\n'.join(params.pi))
            with open(output.s, 'w') as f: f.write('\n'.join(params.s))
            with open(output.y, 'w') as f: f.write('\n'.join(params.y))
            with open(output.nu, 'w') as f: f.write('\n'.join(params.nu))

rule ong_test:
    input:
        y = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/out.batch{{i}}',
    params:
        out = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/rep/out.npy',
        batch = lambda wildcards: ong_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
    resources:
        mem_per_cpu = '5000',
    script: 'bin/ong_test.py'

rule ong_aggReplications:
    input:
        s = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/estS.batch{i}.txt'
                for i in range(len(ong_batches))],
        nu = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/nu.batch{i}.txt' for i in range(len(ong_batches))],
        pi = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/estPI.batch{i}.txt' 
                for i in range(len(ong_batches))],
        out = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(ong_batches))],
    output:
        s = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/estS.txt',
        nu = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/nu.txt',
        pi = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/estPI.txt',
        out = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/out.npy',
    script: 'bin/ong_aggReplications.py'

def ong_agg_out_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, ong_params.loc[ong_params['model']==wildcards.model])
    return expand('analysis/ong/{{model}}/{params}/out.npy', params=subspace.instance_patterns)

def ong_agg_truebeta_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, ong_params.loc[ong_params['model']==wildcards.model])
    return expand('analysis/ong/{{model}}/{params}/celltypebeta.txt', params=subspace.instance_patterns)

def ong_agg_trueV_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, ong_params.loc[ong_params['model']==wildcards.model])
    return expand('analysis/ong/{{model}}/{params}/V.txt', params=subspace.instance_patterns)

rule ong_MLestimates_subspace_plot:
    input:
        out = ong_agg_out_subspace,
        beta = ong_agg_truebeta_subspace,
        V = ong_agg_trueV_subspace,
    output:
        png = 'results/ong/{model}/ML.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: "bin/ong_MLestimates_subspace_plot.py"

rule ong_MLwaldNlrt_subspace_plot:
    input:
        out = ong_agg_out_subspace, 
    output:
        waldNlrt = 'results/ong/{model}/ML.waldNlrt.AGG{arg}.png',
    params: 
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: 'bin/ong_MLwaldNlrt_subspace_plot.py'

rule ong_REMLestimates_subspace_plot:
    input:
        out = ong_agg_out_subspace,
        V = ong_agg_trueV_subspace,
    output:
        png = 'results/ong/{model}/REML.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: 'bin/ong_REMLestimates_subspace_plot.py'

rule ong_REMLwaldNlrt_subspace_plot:
    input:
        out = ong_agg_out_subspace, 
    output:
        waldNlrt = 'results/ong/{model}/REML.waldNlrt.AGG{arg}.png',
    params: 
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: 'bin/ong_REMLwaldNlrt_subspace_plot.py'

rule ong_HEestimates_AGGsubspace_plot:
    input:
        out = ong_agg_out_subspace,
        V = ong_agg_trueV_subspace,
    output:
        png = 'results/ong/{model}/HE.AGG{arg}.png',
    params: 
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: "bin/ong_HEestimates_AGGsubspace_plot.py"

rule ong_HEwald_subspace_plot:
    input:
        out = ong_agg_out_subspace, 
    output:
        waldNlrt = 'results/ong/{model}/HE.wald.AGG{arg}.png',
    params: 
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: 'bin/ong_HEwald_subspace_plot.py'

def ong_MLestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ong/{{model}}/ML.AGG{arg}.png', arg=effective_args)

def ong_MLwaldNlrt_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ong/{{model}}/ML.waldNlrt.AGG{arg}.png', arg=effective_args)

def ong_REMLestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ong/{{model}}/REML.AGG{arg}.png', arg=effective_args)

def ong_REMLwaldNlrt_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ong/{{model}}/REML.waldNlrt.AGG{arg}.png', arg=effective_args)

def ong_HEestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ong/{{model}}/HE.AGG{arg}.png', arg=effective_args)

def ong_HEwald_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ong/{{model}}/HE.wald.AGG{arg}.png', arg=effective_args)

rule ong_AGGarg:
    input:
        ML = ong_MLestimates_AGGarg_fun,
        ML_waldNlrt = ong_MLwaldNlrt_AGGarg_fun,
        REML = ong_REMLestimates_AGGarg_fun,
        REML_waldNlrt = ong_REMLwaldNlrt_AGGarg_fun,
        HE = ong_HEestimates_AGGarg_fun,
        HE_wald = ong_HEwald_AGGarg_fun,
    output:
        flag = touch('staging/ong/{model}/all.flag'),

rule ong_mainfig_LRT:
    input:
        hom_ss = expand('analysis/ong/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        iid_ss = expand('analysis/ong/iid/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='iid']).instance_patterns),
        free_ss = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        hom_a = expand('analysis/ong/hom/{params}/out.npy',
                params=get_subspace('a', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        iid_a = expand('analysis/ong/iid/{params}/out.npy',
                params=get_subspace('a', ong_params.loc[ong_params['model']=='iid']).instance_patterns),
        free_a = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('a', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        #hom_vc = expand('analysis/ong/hom/{params}/out.npy',
        #        params=get_subspace('vc', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        iid_vc = expand('analysis/ong/iid/{params}/out.npy',
                params=get_subspace('vc', ong_params.loc[ong_params['model']=='iid']).instance_patterns),
        free_vc = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('vc', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        free_V_diag = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('V_diag', ong_params.loc[ong_params['model']=='free']).instance_patterns),
    output:
        png = 'results/ong/mainfig.LRT.png',
    params:
        hom_ss = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='hom'])['ss']),
        iid_ss = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='iid'])['ss']),
        free_ss = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='free'])['ss']),
        hom_a = np.array(get_subspace('a', ong_params.loc[ong_params['model']=='hom'])['a']),
        iid_a = np.array(get_subspace('a', ong_params.loc[ong_params['model']=='iid'])['a']),
        free_a = np.array(get_subspace('a', ong_params.loc[ong_params['model']=='free'])['a']),
        #hom_vc = np.array(get_subspace('vc', ong_params.loc[ong_params['model']=='hom'])['vc']),
        iid_vc = np.array(get_subspace('vc', ong_params.loc[ong_params['model']=='iid'])['vc']),
        free_vc = np.array(get_subspace('vc', ong_params.loc[ong_params['model']=='free'])['vc']),
        free_V_diag = np.array(get_subspace('V_diag', ong_params.loc[ong_params['model']=='free'])['V_diag']),
        plot_order = ong_plot_order,
    script: "bin/ong_mainfig_LRT.py"

rule ong_all:
    input:
        flag = expand('staging/ong/{model}/all.flag', model=['null', 'hom', 'iid', 'free', 'full']),
        png = 'results/ong/mainfig.LRT.png',

#########################################################################################
# CT pseudo-bulk non-genetic
#########################################################################################
# par
ctng_replicates = 1000
ctng_batch_no = 100
ctng_batches = np.array_split(range(ctng_replicates), ctng_batch_no)

## declare a dataframe to be a paramspace
#ctng_params = pd.read_table("ctng.params.txt", dtype="str", comment='#')
#if ctng_params.shape[0] != ctng_params.drop_duplicates().shape[0]:
#    sys.exit('Duplicated parameters!\n')
#ctng_par_columns = list(ctng_params.columns)
#ctng_par_columns.remove('model')
#ctng_paramspace = Paramspace(ctng_params[ctng_par_columns])
#
#ctng_plot_order = {
#    'hom':{
#        'ss':['1e2', '5e2', '1e3', '2e3'], 'a':['0.5_2_2_2', '1_2_2_2', '2_2_2_2', '4_2_2_2']
#        },
#    'iid':{
#        'ss':['1e2', '5e2', '1e3', '2e3'], 'a':['0.5_2_2_2', '1_2_2_2', '2_2_2_2', '4_2_2_2'],
#        'vc':['0.2_0.2_0.1_0.3_0.2', '0.2_0.2_0.2_0.2_0.2', '0.2_0.2_0.3_0.1_0.2']
#        }, 
#    'free': {
#        'ss':['1e2', '5e2', '1e3', '2e3', '5e3'], 'a':['0.5_2_2_2', '1_2_2_2', '2_2_2_2', '4_2_2_2'], 
#        'vc':['0.2_0.2_0.1_0.3_0.2', '0.2_0.2_0.2_0.2_0.2', '0.2_0.2_0.3_0.1_0.2'],
#        'V_diag':['64_16_4_1', '27_9_3_1', '8_4_2_1', '1_1_1_1'],
#        },
#    'full':{
#        'ss':['1e2', '5e2', '1e3', '2e3'], 'a':['0.5_2_2_2', '1_2_2_2', '2_2_2_2', '4_2_2_2'],
#        'vc':['0.2_0.2_0.1_0.3_0.2', '0.2_0.2_0.2_0.2_0.2', '0.2_0.2_0.3_0.1_0.2'],
#        'V_diag':['64_64_1_1', '64_16_4_1', '27_9_3_1', '8_4_2_1', '1_1_1_1'],
#        'V_tril':['0.25_0.25_0_-0.25_0_0', '0.5_0.5_0_-0.5_0_0', '0.75_0.75_0_-0.75_0_0', '0.95_0.95_0.95_-0.95_-0.95_-0.95']
#        },
#    }


#rule ctng_celltype_expectedPInSnBETAnV:
#    output:
#        pi = f'analysis/ctng/{{model}}/{ctng_paramspace.wildcard_pattern}/PI.txt',
#        s = f'analysis/ctng/{{model}}/{ctng_paramspace.wildcard_pattern}/S.txt',
#        beta = f'analysis/ctng/{{model}}/{ctng_paramspace.wildcard_pattern}/celltypebeta.txt',
#        V = f'analysis/ctng/{{model}}/{ctng_paramspace.wildcard_pattern}/V.txt',
#    params:
#        simulation=ctng_paramspace.instance,
#    script: "bin/ctng_celltype_expectedPInSnBETAnV.py"

localrules: ctng_generatedata_batch
rule ctng_generatedata_batch:
    output: touch(f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/generatedata.batch')

for _, batch in enumerate(ctng_batches):
    rule: # generate simulation data for each batch
        name: f'ctng_generatedata_batch{_}'
        input:
            flag = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/generatedata.batch',
            beta = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/celltypebeta.txt',
            V = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/V.txt',
        output:
            P = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/P.batch{_}.txt',
            pi = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/estPI.batch{_}.txt',
            s = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/estS.batch{_}.txt',
            nu = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/nu.batch{_}.txt',
            y = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/pseudobulk.batch{_}.txt',
            overall_nu = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/overall.nu.batch{_}.txt',
            overall_y = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/overall.pseudobulk.batch{_}.txt',
            fixed_M = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{_}.txt',
            random_M = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{_}.txt',
        params:
            P = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/P.txt' for i in batch],
            pi = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/estPI.txt' 
                    for i in batch],
            s = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/estS.txt' 
                    for i in batch],# sample prop cov matrix
            nu = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/nu.txt' for i in batch],
            y = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/pseudobulk.txt' for i in batch],
            overall_nu = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/overall.nu.txt' 
                    for i in batch],
            overall_y = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/overall.pseudobulk.txt' 
                    for i in batch],
            fixed_M = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/overall.fixedeffectmatrix.txt' 
                    for i in batch],
            random_M = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep{i}/overall.randomeffectmatrix.txt'
                    for i in batch],
        run:
            # par
            beta = np.loadtxt(input.beta)
            V = np.loadtxt(input.V)
            try:
                hom2 = float(wildcards['vc'].values[0].split('_')[0]) # variance of individual effect
                mean_nu = float(wildcards['vc'].values[0].split('_')[-1]) # mean variance for residual error acros individuals
                var_nu = float(wildcards['var_nu'].values[0]) #variance of variance for residual error across individuals
                a = np.array([float(x) for x in wildcards['a'].values[0].split('_')])
                ss = int(float(wildcards['ss'].values[0]))
                C = len(wildcards['a'].values[0].split('_'))
            except:
                hom2 = float(wildcards['vc'].split('_')[0]) # variance of individual effect
                mean_nu = float(wildcards['vc'].split('_')[-1]) # mean variance for residual error acros individuals
                var_nu = float(wildcards['var_nu']) #variance of variance for residual error across individuals
                a = np.array([float(x) for x in wildcards['a'].split('_')])
                ss = int(float(wildcards['ss']))
                C = len(wildcards['a'].split('_'))
            rng = np.random.default_rng()

            for P_f, pi_f, s_f, y_f, nu_f, overall_y_f, overall_nu_f, fixed_M_f, random_M_f in zip(
                    params.P, params.pi, params.s, params.y, params.nu, params.overall_y, params.overall_nu,
                    params.fixed_M, params.random_M):
                #for P_f, pi_f, s_f, y_f, nu_f in zip(
                #    params.P, params.pi, params.s, params.y, params.nu):
                os.makedirs(os.path.dirname(P_f), exist_ok=True)
                # simulate cell type proportions
                P = rng.dirichlet(alpha=a, size=ss)
                np.savetxt(P_f, P, delimiter='\t')
                pi = np.mean(P, axis=0)
                np.savetxt(pi_f, pi, delimiter='\t')

                # estimate cov matrix S
                ## demeaning P 
                pd = P-pi
                ## covariance
                s = (pd.T @ pd)/ss
                #print(bmatrix(s))
                np.savetxt(s_f, s, delimiter='\t')

                # draw alpha / hom effect
                alpha_overall = rng.normal(loc=0, scale=math.sqrt(hom2), size=(ss))
                alpha = np.outer(alpha_overall, np.ones(C))

                # draw gamma (interaction)
                if wildcards.model != 'hom':
                    gamma = rng.multivariate_normal(np.zeros(C), V, size=ss) 
                    interaction = np.sum(P * gamma, axis=1)

                # draw residual error
                ## draw variance of residual error for each individual from gamma distribution \Gamma(k, theta)
                ## with mean = k * theta, var = k * theta^2, so theta = var / mean, k = mean / theta
                ## since mean = 0.25 and assume var = 0.01, we can get k and theta
                if mean_nu != 0:
                    theta = var_nu / mean_nu 
                    if var_nu < 0:
                        theta = theta * (-1)
                    k = mean_nu / theta 
                    ### variance of residual error for each individual
                    nu_overall = rng.gamma(k, scale=theta, size=ss)
                else:
                    nu_overall = np.zeros(ss)
                np.savetxt(overall_nu_f, nu_overall, delimiter='\t')
                ### variance of residual error for each CT is in the ratio of the inverse of CT proportion
                P_inv = 1 / P
                nu = P_inv * nu_overall.reshape(-1,1)

                # if mor than two CTs of one individual have low nu, 
                # hom model broken because of sigular variance matrix.
                # to solve the problem, regenerate NU
                threshold = 1e-10
                if mean_nu != 0:
                    i = 1
                    while np.any( np.sum(nu < threshold, axis=1) > 1 ):
                        nu_overall = rng.gamma(k, scale=theta, size=ss)
                        np.savetxt(overall_nu_f, nu_overall, delimiter='\t')
                        nu = P_inv * nu_overall.reshape(-1,1)
                        i += 1
                        if i > 5:
                            sys.exit('Generate NU failed!\n')

                # temporary use
                if var_nu < 0:
                    P2 = P**2
                    nu = (1 / P2.sum(axis=1)) * nu_overall
                    nu = np.repeat(nu.reshape(-1,1), C, axis=1)

                np.savetxt(nu_f, nu, delimiter='\t')

                ## draw residual error from normal distribution with variance drawn above
                delta_overall = rng.normal(np.zeros_like(nu_overall), np.sqrt(nu_overall))
                delta = rng.normal(np.zeros_like(nu), np.sqrt(nu))

                # generate pseudobulk
                if wildcards.model == 'hom':
                    y_overall = alpha_overall + P @ beta + delta_overall
                    y = alpha + np.outer(np.ones(ss), beta) + delta
                else:
                    y_overall = alpha_overall + P @ beta + interaction + delta_overall
                    y = alpha + np.outer(np.ones(ss), beta) + gamma + delta 

                # add fixed effect
                if 'fixed' in wildcards.keys():
                    levels = int(wildcards.fixed)
                    if levels > 0:
                        print(np.var(y_overall))
                        fixed = np.arange(levels)
                        fixed = fixed / np.std(fixed) # to set variance to 1
                        fixed_M = np.zeros( (len(y_overall), levels) )
                        j = 0
                        for i, chunk in enumerate( np.array_split(y_overall, levels)):
                            fixed_M[j:(j+len(chunk)), i] = 1
                            j = j + len(chunk)
                        # centralize fixed effect
                        fixed = fixed - np.mean(fixed_M @ fixed)

                        y_overall = y_overall + fixed_M @ fixed
                        print(np.var(y_overall))
                        y = y + (fixed_M @ fixed).reshape(-1,1)
                        # save fixed effect design matrix (get rid last column to avoid colinear)
                        np.savetxt(fixed_M_f, fixed_M[:,:-1], delimiter='\t')

                # add random effect
                if 'random' in wildcards.keys():
                    levels = int(wildcards.random)
                    if levels > 0:
                        print('variance', np.var(y_overall))
                        random_e = rng.normal(0, 1, levels)
                        random_M = np.zeros( (len(y_overall), levels) )
                        j = 0
                        for i, chunk in enumerate(np.array_split(y_overall, levels)):
                            random_M[j:(j+len(chunk)), i] = 1
                            j = j + len(chunk)
                        # centralize random effect
                        random_e = random_e - np.mean(random_M @ random_e)
                        ## double check it's centralized
                        if np.mean( random_M @ random_e) > 1e-3:
                            print('mean', np.mean( random_M @ random_e ) )
                            sys.exit('Centralization error!\n')

                        y_overall = y_overall + random_M @ random_e
                        print('variance', np.var(y_overall))
                        y = y + (random_M @ random_e).reshape(-1,1)
                        # save random effect design matrix
                        np.savetxt(random_M_f, random_M, delimiter='\t')

                # save
                np.savetxt(overall_y_f, y_overall, delimiter='\t')
                np.savetxt(y_f, y, delimiter='\t')

            with open(output.P, 'w') as f: f.write('\n'.join(params.P))
            with open(output.pi, 'w') as f: f.write('\n'.join(params.pi))
            with open(output.s, 'w') as f: f.write('\n'.join(params.s))
            with open(output.y, 'w') as f: f.write('\n'.join(params.y))
            with open(output.nu, 'w') as f: f.write('\n'.join(params.nu))
            with open(output.overall_y, 'w') as f: f.write('\n'.join(params.overall_y))
            with open(output.overall_nu, 'w') as f: f.write('\n'.join(params.overall_nu))
            with open(output.fixed_M, 'w') as f:
                if 'fixed' in wildcards.keys():
                    if int(wildcards.fixed) > 0:
                        f.write('\n'.join(params.fixed_M))
                    else:
                        f.write('NA')
                else:
                    f.write('NA')
            with open(output.random_M, 'w') as f:
                if 'random' in wildcards.keys():
                    if int(wildcards.random) > 0:
                        f.write('\n'.join(params.random_M))
                    else:
                        f.write('NA')
                else:
                    f.write('NA')

rule ctng_test:
    input:
        y = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/out.batch{{i}}',
    params:
        out = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep/out.npy',
        batch = lambda wildcards: ctng_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
    resources:
        mem_per_cpu = '5gb',
        time = '48:00:00',
    priority: 1
    script: 'bin/ctng_test.py'

rule ctng_MLnREML_aggReplications:
    input:
        s = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/estS.batch{i}.txt'
                for i in range(len(ctng_batches))],
        nu = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/nu.batch{i}.txt' 
                for i in range(len(ctng_batches))],
        P = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/P.batch{i}.txt'
                for i in range(len(ctng_batches))],
        pi = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/estPI.batch{i}.txt' 
                for i in range(len(ctng_batches))],
        out = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(ctng_batches))],
    output:
        s = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/estS.txt',
        nu = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/nu.txt',
        pi = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/estPI.txt',
        out = f'analysis/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/out.npy',
    script: 'bin/ctng_waldNlrt_aggReplications.py'

def ctng_agg_out_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, ong_params.loc[ong_params['model']==wildcards.model])
    #ong_params.to_csv(sys.stdout, sep='\t', index=False)
    #subspace.to_csv(sys.stdout, sep='\t', index=False)
    return expand('analysis/ctng/{{model}}/{params}/out.npy', params=subspace.instance_patterns)

def ctng_agg_truebeta_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, ong_params.loc[ong_params['model']==wildcards.model])
    return expand('analysis/ong/{{model}}/{params}/celltypebeta.txt', params=subspace.instance_patterns)

#def ctng_agg_trueV_subspace(wildcards):
#    subspace = get_subspace(wildcards.arg, ong_params.loc[ong_params['model']==wildcards.model])
#    return expand('analysis/ong/{{model}}/{params}/V.txt', params=subspace.instance_patterns)

use rule ong_MLestimates_subspace_plot as ctng_MLestimates_subspace_plot with:
    input:
        out = ctng_agg_out_subspace,
        beta = ctng_agg_truebeta_subspace,
        V = ong_agg_trueV_subspace,
    output:
        png = 'results/ctng/{model}/ML.AGG{arg}.png',

use rule ong_MLestimates_subspace_plot as ctng_MLestimates_subspace_plot2 with:
    input:
        out = ctng_agg_out_subspace,
        beta = ctng_agg_truebeta_subspace,
        V = ong_agg_trueV_subspace,
    output:
        png = 'results/ctng/{model}/ML.excluderareCT.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_excluderareCT_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,

use rule ong_MLwaldNlrt_subspace_plot as ctng_MLwaldNlrt_subspace_plot with:
    input:
        out = ctng_agg_out_subspace, 
    output:
        waldNlrt = 'results/ctng/{model}/ML.waldNlrt.AGG{arg}.png',

use rule ong_MLwaldNlrt_subspace_plot as ctng_MLwaldNlrt_subspace_plot2 with:
    input:
        out = ctng_agg_out_subspace, 
    output:
        waldNlrt = 'results/ctng/{model}/ML.excluderareCT.waldNlrt.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_excluderareCT_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,

use rule ong_HEestimates_AGGsubspace_plot as ctng_HEestimates_AGGsubspace_plot with:
    input:
        out = ctng_agg_out_subspace,
        V = ong_agg_trueV_subspace,
    output:
        png = 'results/ctng/{model}/HE.AGG{arg}.png',

use rule ong_HEwald_subspace_plot as ctng_HEwald_subspace_plot with:
    input:
        out = ctng_agg_out_subspace, 
    output:
        waldNlrt = 'results/ctng/{model}/HE.wald.AGG{arg}.png',

use rule ong_REMLestimates_subspace_plot as ctng_REMLestimates_subspace_plot with:
    input:
        out = ctng_agg_out_subspace,
        #beta = ctng_agg_truebeta_subspace,
        V = ong_agg_trueV_subspace,
    output:
        png = 'results/ctng/{model}/REML.AGG{arg}.png',

rule ctng_REMLestimates_subspace_plot_BSDposter:
    input:
        out = ctng_agg_out_subspace,
        V = ong_agg_trueV_subspace,
    output:
        png = 'results/ctng/{model}/REML.AGG{arg}.poster.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: "bin/ong_MLmodelestimates_subspace_plot.poster.py"

use rule ong_REMLestimates_subspace_plot as ctng_REMLestimates_subspace_plot2 with:
    input:
        out = ctng_agg_out_subspace,
        V = ong_agg_trueV_subspace,
    output:
        png = 'results/ctng/{model}/REML.excluderareCT.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_excluderareCT_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,

use rule ong_REMLwaldNlrt_subspace_plot as ctng_REMLwaldNlrt_subspace_plot with:
    input:
        out = ctng_agg_out_subspace, 
    output:
        waldNlrt = 'results/ctng/{model}/REML.waldNlrt.AGG{arg}.png',
    #script: "bin/ctng_REMLwaldNlrt_subspace_plot.py"

use rule ong_REMLwaldNlrt_subspace_plot as ctng_REMLwaldNlrt_subspace_plot2 with:
    input:
        out = ctng_agg_out_subspace, 
    output:
        waldNlrt = 'results/ctng/{model}/REML.excluderareCT.waldNlrt.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_excluderareCT_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,

def ctng_MLwaldNlrt_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ctng/{{model}}/ML.waldNlrt.AGG{arg}.png', arg=effective_args)

def ctng_MLestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ctng/{{model}}/ML.AGG{arg}.png', arg=effective_args)

def ctng_HEestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ctng/{{model}}/HE.AGG{arg}.png', arg=effective_args)

def ctng_HEwald_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ctng/{{model}}/HE.wald.AGG{arg}.png', arg=effective_args)

def ctng_REMLwaldNlrt_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ctng/{{model}}/REML.waldNlrt.AGG{arg}.png', arg=effective_args)

def ctng_REMLestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(ong_params.loc[ong_params['model']==wildcards.model])
    return expand('results/ctng/{{model}}/REML.AGG{arg}.png', arg=effective_args)

rule ctng_AGGarg:
    input:
        MLmodelestimates = ctng_MLestimates_AGGarg_fun,
        MLwaldNlrt = ctng_MLwaldNlrt_AGGarg_fun,
        HEestimates = ctng_HEestimates_AGGarg_fun,
        HEwald = ctng_HEwald_AGGarg_fun,
        REMLmodelestimates = ctng_REMLestimates_AGGarg_fun,
        REMLwaldNlrt = ctng_REMLwaldNlrt_AGGarg_fun,
        MLmodelestimates2 = 'results/ctng/{model}/ML.excluderareCT.AGGa.png',
        MLwaldNlrt2 = 'results/ctng/{model}/ML.excluderareCT.waldNlrt.AGGa.png',
        REMLmodelestimates2 = 'results/ctng/{model}/REML.excluderareCT.AGGa.png',
        REMLwaldNlrt2 = 'results/ctng/{model}/REML.excluderareCT.waldNlrt.AGGa.png',
    output:
        flag = touch('staging/ctng/{model}/all.flag'),

#use rule ong_mainfig_LRT as ctng_mainfig_LRT with:
#    input:
#        hom_ss = expand('analysis/ctng/hom/{params}/out.npy',
#                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
#        iid_ss = expand('analysis/ctng/iid/{params}/out.npy',
#                params=get_subspace('ss', ong_params.loc[ong_params['model']=='iid']).instance_patterns),
#        free_ss = expand('analysis/ctng/free/{params}/out.npy',
#                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
#        hom_a = expand('analysis/ctng/hom/{params}/out.npy',
#                params=get_subspace('a', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
#        iid_a = expand('analysis/ctng/iid/{params}/out.npy',
#                params=get_subspace('a', ong_params.loc[ong_params['model']=='iid']).instance_patterns),
#        free_a = expand('analysis/ctng/free/{params}/out.npy',
#                params=get_subspace('a', ong_params.loc[ong_params['model']=='free']).instance_patterns),
#        iid_vc = expand('analysis/ctng/iid/{params}/out.npy',
#                params=get_subspace('vc', ong_params.loc[ong_params['model']=='iid']).instance_patterns),
#        free_vc = expand('analysis/ctng/free/{params}/out.npy',
#                params=get_subspace('vc', ong_params.loc[ong_params['model']=='free']).instance_patterns),
#        free_V_diag = expand('analysis/ctng/free/{params}/out.npy',
#                params=get_subspace('V_diag', ong_params.loc[ong_params['model']=='free']).instance_patterns),
#    output:
#        png = 'results/ctng/mainfig.LRT.png',

rule ctng_all:
    input:
        flag = expand('staging/ctng/{model}/all.flag', model=['hom','free', 'full']),
        #png = 'results/ctng/mainfig.LRT.png',

rule ong_test_remlJK:
    input:
        y = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/out.remlJK.batch{{i}}',
    params:
        out = f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/rep/out.remlJK.npy',
        batch = lambda wildcards: ong_batches[int(wildcards.i)],
        ML = False,
        REML = True,
        Free_reml_only = True,
        Free_reml_jk = True,
        HE = False,
    resources:
        mem_per_cpu = '10gb',
    script: 'bin/ong_test.py'

rule ong_test_remlJK_aggReplications:
    input:
        out = [f'staging/ong/{{model}}/{ong_paramspace.wildcard_pattern}/out.remlJK.batch{i}' 
                for i in range(len(ctng_batches))],
    output:
        out = f'analysis/ong/{{model}}/{ong_paramspace.wildcard_pattern}/out.remlJK.npy',
    script: 'bin/mergeBatches.py'

rule ctng_test_remlJK:
    input:
        y = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/out.remlJK.batch{{i}}',
    params:
        out = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep/out.remlJK.npy',
        batch = lambda wildcards: ctng_batches[int(wildcards.i)],
        ML = False,
        REML = True,
        Free_reml_only = True,
        Free_reml_jk = True,
        HE = False,
    resources:
        mem_per_cpu = '10gb',
        time = '200:00:00',
    priority: 1
    script: 'bin/ctng_test.py'

rule ctng_test_remlJK_aggReplications:
    input:
        out = [f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/out.remlJK.batch{i}' 
                for i in range(len(ctng_batches))],
    output:
        out = f'analysis/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/out.remlJK.npy',
    script: 'bin/mergeBatches.py'

rule paper_ongNctng_power:
    input:
        ong_hom = expand('analysis/ong/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        ong_free = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ong_hom_remlJK = expand('analysis/ong/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        #params=get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        ong_free_remlJK = expand('analysis/ong/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        #params=get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        ctng_hom = expand('analysis/ctng/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        ctng_free = expand('analysis/ctng/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ctng_hom_remlJK = expand('analysis/ctng/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        #params=get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        ctng_free_remlJK = expand('analysis/ctng/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        #params=get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
    output:
        png = 'results/ctng/ongNctng.power.paper.png',
    params: 
        hom = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='hom'])['ss']),
        free = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='free'])['ss']),
        hom_remlJK = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='hom'])['ss']),
        #hom_remlJK = np.array(get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)])['ss']),
        free_remlJK = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='free'])['ss']),
        #free_remlJK = np.array(get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)])['ss']),
        plot_order = ong_plot_order,
    script: 'bin/paper_ongNctng_power.py'

rule paper_ongNctng_power_ASHG:
    input:
        ong_hom = expand('analysis/ong/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        ong_free = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ong_hom_remlJK = expand('analysis/ong/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        ong_free_remlJK = expand('analysis/ong/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        ctng_hom = expand('analysis/ctng/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        ctng_free = expand('analysis/ctng/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ctng_hom_remlJK = expand('analysis/ctng/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        ctng_free_remlJK = expand('analysis/ctng/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
    output:
        png = 'results/ctng/ongNctng.power.paper.ASHG.png',
    params: 
        hom = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='hom'])['ss']),
        free = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='free'])['ss']),
        hom_remlJK = np.array(get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)])['ss']),
        free_remlJK = np.array(get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)])['ss']),
        plot_order = ong_plot_order,
    script: 'bin/paper_ongNctng_power.ASHG.py'

rule paper_ongNctng_power_ZJU:
    input:
        ong_hom = expand('analysis/ong/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        ong_free = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ong_hom_remlJK = expand('analysis/ong/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        ong_free_remlJK = expand('analysis/ong/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        ctng_hom = expand('analysis/ctng/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        ctng_free = expand('analysis/ctng/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ctng_hom_remlJK = expand('analysis/ctng/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        ctng_free_remlJK = expand('analysis/ctng/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
    output:
        png = 'results/ctng/ongNctng.power.paper.ZJU.png',
    params: 
        hom = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='hom'])['ss']),
        free = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='free'])['ss']),
        hom_remlJK = np.array(get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)])['ss']),
        free_remlJK = np.array(get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)])['ss']),
        plot_order = ong_plot_order,
    script: 'bin/paper_ongNctng_power.ZJU.py'

rule paper_ctng_power:
    input:
        hom_ss = expand('analysis/ctng/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        hom_ss_remlJK = expand('analysis/ctng/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        free_ss = expand('analysis/ctng/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        free_ss_remlJK = expand('analysis/ctng/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        hom_a = expand('analysis/ctng/hom/{params}/out.npy',
                params=get_subspace('a', ong_params.loc[ong_params['model']=='hom']).instance_patterns),
        hom_a_remlJK = expand('analysis/ctng/hom/{params}/out.remlJK.npy',
                params=get_subspace('a', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        free_a = expand('analysis/ctng/free/{params}/out.npy',
                params=get_subspace('a', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        free_a_remlJK = expand('analysis/ctng/free/{params}/out.remlJK.npy',
                params=get_subspace('a', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
        free_vc = expand('analysis/ctng/free/{params}/out.npy',
                params=get_subspace('vc', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        free_vc_remlJK = expand('analysis/ctng/free/{params}/out.remlJK.npy',
                params=get_subspace('vc', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)]).instance_patterns),
    output:
        png = 'results/ctng/power.paper.supp.png',
    params: 
        arg_ss = 'ss',
        hom_ss = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='hom'])['ss']),
        free_ss = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='free'])['ss']),
        hom_ss_remlJK = np.array(get_subspace('ss', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)])['ss']),
        free_ss_remlJK = np.array(get_subspace('ss', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)])['ss']),
        arg_a = 'a',
        hom_a = np.array(get_subspace('a', ong_params.loc[ong_params['model']=='hom'])['a']),
        free_a = np.array(get_subspace('a', ong_params.loc[ong_params['model']=='free'])['a']),
        hom_a_remlJK = np.array(get_subspace('a', ong_params.loc[(ong_params['model']=='hom') & (ong_params['ss'].astype('float')<=100)])['a']),
        free_a_remlJK = np.array(get_subspace('a', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)])['a']),
        arg_vc = 'vc',
        free_vc = np.array(get_subspace('vc', ong_params.loc[ong_params['model']=='free'])['vc']),
        free_vc_remlJK = np.array(get_subspace('vc', ong_params.loc[(ong_params['model']=='free') & (ong_params['ss'].astype('float')<=100)])['vc']),
        plot_order = ong_plot_order,
    script: 'bin/paper_ctng_power.py'

rule paper_ongNctng_estimates_ss:
    input:
        ong_free = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ctng_free = expand('analysis/ctng/free/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        V = expand('analysis/ong/free/{params}/V.txt',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='free']).instance_patterns),
    output:
        png = 'results/ctng/ongNctng.estimate.ss.paper.supp.png',
    params:
        free = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='free'])['ss']),
        plot_order = ong_plot_order,
        #subspace = lambda wildcards: get_subspace(wildcards.arg,
        #        ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        #colorpalette = colorpalette,
        pointcolor = pointcolor,
        #mycolors = mycolors,
    script: 'bin/paper_ongNctng_estimates_ss.py'

rule paper_ongNctng_estimates_a:
    input:
        ong_free = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('a', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ctng_free = expand('analysis/ctng/free/{params}/out.npy',
                params=get_subspace('a', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        V = expand('analysis/ong/free/{params}/V.txt',
                params=get_subspace('a', ong_params.loc[ong_params['model']=='free']).instance_patterns),
    output:
        png = 'results/ctng/ongNctng.estimate.a.paper.supp.png',
    params:
        free = np.array(get_subspace('a', ong_params.loc[ong_params['model']=='free'])['a']),
        plot_order = ong_plot_order,
        #subspace = lambda wildcards: get_subspace(wildcards.arg,
        #        ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        #colorpalette = colorpalette,
        pointcolor = pointcolor,
        #mycolors = mycolors,
    script: 'bin/paper_ongNctng_estimates_a.py'

rule paper_ongNctng_estimates_vc:
    input:
        ong_free = expand('analysis/ong/free/{params}/out.npy',
                params=get_subspace('vc', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ctng_free = expand('analysis/ctng/free/{params}/out.npy',
                params=get_subspace('vc', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        V = expand('analysis/ong/free/{params}/V.txt',
                params=get_subspace('vc', ong_params.loc[ong_params['model']=='free']).instance_patterns),
        ong_hom = expand('analysis/ong/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom'].iloc[[0]]).instance_patterns),
        ctng_hom = expand('analysis/ctng/hom/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom'].iloc[[0]]).instance_patterns),
        V_hom = expand('analysis/ong/hom/{params}/V.txt',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='hom'].iloc[[0]]).instance_patterns),
    output:
        png = 'results/ctng/ongNctng.estimate.vc.paper.supp.png',
    params:
        arg = 'vc',
        free = np.array(get_subspace('vc', ong_params.loc[ong_params['model']=='free'])['vc']),
        plot_order = ong_plot_order,
        #subspace = lambda wildcards: get_subspace(wildcards.arg,
        #        ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        #colorpalette = colorpalette,
        pointcolor = pointcolor,
        #mycolors = mycolors,
    script: 'bin/paper_ongNctng_estimates_vc.py'

rule paper_ongNctng_estimates_ss_full:
    input:
        ong_full = expand('analysis/ong/full/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='full']).instance_patterns),
        ctng_full = expand('analysis/ctng/full/{params}/out.npy',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='full']).instance_patterns),
        V = expand('analysis/ong/full/{params}/V.txt',
                params=get_subspace('ss', ong_params.loc[ong_params['model']=='full']).instance_patterns),
    output:
        png = 'results/ctng/ongNctng.estimate.ss.full.paper.supp.png',
    params:
        full = np.array(get_subspace('ss', ong_params.loc[ong_params['model']=='full'])['ss']),
        plot_order = ong_plot_order,
        #subspace = lambda wildcards: get_subspace(wildcards.arg,
        #        ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        #colorpalette = colorpalette,
        pointcolor = pointcolor,
        #mycolors = mycolors,
        vc = get_subspace('ss', ong_params.loc[ong_params['model']=='full'])['vc'][0],
    script: 'bin/paper_ongNctng_estimates_ss_full.py'


################ test the degree of freedom for ctng in HE
rule ctng_test_HEdf:
    input:
        y = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/tmp/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/out.{{n_equation}}.batch{{i}}',
    params:
        out = f'staging/tmp/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/rep/out.{{n_equation}}.npy',
        batch = lambda wildcards: ctng_batches[int(wildcards.i)],
        ML = False,
        REML = False,
        HE = True,
        HE_free_only = True,
    resources:
        mem_per_cpu = '1gb',
        time = '48:00:00',
    priority: 1
    script: "bin/ctng_test.HEdf.py"

rule ctng_test_HEdf_mergeBatches:
    input:
        out = [f'staging/tmp/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/out.{{n_equation}}.batch{i}'
                for i in range(len(ctng_batches))],
    output:
        out = f'staging/tmp/ctng/{{model}}/{ong_paramspace.wildcard_pattern}/out.{{n_equation}}.npy',
    script: 'bin/mergeBatches.py'

def ctng_HEdf_agg_out_subsapce(wildcards):
    subspace = get_subspace('ss', ong_params.loc[ong_params['model']==wildcards.model])
    return expand('staging/tmp/ctng/{{model}}/{params}/out.{{n_equation}}.npy', params=subspace.instance_patterns)

rule ctng_HEdf_subsapce:
    input:
        out = ctng_HEdf_agg_out_subsapce,
    output:
        waldNlrt = 'results/tmp/ctng/{model}/{n_equation}.png',
    params:
        subspace = lambda wildcards: get_subspace('ss',
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: 'bin/ctng_HEdf_subsapce.py'

rule ctng_HE_tmp:
    input:
        expand('results/tmp/ctng/{{model}}/{n_equation}.png', n_equation=['ind', 'indXct']),
    output:
        touch('staging/tmp/ctng/{model}.flag'),
#####################################################################################
# Add noise to Nu in ctng 
#####################################################################################
# par
ctng_uncertainNU_replicates = 1000
ctng_uncertainNU_batch_no = 100
ctng_uncertainNU_batches = np.array_split(range(ctng_replicates), ctng_batch_no)

## declare a dataframe to be a paramspace
ctng_uncertainNU_params = pd.read_table("ctng.uncertainNU.params.txt", dtype="str", comment='#', na_filter=False)
if ctng_uncertainNU_params.shape[0] != ctng_uncertainNU_params.drop_duplicates().shape[0]:
    sys.exit('Duplicated parameters!\n')
par_columns = list(ctng_uncertainNU_params.columns)
par_columns.remove('model') # columns after removing 'model'
ctng_uncertainNU_paramspace = Paramspace(ctng_uncertainNU_params[par_columns], filename_params="*")

use rule ong_celltype_expectedPInSnBETAnV as ctng_uncertainNU_celltype_expectedPInSnBETAnV with:
    output:
        pi = f'analysis/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/PI.txt',
        s = f'analysis/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/S.txt',
        beta = f'analysis/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/V.txt',
    params:
        simulation=ctng_uncertainNU_paramspace.instance,

localrules: ctng_uncertainNU_generatedata_batch
rule ctng_uncertainNU_generatedata_batch:
    output: touch(f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/generatedata.batch')

for _, batch in enumerate(ctng_uncertainNU_batches):
    use rule ctng_generatedata_batch0 as ctng_uncertainNU_generatedata_batchx with:
        input:
            flag = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/generatedata.batch',
            beta = f'analysis/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/celltypebeta.txt',
            V = f'analysis/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/V.txt',
        output:
            P = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/P.batch{_}.txt',
            pi = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/estPI.batch{_}.txt',
            s = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/estS.batch{_}.txt',
            nu = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/nu.batch{_}.txt',
            y = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/pseudobulk.batch{_}.txt',
            overall_nu = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/overall.nu.batch{_}.txt',
            overall_y = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/overall.pseudobulk.batch{_}.txt',
            fixed_M = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{_}.txt',
            random_M = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{_}.txt',
        params:
            P = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep{i}/P.txt' for i in batch],
            pi = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep{i}/estPI.txt' 
                    for i in batch],
            s = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep{i}/estS.txt' 
                    for i in batch],# sample prop cov matrix
            nu = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep{i}/nu.txt' for i in batch],
            y = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep{i}/pseudobulk.txt' for i in batch],
            overall_nu = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep{i}/overall.nu.txt' 
                    for i in batch],
            overall_y = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep{i}/overall.pseudobulk.txt' 
                    for i in batch],
            fixed_M = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep{i}/overall.fixedeffectmatrix.txt' 
                    for i in batch],
            random_M = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep{i}/overall.randomeffectmatrix.txt'
                    for i in batch],
        resources:
            burden = 20,
        name: f'ctng_uncertainNU_generatedata_batch{_}'

rule ctng_uncertainNU_addUncertainty:
    input:
        nu = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        nu = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/nu.uncertain.batch{{i}}.txt',
    run:
        output_nus = open(output.nu, 'w')
        rng = np.random.default_rng()
        for line in open(input.nu):
            nu_f = line.strip()
            uncertain_nu_f = nu_f+'.uncertain'
            nu = np.loadtxt(nu_f)
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

use rule ctng_test as ctng_uncertainNU_test with:
    input:
        y = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/nu.uncertain.batch{{i}}.txt',
    output:
        out = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/out.batch{{i}}',
    params:
        out = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep/out.npy',
        batch = lambda wildcards: ctng_uncertainNU_batches[int(wildcards.i)],
        ML = False,
        REML = True,
        HE = True,
    resources:
        mem_per_cpu = '5gb',
        time = '48:00:00',

rule ctng_uncertainNU_test_aggReplications:
    input:
        out = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/out.batch{i}'
                for i in range(len(ctng_uncertainNU_batches))],
    output:
        out = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/out.npy',
    script: "bin/mergeBatches.py"

use rule ctng_test as ctng_uncertainNU_remlJK_test with:
    input:
        y = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/nu.uncertain.batch{{i}}.txt',
    output:
        out = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/out.remlJK.batch{{i}}',
    params:
        out = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/rep/out.remlJK.npy',
        batch = lambda wildcards: ctng_uncertainNU_batches[int(wildcards.i)],
        ML = False,
        REML = True,
        Free_reml_only = True,
        Free_reml_jk = True,
        HE = False,
    resources:
        mem_per_cpu = '6gb',
        time = '48:00:00',

rule ctng_uncertainNU_remlJK_test_aggReplications:
    input:
        out = [f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/out.remlJK.batch{i}'
                for i in range(len(ctng_uncertainNU_batches))],
    output:
        out = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/out.remlJK.npy',
    script: "bin/mergeBatches.py"

rule ctng_uncertainNU_merge:
    input:
        out = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/out.npy',
        out2 = f'staging/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/out.remlJK.npy',
    output:
        out = f'analysis/ctng_uncertainNU/{{model}}/{ctng_uncertainNU_paramspace.wildcard_pattern}/out.npy',
    run:
        out = np.load(input.out, allow_pickle=True).item()
        out2 = np.load(input.out2, allow_pickle=True).item()
        out['reml_JK'] = out2['reml']
        np.save(output.out, out)

def ctng_uncertainNU_aggparams(wildcards):
    ctng_uncertainNU_params = pd.read_table("ctng.uncertainNU.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctng_uncertainNU_params = ctng_uncertainNU_params.loc[ctng_uncertainNU_params['model'] == wildcards.model]
    par_columns = list(ctng_uncertainNU_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctng_uncertainNU_paramspace = Paramspace(ctng_uncertainNU_params[par_columns], filename_params="*")
    return expand('analysis/ctng_uncertainNU/{{model}}/{params}/out.npy', 
            params=ctng_uncertainNU_paramspace.instance_patterns)

def ctng_uncertainNU_V_aggparams(wildcards):
    ctng_uncertainNU_params = pd.read_table("ctng.uncertainNU.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctng_uncertainNU_params = ctng_uncertainNU_params.loc[ctng_uncertainNU_params['model'] == wildcards.model]
    par_columns = list(ctng_uncertainNU_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctng_uncertainNU_paramspace = Paramspace(ctng_uncertainNU_params[par_columns], filename_params="*")
    return expand('analysis/ctng_uncertainNU/{{model}}/{params}/V.txt', 
            params=ctng_uncertainNU_paramspace.instance_patterns)

rule ctng_uncertainNU_V:
    input:
        V = ctng_uncertainNU_V_aggparams,
        out = ctng_uncertainNU_aggparams,
    output:
        png = 'results/ctng_uncertainNU/{model}/V.png',
    params:
        labels = ['0_2_5', '1_2_5', '1_2_4', '1_2_3', '1_2_2'],
    script: 'bin/ctng_uncertainNU_V.py'

rule ctng_uncertainNU_all:
    input:
        png = expand('results/ctng_uncertainNU/{model}/V.png', model=['hom3','hom4','hom5']),

#####################################################################################
# to test scripts: compare CTNG and ONG estimates 
#####################################################################################

ctngVSong_params = pd.read_table("ctngVSong.params.txt", dtype="str", comment='#', na_filter=False)
if ctngVSong_params.shape[0] != ctngVSong_params.drop_duplicates().shape[0]:
    sys.exit('Duplicated parameters!\n')
par_columns = list(ctngVSong_params.columns)
par_columns.remove('model') # columns after removing 'model'
ctngVSong_paramspace = Paramspace(ctngVSong_params[par_columns], filename_params="*")

ctngVSong_replicates = 1000
ctngVSong_batchsize = 2
ctngVSong_batches = [range(i, min(i+ctngVSong_batchsize, ctngVSong_replicates)) 
        for i in range(0, ctngVSong_replicates, ctngVSong_batchsize)]

use rule ong_celltype_expectedPInSnBETAnV as ctngVSong_celltype_expectedPInSnBETAnV with:
    output:
        pi = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/PI.txt',
        s = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/S.txt',
        beta = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/V.txt',
    params:
        simulation=ctngVSong_paramspace.instance,

localrules: ctngVSong_generatedata_batch
rule ctngVSong_generatedata_batch:
    output: touch(f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/generatedata.batch')

for _, batch in enumerate(ctngVSong_batches):
    use rule ctng_generatedata_batch0 as ctngVSong_generatedata_batchx with:
        input:
            flag = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/generatedata.batch',
            beta = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/celltypebeta.txt',
            V = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/V.txt',
        output:
            P = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/P.batch{_}.txt',
            pi = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/estPI.batch{_}.txt',
            s = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/estS.batch{_}.txt',
            nu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/nu.batch{_}.txt',
            y = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/pseudobulk.batch{_}.txt',
            overall_nu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.nu.batch{_}.txt',
            overall_y = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.pseudobulk.batch{_}.txt',
            fixed_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{_}.txt',
            random_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{_}.txt',
        params:
            P = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep{i}/P.txt' 
                    for i in batch],
            pi = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep{i}/estPI.txt' 
                    for i in batch],
            s = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep{i}/estS.txt' 
                    for i in batch],# sample prop cov matrix
            nu = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep{i}/nu.txt' 
                    for i in batch],
            y = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep{i}/pseudobulk.txt' 
                    for i in batch],
            overall_nu = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep{i}/overall.nu.txt' 
                    for i in batch],
            overall_y = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep{i}/overall.pseudobulk.txt' 
                    for i in batch],
            fixed_M = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep{i}/overall.fixedeffectmatrix.txt' 
                    for i in batch],
            random_M = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep{i}/overall.randomeffectmatrix.txt'
                    for i in batch],
            sim = ctngVSong_paramspace.instance,
        resources: 
            burden = 20,
        name: f'ctngVSong_generatedata_batch{_}'

use rule ong_test as ctngVSong_ongtest with:
    input:
        y = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.nu.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.out.batch{{i}}',
    params:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep/overall.out.npy',
        batch = lambda wildcards: ctngVSong_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        HE_as_initial = False,
    resources:
        mem = '5gb',
        time = '48:00:00',

rule ctngVSong_ongAggReplications:
    input:
        out = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.out.batch{i}' 
                for i in range(len(ctngVSong_batches))],
    output:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.out.npy',
    script: "bin/mergeBatches.py"

use rule ctng_test as ctngVSong_ctngtest with:
    input:
        y = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/out.batch{{i}}',
    params:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep/out.npy',
        batch = lambda wildcards: ctngVSong_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        HE_as_initial = False,
    resources:
        mem = '5gb',
        time = '48:00:00',

rule ctngVSong_ctng_aggReplications:
    input:
        out = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(ctngVSong_batches))],
    output:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/out.npy',
    script: "bin/mergeBatches.py"

rule ctngVSong_ongtest_withCovars:
    input:
        y = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.nu.batch{{i}}.txt',
        fixed_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{{i}}.txt',
        random_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.covars.out.batch{{i}}',
    params:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep/overall.covars.out.npy',
        batch = lambda wildcards: ctngVSong_batches[int(wildcards.i)],
        HE_as_initial = False,
    resources:
        mem = '5gb',
        time = '48:00:00',
    script: 'bin/ctngVSong_ongtest.py'

rule ctngVSong_ong_withCovars_AggReplications:
    input:
        out = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.covars.out.batch{i}' 
                for i in range(len(ctngVSong_batches))],
    output:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.covars.out.npy',
    script: "bin/mergeBatches.py"

rule ctngVSong_ongtest_withCovars_usingHEinit:
    input:
        y = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.nu.batch{{i}}.txt',
        fixed_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{{i}}.txt',
        random_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.HEinit.out.batch{{i}}',
    params:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep/overall.HEinit.out.npy',
        batch = lambda wildcards: ctngVSong_batches[int(wildcards.i)],
        HE_as_initial = True,
    resources:
        mem = '5gb',
        time = '48:00:00',
    script: 'bin/ctngVSong_ongtest.py'

rule ctngVSong_ong_withCovars_usingHEinit_AggReplications:
    input:
        out = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.HEinit.out.batch{i}' 
                for i in range(len(ctngVSong_batches))],
    output:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.HEinit.out.npy',
    script: "bin/mergeBatches.py"

rule ctngVSong_ctngtest_withCovars:
    input:
        y = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.nu.batch{{i}}.txt',
        ctnu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        fixed_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{{i}}.txt',
        random_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.batch{{i}}',
    params:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep/covars.out.npy',
        batch = lambda wildcards: ctngVSong_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        HE_as_initial = False,
    resources: 
        mem = '8gb',
        time = '48:00:00',
    script: 'bin/ctngVSong_ctngtest.py'

rule ctngVSong_ctng_withCovars_aggReplications:
    input:
        out = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.batch{i}' 
                for i in range(len(ctngVSong_batches))],
    output:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.npy',
    script: "bin/mergeBatches.py"

rule ctngVSong_ctngtest_withCovars_usingHEinit:
    input:
        y = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.nu.batch{{i}}.txt',
        ctnu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        fixed_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{{i}}.txt',
        random_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/HEinit.out.batch{{i}}',
    params:
        out = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/rep/HEinit.out.npy',
        batch = lambda wildcards: ctngVSong_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        HE_as_initial = True,
    resources: 
        mem = '8gb',
        time = '48:00:00',
    script: 'bin/ctngVSong_ctngtest.py'

rule ctngVSong_ctng_withCovars_usingHEinit_aggReplications:
    input:
        out = [f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/HEinit.out.batch{i}' 
                for i in range(len(ctngVSong_batches))],
    output:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/HEinit.out.npy',
    script: "bin/mergeBatches.py"

rule ctngVSong_hom:
    input:
        ong = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.HEinit.out.npy',
        ctng = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/HEinit.out.npy',
        ong_covar = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.covars.out.npy',
        ctng_covar = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.npy',
    output:
        png = f'results/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/ctngVSong.hom.png',
    script: 'bin/ctngVSong_hom.py'

rule ctngVSong_V:
    input:
        ong = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.HEinit.out.npy',
        ctng = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/HEinit.out.npy',
        ong_covar = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.covars.out.npy',
        ctng_covar = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.npy',
    output:
        png = f'results/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/ctngVSong.V.png',
    script: 'bin/ctngVSong_V.py'

def ctngVSong_agg_hom_model_fun(wildcards):
    ctngVSong_params = pd.read_table("ctngVSong.params.txt", dtype="str", comment='#', na_filter=False)
    ctngVSong_params = ctngVSong_params.loc[ctngVSong_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_paramspace = Paramspace(ctngVSong_params[par_columns], filename_params="*")
    return expand('results/ctngVSong/{{model}}/{params}/ctngVSong.hom.png', 
            params=ctngVSong_paramspace.instance_patterns)

def ctngVSong_agg_V_model_fun(wildcards):
    ctngVSong_params = pd.read_table("ctngVSong.params.txt", dtype="str", comment='#', na_filter=False)
    ctngVSong_params = ctngVSong_params.loc[ctngVSong_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_paramspace = Paramspace(ctngVSong_params[par_columns], filename_params="*")
    return expand('results/ctngVSong/{{model}}/{params}/ctngVSong.V.png', 
            params=ctngVSong_paramspace.instance_patterns)

rule ctngVSong_agg_hom_model:
    input: 
        hom = ctngVSong_agg_hom_model_fun,
        V = ctngVSong_agg_V_model_fun,
    output: touch('staging/ctngVSong/{model}/ctngVSong.flag'),

rule ctngVSong_all:
    input:
        ctngVSong = expand('staging/ctngVSong/{model}/ctngVSong.flag', model=['hom','free']),

# CTNG p value: permute CT for each donor independently 
# permute projected Y rather then original, to keep fixed effects
rule ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect:
    input:
        y = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.nu.batch{{i}}.txt',
        ctnu = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        fixed_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{{i}}.txt',
        random_M = f'staging/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong/{{model}}/permuteCT_holdfixedeffect/{ctngVSong_paramspace.wildcard_pattern}/covars.out.batch{{i}}',
    params:
        out = f'staging/ctngVSong/{{model}}/permuteCT_holdfixedeffect/{ctngVSong_paramspace.wildcard_pattern}/rep/covars.out.npy',
        batch = lambda wildcards: ctngVSong_batches[int(wildcards.i)],
    script: 'bin/ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect.py'

rule ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_aggReplications:
    input:
        out = [f'staging/ctngVSong/{{model}}/permuteCT_holdfixedeffect/{ctngVSong_paramspace.wildcard_pattern}/covars.out.batch{i}' 
                for i in range(len(ctngVSong_batches))],
    output:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.permute.holdfixedeffect.npy',
    script: 'bin/mergeBatches.py'

rule ctngVSong_ctng_withCovars_pvalueplot_permuteCT_holdfixedeffect:
    input:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.permute.holdfixedeffect.npy',
    output:
        png = f'results/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.permute.holdfixedeffect.png',
    script: 'bin/cuomo_ctng_HEpvalue_acrossmodel_plot.py'

rule ctngVSong_ctng_withCovars_permutationDistribution_permuteCT_holdfixedeffect:
    input:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.permute.holdfixedeffect.npy',
    output:
        png = f'results/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/permutationDistribution.permute.holdfixedeffect.png',
    script: 'bin/ctngVSong_NUfromCuomo_ctng_permutationDistribution_permuteCT_holdfixedeffect.py'

rule ctngVSong_ctng_withCovars_estimatesplot_permuteCT_holdfixedeffect:
    input:
        out = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.permute.holdfixedeffect.npy',
    output:
        png = f'results/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/estimates.permute.holdfixedeffect.png',
    script: 'bin/ctngVSong_ctng_withCovars_estimatesplot_permuteCT_holdfixedeffect.py'

def ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_agg_model_fun(wildcards):
    ctngVSong_params = pd.read_table("ctngVSong.params.txt", dtype="str", comment='#', na_filter=False)
    ctngVSong_params = ctngVSong_params.loc[ctngVSong_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_paramspace = Paramspace(ctngVSong_params[par_columns], filename_params="*")
    return expand('results/ctngVSong/{{model}}/{params}/covars.out.permute.holdfixedeffect.png', 
            params=ctngVSong_paramspace.instance_patterns)

def ctngVSong_ctng_withCovars_permutationDistribution_permuteCT_holdfixedeffect_agg_model_fun(wildcards):
    ctngVSong_params = pd.read_table("ctngVSong.params.txt", dtype="str", comment='#', na_filter=False)
    ctngVSong_params = ctngVSong_params.loc[ctngVSong_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_paramspace = Paramspace(ctngVSong_params[par_columns], filename_params="*")
    return expand('results/ctngVSong/{{model}}/{params}/permutationDistribution.permute.holdfixedeffect.png', 
            params=ctngVSong_paramspace.instance_patterns)

def ctngVSong_ctng_withCovars_estimatesplot_permuteCT_holdfixedeffect_agg_model_fun(wildcards):
    ctngVSong_params = pd.read_table("ctngVSong.params.txt", dtype="str", comment='#', na_filter=False)
    ctngVSong_params = ctngVSong_params.loc[ctngVSong_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_paramspace = Paramspace(ctngVSong_params[par_columns], filename_params="*")
    return expand('results/ctngVSong/{{model}}/{params}/estimates.permute.holdfixedeffect.png', 
            params=ctngVSong_paramspace.instance_patterns)

rule ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_agg_model:
    input:
        png = ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_agg_model_fun,
        png2 = ctngVSong_ctng_withCovars_permutationDistribution_permuteCT_holdfixedeffect_agg_model_fun,
        png3 = ctngVSong_ctng_withCovars_estimatesplot_permuteCT_holdfixedeffect_agg_model_fun,
    output: touch('staging/ctngVSong/{model}/permuteCT_holdfixedeffect/ctngVSong.flag'),

rule ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_all:
    input:
        ctngVSong = expand('staging/ctngVSong/{model}/permuteCT_holdfixedeffect/ctngVSong.flag', 
                model=['hom','free']),

rule ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_confirmestimates:
    input:
        original = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.npy',
        permute = f'analysis/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/covars.out.permute.holdfixedeffect.npy',
    output:
        out = f'results/ctngVSong/{{model}}/{ctngVSong_paramspace.wildcard_pattern}/confirmestimates.tmp.txt',
    script: 'bin/ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_confirmestimates.py'

def ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_confirmestimates_agg_model_fun(wildcards):
    ctngVSong_params = pd.read_table("ctngVSong.params.txt", dtype="str", comment='#', na_filter=False)
    ctngVSong_params = ctngVSong_params.loc[ctngVSong_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_paramspace = Paramspace(ctngVSong_params[par_columns], filename_params="*")
    return expand('results/ctngVSong/{{model}}/{params}/confirmestimates.tmp.txt', 
            params=ctngVSong_paramspace.instance_patterns)

rule ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_confirmestimates_agg_model:
    input:
        out = ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_confirmestimates_agg_model_fun,
    output: touch('staging/ctngVSong/{model}/permuteCT_holdfixedeffect/ctngVSong.tmp.flag'),

rule ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect_confirmestimates_all:
    input:
        ctngVSong = expand('staging/ctngVSong/{model}/permuteCT_holdfixedeffect/ctngVSong.tmp.flag', 
                model=['hom','free']),




#########################################################################################
# Cuomo et al 2020 Nature Communications
#########################################################################################
# data
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
                sys.exit('Wrong order!\n')
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

#use rule cuomo_day_pseudobulk_log as cuomo_day_pseudobulk_raw with:
#    input:
#        meta = 'analysis/cuomo/data/meta.txt',
#        counts = 'data/cuomo2020natcommun/raw_counts.csv.gz',
#    output:
#        y = 'analysis/cuomo/data/raw/day.raw.pseudobulk.gz', # donor - day * gene
#        nu = 'analysis/cuomo/data/raw/day.raw.nu.gz', # donor - day * gene

#rule cuomo_day_countspercell_dist:
#    input:
#        log = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz', # donor - day * gene
#        raw = 'data/cuomo2020natcommun/raw_counts.csv.gz', # donor - day * gene
#    output:
#        png = 'analysis/cuomo/data/counts.png',
#    script: 'bin/cuomo_day_pseudobulk_dist.py'

#########################
# replicate GO enrichment
#########################
localrules: cuomo_replicateGO_copy
rule cuomo_replicateGO_copy:
    input: 
        clusters = 'data/cuomo2020natcommun/suppdata5.txt',
    output:
        clusters = 'staging/replicateGO/suppdata5.{i}.txt',
    shell:
        'cp {input.clusters} {output.clusters}'

rule cuomo_replicateGO_enrichment:
    input:
        clusters = 'staging/replicateGO/suppdata5.{i}.txt',
    output:
        go = 'staging/replicateGO/go.{i}.txt',
    resources:
        mem = '10gb',
    shell:
        '''
        module load R/4.0.3
        Rscript bin/cuomo_replicateGO.R {input.clusters} {wildcards.i} {output.go}
        '''

localrules: cuomo_replicateGO_merge
rule cuomo_replicateGO_merge:
    input:
        go = expand('staging/replicateGO/go.{i}.txt', i=range(60)),
    output:
        go = 'staging/replicateGO/go.txt'
    shell:
        '''
        awk '!(FNR==1 && NR!=1) {{print}}' {input.go} > {output.go}
        '''

rule cuomo_replicateGO_plot:
    input:
        old = 'data/cuomo2020natcommun/suppdata6.txt',
        new = 'staging/replicateGO/go.txt',
    output:
        png = 'results/replicateGO/go.png',
    script: 'bin/cuomo_replicateGO_plot.py'

# analysis
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

rule cuomo_day_imputeNinputForONG:
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
        nu_ctng = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_ct_y = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        imputed_ct_nu_ctng = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
    resources: mem = '10gb',
    script: 'bin/cuomo_day_imputeNinputForONG.py'

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

rule cuomo_ong_test:
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
    script: "bin/cuomo_ong_test.py"

#def cuomo_ong_test_agg(wildcards):
#    checkpoint_output = checkpoints.cuomo_split2batches.get(**wildcards).output[0]
#    # snakemake bug
#    par = ''
#    for column in cuomo_params.columns:
#        par = par + f'{column}={wildcards[column]}/'
#    #print(par)
#    return expand(f"staging/cuomo/{par[:-1]}/batch{{i}}/out.txt", 
#            i=glob_wildcards(os.path.join(checkpoint_output, "batch{i}.txt")).i)

rule cuomo_ong_test_mergeBatches:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/out.txt' 
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    script: 'bin/mergeBatches.py'

rule cuomo_ong_corr_plot:
    input:
        base = expand('analysis/cuomo/{params}/out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/CTcorr.png',
    script: 'bin/cuomo_ong_corr_plot.py'

rule cuomo_ong_rVariance_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/rVariance.png',
    script: 'bin/cuomo_ong_rVariance_plot.py'

rule cuomo_ong_variance_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/vc.png',
    params:
        cut_off = {'free':[-1.5,2], 'full':[-3,3]},
    script: 'bin/cuomo_ong_variance_plot.py'

rule cuomo_ong_waldNlrt_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/waldNlrt.png',
    script: 'bin/cuomo_ong_waldNlrt_plot.py'

rule cuomo_ong_experimentR_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy', 
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/hom.png',
    script: 'bin/cuomo_ong_experimentR_plot.py'

rule cuomo_ong_experimentR_all:
    input:
        png = expand('results/cuomo/{params}/hom.png',
                params=Paramspace(cuomo_params.loc[cuomo_params['experiment']=='R'], filename_params="*").instance_patterns),

rule cuomo_ctng_test:
    input:
        y_batch = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
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
    script: 'bin/cuomo_ctng_test.py'

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_mergeBatches with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',

rule cuomo_ctng_test2:
    input:
        y_batch = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out2.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out2.npy',
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
    script: 'bin/cuomo_ctng_test.py'

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test2_mergeBatches with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out2.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out2.npy',

use rule cuomo_ctng_test as cuomo_ctng_test_remlJK with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.remlJK.out.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/remlJK.out.npy',
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

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_remlJK_mergeBatches with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.remlJK.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.remlJK.out.npy',

use rule cuomo_ong_waldNlrt_plot as  cuomo_ctng_waldNlrt_plot with:
    # when using LRT test p value in Free REML
#rule cuomo_ctng_waldNlrt_plot:
    # when using Wald test p value in Free REML
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.waldNlrt.png',
    #script: 'bin/cuomo_ctng_waldNlrt_plot.py'

use rule cuomo_ong_corr_plot as cuomo_ctng_corr_plot with:
    input:
        base = expand('analysis/cuomo/{params}/ctng.out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.CTcorr.png',

use rule cuomo_ong_rVariance_plot as cuomo_ctng_rVariance_plot with:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.rVariance.png',

rule cuomo_ctng_variance_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        nu_ctng = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/nu.ctng.txt'
                for i in range(cuomo_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.vc.png',
    params:
        free = ['hom', 'CT_main', 'ct_random_var', 'nu'],
        cut_off = {'free':[-0.5,0.5], 'full':[-3,3]},
    script: 'bin/cuomo_ctng_variance_plot.py'

rule cuomo_ctng_Vplot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.V.png',
    script: 'bin/cuomo_ctng_Vplot.py'

use rule cuomo_ctng_test as cuomo_ctng_test_miny with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.miny.out.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/miny.out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,  
        REML = True,
        HE = True, 
        jack_knife = True,
    resources: 
        mem = '10gb',
        time = '48:00:00',

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_miny_mergeBatches with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.miny.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.miny.out.npy',

use rule cuomo_ctng_waldNlrt_plot as cuomo_ctng_waldNlrt_plot_miny with:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.miny.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.miny.waldNlrt.png',

use rule cuomo_ong_rVariance_plot as cuomo_ctng_rVariance_plot_miny with:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.miny.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.miny.rVariance.png',

rule cuomo_ctng_HEpvalue_acrossmodel_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.HEpvalue.png',
    script: 'bin/cuomo_ctng_HEpvalue_acrossmodel_plot.py'

###### p values in REML vs HE free
rule cuomo_ctng_pvalue_REMLvsHE:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.free.REMLvsHE.inflated_zeros_{{prop}}.png',
    script: 'bin/cuomo_ctng_pvalue_REMLvsHE.py'

rule cuomo_ctng_pvalue_REMLvsHE_addrVariance:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.free.REMLvsHE.rVariance.png',
    script: 'bin/cuomo_ctng_pvalue_REMLvsHE_addrVariance.py'

#rule cuomo_ctng_bugs:
#    input:
#        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
#        imputed_ct_y = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
#                for i in range(cuomo_batch_no)], # donor - day * gene
#        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
#    output:
#        png = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng_bugs.png',
#    script: 'bin/ctng_bugs.py' 

#rule cuomo_ongVSctng_hom:
#    input:
#        ong = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
#        ctng = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
#    output:
#        hom = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ongVSctng.hom.png',
#        V = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ongVSctng.V.png',
#        beta = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ongVSctng.beta.png',
#    script: "bin/cuomo_ongVSctng_hom.py"

rule cuomo_ng_all:
    input:
        imputation = expand('analysis/cuomo/{params}/imputation.png', 
                params=cuomo_paramspace.instance_patterns),
        ong_CTcorr = expand('results/cuomo/{params}/CTcorr.png', 
                params=cuomo_paramspace.instance_patterns),
        #ong_rVar = expand('results/cuomo/{params}/rVariance.png', 
        #        params=cuomo_paramspace.instance_patterns),
        #ong_varcomponent = expand('results/cuomo/{params}/vc.png', 
        #        params=cuomo_paramspace.instance_patterns),
        ctng_CTcorr = expand('results/cuomo/{params}/ctng.CTcorr.png', 
                params=cuomo_paramspace.instance_patterns),
        ctng_rVar = expand('results/cuomo/{params}/ctng.rVariance.png', 
                params=cuomo_paramspace.instance_patterns),
        #ctng_varcomponent = expand('results/cuomo/{params}/ctng.vc.png', 
        #        params=cuomo_paramspace.instance_patterns),
        #ongVSctng = expand('results/cuomo/{params}/ongVSctng.hom.png',
        #        params=cuomo_paramspace.instance_patterns),
        #ctng_bugs = expand('analysis/cuomo/{params}/ctng_bugs.png',
        #        params=cuomo_paramspace.instance_patterns),
        pca = expand('results/cuomo/{params}/pca.associatedVar.png',
                params=cuomo_paramspace.instance_patterns),
        ong_wald = expand('results/cuomo/{params}/waldNlrt.png',
                params=cuomo_paramspace.instance_patterns),
        ctng_wald = expand('results/cuomo/{params}/ctng.waldNlrt.png',
                params=cuomo_paramspace.instance_patterns),
        #ctng_HE = expand('results/cuomo/{params}/ctng.HEpvalue.png',
        #        params=cuomo_paramspace.instance_patterns),
        ctng_REMLvsHE = expand('results/cuomo/{params}/ctng.free.REMLvsHE.inflated_zeros_1.png',
                params=cuomo_paramspace.instance_patterns),
        #ctng_enrichment = expand('results/cuomo/{params}/enrichment/reml.V_bon.beta_bon.enrich.txt',
        #        params=cuomo_paramspace.instance_patterns),
        #ctng_remlJK = expand('analysis/cuomo/{params}/ctng.remlJK.out.npy',
        #        params=cuomo_paramspace.instance_patterns),

# single cell expression pattern plot
rule cuomo_sc_expressionpattern:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.remlJK.out.npy',
        imputed_ct_y = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_batch_no)], # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/ctng.{{gene}}.png', 
    params:
        mycolors = mycolors,
        paper = True,
    script: 'bin/cuomo_sc_expressionpattern.py'

rule cuomo_sc_expressionpattern_paper:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.remlJK.out.npy',
        imputed_ct_y = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_batch_no)], # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/paper.ctng.png', 
    params:
        mycolors = mycolors,
        genes = ['ENSG00000204531_POU5F1', 'NDUFB4', 'ENSG00000185155_MIXL1', 'ENSG00000163508_EOMES'],
    script: 'bin/cuomo_sc_expressionpattern.paper.py'

rule cuomo_sc_expressionpattern_collect:
    input:
        png = [f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/ctng.{gene}.png'
                for gene in ['ENSG00000111704_NANOG', 'ENSG00000141448_GATA6', 'ENSG00000204531_POU5F1',
                    'ENSG00000181449_SOX2', 'ENSG00000065518_NDUFB4', 'ENSG00000074047_GLI2', 'ENSG00000136997_MYC',
                    'ENSG00000125845_BMP2', 'ENSG00000107984_DKK1', 'ENSG00000234964_FABP5P7', 
                    'ENSG00000166105_GLB1L3', 'ENSG00000237550_UBE2Q2P6', 'ENSG00000230903_RPL9P8']],
    output:
        touch(f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.sc_expressionpattern.flag'),

###### GO pathway enrichment
rule cuomo_ctng_enrichment_Free_genes:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.remlJK.out.npy',
    output:
        genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/all.genes.txt',
        reml_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.genes.txt',
        reml_bon_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.topgenes.txt',
        reml_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.topgenes.txt',
        reml_bon_1 = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_0.01.genes.txt',
        reml_bon_fdr = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_fdr10.genes.txt',
        reml_bon_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_bon.genes.txt',
        he_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.genes.txt',
        he_bon_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.topgenes.txt',
        he_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.topgenes.txt',
        he_bon_1 = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_0.01.genes.txt',
        he_bon_fdr = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_fdr10.genes.txt',
        he_bon_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_bon.genes.txt',
        reml_fdr_1 = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr10.beta_0.01.genes.txt',
        reml_fdr_fdr = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr10.beta_fdr10.genes.txt',
        reml_fdr_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr10.beta_bon.genes.txt',
        he_fdr_1 = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr10.beta_0.01.genes.txt',
        he_fdr_fdr = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr10.beta_fdr10.genes.txt',
        he_fdr_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr10.beta_bon.genes.txt',
        reml_bon_1_sigHE = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_0.01.sigHE.genes.txt',
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/genes.p.png',
    script: 'bin/cuomo_ctng_Free_genes.py'

rule cuomo_ctng_enrichment_GOnPathway:
    input:
        genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/all.genes.txt',
        #reml_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.genes.txt',
        reml_bon_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.topgenes.txt',
        reml_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.topgenes.txt',
        reml_bon_1 = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_0.01.genes.txt',
        reml_bon_fdr = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_fdr10.genes.txt',
        reml_bon_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_bon.genes.txt',
        #he_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.genes.txt',
        he_bon_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.topgenes.txt',
        he_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.topgenes.txt',
        he_bon_1 = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_0.01.genes.txt',
        he_bon_fdr = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_fdr10.genes.txt',
        he_bon_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_bon.genes.txt',
        reml_fdr_1 = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr10.beta_0.01.genes.txt',
        reml_fdr_fdr = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr10.beta_fdr10.genes.txt',
        reml_fdr_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr10.beta_bon.genes.txt',
        he_fdr_1 = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr10.beta_0.01.genes.txt',
        he_fdr_fdr = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr10.beta_fdr10.genes.txt',
        he_fdr_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr10.beta_bon.genes.txt',
        reml_bon_1_sigHE = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_0.01.sigHE.genes.txt',
    output:
        #reml_bon_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.enrich.txt',
        reml_bon_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.top.enrich.txt',
        reml_beta_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.top.enrich.txt',
        reml_bon_1_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_0.01.enrich.txt',
        reml_bon_fdr_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_fdr10.enrich.txt',
        reml_bon_bon_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_bon.enrich.txt',
        #he_bon_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.enrich.txt',
        he_bon_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.top.enrich.txt',
        he_beta_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.top.enrich.txt',
        he_bon_1_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_0.01.enrich.txt',
        he_bon_fdr_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_fdr10.enrich.txt',
        he_bon_bon_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_bon.enrich.txt',
        reml_fdr_1_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr.beta_0.01.enrich.txt',
        reml_fdr_fdr_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr.beta_fdr10.enrich.txt',
        reml_fdr_bon_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr.beta_bon.enrich.txt',
        he_fdr_1_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr.beta_0.01.enrich.txt',
        he_fdr_fdr_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr.beta_fdr10.enrich.txt',
        he_fdr_bon_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr.beta_bon.enrich.txt',
        reml_bon_1_sigHE_enrich = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_0.01.sigHE.enrich.txt',

        #reml_bon_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.gse.txt',
        reml_bon_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.top.gse.txt',
        reml_beta_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.top.gse.txt',
        reml_bon_1_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_0.01.gse.txt',
        reml_bon_fdr_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_fdr10.gse.txt',
        reml_bon_bon_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_bon.gse.txt',
        #he_bon_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.gse.txt',
        he_bon_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.top.gse.txt',
        he_beta_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.top.gse.txt',
        he_bon_1_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_0.01.gse.txt',
        he_bon_fdr_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_fdr10.gse.txt',
        he_bon_bon_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_bon.beta_bon.gse.txt',
        reml_fdr_1_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr.beta_0.01.gse.txt',
        reml_fdr_fdr_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr.beta_fdr10.gse.txt',
        reml_fdr_bon_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr.beta_bon.gse.txt',
        he_fdr_1_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr.beta_0.01.gse.txt',
        he_fdr_fdr_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr.beta_fdr10.gse.txt',
        he_fdr_bon_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V_fdr.beta_bon.gse.txt',
        reml_bon_1_sigHE_gse = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_bon.beta_0.01.sigHE.gse.txt',
    resources:
        mem = '10gb',
        time = '24:00:00',
    shell:
        '''
        module load R/4.0.3
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_bon_1} {input.genes} \
                {output.reml_bon_1_enrich} {output.reml_bon_1_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_bon_fdr} {input.genes} \
                {output.reml_bon_fdr_enrich} {output.reml_bon_fdr_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_bon_bon} {input.genes} \
                {output.reml_bon_bon_enrich} {output.reml_bon_bon_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.he_bon_1} {input.genes} \
                {output.he_bon_1_enrich} {output.he_bon_1_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.he_bon_fdr} {input.genes} \
                {output.he_bon_fdr_enrich} {output.he_bon_fdr_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.he_bon_bon} {input.genes} \
                {output.he_bon_bon_enrich} {output.he_bon_bon_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_bon_1_sigHE} {input.genes} \
                {output.reml_bon_1_sigHE_enrich} {output.reml_bon_1_sigHE_gse}; \
        \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_fdr_1} {input.genes} \
                {output.reml_fdr_1_enrich} {output.reml_fdr_1_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_fdr_fdr} {input.genes} \
                {output.reml_fdr_fdr_enrich} {output.reml_fdr_fdr_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_fdr_bon} {input.genes} \
                {output.reml_fdr_bon_enrich} {output.reml_fdr_bon_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.he_fdr_1} {input.genes} \
                {output.he_fdr_1_enrich} {output.he_fdr_1_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.he_fdr_fdr} {input.genes} \
                {output.he_fdr_fdr_enrich} {output.he_fdr_fdr_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.he_fdr_bon} {input.genes} \
                {output.he_fdr_bon_enrich} {output.he_fdr_bon_gse}; \
        \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_bon_top} {input.genes} \
                {output.reml_bon_top_enrich} {output.reml_bon_top_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.he_bon_top} {input.genes} \
                {output.he_bon_top_enrich} {output.he_bon_top_gse}; \
        \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_beta_top} {input.genes} \
                {output.reml_beta_top_enrich} {output.reml_beta_top_gse}; \
        Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.he_beta_top} {input.genes} \
                {output.he_beta_top_enrich} {output.he_beta_top_gse}; \
        \
        '''
        #Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.reml_bon} {input.genes} \
                #        {output.reml_bon_enrich} {output.reml_bon_gse}; \
        #Rscript bin/cuomo_ctng_enrichment_GOnPathway.R {input.he_bon} {input.genes} \
                #        {output.he_bon_enrich} {output.he_bon_gse};

rule cuomo_ctng_enrichment_CountGO:
    input:
        genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/all.genes.txt',
        reml_fdr_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr10.beta_bon.genes.txt',
    output:
        go = temp(f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/all.genes.GOannotation.txt'),
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/count_GOannotaion_pergene.txt',
    shell:
        '''
        module load R/4.0.3
        Rscript bin/cuomo_ctng_enrichment_CountGO.R {input.genes} {output.go}
        python3 bin/cuomo_ctng_enrichment_CountGO.py {output.go} {input.reml_fdr_bon} {output.png}
        '''

# likelihood
rule cuomo_likelihood_plot:
    input:
        imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt'
                for i in range(cuomo_batch_no)], #donor-day * gene 
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.likelihood.png',
    script: 'bin/cuomo_likelihood_plot.py'

# p value across test methods: HE, ML, REML, Wald, JK
rule cuomo_ctng_p:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        out2 = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out2.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.remlJK.out.npy',
    output:
        hom2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.hom2.png',
        p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.p.png',
    script: 'bin/cuomo_ctng_p.py'

rule cuomo_ongVSctng_p:
    input:
        ong = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.remlJK.out.npy',
    output:
        p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ongVSctng.p.png',
    script: 'bin/cuomo_ongVSctng_p.py'

# find top genes
rule cuomo_geneP:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/ctng.{{gene}}.P.txt',
    script: 'bin/cuomo_geneP.py'

rule cuomo_topgenes:
    input:
        ong = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        ctng = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.remlJK.out.npy',
    output:
        topgenes = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ongNctng.topgenes.txt',
    params:
        ong = ['reml', 'he'],
        ctng = ['remlJK', 'he'],
    script: 'bin/cuomo_topgenes.py'

# paper plot
rule paper_cuomo_ong_pvalue_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/waldNlrt.supp.png',
    script: 'bin/cuomo_ong_waldNlrt_plot_paper.py'

rule paper_cuomo_ctng_pvalue_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.remlJK.out.npy',
    output:
        reml_p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.REMLpvalue.paper.png',
        he_p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.HEpvalue.paper.png',
        qq = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.qq.supp.png',
    script: 'bin/paper_cuomo_ctng_pvalue_plot.py'

rule paper_cuomo_ctng_pvalue_plot_ASHG:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.remlJK.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.REMLpvalue.paper.ASHG.png',
        png2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.HEpvalue.paper.ASHG.png',
    script: 'bin/paper_cuomo_ctng_pvalue_plot.ASHG.py'

rule paper_cuomo_freeNfull_Variance_plot:
    input:
        ong = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        ctng = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.freeNfull.Variance.paper.png',
    script: 'bin/paper_cuomo_freeNfull_Variance_plot.py'

rule paper_cuomo_freeNfull_Variance_plot_ASHG:
    input:
        ong = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        ctng = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.freeNfull.Variance.paper.ASHG.png',
    script: 'bin/paper_cuomo_freeNfull_Variance_plot.ASHG.py'

rule paper_cuomo_freeNfull_Variance_plot_supp:
    input:
        ong = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/out.npy',
        ctng = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        ong = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ong.freeNfull.Variance.supp.png',
        ctng = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.freeNfull.Variance.supp.png',
    script: 'bin/paper_cuomo_freeNfull_Variance_plot_supp.py'

rule paper_cuomo_ctng_corr_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.CTcorr.paper.png',
    script: 'bin/paper_cuomo_ctng_corr_plot.py'


###########################################################################################
# simulate Cuomo genes: a random gene's hom2, ct main variance, nu
###########################################################################################
cuomo_simulateGene_gene_no = 1000
cuomo_simulateGene_batch_no = 100
cuomo_simulateGene_batches = np.array_split(range(cuomo_simulateGene_gene_no), cuomo_simulateGene_batch_no)
rule cuomo_simulateGene_randompickgene:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.nu.ctng.txt'
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
            out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
            genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
            imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.nu.ctng.txt'
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

use rule ctng_test as cuomo_simulateGene_hom_ctng_test with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctng.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/rep/ctng.out.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = True,
        REML = True,
        HE = True,

use rule cuomo_ong_test_mergeBatches as cuomo_simulateGene_hom_ctng_test_mergeBatches with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctng.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctng.out.npy',

use rule ctng_test as cuomo_simulateGene_hom_ctng_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctng.remlJK.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/rep/ctng.remlJK.out.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = False,
        REML = True,
        Free_reml_jk = True,
        HE = False,
    resources:
        mem_per_cpu = '12gb',
        time = '48:00:00',

use rule cuomo_ong_test_mergeBatches as cuomo_simulateGene_hom_ctng_test_remlJK_mergeBatches with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctng.remlJK.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctng.remlJK.out.npy',

nu_noises = ['1_0_0', '1_2_20', '1_2_10', '1_2_5', '1_2_3', '1_2_2']
rule cuomo_simulateGene_hom_ctng_test_powerplot:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.remlJK.out.npy'
                for nu_noise in nu_noises],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctng.hom.power.png',
    params:
        nu_noises = nu_noises,
    script: 'bin/cuomo_simulateGene_hom_ctng_test_powerplot.py'

rule cuomo_simulateGene_hom_ctng_test_estimates:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.remlJK.out.npy'
                for nu_noise in nu_noises],
        real_out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        genes = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{i}.txt' 
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctng.hom.estimates.png',
    params:
        nu_noises = nu_noises,
    script: 'bin/cuomo_simulateGene_hom_ctng_test_estimates.py'

rule cuomo_simulateGene_hom_ctng_test_estimates_paper:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.out.npy'
                for nu_noise in nu_noises],
        #remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.remlJK.out.npy'
        #        for nu_noise in nu_noises],
        #real_out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        #genes = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{i}.txt' 
        #        for i in range(cuomo_simulateGene_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctng.hom.estimates.paper.png',
    params:
        nu_noises = nu_noises,
    script: 'bin/cuomo_simulateGene_hom_ctng_test_estimates_paper.py'

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

use rule ctng_test as cuomo_simulateGene_Free_ctng_test with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctng.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/rep/ctng.out.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = True,
        REML = True,
        HE = True,

use rule cuomo_ong_test_mergeBatches as cuomo_simulateGene_Free_ctng_test_mergeBatches with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctng.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctng.out.npy',

use rule ctng_test as cuomo_simulateGene_Free_ctng_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{nu_noise}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctng.remlJK.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/rep/ctng.remlJK.out.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str'),
        ML = False,
        REML = True,
        Free_reml_jk = True,
        HE = False,
    resources:
        mem_per_cpu = '12gb',
        time = '48:00:00',

use rule cuomo_ong_test_mergeBatches as cuomo_simulateGene_Free_ctng_test_remlJK_mergeBatches with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctng.remlJK.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{{V}}/{{nu_noise}}/ctng.remlJK.out.npy',

V1 = ['0_0_0_0', '0.05_0_0_0','0.1_0_0_0', '0.2_0_0_0', '0.5_0_0_0']
V2 = ['0.05_0.05_0.05_0.05', '0.1_0.1_0.1_0.1', '0.2_0.2_0.2_0.2', '0.5_0.5_0.5_0.5']
V3 = ['0_0_0_0', '0.05_0.1_0.1_0.1', '0.1_0.1_0.1_0.1', '0.2_0.1_0.1_0.1', '0.5_0.1_0.1_0.1']
rule cuomo_simulateGene_Free_ctng_test_powerplot:
    input:
        outs1 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.out.npy'
                for V in V1],
        outs2 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.out.npy'
                for V in V2],
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.out.npy'
                for V in V3],
        remlJKs1 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.remlJK.out.npy'
                for V in V1],
        remlJKs2 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.remlJK.out.npy'
                for V in V2],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.remlJK.out.npy'
                for V in V3],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctng.free.1_2_5.power.png',
    params:
        V1 = V1,
        V2 = V2,
        V3 = V3,
    script: 'bin/cuomo_simulateGene_Free_ctng_test_powerplot.py'

rule cuomo_simulateGene_ctng_test_powerplot_paper:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.remlJK.out.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.out.npy'
                for V in V3],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.remlJK.out.npy'
                for V in V3],
    output:
        png1 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctng.power.paper.supp.png',
        png2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctng.power.paper.png',
    params:
        nu_noises = nu_noises,
        V3 = V3,
    script: 'bin/cuomo/simulateGene_ctng_test_powerplot_paper.py'

rule cuomo_simulateGene_ctng_test_powerplot_paper_ASHG:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.remlJK.out.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.out.npy'
                for V in V3],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.remlJK.out.npy'
                for V in V3],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctng.power.paper.ASHG.png',
    params:
        nu_noises = nu_noises,
        V3 = V3,
    script: 'bin/cuomo_simulateGene_ctng_test_powerplot_paper.ASHG.py'

rule cuomo_simulateGene_ctng_test_powerplot_paper_ZJU:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.out.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctng.remlJK.out.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.out.npy'
                for V in V3],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.remlJK.out.npy'
                for V in V3],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctng.power.paper.ZJU.png',
    params:
        nu_noises = nu_noises,
        V3 = V3,
    script: 'bin/cuomo_simulateGene_ctng_test_powerplot_paper.ZJU.py'

rule cuomo_simulateGene_free_ctng_test_estimates:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.out.npy'
                for V in V3],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctng.remlJK.out.npy'
                for V in V3],
        #real_out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
        #genes = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{i}.txt' 
        #        for i in range(cuomo_simulateGene_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctng.free.estimates.png',
    params:
        V3 = V3,
    script: 'bin/cuomo_simulateGene_free_ctng_test_estimates.py'

###########################################################################################
# CTNG p value: bootstrap cells (rerun model)
###########################################################################################
use rule cuomo_ctng_test as cuomo_ctng_test_bootstrap with:
    output:
        out = f'staging/cuomo/bootstrap/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
    params:
        out = f'staging/cuomo/bootstrap/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,
        REML = False,
        HE = True,
        Full_HE = False,
        HE_as_initial = False,
    resources: 
        mem = '1gb',

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_mergeBatches_bootstrap with:
    input:
        out = [f'staging/cuomo/bootstrap/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out.txt' 
                for i in range(cuomo_batch_no)],
    output:
        out = f'staging/cuomo/bootstrap/{cuomo_paramspace.wildcard_pattern}/ctng.out.nominalP.npy',


# Bootstrap
cuomo_bootstrap_batch_no = 10

rule cuomo_day_bootstrap:
    ### be careful: bootstrap generate duplicate cells
    input:
        meta = 'analysis/cuomo/data/meta.txt',
    output:
        meta = 'staging/cuomo/data/bootstrap/rep{b}/meta.txt',
    script: 'bin/cuomo_day_bootstrap.py'

use rule cuomo_day_pseudobulk_log as cuomo_day_pseudobulk_log_bootstrap with:
    input:
        meta = 'staging/cuomo/data/bootstrap/rep{b}/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        y = temp('staging/cuomo/data/bootstrap/rep{b}/log/day.raw.pseudobulk.gz'), # donor - day * gene
        nu = temp('staging/cuomo/data/bootstrap/rep{b}/log/day.raw.nu.gz'), # donor - day * gene
    params: unique_cell = False,

use rule cuomo_day_filterInds as cuomo_day_filterInds_bootstrap with:
    input:
        meta = 'staging/cuomo/data/bootstrap/rep{b}/meta.txt',
        y = 'staging/cuomo/data/bootstrap/rep{b}/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'staging/cuomo/data/bootstrap/rep{b}/log/day.raw.nu.gz', # donor - day * gene
    output:
        y = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz'), # donor - day * gene
        nu = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz'), # donor - day * gene
        P = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz'), # donor * day
        n = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz'), # donor * day

use rule cuomo_day_filterCTs as cuomo_day_filterCTs_bootstrap with:
    input:
        y = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        n = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
    output:
        y = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz'), # donor - day * gene
        nu = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz'), # donor - day * gene

use rule cuomo_split2batches as cuomo_split2batches_bootstrap with:
    input:
        y = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz',
    output:
        y_batch = [f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/y/batch{i}.txt' 
                for i in range(cuomo_bootstrap_batch_no)],

use rule cuomo_day_imputeGenome as cuomo_day_imputeGenome_bootstrap with:
    input:
        y = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
    output:
        y = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz'), # donor - day * gene
        nu = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz'), # donor - day * gene

use rule cuomo_day_imputeNinputForONG as cuomo_day_imputeNinputForONG_bootstrap with:
    # also exclude individuals with nu = 0 which cause null model fail (some individuals have enough cells, but all cells have no expression of specific gene)
    # seems we should keep nu = 0
    input:
        P = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
        y = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene
        y_batch = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
    output:
        y = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        nu = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
        nu_ctng = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        P = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_ct_y = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        imputed_ct_nu = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        imputed_ct_nu_ctng = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
    resources:
        burden = 10,

use rule cuomo_day_y_collect as cuomo_day_y_collect_bootstrap with:
    input:
        y = [f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{i}/y.txt' 
                for i in range(cuomo_bootstrap_batch_no)],
    output:
        y = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',

use rule cuomo_day_pca as cuomo_day_pca_bootstrap with:
    input:
        y = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',
        P = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
    output:
        evec = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/evec.txt'),
        eval = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/eval.txt'),
        pca = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/pca.txt'),
        png = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/pca.png'),

use rule cuomo_ctng_test as cuomo_ctng_test_rep_bootstrap with:
    input:
        y_batch = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        imputed_ct_nu = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
        P = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'staging/cuomo/data/bootstrap/rep{b}/meta.txt',
    output:
        out = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
    params:
        out = f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,
        REML = False,
        HE = True,
        Full_HE = False,
        HE_as_initial = False,
    resources: mem = '1gb',

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_bootstrap_mergeBatches with:
    input:
        out = [f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out.txt'
                for i in range(cuomo_bootstrap_batch_no)],
    output:
        out = temp(f'staging/cuomo/bootstrap/rep{{b}}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy'),

rule cuomo_ctng_test_bootstrap_p:
    input:
        out = f'staging/cuomo/bootstrap/{cuomo_paramspace.wildcard_pattern}/ctng.out.nominalP.npy',
        boot = [f'staging/cuomo/bootstrap/rep{b}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy' 
                for b in range(500)],
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day # to get number of individuals and days, for degree of freedom
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.bootstrap.out.npy',
    script: 'bin/cuomo_ctng_test_bootstrap_p.py'

use rule cuomo_ctng_HEpvalue_acrossmodel_plot as cuomo_ctng_test_bootstrap_pvalueplot with:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.bootstrap.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.bootstrap.png'
    
rule cuomo_bootstrapVSjackknife:
    input:
        bs = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.bootstrap.out.npy',
        jk = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.bootstrapVSjackknife.png',
    script: 'bin/cuomo_bootstrapVSjackknife.py'

cuomo_permuteCT_paramspace = Paramspace(cuomo_params.loc[cuomo_params['im_genome'] == 'Y'], filename_params="*")
rule cuomo_bootstrap_all:
    input:
        out = expand('results/cuomo/{params}/ctng.bootstrap.png',
                params=cuomo_permuteCT_paramspace.instance_patterns),
        bsVSjk = expand('results/cuomo/{params}/ctng.bootstrapVSjackknife.png',
                params=cuomo_permuteCT_paramspace.instance_patterns),


###########################################################################################
# CTNG p value: bootstrap cells to get nu 
###########################################################################################
rule cuomo_day_pseudobulk_log_splitCounts:
    input:
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        counts = expand('staging/cuomo/bootstrapedNU/data/counts{i}.txt.gz', i=range(100)),
    resources: 
        mem_per_cpu = '10gb',
    script: 'bin/cuomo_day_pseudobulk_log_splitCounts.py'

rule cuomo_day_pseudobulk_log_bootstrapedNU:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'staging/cuomo/bootstrapedNU/data/counts{i}.txt.gz',
    output:
        nu = 'staging/cuomo/bootstrapedNU/data/counts{i}.nu.gz',
    resources: 
        mem_per_cpu = '10gb',
        time = '24:00:00',
    shell: 
        '''
        module load python/3.8.1
        python3 bin/cuomo_day_pseudobulk_log_bootstrapedNU.py {input.meta} {input.counts} {output.nu}
        '''

rule cuomo_day_pseudobulk_log_bootstrapedNU_merge:
    input:
        nu = expand('staging/cuomo/bootstrapedNU/data/counts{i}.nu.gz', i=range(100)),
    output:
        nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.nu.gz', # donor - day * gene
    run:
        nus = [pd.read_table(f, index_col=(0,1)) for f in input.nu]
        # check donor day
        index = nus[0].index
        #donors = nus[0]['donor']
        #days = nus[0]['day']
        for data in nus[1:]:
            #if np.any( donors != data['donor'] ) or np.any( days != data['day'] ):
            if np.any( index != data.index ):
                sys.exit('Wrong order!\n')
        # merge
        data = pd.concat( nus, axis=1 )
        data.to_csv( output.nu, sep='\t')

use rule cuomo_day_filterInds as cuomo_day_filterInds_bootstrapedNU with:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        y = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.nu.gz', # donor - day * gene
    output:
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        P = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day

use rule cuomo_day_filterCTs as cuomo_day_filterCTs_bootstrapedNU with:
    input:
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        n = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
    output:
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene

rule cuomo_day_compareNU_bootstrapedNU:
    input:
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
        boot = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
    output:
        png = f'results/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/data/log/nu.bootstrapedNU.png',
    script: 'bin/cuomo_day_compareNU_bootstrapedNU.py'

use rule cuomo_split2batches as cuomo_split2batches_bootstrapedNU with:
    input:
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz',
    output:
        y_batch = [f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/y/batch{i}.txt' 
                for i in range(cuomo_batch_no)],

use rule cuomo_day_imputeGenome as cuomo_day_imputeGenome_bootstrapedNU with:
    input:
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
    output:
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene

use rule cuomo_day_imputeNinputForONG as cuomo_day_imputeNinputForONG_bootstrapedNU with:
    # also exclude individuals with nu = 0 which cause null model fail (some individuals have enough cells, but all cells have no expression of specific gene)
    # seems we should keep nu = 0
    input:
        P = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene
        y_batch = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
    output:
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
        nu_ctng = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        P = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_ct_y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        imputed_ct_nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        imputed_ct_nu_ctng = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)

use rule cuomo_day_y_collect as cuomo_day_y_collect_bootstrapedNU with:
    input:
        y = [f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{i}/y.txt' 
                for i in range(cuomo_batch_no)],
    output:
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',

use rule cuomo_day_pca as cuomo_day_pca_bootstrapedNU with:
    input:
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',
        P = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
    output:
        evec = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/evec.txt',
        eval = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/eval.txt',
        pca = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        png = f'results/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/pca.png',

use rule cuomo_ong_test as cuomo_ong_test_bootstrapedNU with:
    input:
        y_batch = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt',
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt',
        P = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt',
        pca = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment 
        imputed_ct_nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', # donor - day * gene
    output:
        out = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/out.txt',
    params:
        out = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        HE_as_initial = False,

use rule cuomo_ong_test_mergeBatches as cuomo_ong_test_mergeBatches_bootstrapedNU with:
    input:
        out = [f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{i}/out.txt' 
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/out.npy',

use rule cuomo_ong_corr_plot as cuomo_ong_corr_plot_bootstrapedNU with:
    input:
        base = expand('analysis/cuomo/bootstrapedNU/{params}/out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'analysis/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/bootstrapedNU.CTcorr.png',

use rule cuomo_ong_waldNlrt_plot as cuomo_ong_waldNlrt_plot_bootstrapedNU with:
    input:
        out = f'analysis/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/bootstrapedNU.waldNlrt.png',

use rule cuomo_ctng_test as cuomo_ctng_test_bootstrapedNU with:
    input:
        y_batch = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        imputed_ct_nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
        P = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        out = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
    params:
        out = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,
        REML = True,
        HE = True,
        Full_HE = True,
        jack_knife = True,
        HE_as_initial = False,
    resources:
        mem = '2gb',

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_mergeBatches_bootstrapedNU with:
    input:
        out = [f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',

use rule cuomo_ong_waldNlrt_plot as  cuomo_ctng_waldNlrt_plot_bootstrapedNU with:
    input:
        out = f'analysis/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.bootstrapedNU.waldNlrt.png',

use rule cuomo_ong_corr_plot as cuomo_ctng_corr_plot_bootstrapedNU with:
    input:
        base = expand('analysis/cuomo/bootstrapedNU/{params}/ctng.out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'analysis/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.bootstrapedNU.CTcorr.png',

rule cuomo_ng_bootstrapedNU_all:
    input:
        png = expand('results/cuomo/bootstrapedNU/{params}/data/log/nu.bootstrapedNU.png',
                params=cuomo_paramspace.instance_patterns),
        ong_CTcorr = expand('results/cuomo/{params}/bootstrapedNU.CTcorr.png', 
                params=cuomo_paramspace.instance_patterns),
        ctng_CTcorr = expand('results/cuomo/{params}/ctng.bootstrapedNU.CTcorr.png', 
                params=cuomo_paramspace.instance_patterns),
        ong_wald = expand('results/cuomo/{params}/bootstrapedNU.waldNlrt.png',
                params=cuomo_paramspace.instance_patterns),
        ctng_wald = expand('results/cuomo/{params}/ctng.bootstrapedNU.waldNlrt.png',
                params=cuomo_paramspace.instance_patterns),

###########################################################################################
# CTNG p value: permute CT for each donor independently 
# also permutes fixed effects
###########################################################################################
#use rule cuomo_ctng_test as cuomo_ctng_test_permuteCT with:
#    output:
#        out = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
#    params:
#        out = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
#        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
#        ML = False,
#        REML = False,
#        HE = True,
#        Full_HE = False,
#        HE_as_initial = False,
#    resources: 
#        mem = '1gb',
#
#use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_mergeBatches_permuteCT with:
#    input:
#        out = [f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt' 
#                for i in range(cuomo_batch_no)],
#    output:
#        out = 'analysis/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
#
#
#cuomo_permuteCT_batch_no = 2
#use rule cuomo_split2batches as cuomo_split2batches_permuteCT with:
#    input:
#        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz',
#    output:
#        y_batch = [f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/y/batch{i}.txt' 
#                for i in range(cuomo_permuteCT_batch_no)],
#
#use rule cuomo_day_imputeNinputForONG as cuomo_day_imputeNinputForONG_permuteCT with:
#    input:
#        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
#        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
#        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
#        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene
#        y_batch = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
#    output:
#        y = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
#        nu = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
#        nu_ctng = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
#        P = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
#        imputed_ct_y = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
#        imputed_ct_nu = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
#        imputed_ct_nu_ctng = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
#
#rule cuomo_permuteCT:
#    input:
#        imputed_ct_y = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
#        imputed_ct_nu = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
#        P = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
#        y_batch = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
#    output:
#        imputed_ct_y = f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
#        imputed_ct_nu = f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
#        P = f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
#    script: 'bin/cuomo_permuteCT.py'
#
#use rule cuomo_ctng_test as cuomo_ctng_test_permuted_permuteCT with:
#    input:
#        y_batch = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
#        imputed_ct_y = f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
#        nu = f'staging/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
#        imputed_ct_nu = f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
#        P = f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
#        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
#        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
#        meta = 'analysis/cuomo/data/meta.txt', # experiment
#    output:
#        out = f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
#    params:
#        out = f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
#        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
#        ML = False,
#        REML = False,
#        HE = True,
#        Full_HE = False,
#        HE_as_initial = False,
#    resources: mem = '1gb',
#
#use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_permuted_mergeBatches_permuteCT with:
#    input:
#        out = [f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out.txt'
#                for i in range(cuomo_permuteCT_batch_no)],
#    output:
#        out = f'staging/cuomo/permuteCT/p{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
#
#rule cuomo_permuteCT_p:
#    input:
#        out = 'analysis/cuomo/permuteCT/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
#        outs = [f'staging/cuomo/permuteCT/p{m}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy' for m in range(99)],
#    output:
#        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.permute.png'
#    script: 'bin/cuomo_permuteCT_p.py'
#
#    
#cuomo_permuteCT_paramspace = Paramspace(cuomo_params.loc[cuomo_params['im_genome'] == 'Y'], filename_params="*")
#rule cuomo_permuteCT_all:
#    input:
#        out = expand('results/cuomo/{params}/ctng.permute.png',
#                params=cuomo_permuteCT_paramspace.instance_patterns),
#
###########################################################################################
# CTNG p value: permute CT for each donor independently 
# permute projected Y rather then original, to keep fixed effects
###########################################################################################
# use nu from bootstrap
rule cuomo_ctng_test_permuteCT_holdfixedeffect2:
    input:
        y_batch = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        imputed_ct_nu = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
        P = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'staging/cuomo/bootstrapedNU/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        out = f'staging/cuomo/permuteCT_holdfixedeffect2/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
    params:
        out = f'staging/cuomo/permuteCT_holdfixedeffect2/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        permutes = 99,
    script: 'bin/cuomo_ctng_test_permuteCT_holdfixedeffect.py'

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_mergeBatches_permuteCT_holdfixedeffect2 with:
    input:
        out = [f'staging/cuomo/permuteCT_holdfixedeffect2/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/permuteCT_holdfixedeffect2/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',

use rule cuomo_ctng_HEpvalue_acrossmodel_plot as cuomo_ctng_pvalueplot_permuteCT_holdfixedeffect2 with:
    input:
        out = f'analysis/cuomo/permuteCT_holdfixedeffect2/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.permute.holdfixedeffect2.png',

rule cuomo_ng_permuteCT_holdfixedeffect2_all:
    input:
        ctng_CTcorr = expand('results/cuomo/{params}/ctng.permute.holdfixedeffect2.png',
                params=cuomo_permuteCT_paramspace.instance_patterns),

# use nu from SEM
rule cuomo_ctng_test_permuteCT_holdfixedeffect:
    input:
        y_batch = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        imputed_ct_nu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        out = f'staging/cuomo/permuteCT_holdfixedeffect/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
    params:
        out = f'staging/cuomo/permuteCT_holdfixedeffect/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        permutes = 99,
    script: 'bin/cuomo_ctng_test_permuteCT_holdfixedeffect.py'

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_mergeBatches_permuteCT_holdfixedeffect with:
    input:
        out = [f'staging/cuomo/permuteCT_holdfixedeffect/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/permuteCT_holdfixedeffect/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',

use rule cuomo_ctng_HEpvalue_acrossmodel_plot as cuomo_ctng_pvalueplot_permuteCT_holdfixedeffect with:
    input:
        out = f'analysis/cuomo/permuteCT_holdfixedeffect/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctng.permute.holdfixedeffect.png',

rule cuomo_ng_permuteCT_holdfixedeffect_all:
    input:
        ctng_CTcorr = expand('results/cuomo/{params}/ctng.permute.holdfixedeffect.png',
                params=cuomo_permuteCT_paramspace.instance_patterns),

#############################################################################
# Xuanyao suspect the widespread signals observed in CTNG might be real.
# prove here
#############################################################################
# mashA: permute assignment of cells to CT, so keep cell type proportions
rule cuomo_day_meta_mashCT:
    input:
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        meta = 'staging/cuomo/data/meta.mash{m}.txt',
    script: 'bin/cuomo_day_meta_mashCT.py'

use rule cuomo_day_pseudobulk_log as cuomo_day_pseudobulk_log_mashCT with:
    input:
        meta = 'staging/cuomo/data/meta.mash{m}.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        y = 'staging/cuomo/mash{m}/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'staging/cuomo/mash{m}/data/log/day.raw.nu.gz', # donor - day * gene

use rule cuomo_day_filterInds as cuomo_day_filterInds_mashCT with:
    input:
        meta = 'staging/cuomo/data/meta.mash{m}.txt',
        y = 'staging/cuomo/mash{m}/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'staging/cuomo/mash{m}/data/log/day.raw.nu.gz', # donor - day * gene
    output:
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        P = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day

use rule cuomo_day_filterCTs as cuomo_day_filterCTs_mashCT with:
    input:
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        n = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
    output:
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene

use rule cuomo_split2batches as cuomo_split2batches_mashCT with:
    input:
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
    output:
        y_batch = [f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/y/batch{i}.txt' 
                for i in range(cuomo_batch_no)],

use rule cuomo_day_imputeGenome as cuomo_day_imputeGenome_mashCT with:
    input:
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
    output:
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene

use rule cuomo_day_imputeNinputForONG as cuomo_day_imputeNinputForONG_mashCT with:
    input:
        P = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene
        y_batch = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
    output:
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
        nu_ctng = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        P = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_ct_y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        imputed_ct_nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        imputed_ct_nu_ctng = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)

use rule cuomo_day_y_collect as cuomo_day_y_collect_mashCT with:
    input:
        y = [f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{i}/y.txt' 
                for i in range(cuomo_batch_no)],
    output:
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',

use rule cuomo_day_pca as cuomo_day_pca_mashCT with:
    input:
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',
        P = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
    output:
        evec = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/evec.txt',
        eval = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/eval.txt',
        pca = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        png = f'results/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/pca.png',

use rule cuomo_ong_test as cuomo_ong_test_mashCT with:
    input:
        y_batch = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt',
        nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt',
        P = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt',
        pca = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'staging/cuomo/data/meta.mash{m}.txt', 
        imputed_ct_nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', # donor - day * gene
    output:
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/out.txt',
    params:
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        HE_as_initial = False,

use rule cuomo_ong_test_mergeBatches as cuomo_ong_test_mergeBatches_mashCT with:
    input:
        out = [f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{i}/out.txt' 
                for i in range(cuomo_batch_no)],
    output:
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/out.npy',

use rule cuomo_ong_corr_plot as cuomo_ong_corr_plot_mashCT with:
    input:
        base = expand('staging/cuomo/mash{{m}}/{params}/out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/CTcorr.png',

use rule cuomo_ong_waldNlrt_plot as cuomo_ong_waldNlrt_plot_mashCT with:
    input:
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/waldNlrt.png',

use rule cuomo_ctng_test as cuomo_ctng_test_mashCT with:
    input:
        y_batch = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        imputed_ct_nu = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
        P = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'staging/cuomo/data/meta.mash{m}.txt', 
    output:
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
    params:
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,
        REML = True,
        HE = True,
        HE_as_initial = False,

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_mergeBatches_mashCT with:
    input:
        out = [f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',

use rule cuomo_ong_waldNlrt_plot as  cuomo_ctng_waldNlrt_plot_mashCT with:
    input:
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.waldNlrt.png',

use rule cuomo_ong_corr_plot as cuomo_ctng_corr_plot_mashCT with:
    input:
        base = expand('staging/cuomo/mash{{m}}/{params}/ctng.out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'staging/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/mash{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.CTcorr.png',

rule cuomo_mashCT_all:
    input:
        ong_CTcorr = expand('results/cuomo/mash{m}/{params}/CTcorr.png', 
                m=range(4), params=cuomo_paramspace.instance_patterns),
        ctng_CTcorr = expand('results/cuomo/mash{m}/{params}/ctng.CTcorr.png', 
                m=range(4), params=cuomo_paramspace.instance_patterns),
        ong_wald = expand('results/cuomo/mash{m}/{params}/waldNlrt.png',
                m=range(4), params=cuomo_paramspace.instance_patterns),
        ctng_wald = expand('results/cuomo/mash{m}/{params}/ctng.waldNlrt.png',
                m=range(4), params=cuomo_paramspace.instance_patterns),

# mashB: randomly assign cells to CT, so each cell type has about 25% proportions
rule cuomo_day_meta_mashB_CT:
    input:
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        meta = 'staging/cuomo/data/meta.mashB{m}.txt',
    script: 'bin/cuomo_day_meta_mashB_CT.py'

use rule cuomo_day_pseudobulk_log as cuomo_day_pseudobulk_log_mashB_CT with:
    input:
        meta = 'staging/cuomo/data/meta.mashB{m}.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        y = 'staging/cuomo/mashB{m}/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'staging/cuomo/mashB{m}/data/log/day.raw.nu.gz', # donor - day * gene

use rule cuomo_day_filterInds as cuomo_day_filterInds_mashB_CT with:
    input:
        meta = 'staging/cuomo/data/meta.mashB{m}.txt',
        y = 'staging/cuomo/mashB{m}/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        nu = 'staging/cuomo/mashB{m}/data/log/day.raw.nu.gz', # donor - day * gene
    output:
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        P = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day

use rule cuomo_day_filterCTs as cuomo_day_filterCTs_mashB_CT with:
    input:
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.nu.gz', # donor - day * gene
        n = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
    output:
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene

use rule cuomo_split2batches as cuomo_split2batches_mashB_CT with:
    input:
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
    output:
        y_batch = [f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/y/batch{i}.txt' 
                for i in range(cuomo_batch_no)],

use rule cuomo_day_imputeGenome as cuomo_day_imputeGenome_mashB_CT with:
    input:
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.nu.gz', # donor - day * gene
    output:
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene

use rule cuomo_day_imputeNinputForONG as cuomo_day_imputeNinputForONG_mashB_CT with:
    input:
        P = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        n = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.cellnum.gz', # donor * day
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
        nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.nu.gz', # donor - day * gene
        y_batch = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
    output:
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
        nu_ctng = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        P = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_ct_y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        imputed_ct_nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        imputed_ct_nu_ctng = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)

use rule cuomo_day_y_collect as cuomo_day_y_collect_mashB_CT with:
    input:
        y = [f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{i}/y.txt' 
                for i in range(cuomo_batch_no)],
    output:
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',

use rule cuomo_day_pca as cuomo_day_pca_mashB_CT with:
    input:
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/y.merged.txt',
        P = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
    output:
        evec = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/evec.txt',
        eval = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/eval.txt',
        pca = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        png = f'results/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/pca.png',

use rule cuomo_ong_test as cuomo_ong_test_mashB_CT with:
    input:
        y_batch = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/y.txt',
        nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.txt',
        P = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt',
        pca = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'staging/cuomo/data/meta.mashB{m}.txt', 
        imputed_ct_nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', # donor - day * gene
    output:
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/out.txt',
    params:
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        HE_as_initial = False,

use rule cuomo_ong_test_mergeBatches as cuomo_ong_test_mergeBatches_mashB_CT with:
    input:
        out = [f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{i}/out.txt' 
                for i in range(cuomo_batch_no)],
    output:
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/out.npy',

use rule cuomo_ong_corr_plot as cuomo_ong_corr_plot_mashB_CT with:
    input:
        base = expand('staging/cuomo/mashB{{m}}/{params}/out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/CTcorr.png',

use rule cuomo_ong_waldNlrt_plot as cuomo_ong_waldNlrt_plot_mashB_CT with:
    input:
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/out.npy',
    output:
        png = f'results/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/waldNlrt.png',

use rule cuomo_ctng_test as cuomo_ctng_test_mashB_CT with:
    input:
        y_batch = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        imputed_ct_y = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # donor - day * gene
        nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        imputed_ct_nu = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene 
        P = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        pca = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'staging/cuomo/data/meta.mashB{m}.txt', 
    output:
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng.out.txt',
    params:
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctng/rep/out.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,
        REML = True,
        HE = True,
        HE_as_initial = False,

use rule cuomo_ong_test_mergeBatches as cuomo_ctng_test_mergeBatches_mashB_CT with:
    input:
        out = [f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctng.out.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',

use rule cuomo_ong_waldNlrt_plot as  cuomo_ctng_waldNlrt_plot_mashB_CT with:
    input:
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.waldNlrt.png',

use rule cuomo_ong_corr_plot as cuomo_ctng_corr_plot_mashB_CT with:
    input:
        base = expand('staging/cuomo/mashB{{m}}/{params}/ctng.out.npy', 
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'staging/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.out.npy',
    output:
        png = f'results/cuomo/mashB{{m}}/{cuomo_paramspace.wildcard_pattern}/ctng.CTcorr.png',

rule cuomo_mashB_CT_all:
    input:
        ong_CTcorr = expand('results/cuomo/mashB{m}/{params}/CTcorr.png', 
                m=range(3), params=cuomo_paramspace.instance_patterns),
        ctng_CTcorr = expand('results/cuomo/mashB{m}/{params}/ctng.CTcorr.png', 
                m=range(3), params=cuomo_paramspace.instance_patterns),
        ong_wald = expand('results/cuomo/mashB{m}/{params}/waldNlrt.png',
                m=range(3), params=cuomo_paramspace.instance_patterns),
        ctng_wald = expand('results/cuomo/mashB{m}/{params}/ctng.waldNlrt.png',
                m=range(3), params=cuomo_paramspace.instance_patterns),
################ imputation accuracy ###########################
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

#use rule cuomo_day_imputeNinputForONG as cuomo_imputation_day_imputeNinputForONG with:
# no standardization of y and nu
rule cuomo_imputation_day_imputeNinputForONG:
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
        #nu_ctng = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/nu.ctng.txt', # list
        #P = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/P.txt', # list
        #imputed_ct_y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/ct.y.txt', # donor - day * gene
        #imputed_ct_nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/ct.nu.txt', #donor-day * gene # negative ct_nu set to 0
        #imputed_ct_nu_ctng = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
        y = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        nu = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
        nu_ctng = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        P = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_ct_y = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/ct.y.txt', # list 
        imputed_ct_nu = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.txt', # list # negative ct_nu set to 0
        imputed_ct_nu_ctng = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', # list  # negative ct_nu set to max(ct_nu)
    resources: 
        time= '200:00:00',
        mem = lambda wildcards: '15gb' if wildcards.im_mvn == 'N' else '5gb',
    script: 'bin/cuomo_imputation_day_imputeNinputForONG.py'

rule cuomo_imputation_day_cleanfile:
    input:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        #nu = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/nu.txt', # list
        #nu_ctng = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/nu.ctng.txt', # list
        #P = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/P.txt', # list
        #imputed_ct_nu_ctng = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/batch{{i}}/ct.nu.ctng.txt', #donor-day * gene # negative ct_nu set to max(ct_nu)
        y = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/y.txt', # list # y for each gene is sorted by ind order
        nu = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/nu.txt', # list
        nu_ctng = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/nu.ctng.txt', # list
        P = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/P.txt', # list
        imputed_ct_nu_ctng = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/{cuomo_imput_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctng.txt', # list * gene # negative ct_nu set to max(ct_nu)
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
################ replicate Cuomo et al Figure 1b ###############
batch_no = 1000

rule cuomo_figure1b_generateinput:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz', 
    output:
        y = expand('staging/cuomo/figure1b/sample_{{k}}/rep_{{j}}/y.batch{i}.txt', i=range(batch_no)),
        y_dir = directory('staging/cuomo/figure1b/sample_{k}/rep_{j}/y'),
        donor = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/donor.gz',
        P = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/day.prop.gz',
        experiment = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/experiment.gz',
    script: 'bin/cuomo_figure1b_generateinput.py'

rule cuomo_figure1b_pca:
    input:
        y = expand('staging/cuomo/figure1b/sample_{{k}}/rep_{{j}}/y.batch{i}.txt', i=range(batch_no)),
        P = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/day.prop.gz',
    output:
        png = 'analysis/cuomo/figure1b/sample_{k}/rep_{j}/pca.png',
    script: 'bin/cuomo_figure1b_pca.py'

rule cuomo_figure1b_test:
    input:
        y = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/y.batch{i}.txt',
        donor = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/donor.gz',
        P = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/day.prop.gz',
        experiment = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/experiment.gz',
    output:
        out = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/out.batch{i}.txt',
        out_dir = directory('staging/cuomo/figure1b/sample_{k}/rep_{j}/out.batch{i}'),
    params:
        options = lambda wildcards: [] if int(wildcards.k)==5000 else ['null','null2','null3','null4','null5','null6'],
    resources: 
        time = '124:00:00', 
        mem = '10gb',
    script: 'bin/cuomo_figure1b_test.py'

rule cuomo_figure1b_aggout:
    input:
        out = expand('staging/cuomo/figure1b/sample_{{k}}/rep_{{j}}/out.batch{i}.txt', i=range(batch_no)),
    output:
        out = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/out.npy',
    resources: mem = '10gb'
    script: 'bin/cuomo_figure1b_aggout.py'

rule cuomo_figure1b_plot:
    input:
        out = 'staging/cuomo/figure1b/sample_{k}/rep_{j}/out.npy',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz', 
        supp = 'data/cuomo2020natcommun/suppdata14.txt',
    output:
        png = 'analysis/cuomo/figure1b/sample_{k}/rep_{j}/out.png',
    script: 'bin/cuomo_figure1b_plot.py'

rule cuomo_figure1b_all:
    input:
        #png = expand('analysis/cuomo/figure1b/sample_{k}/rep_{j}/out.png', zip, 
        #        k=[500,500,500,500],j=[1,2,3,4]),
        png = expand('analysis/cuomo/figure1b/sample_{k}/rep_{j}/out.png', zip,  k=[5000,5000,5000,5000],j=[1,2,3,4]),

rule cuomo_figure1b_acrossREPplot:
    input:
        out = expand('staging/cuomo/figure1b/sample_{{k}}/rep_{j}/out.npy', j=[1,2,3,4]),
    output:
        png = 'analysis/cuomo/figure1b/sample_{k}/crossREP.png',
    script: 'bin/cuomo_figure1b_acrossREPplot.py'

rule cuomo_figure1b_acrossREPplot_all:
    input:
        png = expand('analysis/cuomo/figure1b/sample_{k}/crossREP.png', k=['5000']),

###
#largesample_batch_no = len(gzip.open('data/cuomo2020natcommun/log_normalised_counts.csv.gz','rt').readlines())-1
#print(largesample_batch_no)
#
#use rule cuomo_figure1b_generateinput as cuomo_figure1b_generateinput_largesample with:
#    output:
#        y = expand('staging/cuomo/figure1b/largesample/sample_{{k}}/rep_{{j}}/y.batch{i}.txt', 
#                i=range(largesample_batch_no)),
#        y_dir = directory('staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/y'),
#        donor = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/donor.gz',
#        P = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/day.prop.gz',
#        experiment = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/experiment.gz',
#
#use rule cuomo_figure1b_pca as cuomo_figure1b_pca_largesample with:
#    input:
#        y = expand('staging/cuomo/figure1b/largesample/sample_{{k}}/rep_{{j}}/y.batch{i}.txt', 
#                i=range(largesample_batch_no)),
#        P = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/day.prop.gz',
#    output:
#        png = 'analysis/cuomo/figure1b/largesample/sample_{k}/rep_{j}/pca.png',
#
#use rule cuomo_figure1b_test as cuomo_figure1b_test_largesample with:
#    input:
#        y = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/y.batch{i}.txt',
#        donor = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/donor.gz',
#        P = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/day.prop.gz',
#        experiment = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/experiment.gz',
#    output:
#        out = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/out.batch{i}.txt',
#        out_dir = directory('staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/out.batch{i}'),
#    params:
#        options = ['null5','null6'],
#    resources: 
#        time = '300:00:00', 
#        mem = '20gb',
#
#use rule cuomo_figure1b_aggout as cuomo_figure1b_aggout_largesample with:
#    input:
#        out = expand('staging/cuomo/figure1b/largesample/sample_{{k}}/rep_{{j}}/out.batch{i}.txt', 
#                i=range(largesample_batch_no)),
#    output:
#        out = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/out.npy',
#
#use rule cuomo_figure1b_plot as cuomo_figure1b_plot_largesample with:
#    input:
#        out = 'staging/cuomo/figure1b/largesample/sample_{k}/rep_{j}/out.npy',
#        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz', 
#        supp = 'data/cuomo2020natcommun/suppdata14.txt',
#    output:
#        png = 'results/cuomo/figure1b/largesample/sample_{k}/rep_{j}/out.png',
#
#rule cuomo_figure1b_all_largesample:
#    input:
#        png = expand('results/cuomo/figure1b/largesample/sample_{k}/rep_{j}/out.png', zip,
#                k=[5000], j=[1]),
#

################ replicate Cuomo et al Figure 1c ###############
rule cuomo_figure1c_pca:
    input:
        gene = 'analysis/cuomo/figure1c/highly_variable_gene.txt',
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz', 
    output:
        png = 'results/cuomo/figure1c/pca.png',
    resources: mem = '5gb',
    script: 'bin/cuomo_figure1c_pca.py'

rule cuomo_figure1c_pcaALLgene:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz', 
    output:
        png = 'results/cuomo/figure1c/pca.all.png',
    resources: mem = '5gb',
    script: 'bin/cuomo_figure1c_pcaALLgene.py'

###################################################################################
# simlation with real nu from Cuomo et al 
###################################################################################
ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
        dtype="str", comment='#', na_filter=False)
if ctngVSong_NUfromCuomo_params.shape[0] != ctngVSong_NUfromCuomo_params.drop_duplicates().shape[0]:
    sys.exit('Duplicated parameters!\n')
par_columns = list(ctngVSong_NUfromCuomo_params.columns)
par_columns.remove('model') # columns after removing 'model'
ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")

ctngVSong_NUfromCuomo_replicates = 1000
ctngVSong_NUfromCuomo_batchsize = 20
ctngVSong_NUfromCuomo_batches = [range(i, min(i+ctngVSong_NUfromCuomo_batchsize, ctngVSong_NUfromCuomo_replicates)) 
        for i in range(0, ctngVSong_NUfromCuomo_replicates, ctngVSong_NUfromCuomo_batchsize)]

use rule ong_celltype_expectedPInSnBETAnV as ctngVSong_NUfromCuomo_celltype_expectedPInSnBETAnV with:
    output:
        pi = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/PI.txt',
        s = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/S.txt',
        beta = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/V.txt',
    params:
        simulation=ctngVSong_NUfromCuomo_paramspace.instance,

localrules: ctngVSong_NUfromCuomo_generatedata_batch
rule ctngVSong_NUfromCuomo_generatedata_batch:
    output: touch(f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/generatedata.batch')

for _, batch in enumerate(ctngVSong_NUfromCuomo_batches):
    rule: # generate simulation data for each batch
        name: f'ctngVSong_NUfromCuomo_generatedata_batch{_}'
        input:
            flag = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/generatedata.batch',
            beta = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/celltypebeta.txt',
            V = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/V.txt',
            imputed_ct_nu_ctng = expand('staging/cuomo/ind_min_cellnum~100_ct_min_cellnum~10_im_miny~N_im_scale~Y_im_genome~Y_im_mvn~N_sex~Y_PC~11_experiment~R_disease~Y/batch{i}/ct.nu.ctng.txt', 
                    i=range(cuomo_batch_no)), #donor-day * gene # negative ct_nu set to max(ct_nu)
        output:
            P = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/P.batch{_}.txt',
            pi = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/estPI.batch{_}.txt',
            s = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/estS.batch{_}.txt',
            nu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/nu.batch{_}.txt',
            nu_withnoise = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/nu.withnoise.batch{_}.txt',
            y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/pseudobulk.batch{_}.txt',
            overall_nu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.nu.batch{_}.txt',
            overall_nu_withnoise = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.nu.withnoise.batch{_}.txt',
            overall_y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.pseudobulk.batch{_}.txt',
            fixed_M = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{_}.txt',
            random_M = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{_}.txt',
        params:
            P = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/P.txt' 
                    for i in batch],
            pi = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/estPI.txt' 
                    for i in batch],
            s = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/estS.txt' 
                    for i in batch],# sample prop cov matrix
            nu = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/nu.txt' 
                    for i in batch],
            nu_withnoise = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/nu.withnoise.txt' 
                    for i in batch],
            y = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/pseudobulk.txt' 
                    for i in batch],
            overall_nu = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/overall.nu.txt' 
                    for i in batch],
            overall_nu_withnoise = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/overall.nu.withnoise.txt' 
                    for i in batch],
            overall_y = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/overall.pseudobulk.txt' 
                    for i in batch],
            fixed_M = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/overall.fixedeffectmatrix.txt' 
                    for i in batch],
            random_M = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep{i}/overall.randomeffectmatrix.txt'
                    for i in batch],
        resources:
            burden = 20,
        run:
            # par
            beta = np.loadtxt(input.beta)
            V = np.loadtxt(input.V)
            hom2 = float(wildcards['vc'].split('_')[0]) # variance of individual effect
            #mean_nu = float(wildcards['vc'].split('_')[-1]) # mean variance for residual error acros individuals
            #var_nu = float(wildcards['var_nu']) #variance of variance for residual error across individuals
            a = np.array([float(x) for x in wildcards['a'].split('_')])
            ss = int(float(wildcards['ss']))
            C = len(wildcards['a'].split('_'))
            rng = np.random.default_rng()

            for P_f, pi_f, s_f, y_f, nu_f, nu_withnoise_f, overall_y_f, overall_nu_f, overall_nu_withnoise_f, fixed_M_f, random_M_f in zip(
                    params.P, params.pi, params.s, params.y, params.nu, params.nu_withnoise, params.overall_y, 
                    params.overall_nu, params.overall_nu_withnoise,
                    params.fixed_M, params.random_M):
                #for P_f, pi_f, s_f, y_f, nu_f in zip(
                #    params.P, params.pi, params.s, params.y, params.nu):
                os.makedirs(os.path.dirname(P_f), exist_ok=True)
                # simulate cell type proportions
                P = rng.dirichlet(alpha=a, size=ss)
                np.savetxt(P_f, P, delimiter='\t')
                pi = np.mean(P, axis=0)
                np.savetxt(pi_f, pi, delimiter='\t')

                # estimate cov matrix S
                ## demeaning P 
                pdemean = P-pi
                ## covariance
                s = (pdemean.T @ pdemean)/ss
                #print(bmatrix(s))
                np.savetxt(s_f, s, delimiter='\t')

                # draw alpha / hom effect
                alpha_overall = rng.normal(loc=0, scale=math.sqrt(hom2), size=(ss))
                alpha = np.outer(alpha_overall, np.ones(C))

                # draw gamma (interaction)
                if wildcards.model != 'hom':
                    gamma = rng.multivariate_normal(np.zeros(C), V, size=ss) 
                    interaction = np.sum(P * gamma, axis=1)

                # draw residual error
                ## draw variance of residual error for each individual from gamma distribution \Gamma(k, theta)
                ## with mean = k * theta, var = k * theta^2, so theta = var / mean, k = mean / theta
                ## since mean = 0.25 and assume var = 0.01, we can get k and theta
                #theta = var_nu / mean_nu 
                #k = mean_nu / theta 
                ### variance of residual error for each individual
                #nu_overall = rng.gamma(k, scale=theta, size=ss)
                #np.savetxt(overall_nu_f, nu_overall, delimiter='\t')
                ### variance of residual error for each CT is in the ratio of the inverse of CT proportion
                #P_inv = 1 / P
                #nu = P_inv * nu_overall.reshape(-1,1)
                #np.savetxt(nu_f, nu, delimiter='\t')

                # cuomo nu
                ## random sample one gene's ct nu
                imputed_ct_nu_ctng_fs = []
                for f in input.imputed_ct_nu_ctng:
                    for line in open(f):
                        imputed_ct_nu_ctng_fs.append(line.strip())
                nu = pd.read_table( rng.choice(imputed_ct_nu_ctng_fs) )
                nu = nu.pivot(index='donor', columns='day')
                while np.any( (nu < 1e-12).sum(axis=1) > 1 ):
                    # remove genes with more than 1 cts with ctnu =0, otherwise hom and IID is gonna broken in CTNG
                    nu = pd.read_table( rng.choice(imputed_ct_nu_ctng_fs) )
                    nu = nu.pivot(index='donor', columns='day')
                nu = np.array(nu)
                ## random sample ss individuals 
                nu = np.array( [nu[x] for x in rng.choice(nu.shape[0], size=int(float(wildcards.ss)))] )
                if float(wildcards['vc'].split('_')[-1]) == 0:
                    nu = np.zeros_like(nu)
                if wildcards.model == 'hom2':
                    nu = rng.permutation( nu.flatten() ).reshape( nu.shape )
                    # if mor than two CTs of one individual have low nu, 
                    # hom model broken because of sigular variance matrix.
                    # to solve the problem, permute
                    threshold = 1e-10
                    while np.any( np.sum(nu < threshold, axis=1) > 1 ):
                        nu = rng.permutation(nu.flatten()).reshape(nu.shape)

                np.savetxt(nu_f, nu, delimiter='\t')
                nu_overall = np.sum(P * nu, axis=1)
                np.savetxt(overall_nu_f, nu_overall, delimiter='\t')

                ## add noise to nu
                prop = float(wildcards.nu_noise.split('_')[0])
                noise = np.zeros(len(nu.flatten()))
                noise[rng.choice(len(noise), int(len(noise)*prop), replace=False)] = rng.beta( 
                        a=float(wildcards.nu_noise.split('_')[1]), b=float(wildcards.nu_noise.split('_')[2]), 
                        size=int(len(noise)*prop) )
                noise = noise * rng.choice([-1,1], len(noise))
                nu_withnoise = nu * (1 + noise.reshape(nu.shape[0], nu.shape[1]))
                np.savetxt(nu_withnoise_f, nu_withnoise)
                nu_overall_withnoise = np.sum(P * nu_withnoise, axis=1)
                np.savetxt(overall_nu_withnoise_f, nu_overall_withnoise)

                ## draw residual error from normal distribution with variance drawn above
                delta_overall = rng.normal(np.zeros_like(nu_overall), np.sqrt(nu_overall))
                delta = rng.normal(np.zeros_like(nu), np.sqrt(nu))

                # generate pseudobulk
                if wildcards.model == 'hom':
                    y_overall = alpha_overall + P @ beta + delta_overall
                    y = alpha + np.outer(np.ones(ss), beta) + delta
                else:
                    y_overall = alpha_overall + P @ beta + interaction + delta_overall
                    y = alpha + np.outer(np.ones(ss), beta) + gamma + delta 

                # add fixed effect
                if 'fixed' in wildcards.keys():
                    levels = int(wildcards.fixed)
                    if levels > 0:
                        print(np.var(y_overall))
                        fixed = np.arange(levels)
                        fixed = fixed / np.std(fixed) # to set variance to 1
                        fixed_M = np.zeros( (len(y_overall), levels) )
                        j = 0
                        for i, chunk in enumerate( np.array_split(y_overall, levels)):
                            fixed_M[j:(j+len(chunk)), i] = 1
                            j = j + len(chunk)
                        # centralize fixed effect
                        fixed = fixed - np.mean(fixed_M @ fixed)

                        y_overall = y_overall + fixed_M @ fixed
                        print(np.var(y_overall))
                        y = y + (fixed_M @ fixed).reshape(-1,1)
                        # save fixed effect design matrix (get rid last column to avoid colinear)
                        np.savetxt(fixed_M_f, fixed_M[:,:-1], delimiter='\t')

                # add random effect
                if 'random' in wildcards.keys():
                    levels = int(wildcards.random)
                    if levels > 0:
                        print('variance', np.var(y_overall))
                        random_e = rng.normal(0, 1, levels)
                        random_M = np.zeros( (len(y_overall), levels) )
                        j = 0
                        for i, chunk in enumerate(np.array_split(y_overall, levels)):
                            random_M[j:(j+len(chunk)), i] = 1
                            j = j + len(chunk)
                        # centralize random effect
                        random_e = random_e - np.mean(random_M @ random_e)
                        ## double check it's centralized
                        if np.mean( random_M @ random_e) > 1e-3:
                            print('mean', np.mean( random_M @ random_e ) )
                            sys.exit('Centralization error!\n')

                        y_overall = y_overall + random_M @ random_e
                        print('variance', np.var(y_overall))
                        y = y + (random_M @ random_e).reshape(-1,1)
                        # save random effect design matrix
                        np.savetxt(random_M_f, random_M, delimiter='\t')

                # save
                np.savetxt(overall_y_f, y_overall, delimiter='\t')
                np.savetxt(y_f, y, delimiter='\t')

            with open(output.P, 'w') as f: f.write('\n'.join(params.P))
            with open(output.pi, 'w') as f: f.write('\n'.join(params.pi))
            with open(output.s, 'w') as f: f.write('\n'.join(params.s))
            with open(output.y, 'w') as f: f.write('\n'.join(params.y))
            with open(output.nu, 'w') as f: f.write('\n'.join(params.nu))
            with open(output.nu_withnoise, 'w') as f: f.write('\n'.join(params.nu_withnoise))
            with open(output.overall_y, 'w') as f: f.write('\n'.join(params.overall_y))
            with open(output.overall_nu, 'w') as f: f.write('\n'.join(params.overall_nu))
            with open(output.overall_nu_withnoise, 'w') as f: f.write('\n'.join(params.overall_nu_withnoise))
            with open(output.fixed_M, 'w') as f:
                if 'fixed' in wildcards.keys():
                    if int(wildcards.fixed) > 0:
                        f.write('\n'.join(params.fixed_M))
                    else:
                        f.write('NA')
                else:
                    f.write('NA')
            with open(output.random_M, 'w') as f:
                if 'random' in wildcards.keys():
                    if int(wildcards.random) > 0:
                        f.write('\n'.join(params.random_M))
                    else:
                        f.write('NA')
                else:
                    f.write('NA')

use rule ong_test as ctngVSong_NUfromCuomo_ongtest with:
    input:
        y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.nu.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.out.batch{{i}}',
    params:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep/overall.out.npy',
        batch = lambda wildcards: ctngVSong_NUfromCuomo_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,

use rule ctngVSong_ongAggReplications as ctngVSong_NUfromCuomo_ongAggReplications with:
    input:
        out = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.out.batch{i}' 
                for i in range(len(ctngVSong_NUfromCuomo_batches))],
    output:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.out.npy',

use rule ong_test as ctngVSong_NUfromCuomo_ongWithNoisetest with:
    input:
        y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.nu.withnoise.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.withnoise.out.batch{{i}}',
    params:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep/overall.withnoise.out.npy',
        batch = lambda wildcards: ctngVSong_NUfromCuomo_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,

use rule ctngVSong_ongAggReplications as ctngVSong_NUfromCuomo_ongWithNoiseAggReplications with:
    input:
        out = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.withnoise.out.batch{i}' 
                for i in range(len(ctngVSong_NUfromCuomo_batches))],
    output:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.withnoise.out.npy',

use rule ctng_test as ctngVSong_NUfromCuomo_ctngtest with:
    input:
        y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.batch{{i}}',
    params:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep/out.npy',
        batch = lambda wildcards: ctngVSong_NUfromCuomo_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,

use rule cuomo_ong_test_mergeBatches as ctngVSong_NUfromCuomo_ctng_aggReplications with:
    input:
        out = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(ctngVSong_NUfromCuomo_batches))],
    output:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.npy',

use rule ctng_test as ctngVSong_NUfromCuomo_ctngtest2 with:
    input:
        y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out2.batch{{i}}',
    params:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep/out2.npy',
        batch = lambda wildcards: ctngVSong_NUfromCuomo_batches[int(wildcards.i)],
        ML = False,
        REML = True,
        Free_reml_only = True,
        Free_reml_jk = True,
        HE = False,

use rule cuomo_ong_test_mergeBatches as ctngVSong_NUfromCuomo_ctng_aggReplications2 with:
    input:
        out = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out2.batch{i}' 
                for i in range(len(ctngVSong_NUfromCuomo_batches))],
    output:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out2.npy',

use rule ctng_test as ctngVSong_NUfromCuomo_ctngWithNoiseTest with:
    input:
        y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/nu.withnoise.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.batch{{i}}',
    params:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep/out.withnoise.npy',
        batch = lambda wildcards: ctngVSong_NUfromCuomo_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,

use rule cuomo_ong_test_mergeBatches as ctngVSong_NUfromCuomo_ctngWithNoise_aggReplications with:
    input:
        out = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.batch{i}' 
                for i in range(len(ctngVSong_NUfromCuomo_batches))],
    output:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.npy',

use rule ctng_test as ctngVSong_NUfromCuomo_ctngWithNoiseTest2 with:
    # reml JK
    input:
        y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/nu.withnoise.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out2.withnoise.batch{{i}}',
    params:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep/out2.withnoise.npy',
        batch = lambda wildcards: ctngVSong_NUfromCuomo_batches[int(wildcards.i)],
        ML = False,
        REML = True,
        Free_reml_only = True,
        Free_reml_jk = True,
        HE = False,

use rule cuomo_ong_test_mergeBatches as ctngVSong_NUfromCuomo_ctngWithNoise_aggReplications2 with:
    input:
        out = [f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out2.withnoise.batch{i}' 
                for i in range(len(ctngVSong_NUfromCuomo_batches))],
    output:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out2.withnoise.npy',

rule ctngVSong_NUfromCuomo_hom:
    input:
        ong = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.out.npy',
        ong_withnoise = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.withnoise.out.npy',
        ctng=f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.npy',
        ctng_withnoise = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.npy',
    output:
        png = f'results/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/ctngVSong.hom.png',
    script: 'bin/ctngVSong_NUfromCuomo_hom.py'

rule ctngVSong_NUfromCuomo_V:
    input:
        V = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/V.txt',
        ong = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.out.npy',
        ong_withnoise = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/overall.withnoise.out.npy',
        ctng=f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.npy',
        ctng_withnoise = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.npy',
    output:
        png = f'results/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/ctngVSong.V.png',
    script: 'bin/ctngVSong_NUfromCuomo_V.py'

def ctngVSong_NUfromCuomo_agg_hom_model_fun(wildcards):
    ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctngVSong_NUfromCuomo_params = ctngVSong_NUfromCuomo_params.loc[ctngVSong_NUfromCuomo_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_NUfromCuomo_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")
    return expand('results/ctngVSong_NUfromCuomo/{{model}}/{params}/ctngVSong.hom.png', 
            params=ctngVSong_NUfromCuomo_paramspace.instance_patterns)

def ctngVSong_NUfromCuomo_agg_V_model_fun(wildcards):
    ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctngVSong_NUfromCuomo_params = ctngVSong_NUfromCuomo_params.loc[ctngVSong_NUfromCuomo_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_NUfromCuomo_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")
    return expand('results/ctngVSong_NUfromCuomo/{{model}}/{params}/ctngVSong.V.png', 
            params=ctngVSong_NUfromCuomo_paramspace.instance_patterns)

rule ctngVSong_NUfromCuomo_agg_model:
    input: 
        hom = ctngVSong_NUfromCuomo_agg_hom_model_fun,
        V = ctngVSong_NUfromCuomo_agg_V_model_fun,
    output: touch('staging/ctngVSong_NUfromCuomo/{model}/ctngVSong.flag'),

def ctngVSong_NUfromCuomo_aggctng(wildcards):
    ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctngVSong_NUfromCuomo_params = ctngVSong_NUfromCuomo_params.loc[ctngVSong_NUfromCuomo_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_NUfromCuomo_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")
    return expand('analysis/ctngVSong_NUfromCuomo/{{model}}/{params}/out.npy', 
            params=ctngVSong_NUfromCuomo_paramspace.instance_patterns)

def ctngVSong_NUfromCuomo_aggctng2(wildcards):
    ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctngVSong_NUfromCuomo_params = ctngVSong_NUfromCuomo_params.loc[
            (ctngVSong_NUfromCuomo_params['model'] == wildcards.model) 
            & (ctngVSong_NUfromCuomo_params['nu_noise'] == '1_2_5')]
    par_columns = list(ctngVSong_NUfromCuomo_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")
    return expand('analysis/ctngVSong_NUfromCuomo/{{model}}/{params}/out2.npy', 
            params=ctngVSong_NUfromCuomo_paramspace.instance_patterns)

def ctngVSong_NUfromCuomo_aggctngwithNoise(wildcards):
    ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctngVSong_NUfromCuomo_params = ctngVSong_NUfromCuomo_params.loc[ctngVSong_NUfromCuomo_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_NUfromCuomo_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")
    return expand('analysis/ctngVSong_NUfromCuomo/{{model}}/{params}/out.withnoise.npy', 
            params=ctngVSong_NUfromCuomo_paramspace.instance_patterns)

def ctngVSong_NUfromCuomo_aggctngwithNoise2(wildcards):
    ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctngVSong_NUfromCuomo_params = ctngVSong_NUfromCuomo_params.loc[ctngVSong_NUfromCuomo_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_NUfromCuomo_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")
    return expand('analysis/ctngVSong_NUfromCuomo/{{model}}/{params}/out2.withnoise.npy', 
            params=ctngVSong_NUfromCuomo_paramspace.instance_patterns)

rule ctngVSong_NUfromCuomo_power:
    input:
        ctng = ctngVSong_NUfromCuomo_aggctng,
        ctng2 = ctngVSong_NUfromCuomo_aggctng2,
        ctng_withnoise = ctngVSong_NUfromCuomo_aggctngwithNoise,
        ctng_withnoise2 = ctngVSong_NUfromCuomo_aggctngwithNoise2,
    output:
        png = 'results/ctngVSong_NUfromCuomo/{model}/power.png',
    params:
        labels = ['1_2_5', '1_2_4', '1_2_3', '1_2_2'],
    script: 'bin/ctngVSong_NUfromCuomo_power.py'

rule ctngVSong_NUfromCuomo_all:
    input:
        ctngVSong = expand('staging/ctngVSong_NUfromCuomo/{model}/ctngVSong.flag', 
                model=['free','hom','hom2','hom3','hom4']),
        power = expand('results/ctngVSong_NUfromCuomo/{model}/power.png', 
                model=['free','hom','hom2','hom3','hom4']),

###################### CTNG p value: permute CT for each donor independently ######################
######################## permute projected Y rather then original, to keep fixed effects ##########
use rule ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect as ctngVSong_NUfromCuomo_ctng_permuteCT_holdfixedeffect with:
    input:
        y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        ctnu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/permuteCT_holdfixedeffect/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.batch{{i}}',
    params:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/permuteCT_holdfixedeffect/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep/out.npy',
        batch = lambda wildcards: ctngVSong_NUfromCuomo_batches[int(wildcards.i)],

use rule cuomo_ong_test_mergeBatches as ctngVSong_NUfromCuomo_ctng_permuteCT_holdfixedeffect_aggReplications with:
    input:
        out = [f'staging/ctngVSong_NUfromCuomo/{{model}}/permuteCT_holdfixedeffect/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.batch{i}' 
                for i in range(len(ctngVSong_NUfromCuomo_batches))],
    output:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.permuteCT.holdfixedeffect.npy',

use rule cuomo_ctng_HEpvalue_acrossmodel_plot as ctngVSong_NUfromCuomo_ctng_pvalueplot_permuteCT_holdfixedeffect with:
    input:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.permuteCT.holdfixedeffect.npy',
    output:
        png = f'results/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.permuteCT.holdfixedeffect.png',

def ctngVSong_NUfromCuomo_ctng_permuteCT_holdfixedeffect_agg_model_fun(wildcards):
    ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctngVSong_NUfromCuomo_params = ctngVSong_NUfromCuomo_params.loc[ctngVSong_NUfromCuomo_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_NUfromCuomo_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")
    return expand('results/ctngVSong_NUfromCuomo/{{model}}/{params}/out.permuteCT.holdfixedeffect.png', 
            params=ctngVSong_NUfromCuomo_paramspace.instance_patterns)

rule ctngVSong_NUfromCuomo_ctng_permutationDistribution_permuteCT_holdfixedeffect:
    input:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.permuteCT.holdfixedeffect.npy',
    output:
        png = f'results/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.permutationDistribution.permuteCT.holdfixedeffect.png',
    script: 'bin/ctngVSong_NUfromCuomo_ctng_permutationDistribution_permuteCT_holdfixedeffect.py'

def ctngVSong_NUfromCuomo_ctng_permutationDistribution_permuteCT_holdfixedeffect_agg_model_fun(wildcards):
    ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctngVSong_NUfromCuomo_params = ctngVSong_NUfromCuomo_params.loc[ctngVSong_NUfromCuomo_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_NUfromCuomo_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")
    return expand('results/ctngVSong_NUfromCuomo/{{model}}/{params}/out.permutationDistribution.permuteCT.holdfixedeffect.png', 
            params=ctngVSong_NUfromCuomo_paramspace.instance_patterns)

use rule ctngVSong_ctng_withCovars_permuteCT_holdfixedeffect as ctngVSong_NUfromCuomo_ctngWithNoise_permuteCT_holdfixedeffect with:
    input:
        y = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
        P = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        ctnu = f'staging/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/nu.withnoise.batch{{i}}.txt',
    output:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/permuteCT_holdfixedeffect/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.batch{{i}}',
    params:
        out = f'staging/ctngVSong_NUfromCuomo/{{model}}/permuteCT_holdfixedeffect/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/rep/out.withnoise.npy',
        batch = lambda wildcards: ctngVSong_NUfromCuomo_batches[int(wildcards.i)],

use rule cuomo_ong_test_mergeBatches as ctngVSong_NUfromCuomo_ctngWithNoise_permuteCT_holdfixedeffect_aggReplications with:
    input:
        out = [f'staging/ctngVSong_NUfromCuomo/{{model}}/permuteCT_holdfixedeffect/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.batch{i}' 
                for i in range(len(ctngVSong_NUfromCuomo_batches))],
    output:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.permuteCT.holdfixedeffect.npy',

use rule cuomo_ctng_HEpvalue_acrossmodel_plot as ctngVSong_NUfromCuomo_ctngWithNoise_pvalueplot_permuteCT_holdfixedeffect with:
    input:
        out = f'analysis/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.permuteCT.holdfixedeffect.npy',
    output:
        png = f'results/ctngVSong_NUfromCuomo/{{model}}/{ctngVSong_NUfromCuomo_paramspace.wildcard_pattern}/out.withnoise.permuteCT.holdfixedeffect.png',

def ctngVSong_NUfromCuomo_ctngWithNoise_permuteCT_holdfixedeffect_agg_model_fun(wildcards):
    ctngVSong_NUfromCuomo_params = pd.read_table("ctngVSong.NUfromCuomo.params.txt", 
            dtype="str", comment='#', na_filter=False)
    ctngVSong_NUfromCuomo_params = ctngVSong_NUfromCuomo_params.loc[ctngVSong_NUfromCuomo_params['model'] == wildcards.model]
    par_columns = list(ctngVSong_NUfromCuomo_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    ctngVSong_NUfromCuomo_paramspace = Paramspace(ctngVSong_NUfromCuomo_params[par_columns], filename_params="*")
    return expand('results/ctngVSong_NUfromCuomo/{{model}}/{params}/out.withnoise.permuteCT.holdfixedeffect.png', 
            params=ctngVSong_NUfromCuomo_paramspace.instance_patterns)

rule ctngVSong_NUfromCuomo_ctngWithNoise_permuteCT_holdfixedeffect_agg_model:
    input:
        ctng = ctngVSong_NUfromCuomo_ctng_permuteCT_holdfixedeffect_agg_model_fun,
        ctngwithnoise = ctngVSong_NUfromCuomo_ctngWithNoise_permuteCT_holdfixedeffect_agg_model_fun,
        permutation = ctngVSong_NUfromCuomo_ctng_permutationDistribution_permuteCT_holdfixedeffect_agg_model_fun,
    output: touch('staging/ctngVSong_NUfromCuomo/{model}/permuteCT_holdfixedeffect/ctng.flag'),

rule ctngVSong_NUfromCuomo_permuteCT_holdfixedeffect_all:
    input:
        ctngVSong = expand('staging/ctngVSong_NUfromCuomo/{model}/permuteCT_holdfixedeffect/ctng.flag', 
                model=['hom','free']),

###############################################################################
# paper
###############################################################################
#rule paper_modelview:
#    output:
#        png = 'paper/modelview.png',
#    script: 'bin/paper_modelview.py'

rule paper_ongNctng_REML_subspace_plot:
    input:
        ong_out = ong_agg_out_subspace,
        ong_V = ong_agg_trueV_subspace,
        ctng_out = ctng_agg_out_subspace,
    output:
        violin = 'results/paper/{model}/REML.AGG{arg}.violin.png',
        box = 'results/paper/{model}/REML.AGG{arg}.box.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: 'bin/paper_ongNctng_REML.py'

def ong_agg_hom_out_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, ong_params.loc[ong_params['model']=='hom'])
    return expand('analysis/ong/hom/{params}/out.npy', params=subspace.instance_patterns)

def ctng_agg_hom_out_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, ong_params.loc[ong_params['model']=='hom'])
    return expand('analysis/ctng/hom/{params}/out.npy', params=subspace.instance_patterns)

rule paper_ongNctng_REML_waldNlrt_subspace_plot:
    input:
        ong_out = ong_agg_out_subspace,
        ctng_out = ctng_agg_out_subspace,
        ong_hom_out = ong_agg_hom_out_subspace,
        ctng_hom_out = ctng_agg_hom_out_subspace,
    output:
        waldNlrt = 'results/paper/{model}/REML.waldNlrt.AGG{arg}.png',
    params: 
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']==wildcards.model]).iloc[:,:],
        hom_subspace = lambda wildcards: get_subspace(wildcards.arg,
                ong_params.loc[ong_params['model']=='hom']).iloc[:,:],
        plot_order = ong_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
    script: 'bin/paper_ongNctng_REML_waldNlrt_subspace_plot.py'

rule paper_ongNctng_REML_subspace_plot_all:
    input:
        estimates =  expand('results/paper/{model}/REML.AGG{arg}.box.png', 
                zip, model=['free'], arg=['ss']),
        waldNlrt = expand('results/paper/{model}/REML.waldNlrt.AGG{arg}.png', 
                zip, model=['free'], arg=['ss']),

###########
# NEW
###########
include: 'xCTMM.snake'
