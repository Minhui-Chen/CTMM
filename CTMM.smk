colorpalette='bright'
op_excluderareCT_plot_order = copy.deepcopy(op_plot_order)
for model in op_excluderareCT_plot_order.keys():
    op_excluderareCT_plot_order[model]['a'].remove('0.5_2_2_2')


################################################################
# OP
################################################################
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
    script: 'bin/sim/op.py'

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
    script: 'scripts/sim/compare_optim_RvsPython.py'

use rule op_test as op_test_Nelder with:
    output:
        out = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.Nelder.batch{{i}}',
    params:
        out = f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/rep/out.Nelder.npy',
        batch = lambda wildcards: op_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        method = 'BFGS-Nelder',

use rule op_aggReplications as op_Nelder_aggReplications with:
    input:
        out = [f'staging/op/{{model}}/{op_paramspace.wildcard_pattern}/out.Nelder.batch{i}'
                for i in range(len(op_batches))],
    output:
        out = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.Nelder.npy',

rule op_compare_optim_Nelder:
    input:
        r = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.npy',
        p = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/out.Nelder.npy',
    output:
        png = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/optim.BFGSvsBFGSNelder.supp.png',
    script: 'scripts/sim/op_compare_optim_Nelder.py'

def op_agg_out_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, op_params.loc[op_params['model']==wildcards.model])
    return expand('analysis/op/{{model}}/{params}/out.scipy.npy', params=subspace.instance_patterns)
#return expand('analysis/op/{{model}}/{params}/out.npy', params=subspace.instance_patterns)

def op_agg_truebeta_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, op_params.loc[op_params['model']==wildcards.model])
    return expand('analysis/op/{{model}}/{params}/celltypebeta.txt', params=subspace.instance_patterns)

def op_agg_trueV_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, op_params.loc[op_params['model']==wildcards.model])
    return expand('analysis/op/{{model}}/{params}/V.txt', params=subspace.instance_patterns)

rule op_MLestimates_subspace_plot:
    input:
        out = op_agg_out_subspace,
        beta = op_agg_truebeta_subspace,
        V = op_agg_trueV_subspace,
    output:
        png = 'results/op/{model}/ML.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
        method = 'ml',
    script: 'scripts/sim/op.estimates_subspace_plot.py'

rule op_MLwaldNlrt_subspace_plot:
    input:
        out = op_agg_out_subspace,
    output:
        waldNlrt = 'results/op/{model}/ML.waldNlrt.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
        method = 'ml',
    script: 'scripts/sim/op.waldNlrt_subspace_plot.py'

use rule op_MLestimates_subspace_plot as op_REMLestimates_subspace_plot with:
    output:
        png = 'results/op/{model}/REML.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
        method = 'reml',

use rule op_MLwaldNlrt_subspace_plot as op_REMLwaldNlrt_subspace_plot with:
    output:
        waldNlrt = 'results/op/{model}/REML.waldNlrt.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
        method = 'reml',

use rule op_MLestimates_subspace_plot as op_HEestimates_AGGsubspace_plot with:
    output:
        png = 'results/op/{model}/HE.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
        method = 'he',

use rule op_MLwaldNlrt_subspace_plot as op_HEwald_subspace_plot with:
    output:
        waldNlrt = 'results/op/{model}/HE.wald.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
        method = 'he',

def op_MLestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/op/{{model}}/ML.AGG{arg}.png', arg=effective_args)

def op_MLwaldNlrt_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/op/{{model}}/ML.waldNlrt.AGG{arg}.png', arg=effective_args)

def op_REMLestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/op/{{model}}/REML.AGG{arg}.png', arg=effective_args)

def op_REMLwaldNlrt_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/op/{{model}}/REML.waldNlrt.AGG{arg}.png', arg=effective_args)

def op_HEestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/op/{{model}}/HE.AGG{arg}.png', arg=effective_args)

def op_HEwald_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/op/{{model}}/HE.wald.AGG{arg}.png', arg=effective_args)

rule op_AGGarg:
    input:
        ML = op_MLestimates_AGGarg_fun,
        ML_waldNlrt = op_MLwaldNlrt_AGGarg_fun,
        REML = op_REMLestimates_AGGarg_fun,
        REML_waldNlrt = op_REMLwaldNlrt_AGGarg_fun,
        HE = op_HEestimates_AGGarg_fun,
        HE_wald = op_HEwald_AGGarg_fun,
    output:
        flag = touch('staging/op/{model}/all.flag'),

#############################################################################
# CTP
#############################################################################
use rule ctp_test as ctp_test_scipy with:
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

def ctp_agg_out_subspace(wildcards):
    subspace = get_subspace(wildcards.arg, op_params.loc[op_params['model']==wildcards.model])
    #op_params.to_csv(sys.stdout, sep='\t', index=False)
    #subspace.to_csv(sys.stdout, sep='\t', index=False)
    return expand('analysis/ctp/{{model}}/{params}/out.npy', params=subspace.instance_patterns)

use rule op_MLestimates_subspace_plot as ctp_MLestimates_subspace_plot with:
    input:
        out = ctp_agg_out_subspace,
        beta = op_agg_truebeta_subspace,
        V = op_agg_trueV_subspace,
    output:
        png = 'results/ctp/{model}/ML.AGG{arg}.png',

use rule op_MLestimates_subspace_plot as ctp_MLestimates_subspace_plot2 with:
    input:
        out = ctp_agg_out_subspace,
        beta = op_agg_truebeta_subspace,
        V = op_agg_trueV_subspace,
    output:
        png = 'results/ctp/{model}/ML.excluderareCT.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_excluderareCT_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
        method = 'ml',

use rule op_MLwaldNlrt_subspace_plot as ctp_MLwaldNlrt_subspace_plot with:
    input:
        out = ctp_agg_out_subspace,
    output:
        waldNlrt = 'results/ctp/{model}/ML.waldNlrt.AGG{arg}.png',

use rule op_MLwaldNlrt_subspace_plot as ctp_MLwaldNlrt_subspace_plot2 with:
    input:
        out = ctp_agg_out_subspace,
    output:
        waldNlrt = 'results/ctp/{model}/ML.excluderareCT.waldNlrt.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_excluderareCT_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,

use rule op_HEestimates_AGGsubspace_plot as ctp_HEestimates_AGGsubspace_plot with:
    input:
        out = ctp_agg_out_subspace,
        V = op_agg_trueV_subspace,
    output:
        png = 'results/ctp/{model}/HE.AGG{arg}.png',

use rule op_HEwald_subspace_plot as ctp_HEwald_subspace_plot with:
    input:
        out = ctp_agg_out_subspace,
    output:
        waldNlrt = 'results/ctp/{model}/HE.wald.AGG{arg}.png',

use rule op_MLestimates_subspace_plot as ctp_REMLestimates_subspace_plot with:
    input:
        out = ctp_agg_out_subspace,
        beta = op_agg_truebeta_subspace,
        V = op_agg_trueV_subspace,
    output:
        png = 'results/ctp/{model}/REML.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
        method = 'reml',

use rule op_MLestimates_subspace_plot as ctp_REMLestimates_subspace_plot2 with:
    input:
        out = ctp_agg_out_subspace,
        beta = op_agg_truebeta_subspace,
        V = op_agg_trueV_subspace,
    output:
        png = 'results/ctp/{model}/REML.excluderareCT.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_excluderareCT_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,
        method = 'reml',

use rule op_REMLwaldNlrt_subspace_plot as ctp_REMLwaldNlrt_subspace_plot with:
    input:
        out = ctp_agg_out_subspace,
    output:
        waldNlrt = 'results/ctp/{model}/REML.waldNlrt.AGG{arg}.png',

use rule op_REMLwaldNlrt_subspace_plot as ctp_REMLwaldNlrt_subspace_plot2 with:
    input:
        out = ctp_agg_out_subspace,
    output:
        waldNlrt = 'results/ctp/{model}/REML.excluderareCT.waldNlrt.AGG{arg}.png',
    params:
        subspace = lambda wildcards: get_subspace(wildcards.arg,
                op_params.loc[op_params['model']==wildcards.model]).iloc[:,:],
        plot_order = op_excluderareCT_plot_order,
        colorpalette = colorpalette,
        pointcolor = pointcolor,
        mycolors = mycolors,

def ctp_MLwaldNlrt_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/ctp/{{model}}/ML.waldNlrt.AGG{arg}.png', arg=effective_args)

def ctp_MLestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/ctp/{{model}}/ML.AGG{arg}.png', arg=effective_args)

def ctp_HEestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/ctp/{{model}}/HE.AGG{arg}.png', arg=effective_args)

def ctp_HEwald_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/ctp/{{model}}/HE.wald.AGG{arg}.png', arg=effective_args)

def ctp_REMLwaldNlrt_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/ctp/{{model}}/REML.waldNlrt.AGG{arg}.png', arg=effective_args)

def ctp_REMLestimates_AGGarg_fun(wildcards):
    effective_args = get_effective_args(op_params.loc[op_params['model']==wildcards.model])
    return expand('results/ctp/{{model}}/REML.AGG{arg}.png', arg=effective_args)

rule ctp_AGGarg:
    input:
        MLmodelestimates = ctp_MLestimates_AGGarg_fun,
        MLwaldNlrt = ctp_MLwaldNlrt_AGGarg_fun,
        HEestimates = ctp_HEestimates_AGGarg_fun,
        HEwald = ctp_HEwald_AGGarg_fun,
        REMLmodelestimates = ctp_REMLestimates_AGGarg_fun,
        REMLwaldNlrt = ctp_REMLwaldNlrt_AGGarg_fun,
        MLmodelestimates2 = 'results/ctp/{model}/ML.excluderareCT.AGGa.png',
        MLwaldNlrt2 = 'results/ctp/{model}/ML.excluderareCT.waldNlrt.AGGa.png',
        REMLmodelestimates2 = 'results/ctp/{model}/REML.excluderareCT.AGGa.png',
        REMLwaldNlrt2 = 'results/ctp/{model}/REML.excluderareCT.waldNlrt.AGGa.png',
    output:
        flag = touch('staging/ctp/{model}/all.flag'),

rule ctp_all:
    input:
        flag = expand('staging/ctp/{model}/all.flag', model=['hom','free', 'full']),

#rule op_mainfig_LRT:
#    input:
#        hom_ss = expand('analysis/op/hom/{params}/out.npy',
#                params=get_subspace('ss', op_params.loc[op_params['model']=='hom']).instance_patterns),
#        iid_ss = expand('analysis/op/iid/{params}/out.npy',
#                params=get_subspace('ss', op_params.loc[op_params['model']=='iid']).instance_patterns),
#        free_ss = expand('analysis/op/free/{params}/out.npy',
#                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
#        hom_a = expand('analysis/op/hom/{params}/out.npy',
#                params=get_subspace('a', op_params.loc[op_params['model']=='hom']).instance_patterns),
#        iid_a = expand('analysis/op/iid/{params}/out.npy',
#                params=get_subspace('a', op_params.loc[op_params['model']=='iid']).instance_patterns),
#        free_a = expand('analysis/op/free/{params}/out.npy',
#                params=get_subspace('a', op_params.loc[op_params['model']=='free']).instance_patterns),
#        #hom_vc = expand('analysis/op/hom/{params}/out.npy',
#        #        params=get_subspace('vc', op_params.loc[op_params['model']=='hom']).instance_patterns),
#        iid_vc = expand('analysis/op/iid/{params}/out.npy',
#                params=get_subspace('vc', op_params.loc[op_params['model']=='iid']).instance_patterns),
#        free_vc = expand('analysis/op/free/{params}/out.npy',
#                params=get_subspace('vc', op_params.loc[op_params['model']=='free']).instance_patterns),
#        free_V_diag = expand('analysis/op/free/{params}/out.npy',
#                params=get_subspace('V_diag', op_params.loc[op_params['model']=='free']).instance_patterns),
#    output:
#        png = 'results/op/mainfig.LRT.png',
#    params:
#        hom_ss = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom'])['ss']),
#        iid_ss = np.array(get_subspace('ss', op_params.loc[op_params['model']=='iid'])['ss']),
#        free_ss = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free'])['ss']),
#        hom_a = np.array(get_subspace('a', op_params.loc[op_params['model']=='hom'])['a']),
#        iid_a = np.array(get_subspace('a', op_params.loc[op_params['model']=='iid'])['a']),
#        free_a = np.array(get_subspace('a', op_params.loc[op_params['model']=='free'])['a']),
#        #hom_vc = np.array(get_subspace('vc', op_params.loc[op_params['model']=='hom'])['vc']),
#        iid_vc = np.array(get_subspace('vc', op_params.loc[op_params['model']=='iid'])['vc']),
#        free_vc = np.array(get_subspace('vc', op_params.loc[op_params['model']=='free'])['vc']),
#        free_V_diag = np.array(get_subspace('V_diag', op_params.loc[op_params['model']=='free'])['V_diag']),
#        plot_order = op_plot_order,
#    script: "bin/op_mainfig_LRT.py"
#
rule op_all:
    input:
        flag = expand('staging/op/{model}/all.flag', model=['null', 'hom', 'iid', 'free', 'full']),
        png = 'results/op/mainfig.LRT.png',


###############################################################
# supp figures 
###############################################################
hom_C_params = Paramspace(op_params.loc[op_params['model'] == 'hom2'].drop('model', axis=1), filename_params="*")
free_C_params = Paramspace(op_params.loc[op_params['model'] == 'free2'].drop('model', axis=1), filename_params="*")

rule paper_ctp_power:
    input:
        hom_ss = expand('analysis/ctp/hom/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom']).instance_patterns),
        hom_ss_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom']).instance_patterns),
        free_ss = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
        free_ss_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
        hom_a = expand('analysis/ctp/hom/{params}/out.npy',
                params=get_subspace('a', op_params.loc[op_params['model']=='hom']).instance_patterns),
        hom_a_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
                params=get_subspace('a', op_params.loc[op_params['model']=='hom']).instance_patterns),
        free_a = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('a', op_params.loc[op_params['model']=='free']).instance_patterns),
        free_a_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
                params=get_subspace('a', op_params.loc[op_params['model']=='free']).instance_patterns),
        hom_c = expand('analysis/ctp/hom2/{params}/out.npy', params=hom_C_params.instance_patterns),
        hom_c_remlJK = expand('analysis/ctp/hom2/{params}/out.remlJK.npy', params=hom_C_params.instance_patterns),
        free_c = expand('analysis/ctp/free2/{params}/out.npy', params=free_C_params.instance_patterns),
        free_c_remlJK = expand('analysis/ctp/free2/{params}/out.remlJK.npy', params=free_C_params.instance_patterns),
        free_vc = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('vc', op_params.loc[op_params['model']=='free']).instance_patterns),
        free_vc_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
                params=get_subspace('vc', op_params.loc[op_params['model']=='free']).instance_patterns),
    output:
        png = 'results/ctp/power.paper.supp.png',
    params:
        arg_ss = 'ss',
        hom_ss = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom'])['ss']),
        free_ss = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free'])['ss']),
        hom_ss_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom'])['ss']),
        free_ss_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free'])['ss']),
        arg_a = 'a',
        hom_a = np.array(get_subspace('a', op_params.loc[op_params['model']=='hom'])['a']),
        free_a = np.array(get_subspace('a', op_params.loc[op_params['model']=='free'])['a']),
        hom_a_remlJK = np.array(get_subspace('a', op_params.loc[op_params['model']=='hom'])['a']),
        free_a_remlJK = np.array(get_subspace('a', op_params.loc[op_params['model']=='free'])['a']),
        arg_c = 'c',
        hom_c = [len(x.split('_')) for x in hom_C_params['a'].to_list()],
        free_c = [len(x.split('_')) for x in free_C_params['a'].to_list()],
        plot_order_c = [4, 8, 12],
        arg_vc = 'vc',
        free_vc = np.array(get_subspace('vc', op_params.loc[op_params['model']=='free'])['vc']),
        free_vc_remlJK = np.array(get_subspace('vc', op_params.loc[op_params['model']=='free'])['vc']),
        plot_order = op_plot_order,
    script: 'scripts/sim/ctp_power.py'


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
        pointcolor = pointcolor,
    script: 'scripts/sim/opNctp_estimates_ss.paper.py'


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
        pointcolor = pointcolor,
    script: 'scripts/sim/opNctp_estimates_a.paper.py'


rule paper_opNctp_estimates_C:
    input:
        op_free = expand('analysis/op/free2/{params}/out.npy',
                params=free_C_params.instance_patterns),
        ctp_free = expand('analysis/ctp/free2/{params}/out.npy',
                params=free_C_params.instance_patterns),
        V = expand('analysis/op/free2/{params}/V.txt',
                params=free_C_params.instance_patterns),
    output:
        png = 'results/ctp/ctp.estimate.C.paper.supp.png',
    params: 
        hom = [len(x.split('_')) for x in hom_C_params['a'].to_list()],
        free = [len(x.split('_')) for x in free_C_params['a'].to_list()],
        plot_order = [4, 8, 12],
        pointcolor = pointcolor,
    script: 'scripts/sim/opNctp_estimates_C.paper.py'


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
        pointcolor = pointcolor,
    script: 'scripts/sim/opNctp_estimates_vc.paper.py'


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
        pointcolor = pointcolor,
        vc = get_subspace('ss', op_params.loc[op_params['model']=='full'])['vc'][0],
    script: 'scripts/sim/opNctp_estimates_ss_full.paper.py'


use rule ctp_test as ctp_iid with:
    output:
        out = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.iid.batch{{i}}',
    params:
        out = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/rep/out.iid.npy',
        batch = lambda wildcards: ctp_batches[int(wildcards.i)],
        ML = False,
        REML = True,
        HE = True,
        he_free_jk = False,
        he_iid_jk = True,
        optim_by_R = True,
    resources:
        mem_mb = '5gb',
        time = '148:00:00',


use rule op_aggReplications as ctp_iid_aggReplications with:
    input:
        out = [f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.iid.batch{i}' 
                for i in range(len(ctp_batches))],
    output:
        out = f'analysis/ctp/{{model}}/{op_paramspace.wildcard_pattern}/out.iid.npy',


rule paper_ctp_iid_power:
    input:
        ctp_hom = expand('analysis/ctp/hom/{params}/out.iid.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom']).instance_patterns),
        ctp_free = expand('analysis/ctp/free/{params}/out.iid.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free']).instance_patterns),
    output:
        png = 'results/ctp/ctp.iid.power.paper.supp.png',
    params: 
        hom = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom'])['ss']),
        free = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free'])['ss']),
        plot_order = op_plot_order,
    script: 'scripts/sim/ctp_iid_power.py'


rule paper_ctp_C_power:
    input:
        ctp_hom = expand('analysis/ctp/hom2/{params}/out.npy',
                params=hom_C_params.instance_patterns),
        ctp_hom_remlJK = expand('analysis/ctp/hom2/{params}/out.remlJK.npy',
                params=hom_C_params.instance_patterns),
        ctp_free = expand('analysis/ctp/free2/{params}/out.npy',
                params=free_C_params.instance_patterns),
        ctp_free_remlJK = expand('analysis/ctp/free2/{params}/out.remlJK.npy',
                params=free_C_params.instance_patterns),
    output:
        png = 'results/ctp/ctp.C.power.paper.supp.png',
    params: 
        hom = [len(x.split('_')) for x in hom_C_params['a'].to_list()],
        free = [len(x.split('_')) for x in free_C_params['a'].to_list()],
        plot_order = [4, 8, 12],
    script: 'scripts/sim/ctp_C_power.py'


rule opNctp_supp_all:
    input:
        power_ctp = 'results/ctp/power.paper.supp.png',
        estimates_ss = 'results/ctp/opNctp.estimate.ss.paper.supp.png',
        estimates_a = 'results/ctp/opNctp.estimate.a.paper.supp.png',
        estimates_vc = 'results/ctp/opNctp.estimate.vc.paper.supp.png',
        estiamtes_ss_full = 'results/ctp/opNctp.estimate.ss.full.paper.supp.png',
        iid_power = 'results/ctp/ctp.iid.power.paper.supp.png',
        C_power = 'results/ctp/ctp.C.power.paper.supp.png',




# #####################################################################################
# # test: restrict P_ic > 1%. if P_ic is too small, v_ic would be very large
# #####################################################################################

# rule ctp_pcut_simulation:
#     input:
#         beta = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/celltypebeta.txt',
#         V = f'analysis/op/{{model}}/{op_paramspace.wildcard_pattern}/V.txt',
#     output:
#         P = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
#         pi = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/estPI.batch{{i}}.txt',
#         s = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/estS.batch{{i}}.txt',
#         nu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
#         y = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
#         ctnu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
#         cty = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
#         fixed = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/fixed.X.batch{{i}}.txt',
#         random = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/random.X.batch{{i}}.txt',
#     params:
#         batch = lambda wildcards: ctp_batches[int(wildcards.i)],
#         P = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/P.txt',
#         pi = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/estPI.txt',
#         s = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/estS.txt',
#         nu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/nu.txt',
#         y = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/y.txt',
#         ctnu = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/ctnu.txt',
#         cty = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/cty.txt',
#         fixed = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/fixed.X.txt',
#         random = f'staging/ctp/{{model}}/{op_paramspace.wildcard_pattern}/repX/random.X.txt',
#         seed = 376487,
#     script: 'scripts/sim/opNctp_simulation.py'


#####################################################################################
# find simulation parameters
#####################################################################################
use rule paper_opNctp_power as paper_opNctp_power3 with:
    input:
        op_hom = expand('analysis/op/hom/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom3']).instance_patterns),
        op_free = expand('analysis/op/free/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free3']).instance_patterns),
        op_hom_remlJK = expand('analysis/op/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom3']).instance_patterns),
        op_free_remlJK = expand('analysis/op/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free3']).instance_patterns),
        ctp_hom = expand('analysis/ctp/hom/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom3']).instance_patterns),
        ctp_free = expand('analysis/ctp/free/{params}/out.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free3']).instance_patterns),
        ctp_hom_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='hom3']).instance_patterns),
        ctp_free_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
                params=get_subspace('ss', op_params.loc[op_params['model']=='free3']).instance_patterns),
    output:
        png = 'results/ctp/opNctp.power.sim3.paper.png',
    params: 
        hom = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom3'])['ss']),
        free = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free3'])['ss']),
        hom_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom3'])['ss']),
        free_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free3'])['ss']),
        plot_order = op_plot_order,


# use rule paper_opNctp_power as paper_opNctp_power4 with:
#     input:
#         op_hom = expand('analysis/op/hom/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom4']).instance_patterns),
#         op_free = expand('analysis/op/free/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free4']).instance_patterns),
#         op_hom_remlJK = expand('analysis/op/hom/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom4']).instance_patterns),
#         op_free_remlJK = expand('analysis/op/free/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free4']).instance_patterns),
#         ctp_hom = expand('analysis/ctp/hom/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom4']).instance_patterns),
#         ctp_free = expand('analysis/ctp/free/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free4']).instance_patterns),
#         ctp_hom_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom4']).instance_patterns),
#         ctp_free_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free4']).instance_patterns),
#     output:
#         png = 'results/ctp/opNctp.power.sim4.paper.png',
#     params: 
#         hom = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom4'])['ss']),
#         free = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free4'])['ss']),
#         hom_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom4'])['ss']),
#         free_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free4'])['ss']),
#         plot_order = op_plot_order,


# use rule paper_opNctp_power as paper_opNctp_power5 with:
#     input:
#         op_hom = expand('analysis/op/hom/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom5']).instance_patterns),
#         op_free = expand('analysis/op/free/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free5']).instance_patterns),
#         op_hom_remlJK = expand('analysis/op/hom/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom5']).instance_patterns),
#         op_free_remlJK = expand('analysis/op/free/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free5']).instance_patterns),
#         ctp_hom = expand('analysis/ctp/hom/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom5']).instance_patterns),
#         ctp_free = expand('analysis/ctp/free/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free5']).instance_patterns),
#         ctp_hom_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom5']).instance_patterns),
#         ctp_free_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free5']).instance_patterns),
#     output:
#         png = 'results/ctp/opNctp.power.sim5.paper.png',
#     params: 
#         hom = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom5'])['ss']),
#         free = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free5'])['ss']),
#         hom_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom5'])['ss']),
#         free_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free5'])['ss']),
#         plot_order = op_plot_order,


# use rule paper_opNctp_power as paper_opNctp_power6 with:
#     input:
#         op_hom = expand('analysis/op/hom/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom6']).instance_patterns),
#         op_free = expand('analysis/op/free/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free6']).instance_patterns),
#         op_hom_remlJK = expand('analysis/op/hom/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom6']).instance_patterns),
#         op_free_remlJK = expand('analysis/op/free/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free6']).instance_patterns),
#         ctp_hom = expand('analysis/ctp/hom/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom6']).instance_patterns),
#         ctp_free = expand('analysis/ctp/free/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free6']).instance_patterns),
#         ctp_hom_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom6']).instance_patterns),
#         ctp_free_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free6']).instance_patterns),
#     output:
#         png = 'results/ctp/opNctp.power.sim6.paper.png',
#     params: 
#         hom = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom6'])['ss']),
#         free = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free6'])['ss']),
#         hom_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom6'])['ss']),
#         free_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free6'])['ss']),
#         plot_order = op_plot_order,


# use rule paper_opNctp_power as paper_opNctp_power7 with:
#     input:
#         op_hom = expand('analysis/op/hom/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom7']).instance_patterns),
#         op_free = expand('analysis/op/free/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free7']).instance_patterns),
#         op_hom_remlJK = expand('analysis/op/hom/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom7']).instance_patterns),
#         op_free_remlJK = expand('analysis/op/free/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free7']).instance_patterns),
#         ctp_hom = expand('analysis/ctp/hom/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom7']).instance_patterns),
#         ctp_free = expand('analysis/ctp/free/{params}/out.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free7']).instance_patterns),
#         ctp_hom_remlJK = expand('analysis/ctp/hom/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='hom7']).instance_patterns),
#         ctp_free_remlJK = expand('analysis/ctp/free/{params}/out.remlJK.npy',
#                 params=get_subspace('ss', op_params.loc[op_params['model']=='free7']).instance_patterns),
#     output:
#         png = 'results/ctp/opNctp.power.sim7.paper.png',
#     params: 
#         hom = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom7'])['ss']),
#         free = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free7'])['ss']),
#         hom_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='hom7'])['ss']),
#         free_remlJK = np.array(get_subspace('ss', op_params.loc[op_params['model']=='free7'])['ss']),
#         plot_order = op_plot_order,

#####################################################################################
# test CTP and OP with covariates
#####################################################################################
cov_test_params = pd.read_table("cov_test.params.txt", dtype="str", comment='#', na_filter=False)
if cov_test_params.shape[0] != cov_test_params.drop_duplicates().shape[0]:
    sys.exit('Duplicated parameters!\n')
par_columns = list(cov_test_params.columns)
par_columns.remove('model') # columns after removing 'model'
cov_test_paramspace = Paramspace(cov_test_params[par_columns], filename_params="*")

cov_test_replicates = 1000
cov_test_batchsize = 2
cov_test_batches = [range(i, min(i+cov_test_batchsize, cov_test_replicates))
        for i in range(0, cov_test_replicates, cov_test_batchsize)]

use rule op_parameters as cov_test_parameters with:
    output:
        pi = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/PI.txt',
        s = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/S.txt',
        beta = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/V.txt',

use rule op_simulation as cov_test_simulation with:
    input:
        beta = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/celltypebeta.txt',
        V = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/V.txt',
    output:
        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        pi = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/estPI.batch{{i}}.txt',
        s = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/estS.batch{{i}}.txt',
        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
        ctnu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
        cty = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        # add a test fixed effect. if it's not needed, this file is 'NA'
        fixed = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/fixed.X.batch{{i}}.txt',
        # add a test random effect. if it's not needed, this file is 'NA'
        random = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/random.X.batch{{i}}.txt',
    params:
        batch = lambda wildcards: cov_test_batches[int(wildcards.i)],
        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/repX/P.txt',
        pi = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/repX/estPI.txt',
        s = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/repX/estS.txt',
        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/repX/nu.txt',
        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/repX/y.txt',
        ctnu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/repX/ctnu.txt',
        cty = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/repX/cty.txt',
        fixed = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/repX/fixed.X.txt',
        random = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/repX/random.X.txt',

use rule op_test as cov_test_op_test with:
    input:
        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
    output:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.batch{{i}}',
    params:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/rep/op.npy',
        batch = lambda wildcards: cov_test_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        HE_as_initial = False,

use rule op_aggReplications as cov_test_op_aggReplications with:
    input:
        out = [f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.batch{i}'
                for i in range(len(cov_test_batches))],
    output:
        out = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.npy',

use rule ctp_test as cov_test_ctp_test with:
    input:
        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.batch{{i}}',
    params:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/rep/ctp.npy',
        batch = lambda wildcards: cov_test_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,

use rule op_aggReplications as cov_test_ctp_aggReplications with:
    input:
        out = [f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.batch{i}'
                for i in range(len(cov_test_batches))],
    output:
        out = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.npy',

rule cov_test_op_withCovars:
    input:
        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        fixed = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/fixed.X.batch{{i}}.txt',
        random = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/random.X.batch{{i}}.txt',
    output:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.batch{{i}}',
    params:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/rep/op.covars.npy',
        batch = lambda wildcards: cov_test_batches[int(wildcards.i)],
    resources:
        mem = '5gb',
        time = '48:00:00',
    script: 'scripts/sim/cov_test_optest.py'

use rule op_aggReplications as cov_test_op_withCovars_aggReplications with:
    input:
        out = [f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.batch{i}'
                for i in range(len(cov_test_batches))],
    output:
        out = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.npy',

rule cov_test_op_withCovars_scipy:
    input:
        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/y.batch{{i}}.txt',
        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        ctnu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
        fixed = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/fixed.X.batch{{i}}.txt',
        random = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/random.X.batch{{i}}.txt',
    output:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.scipy.batch{{i}}',
    params:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/rep/op.covars.scipy.npy',
        batch = lambda wildcards: cov_test_batches[int(wildcards.i)],
    resources:
        mem = '5gb',
        time = '48:00:00',
    script: 'scripts/sim/cov_test_optest.scipy.py'

use rule op_aggReplications as cov_test_op_withCovars_scipy_aggReplications with:
    input:
        out = [f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.scipy.batch{i}'
                for i in range(len(cov_test_batches))],
    output:
        out = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.scipy.npy',

use rule op_compare_optim_RvsPython as cov_test_op_compare_optim_RvsPython with:
    input:
        r = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.npy',
        p = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.scipy.npy',
    output:
        png = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.optim.RvsPython.png',


#use rule cov_test_op_withCovars as cov_test_ongtest_withCovars_usingHEinit with:
#    input:
#        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.pseudobulk.batch{{i}}.txt',
#        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
#        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.nu.batch{{i}}.txt',
#        fixed_M = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{{i}}.txt',
#        random_M = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{{i}}.txt',
#    output:
#        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.HEinit.out.batch{{i}}',
#    params:
#        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/rep/overall.HEinit.out.npy',
#        batch = lambda wildcards: cov_test_batches[int(wildcards.i)],
#        HE_as_initial = True,
#    resources:
#        mem = '5gb',
#        time = '48:00:00',
#
#rule cov_test_ong_withCovars_usingHEinit_AggReplications:
#    input:
#        out = [f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.HEinit.out.batch{i}'
#                for i in range(len(cov_test_batches))],
#    output:
#        out = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.HEinit.out.npy',
#    script: "bin/mergeBatches.py"
#
rule cov_test_ctp_withCovars:
    input:
        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        ctnu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
        fixed = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/fixed.X.batch{{i}}.txt',
        random = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/random.X.batch{{i}}.txt',
    output:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.batch{{i}}',
    params:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/rep/ctp.covars.npy',
        batch = lambda wildcards: cov_test_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        HE_as_initial = False,
    resources:
        mem = '8gb',
        time = '48:00:00',
    script: 'scripts/sim/cov_test_ctptest.py'

use rule op_aggReplications as cov_test_ctp_withCovars_aggReplications with:
    input:
        out = [f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.batch{i}'
                for i in range(len(cov_test_batches))],
    output:
        out = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.npy',

rule cov_test_ctp_withCovars_scipy:
    input:
        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/cty.batch{{i}}.txt',
        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
        ctnu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctnu.batch{{i}}.txt',
        fixed = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/fixed.X.batch{{i}}.txt',
        random = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/random.X.batch{{i}}.txt',
    output:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.scipy.batch{{i}}',
    params:
        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/rep/ctp.covars.scipy.npy',
        batch = lambda wildcards: cov_test_batches[int(wildcards.i)],
        ML = True,
        REML = True,
        HE = True,
        optim_by_r = False,
        method = 'BFGS',
    resources:
        time = '48:00:00',
    script: 'scripts/sim/cov_test_ctptest.scipy.py'

use rule op_aggReplications as cov_test_ctp_withCovars_scipy_aggReplications with:
    input:
        out = [f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.scipy.batch{i}'
                for i in range(len(cov_test_batches))],
    output:
        out = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.scipy.npy',

use rule op_compare_optim_RvsPython as cov_test_compare_optim_RvsPython with:
    input:
        r = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.npy',
        p = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.scipy.npy',
    output:
        png = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/optim.RvsPython.png',


#use rule cov_test_ctp_withCovars as cov_test_ctngtest_withCovars_usingHEinit with:
#    input:
#        y = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/pseudobulk.batch{{i}}.txt',
#        P = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/P.batch{{i}}.txt',
#        nu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.nu.batch{{i}}.txt',
#        ctnu = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/nu.batch{{i}}.txt',
#        fixed_M = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.fixedeffectmatrix.batch{{i}}.txt',
#        random_M = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/overall.randomeffectmatrix.batch{{i}}.txt',
#    output:
#        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/HEinit.out.batch{{i}}',
#    params:
#        out = f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/rep/HEinit.out.npy',
#        batch = lambda wildcards: cov_test_batches[int(wildcards.i)],
#        ML = True,
#        REML = True,
#        HE = True,
#        HE_as_initial = True,
#    resources:
#        mem = '8gb',
#        time = '48:00:00',
#
#rule cov_test_ctng_withCovars_usingHEinit_aggReplications:
#    input:
#        out = [f'staging/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/HEinit.out.batch{i}'
#                for i in range(len(cov_test_batches))],
#    output:
#        out = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/HEinit.out.npy',
#    script: "bin/mergeBatches.py"
#
rule cov_test_hom:
    input:
        op = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.npy',
        ctp = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.npy',
        op_covar = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.npy',
        ctp_covar = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.npy',
    output:
        png = f'results/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/cov_test.hom.png',
    script: 'scripts/sim/cov_test_hom.py'

rule cov_test_V:
    input:
        op = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.npy',
        ctp = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.npy',
        op_covar = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/op.covars.npy',
        ctp_covar = f'analysis/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/ctp.covars.npy',
    output:
        png = f'results/cov_test/{{model}}/{cov_test_paramspace.wildcard_pattern}/cov_test.V.png',
    script: 'scripts/sim/cov_test_V.py'

def cov_test_agg_hom_model_fun(wildcards):
    cov_test_params = pd.read_table("cov_test.params.txt", dtype="str", comment='#', na_filter=False)
    cov_test_params = cov_test_params.loc[cov_test_params['model'] == wildcards.model]
    par_columns = list(cov_test_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    cov_test_paramspace = Paramspace(cov_test_params[par_columns], filename_params="*")
    return expand('results/cov_test/{{model}}/{params}/cov_test.hom.png',
            params=cov_test_paramspace.instance_patterns)

def cov_test_agg_V_model_fun(wildcards):
    cov_test_params = pd.read_table("cov_test.params.txt", dtype="str", comment='#', na_filter=False)
    cov_test_params = cov_test_params.loc[cov_test_params['model'] == wildcards.model]
    par_columns = list(cov_test_params.columns)
    par_columns.remove('model') # columns after removing 'model'
    cov_test_paramspace = Paramspace(cov_test_params[par_columns], filename_params="*")
    return expand('results/cov_test/{{model}}/{params}/cov_test.V.png',
            params=cov_test_paramspace.instance_patterns)

rule cov_test_agg_hom_model:
    input:
        hom = cov_test_agg_hom_model_fun,
        V = cov_test_agg_V_model_fun,
    output: touch('staging/cov_test/{model}/cov_test.flag'),

rule cov_test_all:
    input:
        cov_test = expand('staging/cov_test/{model}/cov_test.flag', model=['hom','free']),


###################################################################################################
# Cuomo
###################################################################################################
rule cuomo_P_plot:
    input:
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/P.png',
    run:
        P = pd.read_table(input.P, index_col=0)

        plt.rcParams.update({'font.size' : 6})
        fig, ax = plt.subplots(figsize=(8,4), dpi=600)
        sns.violinplot(data=P, scale='width', cut=0)
        ax.axhline(y=0, color='0.9', ls='--', zorder=0)
        ax.set_xlabel('Cell type', fontsize=10)
        ax.set_ylabel('Cell type proportion', fontsize=10)
        plt.tight_layout()
        fig.savefig(output.png)


rule cuomo_size_factor:
    input:
        meta = 'data/cuomo2020natcommun/cell_metadata_cols.tsv',
        raw = 'data/cuomo2020natcommun/raw_counts.csv.gz',
        log = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        size_factor = 'analysis/cuomo/data/size_factor.gz',
        png = 'results/cuomo/data/size_factor.png',
    run:
        meta = pd.read_table(input.meta, index_col=0)
        meta = meta.loc[meta['experiment'] == 'expt_09']
        raw = pd.read_table(input.raw, index_col=0, nrows=10)
        log = pd.read_table(input.log, index_col=0, nrows=10)


        gene = raw.index[0]
        raw = raw.loc[gene]
        log = log.loc[gene]

        data = pd.merge(raw, log, left_index=True, right_index=True)
        print(data.shape)
        print(data.head())
        print(meta.shape)
        print(meta['total_counts_endogenous'])
        data = data.merge(meta['total_counts_endogenous'], left_index=True, right_index=True)
        print(data.shape)
        data.columns = ['raw', 'log', 'total_counts']
        data.index.name = 'cell'

        f = lambda row: (row['raw'] * 1e6) / ((2**row['log'] - 1) * row['total_counts'])
        data['size factor'] = data.apply(f, axis=1)

        data.to_csv(output.size_factor, sep='\t')

        fig, ax = plt.subplots()
        ax.hist(data['size factor'], bins=100)
        fig.savefig(output.png)


rule cuomo_metadata_columns:
    input:
        meta = 'data/cuomo2020natcommun/cell_metadata_cols.tsv',
    output:
        info = 'data/cuomo2020natcommun/data.info',
    script: 'scripts/cuomo/data_explore.py'


rule cuomo_cellnum_summary:
    input:
        meta = 'data/cuomo2020natcommun/cell_metadata_cols.tsv',
    output:
        summary = 'analysis/cuomo/data/cellnum.txt',
        png = 'analysis/cuomo/data/cellnum.png',
        png2 = 'analysis/cuomo/data/cellnum2.png',
    script: 'scripts/cuomo/cellnum_summary.py'


rule cuomo_varaince_nu:
    input:
        nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/nu.ctp.txt'
                for i in range(cuomo_batch_no)],
    output:
        var = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/nu.ctp.var.txt',
    run:
        vars = []
        for f1 in input.nu:
            for f2 in open(f1):
                nu = np.loadtxt(f2.strip())
                vars.append(np.var(nu))
        np.savetxt(output.var, vars)


rule cuomo_varNU_dist:
    input:
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
    output:
        png = 'results/cuomo/data/log/bootstrapedNU/day.raw.var_nu.png',
    script: 'scripts/cuomo/varNU_dist.py'


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
    script: 'scripts/cuomo/day_summary_imputation.py'

rule cuomo_day_PCassociatedVar:
    input:
        pca = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.txt',
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        supp = f'data/cuomo2020natcommun/suppdata2.txt', # sex disease
        meta = 'analysis/cuomo/data/meta.txt', # experiment
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/pca.associatedVar.png',
    script: 'scripts/cuomo/day_PCassociatedVar.py'

rule cuomo_op_corr_plot:
    input:
        base = expand('analysis/cuomo/{params}/op.out.npy',
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.CTcorr.png',
    script: 'scripts/cuomo/op_corr_plot.py'

rule cuomo_op_rVariance_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.rVariance.png',
    script: 'scripts/cuomo/op_rVariance_plot.py'

rule cuomo_op_variance_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.vc.png',
    params:
        cut_off = {'free':[-1.5,2], 'full':[-3,3]},
    script: 'scripts/cuomo/op_variance_plot.py'

rule cuomo_op_waldNlrt_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.waldNlrt.png',
    script: 'scripts/cuomo/op_waldNlrt_plot.py'

rule cuomo_op_experimentR_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.hom.png',
    script: 'scripts/cuomo/op_experimentR_plot.py'

#rule cuomo_op_experimentR_all:
#    input:
#        png = expand('results/cuomo/{params}/op.hom.png',
#                params=Paramspace(cuomo_params.loc[cuomo_params['experiment']=='R'], filename_params="*").instance_patterns),

use rule cuomo_op_waldNlrt_plot as  cuomo_ctp_waldNlrt_plot with:
    # when using LRT test p value in Free REML
    # rule cuomo_ctp_waldNlrt_plot:
    # when using Wald test p value in Free REML
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.waldNlrt.png',
    #script: 'scripts/cuomo/ctp_waldNlrt_plot.py'

use rule cuomo_op_corr_plot as cuomo_ctp_corr_plot with:
    input:
        base = expand('analysis/cuomo/{params}/ctp.npy',
                params=Paramspace(cuomo_params.iloc[[0]], filename_params="*").instance_patterns),
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.CTcorr.png',

use rule cuomo_op_rVariance_plot as cuomo_ctp_rVariance_plot with:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.rVariance.png',

rule cuomo_ctp_variance_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        nu_ctp = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/nu.ctp.txt'
                for i in range(cuomo_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.vc.png',
    params:
        free = ['hom', 'CT_main', 'ct_random_var', 'nu'],
        cut_off = {'free':[-0.5,0.5], 'full':[-3,3]},
    script: 'scripts/cuomo/ctp_variance_plot.py'

rule cuomo_ctp_Vplot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.V.png',
    script: 'scripts/cuomo/ctp_Vplot.py'

use rule cuomo_ctp_test as cuomo_ctp_test_miny with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.miny.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.miny.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = True,
        REML = True,
        HE = True,
        jack_knife = True,
    resources:
        mem = '10gb',
        time = '48:00:00',

rule cuomo_ctp_HEpvalue_acrossmodel_plot:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.HEpvalue.png',
    script: 'scripts/cuomo/ctp_HEpvalue_acrossmodel_plot.py'

#use rule op_aggReplications as cuomo_ctp_miny_aggReplications with:
#    input:
#        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.miny.txt'
#                for i in range(cuomo_batch_no)],
#    output:
#        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.npy',
#
#use rule cuomo_ctp_waldNlrt_plot as cuomo_ctp_waldNlrt_plot_miny with:
#    input:
#        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.npy',
#    output:
#        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.waldNlrt.png',
#
#use rule cuomo_op_rVariance_plot as cuomo_ctp_rVariance_plot_miny with:
#    input:
#        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.npy',
#    output:
#        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.miny.rVariance.png',

rule cuomo_ctp_pvalue_REMLvsHE:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.free.REMLvsHE.inflated_zeros_{{prop}}.png',
    script: 'scripts/cuomo/ctp_pvalue_REMLvsHE.py'

rule cuomo_ctp_pvalue_REMLvsHE_addrVariance:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        y = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.free.REMLvsHE.rVariance.png',
    script: 'scripts/cuomo/ctp_pvalue_REMLvsHE_addrVariance.py'

rule cuomo_ng_all:
    input:
        imputation = expand('analysis/cuomo/{params}/imputation.png',
                params=cuomo_paramspace.instance_patterns),
        op_CTcorr = expand('results/cuomo/{params}/CTcorr.png',
                params=cuomo_paramspace.instance_patterns),
        ctp_CTcorr = expand('results/cuomo/{params}/ctp.CTcorr.png',
                params=cuomo_paramspace.instance_patterns),
        ctp_rVar = expand('results/cuomo/{params}/ctp.rVariance.png',
                params=cuomo_paramspace.instance_patterns),
        pca = expand('results/cuomo/{params}/pca.associatedVar.png',
                params=cuomo_paramspace.instance_patterns),
        op_wald = expand('results/cuomo/{params}/waldNlrt.png',
                params=cuomo_paramspace.instance_patterns),
        ctp_wald = expand('results/cuomo/{params}/ctp.waldNlrt.png',
                params=cuomo_paramspace.instance_patterns),
        ctp_REMLvsHE = expand('results/cuomo/{params}/ctp.free.REMLvsHE.inflated_zeros_1.png',
                params=cuomo_paramspace.instance_patterns),

rule cuomo_sc_expressionpattern:
    # single cell expression pattern plot
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
        imputed_ct_y = [f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ct.y.txt'
                for i in range(cuomo_batch_no)], # donor - day * gene
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/ctp.{{gene}}.png',
    params:
        mycolors = mycolors,
        paper = True,
    script: 'scripts/cuomo/sc_expressionpattern.py'

rule cuomo_sc_expressionpattern_collect:
    input:
        png = [f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/ctp.{gene}.png'
                for gene in ['ENSG00000111704_NANOG', 'ENSG00000141448_GATA6', 'ENSG00000204531_POU5F1',
                    'ENSG00000181449_SOX2', 'ENSG00000065518_NDUFB4', 'ENSG00000074047_GLI2', 'ENSG00000136997_MYC',
                    'ENSG00000125845_BMP2', 'ENSG00000107984_DKK1', 'ENSG00000234964_FABP5P7',
                    'ENSG00000166105_GLB1L3', 'ENSG00000237550_UBE2Q2P6', 'ENSG00000230903_RPL9P8']],
    output:
        flag = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.sc_expressionpattern.flag',
    shell:
        'touch {output.flag}'

# likelihood
rule cuomo_likelihood_plot:
    input:
        imputed_ct_nu = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ct.nu.ctp.txt'
                for i in range(cuomo_batch_no)], #donor-day * gene
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.likelihood.png',
    script: 'scripts/cuomo/likelihood_plot.py'

# p value across test methods: HE, ML, REML, Wald, JK
rule cuomo_ctp_p:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        out2 = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp2.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
    output:
        # hom2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.hom2.png',
        p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.p.png',
    script: 'scripts/cuomo/ctp_p.py'
    

rule cuomo_opVSctp_p:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
    output:
        p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/opVSctp.p.png',
    script: 'scripts/cuomo/opVSctp_p.py'

# find top genes
rule cuomo_geneP:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        p = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/genes/ctp.{{gene}}.P.txt',
    script: 'scripts/cuomo/geneP.py'

rule cuomo_topgenes:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
        ctp = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
    output:
        topgenes = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/opNctp.topgenes.txt',
    params:
        op = ['reml', 'he'],
        ctp = ['remlJK', 'he'],
    script: 'scripts/cuomo/topgenes.py'

rule cuomo_op_pvalue_paper:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.p.supp.png',
    script: 'scripts/cuomo/op.p.paper.py'

rule cuomo_freeNfull_Variance_paper:
    input:
        op = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/op.npy',
        ctp = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        op = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.freeNfull.Variance.supp.png',
        op_free_data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.freeNfull.Variance.supp.free_source_data.txt',
        op_full_data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/op.freeNfull.Variance.supp.full_source_data.txt',
        ctp = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.supp.png',
        ctp_free_data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.supp.free_source_data.txt',
        ctp_full_data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.freeNfull.Variance.supp.full_source_data.txt',
    script: 'bin/cuomo/freeNfull_Variance.supp.py'

#rule cuomo_ctp_corr_paper:
#    input:
#        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
#    output:
#        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.CTcorr.paper.png',
#    script: 'scripts/cuomo/ctp_corr.paper.py'

################# compare OLD and NEW HE fun ####################
rule cuomo_ctp_test_compareHE_old:
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
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.old.HE.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.old.HE.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,
        REML = False,
        HE = True,
        jack_knife = True,
        IID = True,
        Hom = True,
        optim_by_r = True,
    resources:
        mem = '10gb',
        time = '48:00:00',
    script: 'scripts/cuomo/ctp_test.py'

use rule op_aggReplications as cuomo_ctp_compareHE_old_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.old.HE.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.old.HE.npy',

use rule cuomo_ctp_test as cuomo_ctp_test_compareHE_new with:
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
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.new.HE.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.new.HE.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,
        REML = False,
        HE = True,
        jack_knife = True,
        IID = True,
        Hom = True,
        optim_by_r = True,
    resources:
        mem = '10gb',
        time = '48:00:00',

use rule op_aggReplications as cuomo_ctp_compareHE_new_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.new.HE.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.new.HE.npy',

use rule op_compare_optim_RvsPython as cuomo_ctp_compareHE_oldVSnew with:
    input:
        r = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.old.HE.npy',
        p = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.new.HE.npy',
    output:
        png = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.HE.oldVSnew.png',





###################################################################################################
# Cuomo: test HE free gls / ols test for mean differentiation
###################################################################################################
rule cuomo_ctp_ols:
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
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.ols.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.ols.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,  
        REML = False,
        HE = True, 
        jack_knife = False,
        he_free_ols = True,
        IID = False,
        Hom = False,
        optim_by_R = True,
    resources: 
        mem_mb = '10gb',
        time = '48:00:00',
    script: 'scripts/cuomo/ctp.py'


use rule op_aggReplications as cuomo_ctp_ols_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.ols.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.ols.npy',


use rule cuomo_ctp_ols as cuomo_ctp_olsjk with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.olsjk.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.olsjk.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,  
        REML = False,
        HE = True, 
        jack_knife = True,
        he_free_ols_jk = True,
        IID = False,
        Hom = False,
        optim_by_R = True, 


use rule op_aggReplications as cuomo_ctp_olsjk_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.olsjk.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.olsjk.npy',


use rule cuomo_ctp_ols as cuomo_ctp_gls with:
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/ctp.gls.txt',
    params:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.gls.npy',
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        ML = False,  
        REML = False,
        HE = True, 
        jack_knife = False,
        he_free_gls = True,
        IID = False,
        Hom = False,
        optim_by_R = True, 


use rule op_aggReplications as cuomo_ctp_gls_aggReplications with:
    input:
        out = [f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/batch{i}/ctp.gls.txt'
                for i in range(cuomo_batch_no)],
    output:
        out = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.gls.npy',


rule cuomo_ls_betacompare:
    input:
        ols = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.ols.npy',
        ols_jk = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.olsjk.npy',
        gls = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.gls.npy',
        gls_jk = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.ls.beta.png',
    script: 'scripts/yazar/betacompare.py'



##########################################################################################
# GO pathway enrichment analysis
##########################################################################################
rule cuomo_enrichment:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
    output:
        genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/all.genes.txt',
        reml_beta_bot_genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.bot.genes.txt',
        he_beta_bot_genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.bot.genes.txt',
        reml_V_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.topgenes.txt',
        reml_V_merge = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.merge.genes.txt',
        reml_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.topgenes.txt',
        reml_beta_merge = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.merge.genes.txt',
        reml_V_sig_beta_bot = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.sig.beta.botgenes.txt',
        reml_hom_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.hom.topgenes.txt',
        he_V_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.topgenes.txt',
        he_V_merge = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.merge.genes.txt',
        he_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.topgenes.txt',
        he_beta_merge = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.merge.genes.txt',
        he_V_sig_beta_bot = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.sig.beta.botgenes.txt',
        he_hom_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.hom.topgenes.txt',
    script: 'scripts/cuomo/homvsfree_genes.py'


rule cuomo_enrichment_GOnPathway:
    input:
        genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/all.genes.txt',
        reml_beta_bot_genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.bot.genes.txt',
        he_beta_bot_genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.bot.genes.txt',
        reml_V_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.topgenes.txt',
        reml_V_merge = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.merge.genes.txt',
        reml_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.topgenes.txt',
        reml_beta_merge = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.merge.genes.txt',
        reml_V_sig_beta_bot = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.sig.beta.botgenes.txt',
        reml_hom_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.hom.topgenes.txt',
        he_V_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.topgenes.txt',
        he_V_merge = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.merge.genes.txt',
        he_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.topgenes.txt',
        he_beta_merge = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.merge.genes.txt',
        he_V_sig_beta_bot = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.sig.beta.botgenes.txt',
        he_hom_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.hom.topgenes.txt',
    output:
        reml_V_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.beta.topgenes.txt',
        he_V_beta_top = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.beta.topgenes.txt',
        reml_V_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.top.enrich.txt',
        reml_V_merge_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.merge.enrich.txt',
        reml_beta_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.top.enrich.txt',
        reml_beta_merge_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.merge.enrich.txt',
        reml_V_beta_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.beta.top.enrich.txt',
        reml_V_sig_beta_bot_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.sig.beta.bot.enrich.txt',
        reml_V_sig_beta_bot_enrich2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.sig.beta.bot.enrich2.txt',
        reml_hom_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.hom.top.enrich.txt',
        he_V_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.top.enrich.txt',
        he_V_merge_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.merge.enrich.txt',
        he_beta_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.top.enrich.txt',
        he_beta_merge_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.merge.enrich.txt',
        he_V_beta_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.beta.top.enrich.txt',
        he_V_sig_beta_bot_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.sig.beta.bot.enrich.txt',
        he_V_sig_beta_bot_enrich2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.sig.beta.bot.enrich2.txt',
        he_hom_top_enrich = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.hom.top.enrich.txt',
        reml_V_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.top.gse.txt',
        reml_V_merge_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.merge.gse.txt',
        reml_beta_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.top.gse.txt',
        reml_beta_merge_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.beta.merge.gse.txt',
        reml_V_beta_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.beta.top.gse.txt',
        reml_V_sig_beta_bot_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.sig.beta.bot.gse.txt',
        reml_V_sig_beta_bot_gse2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V.sig.beta.bot.gse2.txt',
        reml_hom_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.hom.top.gse.txt',
        he_V_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.top.gse.txt',
        he_V_merge_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.merge.gse.txt',
        he_beta_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.top.gse.txt',
        he_beta_merge_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.beta.merge.gse.txt',
        he_V_beta_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.beta.top.gse.txt',
        he_V_sig_beta_bot_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.sig.beta.bot.gse.txt',
        he_V_sig_beta_bot_gse2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.V.sig.beta.bot.gse2.txt',
        he_hom_top_gse = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/he.hom.top.gse.txt',
    resources:
        mem_mb = '10gb',
    shell:
        '''
        cat {input.reml_V_top} {input.reml_beta_top} > {output.reml_V_beta_top}
        cat {input.he_V_top} {input.he_beta_top} > {output.he_V_beta_top}

        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.reml_V_top} {input.genes} \
                {output.reml_V_top_enrich} {output.reml_V_top_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.reml_V_merge} {input.genes} \
                {output.reml_V_merge_enrich} {output.reml_V_merge_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.reml_beta_top} {input.genes} \
                {output.reml_beta_top_enrich} {output.reml_beta_top_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.reml_beta_merge} {input.genes} \
                {output.reml_beta_merge_enrich} {output.reml_beta_merge_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.reml_V_sig_beta_bot} {input.reml_beta_bot_genes} \
                {output.reml_V_sig_beta_bot_enrich} {output.reml_V_sig_beta_bot_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.reml_V_sig_beta_bot} {input.genes} \
                {output.reml_V_sig_beta_bot_enrich2} {output.reml_V_sig_beta_bot_gse2}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.reml_hom_top} {input.genes} \
                {output.reml_hom_top_enrich} {output.reml_hom_top_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.he_V_top} {input.genes} \
                {output.he_V_top_enrich} {output.he_V_top_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.he_V_merge} {input.genes} \
                {output.he_V_merge_enrich} {output.he_V_merge_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.he_beta_top} {input.genes} \
                {output.he_beta_top_enrich} {output.he_beta_top_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.he_beta_merge} {input.genes} \
                {output.he_beta_merge_enrich} {output.he_beta_merge_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.he_V_sig_beta_bot} {input.he_beta_bot_genes} \
                {output.he_V_sig_beta_bot_enrich} {output.he_V_sig_beta_bot_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.he_V_sig_beta_bot} {input.genes} \
                {output.he_V_sig_beta_bot_enrich2} {output.he_V_sig_beta_bot_gse2}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {input.he_hom_top} {input.genes} \
                {output.he_hom_top_enrich} {output.he_hom_top_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {output.reml_V_beta_top} {input.genes} \
                {output.reml_V_beta_top_enrich} {output.reml_V_beta_top_gse}; \
        Rscript scripts/cuomo/ctp_enrichment_GOnPathway.R {output.he_V_beta_top} {input.genes} \
                {output.he_V_beta_top_enrich} {output.he_V_beta_top_gse}; \
        '''


# rule cuomo_homvsfree_enrichment_CountGO:
#     input:
#         genes = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/all.genes.txt',
#         reml_fdr_bon = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/reml.V_fdr10.beta_bon.genes.txt',
#     output:
#         go = temp(f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/all.genes.GOannotation.txt'),
#         png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/enrichment/count_GOannotaion_pergene.txt',
#     shell:
#         '''
#         module load R/4.0.3
#         Rscript scripts/cuomo/ctp_enrichment_CountGO.R {input.genes} {output.go}
#         python3 scripts/cuomo/ctp_enrichment_CountGO.py {output.go} {input.reml_fdr_bon} {output.png}
#         '''


rule cuomo_enrichment_all:
    input:
        reml_V = expand('results/cuomo/{params}/enrichment/reml.V.top.enrich.txt',
                params=cuomo_paramspace.instance_patterns),



rule yazar_enrichment:
    input:
        out = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        genes = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/enrichment/all.genes.txt',
        he_V_top = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/enrichment/he.V.topgenes.txt',
    script: 'scripts/yazar/homvsfree_genes.py'


rule yazar_enrichment_GOnPathway:
    input:
        genes = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/enrichment/all.genes.txt',
        he_V_top = f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/enrichment/he.V.topgenes.txt',
    output:
        he_V_top_enrich = f'results/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/enrichment/he.V.top.enrich.txt',
        he_V_top_gse = f'results/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/enrichment/he.V.top.gse.txt',
    resources:
        mem_mb = '10gb',
    shell:
        '''
        Rscript scripts/yazar/ctp_enrichment_GOnPathway.R {input.he_V_top} {input.genes} \
                {output.he_V_top_enrich} {output.he_V_top_gse}
        '''


rule yazar_enrichment_all:
    input:
        he_V = expand('results/yazar/nomissing/{params}/enrichment/he.V.top.enrich.txt',
                params=yazar_paramspace.instance_patterns),



##########################################################################
#  correlation with gene features
##########################################################################
rule cuomo_eds:
    input:
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz', 
        eds = 'data/Wang_Goldstein.tableS1.txt',
        gnomad = 'data/gnomad.v2.1.1.lof_metrics.by_gene.txt.gz',
        tf = 'data/Dey2022CellGenomics/Dey_Enhancer_MasterReg/genesets/TF_genes_curated.txt',
    output:
        eds = 'analysis/cuomo/eds.txt',
    run:
        counts = pd.read_table(input.counts, index_col=0)
        eds = pd.read_table(input.eds)

        print(counts.shape, eds.shape)
        print(len(np.unique(counts.index)))
        print(len(np.unique(eds['GeneSymbol'])))

        # TODO: drop dozens of duplicated genes
        eds = eds.drop_duplicates(subset=['GeneSymbol'], keep=False, ignore_index=True)

        eds = eds.loc[eds['GeneSymbol'].isin(counts.index.str.split('_').str.get(0))]
        # print(counts.index.str.split('_').str.get(0))

        # drop pLI in eds, instead using pLI from gnomad
        eds = eds.drop('pLI', axis=1)

        # read gene length from gnomad
        gnomad = pd.read_table(input.gnomad, usecols=['gene', 'gene_id', 'gene_length', 'pLI', 'oe_lof_upper'])
        gnomad = gnomad.rename(columns={'gene_id': 'GeneSymbol', 'oe_lof_upper': 'LOEUF'})
        # lose dozens of genes that are not exist in gnomad
        eds = eds.merge(gnomad[['GeneSymbol', 'gene_length', 'pLI', 'LOEUF']])

        # read TF
        tf = [line.strip().split()[0] for line in open(input.tf)]
        tf_gnomad = gnomad.loc[gnomad['gene'].isin(tf)]
        # check how many genes in tf has gene_id  # TODO: gnomad has the same gene name but different id
        print(len(tf), tf_gnomad.shape[0], len(tf_gnomad['gene'].unique()), len(tf_gnomad['GeneSymbol'].unique()))

        eds['TF'] = 0
        eds.loc[eds['GeneSymbol'].isin(tf_gnomad['GeneSymbol']), 'TF'] = 1


        eds.to_csv(output.eds, sep='\t', index=False)


rule cuomo_eds_cor:
    input:
        y = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
        eds = 'analysis/cuomo/eds.txt',
    output: 
        matrix = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/matrix.cor.png',
    params:
        method = 'reml',
    priority: 1
    script: 'scripts/cuomo/eds.cor.py'


use rule cuomo_eds_cor as cuomo_eds_cor_he with:
    input:
        y = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        eds = 'analysis/cuomo/eds.txt',
    output: 
        matrix = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/matrix.cor.he.png',
    params:
        method = 'he',


rule cuomo_eds_all:
    input:
        png = expand('results/cuomo/{params}/matrix.cor.png', params=cuomo_paramspace.instance_patterns),
        he = expand('results/cuomo/{params}/matrix.cor.he.png', params=cuomo_paramspace.instance_patterns),


use rule cuomo_eds_cor as yazar_eds_cor with:
    input:
        y = 'data/Yazar2022Science/ctp.gz',
        out = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
        eds = 'analysis/cuomo/eds.txt',
    output: 
        matrix = f'results/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/matrix.cor.png',
    params:
        method = 'he',


rule yazar_eds_all:
    input:
        png = expand('results/yazar/nomissing/{params}/matrix.cor.png', params=yazar_paramspace.instance_patterns),


###########################################################################################
# simulate Cuomo genes: a random gene's hom2, ct main variance, nu
###########################################################################################
V1 = ['0_0_0_0', '0.05_0_0_0','0.1_0_0_0', '0.2_0_0_0', '0.5_0_0_0']
V2 = ['0.05_0.05_0.05_0.05', '0.1_0.1_0.1_0.1', '0.2_0.2_0.2_0.2', '0.5_0.5_0.5_0.5']

rule cuomo_simulateGene_hom_ctp_test_powerplot:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.remlJK.npy'
                for nu_noise in nu_noises],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.hom.power.png',
    params:
        nu_noises = nu_noises,
    script: 'scripts/cuomo/simulateGene_hom_ctp_test_powerplot.py'

#rule cuomo_simulateGene_hom_ctp_test_estimates:
#    input:
#        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.npy'
#                for nu_noise in nu_noises],
#        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{nu_noise}/ctp.remlJK.npy'
#                for nu_noise in nu_noises],
#        real_out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
#        genes = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{i}.txt'
#                for i in range(cuomo_simulateGene_batch_no)],
#    output:
#        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.hom.estimates.png',
#    params:
#        nu_noises = nu_noises,
#    script: 'scripts/cuomo/simulateGene_hom_ctp_test_estimates.py'

rule cuomo_simulateGene_hom_ctp_test_estimates_paper:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/pseudo/{nu_noise}/ctp.npy'
                for nu_noise in nu_noises],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.hom.estimates.supp.png',
    params:
        nu_noises = nu_noises,
    script: 'scripts/cuomo/simulateGene_hom_ctp_estimates.paper.py'

rule cuomo_simulateGene_Free_ctp_test_powerplot:
    input:
        outs1 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.npy'
                for V in V1],
        outs2 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.npy'
                for V in V2],
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.npy'
                for V in V3],
        remlJKs1 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.npy'
                for V in V1],
        remlJKs2 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.npy'
                for V in V2],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.npy'
                for V in V3],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.free.1_2_5.power.png',
    params:
        V1 = V1,
        V2 = V2,
        V3 = V3,
    script: 'scripts/cuomo/simulateGene_Free_ctp_test_powerplot.py'

rule cuomo_simulateGene_free_ctp_test_estimates:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{V}/1_2_5/ctp.npy'
                for V in V3],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/pseudo/V_{V}/1_2_5/ctp.remlJK.npy'
                for V in V3],
        #real_out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        #genes = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/genes.batch{i}.txt'
        #        for i in range(cuomo_simulateGene_batch_no)],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.free.estimates.png',
    params:
        V3 = V3,
    script: 'scripts/cuomo/simulateGene_free_ctp_test_estimates.py'


use rule cuomo_simulateGene_free_ctp_test_estimates as cuomo_simulateGene_free_ctp_test_estimates_tmp with:
    input:
        outs = [f'analysis_20230510/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.npy'
                for V in V3],
        remlJK_outs = [f'analysis_20230510/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/V_{V}/1_2_5/ctp.remlJK.npy'
                for V in V3],
    output:
        png = f'results_20230510/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/ctp.free.estimates.png',






###########################################################################################
# simulate Cuomo genes: single cell simulation
###########################################################################################
#################### 10 cells per ind-ct
use rule cuomo_simulateGene_sc_bootstrap_hom as cuomo_simulateGene_sc_bootstrap_hom_10 with:
    output:
        genes = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/genes.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        P = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/P.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        ctnu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        cty = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/cty.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    params:
        batch = cuomo_sc_batches,
        cty = lambda wildcards, output: os.path.dirname(output.cty[0]),
        seed = 145342,
        resample_inds = False,
        cell_count = True,


use rule cuomo_simulateGene_hom_addUncertainty as cuomo_simulateGene_sc_bootstrap_hom_addUncertainty_10 with:
    input:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    output:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctnu.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    params:
        seed = 237632,


use rule ctp_test as cuomo_simulateGene_sc_bootstrap_hom_ctp_test_10 with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = False,
        REML = False,
        HE = True,
        HE_free_only = True,
        optim_by_R = True,


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_hom_ctp_aggReplications_10 with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.npy',


use rule ctp_test_remlJK as cuomo_simulateGene_sc_bootstrap_hom_ctp_test_remlJK_10 with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        optim_by_R = True,


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_hom_ctp_remlJK_aggReplications_10 with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.remlJK.npy',


use rule cuomo_simulateGene_sc_bootstrap_hom as cuomo_simulateGene_sc_bootstrap_free_10 with:
    output:
        genes = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/genes.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        P = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/P.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        ctnu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        cty = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/cty.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    params:
        batch = cuomo_sc_batches,
        cty = lambda wildcards, output: os.path.dirname(output.cty[0]), 
        seed = 143352,
        resample_inds = False,
        cell_count = True,
    resources:
        mem_mb = '30gb',
        partition = 'tier2q',


use rule cuomo_simulateGene_hom_addUncertainty as cuomo_simulateGene_sc_bootstrap_free_addUncertainty_10 with:
    input:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    output:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctnu.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    params:
        seed = 211313,


use rule ctp_test as cuomo_simulateGene_sc_bootstrap_Free_ctp_test_10 with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = False,
        REML = False,
        HE = True,
        HE_free_only = True,
        optim_by_R = True,


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_Free_ctp_aggReplications_10 with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.npy',


use rule ctp_test_remlJK as cuomo_simulateGene_sc_bootstrap_Free_ctp_test_remlJK_10 with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        optim_by_R = True,
        method = 'BFGS-Nelder',
    resources:
        mem_mb = '12gb',
        time = '68:00:00',


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_Free_ctp_remlJK_aggReplications_10 with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.remlJK.npy',


use rule cuomo_simulateGene_sc_bootstrap_ctp_test_powerplot_merge_paper as cuomo_simulateGene_sc_bootstrap_ctp_test_powerplot_merge_paper_including_10cellsimulation with:
    input:
        cellno_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{cell_no}/1/{nu_noise}/{{option}}/ctp.npy'
                for cell_no in cell_nos for nu_noise in nu_noises],
        cellcount_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/10/1/{nu_noise}/{{option}}/ctp.npy'
                for nu_noise in nu_noises],
        depth_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/1/{depth}/{nu_noise}/{{option}}/ctp.npy'
                for depth in depths for nu_noise in nu_noises],
        remlJK_cellno_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{cell_no}/1/{nu_noise}/{{option}}/ctp.remlJK.npy'
                for cell_no in cell_nos for nu_noise in nu_noises],
        remlJK_cellcount_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/cell_count/10/1/{nu_noise}/{{option}}/ctp.remlJK.npy'
                for nu_noise in nu_noises],
        remlJK_depth_outs = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/1/{depth}/{nu_noise}/{{option}}/ctp.remlJK.npy'
                for depth in depths for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        cellno_outs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{cell_no}/1/V_{V}/1_2_5/{{option}}/ctp.npy'
                for cell_no in cell_nos for V in V4],
        cellcount_outs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/10/1/V_{V}/1_2_5/{{option}}/ctp.npy'
                for V in V4],
        depth_outs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/1/{depth}/V_{V}/1_2_5/{{option}}/ctp.npy'
                for depth in depths for V in V4],
        cellno_remlJKs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/{cell_no}/1/V_{V}/1_2_5/{{option}}/ctp.remlJK.npy'
                for cell_no in cell_nos for V in V4],
        cellcount_remlJKs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/cell_count/10/1/V_{V}/1_2_5/{{option}}/ctp.remlJK.npy'
                for V in V4],
        depth_remlJKs3 = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/free/1/{depth}/V_{V}/1_2_5/{{option}}/ctp.remlJK.npy'
                for depth in depths for V in V4],
        real = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/ctp.sc.power.{{option}}.png',
        data = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/ctp.sc.power.{{option}}.txt',



# quality check of sc simulation
rule cuomo_simulateGene_sc_bootstrap_countsimqc_transform:
    input:
        raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.gz',
        sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.gz',
    output:
        raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.dds.gz',
        sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.dds.gz',
        raw_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.meta.dds.gz',
        sim_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.meta.dds.gz',
    script: 'scripts/cuomo/countsimqc_transform.py'


rule cuomo_simulateGene_sc_bootstrap_countsimqc:
    input:
        raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.dds.gz',
        sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.dds.gz',
        raw_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.meta.dds.gz',
        sim_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.meta.dds.gz',
    output:
        html = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/hom/{{cell_no}}/{{depth}}/{{option}}/countsimqc.html',
    resources:
        time = '124:00:00'
    conda: "envs/countsimQC.yaml"
    script: 'scripts/cuomo/countsimqc.R' 


# rule cuomo_simulateGene_sc_bootstrap_countsimqc_test:
#     input:
#         raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.dds.gz',
#         sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.dds.gz',
#         raw_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.meta.dds.gz',
#         sim_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.meta.dds.gz',
#     output:
#         html = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/hom/{{cell_no}}/{{depth}}/{{option}}/test/countsimqc.html',
#     resources:
#         time = '124:00:00'
#     conda: "envs/countsimQC.yaml"
#     script: 'scripts/cuomo/countsimqc.test.R'


# test: including all 1,000 genes. conclusion: the package doesn't support missing values in count matrix
# rule cuomo_simulateGene_sc_bootstrap_countsimqc_transform2:
#     input:
#         raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.gz',
#         sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.gz',
#     output:
#         raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.dds.tmp.gz',
#         sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.dds.tmp.gz',
#         raw_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.meta.dds.tmp.gz',
#         sim_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.meta.dds.tmp.gz',
#     script: 'scripts/cuomo/countsimqc_transform2.py'


# use rule cuomo_simulateGene_sc_bootstrap_countsimqc as cuomo_simulateGene_sc_bootstrap_countsimqc2 with:
#     input:
#         raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.dds.tmp.gz',
#         sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.dds.tmp.gz',
#         raw_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.meta.dds.tmp.gz',
#         sim_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.meta.dds.tmp.gz',
#     output:
#         html = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/hom/{{cell_no}}/{{depth}}/{{option}}/tmp/countsimqc.html',


# # test: the inflated variance is because of round of counts in raw data?
# rule cuomo_simulateGene_sc_bootstrap_countsimqc_transform3:
#     input:
#         raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.gz',
#         sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.gz',
#     output:
#         raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.dds.tmp3.gz',
#         sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.dds.tmp3.gz',
#         raw_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.meta.dds.tmp3.gz',
#         sim_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.meta.dds.tmp3.gz',
#     script: 'scripts/cuomo/countsimqc_transform3.py'


# use rule cuomo_simulateGene_sc_bootstrap_countsimqc as cuomo_simulateGene_sc_bootstrap_countsimqc3 with:
#     input:
#         raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.count.dds.tmp3.gz',
#         sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.count.dds.tmp3.gz',
#         raw_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/raw.meta.dds.tmp3.gz',
#         sim_meta = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/sim.meta.dds.tmp3.gz',
#     output:
#         html = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/hom/{{cell_no}}/{{depth}}/{{option}}/tmp3/countsimqc.html',




#########################################################################
#  Cuomo sc simulation test for pseudocount in simulation of mean difference
#########################################################################
rule cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom:
    input:
        counts = 'data/cuomo2020natcommun/raw_counts.csv.gz',
        meta = 'analysis/cuomo/data/meta.txt',
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
    output:
        genes = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/genes.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        P = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/P.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        ctnu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        cty = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/cty.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/raw.count.gz',
        sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/sim.count.gz',
    params:
        batch = cuomo_sc_batches,
        cty = lambda wildcards, output: os.path.dirname(output.cty[0]),
        seed = 145342,
        resample_inds = False,
    resources:
        mem_mb = '10gb',
    priority: 100
    script: 'scripts/cuomo/simulateGene_sc_bootstrap.py'


use rule cuomo_simulateGene_hom_addUncertainty as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom_addUncertainty with:
    input:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    output:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/ctnu.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],


use rule ctp_test as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/ctp.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = False,
        REML = False,
        HE = True,
        HE_free_only = True,
        optim_by_R = True,


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/ctp.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/ctp.npy',
    priority: 100


use rule ctp_test_remlJK as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/ctp.remlJK.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        optim_by_R = True,
    resources:
        mem_mb = '12gb',
        time = '48:00:00',
    priority: 97


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/ctp.remlJK.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/ctp.remlJK.npy',
    priority: 100


# use rule cuomo_simulateGene_sc_bootstrap_meandifference_hom_plot as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom_plot with:
#     input:
#         he = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/sdc_{{pseudocount}}/1/1/1_2_5/2/{meandifference}/ctp.npy'
#                 for meandifference in mean_differences],
#         reml = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/hom/sdc_{{pseudocount}}/1/1/1_2_5/2/{meandifference}/ctp.remlJK.npy'
#                 for meandifference in mean_differences],
#     output:
#         png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/sdc_{{pseudocount}}/meandifference.2.supp.png',

pseudocounts = [1, 1e-4]
# pseudocounts = [1, 1e-2, 1e-4]
rule cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom_plot:
    input:
        raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_1/option_{{option}}/md_0/raw.count.gz',
        he = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{pseudocount}/option_{{option}}/md_{meandifference}/nu_{{nu_noise}}/ctp.npy'
                for pseudocount in pseudocounts for meandifference in mean_differences],
        reml = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{pseudocount}/option_{{option}}/md_{meandifference}/nu_{{nu_noise}}/ctp.remlJK.npy'
                for pseudocount in pseudocounts for meandifference in mean_differences],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/nu_{{nu_noise}}/meandifference.{{option}}.png',
    params:
        pseudocounts = pseudocounts,
        mean_differences = mean_differences,
    script: 'scripts/cuomo/meandifference_pseudocount_hom_plot.py'



rule cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_all:
    input:
        png = expand('results/cuomo/{params}/simulateGene/bootstrap/nu_{nu_noise}/meandifference.{option}.png',
                    params=cuomo_paramspace.instance_patterns, nu_noise=['1_0_0'], option=[2, 3]),
        # nu_noise=['1_0_0', '1_2_5']


rule cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_mergeplot:
    input:
        raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_1/option_2/md_0/raw.count.gz',
        null = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_1/option_3/md_{meandifference}/nu_{{nu_noise}}/ctp.remlJK.npy'
                for meandifference in mean_differences],
        hom = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_1/option_2/md_{meandifference}/nu_{{nu_noise}}/ctp.remlJK.npy'
                for meandifference in mean_differences],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/nu_{{nu_noise}}/meandifference.supp.png',
    params:
        mean_differences = mean_differences,
    script: 'scripts/cuomo/meandifference_pseudocount_merge_plot.py'


#################### test prior gene proportion
use rule cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_prior_hom with:
    output:
        genes = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.genes.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        P = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.P.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        ctnu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        cty = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.cty.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
        raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.raw.count.gz',
        sim = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.sim.count.gz',
    params:
        batch = cuomo_sc_batches,
        cty = lambda wildcards, output: os.path.dirname(output.cty[0]),
        seed = 145342,
        resample_inds = False,
        prior = True,


use rule cuomo_simulateGene_hom_addUncertainty as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_prior_hom_addUncertainty with:
    input:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    output:
        nu = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_{{nu_noise}}/prior.ctnu.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],


use rule ctp_test as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_prior_hom_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_1_2_5/prior.ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.ctp.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/rep/prior.ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = False,
        REML = False,
        HE = True,
        HE_free_only = True,
        optim_by_R = True,


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_prior_hom_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.ctp.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.ctp.npy',
    priority: 100


use rule ctp_test_remlJK as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_prior_hom_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/nu_1_2_5/prior.ctnu.batch{{i}}.npy',
    output:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.ctp.remlJK.batch{{i}}.npy',
    params:
        out = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/rep/prior.ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        optim_by_R = True,
    resources:
        mem_mb = '12gb',
        time = '48:00:00',
    priority: 97


use rule op_aggReplications as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_prior_hom_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.ctp.remlJK.batch{i}.npy'
                for i in range(cuomo_sc_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{{pseudocount}}/option_{{option}}/md_{{meandifference}}/prior.ctp.remlJK.npy',
    priority: 100


use rule cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_hom_plot as cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_prior_hom_plot with:
    input:
        raw = f'staging/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_1/option_{{option}}/md_0/prior.raw.count.gz',
        he = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{pseudocount}/option_{{option}}/md_{meandifference}/prior.ctp.npy'
                for pseudocount in pseudocounts for meandifference in mean_differences],
        reml = [f'analysis/cuomo/simulateGene/bootstrap/{cuomo_paramspace.wildcard_pattern}/sdc_{pseudocount}/option_{{option}}/md_{meandifference}/prior.ctp.remlJK.npy'
                for pseudocount in pseudocounts for meandifference in mean_differences],
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/bootstrap/option_{{option}}/meandifference.prior.2.png',



rule cuomo_simulateGene_sc_bootstrap_meandifference_pseudocount_prior_all:
    input:
        png = expand('results/cuomo/{params}/simulateGene/bootstrap/option_{option}/meandifference.prior.2.png',
                    params=cuomo_paramspace.instance_patterns, option=[2]),





#########################################################################
#     imputation accuracy
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

use rule cuomo_filterInds as cuomo_imputation_day_filterInds with:
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

use rule cuomo_filterCTs as cuomo_imputation_day_filterCTs with:
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
    script: 'scripts/cuomo/imputation_AddMissing.py'

use rule cuomo_split2batches as cuomo_imputation_split2batches with:
    input:
        #y = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/rep{{k}}/day.masked.pseudobulk.gz', # donor - day * gene
        y = 'staging/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/rep{k}/day.masked.pseudobulk.gz', # donor - day * gene
    output:
        y_batch = expand('staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{{k}}/y/batch{i}.txt',
                i=range(cuomo_imput_batch_no)),

use rule cuomo_imputeGenome as cuomo_imputation_day_imputeGenome with:
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
    script: 'scripts/cuomo/imputation_day_imputeNinputForop.py'

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
    script: 'scripts/cuomo/imputation_merge.py'

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
        imputed_nu = [f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/rep{k}/{cuomo_imput_paramspace.wildcard_pattern}/merged.ct.nu.gz'                                                  for k in cuomo_imputataion_reps], #donor-day * gene # negative ct_nu kept                                                                           output:
        #y_mse = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/y.mse', # series for gene-ct
        #nu_mse = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/nu.mse', # series for gene-ct
        #y_cor = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/y.cor', # series for gene-ct                                                              #nu_cor = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/nu.cor', # series for gene-ct
        #nu_png = f'analysis/imp/{cuomo_imput_paramspace.wildcard_pattern}/nu.png', # series for gene-ct
        #raw_nu_standradized = f'staging/imp/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.nu.std.gz', # donor - day * gene 
        y_mse = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.mse', # series for gene-ct
        nu_mse = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.mse', # series for gene-ct                                            y_cor = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.cor', # series for gene-ct
        nu_cor = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.cor', # series for gene-ct
        y_mse_within = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.withinrep.mse', # series for gene-ct                             y_mse_within_tmp = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.withinrep.tmp.mse', # series for gene-ct                     nu_mse_within = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.withinrep.mse', # series for gene-ct
        y_cor_within = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/y.withinrep.cor', # series for gene-ct
        nu_cor_within = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.withinrep.cor', # series for gene-ct
        #nu_png = f'analysis/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/nu.png', # series for gene-ct
        #raw_nu_standradized = f'staging/imp/ind_min_cellnum{{ind_min_cellnum}}/ct_min_cellnum{{ct_min_cellnum}}/missingness{{missingness}}/{cuomo_imput_paramspace.wildcard_pattern}/day.filterCTs.nu.std.gz', # donor - day * gene
    resources:
        time = '48:00:00',
        mem = '10gb',
    script: 'scripts/cuomo/imputation_accuracy.py'

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
    script: 'scripts/cuomo/imputation_accuracy_plot.py'

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
    script: 'scripts/cuomo/imputation_accuracy_plot.paper.py'

rule paper_cuomo_imputation_all:
    input:
        png = expand('results/imp/ind_min_cellnum{ind_min_cellnum}/ct_min_cellnum{ct_min_cellnum}/missingness{missingness}/imputation.accuracy.withinrep.paper.png',
                ind_min_cellnum=100, ct_min_cellnum=10, missingness=[0.1]),







###################################################################################################
# Cuomo compare top genes and bottom genes
###################################################################################################
rule cuomo_topVSbot_genes:
    input:
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
    output:
        top = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/top.genes',
        bot = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/bot.genes',
    params:
        n = 1000,
    script: 'scripts/cuomo/top_bot_genes.py'


# rule cuomo_topVSbot_overall_meanNvarNctspecificmean:
#     input:
#         top = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/top.genes',
#         bot = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/bot.genes',
#         cty = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterCTs.pseudobulk.gz', # donor - day * gene
#         P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
#         remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
#     output:
#         mean = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/overall_mean.png',
#         ct_mean = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/ct_mean.png',
#         var = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/overall_var.png',
#         beta = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/ct_beta.png',
#     script: 'scripts/cuomo/top_bot_overall_mean.py'


rule cuomo_topVSbot_plot:
    input:
        top = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/top.genes',
        bot = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot/bot.genes',
        cty = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.pseudobulk.gz', # donor - day * gene
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.filterInds.prop.gz', # donor * day
        remlJK = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.remlJK.npy',
    output:
        png = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/topvsbot.paper.png',
    script: 'scripts/cuomo/top_bot.paper.py'


################################################################
# simulate single cells using cell level lmm (inflated fpr)
################################################################


rule cuomo_simulateGene_sc_hom:
    input:
        raw_cty = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/day.Gimputed.pseudobulk.gz', # imputed cty before std
        std_cty = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ct.y.merge.txt',
        out = f'analysis/cuomo/{cuomo_paramspace.wildcard_pattern}/ctp.npy',
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/genes.txt',
        ctnu = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/ct.nu.ctp.merge.txt',
        P = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/P.merge.txt',
        n = f'staging/cuomo/{cuomo_paramspace.wildcard_pattern}/n.merge.txt', 
    output:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        n = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/n.batch{{i}}.txt',
        ctnu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/ctnu.batch{{i}}.txt',
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/cty.batch{{i}}.txt',
    params:
        batch = lambda wildcards: cuomo_simulateGene_batches[int(wildcards.i)],
        cty = lambda wildcards, output: os.path.dirname(output.cty),
        seed = 1452,
    resources:
        mem_mb = '10gb',
        burden = 100,
    script: 'scripts/cuomo/simulateGene_sc.py'


use rule cuomo_simulateGene_hom_addUncertainty as cuomo_simulateGene_sc_hom_addUncertainty with:
    input:
        nu = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    output:
        nu = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    params:
        seed = 2376311,


use rule ctp_test as cuomo_simulateGene_sc_hom_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready', 
        ML = False,
        REML = True,
        HE = True,
        optim_by_R = True,


use rule op_aggReplications as cuomo_simulateGene_sc_hom_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.npy',


use rule ctp_test_remlJK as cuomo_simulateGene_sc_hom_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        optim_by_R = True,
    resources:
        mem_mb = '12gb',
        time = '48:00:00',
        partition = 'tier3q',


use rule op_aggReplications as cuomo_simulateGene_sc_hom_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{nu_noise}}/{{option}}/ctp.remlJK.npy',


use rule cuomo_simulateGene_sc_hom as cuomo_simulateGene_sc_free with:
    output:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/genes.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/P.batch{{i}}.txt',
        n = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/n.batch{{i}}.txt',
        ctnu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/ctnu.batch{{i}}.txt',
        cty = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/cty.batch{{i}}.txt',
    params:
        batch = lambda wildcards: cuomo_simulateGene_batches[int(wildcards.i)],
        cty = lambda wildcards, output: os.path.dirname(output.cty), 
        seed = 143352,


use rule cuomo_simulateGene_hom_addUncertainty as cuomo_simulateGene_sc_free_addUncertainty with:
    input:
        nu = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    output:
        nu = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctnu.batch{i}.txt'
                for i in range(cuomo_sc_batch_no)],
    params:
        seed = 23762123,


use rule ctp_test as cuomo_simulateGene_sc_Free_ctp_test with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/rep/ctp.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        ML = False,
        REML = True,
        HE = True,
        optim_by_R = True,


use rule op_aggReplications as cuomo_simulateGene_sc_Free_ctp_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.npy',


use rule ctp_test_remlJK as cuomo_simulateGene_sc_Free_ctp_test_remlJK with:
    input:
        genes = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/genes.batch{{i}}.txt',
        y = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{option}}/cty.batch{{i}}.txt',
        P = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{{option}}/P.batch{{i}}.txt',
        nu = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctnu.batch{{i}}.txt',
    output:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{{i}}.out',
    params:
        out = f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/rep/ctp.remlJK.npy',
        batch = lambda wildcards, input: np.loadtxt(input.genes, dtype='str') if os.path.exists(input.genes) else 'not ready',
        optim_by_R = True,
    resources:
        mem_mb = '12gb',
        time = '48:00:00',


use rule op_aggReplications as cuomo_simulateGene_sc_Free_ctp_remlJK_aggReplications with:
    input:
        out = [f'staging/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.remlJK.batch{i}.out'
                for i in range(cuomo_simulateGene_batch_no)],
    output:
        out = f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{{V}}/{{nu_noise}}/{{option}}/ctp.remlJK.npy',


use rule cuomo_simulateGene_ctp_test_powerplot_paper as cuomo_simulateGene_sc_ctp_test_powerplot_paper with:
    input:
        outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{nu_noise}/{{option}}/ctp.npy'
                for nu_noise in nu_noises],
        remlJK_outs = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/hom/{{cell_no}}/{{depth}}/{nu_noise}/{{option}}/ctp.remlJK.npy'
                for nu_noise in nu_noises],
        nu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        var_nu = 'analysis/cuomo/data/log/bootstrapedNU/day.raw.var_nu.gz', # donor - day * gene
        outs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{V}/1_2_5/{{option}}/ctp.npy'
                for V in V3],
        remlJKs3 = [f'analysis/cuomo/simulateGene/{cuomo_paramspace.wildcard_pattern}/free/{{cell_no}}/{{depth}}/V_{V}/1_2_5/{{option}}/ctp.remlJK.npy'
                for V in V3],
    output:
        png1 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/{{cell_no}}/{{depth}}/{{option}}/ctp.sc.power.paper.supp.png',
        png2 = f'results/cuomo/{cuomo_paramspace.wildcard_pattern}/simulateGene/{{cell_no}}/{{depth}}/{{option}}/ctp.sc.power.paper.png',


rule cuomo_simulateGene_sc_all:
    input:
        png2 = expand('results/cuomo/{params}/simulateGene/{cell_no}/{depth}/{option}/ctp.sc.power.paper.png',
                params=cuomo_paramspace.instance_patterns,
                cell_no=[100], depth=[0.02, 0.2], option=[3]),
                # cell_no=[50, 100], depth=[0.02, 0.2, 1]),






###################################################################################################
# Yazar
###################################################################################################
rule yazar_varNU_dist:
    input:
        var_nu = 'analysis/yazar/var_ctnu.gz',
        nu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.gz',
    output:
        cv = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/cv.gz',
        png = f'results/yazar/{yazar_paramspace.wildcard_pattern}/var_nu.png',
    script: 'scripts/yazar/varNU_dist.py'

rule yazar_large_cv:
    input:
        cv = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/cv.gz',
        X = 'staging/data/yazar/X.npz',
        obs = 'staging/data/yazar/obs.gz',
        var = 'staging/data/yazar/var.gz',
    output:
        cv = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/var_nu.large_cv.gz',
    params:
        ind = yazar_ind_col,
        ct = yazar_ct_col,
    script: 'scripts/yazar/large_cv.py'


############################################################
# MVN imputation
############################################################
rule yazar_mvn_ctp:
    input:
        data = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctp.gz',
    output:
        data = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
    resources:
        mem_mb = '10G',
    run:
        from ctmm import preprocess
        data = pd.read_table(input.data, index_col=(0,1)).astype('float32')
        preprocess.mvn(data).to_csv(output.data, sep='\t')


use rule yazar_mvn_ctp as yazar_mvn_ctnu with:
    input:
        data = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.gz',
    output:
        data = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',


rule yazar_zero_ctnu:
    # >1 cts within a ind having ctnu=0, hom break
    input: 
        ctnu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',
    output:
        touch(f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.zeros'),
    run:
        ctnu = pd.read_table(input.ctnu, index_col=(0,1))
        print(ctnu.shape[1])
        grouped = (ctnu==0).groupby(level=0)
        counts_ind = grouped.sum()
        counts_gene = (counts_ind > 1).sum(axis=0)
        print((counts_gene > 0).sum())


use rule yazar_std_op as yazar_mvn_std_op with:
    input:
        ctp = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
        ctnu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',
        P = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/P.gz',
    output:
        op = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/op.mvn.gz',
        nu = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/nu.mvn.gz',
        ctp = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
        ctnu = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',
    

use rule yazar_op_pca as yazar_mvn_op_pca with:
    input:
        op = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/op.mvn.gz',
    output:
        evec = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/evec.txt',
        eval = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/eval.txt',
        pca = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/pca.txt',
        png = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/pca.png',


use rule cuomo_split2batches as yazar_mvn_split2batches with:
    input:
        y = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
    output:
        y_batch = [f'staging/yazar/{yazar_paramspace.wildcard_pattern}/y/batch{i}.txt' 
                for i in range(yazar_batch_no)],


use rule yazar_split_ctp as yazar_mvn_split_ctp with:
    input:
        ctp = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
        ctnu = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',
        y_batch = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt',
    output:
        ctp = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.mvn.gz', 
        ctnu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.mvn.gz', 


use rule yazar_ctp_HE_free as yazar_mvn_ctp_HE_free with:
    input:
        y_batch = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        ctp = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.mvn.gz', 
        ctnu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.mvn.gz', 
        P = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/P.gz',
        pca = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/pca.txt',
        meta = 'staging/data/yazar/obs.gz',
    output:
        out = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'he',
        model = 'free',
        he_free_ols = True,


use rule yazar_mvn_ctp_HE_free as yazar_mvn_ctp_HE_full with:
    output:
        out = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'he',
        model = 'full',
    resources:
        mem_mb = '60gb',
        partition = 'tier3q',
        burden = 20,


use rule yazar_ctp_HE as yazar_mvn_ctp_HE with:
    input:
        free = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
        full = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    output:
        out = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.npy',


use rule op_aggReplications as yazar_mvn_ctp_HE_aggReplications with:
    input:
        out = [f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.npy'
                for i in range(yazar_batch_no)],
    output:
        out = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',


use rule yazar_ctp_freeNfull_Variance_paper as yazar_mvn_ctp_freeNfull_Variance_paper with:
    input:
        P = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/P.gz',
        ctp = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        free = f'results/yazar/{yazar_paramspace.wildcard_pattern}/ctp.free.Variance.paper.png',
        full = f'results/yazar/{yazar_paramspace.wildcard_pattern}/ctp.full.Variance.paper.png',


use rule yazar_ctp_pvalue_paper as yazar_mvn_ctp_pvalue_paper with:
    input:
        out = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        png = f'results/yazar/{yazar_paramspace.wildcard_pattern}/ctp.he.pvalue.png',


rule yazar_mvn_all:
    input:
        ctp = expand('results/yazar/{params}/ctp.free.Variance.paper.png', 
                params=yazar_paramspace.instance_patterns),
        p = expand('results/yazar/{params}/ctp.he.pvalue.png',
                params=yazar_paramspace.instance_patterns),


###################################################################################################
# Yazar: 1.2 test L = 1e4 1e6 total counts
###################################################################################################

rule yazar_L_ctp_extractX:
    input:
        h5ad = 'data/Yazar2022Science/OneK1K_cohort_gene_expression_matrix_14_celltypes.h5ad.gz',
        var = 'data/Yazar2022Science/var.txt',
        obs = 'analysis/yazar/exclude_repeatedpool.obs.txt',
    output:
        X = 'staging/data/yazar/L{l}/X.npz',
        obs = 'staging/data/yazar/L{l}/obs.gz',
        var = 'staging/data/yazar/L{l}/var.gz',
    params:
        ind_col = yazar_ind_col,
        ct_col = yazar_ct_col,
    resources:
        mem_mb = '40G',
    run:
        import scanpy as sc
        from scipy import sparse
        from ctmm import preprocess

        genes = pd.read_table(input.var)
        if 'feature_is_filtered' in genes.columns:
            genes = genes.loc[~genes['feature_is_filtered'], 'feature'].to_numpy()
        else:
            genes = genes['feature'].to_numpy()

        if 'subset_gene' in params.keys():
            # random select genes
            rng = np.random.default_rng(seed=params.seed)
            genes = rng.choice(genes, params.subset_gene, replace=False)

        obs = pd.read_table(input.obs)
        ind_pool = np.unique(obs[params.ind_col].astype('str')+'+'+obs['pool'].astype('str'))

        ann = sc.read_h5ad(input.h5ad, backed='r')
        data = ann[(~ann.obs[params.ind_col].isna())
                & (~ann.obs[params.ct_col].isna())
                & (ann.obs[params.ind_col].astype('str')+'+'+ann.obs['pool'].astype('str')).isin(ind_pool), genes]
        # normalize and natural logarithm of one plus the input array
        X = preprocess.normalize(data.X, float(wildcards.l)).log1p()
        sparse.save_npz(output.X, X)

        data.obs.rename_axis('cell').to_csv(output.obs, sep='\t')
        data.var.rename_axis('feature').to_csv(output.var, sep='\t')


use rule yazar_ctp as yazar_L_ctp with:
    input:
        X = 'staging/data/yazar/L{l}/X.npz',
        obs = 'staging/data/yazar/L{l}/obs.gz',
        var = 'staging/data/yazar/L{l}/var.gz',
    output:
        ctp = 'staging/data/yazar/L{l}/ctp.gz',
        ctnu = 'staging/data/yazar/L{l}/ctnu.gz',
        P = 'staging/data/yazar/L{l}/P.gz',
        n = 'staging/data/yazar/L{l}/n.gz',


use rule yazar_rm_rareINDnCT_filterGenes as yazar_L_rm_rareINDnCT_filterGenes with:
    input:
        ctp = 'staging/data/yazar/L{l}/ctp.gz',
        ctnu = 'staging/data/yazar/L{l}/ctnu.gz',
        n = 'staging/data/yazar/L{l}/n.gz',
    output:
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.gz',
        ctnu = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctnu.gz',
        P = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/P.gz',
        n = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/n.gz',


use rule yazar_mvn_ctp as yazar_L_mvn_ctp with:
    input:
        data = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.gz',
    output:
        data = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',


use rule yazar_mvn_ctp as yazar_L_mvn_ctnu with:
    input:
        data = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctnu.gz',
    output:
        data = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',


use rule yazar_std_op as yazar_L_std_op with:
    input:
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
        ctnu = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',
        P = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/P.gz',
    output:
        op = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/op.std.gz',
        nu = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/nu.std.gz',
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
        ctnu = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctnu.std.gz',


use rule yazar_op_pca as yazar_L_op_pca with:
    input:
        op = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/op.std.gz',
    output:
        evec = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/evec.txt',
        eval = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/eval.txt',
        pca = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/pca.txt',
        png = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/pca.png',


use rule cuomo_split2batches as yazar_L_split2batches with:
    input:
        y = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
    output:
        y_batch = [f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/y/batch{i}.txt' 
                for i in range(yazar_batch_no)],


use rule yazar_split_ctp as yazar_L_split_ctp with:
    input:
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
        ctnu = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctnu.std.gz',
        y_batch = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt',
    output:
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.std.gz', 
        ctnu = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.std.gz', 


use rule yazar_ctp_HE_free as yazar_L_ctp_HE_free with:
    input:
        y_batch = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.std.gz', 
        ctnu = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.std.gz', 
        P = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/P.gz',
        pca = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/pca.txt',
        meta = 'staging/data/yazar/obs.gz',
    output:
        out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',


use rule yazar_L_ctp_HE_free as yazar_L_ctp_HE_full with:
    output:
        out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'he',
        model = 'full',
    resources:
        mem_mb = '90gb',
        partition = 'tier3q',
        burden = 50,


use rule yazar_ctp_HE as yazar_L_ctp_HE with:
    input:
        free = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
        full = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    output:
        out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.npy',


use rule op_aggReplications as yazar_L_ctp_HE_aggReplications with:
    input:
        out = [f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.npy'
                for i in range(yazar_batch_no)],
    output:
        out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',


use rule yazar_ctp_freeNfull_Variance_paper as yazar_L_ctp_freeNfull_Variance_paper with:
    input:
        P = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/P.gz',
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        free = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.free.Variance.paper.png',
        full = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.full.Variance.paper.png',


use rule yazar_ctp_pvalue_paper as yazar_L_ctp_pvalue_paper with:
    input:
        out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        png = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.he.pvalue.png',


rule yazar_L_all:
    input:
        ctp = expand('results/yazar/L{l}/{params}/ctp.free.Variance.paper.png', 
                l=['1e4'], params=yazar_paramspace.instance_patterns),
        p = expand('results/yazar/L{l}/{params}/ctp.he.pvalue.png',
                l=['1e4'], params=yazar_paramspace.instance_patterns),


rule yazar_L_ls_betacompare_v2:
    input:
        out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        png = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.beta.v2.png',
    script: 'scripts/yazar/betacompare.v2.py'


rule yazar_L_ctp_pvalue_olsbeta_paper:
    output:
        png = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.he.pvalue.olsbeta.png',
    params:
        out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    script: 'scripts/yazar/ctp.p.olsbeta.paper.py'


rule yazar_L_ctp_freeNfull_Variance_paper_tmp:
    output:
        free = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.free.Variance.paper.tmp.png',
        full = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.full.Variance.paper.tmp.png',
    params:
        P = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/P.gz',
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    script: 'scripts/yazar/ctp_freeNfull_Variance.paper.tmp.py'


###################################################################################################
# Yazar: 1.2.3.4 plot sc expression
###################################################################################################

rule yazar_L_sc_expressionpattern:
# single cell expression pattern plot
    input:
        out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
        P = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/P.gz',
        var = 'staging/data/yazar/L{l}/var.gz',
    output:
        png = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/genes/ctp.{{gene}}.png',
    params:
        mycolors = mycolors,
    script: 'scripts/yazar/sc_expressionpattern.py'


rule yazar_L_sc_expressionpattern_collect:
    input:
        png = [f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/genes/ctp.{gene}.png'
                for gene in ['ENSG00000197728']],
    output:
        touch(f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.sc_expressionpattern.flag'),





###################################################################################################
# Yazar: 1.2.3 rm missing inds (i.e. main analysis)
###################################################################################################

rule yazar_nomissing_compare:
    input:
        out9 = f'analysis/yazar/nomissing/ind_min_cellnum~10_ct_min_cellnum~10_prop~0.9/ctp.HE.npy',
        out5 = f'analysis/yazar/nomissing/ind_min_cellnum~10_ct_min_cellnum~10_prop~0.5/ctp.HE.npy',
    output:
        png = f'results/yazar/nomissing/ctp.he.pvalue.qq.png',
    script: 'scripts/yazar/pvalue.qq.py'


use rule yazar_L_sc_expressionpattern as yazar_nomissing_sc_expressionpattern with:
# single cell expression pattern plot
    input:
        out = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
        ctp = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
        P = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/P.gz',
        var = 'staging/data/yazar/var.gz',
    output:
        png = f'results/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/genes/ctp.{{gene}}.png',


rule yazar_nomissing_sc_expressionpattern_weakestvariance:
    input:
        out = f'analysis/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        out = touch(f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/weakestvariance.txt'),
    run:
        out = np.load(input.out, allow_pickle=True).item()
        p_beta = out['he']['wald']['free']['ct_beta']
        genes = out['gene'][p_beta == 0]
        p_V = out['he']['wald']['free']['V'][p_beta == 0]
        print(genes[np.argmax(p_V)])


rule yazar_nomissing_sc_expressionpattern_collect:
    input:
        png = [f'results/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/genes/ctp.{gene}.png'
                for gene in ['ENSG00000197728', 'ENSG00000096093']],
    output:
        touch(f'staging/yazar/nomissing/{yazar_paramspace.wildcard_pattern}/ctp.sc_expressionpattern.flag'),


rule yazar_nomissing_sc_expressionpattern_collect_prop5:
    input:
        png = [f'results/yazar/nomissing/ind_min_cellnum~10_ct_min_cellnum~10_prop~0.5/genes/ctp.{gene}.png'
                for gene in ['ENSG00000106565'] + # + ['ENSG00000100079', 'ENSG00000257764', 'ENSG00000121552', 'ENSG00000233864', 'ENSG00000105835', 'ENSG00000085265', 'ENSG00000197766', 'ENSG00000120738'] + 
                ['ENSG00000197728', 'ENSG00000196154', 'ENSG00000198502', 'ENSG00000237541']], # significant in prop 0.5 but not prop 0.9
    output:
        touch(f'staging/yazar/nomissing/ind_min_cellnum~10_ct_min_cellnum~10_prop~0.5/ctp.sc_expressionpattern.flag'),


rule yazar_nomissing_sc_expressionpattern_paper:
    input:
        out = f'analysis/yazar/nomissing/ind_min_cellnum~10_ct_min_cellnum~10_prop~0.9/ctp.HE.npy',
        ctp = f'analysis/yazar/nomissing/ind_min_cellnum~10_ct_min_cellnum~10_prop~0.9/ctp.std.gz',
        P = f'analysis/yazar/nomissing/ind_min_cellnum~10_ct_min_cellnum~10_prop~0.9/P.gz',
        var = 'staging/data/yazar/var.gz',
    output:
        png = f'results/yazar/nomissing/ind_min_cellnum~10_ct_min_cellnum~10_prop~0.9/genes/ctp.paper.png',
        data = f'results/yazar/nomissing/ind_min_cellnum~10_ct_min_cellnum~10_prop~0.9/genes/ctp.paper.sourcedata.txt',
    params:
        genes = ['ENSG00000197728', 'ENSG00000096093'],
        mycolors = mycolors,
    script: 'scripts/yazar/sc_expressionpattern.paper.py'



###################################################################################################
# Yazar: 1.2.3 test HE free gls / ols test for mean differentiation
###################################################################################################
# use rule yazar_L_ctp_HE_free as yazar_L_ctp_HE_free_ols with:
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols.npy',
#     params:
#         genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
#         test = 'he',
#         model = 'free',
#         he_jk = False,
#         he_free_ols = True,


# rule yazar_L_ctp_HE_free_ols_merge:
#     input:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
#         ls = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols.npy',
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols.merged.npy',
#     run:
#         out = np.load(input.out, allow_pickle=True)
#         ls = np.load(input.ls, allow_pickle=True)

#         for i in range(len(out)):
#             out[i]['he']['wald']['free']['ct_beta'] = ls[i]['he']['wald']['free']['ct_beta']
        
#         np.save(output.out, out)


# use rule yazar_ctp_HE as yazar_L_ctp_HE_ols with:
#     input:
#         free = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols.merged.npy',
#         full = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.ols.npy',


# use rule op_aggReplications as yazar_L_ctp_HE_ols_aggReplications with:
#     input:
#         out = [f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.ols.npy'
#                 for i in range(yazar_batch_no)],
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols.npy',


# use rule yazar_ctp_pvalue_paper as yazar_L_ctp_pvalue_ols_paper with:
#     input:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols.npy',
#     output:
#         png = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.he.ols.pvalue.png',


# use rule yazar_L_ctp_HE_free as yazar_L_ctp_HE_free_olsjk with:
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols_jk.npy',
#     params:
#         genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
#         test = 'he',
#         model = 'free',
#         he_free_ols_jk = True,


# use rule yazar_ctp_HE as yazar_L_ctp_HE_olsjk with:
#     input:
#         free = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols_jk.npy',
#         full = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.ols_jk.npy',


# use rule op_aggReplications as yazar_L_ctp_HE_olsjk_aggReplications with:
#     input:
#         out = [f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.ols_jk.npy'
#                 for i in range(yazar_batch_no)],
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols_jk.npy',


# use rule yazar_ctp_pvalue_paper as yazar_L_ctp_pvalue_olsjk_paper with:
#     input:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols_jk.npy',
#     output:
#         png = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.he.ols_jk.pvalue.png',



# use rule yazar_L_ctp_HE_free as yazar_L_ctp_HE_free_ols_ew with:
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols.ew.npy',
#     params:
#         genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
#         test = 'he',
#         model = 'free',
#         he_jk = False,
#         he_free_ols_ew = True,


# use rule yazar_L_ctp_HE_free_ols_merge as yazar_L_ctp_HE_free_ols_ew_merge with:
#     input:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
#         ls = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols.ew.npy',
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols.ew.merged.npy',


# use rule yazar_ctp_HE as yazar_L_ctp_HE_ols_ew with:
#     input:
#         free = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols.ew.merged.npy',
#         full = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.ols.ew.npy',


# use rule op_aggReplications as yazar_L_ctp_HE_ols_ew_aggReplications with:
#     input:
#         out = [f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.ols.ew.npy'
#                 for i in range(yazar_batch_no)],
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols.ew.npy',


# use rule yazar_ctp_pvalue_paper as yazar_L_ctp_pvalue_ols_ew_paper with:
#     input:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols.ew.npy',
#     output:
#         png = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.he.ols.ew.pvalue.png',


# use rule yazar_L_ctp_HE_free as yazar_L_ctp_HE_free_gls with:
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.gls.npy',
#     params:
#         genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
#         test = 'he',
#         model = 'free',
#         he_jk = False,
#         he_free_gls = True,


# use rule yazar_L_ctp_HE_free_ols_merge as yazar_L_ctp_HE_free_gls_merge with:
#     input:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
#         ls = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.gls.npy',
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.gls.merged.npy',


# use rule yazar_ctp_HE as yazar_L_ctp_HE_gls with:
#     input:
#         free = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.gls.merged.npy',
#         full = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.gls.npy',


# use rule op_aggReplications as yazar_L_ctp_HE_gls_aggReplications with:
#     input:
#         out = [f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.gls.npy'
#                 for i in range(yazar_batch_no)],
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.gls.npy',


# use rule yazar_ctp_pvalue_paper as yazar_L_ctp_pvalue_gls_paper with:
#     input:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.gls.npy',
#     output:
#         png = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.he.gls.pvalue.png',



# rule yazar_L_ls_all:
#     input:
#         ols = expand('results/yazar/L{l}/{params}/ctp.he.ols.pvalue.png',
#                 l=['1e4'], params=yazar_paramspace.instance_patterns),
#         ols_jk = expand('results/yazar/L{l}/{params}/ctp.he.ols_jk.pvalue.png',
#                 l=['1e4'], params=yazar_paramspace.instance_patterns),
#         # ols_ew = expand('results/yazar/L{l}/{params}/ctp.he.ols.ew.pvalue.png',
#                 # l=['1e4'], params=yazar_paramspace.instance_patterns),
#         gls = expand('results/yazar/L{l}/{params}/ctp.he.gls.pvalue.png',
#                 l=['1e4'], params=yazar_paramspace.instance_patterns),


# rule yazar_L_ls_betacompare:
#     input:
#         ols = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols.npy',
#         ols_jk = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols_jk.npy',
#         gls = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.gls.npy',
#         gls_jk = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
#     output:
#         png = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.beta.png',
#     script: 'scripts/yazar/betacompare.py'







###################################################################################################
# Yazar: 1.2.3 subset cell types from imputed data
###################################################################################################

subset = {  'A': ['CD4 NC', 'CD8 ET', 'NK', 'CD8 NC'],
            'B': ['CD4 NC', 'CD8 ET']}

rule yazar_L_subset:
    input:
        ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
        ctnu = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',
        n = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/n.gz',
    output:
        ctp = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
        ctnu = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',
        n = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/n.gz',
        P = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/P.gz',
    params:
        cts = lambda wildcards: subset[wildcards.group],
    run:
        ctp = pd.read_table(input.ctp)
        ctp.loc[ctp['ct'].isin(params.cts)].to_csv(output.ctp, sep='\t', index=False)

        ctnu = pd.read_table(input.ctnu)
        ctnu.loc[ctnu['ct'].isin(params.cts)].to_csv(output.ctnu, sep='\t', index=False)

        n = pd.read_table(input.n, index_col=0)
        cts = n.columns.to_numpy()
        cts = cts[np.isin(cts, params.cts)]
        n = n[cts]
        n.to_csv(output.n, sep='\t')
        
        P = n.divide(n.sum(axis=1), axis=0)
        P.to_csv(output.P, sep='\t')


use rule yazar_std_op as yazar_L_subset_std_op with:
    input:
        ctp = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
        ctnu = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctnu.mvn.gz',
        P = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/P.gz',
    output:
        op = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/op.std.gz',
        nu = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/nu.std.gz',
        ctp = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
        ctnu = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctnu.std.gz',


use rule yazar_op_pca as yazar_L_subset_op_pca with:
    input:
        op = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/op.std.gz',
    output:
        evec = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/evec.txt',
        eval = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/eval.txt',
        pca = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/pca.txt',
        png = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/pca.png',


use rule cuomo_split2batches as yazar_L_subset_split2batches with:
    input:
        y = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
    output:
        y_batch = [f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/y/batch{i}.txt'
                for i in range(yazar_batch_no)],


use rule yazar_split_ctp as yazar_L_subset_split_ctp with:
    input:
        ctp = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.std.gz',
        ctnu = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctnu.std.gz',
        y_batch = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt',
    output:
        ctp = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.std.gz',
        ctnu = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.std.gz',


use rule yazar_ctp_HE_free as yazar_L_subset_ctp_HE_free with:
    input:
        y_batch = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt',
        ctp = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.std.gz',
        ctnu = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.std.gz',
        P = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/P.gz',
        pca = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/pca.txt',
        meta = 'staging/data/yazar/obs.gz',
    output:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
    resources:
        mem_mb = '15gb',
        time = '120:00:00',


use rule yazar_L_subset_ctp_HE_free as yazar_L_subset_ctp_HE_full with:
    output:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'he',
        model = 'full',
    resources:
        mem_mb = '90gb',
        partition = 'tier3q',
        burden = 50,


use rule yazar_ctp_HE as yazar_L_subset_ctp_HE with:
    input:
        free = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.npy',
        full = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    output:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.npy',


use rule op_aggReplications as yazar_L_subset_ctp_HE_aggReplications with:
    input:
        out = [f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.npy'
                for i in range(yazar_batch_no)],
    output:
        out = f'analysis/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',


use rule yazar_ctp_freeNfull_Variance_paper as yazar_L_subset_ctp_freeNfull_Variance_paper with:
    input:
        P = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/P.gz',
        ctp = f'analysis/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        free = f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.free.Variance.paper.png',
        full = f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.full.Variance.paper.png',


use rule yazar_ctp_pvalue_paper as yazar_L_subset_ctp_pvalue_paper with:
    input:
        out = f'analysis/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
    output:
        png = f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.he.pvalue.png',


rule yazar_L_subset_all:
    input:
        ctp = expand('results/yazar/L{l}/subset{group}/{params}/ctp.free.Variance.paper.png',
                l=['1e4'], group=['B'], params=yazar_paramspace.instance_patterns),
        p = expand('results/yazar/L{l}/subset{group}/{params}/ctp.he.pvalue.png',
                l=['1e4'], group=['B'], params=yazar_paramspace.instance_patterns),


###################################################################################################
# Yazar: 1.2.3 subset cell types from imputed data  4. ols jk
###################################################################################################

use rule yazar_L_subset_ctp_HE_free as yazar_L_subset_ctp_HE_free_olsjk with:
    output:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols_jk.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'he',
        model = 'free',
        he_free_ols_jk = True,


use rule yazar_ctp_HE as yazar_L_subset_ctp_HE_olsjk with:
    input:
        free = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.free.ols_jk.npy',
        full = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    output:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.ols_jk.npy',


use rule op_aggReplications as yazar_L_subset_ctp_HE_olsjk_aggReplications with:
    input:
        out = [f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.ols_jk.npy'
                for i in range(yazar_batch_no)],
    output:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols_jk.npy',


use rule yazar_ctp_pvalue_paper as yazar_L_subset_ctp_pvalue_olsjk_paper with:
    input:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.ols_jk.npy',
    output:
        png = f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.he.ols_jk.pvalue.png',


rule yazar_L_subset_olsjk_all:
    input:
        p = expand('results/yazar/L{l}/subset{group}/{params}/ctp.he.ols_jk.pvalue.png',
                l=['1e4'], group=['B'], params=yazar_paramspace.instance_patterns),


###################################################################################################
# Yazar: 1.2.3 subset cell types from imputed data  4. REML (too slow, 2h/gene for two cts)
###################################################################################################

use rule yazar_L_subset_ctp_HE_free as yazar_L_subset_ctp_REML_free with:
    output:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.reml.free.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'reml',
        model = 'free',
        jk = True,
    resources:
        mem_mb = '10gb',
        time = '200:00:00',


use rule op_aggReplications as yazar_L_subset_ctp_REML_free_aggReplications with:
    input:
        out = [f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.reml.free.npy'
                for i in range(yazar_batch_no)],
    output:
        out = f'analysis/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.reml.free.npy',


use rule yazar_ctp_pvalue_paper as yazar_L_subset_ctp_reml_pvalue_paper with:
    input:
        out = f'analysis/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.reml.free.npy',
    output:
        png = f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.reml.pvalue.png',
    params:
        method = 'reml',


rule yazar_L_subset_reml_all:
    input:
        p = expand('results/yazar/L{l}/subset{group}/{params}/ctp.reml.pvalue.png',
                l=['1e4'], group=['B'], params=yazar_paramspace.instance_patterns),


###################################################################################################
# Yazar: 1.2.3 subset cell types from imputed data  4. ct specific random (batch)
###################################################################################################

use rule yazar_L_subset_ctp_HE_free as yazar_L_subset_ctrandom_ctp_HE_free with:
    output:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.ctrandom.HE.free.npy',
    params:
        genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
        test = 'he',
        model = 'free',
        he_free_ct_specific_random = True,
    resources:
        mem_mb = '20gb',
        time = '120:00:00',


use rule yazar_ctp_HE as yazar_L_subset_ctrandom_ctp_HE with:
    input:
        free = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.ctrandom.HE.free.npy',
        full = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.full.npy',
    output:
        out = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.ctrandom.HE.npy',


use rule op_aggReplications as yazar_L_subset_ctrandom_ctp_HE_aggReplications with:
    input:
        out = [f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.ctrandom.HE.npy'
                for i in range(yazar_batch_no)],
    output:
        out = f'analysis/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.ctrandom.HE.npy',


use rule yazar_ctp_freeNfull_Variance_paper as yazar_L_subset_ctrandom_ctp_freeNfull_Variance_paper with:
    input:
        P = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/P.gz',
        ctp = f'analysis/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.ctrandom.HE.npy',
    output:
        free = f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.ctrandom.free.Variance.paper.png',
        full = f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.ctrandom.full.Variance.paper.png',


use rule yazar_ctp_pvalue_paper as yazar_L_subset_ctrandom_ctp_pvalue_paper with:
    input:
        out = f'analysis/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.ctrandom.HE.npy',
    output:
        png = f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.ctrandom.he.pvalue.png',


rule yazar_L_subset_ctrandom_all:
    input:
        ctp = expand('results/yazar/L{l}/subset{group}/{params}/ctp.ctrandom.free.Variance.paper.png',
                l=['1e4'], group=['B'], params=yazar_paramspace.instance_patterns),
        p = expand('results/yazar/L{l}/subset{group}/{params}/ctp.ctrandom.he.pvalue.png',
                l=['1e4'], group=['B'], params=yazar_paramspace.instance_patterns),




###################################################################################################
# Yazar: 1.2.3.4 plot sc expression
###################################################################################################

# use rule yazar_L_sc_expressionpattern as yazar_sc_expressionpattern with:
# # single cell expression pattern plot
#     input:
#         out = f'analysis/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.npy',
#         ctp = f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.mvn.gz',
#     output:
#         png = f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/genes/ctp.{{gene}}.png',


# rule yazar_sc_expressionpattern_collect:
#     input:
#         png = [f'results/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/genes/ctp.{gene}.png'
#                 for gene in ['ENSG00000189159', 'ENSG00000185475', 'ENSG00000114353', 'ENSG00000141759', 'ENSG00000105254'] +
#                 ['ENSG00000100450', 'ENSG00000177954', 'ENSG00000168685', 'ENSG00000171223', 'ENSG00000113088'] + 
#                 ['ENSG00000181817', 'ENSG00000178980', 'ENSG00000254505', 'ENSG00000130706', 'ENSG00000100142']], # top mean genes, top var means but not sig mean, top mean genes but not sig var
#     output:
#         touch(f'staging/yazar/L{{l}}/subset{{group}}/{yazar_paramspace.wildcard_pattern}/ctp.sc_expressionpattern.flag'),



###################################################################################################
# Yazar: 1.2.3 test l = 10,000 total counts without correction of batch effect
###################################################################################################

# use rule yazar_ctp_HE as yazar_L_ctp_HE_nobatch with:
#     input:
#         y_batch = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
#         ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.std.gz', 
#         ctnu = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.std.gz', 
#         P = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/P.gz',
#         pca = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/pca.txt',
#         meta = 'staging/data/yazar/obs.gz',
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.HE.nobatch.txt',
#     params:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{{i}}/rep/ctp.HE.nobatch.npy',
#         genes = lambda wildcards, input: [line.strip() for line in open(input.y_batch)],
#         test = 'he',
#         batch = False,


# use rule op_aggReplications as yazar_L_ctp_HE_nobatch_aggReplications with:
#     input:
#         out = [f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/batch{i}/ctp.HE.nobatch.txt'
#                 for i in range(yazar_batch_no)],
#     output:
#         out = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.nobatch.npy',


# rule yazar_L_ctp_freeNfull_Variance_nobatch_paper:
#     input:
#         P = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/P.gz',
#         ctp = f'staging/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.HE.nobatch.npy',
#     output:
#         free = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.free.Variance.nobatch.paper.png',
#         # full = f'results/yazar/L{{l}}/{yazar_paramspace.wildcard_pattern}/ctp.full.Variance.adjustbatch.paper.png',
#     script: 'scripts/yazar/ctp_freeNfull_Variance.paper.py'


# rule yazar_L_nobatch_all:
#     input:
#         ctp = expand('results/yazar/L{l}/{params}/ctp.free.Variance.nobatch.paper.png', 
#                 l=['1e4', '1e6'], params=yazar_paramspace.instance_patterns),


###################################################################################################
# test AnnData ctp and ctnu codes using cuomo data
###################################################################################################
rule cuomo_pseudobulk_withanncode:
    input:
        meta = 'analysis/cuomo/data/meta.txt',
        counts = 'data/cuomo2020natcommun/log_normalised_counts.csv.gz',
    output:
        ctp = 'staging/cuomo/testanncode/ctp.gz',
        ctnu = 'staging/cuomo/testanncode/ctnu.gz',
    run:
        from ctmm import preprocess
        obs = pd.read_table(input.meta)
        obs.set_index('cell_name', inplace=True)
        counts = pd.read_table(input.counts, index_col=0).T
        print(counts.shape)
        counts = counts.loc[obs.index]
        print(counts.shape)
        var = counts.iloc[[0]].T
        print(var.head())
        ctp, ctnu, _, _ = preprocess.pseudobulk(X=counts.to_numpy(), obs=obs, var=var, ind_col='donor', ct_col='day')
        ctp.to_csv(output.ctp, sep='\t')
        ctnu.to_csv(output.ctnu, sep='\t')


rule cuomo_pseudobulk_withanncode_compare:
    input:
        old_ctp = 'analysis/cuomo/data/log/day.raw.pseudobulk.gz', # donor - day * gene
        new_ctp = 'staging/cuomo/testanncode/ctp.gz',
        old_ctnu = 'analysis/cuomo/data/log/day.raw.nu.gz', # donor - day * gene
        new_ctnu = 'staging/cuomo/testanncode/ctnu.gz',
    output:
        touch('staging/cuomo/testanncode/ctp.compare.txt'),
    run:
        old_ctp = pd.read_table(input.old_ctp, index_col=(0, 1))
        new_ctp = pd.read_table(input.new_ctp, index_col=(0, 1))
        print(old_ctp.shape, new_ctp.shape)
        new_ctp = new_ctp.loc[old_ctp.index, old_ctp.columns]
        old_ctnu = pd.read_table(input.old_ctnu, index_col=(0, 1))
        new_ctnu = pd.read_table(input.new_ctnu, index_col=(0, 1))
        print(old_ctnu.shape, new_ctnu.shape)
        new_ctnu = new_ctnu.loc[old_ctnu.index, old_ctnu.columns]

        print(np.allclose(old_ctp, new_ctp))
        print(np.allclose(old_ctnu, new_ctnu))






###################################################################################################
# Yazar: test if signals weakened when testing similar cell types: CD4 ET  CD4 NC
###################################################################################################
# wildcard_constraints: subset='[^/]+' 

# subset = {
#             'CD4': ['CD4 ET', 'CD4 NC'],
#             'CD8': ['CD8 ET', 'CD8 NC']}

# rule yazar_subset_cts_P:
#     input:
#         n = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/n.gz',
#     output:
#         n = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/n.gz',
#         P = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/P.gz',
#     params:
#         cts = lambda wildcards: subset[wildcards.subset],
#     run:
#         n = pd.read_table(input.n, index_col=0)
#         n = n[params.cts]
#         n.to_csv(output.n, sep='\t')

#         P = n.divide(n.sum(axis=1), axis=0)
#         P.to_csv(output.P, sep='\t')


# rule yazar_subset_cts:
#     input:
#         ctp = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctp.mvn.gz', 
#         ctnu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/batch{{i}}/ctnu.mvn.gz', 
#     output:
#         ctp = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/batch{{i}}/ctp.mvn.gz', 
#         ctnu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/batch{{i}}/ctnu.mvn.gz', 
#     params:
#         cts = lambda wildcards: subset[wildcards.subset],
#     run:
#         ctp = pd.read_table(input.ctp, index_col=(0, 1))
#         ctnu = pd.read_table(input.ctnu, index_col=(0, 1))
#         ctp = ctp.loc[ctp.index.get_level_values(1).isin(params.cts)]
#         ctnu = ctnu.loc[ctnu.index.get_level_values(1).isin(params.cts)]

#         ctp.to_csv(output.ctp, sep='\t')
#         ctnu.to_csv(output.ctnu, sep='\t')


# use rule yazar_ctp_HE as yazar_subset_ctp_HE with:
#     input:
#         y_batch = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/y/batch{{i}}.txt', # genes
#         ctp = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/batch{{i}}/ctp.mvn.gz', 
#         ctnu = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/batch{{i}}/ctnu.mvn.gz', 
#         P = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/P.gz',
#         pca = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/pca.txt',
#         meta = 'staging/data/yazar/obs.gz',
#     output:
#         out = f'staging/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/batch{{i}}/ctp.HE.txt',


# use rule op_aggReplications as yazar_subset_ctp_HE_aggReplications with:
#     input:
#         out = [f'staging/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/batch{i}/ctp.HE.txt'
#                 for i in range(yazar_batch_no)],
#     output:
#         out = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/ctp.HE.npy',


# use rule yazar_ctp_pvalue_paper as yazar_subset_ctp_pvalue_paper with:
#     input:
#         out = f'analysis/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/ctp.HE.npy',
#     output:
#         png = f'results/yazar/{yazar_paramspace.wildcard_pattern}/{{subset}}/ctp.he.pvalue.png',


# rule yazar_subset_all:
#     input:
#         p = expand('results/yazar/{params}/{subset}/ctp.he.pvalue.png',
#                 params=yazar_paramspace.instance_patterns,
#                 subset=['CD4', 'CD8']),
