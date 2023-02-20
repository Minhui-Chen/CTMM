# CTMM
Python and R packages to fit CTMM (Cell Type-specific linear Mixed Model). CTMM estimates the variance of gene expression specific to each cell type and shared across cell types. For a full description of CTMM, including strengths and limitations, see:
  
* M Chen, A Dahl. (2023) A robust model for cell type-specific interindividual variation in single-cell RNA sequencing data. bioRxiv.

## Analysis scripts
This repository contains scripts for simulations and real data analysis for our paper.

* [Snakefile](Snakefile) contains steps for perform simulations and iPSCs analyses

* [sim](bin/sim) in bin/sim contains scripts to perform simulations

* [cuomo](bin/cuomo) in bin/cuomo contains scripts to perform analyses on iPSCs from Cuomo et al. 2020 Nature Communications

## Installation
Users can download the latest repository and then use ``pip``:

    git clone https://github.com/Minhui-Chen/CTMM.git
    cd CTMM
    pip install .

## Input format
CTMM can fit two types of pseudobulk gene expression data: Overall Pseudobulk (OP) and Cell Type-specific Pseudobulk (CTP).

* OP: Overall Pseudobulk gene expression for each individual. The file should have one column without header. (only needed when fitting OP)

* CTP: Cell Type-specific Pseudobulk gene expression for each individual. The file should have one column for each cell type and without header. (only needed when fitting CTP)

* P: cell type proportions for each individual. The file should have one column for each cell type and without header.

* nu: variance of measurement noise for each individual. The file should have one column without header. (only needed when fitting OP)

* ctnu: variance of measurement noise for each pair of individual and cell type. The file should have one column for each cell type and without header. (only needed when fitting CTP)

## Output
The output of CTMM have two dictionaries.

* The first one contains estimates of model parameters, including cell type-specific mean expression (beta), variance of cell type-shared random effect (hom2), variance of cell type-specific random effect (V), and others, e.g. loglikelihood (l).

* The second one contains p values from Wald test on e.g. differentiation of mean expression between cell types (ct_beta) and differentiation of expression variance between cell types (V). 

## Running CTMM

CTMM can be fit using OP and CTP data under hom, iid, free, and full models with ML (maximum likelihood), REML (restricted maximum likelihood), and HE (Haseman-Elston regression, a method-of-moments) methods.
To fit OP, import ``op`` module and call the function ``[model]_[method]``. For example, to fit Free model using HE, call ``op.free_HE()``. It's the same to fit CTP except to import ``ctp`` module. For example, to fit Full model using ML, call ``ctp.full_HE()``. Funcation arguments can be found using the ``help()``, e.g., ``help(ctp.full_HE())``.

To illustarte the usage of CTMM, here is an example of CTMM on OP and CTP from 50 individuals * 4 cell types:
```python
from ctmm import op, ctp, util

# Fit OP (Overall Pseudobulk)
OP_f = 'test/OP.gz' # overall pseudobulk
P_f = 'test/P.gz' # cell type proportions
nu_f = 'test/nu.gz' # overall variance of measurement noise for each individual

## fit with REML on Free model
reml_op, p_op = op.free_REML(y_f=OP_f, P_f=P_f, nu_f=nu_f, method='BFGS', optim_by_R=True) # use BFGS in R optim function for optimization
print( reml_op['hom2'] )  # variance of cell type-shared random effect (\sigma_\alpha^2)
print( reml_op['V'] )     # variance of cell type-specific random effect 
print( reml_op['beta']['ct_beat'] ) # cell type-specific fixed effect i.e. mean expression
print( p_op['V'] )        # Wald test on expression variance differentiation between cell types (V_1 =V_2 = 0)
print( p_op['ct_beta'] )  # Wald test on mean expression differentiation between cell types (beta_1 = beta_2)

# Fit CTP (Cell Type-specific Pseudobulk)
CTP_f = 'test/CTP.gz' # Cell Type-specific Pseudobulk
ctnu_f = 'test/ctnu.gz' # variance of measurement noise for each pair of individual and cell type

## fit with REML on Free model
free, p_wald = ctp.free_REML(y_f=CTP_f, P_f=P_f, ctnu_f=ctnu_f, method='BFGS', optim_by_R=True) # use BFGS in scipy package
### to conduct jackknife-based Wald test 
free_jk, p_jk = ctp.free_REML(y_f=CTP_f, P_f=P_f, ctnu_f=ctnu_f, method='BFGS', optim_by_R=True, jack_knife=True)

# Likelihood-ratio test (LRT)
## fit with REML on Hom model
hom, _ = ctp.hom_REML(y_f=CTP_f, P_f=P_f, ctnu_f=ctnu_f, method='BFGS', optim_by_R=True)
C = 4 # number of cell types
p_lrt = util.lrt(free['l'], hom['l'], C) # LRT on variance differentiation (V=0)
```

Now that the test data has been simulated, we need to run the three GxEMM models. Note you need to point GxEMM to the location of LDAK on your computer, and the location I've used here won't work for you:
```R
library(GxEMM)

ldak_loc  <- "~/GxEMM/code/ldak5.linux "
out_hom		<- GxEMM( y, X, K, Z, gtype='hom', ldak_loc=ldak_loc )
out_iid		<- GxEMM( y, X, K, Z, gtype='iid', ldak_loc=ldak_loc ) ### need to add etype='iid' for non-discrete environments
out_free	<- GxEMM( y, X, K, Z, gtype='free', etype='free', ldak_loc=ldak_loc )
```

Now that we've run the core three models, we can compare them:
```R
### test whether there is any heritability assuming the Hom model
Waldtest( out_hom$h2, out_hom$h2Covmat[1,1] )   

### test for genetic heterogeneity using IID model, which assumes that h2 is equal across all environments
Waldtest( out_iid$h2[2], out_iid$h2Covmat[2,2] )

### tests for genetic heterogeneity using Free model
MVWaldtest( out_free$sig2s[2:3], out_free$sig2Var[2:3,2:3] ) 
```
The details for these tests can be found in the AJHG paper. But the idea is to test whether key variance components are nonzero for each of these three models:

* In the Hom model, the focus is on the overall genetic variance. 
* In the IID model, the focus is on the single parameter that summarizes heterogeneous genetic variance that is shared, in magnitude, across all environments.
* In the Free model, the focus is on the vector of all environment-specific genetic variances, and the test is whether any is nonzero. 

### Additional tests

In general, many other tests can be performed that may be useful. For example:
```R
### tests for non-genetic heterogeneity in variance using Free model
### Because Z is discrete and there are 2 environments, sig2s[4]+sig2s[5] = sig2e[1], and sig2s[5]=sig2e[2]
### Contact Andy Dahl if studying a different Z and parameterization is too complicated
Waldtest( out_free$sig2s[4], out_free$sig2Var[4,4] )

### tests for any heterogeneity in variance using Free model
MVWaldtest( out_free$sig2s[2:4], out_free$sig2Var[2:4,2:4] )
```
