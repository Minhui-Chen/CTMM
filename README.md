# CTMM
Python and R packages to fit CTMM (Cell Type-specific linear Mixed Model). CTMM estimates the variance of gene expression specific to each cell type and shared across cell types. For a full description of CTMM, including strengths and limitations, see:
  
* M Chen, A Dahl. (2023) A robust model for cell type-specific interindividual variation in single-cell RNA sequencing data. bioRxiv.

## Analysis scripts
This repository contains scripts for simulations and real data analysis for our paper.

* [Snakefile](Snakefile) contains steps for perform simulations and iPSCs analyses

* [OP](bin/OP) in bin/OP contains R scripts to fit CTMM using ML and REML with Overall Pseudobulk (OP) data

* [CTP](bin/CTP) in bin/CTP contains R scripts to fit CTMM using ML and REML with Cell Type-specific Pseudobulk (CTP) data

* [op.py](bin/op.py) and [ctp.py](bin/ctp.py) in bin/ are python scripts to fit CTMM using ML, REML, and HE with OP and CTP data.


## Running GxEMM: Hom, IID, and Free models

First, I'll simulate some test data. The details are not important for understanding how to use the package:
```R
set.seed(1234)
N <- 1e3 # sample size
S <- 1e2 # number SNPs

Z1  <- rbinom( N, 1, .5 )
Z   <- cbind( Z1, 1-Z1 ) ### two discrete environments

snps  <- scale( matrix( rbinom(N*S,2,.1), N, S ) )
K     <- snps %*% t(snps) / S

X     <- Z[,-1] # fixed effect covariates. Must include Z! Column is dropped here so that cbind(1,X) is full rank

#genetic variances--assumed heterogeneous in this simulation
sig2hom <- 0
sig2het <- c( .1, .4 )

# noise--assumed homogeneous in this simulation
epsilon	<- sqrt(1-sig2hom-sum( colMeans( Z^2 ) * sig2het )) * rnorm(N)

# heterogeneous SNP effects--details of this expression are not so important
betas	<- sapply( sig2het, function(sig) rnorm( S, sd=sqrt( sig/S ) ) )
uhet	<- sapply( 1:nrow(Z), function(i) snps[i,] %*% ( betas %*% Z[i,] ) )

y   <- as.numeric( Z %*% c(.1,-.5) + uhet + epsilon )
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
