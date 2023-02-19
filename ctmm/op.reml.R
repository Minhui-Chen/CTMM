library(numDeriv)
library(MASS)
source('util.R')

LL <- function(y, P, X, C, vs, hom2, V, random_variances=NULL, random_MMT=NULL){

    sig2s <- hom2 + diag( P %*% V %*% t(P) ) + vs
	if( any( sig2s < 0 ) ) return(1e12)

    if ( is.null( random_MMT ) ) {
        sig2s_inv <- 1/sig2s
        A <- sweep( t(X), 2, sig2s_inv, '*') # t(X) %*% diag(sig2s_inv)
        B <- A %*% X
        eval <- eigen(B,symmetric=TRUE)$values
        if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
        M <- diag(sig2s_inv) - t(A) %*% solve(B) %*% A
        L <- sum(log( sig2s ))
    } else {
        sig2s <- diag(sig2s)
        for (i in 1:length(random_MMT)) {
            sig2s <- sig2s + random_variances[i] * random_MMT[[i]]
        }
        eval <- eigen(sig2s,symmetric=TRUE)$values
        if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
        sig2s_inv <- solve( sig2s )
        A <- t(X) %*% sig2s_inv
        B <- A %*% X
        eval <- eigen(B,symmetric=TRUE)$values
        if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
        M <- sig2s_inv - t(A) %*% solve(B) %*% A
        L <- determinant(sig2s, logarithm=TRUE)$modulus
    }

    det_B <- determinant(B, logarithm=TRUE)$modulus
    L <- 0.5 * ( L + det_B + t(y) %*% M %*% y )
    return ( L )
}

gls <- function(y, P, X, vs, hom2, V, random, r2){

    sig2s <- hom2 + diag( P %*% V %*% t(P) ) + vs

    if ( !is.null( random ) ) {
        sig2s <- diag(sig2s)
        for (i in 1:length(random)) {
            sig2s <- sig2s + r2[i] * random[[i]] %*% t(random[[i]])
        }
        sig2s_inv <- solve( sig2s )
    } else {
        sig2s_inv <- diag( 1/sig2s )
    }

    A <- t(X) %*% sig2s_inv
    B <- A %*% X
    beta <- c(solve( B ) %*% A %*% y)

    return( beta )
}

screml_hom <- function(y, P, vs, fixed=NULL, random=NULL, method='BFGS', hessian=TRUE, nrep=10, 
                       overVariance_cut=5, par=NULL){

	C <- ncol(P) 

    X <- make_X(P, fixed)
    random_MMT <- make_MMT( random )

    if ( is.null( par ) ) {
        hom2 <- var(y) / (length(random)+1)
        par <- rep(hom2, length(random)+1)
    }
    
    args <- list(y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)
    out <- optim_wrap( par, screml_hom_loglike, args, method, hessian )

	hom2 <- out$par[1]
    r2 <- out$par[2:length(out$par)]
    beta <- gls(y, P, X, vs, hom2, matrix(rep(0,C*C),nrow=C), random, r2)
    l <- out$value * (-1)

    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, 0, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim( out, screml_hom_loglike, par, args, method, nrep, hessian )

        hom2 <- out$par[1]
        r2 <- out$par[2:length(out$par)]
        beta <- gls(y, P, X, vs, hom2, matrix(rep(0,C*C),nrow=C), random, r2)
        l <- out$value * (-1)

        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_hom_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, l=l, hess=hess, beta=beta, fixedeffect_vars=fixed_vars,
                  randomeffect_vars=random_vars, r2=r2, convergence=out$convergence) )
}

screml_hom_loglike <- function( par, args ){
    y <- args[['y']]
    P <- args[['P']] 
    X <- args[['X']]
    C <- args[['C']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]

	hom2 <- par[1]
    V <- matrix( rep(0,C*C), nrow=C )
    r2 <- par[2:length(par)]

    l <- LL(y, P, X, C, vs, hom2, V, r2, random_MMT)
    return( l )
}

screml_iid <- function(y, P, vs, fixed=NULL, random=NULL, method='BFGS', hessian=TRUE, nrep=10,
                       overVariance_cut=5, par=NULL){

	C      <- ncol(P) 
    pi  <- colMeans(P)
    pd  <- scale(P, scale=F)
    S   <- (t(pd) %*% pd ) / nrow(P)

    X <- make_X(P, fixed)
    random_MMT <- make_MMT( random )

    if ( is.null( par ) ) {
        hom2 <- var(y) / (length(random)+2)
        par <- rep(hom2, length(random)+2)
    }
    
    args <- list(y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)
    out <- optim_wrap( par, screml_iid_loglike, args, method, hessian )

	hom2 <- out$par[1]
	V <- diag(C) * out$par[2]
    r2 <- out$par[3:length(out$par)]
    beta <- gls(y, P, X, vs, hom2, V, random, r2)
    l <- out$value * (-1)
    
    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim( out, screml_iid_loglike, par, args, method, nrep, hessian )

        hom2 <- out$par[1]
        V <- diag(C) * out$par[2]
        r2 <- out$par[3:length(out$par)]
        beta <- gls(y, P, X, vs, hom2, V, random, r2)
        l <- out$value * (-1)
        
        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian
    hess = hessian(screml_iid_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixed_vars, 
                  randomeffect_vars=random_vars, r2=r2, convergence=out$convergence) )
}

screml_iid_loglike<- function(par, args){
    y <- args[['y']]
    P <- args[['P']] 
    X <- args[['X']]
    C <- args[['C']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]

	hom2 <- par[1]
	V <- diag(C) * par[2]
    r2 <- par[3:length(par)]

    return( LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )
}

screml_free <- function(y, P, vs, fixed=NULL, random=NULL, method='BFGS', hessian=TRUE, nrep=10,
                        overVariance_cut=5, par=NULL){

	C   <- ncol(P) 

    X <- make_X(P, fixed)
    random_MMT <- make_MMT( random )

    if ( is.null( par ) ) {
        hom2 <- var(y) / (length(random)+2)
        par <- rep(hom2, length(random)+1+C)
    }

    args <- list(y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)
    out <- optim_wrap( par, screml_free_loglike, args, method, hessian )

	hom2 <- out$par[1]
	V <- diag(out$par[1+1:C])
    r2 <- out$par[(C+2):length(out$par)]
    beta <- gls(y, P, X, vs, hom2, V, random, r2)
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim( out, screml_free_loglike, par, args, method, nrep, hessian )

        hom2 <- out$par[1]
        V <- diag(out$par[1+1:C])
        r2 <- out$par[(C+2):length(out$par)]
        beta <- gls(y, P, X, vs, hom2, V, random, r2)
        l <- out$value * (-1)

        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_free_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixed_vars,
                 randomeffect_vars=random_vars, r2=r2, convergence=out$convergence ))
}

screml_free_loglike <- function(par, args){
    y <- args[['y']]
    P <- args[['P']] 
    X <- args[['X']]
    C <- args[['C']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]

	hom2 <- par[1]
	V <- diag(par[1+1:C])
    r2 <- par[(C+2):length(par)]

    return ( LL(y, P, X, C, vs, hom2, V, r2, random_MMT) )
}

screml_full <- function(y, P, vs, fixed=NULL, random=NULL, method='BFGS', hessian=TRUE, nrep=10,
                        overVariance_cut=5, par=NULL){

	C      <- ncol(P)
	ngam   <- C*(C+1)/2

    X <- make_X(P, fixed)
    random_MMT <- make_MMT( random )

    if ( is.null( par ) ) {
        v1 <- var(y) / (length(random)+1)
        V <- diag(C)[ lower.tri(diag(C),diag=T) ] * c(v1)
        par <- c( V, rep(v1,length(random)) )
    }

    args <- list(y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)
    out <- optim_wrap( par, screml_full_loglike, args, method, hessian )

    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]  <- out$par[1:ngam]
    V <- V + t(V)
    r2 <- out$par[(ngam+1):length(out$par)]
    hom2 <- 0
    beta <- gls(y, P, X, vs, hom2, V, random, r2)
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim( out, screml_full_loglike, par, args, method, nrep, hessian )

        V <- matrix( 0, C, C )
        V[lower.tri(V,diag=T)]  <- out$par[1:ngam]
        V <- V + t(V)
        r2 <- out$par[(ngam+1):length(out$par)]
        hom2 <- 0
        beta <- gls(y, P, X, vs, hom2, V, random, r2)
        l <- out$value * (-1)

        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_full_loglike, x=out$par, args=args)

    return ( list( V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixed_vars,
                 randomeffect_vars=random_vars, r2=r2, convergence=out$convergence ) )
}

screml_full_loglike <- function(par, args){
    y <- args[['y']]
    P <- args[['P']] 
    X <- args[['X']]
    C <- args[['C']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]

	ngam   <- C*(C+1)/2
    hom2 <- 0
    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]<- par[1:ngam]
    V <- V + t(V)
    random_variances <- par[(ngam+1):length(par)]

    return ( LL(y, P, X, C, vs, hom2, V, random_variances, random_MMT) )
}


################################
# runs only when script is run by itself
if (sys.nframe() == 0){
    args <- commandArgs(trailingOnly=TRUE)
    y_f <- args[1]
    P_f <- args[2]
    vs_f <- args[3]
    model <- args[4]
    out_f <- args[5]

    # read data
    y <- scan(y_f)
    P <- as.matrix(read.table(P_f))
    #print(P)
    vs <- scan(vs_f)

    if (model == 'null') {
        out = screml_null(y, P, vs)
        save(out, file=out_f)
    } else if (model == 'hom') {
        out = screml_hom(y, P, vs)
        save(out, file=out_f)
    } else if (model == 'iid') {
        out = screml_iid(y, P, vs)
        save(out, file=out_f)
    } else if (model == 'free') {
        out = screml_free(y, P, vs)
        save(out, file=out_f)
    } else if (model == 'full') {
        out = screml_full(y, P, vs)
        save(out, file=out_f)
    }
}
