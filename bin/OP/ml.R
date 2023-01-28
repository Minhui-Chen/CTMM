# random effect V currently only support homogeneous variance i.e. \sigma * I
library(numDeriv)
library(mvtnorm)
source('bin/util.R')

LL <- function(y, P, X, C, vs, beta, hom2, V, random_variances, random_MMT){
    sig2s <- hom2 + diag( P %*% V %*% t(P) ) + vs
    if ( !is.null( random_MMT ) ) {
        sig2s <- diag(sig2s)
        for (i in 1:length(random_MMT)) {
            sig2s <- sig2s + random_variances[i] * random_MMT[[i]]
        }

        eval <- eigen(sig2s, symmetric=TRUE)$values
        if( max(eval) / (min(eval)+1e-99) > 1e6 | min(eval)<0 ) return(1e12)

        dmvnorm(y, mean=X %*% beta, sigma = sig2s, log=TRUE) * (-1)

    } else {
        if( max(sig2s) / (min(sig2s)+1e-99) > 1e6 | min(sig2s)<0 ) return(1e12)
        (sum(log( sig2s )) + sum( (y - X %*% beta)^2 / sig2s ))/2   
    }
}


screml_hom <- function(
y, P, vs, fixed=NULL, random=NULL, nrep=10, method='BFGS', hessian=TRUE, overVariance_cut=5, par=NULL
){
	C <- ncol(P)  # cell type number
    N <- nrow(P)

    X <- make_X(P, fixed)
    random_MMT <- make_MMT( random )

    if ( is.null( par ) ) {
        beta <- solve( t(X) %*% X ) %*% ( t(X) %*% y ) 
        hom2 <- var(y - X %*% beta) / (length(random)+1)
        par <- c( beta, rep(hom2,length(random)+1) )
    }

    args <- list(y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)
    out <- optim_wrap( par, screml_hom_loglike, args, method, hessian )

	beta <- out$par[1:ncol(X)]
	hom2 <- out$par[ncol(X)+1]
    r2 <- out$par[(ncol(X)+2):length(out$par)]
    l <- out$value * (-1)

    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, 0, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim(out, screml_hom_loglike, par, args, method, nrep, hessian)

        beta <- out$par[1:ncol(X)]
        hom2 <- out$par[ncol(X)+1]
        r2 <- out$par[(ncol(X)+2):length(out$par)]
        l <- out$value * (-1)

        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_hom_loglike, x=out$par, args=args)

    return( list(hom2=hom2, beta=beta, l=l, fixedeffect_vars=fixed_vars, 
                 randomeffect_vars=random_vars, r2=r2, convergence=out$convergence,
                 hess=hess) )
}

screml_hom_loglike<- function( par, args ){
    y <- args[['y']]
    P <- args[['P']]
    X <- args[['X']]
    C <- args[['C']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]

	beta <- par[1:ncol(X)] 
	hom2 <- par[ncol(X)+1]
    V <- matrix(rep(0,C*C), nrow=C)
    r2 <- par[(ncol(X)+2):length(par)]

    return( LL(y, P, X, C, vs, beta, hom2, V, r2, random_MMT) )
}

screml_iid <- function(
y, P, vs, fixed=NULL, random=NULL, nrep=10, method='BFGS', hessian=TRUE, overVariance_cut=5, par=NULL
) {

	C   <- ncol(P) 
	N   <- nrow(P)  

    X <- make_X(P, fixed)
    random_MMT <- make_MMT( random )

    if ( is.null( par ) ) {
        beta   <- solve( t(X) %*% X ) %*% ( t(X) %*% y ) 
        hom2 <- var(y - X %*% beta) / (length(random)+2)
        par <- c( beta, rep(hom2,length(random)+2) )
    }

    args <- list(y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)
    out <- optim_wrap( par, screml_iid_loglike, args, method, hessian )

	beta  <- out$par[1:ncol(X)]
	hom2 <- out$par[ncol(X)+1]
	V <- diag(C) * out$par[ncol(X)+2]
    r2 <- out$par[(ncol(X)+3):length(out$par)]
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim(out, screml_iid_loglike, par, args, method, nrep, hessian)

        beta  <- out$par[1:ncol(X)]
        hom2 <- out$par[ncol(X)+1]
        V <- diag(C) * out$par[ncol(X)+2]
        r2 <- out$par[(ncol(X)+3):length(out$par)]
        l <- out$value * (-1)

        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    
    # estimate hessian
    hess = hessian(screml_iid_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, beta=beta, V=V, l=l, fixedeffect_vars=fixed_vars, 
                  randomeffect_vars=random_vars, r2=r2, convergence=out$convergence,
                  hess=hess) )
}

screml_iid_loglike<- function(par, args){
    y <- args[['y']]
    P <- args[['P']]
    X <- args[['X']]
    C <- args[['C']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]

	beta    <- par[1:ncol(X)] 
	hom2 <- par[ncol(X)+1]
	V <- diag(C) * par[ncol(X)+2]
    r2 <- par[(ncol(X)+3):length(par)]

    return( LL(y, P, X, C, vs, beta, hom2, V, r2, random_MMT) )

}

screml_free <- function(
y, P, vs, fixed=NULL, random=NULL, nrep=10, method='BFGS', hessian=TRUE, overVariance_cut=5, par=NULL
){
	C   <- ncol(P) 
	N   <- ncol(P)

    X <- make_X(P, fixed)
    random_MMT <- make_MMT( random )

    if ( is.null( par ) ) {
        beta <- solve( t(X) %*% X ) %*% ( t(X) %*% y ) # cell type effect
        hom2 <- var(y - X %*% beta) / (length(random)+2)
        par <- c(beta, rep(hom2,length(random)+1+C))
    }

    args <- list(y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)
    out <- optim_wrap( par, screml_free_loglike, args, method, hessian )

	beta  <- out$par[1:ncol(X)]
	hom2 <- out$par[ncol(X)+1]
	V <- diag(out$par[ncol(X)+1+1:C])
    r2 <- out$par[(C+2+ncol(X)):length(out$par)]
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim(out, screml_free_loglike, par, args, method, nrep, hessian)

        beta  <- out$par[1:ncol(X)]
        hom2 <- out$par[ncol(X)+1]
        V <- diag(out$par[ncol(X)+1+1:C])
        r2 <- out$par[(C+2+ncol(X)):length(out$par)]
        l <- out$value * (-1)

        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_free_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, beta=beta, V=V, l=l, fixedeffect_vars=fixed_vars, 
                  randomeffect_vars=random_vars, r2=r2, convergence=out$convergence,
                  hess=hess) )
}

screml_free_loglike<- function( par, args ){
    y <- args[['y']]
    P <- args[['P']]
    X <- args[['X']]
    C <- args[['C']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]

	beta <- par[1:ncol(X)] 
	hom2 <- par[ncol(X)+1]
	V <- diag(par[ncol(X)+1+1:C])
    r2 <- par[(ncol(X)+C+2):length(par)]
    l <- LL(y, P, X, C, vs, beta, hom2, V, r2, random_MMT)
    return( l )

}

screml_full <- function(
y, P, vs, fixed=NULL, random=NULL, nrep=10, method='BFGS', hessian=TRUE, overVariance_cut=5, par=NULL
){

	C      <- ncol(P) 
	N      <- nrow(P)  
	ngam   <- C*(C+1)/2

    X <- make_X(P, fixed)
    random_MMT <- make_MMT( random )

    if ( is.null( par ) ) {
        beta   <- solve( t(X) %*% X ) %*% ( t(X) %*% y ) # cell type effect
        v1 <- var(y - X %*% beta) / (length(random)+1)
        V <- diag(C)[ lower.tri(diag(C),diag=T) ] * c(v1)
        par <- c( beta, V, rep(v1,length(random)) )
    }

    args <- list(y=y, P=P, X=X, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT)
    out <- optim_wrap( par, screml_full_loglike, args, method, hessian )
    
	beta <- out$par[1:ncol(X)]
    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]  <- out$par[ncol(X)+1:ngam]
    V <- V + t(V)
    r2 <- out$par[(ngam+ncol(X)+1):length(out$par)]
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, 0, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim(out, screml_full_loglike, par, args, method, nrep, hessian)

        beta <- out$par[1:ncol(X)]
        V <- matrix( 0, C, C )
        V[lower.tri(V,diag=T)] <- out$par[ncol(X)+1:ngam]
        V <- V + t(V)
        r2 <- out$par[(ngam+ncol(X)+1):length(out$par)]
        l <- out$value * (-1)

        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]

    }

    # estimate hessian matrix
    hess = hessian(screml_full_loglike, x=out$par, args=args)

    return ( list(beta=beta, V=V, l=l, fixedeffect_vars=fixed_vars, 
                  randomeffect_vars=random_vars, r2=r2, convergence=out$convergence,
                  hess=hess) )
}

screml_full_loglike <- function( par, args ){
    y <- args[['y']]
    P <- args[['P']]
    X <- args[['X']]
    C <- args[['C']]
    ngam <- args[['ngam']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]

	beta <- par[1:ncol(X)] 
    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]<- par[ncol(X)+1:ngam]
    V <- V + t(V)
    hom2 <- 0
    r2 <- par[(ncol(X)+ngam+1):length(par)]

    return( LL(y, P, X, C, vs, beta, hom2, V, r2, random_MMT) )

}

##################
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

    if (model == 'hom') {
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
