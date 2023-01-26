library(numDeriv)
library(mvtnorm)
source('bin/util.R')


LL <- function(y, X, vs, hom2, beta, V, r2=NULL, random_MMT=NULL){
    N <- nrow( vs )
    C <- ncol( vs )

    A  <- matrix(rep(1, C*C), nrow=C) * hom2 + V

    if ( is.null(random_MMT) ){
        yd <- y - X %*% beta
        Yd <- matrix(yd, ncol=C, byrow=TRUE)

        l <- 0
        for (i in 1:N) {
            D_i <- diag(vs[i,])
            AD <- A + D_i

            eval   <- eigen(AD, symmetric=TRUE)$values
            if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
            if( any( diag(AD) < 0 ) ) return(1e12)

            det_AD <- determinant(AD, logarithm=TRUE)$modulus
            inv_AD <- solve(AD)
            l <- l + det_AD + Yd[i,] %*% inv_AD %*% Yd[i,]
        }

        l <- 0.5 * l
    } else if (length(random_MMT) == 1) {
        # assmue M is sorted with structure 1_a, 1_b, 1_c, so is MMT
        yd <- y - X %*% beta
        Yd <- matrix(yd, ncol=C, byrow=TRUE)

        sig2s <- kronecker( diag(N), A ) + diag( as.vector(t(vs)) )
        sig2s <- sig2s + r2[1] * random_MMT[[1]]

        l <- 0
        i <- 1 
        while (i <= ncol(random_MMT[[1]])) {
            j <- i + sum(random_MMT[[1]][,i]) - 1
            sig2s_k <- sig2s[i:j, i:j]

            eval <- eigen(sig2s_k, symmetric=TRUE)$values
            if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
            if( any( diag(sig2s_k) < 0 ) ) return(1e12)

            det_sig2s_k <- determinant(sig2s_k, logarithm=TRUE)$modulus
            inv_sig2s_k <- solve(sig2s_k)
            l <- l + det_sig2s_k + yd[i:j] %*% inv_sig2s_k %*% yd[i:j]

            i <- j+1
        }

        l <- 0.5 * l

    } else {
        sig2s <- kronecker( diag(N), A ) + diag( as.vector(t(vs)) )
        for (i in 1:length(random_MMT)){
            sig2s <- sig2s + r2[i] * random_MMT[[i]]
        }
        l <- dmvnorm(y, mean=X %*% beta, sigma=sig2s, log=TRUE) * (-1)
    }
    return (l)
}

screml_hom <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_cut=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

    X <- make_ctp_X(N, C, fixed)
    random_MMT <- make_ctp_MMT( random, C )

    if ( is.null(par) ) {
        beta   <- solve(t(X)%*%X) %*% ( t(X) %*% y ) 
        hom2 <- var(y - X %*% beta) / (length(random)+1)
        par <- c( hom2, beta, rep(hom2,length(random)) )
    }
    print( par )
    
    args <- list( y=y, X=X, vs=vs, random_MMT=random_MMT )
    out <- optim_wrap( par, screml_hom_loglike, args, method, FALSE)

	hom2 <- out$par[1]
	beta  <- out$par[1+1:ncol(X)]
    r2 <- out$par[(2+ncol(X)):length(out$par)]
    l <- out$value * (-1)

    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, 0, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim(out, screml_hom_loglike, par, args, method, nrep, FALSE)

        hom2 <- out$par[1]
        beta  <- out$par[1+1:ncol(X)]
        r2 <- out$par[(2+ncol(X)):length(out$par)]
        l <- out$value * (-1)

        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_hom_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, beta=beta, l=l, hess=hess, fixedeffect_vars=fixed_vars,
                  randomeffect_vars=random_vars, r2=r2, convergence=out$convergence,
                 method=method ) )
}

screml_hom_loglike<- function(par, args){
    y <- args[['y']]
    X <- args[['X']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]
    C <- ncol(vs)

	hom2 <- par[1]
	beta <- par[1+1:ncol(X)] 
    V <- matrix(rep(0, C*C), nrow=C)
    r2 <- par[(ncol(X)+2):length(par)]

    return( LL(y, X, vs, hom2, beta, V, r2, random_MMT) )

}

screml_iid <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_cut=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

    X <- make_ctp_X(N, C, fixed)
    random_MMT <- make_ctp_MMT( random, C )

    if ( is.null( par ) ) {
        beta   <- solve(t(X)%*%X) %*% ( t(X) %*% y ) 
        hom2 <- var(y - X %*% beta) / (length(random)+2)
        par <- c( hom2, hom2, beta, rep(hom2, length(random)) )
    }

    args <- list( y=y, X=X, vs=vs, random_MMT=random_MMT )
    out<- optim_wrap( par, screml_iid_loglike, args, method, FALSE )

	hom2 <- out$par[1]
	V <- diag(C) * out$par[2]
	beta  <- out$par[2+1:ncol(X)]
    r2 <- out$par[(3+ncol(X)):length(out$par)]
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim(out, screml_iid_loglike, par, args, method, nrep, FALSE)

        hom2 <- out$par[1]
        V <- diag(C) * out$par[2]
        beta  <- out$par[2+1:ncol(X)]
        r2 <- out$par[(3+ncol(X)):length(out$par)]
        l <- out$value * (-1)

        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian
    hess = hessian(screml_iid_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, beta=beta, V=V, l=l, hess=hess, fixedeffect_vars=fixed_vars, 
                  randomeffect_vars=random_vars, r2=r2, convergence=out$convergence,
                  method=method) )
}

screml_iid_loglike<- function(par, args){
    y <- args[['y']]
    X <- args[['X']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]
    C <- ncol(vs)

	hom2 <- par[1]
	V <- diag(C) * par[2]
	beta    <- par[2+1:ncol(X)] 
    r2 <- par[(ncol(X)+3):length(par)]

    return( LL(y, X, vs, hom2, beta, V, r2, random_MMT) )
}

screml_free <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_cut=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

    X <- make_ctp_X(N, C, fixed)
    random_MMT <- make_ctp_MMT( random, C )

    if ( is.null( par ) ) {
        beta   <- solve(t(X)%*%X) %*% ( t(X) %*% y )
        hom2 <- var(y - X %*% beta) / (length(random)+2)
        par <- c( rep(hom2, C+1), beta, rep(hom2, length(random)) )
    }

    args <- list( y=y, X=X, vs=vs, random_MMT=random_MMT )
    out <- optim_wrap( par, screml_free_loglike, args, method, FALSE)

	hom2 <- out$par[1]
	V <- diag(out$par[1+1:C])
	beta  <- out$par[C+1+1:ncol(X)]
    r2 <- out$par[(C+2+ncol(X)):length(out$par)]
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim(out, screml_free_loglike, par, args, method, nrep, FALSE)

        hom2 <- out$par[1]
        V <- diag(out$par[1+1:C])
        beta  <- out$par[C+1+1:ncol(X)]
        r2 <- out$par[(C+2+ncol(X)):length(out$par)]
        l <- out$value * (-1)

        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_free_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, beta=beta, V=V, l=l, hess=hess, fixedeffect_vars=fixed_vars, 
                 randomeffect_vars=random_vars, r2=r2, convergence=out$convergence,
                method=method ))
}

screml_free_loglike<- function(par, args){
    y <- args[['y']]
    X <- args[['X']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]
    C <- ncol(vs)

	hom2 <- par[1]
	V <- diag(par[1+1:C])
	beta <- par[C+1+1:ncol(X)] 
    r2 <- par[(ncol(X)+C+2):length(par)]

    return( LL(y, X, vs, hom2, beta, V, r2, random_MMT) )
}

screml_full <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_cut=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)
    ngam   <- C*(C+1)/2

    y <- as.vector(t(Y))

    X <- make_ctp_X(N, C, fixed)
    random_MMT <- make_ctp_MMT( random, C )

    if ( is.null( par ) ) {
        beta   <- solve(t(X)%*%X) %*% ( t(X) %*% y ) 
        v1 <- var(y - X %*% beta) / (length(random)+1)
        V <- diag(C)[ lower.tri(diag(C),diag=T) ] * c(v1)
        par <- c( V, beta, rep(v1,length(random)) )
    }

    args <- list( y=y, X=X, vs=vs, random_MMT=random_MMT )
    out <- optim( par=par, fn=screml_full_loglike, args=args, method=method, hessian=FALSE)

    V   <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]  <- out$par[1:ngam]
    V <- V + t(V)
	beta    <- out$par[ngam+1:ncol(X)]
    r2 <- out$par[(ngam+ncol(X)+1):length(out$par)]
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, 0, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim(out, screml_full_loglike, par, args, method, nrep, FALSE)

        V   <- matrix( 0, C, C )
        V[lower.tri(V,diag=T)]  <- out$par[1:ngam]
        V <- V + t(V)
        beta    <- out$par[ngam+1:ncol(X)]
        r2 <- out$par[(ngam+ncol(X)+1):length(out$par)]
        l <- out$value * (-1)

        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_full_loglike, x=out$par, args=args)

    return ( list(beta=beta, V=V, l=l, hess=hess, fixedeffect_vars=fixed_vars,
                 randomeffect_vars=random_vars, r2=r2, convergence=out$convergence, method=method ))
}

screml_full_loglike <- function(par, args){
    y <- args[['y']]
    X <- args[['X']]
    vs <- args[['vs']]
    random_MMT <- args[['random_MMT']]
    N <- nrow(vs)
    C <- ncol(vs)
    ngam   <- C*(C+1)/2

    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]<- par[1:ngam]
    V <- V + t(V)
	beta <- par[ngam+1:ncol(X)] 
    hom2 <- 0
    r2 <- par[(ncol(X)+ngam+1):length(par)]

    return( LL(y, X, vs, hom2, beta, V, r2, random_MMT) )
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
    Y <- as.matrix(read.table(y_f))
    vs <- as.matrix(read.table(vs_f))
    P <- as.matrix(read.table(P_f))
    #print(P)

    if (model == 'hom') {
        out = screml_hom(Y, P, vs)
        save(out, file=out_f)
    } else if (model == 'iid') {
        out = screml_iid(Y, P, vs)
        save(out, file=out_f)
    } else if (model == 'free') {
        out = screml_free(Y, P, vs)
        save(out, file=out_f)
    } else if (model == 'full') {
        out = screml_full(Y, P, vs)
        save(out, file=out_f)
    }
}
