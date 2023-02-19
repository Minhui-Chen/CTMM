library(numDeriv)
library(Matrix)
source('util.R')

LL <- function(Y, vs, hom2, V, fixed=NULL, r2=NULL, random_MMT=NULL){
    N <- nrow( Y )
    C <- ncol( Y )
    A <- matrix(1, C, C) * hom2 + V

    if ( is.null(fixed) & is.null(random_MMT) ) {
        AD_inv_sum <- matrix(rep(0, C*C), nrow=C)
        AD_det_sum <- 0
        yADy_sum <- 0
        ADy_sum <- rep(0, C)

        for (i in 1:N) {
            AD <- A + diag(vs[i,])
            #print(vs[i,])

            eval   <- eigen(AD, symmetric=TRUE)$values
            if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
            if( any( diag(AD) < 0 ) ) return(1e12)

            AD_inv <- solve(AD)
            AD_det <- determinant(AD, logarithm=TRUE)$modulus
            yADy <- Y[i,] %*% AD_inv %*% Y[i,]
            ADy <- AD_inv %*% Y[i,]

            AD_inv_sum <- AD_inv_sum + AD_inv
            AD_det_sum <- AD_det_sum + AD_det
            yADy_sum <- yADy_sum + yADy
            ADy_sum <- ADy_sum + ADy
        }

        AD_inv_sum_det <- determinant(AD_inv_sum, logarithm=TRUE)$modulus
        AD_inv_sum_inv <- solve(AD_inv_sum)
        
        #ADy_sum <- as.vector(ADy_sum)  
        L <- AD_det_sum + AD_inv_sum_det + yADy_sum - t(ADy_sum) %*% AD_inv_sum_inv %*% ADy_sum
        L <- 0.5 * L

    } else if ( !is.null(fixed) & is.null(random_MMT) ) {
        AD_det_sum <- 0
        yADy_sum <- 0
        XADy_sum <- rep(0, C)

        Xi <- diag(C)
        for (M in fixed) {
            Xi <- cbind( Xi, kronecker(t(M[1,]),rep(1,C)) )
        }
        XADX_sum <- matrix(rep(0, ncol(Xi) * ncol(Xi)), nrow=ncol(Xi))

        for (i in 1:N) {
            AD <- A + diag(vs[i,])

            eval   <- eigen(AD, symmetric=TRUE)$values
            if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
            if( any( diag(AD) < 0 ) ) return(1e12)

            Xi <- diag(C)
            for (M in fixed) {
                Xi <- cbind( Xi, kronecker(t(M[i,]),rep(1,C)) )
            }

            AD_inv <- solve(AD)
            XADX <- t(Xi) %*% AD_inv %*% Xi
            AD_det <- determinant(AD, logarithm=TRUE)$modulus
            yADy <- Y[i,] %*% AD_inv %*% Y[i,]
            XADy <- t(Xi) %*% AD_inv %*% Y[i,]

            XADX_sum <- XADX_sum + XADX
            AD_det_sum <- AD_det_sum + AD_det
            yADy_sum <- yADy_sum + yADy
            XADy_sum <- XADy_sum + XADy
        }

        eval   <- eigen(XADX_sum, symmetric=TRUE)$values
        if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
        XADX_sum_det <- determinant(XADX_sum, logarithm=TRUE)$modulus
        XADX_sum_inv <- solve(XADX_sum)

        L <- AD_det_sum + XADX_sum_det + yADy_sum - t(XADy_sum) %*% XADX_sum_inv %*% XADy_sum
        L <- 0.5 * L
    } else if ( !is.null(random_MMT) ) {
        X <- kronecker( rep(1,N), diag(C) )
        if ( !is.null(fixed) ) {
            for (M in fixed) {
                X <- cbind( X, kronecker( M, rep(1,C) ) )
            }
        }

        sig2s <- kronecker(diag(N), A) + diag(as.vector(t(vs)))
        for (i in 1:length(random_MMT)){
            sig2s <- sig2s + r2[i] * random_MMT[[i]]
        }
        #print( min(eigen(sig2s, symmetric=TRUE)$values) )

        if ( length(random_MMT) == 1) {
            sig2s_inv <- list()
            det_sig2s <- 0
            i <- 1
            k <- 1
            while (i <= ncol(random_MMT[[1]])) {
                j <- i + sum(random_MMT[[1]][,i]) - 1
                sig2s_k <- sig2s[i:j, i:j]

                eval <- eigen(sig2s_k, symmetric=TRUE)$values
                if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval) < 0) return(1e12)
                #print( min(eval) )
                if( any( diag(sig2s_k) < 0 ) ) return(1e12)

                det_sig2s <- det_sig2s + determinant(sig2s_k, logarithm=TRUE)$modulus
                sig2s_inv[[k]] <- solve(sig2s_k)

                i <- j+1
                k <- k+1
            }
            sig2s_inv <- as.matrix( bdiag(sig2s_inv) )

        } else {
            det_sig2s <- determinant(sig2s, logarithm=TRUE)$modulus
            eval   <- eigen(sig2s,symmetric=TRUE)$values
            if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
            sig2s_inv <- solve( sig2s )
        }

        F <- t(X) %*% sig2s_inv
        B <- F %*% X
        eval   <- eigen(B,symmetric=TRUE)$values
        if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
        M <- sig2s_inv - t(F) %*% solve(B) %*% F
        det_B <- determinant(B, logarithm=TRUE)$modulus
        y <- as.vector(t(Y))
        L <- 0.5 * ( det_sig2s + det_B + y %*% M %*% y )
    }

    return(L)
}

gls <- function(y, X, vs, hom2, V, random_MMT, r2){
    N <- nrow( vs )
    C <- ncol( vs )

    A <- matrix(1, C, C)*hom2 + V
    sig2s <- kronecker(diag(N), A) + diag( as.vector(t(vs)) )

    if ( !is.null( random_MMT ) ) {
        for (i in 1:length(random_MMT)) {
            sig2s <- sig2s + r2[i] * random_MMT[[i]]
        }
    }

    sig2s_inv <- solve(sig2s)
    F <- t(X) %*% sig2s_inv
    B <- F %*% X
    beta <- c(solve( B ) %*% F %*% y)
    return( beta )
}

screml_hom <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_cut=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

    X <- make_ctp_X(N, C, fixed)
    random_MMT <- make_ctp_MMT( random, C )

    if ( is.null( par ) ) {
        hom2 <- var(y) / (length(random)+1)
        par <- rep(hom2, length(random)+1)
    }

    args <- list( Y=Y, vs=vs, fixed=fixed, random_MMT=random_MMT )
    out <- optim_wrap( par, screml_hom_loglike, args, method, FALSE)

	hom2 <- out$par[1]
    r2 <- out$par[2:length(out$par)]
    beta <- gls(y, X, vs, hom2, matrix(0,C,C), random_MMT, r2)
    l <- out$value * (-1)

    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, 0, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim( out, screml_hom_loglike, par, args, method, nrep, FALSE )

        hom2 <- out$par[1]
        r2 <- out$par[2:length(out$par)]
        beta <- gls(y, X, vs, hom2, matrix(rep(0,C*C),nrow=C), random_MMT, r2)
        l <- out$value * (-1)

        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_hom_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, l=l, hess=hess, beta=beta, fixedeffect_vars=fixed_vars,
                  randomeffect_vars=random_vars, r2=r2, convergence=out$convergence, 
                  method=method) )
}

screml_hom_loglike<- function(par, args){
    Y <- args[['Y']]
    vs <- args[['vs']]
    fixed <- args[['fixed']]
    random_MMT <- args[['random_MMT']]
    C <- ncol( Y )

	hom2 <- par[1]
    V <- matrix(0, C, C)
    r2 <- par[2:length(par)]

    ll <- LL(Y, vs, hom2, V, fixed, r2, random_MMT) 
    return( ll )
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
        hom2 <- var(y) / (length(random)+2)
        par <- rep(hom2, length(random)+2)
    }
    
    args <- list( Y=Y, vs=vs, fixed=fixed, random_MMT=random_MMT )
    out <- optim_wrap( par, screml_iid_loglike, args, method, FALSE)

	hom2 <- out$par[1]
	V <- diag(C) * out$par[2]
    r2 <- out$par[3:length(out$par)]
    beta <- gls(y, X, vs, hom2, V, random_MMT, r2)
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim( out, screml_iid_loglike, par, args, method, nrep, FALSE )

        hom2 <- out$par[1]
        V <- diag(C) * out$par[2]
        r2 <- out$par[3:length(out$par)]
        beta <- gls(y, X, vs, hom2, V, random_MMT, r2)
        l <- out$value * (-1)

        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }
    
    # estimate hessian
    hess = hessian(screml_iid_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixed_vars,
                  randomeffect_vars=random_vars, r2=r2, convergence=out$convergence,
                  method=method) )
}

screml_iid_loglike <- function(par, args){
    Y <- args[['Y']]
    vs <- args[['vs']]
    fixed <- args[['fixed']]
    random_MMT <- args[['random_MMT']]
    C <- ncol( Y )

	hom2 <- par[1]
	V <- diag(C) * par[2]
    r2 <- par[3:length(par)]

    L <- LL(Y, vs, hom2, V, fixed, r2, random_MMT)

    return( L )
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
        hom2 <- var(y) / (length(random)+2)
        par <- rep(hom2, length(random)+1+C)
    }

    args <- list( Y=Y, vs=vs, fixed=fixed, random_MMT=random_MMT )
    out <- optim_wrap( par, screml_free_loglike, args, method, FALSE)

	hom2 <- out$par[1]
	V <- diag(out$par[1+1:C])
    r2 <- out$par[(C+2):length(out$par)]
    beta <- gls(y, X, vs, hom2, V, random_MMT, r2)
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim( out, screml_free_loglike, par, args, method, nrep, FALSE )

        hom2 <- out$par[1]
        V <- diag(out$par[1+1:C])
        r2 <- out$par[(C+2):length(out$par)]
        beta <- gls(y, X, vs, hom2, V, random_MMT, r2)
        l <- out$value * (-1)

        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_free_loglike, x=out$par, args=args)

    return ( list(hom2=hom2, V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixed_vars,
                 randomeffect_vars=random_vars, r2=r2, convergence=out$convergence, method=method ))
}

screml_free_loglike<- function(par, args){
    Y <- args[['Y']]
    vs <- args[['vs']]
    fixed <- args[['fixed']]
    random_MMT <- args[['random_MMT']]
    C <- ncol( Y )

	hom2 <- par[1]
	V <- diag(par[1+1:C])
    r2 <- par[(C+2):length(par)]

    return( LL(Y, vs, hom2, V, fixed, r2, random_MMT) )
}

screml_full <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_cut=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))
	ngam   <- C*(C+1)/2

    X <- make_ctp_X(N, C, fixed)
    random_MMT <- make_ctp_MMT( random, C )

    if ( is.null( par ) ) {
        v1 <- var(y) / (length(random)+1)
        V <- diag(C)[ lower.tri(diag(C),diag=T) ] * c(v1)
        par <- c( V, rep(v1,length(random)) )
    }

    args <- list( Y=Y, vs=vs, fixed=fixed, random_MMT=random_MMT )
    out <- optim_wrap(par, screml_full_loglike, args, method, FALSE)

    V   <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]  <- out$par[1:ngam]
    V <- V + t(V)
    r2 <- out$par[(ngam+1):length(out$par)]
    hom2 <- 0
    beta <- gls(y, X, vs, hom2, V, random_MMT, r2)
    l <- out$value * (-1)

    ct_overall_var <- RandomeffectVariance_( V, P )
    fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
    random_vars <- RandomeffectVariance( r2, random )[[2]]

    if ( check_optim(out, hom2, ct_overall_var, fixed_vars, random_vars, overVariance_cut) ) {
        out <- re_optim( out, screml_full_loglike, par, args, method, nrep, FALSE )

        V  <- matrix( 0, C, C )
        V[lower.tri(V,diag=T)]  <- out$par[1:ngam]
        V <- V + t(V)
        r2 <- out$par[(ngam+1):length(out$par)]
        hom2 <- 0
        beta <- gls(y, X, vs, hom2, V, random_MMT, r2)
        l <- out$value * (-1)

        ct_overall_var <- RandomeffectVariance_( V, P )
        fixed_vars <- FixedeffectVariance( beta, c(list(P), fixed) )
        random_vars <- RandomeffectVariance( r2, random )[[2]]
    }

    # estimate hessian matrix
    hess = hessian(screml_full_loglike, x=out$par, args=args)

    return ( list( V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixed_vars,
                 randomeffect_vars=random_vars, r2=r2, convergence=out$convergence, method=method ))
}

screml_full_loglike <- function(par, args){
    Y <- args[['Y']]
    vs <- args[['vs']]
    fixed <- args[['fixed']]
    random_MMT <- args[['random_MMT']]
    C <- ncol( Y )
	ngam <- C*(C+1)/2

    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]<- par[1:ngam]
    V <- V + t(V)
    hom2 <- 0
    r2 <- par[(ngam+1):length(par)]

    return( LL(Y, vs, hom2, V, fixed, r2, random_MMT) )
}

##################
# runs only when script is run by itself
if (sys.nframe() == 0){
    args <- commandArgs(trailingOnly=TRUE)
    if (length( args ) == 1) {
        source( args[1] )
    } else {
        y_f <- args[1]
        P_f <- args[2]
        vs_f <- args[3]
        model <- args[4]
        out_f <- args[5]

        # read data
        Y <- as.matrix(read.table(y_f))
        vs <- as.matrix(read.table(vs_f))
        P <- as.matrix(read.table(P_f))

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
}
