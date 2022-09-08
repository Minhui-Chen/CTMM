library(numDeriv)
library(mvtnorm)
source('bin/reml.R')


LL <- function(y, X, N, C, vs, hom2, beta, V, random_variances=NULL, random_MMT=NULL){
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
        sig2s <- sig2s + random_variances[1] * random_MMT[[1]]

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
            sig2s <- sig2s + random_variances[i] * random_MMT[[i]]
        }
        l <- dmvnorm(y, mean=X %*% beta, sigma=sig2s, log=TRUE) * (-1)
    }
    return (l)
}

screml_hom <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, method="BFGS", par=NULL, nrep=10
){
    # the script actually run 2*nrep replications of optimization
    # if it's not converged in the first run
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

    X <- kronecker(rep(1, N), diag(C))
    if ( !is.null( fixed ) ) {
        for ( covar in fixed ) {
            X <- cbind( X, kronecker(covar, rep(1,C)) )
        }
    }

	beta   <- solve(t(X)%*%X) %*% ( t(X) %*% y ) 
	hom2 <- var(y - X %*% beta) 
    par2 <- c(hom2, beta)
    if ( !is.null( random ) ){
        hom2 <- par2[1] / (length(random)+1)
        par2[1] <- hom2
        for (i in 1:length(random)) {
            par2 <- c(par2, hom2)
        }
    }
    if ( is.null(par) ) {
        par <- par2
    }
    
    random_MMT <- NULL
    if ( !is.null( random ) ) {
        random_MMT <- list()
        for (i in 1:length(random)) {
            random[[i]] <- kronecker( random[[i]], rep(1,C) )
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

    out <- optim( par=par, fn=screml_hom_loglike, 
        y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

	hom2_ <- out$par[1]
	beta_  <- out$par[1+1:ncol(X)]
    overVariance <- (hom2_ > overVariance_threshold)
    overVariance <- overVariance | (FixedeffectVariance( beta_, c(list(P), fixed), overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        overVariance <- overVariance | (RandomeffectVariance( out$par[(2+ncol(X)):length(out$par)], random,
                                                             overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:nrep){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_hom_loglike, 
                    y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

            if ( out_$value < out$value ) {
                out <- out_
            }

            par2_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par2_, fn=screml_hom_loglike, 
                    y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

            if ( out_$value < out$value ) {
                out <- out_
            }
        }
    }

	hom2 <- out$par[1]
	beta  <- out$par[1+1:ncol(X)]
    r2 <- out$par[(ncol(X)+2):length(out$par)]
    if ( !is.null( random ) ) {
        randomeffect_vars <- RandomeffectVariance(r2, random)[[2]]
    } else {
        randomeffect_vars <- NULL
        r2 <- NULL
    }
    l <- out$value * (-1)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]

    # estimate hessian matrix
    hess = hessian(screml_hom_loglike, x=out$par, y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT)

    return ( list(hom2=hom2, beta=beta, l=l, hess=hess, fixedeffect_vars=fixedeffect_vars,
                  randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence,
                 method=method ) )
                  #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence,
}

screml_hom_loglike<- function(par, y, X, N, C, vs, random_MMT){
	hom2 <- par[1]
	beta    <- par[1+1:ncol(X)] 
    V <- matrix(rep(0, C*C), nrow=C)
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ncol(X)+2):length(par)]
    }

    return( LL(y, X, N, C, vs, hom2, beta, V, random_variances, random_MMT) )

}

screml_iid <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, method="BFGS", par=NULL, nrep=10
){
    # the script actually run 2*nrep replications of optimization
    # if it's not converged in the first run
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

    X <- kronecker(rep(1, N), diag(C))
    if ( !is.null( fixed ) ) {
        for ( covar in fixed ) {
            X <- cbind( X, kronecker(covar, rep(1,C)) )
        }
    }

	beta   <- solve(t(X)%*%X) %*% ( t(X) %*% y ) 
	hom2 <- var(y - X %*% beta) / 2
	V <- hom2
    par2 <- c(hom2, V, beta)
    if ( !is.null( random ) ) {
        hom2 <- 2 * par2[1] / (length(random)+2)
        par2[1:2] <- hom2
        for (i in 1:length(random)) {
            par2 <- c(par2, hom2)
        }
    }
    if ( is.null( par ) ) {
        par <- par2
    }

    random_MMT <- NULL
    if ( !is.null( random ) ) {
        random_MMT <- list()
        for (i in 1:length(random)) {
            random[[i]] <- kronecker( random[[i]], rep(1,C) )
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

    out<- optim( par=par, fn=screml_iid_loglike, 
        y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

	hom2_ <- out$par[1]
	V_ <- diag(C) * out$par[2]
	beta_  <- out$par[2+1:ncol(X)]
    overVariance <- (hom2_ > overVariance_threshold)
    overVariance <- overVariance | (RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]])
    overVariance <- overVariance | (FixedeffectVariance( beta_, c(list(P), fixed), overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        overVariance <- overVariance | (RandomeffectVariance( out$par[(3+ncol(X)):length(out$par)], random,
                                                             overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:nrep){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_iid_loglike, 
                    y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

            if (out_$value < out$value) {
                out <- out_
            }

            par2_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par2_, fn=screml_iid_loglike, 
                    y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

            if (out_$value < out$value) {
                out <- out_
            }
        }
    }

	hom2 <- out$par[1]
	V <- diag(C) * out$par[2]
	beta  <- out$par[2+1:ncol(X)]
    r2 <- out$par[(ncol(X)+3):length(out$par)]
    if ( !is.null( random ) ) {
        randomeffect_vars <- RandomeffectVariance(r2, random)[[2]]
    } else {
        randomeffect_vars <- NULL
        r2 <- NULL
    }
    l <- out$value * (-1)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]
    
    # estimate hessian
    hess = hessian(screml_iid_loglike, x=out$par, y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT)

    return ( list(hom2=hom2, beta=beta, V=V, l=l, hess=hess, fixedeffect_vars=fixedeffect_vars, 
                  randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence,
                  method=method) )
                  #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence,
}

screml_iid_loglike<- function(par, y, X, N, C, vs, random_MMT){
	hom2 <- par[1]
	V <- diag(C) * par[2]
	beta    <- par[2+1:ncol(X)] 
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ncol(X)+3):length(par)]
    }

    return( LL(y, X, N, C, vs, hom2, beta, V, random_variances, random_MMT) )
}

screml_free <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, method="BFGS", par=NULL, nrep=10
){
    # the script actually run 2*nrep replications of optimization
    # if it's not converged in the first run
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

    X <- kronecker(rep(1, N), diag(C))
    if ( !is.null( fixed ) ) {
        for ( covar in fixed ) {
            X <- cbind( X, kronecker(covar, rep(1,C)) )
        }
    }

	beta   <- solve(t(X)%*%X) %*% ( t(X) %*% y ) # cell type effect
	hom2 <- var(y - X %*% beta) / 2
	V <- rep(1,C) * as.vector(hom2)
    par2 <- c(hom2, V, beta)
    if ( !is.null( random ) ) {
        hom2 <- 2 * par2[1] / (length(random)+2)
        par2[1:(C+1)] <- hom2
        for (i in 1:length(random)) {
            par2 <- c(par2, hom2)
        }
    }
    if ( is.null( par ) ) {
        par <- par2
    }

    random_MMT <- NULL
    if ( !is.null( random ) ) {
        random_MMT <- list()
        for (i in 1:length(random)) {
            random[[i]] <- kronecker( random[[i]], rep(1,C) )
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

    out <- optim( par=par, fn=screml_free_loglike, 
        y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

	hom2_ <- out$par[1]
	V_ <- diag(out$par[1+1:C])
	beta_  <- out$par[C+1+1:ncol(X)]
    overVariance <- (hom2_ > overVariance_threshold)
    overVariance <- overVariance | (RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]])
    overVariance <- overVariance | (FixedeffectVariance( beta_, c(list(P), fixed),overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        overVariance <- overVariance | (RandomeffectVariance( out$par[(C+2+ncol(X)):length(out$par)], random,
                                                             overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:nrep){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_free_loglike, 
                    y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

            if (out_$value < out$value) {
                out <- out_
            }

            par2_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par2_, fn=screml_free_loglike, 
                    y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

            if (out_$value < out$value) {
                out <- out_
            }
        }
    }

	hom2 <- out$par[1]
	V <- diag(out$par[1+1:C])
	beta  <- out$par[C+1+1:ncol(X)]
    r2 <- out$par[(C+2+ncol(X)):length(out$par)]
    if ( !is.null( random ) ) {
        randomeffect_vars <- RandomeffectVariance(r2, random)[[2]]
    } else {
        randomeffect_vars <- NULL
        r2 <- NULL
    }
    l <- out$value * (-1)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]

    # estimate hessian matrix
    hess = hessian(screml_free_loglike, x=out$par, y=y, X=X, N=N, C=C, vs=vs, random_MMT=random_MMT)

    return ( list(hom2=hom2, beta=beta, V=V, l=l, hess=hess, fixedeffect_vars=fixedeffect_vars, 
                 randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence,
                method=method ))
}

screml_free_loglike<- function(par, y, X, N, C, vs, random_MMT){
	hom2 <- par[1]
	V <- diag(par[1+1:C])
	beta <- par[C+1+1:ncol(X)] 
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ncol(X)+C+2):length(par)]
    }

    return( LL(y, X, N, C, vs, hom2, beta, V, random_variances, random_MMT) )
}

screml_full <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, method="BFGS", par=NULL, nrep=10
){
    # the script actually run 2*nrep replications of optimization
    # if it's not converged in the first run
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))
    ngam   <- C*(C+1)/2

    X <- kronecker(rep(1, N), diag(C))
    if ( !is.null( fixed ) ) {
        for ( covar in fixed ) {
            X <- cbind( X, kronecker(covar, rep(1,C)) )
        }
    }

	beta   <- solve(t(X)%*%X) %*% ( t(X) %*% y ) 
	V <- as.numeric( diag(C)[ lower.tri(diag(C),diag=T) ] ) * as.vector(var(y - X %*% beta))
    par2 <- c(V, beta)
    if ( !is.null( random ) ) {
        v1 <- par2[1] / (length(random)+1)
        V <- as.numeric( diag(C)[ lower.tri(diag(C),diag=T) ] ) * v1
        par2 <- c(V, beta)
        for (i in 1:length(random)) {
            par2 <- c(par2, v1)
        }
    }
    if ( is.null( par ) ) {
        par <- par2
    }

    random_MMT <- NULL
    if ( !is.null( random ) ) {
        random_MMT <- list()
        for (i in 1:length(random)) {
            random[[i]] <- kronecker( random[[i]], rep(1,C) )
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

    out <- optim( par=par, fn=screml_full_loglike, 
        y=y, X=X, N=N, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)

    V_   <- matrix( 0, C, C )
    V_[lower.tri(V_,diag=T)]  <- out$par[1:ngam]
    V_ <- V_ + t(V_)
	beta_    <- out$par[ngam+1:ncol(X)]
    overVariance <- FixedeffectVariance( beta_, c(list(P), fixed), overVariance_threshold )[[1]]
    overVariance <- overVariance | (RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        overVariance <- overVariance | (RandomeffectVariance( out$par[(ngam+ncol(X)+1):length(out$par)], random,
                                                             overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:nrep){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_full_loglike, 
                    y=y, X=X, N=N, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)
            
            if (out_$value < out$value) {
                out <- out_
            }

            par2_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par2_, fn=screml_full_loglike, 
                    y=y, X=X, N=N, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT, method=method, hessian=FALSE)
            
            if (out_$value < out$value) {
                out <- out_
            }
        }
    }

    V<- matrix( 0, C, C )
    V[lower.tri(V,diag=T)] <- out$par[1:ngam]
    V<- V + t(V)
	beta  <- out$par[ngam+1:ncol(X)]
    r2 <- out$par[(ngam+ncol(X)+1):length(out$par)]
    if ( !is.null( random ) ) {
        randomeffect_vars <- RandomeffectVariance(r2, random)[[2]]
    } else {
        randomeffect_vars <- NULL
        r2 <- NULL
    }
    l <- out$value * (-1)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]

    # estimate hessian matrix
    hess = hessian(screml_full_loglike, x=out$par, y=y, X=X, N=N, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT)

    return ( list(beta=beta, V=V, l=l, hess=hess, fixedeffect_vars=fixedeffect_vars,
                 randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence, method=method ))
}

screml_full_loglike <- function(par, y, X, N, C, ngam, vs, random_MMT){
    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]<- par[1:ngam]
    V <- V + t(V)
	beta <- par[ngam+1:ncol(X)] 
    hom2 <- 0
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ncol(X)+ngam+1):length(par)]
    }

    return( LL(y, X, N, C, vs, hom2, beta, V, random_variances, random_MMT) )
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

    if (model == 'null') {
        out = screml_null(Y, P, vs)
        save(out, file=out_f)
    } else if (model == 'hom') {
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
