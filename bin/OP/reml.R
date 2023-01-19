library(numDeriv)
library(MASS)
source('bin/reml.R')

LL <- function(y, P, X, C, vs, hom2, V, random_variances=NULL, random_MMT=NULL){

    sig2s <- hom2 + diag( P %*% V %*% t(P) ) + vs
	if( any( sig2s < 0 ) ) return(1e12)

    if ( is.null( random_MMT ) ) {
        sig2s_inv <- 1/sig2s
        A <- sweep( t(X), 2, sig2s_inv, '*') # t(X) %*% diag(sig2s_inv)
        B <- A %*% X
        eval   <- eigen(B,symmetric=TRUE)$values
        if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
        M <- diag(sig2s_inv) - t(A) %*% solve(B) %*% A
        L <- sum(log( sig2s ))
    } else {
        sig2s <- diag(sig2s)
        for (i in 1:length(random_MMT)) {
            sig2s <- sig2s + random_variances[i] * random_MMT[[i]]
        }
        eval   <- eigen(sig2s,symmetric=TRUE)$values
        if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
        sig2s_inv <- solve( sig2s )
        A <- t(X) %*% sig2s_inv
        B <- A %*% X
        eval   <- eigen(B,symmetric=TRUE)$values
        if (max(eval)/(min(eval)+1e-99) > 1e8 | min(eval)<0) return(1e12)
        M <- sig2s_inv - t(A) %*% solve(B) %*% A
        L <- determinant(sig2s, logarithm=TRUE)$modulus
    }

    det_B <- determinant(B, logarithm=TRUE)$modulus
    L <- 0.5 * ( L + det_B + t(y) %*% M %*% y )
    return ( L )
}

fixed_random_vars <- function(y, P, X, C, vs, hom2, V, fixed, random, r2){

    sig2s <- hom2 + diag( P %*% V %*% t(P) ) + vs

    if ( !is.null( random ) ) {
        randomeffect_vars <-RandomeffectVariance(r2, random)[[2]]
        #randomV <- RandomeffectVariance(r2, random)[[3]]

        sig2s <- diag(sig2s)
        for (i in 1:length(random)) {
            sig2s <- sig2s + r2[i] * random[[i]] %*% t(random[[i]])
        }
        sig2s_inv <- solve( sig2s )
        A <- t(X) %*% sig2s_inv

    } else {
        sig2s_inv <- 1/sig2s
        A <- sweep( t(X), 2, sig2s_inv, '*') # t(X) %*% diag(sig2s_inv)
        randomeffect_vars <- NULL
        #randomV <- NULL
    }

    B <- A %*% X
    beta <- c(solve( B ) %*% A %*% y)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]

    return( list(beta=beta, fixedeffect_vars=fixedeffect_vars, randomeffect_vars=randomeffect_vars) )
    #return( list(beta=beta, fixedeffect_vars=fixedeffect_vars, randomV=randomV, randomeffect_vars=randomeffect_vars) )
}

screml_hom <- function(y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, par=NULL){

	C      <- ncol(P)  # cell type number

    X <- P
    if ( !is.null( fixed ) )  {
        for (covar in fixed) {
            X <- cbind(X, covar) 
        }
    }

	hom2 <- var(y) 
    par2 <- c(hom2)
    if ( !is.null( random ) ) {
        hom2 <- par2[1] / (length(random)+1)
        par2[1] <- hom2
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
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

	out <- optim( par=par, fn=screml_hom_loglike, 
		y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

	hom2_ <- out$par[1]
    overVariance <- (hom2_ > overVariance_threshold ) 
    if ( !is.null( random ) ) {
        random_variances <- out$par[2:length(out$par)]
        overVariance <- overVariance | (RandomeffectVariance( random_variances, random, 
                                                             overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:10){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_hom_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_hom_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

	hom2 <- out$par[1]
    V <- matrix(rep(0,C*C), nrow=C)
    if ( !is.null( random ) ) {
        r2 <- out$par[2:length(out$par)]
    } else {
        r2 <- NULL
    }
    l <- out$value * (-1)

    vars <- fixed_random_vars(y, P, X, C, vs, hom2, V, fixed, random, r2)
    beta <- vars$beta
    fixedeffect_vars <- vars$fixedeffect_vars
    #randomV <- vars$randomV
    randomeffect_vars <- vars$randomeffect_vars

    # estimate hessian matrix
    hess = hessian(screml_hom_loglike, x=out$par, y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('Hom', err))
    #})
    #try(solve(out$hessian))

    return ( list(hom2=hom2, l=l, hess=hess, beta=beta, fixedeffect_vars=fixedeffect_vars,
                  randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence) )
                  #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence) )
}

screml_hom_loglike <- function(par, y, P, X, C, vs, random_MMT){
	hom2 <- par[1]
    #print(hom2)
    #Sys.sleep(1)
    V <- matrix( rep(0,C*C), nrow=C )
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[2:length(par)]
    }

    l <- LL(y, P, X, C, vs, hom2, V, random_variances, random_MMT)
    #print(l)

    return( l )

}

screml_iid <- function(y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, par=NULL){

	C      <- ncol(P)  # cell type number
    pi  <- colMeans(P)
    pd  <- scale(P, scale=F)
    S   <- (t(pd) %*% pd ) / nrow(P)

    X <- P
    if ( !is.null( fixed ) )  {
        for (covar in fixed) {
            X <- cbind(X, covar) 
        }
    }

	hom2 <- var(y) / 2
	V <- hom2
    par2 <- c(hom2, V)
    if ( !is.null( random ) ) {
        hom2 <- 2 * par2[1] / (length(random)+2)
        par2[c(1,2)] <- hom2
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
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

	out <- optim( par=par, fn=screml_iid_loglike,
		y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

	hom2_ <- out$par[1]
	V_ <- diag(C) * out$par[2]
    overVariance <- (hom2_ > overVariance_threshold)
    overVariance <- overVariance | (RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        random_variances <- out$par[3:length(out$par)]
        overVariance <- overVariance | (RandomeffectVariance( random_variances, random, 
                                                             overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:10){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_iid_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_iid_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

	hom2 <- out$par[1]
	V <- diag(C) * out$par[2]
    if ( !is.null( random ) ) {
        r2 <- out$par[3:length(out$par)]
    } else {
        r2 <- NULL
    }
    l <- out$value * (-1)

    vars <- fixed_random_vars(y, P, X, C, vs, hom2, V, fixed, random, r2)
    beta <- vars$beta
    fixedeffect_vars <- vars$fixedeffect_vars
    #randomV <- vars$randomV
    randomeffect_vars <- vars$randomeffect_vars
    
    # estimate hessian
    hess = hessian(screml_iid_loglike, x=out$par, y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('IID', err))
    #})
    #try(solve(out$hessian))
    return ( list(hom2=hom2, V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixedeffect_vars, 
                  randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence) )
                  #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence) )
}

screml_iid_loglike<- function(par, y, P, X, C, vs, random_MMT){
	hom2 <- par[1]
	V <- diag(C) * par[2]
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[3:length(par)]
    }

    return( LL(y, P, X, C, vs, hom2, V, random_variances, random_MMT) )
}

screml_free <- function(y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, par=NULL){

	C   <- ncol(P)  # cell type number
    pi  <- colMeans(P)
    pd  <- scale(P, scale=F)
    S   <- (t(pd) %*% pd ) / nrow(P)

    X <- P
    if ( !is.null( fixed ) )  {
        for (covar in fixed) {
            X <- cbind(X, covar) 
        }
    }

	hom2 <- var(y) / 2
	V <- rep(1,C) * as.vector( hom2 )
    par2 <- c(hom2, V)
    if ( !is.null( random ) ) {
        hom2 <- 2 * par2[1] / (length(random)+2)
        par2[1:length(par2)] <- hom2
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
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

	out<- optim( par=par, fn=screml_free_loglike, 
		y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

	hom2_ <- out$par[1]
	V_ <- diag(out$par[1+1:C])
    overVariance <- (hom2_ > overVariance_threshold)
    overVariance <- overVariance | (RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        random_variances <- out$par[(C+2):length(out$par)]
        overVariance <- overVariance | (RandomeffectVariance(random_variances, random, overVariance_threshold)[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:10){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_free_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_free_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

	hom2 <- out$par[1]
	V <- diag(out$par[1+1:C])
    if ( !is.null( random ) ) {
        r2 <- out$par[(C+2):length(out$par)]
    } else {
        r2 <- NULL
    }
    l <- out$value * (-1)

    vars <- fixed_random_vars(y, P, X, C, vs, hom2, V, fixed, random, r2)
    beta <- vars$beta
    fixedeffect_vars <- vars$fixedeffect_vars
    #randomV <- vars$randomV
    randomeffect_vars <- vars$randomeffect_vars

    # estimate hessian matrix
    hess = hessian(screml_free_loglike, x=out$par, y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('Free', err))
    #})
    return ( list(hom2=hom2, V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixedeffect_vars,
                 randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence ))
                 #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence ))
}

screml_free_loglike <- function(par, y, P, X, C, vs, random_MMT){
	hom2 <- par[1]
	V <- diag(par[1+1:C])
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(C+2):length(par)]
    }

    return ( LL(y, P, X, C, vs, hom2, V, random_variances, random_MMT) )
}

screml_full <- function(y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, par=NULL){

	C      <- ncol(P)  # cell type number
	ngam   <- C*(C+1)/2 # number of entries in gamma matrix # should it be C + C*(C+1)/2?
    pi  <- colMeans(P)
    pd  <- scale(P, scale=F)
    S   <- (t(pd) %*% pd ) / nrow(P)

    X <- P
    if ( !is.null( fixed ) )  {
        for (covar in fixed) {
            X <- cbind(X, covar) 
        }
    }

	V <- as.numeric( diag(C)[ lower.tri(diag(C),diag=T) ] ) * as.vector(var(y))
    par2 <- c(V)
    if ( !is.null( random ) ) {
        v1 <- par2[1] / (length(random)+1)
        par2 <- as.numeric( diag(C)[ lower.tri(diag(C),diag=T) ] ) * v1
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
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

	out <- optim( par=par, fn=screml_full_loglike, 
		y=y, P=P, X=X, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)

    V_   <- matrix( 0, C, C )
    V_[lower.tri(V_,diag=T)]  <- out$par[1:ngam]
    V_ <- V_ + t(V_)
    overVariance <- RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]]
    if ( !is.null( random ) ) {
        random_variances <- out$par[(ngam+1):length(out$par)]
        overVariance <- overVariance | (RandomeffectVariance( random_variances, random, overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:10){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_full_loglike, 
                y=y, P=P, X=X, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)
            
            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_full_loglike, 
                y=y, P=P, X=X, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT, method = "BFGS", hessian = TRUE)
            
            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)] <- out$par[1:ngam]
    V<- V + t(V)
    if ( !is.null( random ) ) {
        r2 <- out$par[(ngam+1):length(out$par)]
    } else {
        r2 <- NULL
    }
    hom2 <- 0
    l <- out$value * (-1)
    
    vars <- fixed_random_vars(y, P, X, C, vs, hom2, V, fixed, random, r2)
    beta <- vars$beta
    fixedeffect_vars <- vars$fixedeffect_vars
    #randomV <- vars$randomV
    randomeffect_vars <- vars$randomeffect_vars

    # estimate hessian matrix
    hess = hessian(screml_full_loglike, x=out$par, y=y, P=P, X=X, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('Full', err))
    #})
    return ( list( V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixedeffect_vars,
                 randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence ) )
                 #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence ) )
}

screml_full_loglike <- function(par, y, P, X, C, ngam, vs, random_MMT){
    hom2 <- 0
    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]<- par[1:ngam]
    V <- V + t(V)
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ngam+1):length(par)]
    }

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
