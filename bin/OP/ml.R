# random effect V currently only support homogeneous variance i.e. \sigma * I
library(numDeriv)
library(mvtnorm)
source('bin/reml.R')

LL <- function(y, P, X, C, vs, beta, hom2, V, random_variances, random_MMT){
    sig2s <- hom2 + diag( P %*% V %*% t(P) ) + vs
    if ( !is.null( random_MMT ) ) {
        sig2s <- diag(sig2s)
        for (i in 1:length(random_MMT)) {
            sig2s <- sig2s + random_variances[i] * random_MMT[[i]]
        }
        tryCatch({
            eval <- eigen(sig2s, symmetric=TRUE)$values # somehow, in figure1b analysis, sometimes eigen got error: error code 1 from Lapack routine 'dsyevr'>, but it's not a problem in dmvnorm
            if( max(eval) / (min(eval)+1e-99) > 1e8 | min(eval)<0 ) return(1e12)
        }, error=function(err) {
            print((sig2s[1:10,1:10]))
            print(dim(sig2s))
            print(err)
        })
        dmvnorm(y, mean=X %*% beta, sigma = sig2s, log=TRUE) * (-1)
    } else {
        if( any( sig2s < 0 ) ) return(1e12)
        (sum(log( sig2s )) + sum( (y - X %*% beta)^2 / sig2s ))/2   
    }
}



screml_hom <- function(
y, P, vs, fixed=NULL, random=NULL, rep=0, method='BFGS', hessian=TRUE, overVariance_threshold=5, par=NULL
){

	C <- ncol(P)  # cell type number
    N <- nrow(P)

    X <- P

    if ( !is.null( fixed ) ) {
        for ( covar in fixed ) {
            X <- cbind(X, covar)
        }
    }

	beta <- solve( t(X) %*% X ) %*% ( t(X) %*% y ) 
	hom2 <- var(y - X %*% beta) 
    par2 <- c(beta, hom2)
    if ( !is.null( random ) ) {
        hom2 <- par2[length(par2)] / (length(random)+1)
        par2[length(par2)] <- hom2
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
		y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian=hessian)

    if ( rep > 0 ) {
        for (i in 1:rep) {
            par_ <- par * rgamma( length(par), 2, scale=1/2 )
            out_ <- optim( par=par_, fn=screml_hom_loglike,
                          y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian=hessian)
            if ( out_$value < out$value ) {
                out <- out_
            }
        }
    }

	beta_  <- out$par[1:ncol(X)]
	hom2_ <- out$par[ncol(X)+1]
    overVariance <- (hom2_ > overVariance_threshold) 
    overVariance <- overVariance | (FixedeffectVariance( beta_, c(list(P), fixed), overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        overVariance <- overVariance | (RandomeffectVariance( out$par[(2+ncol(X)):length(out$par)], random,
                                                           overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:10){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_hom_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian=hessian)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par2_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par2_, fn=screml_hom_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian=hessian)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

	beta  <- out$par[1:ncol(X)]
	hom2 <- out$par[ncol(X)+1]
    r2 <- out$par[(ncol(X)+2):length(out$par)]
    if ( !is.null( random ) ) {
        randomeffect_vars <- RandomeffectVariance(r2, random)[[2]]
        #randomV <- RandomeffectVariance(r2, random)[[3]]
    } else {
        randomeffect_vars <- NULL
        #randomV <- NULL
    }
    l <- out$value * (-1)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]

    # estimate hessian matrix
    hess = hessian(screml_hom_loglike, x=out$par, y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('Hom', err))
    #})
    #try(solve(out$hessian))

    return( list(hom2=hom2, beta=beta, l=l, hess=hess, fixedeffect_vars=fixedeffect_vars, 
                 randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence) )
                 #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence) )
}

screml_hom_loglike<- function(par, y, P, X, C, vs, random_MMT){

	beta <- par[1:ncol(X)] 
	hom2 <- par[ncol(X)+1]
    V <- matrix(rep(0,C*C), nrow=C)
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ncol(X)+2):length(par)]
    }

    return( LL(y, P, X, C, vs, beta, hom2, V, random_variances, random_MMT) )
}

screml_iid <- function(
y, P, vs, fixed=NULL, random=NULL, rep=0, method='BFGS', hessian=TRUE, overVariance_threshold=5, par=NULL
) {

	C   <- ncol(P)  # cell type number
	N   <- nrow(P)  

    X <- P
    if ( !is.null( fixed ) ) {
        for ( covar in fixed ) {
            X <- cbind( X, covar )
        }
    }

	beta   <- solve( t(X) %*% X ) %*% ( t(X) %*% y ) # cell type effect
	hom2 <- var(y - X %*% beta) / 2
	V <- hom2
    par2 <- c(beta, hom2, V)
    if ( !is.null( random ) ) {
        hom2 <- 2 * hom2 / (length(random)+2)
        par2[ncol(X)+1:2] <- hom2
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

	out<- optim( par=par, fn=screml_iid_loglike, 
		y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)

    if ( rep > 0 ) {
        for (i in 1:rep) {
            par_ <- par * rgamma( length(par), 2, scale=1/2 )
            out_ <- optim( par=par_, fn=screml_iid_loglike,
                          y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)
            if ( out_$value < out$value ) {
                out <- out_
            }
        }
    }

	beta_  <- out$par[1:ncol(X)]
	hom2_ <- out$par[ncol(X)+1]
	V_ <- diag(C) * out$par[ncol(X)+2]
    overVariance <- (hom2_ > overVariance_threshold)
    overVariance <- overVariance | (RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]])
    overVariance <- overVariance | (FixedeffectVariance( beta_, c(list(P), fixed), overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        overVariance <- overVariance | (RandomeffectVariance( out$par[(3+ncol(X)):length(out$par)], random,
                                                           overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:10){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_iid_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_iid_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

	beta  <- out$par[1:ncol(X)]
	hom2 <- out$par[ncol(X)+1]
	V <- diag(C) * out$par[ncol(X)+2]
    r2 <- out$par[(ncol(X)+3):length(out$par)]
    if ( !is.null( random ) ) {
        randomeffect_vars <- RandomeffectVariance(r2, random)[[2]]
        #randomV <- RandomeffectVariance(r2, random)[[3]]
    } else {
        randomeffect_vars <- NULL
        r2 <- NULL
        #randomV <- NULL
    }
    l <- out$value * (-1)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]
    
    # estimate hessian
    hess = hessian(screml_iid_loglike, x=out$par, y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('IID', err))
    #})
    #try(solve(out$hessian))
    return ( list(hom2=hom2, beta=beta, V=V, l=l, hess=hess, fixedeffect_vars=fixedeffect_vars, 
                  randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence) )
                  #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence) )
}

screml_iid_loglike<- function(par, y, P, X, C, vs, random_MMT){

	beta    <- par[1:ncol(X)] 
	hom2 <- par[ncol(X)+1]
	V <- diag(C) * par[ncol(X)+2]
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ncol(X)+3):length(par)]
    }

    return( LL(y, P, X, C, vs, beta, hom2, V, random_variances, random_MMT) )

}

screml_free <- function(
y, P, vs, fixed=NULL, random=NULL, rep=0, method='BFGS', hessian=TRUE, overVariance_threshold=5, par=NULL
){
    #print(rep)

	C   <- ncol(P)  # cell type number
	N   <- ncol(P)  # cell type number

    X <- P
    if ( !is.null( fixed ) ) {
        for ( covar in fixed ) {
            X <- cbind( X, covar )
        }
    }

	beta   <- solve( t(X) %*% X ) %*% ( t(X) %*% y ) # cell type effect
	hom2 <- var(y - X %*% beta) / 2
	V <- rep(1,C) * as.vector(hom2)
    par2 <- c(beta, hom2, V)
    if ( !is.null( random ) ) {
        hom2 <- 2 * hom2 / (length(random)+2)
        par2[ncol(X)+1:(C+1)] <- hom2
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
		y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)

    if ( rep > 0 ) {
        for (i in 1:rep) {
            print(i)
            par_ <- par * rgamma( length(par), 2, scale=1/2 )
            out_ <- optim( par=par_, fn=screml_free_loglike,
                          y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)
            if ( out_$value < out$value ) {
                out <- out_
            }
        }
    }

	beta_  <- out$par[1:ncol(X)]
	hom2_ <- out$par[ncol(X)+1]
	V_ <- diag(out$par[ncol(X)+1+1:C])
    overVariance <- (hom2_ > overVariance_threshold) 
    overVariance <- overVariance | (RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]])
    overVariance <- overVariance | (FixedeffectVariance( beta_, c(list(P), fixed),overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        overVariance <- overVariance | (RandomeffectVariance( out$par[(C+2+ncol(X)):length(out$par)], random,
                                                           overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:10){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_free_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_free_loglike, 
                y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)

            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

	beta  <- out$par[1:ncol(X)]
	hom2 <- out$par[ncol(X)+1]
	V <- diag(out$par[ncol(X)+1+1:C])
    r2 <- out$par[(C+2+ncol(X)):length(out$par)]
    if ( !is.null( random ) ) {
        randomeffect_vars <- RandomeffectVariance(r2, random)[[2]]
        #randomV <- RandomeffectVariance(r2, random)[[3]]
    } else {
        randomeffect_vars <- NULL
        #randomV <- NULL
    }
    l <- out$value * (-1)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]
    #print(l)
	#print(hom2)
	#print(beta)
    #print(V)

    # estimate hessian matrix
    hess = hessian(screml_free_loglike, x=out$par, y=y, P=P, X=X, C=C, vs=vs, random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('Free', err))
    #})
    return ( list(hom2=hom2, beta=beta, V=V, l=l, hess=out$hessian, hess2=hess, fixedeffect_vars=fixedeffect_vars, 
                  randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence) )
                  #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence) )
}

screml_free_loglike<- function(par, y, P, X, C, vs, random_MMT){

    #print(par)
	beta <- par[1:ncol(X)] 
	hom2 <- par[ncol(X)+1]
	V <- diag(par[ncol(X)+1+1:C])
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ncol(X)+C+2):length(par)]
    }
    #print(par)
    l <- LL(y, P, X, C, vs, beta, hom2, V, random_variances, random_MMT)
    #print(l)
    return( l )

}

screml_full <- function(
y, P, vs, fixed=NULL, random=NULL, rep=0, method='BFGS', hessian=TRUE, overVariance_threshold=5, par=NULL
){

	C      <- ncol(P)  # cell type number
	N      <- nrow(P)  
	ngam   <- C*(C+1)/2 # number of entries in gamma matrix # should it be C + C*(C+1)/2?

    X <- P
    if ( !is.null( fixed ) ) {
        for ( covar in fixed ) {
            X <- cbind( X, covar )
        }
    }

	beta   <- solve( t(X) %*% X ) %*% ( t(X) %*% y ) # cell type effect
    v1 <- c(var(y - X %*% beta))
	V <- diag(C)[ lower.tri(diag(C),diag=T) ] * v1
    par2 <- c(beta, V)
    if ( !is.null( random ) ) {
        v1 <- v1 / (length(random)+1)
        V <- diag(C)[ lower.tri(diag(C),diag=T) ] * v1
        par2 <- c(beta, V)
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
		y=y, P=P, X=X, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT, method=method, hessian=hessian)
    
    if ( rep > 0 ) {
        for (i in 1:rep) {
            par_ <- par * rgamma( length(par), 2, scale=1/2 )
            out_ <- optim( par=par_, fn=screml_full_loglike,
                          y=y, P=P, X=X, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT,  
                          method=method, hessian=hessian)
            if ( out_$value < out$value ) {
                out <- out_
            }
        }
    }

	beta_ <- out$par[1:ncol(X)]
    V_   <- matrix( 0, C, C )
    V_[lower.tri(V_,diag=T)]  <- out$par[ncol(X)+1:ngam]
    V_ <- V_ + t(V_)
    overVariance <- FixedeffectVariance( beta_, c(list(P), fixed), overVariance_threshold )[[1]]
    overVariance <- overVariance | (RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        overVariance <- overVariance | (RandomeffectVariance( out$par[(ngam+ncol(X)+1):length(out$par)], random,
                                                           overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:10){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_full_loglike, 
                y=y, X=X, P=P, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)
            
            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_full_loglike, 
                y=y, X=X, P=P, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT, method = method, hessian = hessian)
            
            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

	beta <- out$par[1:ncol(X)]
    V<- matrix( 0, C, C )
    V[lower.tri(V,diag=T)] <- out$par[ncol(X)+1:ngam]
    V<- V + t(V)
    r2 <- out$par[(ngam+ncol(X)+1):length(out$par)]
    if ( !is.null( random ) ) {
        randomeffect_vars <- RandomeffectVariance(r2, random)[[2]]
        #randomV <- RandomeffectVariance(r2, random)[[3]]
    } else {
        randomeffect_vars <- NULL
        #randomV <- NULL
    }
    l <- out$value * (-1)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]

    # estimate hessian matrix
    hess = hessian(screml_full_loglike, x=out$par, y=y, P=P, X=X, C=C, ngam=ngam, vs=vs, random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('Full', err))
    #})
    return ( list(beta=beta, V=V, l=l, hess=hess, fixedeffect_vars=fixedeffect_vars, 
                  randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence) )
                  #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence) )
}

screml_full_loglike <- function(par, y, P, X, C, ngam, vs, random_MMT){
	beta <- par[1:ncol(X)] 
    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]<- par[ncol(X)+1:ngam]
    V <- V + t(V)
    hom2 <- 0
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ncol(X)+ngam+1):length(par)]
    }

    return( LL(y, P, X, C, vs, beta, hom2, V, random_variances, random_MMT) )

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
