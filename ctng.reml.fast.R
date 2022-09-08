library(numDeriv)
library(Matrix)
source('bin/reml.R')

LL <- function(Y, vs, N, C, hom2, V, fixed=NULL, random_variances=NULL, random_MMT=NULL){
    A <- matrix(rep(1, C*C), nrow=C) * hom2 + V

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
            sig2s <- sig2s + random_variances[i] * random_MMT[[i]]
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

screml_hom <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

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
            random[[i]] <- kronecker( random[[i]], rep(1,C) )
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

    result <- tryCatch({
        out <- optim( par=par, fn=screml_hom_loglike, 
            Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
        list( out=out, method=method)
    }, error=function(err) {
        message(err)
        method <- "Nelder-Mead"
        out <- optim( par=par, fn=screml_hom_loglike, 
            Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
        list( out=out, method=method)
    })
    out <- result$out
    method <- result$method

	hom2_ <- out$par[1]
    overVariance <- (hom2_ > overVariance_threshold)
    if ( !is.null( random ) ) {
        random_variances <- out$par[2:length(out$par)]
        overVariance <- overVariance | (RandomeffectVariance( random_variances, random, 
                                                             overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:nrep){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_hom_loglike, 
                Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)

            if ((out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par2_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par2_, fn=screml_hom_loglike, 
                Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)

            if ((out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

	hom2 <- out$par[1]
    l <- out$value * (-1)

    X <- kronecker(rep(1,N), diag(C))
    if ( !is.null(fixed) ) {
        for (M in fixed) { X <- cbind(X, kronecker(M,rep(1,C))) }
    }

    sig2s <- kronecker(diag(N), matrix(rep(1, C*C), nrow=C)*hom2) + diag( as.vector(t(vs)) )
    if ( !is.null(random_MMT) ) {
        r2 <- out$par[2:length(out$par)]
        randomeffect_vars <-RandomeffectVariance(r2, random)[[2]]
        #randomV <- RandomeffectVariance(r2, random)[[3]]

        for (i in 1:length(random_MMT)){
            sig2s <- sig2s + r2[i] * random_MMT[[i]]
        }
    } else {
        randomeffect_vars <- NULL
        r2 <- NULL
        #randomV <- NULL
    }
    sig2s_inv <- solve(sig2s)
    H <- t(X) %*% sig2s_inv
    B <- H %*% X
    beta <- as.vector(solve( B ) %*% H %*% y)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]

    # estimate hessian matrix
    hess = hessian(screml_hom_loglike, x=out$par, Y=Y, N=N, C=C, vs=vs, fixed=fixed, 
                   random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('Hom', err))
    #})
    return ( list(hom2=hom2, l=l, hess=hess, beta=beta, fixedeffect_vars=fixedeffect_vars,
                  randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence, 
                  method=method) )
                  #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence, 
}

screml_hom_loglike<- function(par, Y, N, C, vs, fixed, random_MMT){
	hom2 <- par[1]
    V <- matrix(rep(0, C*C), nrow=C)
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[2:length(par)]
    }

    ll <- LL(Y, vs, N, C, hom2, V, fixed, random_variances, random_MMT) 
    return( ll )
}

screml_iid <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

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
            random[[i]] <- kronecker( random[[i]], rep(1,C) )
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

    result <- tryCatch({
        out<- optim( par=par, fn=screml_iid_loglike,
            Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
        list( out=out, method=method )
    }, error=function(err) {
        message(err)
        method <- "Nelder-Mead"
        out<- optim( par=par, fn=screml_iid_loglike,
            Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
        list( out=out, method=method )
    })
    out <- result$out
    method <- result$method

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
        for (i in 1:nrep){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_iid_loglike, 
                Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)

            if ((out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par2_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par2_, fn=screml_iid_loglike, 
                Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)

            if ((out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }
    
	hom2 <- out$par[1]
	V <- diag(C) * out$par[2]
    l <- out$value * (-1)

    X <- kronecker(rep(1,N), diag(C))
    if ( !is.null(fixed) ) {
        for (M in fixed) { X <- cbind(X, kronecker(M,rep(1,C))) }
    }

    sig2s <- kronecker(diag(N), hom2 * matrix(rep(1, C*C), nrow=C) + V) + diag( as.vector(t(vs)) )
    if ( !is.null(random_MMT) ) {
        r2 <- out$par[3:length(out$par)]
        randomeffect_vars <-RandomeffectVariance(r2, random)[[2]]
        #randomV <- RandomeffectVariance(r2, random)[[3]]

        for (i in 1:length(random_MMT)){
            sig2s <- sig2s + random_variances[i] * random_MMT[[i]]
        }
    } else {
        randomeffect_vars <- NULL
        r2 <- NULL
        #randomV <- NULL
    }
    sig2s_inv <- solve(sig2s)
    F <- t(X) %*% sig2s_inv
    B <- F %*% X
    beta <- as.vector(solve( B ) %*% F %*% y)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]
    
    # estimate hessian
    hess = hessian(screml_iid_loglike, x=out$par, Y=Y, N=N, C=C, vs=vs, fixed=fixed,
                   random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('IID', err))
    #})
    #try(solve(out$hessian))
    return ( list(hom2=hom2, V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixedeffect_vars,
                  randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence,
                  method=method) )
                  #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence,
}

screml_iid_loglike <- function(par, Y, N, C, vs, fixed, random_MMT){
	hom2 <- par[1]
	V <- diag(C) * par[2]
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[3:length(par)]
    }
    L <- LL(Y, vs, N, C, hom2, V, fixed, random_variances, random_MMT)
    #print( L )

    return( L )
}

screml_free <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))

	hom2 <- var(y) / 2
	V <- rep(1,C) * as.vector(hom2)
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
            random[[i]] <- kronecker( random[[i]], rep(1,C) )
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }

    result <- tryCatch({
        out<- optim( par=par, fn=screml_free_loglike, 
            Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
        list( out=out, method=method )
    }, error=function(err) {
        message(err)
        method <- "Nelder-Mead"
        out<- optim( par=par, fn=screml_free_loglike, 
            Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
        list( out=out, method=method )
    })
    out <- result$out
    method <- result$method

	hom2_ <- out$par[1]
	V_ <- diag(out$par[1+1:C])
    overVariance <- (hom2_ > overVariance_threshold)
    overVariance <- overVariance | (RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]])
    if ( !is.null( random ) ) {
        random_variances <- out$par[(C+2):length(out$par)]
        overVariance <- overVariance | (RandomeffectVariance( random_variances, random, 
                                                             overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        print('rep')
        for (i in 1:nrep){
            print(i)
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_free_loglike, 
                Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)

            if ((out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par2_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par2_, fn=screml_free_loglike, 
                Y=Y, N=N, C=C, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)

            if ((out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

	hom2 <- out$par[1]
	V <- diag(out$par[1+1:C])
    l <- out$value * (-1)

    X <- kronecker(rep(1,N), diag(C))
    if ( !is.null(fixed) ) {
        for (M in fixed) { X <- cbind(X, kronecker(M,rep(1,C))) }
    }

    sig2s <- kronecker(diag(N), hom2 * matrix(rep(1, C*C), nrow=C) + V) + diag( as.vector(t(vs)) )
    if ( !is.null(random_MMT) ) {
        r2 <- out$par[(C+2):length(out$par)]
        randomeffect_vars <- RandomeffectVariance(r2, random)[[2]]
        #randomV <- RandomeffectVariance(r2, random)[[3]]

        for (i in 1:length(random_MMT)){
            sig2s <- sig2s + r2[i] * random_MMT[[i]]
        }
    } else {
        randomeffect_vars <- NULL
        r2 <- NULL
        #randomV <- NULL
    }
    sig2s_inv <- solve(sig2s)
    F_ <- t(X) %*% sig2s_inv
    B <- F_ %*% X
    beta <- as.vector(solve( B ) %*% F_ %*% y)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]

    # estimate hessian matrix
    hess = hessian(screml_free_loglike, x=out$par, Y=Y, N=N, C=C, vs=vs, fixed=fixed,
                   random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('Free', err))
    #})
    return ( list(hom2=hom2, V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixedeffect_vars,
                 randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence, method=method ))
                 #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence, method=method ))
}

screml_free_loglike<- function(par, Y, N, C, vs, fixed, random_MMT){
	hom2 <- par[1]
	V <- diag(par[1+1:C])
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(C+2):length(par)]
    }

    return( LL(Y, vs, N, C, hom2, V, fixed, random_variances, random_MMT) )
}

screml_full <- function(
Y, P, vs, fixed=NULL, random=NULL, overVariance_threshold=5, method="BFGS", par=NULL, nrep=10
){
    N <- nrow(Y)
    C <- ncol(Y)

    y <- as.vector(t(Y))
	ngam   <- C*(C+1)/2 # number of entries in gamma matrix # should it be C + C*(C+1)/2?

	V <- as.numeric( diag(C)[ lower.tri(diag(C),diag=T) ] ) * as.vector(var(y))
    par2 <- c(V)
    if ( !is.null( random ) ) {
        v1 <- par2[1] / (length(random)+1)
        par2 <- as.numeric( diag(C)[ lower.tri(diag(C),diag=T) ] ) * as.vector(v1)
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

    result <- tryCatch({
        out <- optim( par=par, fn=screml_full_loglike, 
            Y=Y, N=N, C=C, ngam=ngam, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
        list( out=out, method=method )
    }, error=function(err) {
        message(err)
        method <- "Nelder-Mead"
        out <- optim( par=par, fn=screml_full_loglike, 
            Y=Y, N=N, C=C, ngam=ngam, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
        list( out=out, method=method )
    })
    out <- result$out
    method <- result$method

    V_   <- matrix( 0, C, C )
    V_[lower.tri(V_,diag=T)]  <- out$par[1:ngam]
    V_ <- V_ + t(V_)
    overVariance <- RandomeffectVariance( list(V_), list(P), overVariance_threshold )[[1]]
    if ( !is.null( random ) ) {
        random_variances <- out$par[(ngam+1):length(out$par)]
        overVariance <- overVariance | (RandomeffectVariance( random_variances, random,
                                                             overVariance_threshold )[[1]])
    }

    if (out$convergence != 0 | overVariance | out$value > 1e10) {
        for (i in 1:nrep){
            par_ <- par * rgamma(length(par), 2, scale=1/2)
            out_ <- optim( par=par_, fn=screml_full_loglike, 
                Y=Y, N=N, C=C, ngam=ngam, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
            
            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }

            par2_ <- par2 * rgamma(length(par2), 2, scale=1/2)
            out_ <- optim( par=par2_, fn=screml_full_loglike, 
                Y=Y, N=N, C=C, ngam=ngam, vs=vs, fixed=fixed, random_MMT=random_MMT, method=method, hessian=FALSE)
            
            if ( (out_$value * (-1)) > (out$value * (-1)) ) {
                out <- out_
            }
        }
    }

    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)] <- out$par[1:ngam]
    V<- V + t(V)
    l <- out$value * (-1)

    X <- kronecker(rep(1,N), diag(C))
    if ( !is.null(fixed) ) {
        for (M in fixed) { X <- cbind(X, kronecker(M,rep(1,C))) }
    }

    sig2s <- kronecker(diag(N), V) + diag( as.vector(t(vs)) )
    if ( !is.null(random_MMT) ) {
        r2 <- out$par[(ngam+1):length(out$par)]
        randomeffect_vars <-RandomeffectVariance(r2, random)[[2]]
        #randomV <- RandomeffectVariance(r2, random)[[3]]

        for (i in 1:length(random_MMT)){
            sig2s <- sig2s + r2[i] * random_MMT[[i]]
        }
    } else {
        randomeffect_vars <- NULL
        r2 <- NULL
        #randomV <- NULL
    }
    sig2s_inv <- solve(sig2s)
    F <- t(X) %*% sig2s_inv
    B <- F %*% X
    beta <- as.vector(solve( B ) %*% F %*% y)
    fixedeffect_vars <- FixedeffectVariance( beta, c(list(P), fixed) )[[2]]

    # estimate hessian matrix
    hess = hessian(screml_full_loglike, x=out$par, Y=Y, N=N, C=C, ngam=ngam, vs=vs, fixed=fixed,
                   random_MMT=random_MMT)

    #tryCatch({
    #    solve(hess)
    #}, error=function(err) {
    #    print(paste('Full', err))
    #})
    return ( list( V=V, l=l, hess=hess, beta=beta, fixedeffect_vars=fixedeffect_vars,
                 randomeffect_vars=randomeffect_vars, r2=r2, convergence=out$convergence, method=method ))
                 #randomeffect_vars=randomeffect_vars, randomV=randomV, convergence=out$convergence, method=method ))
}

screml_full_loglike <- function(par, Y, N, C, ngam, vs, fixed, random_MMT){
    V <- matrix( 0, C, C )
    V[lower.tri(V,diag=T)]<- par[1:ngam]
    V <- V + t(V)
    hom2 <- 0
    random_variances <- NULL
    if ( !is.null( random_MMT ) ) {
        random_variances <- par[(ngam+1):length(par)]
    }

    return( LL(Y, vs, N, C, hom2, V, fixed, random_variances, random_MMT) )
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
}
