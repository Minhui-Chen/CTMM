make_X <- function(P, fixed){
    X <- P

    if ( !is.null( fixed ) ) {
        for ( covar in fixed ) {
            X <- cbind(X, covar)
        }
    }

    return(X)
}

make_MMT <- function( random ){
    random_MMT <- NULL
    if ( !is.null( random ) ) {
        random_MMT <- list()
        for (i in 1:length(random)) {
            random_MMT[[i]] <- random[[i]] %*% t(random[[i]])
        }
    }
    return( random_MMT )
}

check_optim <- function(out, hom2, ct_overall_var, fixed_var, random_var, cut){
    if (out$convergence != 0 | out$value >1e10 | 
        any( c(hom2,ct_overall_var,unlist(fixed_var),unlist(random_var)) > cut ) ) {
        return(TRUE)
    } else {
        return(FALSE)
    }
}

re_optim <- function(out, fun, par, args, method, nrep, hessian){
    for (i in 1:nrep){
        par_ <- par * rgamma(length(par), 2, scale=1/2)
        out_ <- optim( par=par_, fn=fun, args=args, method = method, hessian = hessian)
        if (out$convergence != 0 & out_$convergence == 0) {
            out <- out_
        } else if (out$convergence == out_$convergence & out$value > out_$value) {
            out <- out_
        }
    }
    return( out )
}

FixedeffectVariance_ <- function( beta, x ) {
    xd <- scale(x, scale=F)
    s <- ( t(xd) %*% xd ) / nrow(x)
    variance <- beta %*% s %*% beta
    return ( variance )
}

FixedeffectVariance <- function( beta, xs ) {
    if ( length( xs ) == 0 ) return( NULL )

    j <- 0
    vars <- list()
    for ( i in 1:length(xs) ) {
        x <- xs[[i]]
        vars[[i]] <- FixedeffectVariance_( beta[j+1:ncol(x)], x)
        j <- j + ncol(x)
    }
    return ( vars )
}

RandomeffectVariance_ <- function( V, X ) {
    if ( !is.matrix(V) ) V <- as.matrix( V )
    if ( !is.matrix(X) ) X <- as.matrix( X )
    sum( diag( V %*% (t(X) %*% X) ) ) / nrow(X)
}

RandomeffectVariance <- function( Vs, Xs ) {
    if (length( Xs ) == 0) return( list(NULL, NULL) )

    if ( !is.list( Vs ) ) {
        Vs_ <- list()
        for (i in 1:length(Vs)) {
            Vs_[[i]] <- Vs[i] * diag( ncol(Xs[[i]]) )
        }
        Vs <- Vs_
    }

    vars <- list()
    for ( i in 1:length(Xs) ) {
        vars[[i]] <- RandomeffectVariance_( Vs[[i]], Xs[[i]] )
    }
    return( list(Vs, vars) )
}

