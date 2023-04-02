install.packages( setdiff(c('softImpute'), rownames(installed.packages())) )
library(softImpute)

my_softImpute <- function( Y, scale=F, biscale=F, lambda.len=100, maxrank=TRUE, 
                          fixed.maxrank=min(dim(Y))-1, nfolds=10, verbose=T, return_out=TRUE, seed=NULL ){

    options(warn=1)
	print( 'Start softImpute' )

    if ( !is.null( seed ) ) set.seed( seed )
    # Y has to be matrix not data.frame

    # scale: whether to scale (by column) before impute
    Y_original <- Y
    if ( scale ) {
        print('Scaling')
        #print(Y[1:10,1:min(ncol(Y),10)])
        Y <- scale(Y)
        #print(Y[1:10,1:min(ncol(Y),10)])
        print('Scaling')
    } else if ( biscale ) {
        print('biScaling')
        #print(Y[1:10,1:min(ncol(Y),10)])
        Y <- biScale(Y) # biScale not working well in midway, don't know why
        #print(Y[1:10,1:min(ncol(Y),10)])
    }


	obs	  <- which( ! is.na( Y ) )
	n_obs	  <- length(obs)
	folds	  <- split( obs, sample( rep( 1:nfolds, n_obs )[1:n_obs], n_obs ) )

	cv.loss <- matrix( NA, lambda.len, nfolds )

	lam0    <- lambda0(Y)
	lamseq  <- exp(seq(from=log(lam0+.2),to=log(.001),length=lambda.len)) 
	##### from Hastie blog: http://web.stanford.edu/~hastie/swData/softImpute/vignette.html
	#### lambda.len	<- 10
	#### lamseq			<- exp(seq(from=log(lam0+.2),to=log(1),length=lambda.len))

	for( f in 1:nfolds ){
		if( verbose ) print( f )

		Y.train	<- Y
		Y.train[ folds[[f]] ]	<- NA

		ranks     <- as.integer( lamseq )
		rank.max  <- ifelse( maxrank, 2, fixed.maxrank )
		#out      <- NULL
		warm      <- NULL
		for( i in seq(along=lamseq)){
			if( verbose ) cat( i, ' ' )
			#out	    <- softImpute( x=Y.train, lambda=lamseq[i], rank=rank.max, warm=out, maxit=1000 )
			out	    <- softImpute( x=Y.train, lambda=lamseq[i], rank=rank.max, warm=warm, maxit=1000 )

			#Yhat	    <- complete( Y.train, out ) 
			#cv.loss[i,f]  <- mean( (Yhat - Y)^2, na.rm=TRUE )
			#rm( Yhat )

			cv.loss[i,f]  <- mean( (complete( Y.train, out ) - Y)^2, na.rm=TRUE )

			warm      <- out
			if( maxrank ){
				ranks[i]  <- sum(round(out$d,4)>0)
				rank.max  <- min( ranks[i]+2, fixed.maxrank ) ### grows by at most 2, bounded by P/2
			}
			rm( out ); gc()
		}
		if( verbose ) cat( '\n' )
		rm( Y.train ); gc()
		print( gc() )
	}
	print( 'Done with CV' )
	loss		<- rowMeans( cv.loss )
	lambda	<- lamseq[ which.min(loss) ]

	out			<- softImpute( x=Y, rank.max = fixed.maxrank, lambda = lambda, maxit=1000 )
    #print( round(out$d, 4) )
	Yhat		<- complete( Y_original, out ) 
    #print( Yhat[1:10,1:min(ncol(Yhat),10)] )
	print( 'Done with impute' )

	if( !return_out )
		out	<- NA

    #print(out)
    #print(str(out))	
	#Y_overwrite <- out$u %*% diag(out$d) %*% t(out$v)

	return(list( 
		lambda  = lambda,
		lamseq	= lamseq,
		out			= out,
		loss    = loss, 
		Y       = Yhat
		#Y_overwrite=Y_overwrite
	))
}

#####
# runs only when script is run by itself
if (sys.nframe() == 0){
    args <- commandArgs(trailingOnly=TRUE)
    raw_f <- args[1]
    imputed_f <- args[2]
    scale <- F
    if (length(args) == 3) {
        if (args[3] == 'scale') scale <- TRUE
    } 
    #overwrite <- FALSE
    #if (length(args) == 3) {
    #    if (args[3] == 'overwrite') overwrite <- TRUE
    #} 


    raw <- read.csv(raw_f, header=TRUE, row.names=1, sep='\t')
    imputed <- my_softImpute( as.matrix(raw), scale=scale )
    #if ( overwrite ) {
    #    write.table(imputed$Y_overwrite, imputed_f, sep='\t', quote=FALSE)
    #} else {
        #print(imputed$Y)
    imputed <- imputed$Y
    rownames(imputed) <- rownames(raw)
    colnames(imputed) <- colnames(raw)
    write.table(imputed, imputed_f, sep='\t', quote=FALSE)
    #}
}
