
## make obs vs predicted plot



obs.gdm.plot <- function(model,title,w,Is){

	par(mfcol=c(2,2))

	

	## binned observed vs predicted
	plot(fitted(model), model$data$Match, xlab = "Predicted Miss ratio",ylab = "Observed binned Miss ratio", type = "n",ylim=c(0,1),xlim=c(0,1),main=title)
	
	sqs <- seq(0.05,0.95,by=0.1)
	rat <-unlist(lapply(sqs,function(x){
		data <- model$data$Match[fitted(model) >= (x-0.05) & fitted(model) <= (x+0.05)]
		length(data[data==1]) / length(data)
	}))

	points(sqs, rat, pch = 20, cex = 1,col = rgb(0,0,1))
	overlayX <- overlayY <- seq(from = min(fitted(model)),to = max(fitted(model)), length = 200)
	lines(overlayX, overlayY, lwd = 1,lty="dashed")


	## observed vs predicted
	plot(model$linear.predictors, model$data$Match, xlab = "Predicted Ecological Distance",ylab = "Observed Binned Miss ratio", type = "n")
    	##points(model$ecological, model$observed, pch = 20, cex = 0.25,col = rgb(0,0,1))
   	overlayX <- seq(from = min(model$linear.predictors), to = max(model$linear.predictors),length = 200)
    	overlayY <- inv.logit(overlayX)
    
	
	ecoR <- range(model$linear.predictors)
	
	
	tt <- try(seq(ecoR[1],ecoR[2],by=0.1),silent=T)

	if(!inherits(tt, "try-error")){
		sqs <- seq(ecoR[1],ecoR[2],by=0.1)
	}

	if(inherits(tt, "try-error")){
		sqs <- seq(ecoR[1],10,by=0.1)
	}	
	
	rat <-unlist(lapply(sqs,function(x){
		data <- model$data$Match[model$linear.predictors >= (x-0.05) & model$linear.predictors <= (x+0.05)]
		length(data[data==1]) / length(data)
	}))

	points(sqs, rat, pch = 20, cex = 1,col = rgb(0,0,1))
	lines(overlayX, overlayY, lwd = 2,lty="dashed",col="green")

	## density of observed
	match <- density(x=model$linear.predictors[model$data$Match == 0])
	miss <- density(x=model$linear.predictors[model$data$Match == 1])
	mx <- max(c(match$y,miss$y))
	xrng <- range(model$linear.predictors)
	plot(1,1,type="n",xlim=xrng,ylim=c(0,mx),xlab="Predicted Ecological Distance",ylab="Density")
	lines(match,col="red")
	lines(miss,col="blue")
	legend("topright",legend=c("match","miss"),col=c("red","blue"),lty=1)


	# ## transform predictions.

		# ## calculate sorenson vs binned obs match ratio
		# brk <- seq(0.0,1,by=0.01)
		# raw <- cbind(data2,Sorenson=SORENSON,Richness.S1=RICHNESS[,1],Richness.S2=RICHNESS[,2])
		
		# ## temp until Richness values available
		# raw2 <- raw
		# raw2 <- raw[raw$Richness.S1 > 3 & raw$Richness.S2 > 3,]
		# binnedSor <- lapply(brk,function(x){length(raw2$Match[raw2$Match == 1 & raw2$Sorenson >= (x-0.05) & raw2$Sorenson <= (x+0.05)]) / length(raw2$Match[raw2$Sorenson >= (x-0.05) & raw2$Sorenson <= (x+0.05)])})
		# raw2 <- raw[TARGET,]
		# raw2 <- raw2[raw2$Richness.S1 > 3 & raw2$Richness.S2 > 3,]
		# binnedSor2 <- lapply(brk,function(x){length(raw2$Match[raw2$Match == 1 & raw2$Sorenson >= (x-0.05) & raw2$Sorenson <= (x+0.05)]) / length(raw2$Match[raw2$Sorenson >= (x-0.05) & raw2$Sorenson <= (x+0.05)])})
	
		# dat2 <- data.frame(y=brk,x=unlist(binnedSor),x2=unlist(binnedSor2))
		# dat2 <- dat2[dat2$x != 0,]
		# ##save(dat2,file=paste(rlm,"DissVsMatchRatio.RData",sep=""))

		# ## temp
		# if(!(all(is.na(dat2$x)) | all(is.na(dat2$y)))){
			# plot(dat2$y~dat2$x,ylab="Dissimilarity",xlab="Miss Ratio/Predicted miss ratio")
			# cols <- c("red","blue","green")
			# for(i in 1:3){
				# I <- Is[i]
				# p0 <- inv.logit(I)
				# modOut <- seq(p0,1,length.out=100)
				# tran <- ObsTrans(p0,w,modOut)
				# lines(modOut,tran$out,col=cols[i])
			# }
			# legend("bottomright",legend=c("St1","St2","St3"),col=cols,lty="solid")
		# }
}