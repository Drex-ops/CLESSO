## VAS

### check the variation between coeffieicents

grp <- "VAS"



## load data
setwd("\\\\lw-osm-02-cdc\\OSM_CBR_LW_BACKCAST_work\\DEV\\hos06j\\")

files <- list.files(pattern=grp)
files <- files[grep("coefficients",files)]

samp_size <- strsplit(files,split="_")
samp_size <- unlist(lapply(1:length(samp_size),function(x){samp_size[[x]][2]}))
samp_size <- as.numeric(gsub("mil","",samp_size))

odr <- order(samp_size)

load(files[odr[1]])

coef_df <- data.frame(coefs)

for(i in odr[-1]){
	load(files[i])
	coef_df <- cbind(coef_df,coefs)
}

colnames(coef_df) <- paste("SampSize",samp_size[odr],"X",sep="_")
#colnames(coef_df) <- files[odr]
coef_df <- t(coef_df)

## calc eucledian distances
dist_mat <- dist(coef_df)

unq_smp <- unique(samp_size)
unq_smp <- unq_smp[order(unq_smp)]

mns <- c()
vrs <- c()

dm_resp <- c()
dm_pred <- c()

for(u in unq_smp){

	if(u != 1){
		coef_sm <- coef_df[grep(u,rownames(coef_df)),]
	}
	
	if(u ==1){
		coef_sm <- coef_df[grep("SampSize_1_X",rownames(coef_df)),]
		
	}
	
	dm <- dist(coef_sm)
	mn <- mean(dm)
	vr <- var(dm)
	
	mns <- c(mns,mn)
	vrs <- c(vrs,vr)
	
	## for loess
	dm_resp <- c(dm_resp,dm)
	dm_pred <- c(dm_pred,rep(u,length(dm)))

}

l1 <- loess(dm_resp ~ dm_pred)


## plots
par(mfcol=c(1,1))
scatter.smooth(dm_resp ~ dm_pred,col="grey",pch=16,cex=0.5,
				lpars = list(col = "black", lwd = 3, lty = 1),
				xlab="Number of millions of observation pairs sampled",
				ylab="Between model difference in fit")
abline(v=0.65,lty=3,col="red")
