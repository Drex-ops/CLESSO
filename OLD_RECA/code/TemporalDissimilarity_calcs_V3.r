library(ffbase)
library(raster)
library(dynowindow)
library(feather)
library(Gdm01)
library(SDMTools)
library(raster)
library(nnls)
library(lubridate)

source("//ces-10-cdc/OSM_CDC_MMRG_work/users/GPAAG_refuge/BIOL/G/DEV/_tools/ObsPairSampler/site-richness-extractor-bigData.R")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/obsPairSampler-bigData-RECA.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/siteAggregator.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/dynowindow.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/getSamples_AND_ENV.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/obs.gdm.plot_logit.R")
source("Y:\\MOD\\G\\GDM\\_tools\\create_5splineOutput2.R")
source("Y:\\MOD\\G\\GDM\\_tools\\gdm.spline.plot.R")
source("U:\\users\\hos06j\\R scripts\\nnnpls.fit2_(good).R")
source("U:\\users\\hos06j\\R scripts\\negexp_GDMlink.R")
source("U:\\users\\hos06j\\R scripts\\fitGDM.R")

setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/")

RsqGLM <- function(obs = NULL, pred = NULL, model = NULL) {
  # version 1.2 (3 Jan 2015)
 
  model.provided <- ifelse(is.null(model), FALSE, TRUE)
 
  if (model.provided) {
    if (!("glm" %in% class(model))) stop ("'model' must be of class 'glm'.")
    if (!is.null(pred)) message("Argument 'pred' ignored in favour of 'model'.")
    if (!is.null(obs)) message("Argument 'obs' ignored in favour of 'model'.")
    obs <- model$y
    pred <- model$fitted.values
 
  } else { # if model not provided
    if (is.null(obs) | is.null(pred)) stop ("You must provide either 'obs' and 'pred', or a 'model' object of class 'glm'")
    if (length(obs) != length(pred)) stop ("'obs' and 'pred' must be of the same length (and in the same order).")
    #if (!(obs %in% c(0, 1)) | pred < 0 | pred > 1) stop ("Sorry, 'obs' and 'pred' options currently only implemented for binomial GLMs (binary response variable with values 0 or 1) with logit link.")
    logit <- log(pred / (1 - pred))
    model <- glm(obs ~ logit, family = "binomial")
  }
 
  null.mod <- glm(obs ~ 1, family = family(model))
  loglike.M <- as.numeric(logLik(model))
  loglike.0 <- as.numeric(logLik(null.mod))
  N <- length(obs)
 
  # based on Nagelkerke 1991:
  CoxSnell <- 1 - exp(-(2 / N) * (loglike.M - loglike.0))
  Nagelkerke <- CoxSnell / (1 - exp((2 * N ^ (-1)) * loglike.0))
 
  # based on Allison 2014:
  McFadden <- 1 - (loglike.M / loglike.0)
  Tjur <- mean(pred[obs == 1]) - mean(pred[obs == 0])
  sqPearson <- cor(obs, pred) ^ 2
 
  return(list(CoxSnell = CoxSnell, Nagelkerke = Nagelkerke, McFadden = McFadden, Tjur = Tjur, sqPearson = sqPearson))
}


## inverse link 
inv.logit <- function(x){exp(x)/(1+exp(x))}

## transform obs to diss
ObsTrans <- function(p0,w,p){

	prw <- (p*w) / ((1-p) + (p*w))

	p0w <- (p0*w) / ((1-p0) + (p0*w))

	out <- 1 - ((1-prw) / (1-p0w))
	return(list(prw=prw,out=out))
}

## fit obs GDM
fitGDM <- function(formula=NULL,data=NULL){

	fit <- glm(formula,family=binomial(),data=data,control=list(maxit=500),method='nnls.fit')

	return(fit)
}

rasterOptions(tmpdir="S:\\rasterTemp")



#### plot temporal community change from spatio-temporal 



## calculate transformation parametres
if(FALSE){

## not run needs thinking....
	c_yr <- 51
	setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j")
	load("AVES_aggregated_basicFilt.RData")
	## reduce to dates where we have env data
	datRED <- datRED[datRED$eventDate != "",]
	date_test <- as.Date(datRED$eventDate) > (as.Date("1911-01-01") %m+% years(c_yr))
	datRED <- datRED[date_test,]
	date_test <- as.Date(datRED$eventDate) < as.Date("2018-01-01")
	datRED <- datRED[date_test,]
	datRED <- droplevels(datRED)
	

	datRED$year <- substr(as.character(datRED$eventDate),1,4)
			
	nMatch <- 2000000
	nSamples <- nMatch*4
	idx <- 1:nrow(datRED)

	ws <- c()

	for(i in 1:100){
	s1 <- sample(idx,nSamples,replace=T)
	s2 <- sample(idx,nSamples,replace=T)
	
		## make site ID

	siteID_1 <- datRED$ID[s1]
	siteID_2 <- datRED$ID[s2]

	same_site <- siteID_1 == siteID_2
	same_time <- datRED$year[s1] == datRED$year[s2]
	same_time[same_site] <- FALSE
	
	miss <- datRED$gen_spec[s1] != datRED$gen_spec[s2]
	miss <- miss[same_site | same_time]

	missCount <- sum(miss)
	matchCount <- length(miss) - missCount
	propMiss <- missCount / length(miss)
	propMatch <- matchCount / length(miss)
	w <- propMiss/propMatch
	ws <- c(ws,w)
	}
	w <- quantile(ws,0.5)
}

### get samples
nSamples <- 27400
c_yr <- 50
w_yr <- 1
w <- 41.2789 ## from above

## get grid parametres from env grid\\ces-10-cdc\OSM_CDC_GISDATA_work\_DEV\her134\SUBS\out\DES_mean.flt
ras <- "//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/AWAP_monthly/RECA/monthly/years1901_2017/Centered1960/FWPT_mean_Cmean_mean_1946_1975.flt" ##"//ces-10-cdc/OSM_CDC_GISDATA_work/_DEV/her134/SUBS/out/DES_mean.flt"
ras_sp <- raster(ras)
vals <- values(ras_sp)
crds <- coordinates(ras_sp)
crds <- crds[!is.na(vals),]

smp_idx <- 1:nrow(crds)
idx <- sample(smp_idx,nSamples,replace=FALSE)

ext_points <- crds[idx,]


## get model and quantiles
load("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/AVES_1mil_51climYr_1weathYr_ObsEnvTable.RData")
load("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/V3_AVES_1mil_51climYr_1weathYr__coefficients.RData")

X  <- obsPairs_out[,c(23:ncol(obsPairs_out))]
X <- X[,-c(10:18,47:55)]
	## fold X and create site vector
	nc <- ncol(X)
	nc2 <- nc/2
	if(nc %% 2 != 0){stop("X must be a matrix with even columns")}
	X1 <- X[,1:nc2]
	X2 <- X[,(nc2+1):nc]
	nms <- colnames(X1)
	colnames(X2) <- nms
	## site vector
	sv <- c(rep(1,nrow(X1)),rep(2,nrow(X2)))
	XX <- rbind(X1,X2)
	XX <- cbind(XX[,1:9],XX[,18:28],XX[,1:9],XX[,18:28],XX[,10:17])
	
splines <- 	rep(3,ncol(XX))
quantiles <- unlist(lapply(1:ncol(XX),function(x){quantile(XX[,x],c(0,0.5,1))}))


## get env data
require(raster)
require(feather)
require(doSNOW)
require(foreach)
require(parallel)

	init_params <- list(
		list(variables = c("mean_PT_191101-201712"),
			mstat = 'mean', 
			cstat = 'mean', 
			window = c_yr,
			prefix=paste("XbrXbr",c_yr,sep="_")),
		list(variables = c("TNn_191101-201712", "FWPT_191101-201712"),
			mstat = 'mean', 
			cstat = 'min', 
			window = c_yr,
			prefix=paste("MinXbr",c_yr,sep="_")),	
		list(variables = c("max_PT_191101-201712", "FWPT_191101-201712"),
			mstat = 'mean', 
			cstat = 'max', 
			window = c_yr,
			prefix=paste("MaxXbr",c_yr,sep="_")),		
		list(variables = c("FD_191101-201712", "TXx_191101-201712"),
			mstat = 'mean', 
			cstat = 'max', 
			window = c_yr,
			prefix=paste("MaxXbr",c_yr,sep="_")),		
		list(variables = c("TNn_191101-201712", "PD_191101-201712"),
			mstat = 'mean', 
			cstat = 'max', 
			window = c_yr,
			prefix=paste("MaxXbr",c_yr,sep="_")),
			
		list(variables = c("mean_PT_191101-201712"),
			mstat = 'mean', 
			cstat = 'mean', 
			window = w_yr,
			prefix=paste("XbrXbr",w_yr,sep="_")),
		list(variables = c("TNn_191101-201712", "FWPT_191101-201712"),
			mstat = 'mean', 
			cstat = 'min', 
			window = w_yr,
			prefix=paste("MinXbr",w_yr,sep="_")),	
		list(variables = c("max_PT_191101-201712", "FWPT_191101-201712"),
			mstat = 'mean', 
			cstat = 'max', 
			window = w_yr,
			prefix=paste("MaxXbr",w_yr,sep="_")),		
		list(variables = c("FD_191101-201712", "TXx_191101-201712"),
			mstat = 'mean', 
			cstat = 'max', 
			window = w_yr,
			prefix=paste("MaxXbr",w_yr,sep="_")),		
		list(variables = c("TNn_191101-201712", "PD_191101-201712"),
			mstat = 'mean', 
			cstat = 'max', 
			window = w_yr,
			prefix=paste("MaxXbr",w_yr,sep="_"))
	)

	yrs <- 1966:2018
dissimilarities <- matrix(ncol=length(yrs),nrow=nSamples)
	## cumulative dissimilarity
for(i in 1:length(yrs)){

	yr <- yrs[i];print(yr)
	obs_tab <- data.frame(ext_points,year1=yr-1,month1=12,ext_points,year2=yr,month2=12)
	
	cl <- makeCluster(length(init_params))
	registerDoSNOW(cl)	
		env_out <- foreach(x=1:length(init_params),.combine='cbind',.packages="feather") %dopar% {
		out <- gen_windows(pairs=obs_tab,
		variables = init_params[[x]]$variables,
		mstat =  init_params[[x]]$mstat, 
		cstat =  init_params[[x]]$cstat, 
		window =  init_params[[x]]$window,
		)
		colnames(out) <- paste(init_params[[x]]$prefix,colnames(out),sep="_")
		out[,9:ncol(out)]
	}
	stopCluster(cl)
	registerDoSEQ()
	
	
	## reshuffle env_out
	env_out1 <- env_out[,grep("201712_1",names(env_out))]
	env_out2 <- env_out[,grep("201712_2",names(env_out))]
	
	## make anomalies
	env_anom_out1 <- env_out1[,grep(paste("_",w_yr,"_",sep=""),colnames(env_out1))] - env_out1[,grep(paste("_",c_yr,"_",sep=""),colnames(env_out1))]
	colnames(env_anom_out1) <- gsub("191101-201712","anom",colnames(env_anom_out1))
	env_anom_out2 <- env_out2[,grep(paste("_",w_yr,"_",sep=""),colnames(env_out2))] - env_out2[,grep(paste("_",c_yr,"_",sep=""),colnames(env_out2))]
	colnames(env_anom_out2) <- gsub("191101-201712","anom",colnames(env_anom_out2))
	
	pnt1 <- SpatialPoints(data.frame(obs_tab$x,obs_tab$y))
	pnt2 <- SpatialPoints(data.frame(obs_tab$x.1,obs_tab$y.1))
	
	subs_brk <- brick("SUBS_brk_AVES.grd")
	env1_subs <- extract(subs_brk,pnt1)
	env2_subs <- extract(subs_brk,pnt2)
	
	nms <- paste(colnames(env1_subs),"_1",sep="")
	colnames(env1_subs) <- nms
	nms <- paste(colnames(env2_subs),"_2",sep="")
	colnames(env2_subs) <- nms
	

	
	Trng_1 <- abs(env_out1[,c(grep(paste("MinXbr_",c_yr,"_TNn_191101-201712",sep=""),colnames(env_out1)),grep(paste("MinXbr_",w_yr,"_TNn_191101-201712",sep=""),colnames(env_out1)))] - env_out1[,c(grep(paste("MaxXbr_",c_yr,"_TXx_191101-201712",sep=""),colnames(env_out1)),grep(paste("MaxXbr_",w_yr,"_TXx_191101-201712",sep=""),colnames(env_out1)))])
	names(Trng_1) <- c("Trng_15_191101-201712_1","Trng_1_191101-201712_1")
	Trng_2 <- abs(env_out2[,c(grep(paste("MinXbr_",c_yr,"_TNn_191101-201712",sep=""),colnames(env_out2)),grep(paste("MinXbr_",w_yr,"_TNn_191101-201712",sep=""),colnames(env_out2)))] - env_out2[,c(grep(paste("MaxXbr_",c_yr,"_TXx_191101-201712",sep=""),colnames(env_out2)),grep(paste("MaxXbr_",w_yr,"_TXx_191101-201712",sep=""),colnames(env_out2)))])
	names(Trng_2) <- c("Trng_15_191101-201712_2","Trng_1_191101-201712_2")
	
	
	toSpline <- cbind(
						env_out1[,1:9],Trng_1,env_anom_out1,env_out1[,1:9],Trng_1,env_anom_out1,env1_subs,
						env_out2[,1:9],Trng_2,env_anom_out2,env_out2[,1:9],Trng_2,env_anom_out2,env2_subs)

	splined <- splineData(toSpline,splines=splines,quantiles=quantiles)
	splined[,61:ncol(splined)] <- 0

	
	## adjust splined data
	
	splined <- cbind(1,splined)
	
	
	
	pred <- rowSums(t(coefs * t(splined)))
	pred <- inv.logit(pred)
	p0 <- inv.logit(coefs[1])
	dissimilarity <- ObsTrans(p0,w,pred)$out
	dissimilarities[,i] <- dissimilarity
}


dissimilarities_cum <- cbind(0,dissimilarities)



for(c in 2:ncol(dissimilarities_cum)){
	dissimilarities_cum[,c] <- dissimilarities_cum[,c] + dissimilarities_cum[,c-1]
}

save(dissimilarities_cum,file=paste("V3_",nSamples,"dissimilarities_cum.RData",sep=""))
save(dissimilarities,file=paste("V3_",nSamples,"dissimilarities_yrONyr.RData",sep=""))

mns <- c()
q25 <- c()
q97 <- c()

#dissimilarities_epoch2 <- dissimilarities_epoch[,-ncol(dissimilarities_epoch)]
dissimilarities_cum2 <- dissimilarities_cum[,-ncol(dissimilarities_cum)]

for(c in 1:ncol(dissimilarities_cum2)){

	qs <- quantile(dissimilarities_cum2[,c],c(0.025,0.5,0.975))
	mns <- c(mns,qs[2])
	q25 <- c(q25,qs[1])
	q97 <- c(q97,qs[3])

}

png("Dissimilarities_cumulative.png")
plot(x=1965:2017,y=mns,type="l",ylim=c(0,8),xlab=NA,ylab="Cumulative bird community dissimilarity")
lines(x=1965:2017,y=q25,lty=3)
lines(x=1965:2017,y=q97,lty=3)
dev.off()

mns <- c()
q25 <- c()
q97 <- c()

for(c in 1:ncol(dissimilarities[,-ncol(dissimilarities)])){

	qs <- quantile(dissimilarities[,c],c(0.025,0.5,0.975))
	mns <- c(mns,qs[2])
	q25 <- c(q25,qs[1])
	q97 <- c(q97,qs[3])

}

png("V3_Dissimilarities_yrOnyr.png")
plot(x=1966:2017,y=mns,type="l",ylim=c(0,0.3),xlab=NA,ylab="Year on Year bird community dissimilarity")
lines(x=1966:2017,y=q25,lty=3)
lines(x=1966:2017,y=q97,lty=3)
abline(lm(mns[1:length(1966:1990)] ~ c(1966:1990)),col="blue",lty=2)
abline(lm(mns[26:length(1966:2017)] ~ c(1991:2017)),col="red",lty=2)
dev.off()

png("V3_Dissimilarities_slope.png")
plot(x=0:52,y=c(0,mns),type="n",ylim=c(0,10),xlab=NA,ylab="Slope of yearly change bird community dissimilarity")
cols <- two.colors(n=55, start="darkgreen", end="red", middle="white",alpha=1.0)
cols <- cols[-c(1:3)]

cols <- c( rep("blue",length(1991:2017)),rep("red",length(1991:2017)))
for(i in 1:length(mns)){abline(a=0,b=mns[i],col=cols[i])}
legend("topleft",legend=c("Pre 1990","Post 1990"),fill=unique(cols))
dev.off()


	yrs <- 1966:2018
dissimilarities_epoch <- matrix(ncol=length(yrs),nrow=nSamples)
	## change from epoch dissimilarity
for(i in 1:length(yrs)){

	yr <- yrs[i];print(yr)
	obs_tab <- data.frame(ext_points,year1=1965,month1=12,ext_points,year2=yr,month2=12)
	
	cl <- makeCluster(length(init_params))
	registerDoSNOW(cl)	
		env_out <- foreach(x=1:length(init_params),.combine='cbind',.packages="feather") %dopar% {
		out <- gen_windows(pairs=obs_tab,
		variables = init_params[[x]]$variables,
		mstat =  init_params[[x]]$mstat, 
		cstat =  init_params[[x]]$cstat, 
		window =  init_params[[x]]$window,
		)
		colnames(out) <- paste(init_params[[x]]$prefix,colnames(out),sep="_")
		out[,9:ncol(out)]
	}
	stopCluster(cl)
	registerDoSEQ()
	
	
	## reshuffle env_out
	env_out1 <- env_out[,grep("201712_1",names(env_out))]
	env_out2 <- env_out[,grep("201712_2",names(env_out))]
	
	## make anomalies
	env_anom_out1 <- env_out1[,grep(paste("_",w_yr,"_",sep=""),colnames(env_out1))] - env_out1[,grep(paste("_",c_yr,"_",sep=""),colnames(env_out1))]
	colnames(env_anom_out1) <- gsub("191101-201712","anom",colnames(env_anom_out1))
	env_anom_out2 <- env_out2[,grep(paste("_",w_yr,"_",sep=""),colnames(env_out2))] - env_out2[,grep(paste("_",c_yr,"_",sep=""),colnames(env_out2))]
	colnames(env_anom_out2) <- gsub("191101-201712","anom",colnames(env_anom_out2))
	
	pnt1 <- SpatialPoints(data.frame(obs_tab$x,obs_tab$y))
	pnt2 <- SpatialPoints(data.frame(obs_tab$x.1,obs_tab$y.1))
	
	subs_brk <- brick("SUBS_brk_AVES.grd")
	env1_subs <- extract(subs_brk,pnt1)
	env2_subs <- extract(subs_brk,pnt2)
	
	nms <- paste(colnames(env1_subs),"_1",sep="")
	colnames(env1_subs) <- nms
	nms <- paste(colnames(env2_subs),"_2",sep="")
	colnames(env2_subs) <- nms
	

	
	Trng_1 <- abs(env_out1[,c(grep(paste("MinXbr_",c_yr,"_TNn_191101-201712",sep=""),colnames(env_out1)),grep(paste("MinXbr_",w_yr,"_TNn_191101-201712",sep=""),colnames(env_out1)))] - env_out1[,c(grep(paste("MaxXbr_",c_yr,"_TXx_191101-201712",sep=""),colnames(env_out1)),grep(paste("MaxXbr_",w_yr,"_TXx_191101-201712",sep=""),colnames(env_out1)))])
	names(Trng_1) <- c("Trng_15_191101-201712_1","Trng_1_191101-201712_1")
	Trng_2 <- abs(env_out2[,c(grep(paste("MinXbr_",c_yr,"_TNn_191101-201712",sep=""),colnames(env_out2)),grep(paste("MinXbr_",w_yr,"_TNn_191101-201712",sep=""),colnames(env_out2)))] - env_out2[,c(grep(paste("MaxXbr_",c_yr,"_TXx_191101-201712",sep=""),colnames(env_out2)),grep(paste("MaxXbr_",w_yr,"_TXx_191101-201712",sep=""),colnames(env_out2)))])
	names(Trng_2) <- c("Trng_15_191101-201712_2","Trng_1_191101-201712_2")
	
	toSpline <- cbind(
						env_out1[,1:9],Trng_1,env_anom_out1,env_out1[,1:9],Trng_1,env_anom_out1,env1_subs,
						env_out2[,1:9],Trng_2,env_anom_out2,env_out2[,1:9],Trng_2,env_anom_out2,env2_subs)

	splined <- splineData(toSpline,splines=splines,quantiles=quantiles)
	splined[,61:ncol(splined)] <- 0
	
	splined <- cbind(1,splined)
	pred <- rowSums(t(coefs * t(splined)))
	pred <- inv.logit(pred)
	p0 <- inv.logit(coefs[1])
	dissimilarity <- ObsTrans(p0,w,pred)$out
	dissimilarities_epoch[,i] <- dissimilarity
}

save(dissimilarities_epoch,file=paste("V3_",nSamples,"dissimilarities_epoch.RData",sep=""))

mns <- c()
q25 <- c()
q97 <- c()

#dissimilarities_epoch2 <- dissimilarities_epoch[,-ncol(dissimilarities_epoch)]
dissimilarities_epoch2 <- dissimilarities_epoch

for(c in 1:ncol(dissimilarities_epoch2)){

	qs <- quantile(dissimilarities_epoch2[,c],c(0.025,0.5,0.975))
	mns <- c(mns,qs[2])
	q25 <- c(q25,qs[1])
	q97 <- c(q97,qs[3])

}

png("V3_Dissimilarities_1965Epoch.png")
plot(x=1966:2017,y=mns,type="l",ylim=c(0,0.5),xlab=NA,ylab="Bird community dissimilarity from 1966 conditions")
lines(x=1966:2017,y=q25,lty=3)
lines(x=1966:2017,y=q97,lty=3)
dev.off()
