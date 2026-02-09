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




setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/")

# which data file to use
file_name <- "//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/biol/birds/from-ala/filtered/filtered_data_2018-11-20.csv"

## get grid parametres from env grid\\ces-10-cdc\OSM_CDC_GISDATA_work\_DEV\her134\SUBS\out\DES_mean.flt
ras <- "//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/AWAP_monthly/RECA/monthly/years1901_2017/Centered1960/FWPT_mean_Cmean_mean_1946_1975.flt" ##"//ces-10-cdc/OSM_CDC_GISDATA_work/_DEV/her134/SUBS/out/DES_mean.flt"
ras_sp <- raster(ras)
res <- res(ras_sp)[1] ## resolution
box <- extent(ras_sp) ## bounding box

## load raw specise observations
dat <- read.csv(file_name)

## aggregate to site
datRED <- siteAggregator(dat,res,box)

## clean sites where substrate gives NAs
writeRaster(ras_sp,file="TEMP_RAS.grd")
ras_sp <- raster("TEMP_RAS.grd")
test <- is.na(extract(ras_sp,datRED[,c(6,7)]))
datRED <- datRED[!test,]
file.remove("TEMP_RAS.grd")
file.remove("TEMP_RAS.gri")

setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j")
write.csv(datRED,"AVES_aggregated_basicFilt.csv",row.names=FALSE)
save(datRED,file="AVES_aggregated_basicFilt.RData")

### load datRED from siteAggregator process
#
s1 <- Sys.time()
load("AVES_aggregated_basicFilt.RData")

##nMatch <- 1000000


## reduce to dates where we have env data
datRED <- datRED[datRED$eventDate != "",]
date_test <- as.Date(datRED$eventDate) > as.Date("1926-01-01")
datRED <- datRED[date_test,]
date_test <- as.Date(datRED$eventDate) < as.Date("2018-01-01")
datRED <- datRED[date_test,]
datRED <- droplevels(datRED)

## calculate obsPair transform weights
nMatch <- 2000000
nSamples <- nMatch*4
idx <- 1:nrow(datRED)
s1 <- sample(idx,nSamples,replace=T)
s2 <- sample(idx,nSamples,replace=T)
miss <- datRED$gen_spec[s1] != datRED$gen_spec[s2]

missCount <- sum(miss)
matchCount <- nSamples - missCount
propMiss <- missCount / nSamples
propMatch <- matchCount / nSamples
w <- propMiss/propMatch
w

biol_group="AVES"

run <- 8
nMatches <- c(rep(7500000,10),rep(800000,10),rep(850000,10))

for(idx1 in 1:length(nMatches)){

nMatch <- nMatches[idx1]
save_prefix <- paste("AVES_",nMatch/1000000,"mil_",idx1,"_run",run,"_",sep="")

## get samples and env data

#obsPairs_out <- getSamples_AND_Env_RECA(datRED=datRED,nMatch=nMtch,biol_group="AVES",gen_windows2=gen_windows,exe2=exe)
if(TRUE){
	#require(dynowindow)
	require(raster)
	require(feather)
	require(doSNOW)
	require(foreach)
	require(parallel)
	#source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/dynowindow.r")
	
	
	if(is.null(nMatch)){stop("nMatch == null: Need to specify number of matches to extract.")}
	
	nMatch <- nMatch
	exe <- exe

	data <- data.frame(ID=datRED$ID,Latitude=datRED$latID,Longitude=datRED$lonID,species=datRED$gen_spec,nRecords=datRED$nRecords,nRecords.exDateLocDups=datRED$nRecords.exDateLocDups,nSiteVisits=datRED$nSiteVisits,richness=datRED$richness,stringsAsFactors=FALSE)
	## reduce data to unique site by species
	## calculate number of unique species per site
	LocDups <- paste(data$ID,data$species,sep=":")
	test <- duplicated(LocDups)
	data <- data[!test,]
	## sort by site
	odr <- order(data$ID)
	data <- data[odr,]
	## add row.count
	data$row.count <- 1:nrow(data)
	## species list to factor
	data$species <- as.factor(data$species)

	# ## count the number of unique species*site records
	 LocDups <- as.factor(LocDups)
	 ones <- rep(1,nrow(datRED))
	 count <- bySum(ones,LocDups)


	## use site.richness.extractor to get site x species matrix - used in calculating sorenson - not needed but useful for validation
	sitesPerIteration=50000
	frog.auGrid <- site.richness.extractor.bigData(frog.auGrid=data)

	## back to full dataset
	frog.auGrid <- data.frame(ID=datRED$ID,Latitude=datRED$latID,Longitude=datRED$lonID,species=datRED$gen_spec,eventDate=as.character(datRED$eventDate),nRecords=datRED$nRecords,nRecords.exDateLocDups=datRED$nRecords.exDateLocDups,nSiteVisits=datRED$nSiteVisits,richness=datRED$richness,stringsAsFactors=FALSE,Site.Richness=datRED$richness)

	## expand m1 back to full dataset
	row.nums <- 1:nrow(m1)
	row.vect <- rep(row.nums,count)
	m2 <- m1[row.vect,]

	# garbage collection
	gc()

	## run obsPair sampler
	obsPairs_out <- obsPairSampler.bigData.RECA(frog.auGrid,nMatch,m1=m2,richness=TRUE,speciesThreshold=500,coresToUse=detectCores()-1)

	registerDoSEQ()
	
	
	s1 <- Sys.time()
	## extract env data - current hardcode window lengths
	ext_data <- obsPairs_out[,2:9]

	init_params <- list(
		list(variables = c("TNn_191101-201712", "TXx_191101-201712"),
			mstat = 'mean', 
			cstat = 'mean', 
			window = 15,
			prefix="XbrXbr_15"),
		list(variables = c("Precip_191101-201712", "FWPT_191101-201712"),
			mstat = 'min', 
			cstat = 'mean', 
			window = 15,
			prefix="MinXbr_15"),
		list(variables = c("FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
			mstat = 'min', 
			cstat = 'mean', 
			window = 15,
			prefix="MinXbr_15"),
		list(variables = c("Precip_191101-201712", "FWPT_191101-201712"),
			mstat = 'max', 
			cstat = 'mean', 
			window = 15,
			prefix="MaxXbr_15"),
		list(variables = c("TNn_191101-201712","TXx_191101-201712"),
			mstat = 'max', 
			cstat = 'mean', 
			window = 15,
			prefix="MaxXbr_15"),
		list(variables = c("TNn_191101-201712", "TXx_191101-201712"),
			mstat = 'mean', 
			cstat = 'mean', 
			window = 1,
			prefix="XbrXbr_1"),
		list(variables = c("Precip_191101-201712", "FWPT_191101-201712"),
			mstat = 'min', 
			cstat = 'mean', 
			window = 1,
			prefix="MinXbr_1"),
		list(variables = c("FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
			mstat = 'min', 
			cstat = 'mean', 
			window = 1,
			prefix="MinXbr_1"),
		list(variables = c("Precip_191101-201712", "FWPT_191101-201712"),
			mstat = 'max', 
			cstat = 'mean', 
			window = 1,
			prefix="MaxXbr_1"),
		list(variables = c("TNn_191101-201712","TXx_191101-201712"),
			mstat = 'max', 
			cstat = 'mean', 
			window = 1,
			prefix="MaxXbr_1")
	)
	
	
	cl <- makeCluster(10)
	registerDoSNOW(cl)
	
	env_out <- foreach(x=1:10,.combine='cbind',.packages="feather") %dopar% {
		out <- gen_windows(pairs=ext_data,
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
	
	# env15_meanmean <- gen_windows(pairs = ext_data,
		# variables = c("TNn_191101-201712", "TXx_191101-201712"),
		# mstat = 'mean', 
		# cstat = 'mean', 
		# window = 15)
	# env15_minmean <- gen_windows(pairs = ext_data,
		# variables = c("Precip_191101-201712", "FWPT_191101-201712","FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
		# mstat = 'min', 
		# cstat = 'mean', 
		# window = 15)
	# env15_maxmean <- gen_windows(pairs = ext_data,
		# variables = c("Precip_191101-201712", "FWPT_191101-201712","TNn_191101-201712","TXx_191101-201712"),
		# mstat = 'max', 
		# cstat = 'mean', 
		# window = 15)

	  # env1_meanmean <- gen_windows(pairs = ext_data[1:10,],
		  # variables = c("TNn_191101-201712", "TXx_191101-201712"),
		  # mstat = 'mean', 
		  # cstat = 'mean', 
		  # window = 15)
	 # env1_minmean <- gen_windows(pairs = ext_data[1:10,],
		 # variables = c("Precip_191101-201712"),
		 # mstat = 'min', 
		 # cstat = 'mean', 
		 # window = 1) ##, "FWPT_191101-201712","FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
		
	# env1_maxmean <- gen_windows(pairs = ext_data,
		# variables = c("Precip_191101-201712", "FWPT_191101-201712","TNn_191101-201712","TXx_191101-201712"),
		# mstat = 'max', 
		# cstat = 'mean', 
		# window = 1)
		
	## reshuffle env_out
	env_out1 <- env_out[,grep("201712_1",names(env_out))]
	env_out2 <- env_out[,grep("201712_2",names(env_out))]
	
	## make anomalies
	env_anom_out1 <- env_out1[,12:22] - env_out1[,1:11]
	colnames(env_anom_out1) <- gsub("191101-201712","anom",colnames(env_anom_out1))
	env_anom_out2 <- env_out2[,12:22] - env_out2[,1:11]
	colnames(env_anom_out2) <- gsub("191101-201712","anom",colnames(env_anom_out2))
		
	pnt1 <- SpatialPoints(data.frame(ext_data$Lon1,ext_data$Lat1))
	pnt2 <- SpatialPoints(data.frame(ext_data$Lon2,ext_data$Lat2))

	subs_brk <- brick("SUBS_brk.grd")
	env1_subs <- extract(subs_brk,pnt1)
	env2_subs <- extract(subs_brk,pnt2)
	
	
	nms <- paste(colnames(env1_subs),"_1",sep="")
	colnames(env1_subs) <- nms
	nms <- paste(colnames(env2_subs),"_2",sep="")
	colnames(env2_subs) <- nms
	
	Sys.time() - s1
	
	Trng_1 <- abs(env_out1[,c(11,21)] - env_out1[,c(7,17)])
	names(Trng_1) <- c("Trng_15_191101-201712_1","Trng_1_191101-201712_1")
	Trng_2 <- abs(env_out2[,c(11,21)] - env_out2[,c(7,17)])
	names(Trng_2) <- c("Trng_15_191101-201712_2","Trng_1_191101-201712_2")
	
	Sys.time()-s1
	
	# obsPairs_out <- cbind(obsPairs_out,
							# env15_meanmean[,9:12],env15_minmean[,9:18],env15_maxmean[,9:16],
							# env1_meanmean[,9:12],env1_minmean[,9:18],env1_maxmean[,9:16],
							# env1_subs,env2_subs)
	obsPairs_out <- cbind(obsPairs_out,
							env_out1,env1_subs,Trng_1,env_anom_out1,
							env_out2,env2_subs,Trng_2,env_anom_out2)

	## clean up
	rm(ext_data)
	rm(frog.auGrid)
	rm(data)
	rm(m1)
	rm(list=c("env_out","env_out1","env_out2","env1_subs","env2_subs","Trng_1","Trng_2"))
	gc()
	
	## save
	fn=paste("ObsPairsTable_RECA_",biol_group,".RData",sep="")
	save(obsPairs_out,file=fn)
	#write.csv()
	}




## temporalily remove NAs - this to be fixed in getSamples before sampling
test <- is.na(rowSums(obsPairs_out[,23:102]))
obsPairs_out <- obsPairs_out[!test,]

## there's NAflags present (-9999). Find and remove
tst <- c()
for(c in 23:102){
	if(any(obsPairs_out[,c] == -9999)){
	tst <- c(tst,c);print(c)}
	}
test <- rep(0,nrow(obsPairs_out))
for(c in 23:102){
	test <- test + (obsPairs_out[,c] == -9999)
}
test <- test == 0

obsPairs_out <- obsPairs_out[test,]


# ## check correlctions
# #cor_test <- cor(obsPairs_out[,23:78])
# cor_test <- cor(obsPairs_out[,23:102])

# prs <- c()
# cors <- c()

# for(i in 1:nrow(cor_test)){
	# for(j in 1:ncol(cor_test)){
		# if(j > i){
			# tst <-abs(cor_test[i,j] > 0.7 & cor_test[i,j] < 1.0)
			# if(tst){
			# prs <- c(prs,paste(rownames(cor_test)[i],colnames(cor_test)[j],sep="~"))
			# cors <- c(cors,cor_test[i,j] )
			# }
		# }
	# }
# }

# #cbind(prs,cors)


# cor(obsPairs_out[tst,c(23:33)[3:4]],pch="+")

## spline data
toSpline  <- obsPairs_out[,c(23:33,45:73,85:102)]
toSpline <- toSpline[,-c(grep("XbrXbr",colnames(toSpline)),grep("MinXbr_1_TNn_",colnames(toSpline)),grep("MinXbr_15_TNn_",colnames(toSpline)),grep("MinXbr_1_TNn_anom",colnames(toSpline)))]
#toSpline  <- obsPairs_out[,c(23,24,51,52)] ## testing
splined <- splineData(toSpline)

#splined <- splined[,1:3] ## testing

## fit model
mod_ready <- cbind(Match=obsPairs_out$Match,as.data.frame(splined)) ## make table
colnames(mod_ready) <- gsub("191101-201712_","",colnames(mod_ready)) ## make colnames r friendly
f1 <- paste(colnames(mod_ready)[-1],collapse="+") ## fformula
formula <- as.formula(paste(colnames(mod_ready)[1],"~",f1,sep="")) ## formula
obsGDM_1 <- fitGDM(formula=formula,data=mod_ready) ## fit

## rejoin everything to fit for diagnostic plots
fit <- list()
fit$intercept <- coef(obsGDM_1)[1]
fit$sample <- nrow(data)

fit$predictors <- gsub("191101-201712_","",gsub("_spl1","",colnames(splined)[grep("_spl1",colnames(splined))]))
fit$coefficients <- coef(obsGDM_1)[-1]
fit$coefficients[is.na(fit$coefficients)] <- 0
fitnms <- names(fit$quantiles)
## fold X and create site vector
nc <- ncol(toSpline)
nc2 <- nc/2
X1 <- toSpline[,1:nc2]
X2 <- toSpline[,(nc2+1):nc]
nms <- names(X1)
names(X2) <- nms
## site vector
sv <- c(rep(1,nrow(X1)),rep(2,nrow(X2)))
XX <- rbind(X1,X2)
fit$quantiles <- unlist(lapply(1:ncol(XX),function(x){quantile(XX[,x],c(0,0.5,1))}))
#names(fit$quantiles) <- c(fitnms,names(MDSquan))
fit$splines <- rep(3,ncol(XX))
fit$predicted <- fitted(obsGDM_1)
fit$ecological <- obsGDM_1$linear.predictors

save(fit,file=paste(save_prefix,"fittedGDM.RData",sep="_"))
coefs<-coef(obsGDM_1)
save(coefs,file=paste(save_prefix,"coefficients.RData",sep="_"))
D2 <- (obsGDM_1$null.deviance - obsGDM_1$deviance) /obsGDM_1$null.deviance
print(idx1)
print(paste('Deviance exp:',D2))
save(D2,file=paste(save_prefix,"deviance.RData",sep="_"))
#save(obsGDM,filename=paste(save_prefix,"fittedMODobj.RData",sep="_"))

## diagnostic plots
tiff(paste(save_prefix,"GDM-ObsDiag.tif",sep="_"),height=6,width=6,units="in",res=200,compression="lzw")
obs.gdm.plot(obsGDM_1,paste(save_prefix),w,Is=fit$intercept)
dev.off()

pdf(paste(save_prefix,"GDM-gdmDiag.pdf",sep="_"))
gdm.spline.plot(fit)		
dev.off()


} ## end for loop



run <- 9
nMatches <- c(rep(950000,10),rep(1100000,10),rep(1200000,10),rep(1600000,10))

for(idx1 in 1:length(nMatches)){

nMatch <- nMatches[idx1]
save_prefix <- paste("AVES_",nMatch/1000000,"mil_",idx1,"_run",run,"_",sep="")

## get samples and env data

#obsPairs_out <- getSamples_AND_Env_RECA(datRED=datRED,nMatch=nMtch,biol_group="AVES",gen_windows2=gen_windows,exe2=exe)
if(TRUE){
	#require(dynowindow)
	require(raster)
	require(feather)
	require(doSNOW)
	require(foreach)
	require(parallel)
	#source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/dynowindow.r")
	
	
	if(is.null(nMatch)){stop("nMatch == null: Need to specify number of matches to extract.")}
	
	nMatch <- nMatch
	exe <- exe

	data <- data.frame(ID=datRED$ID,Latitude=datRED$latID,Longitude=datRED$lonID,species=datRED$gen_spec,nRecords=datRED$nRecords,nRecords.exDateLocDups=datRED$nRecords.exDateLocDups,nSiteVisits=datRED$nSiteVisits,richness=datRED$richness,stringsAsFactors=FALSE)
	## reduce data to unique site by species
	## calculate number of unique species per site
	LocDups <- paste(data$ID,data$species,sep=":")
	test <- duplicated(LocDups)
	data <- data[!test,]
	## sort by site
	odr <- order(data$ID)
	data <- data[odr,]
	## add row.count
	data$row.count <- 1:nrow(data)
	## species list to factor
	data$species <- as.factor(data$species)

	# ## count the number of unique species*site records
	 LocDups <- as.factor(LocDups)
	 ones <- rep(1,nrow(datRED))
	 count <- bySum(ones,LocDups)


	## use site.richness.extractor to get site x species matrix - used in calculating sorenson - not needed but useful for validation
	sitesPerIteration=50000
	frog.auGrid <- site.richness.extractor.bigData(frog.auGrid=data)

	## back to full dataset
	frog.auGrid <- data.frame(ID=datRED$ID,Latitude=datRED$latID,Longitude=datRED$lonID,species=datRED$gen_spec,eventDate=as.character(datRED$eventDate),nRecords=datRED$nRecords,nRecords.exDateLocDups=datRED$nRecords.exDateLocDups,nSiteVisits=datRED$nSiteVisits,richness=datRED$richness,stringsAsFactors=FALSE,Site.Richness=datRED$richness)

	## expand m1 back to full dataset
	row.nums <- 1:nrow(m1)
	row.vect <- rep(row.nums,count)
	m2 <- m1[row.vect,]

	# garbage collection
	gc()

	## run obsPair sampler
	obsPairs_out <- obsPairSampler.bigData.RECA(frog.auGrid,nMatch,m1=m2,richness=TRUE,speciesThreshold=500,coresToUse=detectCores()-1)

	registerDoSEQ()
	
	
	s1 <- Sys.time()
	## extract env data - current hardcode window lengths
	ext_data <- obsPairs_out[,2:9]

	init_params <- list(
		list(variables = c("TNn_191101-201712", "TXx_191101-201712"),
			mstat = 'mean', 
			cstat = 'mean', 
			window = 15,
			prefix="XbrXbr_15"),
		list(variables = c("Precip_191101-201712", "FWPT_191101-201712"),
			mstat = 'min', 
			cstat = 'mean', 
			window = 15,
			prefix="MinXbr_15"),
		list(variables = c("FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
			mstat = 'min', 
			cstat = 'mean', 
			window = 15,
			prefix="MinXbr_15"),
		list(variables = c("Precip_191101-201712", "FWPT_191101-201712"),
			mstat = 'max', 
			cstat = 'mean', 
			window = 15,
			prefix="MaxXbr_15"),
		list(variables = c("TNn_191101-201712","TXx_191101-201712"),
			mstat = 'max', 
			cstat = 'mean', 
			window = 15,
			prefix="MaxXbr_15"),
		list(variables = c("TNn_191101-201712", "TXx_191101-201712"),
			mstat = 'mean', 
			cstat = 'mean', 
			window = 1,
			prefix="XbrXbr_1"),
		list(variables = c("Precip_191101-201712", "FWPT_191101-201712"),
			mstat = 'min', 
			cstat = 'mean', 
			window = 1,
			prefix="MinXbr_1"),
		list(variables = c("FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
			mstat = 'min', 
			cstat = 'mean', 
			window = 1,
			prefix="MinXbr_1"),
		list(variables = c("Precip_191101-201712", "FWPT_191101-201712"),
			mstat = 'max', 
			cstat = 'mean', 
			window = 1,
			prefix="MaxXbr_1"),
		list(variables = c("TNn_191101-201712","TXx_191101-201712"),
			mstat = 'max', 
			cstat = 'mean', 
			window = 1,
			prefix="MaxXbr_1")
	)
	
	
	cl <- makeCluster(10)
	registerDoSNOW(cl)
	
	env_out <- foreach(x=1:10,.combine='cbind',.packages="feather") %dopar% {
		out <- gen_windows(pairs=ext_data,
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
	
	# env15_meanmean <- gen_windows(pairs = ext_data,
		# variables = c("TNn_191101-201712", "TXx_191101-201712"),
		# mstat = 'mean', 
		# cstat = 'mean', 
		# window = 15)
	# env15_minmean <- gen_windows(pairs = ext_data,
		# variables = c("Precip_191101-201712", "FWPT_191101-201712","FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
		# mstat = 'min', 
		# cstat = 'mean', 
		# window = 15)
	# env15_maxmean <- gen_windows(pairs = ext_data,
		# variables = c("Precip_191101-201712", "FWPT_191101-201712","TNn_191101-201712","TXx_191101-201712"),
		# mstat = 'max', 
		# cstat = 'mean', 
		# window = 15)

	  # env1_meanmean <- gen_windows(pairs = ext_data[1:10,],
		  # variables = c("TNn_191101-201712", "TXx_191101-201712"),
		  # mstat = 'mean', 
		  # cstat = 'mean', 
		  # window = 15)
	 # env1_minmean <- gen_windows(pairs = ext_data[1:10,],
		 # variables = c("Precip_191101-201712"),
		 # mstat = 'min', 
		 # cstat = 'mean', 
		 # window = 1) ##, "FWPT_191101-201712","FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
		
	# env1_maxmean <- gen_windows(pairs = ext_data,
		# variables = c("Precip_191101-201712", "FWPT_191101-201712","TNn_191101-201712","TXx_191101-201712"),
		# mstat = 'max', 
		# cstat = 'mean', 
		# window = 1)
		
	## reshuffle env_out
	env_out1 <- env_out[,grep("201712_1",names(env_out))]
	env_out2 <- env_out[,grep("201712_2",names(env_out))]
	
	## make anomalies
	env_anom_out1 <- env_out1[,12:22] - env_out1[,1:11]
	colnames(env_anom_out1) <- gsub("191101-201712","anom",colnames(env_anom_out1))
	env_anom_out2 <- env_out2[,12:22] - env_out2[,1:11]
	colnames(env_anom_out2) <- gsub("191101-201712","anom",colnames(env_anom_out2))
		
	pnt1 <- SpatialPoints(data.frame(ext_data$Lon1,ext_data$Lat1))
	pnt2 <- SpatialPoints(data.frame(ext_data$Lon2,ext_data$Lat2))

	subs_brk <- brick("SUBS_brk.grd")
	env1_subs <- extract(subs_brk,pnt1)
	env2_subs <- extract(subs_brk,pnt2)
	
	
	nms <- paste(colnames(env1_subs),"_1",sep="")
	colnames(env1_subs) <- nms
	nms <- paste(colnames(env2_subs),"_2",sep="")
	colnames(env2_subs) <- nms
	
	Sys.time() - s1
	
	Trng_1 <- abs(env_out1[,c(11,21)] - env_out1[,c(7,17)])
	names(Trng_1) <- c("Trng_15_191101-201712_1","Trng_1_191101-201712_1")
	Trng_2 <- abs(env_out2[,c(11,21)] - env_out2[,c(7,17)])
	names(Trng_2) <- c("Trng_15_191101-201712_2","Trng_1_191101-201712_2")
	
	Sys.time()-s1
	
	# obsPairs_out <- cbind(obsPairs_out,
							# env15_meanmean[,9:12],env15_minmean[,9:18],env15_maxmean[,9:16],
							# env1_meanmean[,9:12],env1_minmean[,9:18],env1_maxmean[,9:16],
							# env1_subs,env2_subs)
	obsPairs_out <- cbind(obsPairs_out,
							env_out1,env1_subs,Trng_1,env_anom_out1,
							env_out2,env2_subs,Trng_2,env_anom_out2)

	## clean up
	rm(ext_data)
	rm(frog.auGrid)
	rm(data)
	rm(m1)
	rm(list=c("env_out","env_out1","env_out2","env1_subs","env2_subs","Trng_1","Trng_2"))
	gc()
	
	## save
	fn=paste("ObsPairsTable_RECA_",biol_group,".RData",sep="")
	save(obsPairs_out,file=fn)
	#write.csv()
	}




## temporalily remove NAs - this to be fixed in getSamples before sampling
test <- is.na(rowSums(obsPairs_out[,23:102]))
obsPairs_out <- obsPairs_out[!test,]

## there's NAflags present (-9999). Find and remove
tst <- c()
for(c in 23:102){
	if(any(obsPairs_out[,c] == -9999)){
	tst <- c(tst,c);print(c)}
	}
test <- rep(0,nrow(obsPairs_out))
for(c in 23:102){
	test <- test + (obsPairs_out[,c] == -9999)
}
test <- test == 0

obsPairs_out <- obsPairs_out[test,]


# ## check correlctions
# #cor_test <- cor(obsPairs_out[,23:78])
# cor_test <- cor(obsPairs_out[,23:102])

# prs <- c()
# cors <- c()

# for(i in 1:nrow(cor_test)){
	# for(j in 1:ncol(cor_test)){
		# if(j > i){
			# tst <-abs(cor_test[i,j] > 0.7 & cor_test[i,j] < 1.0)
			# if(tst){
			# prs <- c(prs,paste(rownames(cor_test)[i],colnames(cor_test)[j],sep="~"))
			# cors <- c(cors,cor_test[i,j] )
			# }
		# }
	# }
# }

# #cbind(prs,cors)


# cor(obsPairs_out[tst,c(23:33)[3:4]],pch="+")

## spline data
toSpline  <- obsPairs_out[,c(23:33,45:73,85:102)]
toSpline <- toSpline[,-c(grep("XbrXbr",colnames(toSpline)),grep("MinXbr_1_TNn_",colnames(toSpline)),grep("MinXbr_15_TNn_",colnames(toSpline)),grep("MinXbr_1_TNn_anom",colnames(toSpline)))]
#toSpline  <- obsPairs_out[,c(23,24,51,52)] ## testing
splined <- splineData(toSpline)

#splined <- splined[,1:3] ## testing

## fit model
mod_ready <- cbind(Match=obsPairs_out$Match,as.data.frame(splined)) ## make table
colnames(mod_ready) <- gsub("191101-201712_","",colnames(mod_ready)) ## make colnames r friendly
f1 <- paste(colnames(mod_ready)[-1],collapse="+") ## fformula
formula <- as.formula(paste(colnames(mod_ready)[1],"~",f1,sep="")) ## formula
obsGDM_1 <- fitGDM(formula=formula,data=mod_ready) ## fit

## rejoin everything to fit for diagnostic plots
fit <- list()
fit$intercept <- coef(obsGDM_1)[1]
fit$sample <- nrow(data)

fit$predictors <- gsub("191101-201712_","",gsub("_spl1","",colnames(splined)[grep("_spl1",colnames(splined))]))
fit$coefficients <- coef(obsGDM_1)[-1]
fit$coefficients[is.na(fit$coefficients)] <- 0
fitnms <- names(fit$quantiles)
## fold X and create site vector
nc <- ncol(toSpline)
nc2 <- nc/2
X1 <- toSpline[,1:nc2]
X2 <- toSpline[,(nc2+1):nc]
nms <- names(X1)
names(X2) <- nms
## site vector
sv <- c(rep(1,nrow(X1)),rep(2,nrow(X2)))
XX <- rbind(X1,X2)
fit$quantiles <- unlist(lapply(1:ncol(XX),function(x){quantile(XX[,x],c(0,0.5,1))}))
#names(fit$quantiles) <- c(fitnms,names(MDSquan))
fit$splines <- rep(3,ncol(XX))
fit$predicted <- fitted(obsGDM_1)
fit$ecological <- obsGDM_1$linear.predictors

save(fit,file=paste(save_prefix,"fittedGDM.RData",sep="_"))
coefs<-coef(obsGDM_1)
save(coefs,file=paste(save_prefix,"coefficients.RData",sep="_"))
D2 <- (obsGDM_1$null.deviance - obsGDM_1$deviance) /obsGDM_1$null.deviance
print(idx1)
print(paste('Deviance exp:',D2))
save(D2,file=paste(save_prefix,"deviance.RData",sep="_"))
#save(obsGDM,filename=paste(save_prefix,"fittedMODobj.RData",sep="_"))

## diagnostic plots
tiff(paste(save_prefix,"GDM-ObsDiag.tif",sep="_"),height=6,width=6,units="in",res=200,compression="lzw")
obs.gdm.plot(obsGDM_1,paste(save_prefix),w,Is=fit$intercept)
dev.off()

pdf(paste(save_prefix,"GDM-gdmDiag.pdf",sep="_"))
gdm.spline.plot(fit)		
dev.off()


} ## end for loop


