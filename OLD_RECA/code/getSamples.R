#######
##
## Make obs-pair sampled set
##
#######

library(ffbase)
library(raster)

source("//ces-10-cdc/OSM_CDC_MMRG_work/users/GPAAG_refuge/BIOL/G/DEV/_tools/ObsPairSampler/site-richness-extractor-bigData.R")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/obsPairSampler-bigData-RECA.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/siteAggregator.r")

setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/")

# which data file to use
file_name <- "//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/biol/birds/from-ala/filtered/filtered_data_2018-11-20.csv"

## get grid parametres from env grid\\ces-10-cdc\OSM_CDC_GISDATA_work\_DEV\her134\SUBS\out\DES_mean.flt
ras <- "//ces-10-cdc/OSM_CDC_GISDATA_work/_DEV/her134/SUBS/out/DES_mean.flt"
ras_sp <- raster(ras)
res <- res(ras_sp)[1] ## resolution
box <- extent(ras_sp) ## bounding box

## load raw specise observations
dat <- read.csv(file_name)

## aggregate to site
datRED <- siteAggregator(dat,res,box)
setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j")
write.csv(datRED,"AVES_aggregated_basicFilt.csv",row.names=FALSE)
save(datRED,file="AVES_aggregated_basicFilt.RData")

### load datRED from siteAggregator process
#load("AVES_aggregated_basicFilt.RData")

## reduce to dates where we have env data
datRED <- datRED[datRED$eventDate != "",]
date_test <- as.Date(datRED$eventDate) > as.Date("1926-01-01")
datRED <- datRED[date_test,]
date_test <- as.Date(datRED$eventDate) < as.Date("2018-01-01")
datRED <- datRED[date_test,]
datRED <- droplevels(datRED)



getEnv_RECA(datRED=datRED,nMatch=1000000){
	require(dynowindow)
	require(raster)

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
	# LocDups <- as.factor(LocDups)
	# ones <- rep(1,nrow(datRED))
	# count <- bySum(ones,LocDups)


	## use site.richness.extractor to get site x species matrix - used in calculating sorenson - not needed but useful for validation
	sitesPerIteration=50000
	frog.auGrid <- site.richness.extractor.bigData(frog.auGrid=data)

	## back to full dataset
	frog.auGrid <- data.frame(ID=datRED$ID,Latitude=datRED$latID,Longitude=datRED$lonID,species=datRED$gen_spec,eventDate=as.character(datRED$eventDate),nRecords=datRED$nRecords,nRecords.exDateLocDups=datRED$nRecords.exDateLocDups,nSiteVisits=datRED$nSiteVisits,richness=datRED$richness,stringsAsFactors=FALSE,Site.Richness=datRED$richness)

	## expand m1 back to full dataset
	row.nums <- 1:nrow(m1)
	row.vect <- rep(row.nums,count)
	m1 <- m1[row.vect,]

	# garbase collection
	

	## run obsPair sampler
	obsPairs_out <- obsPairSampler.bigData.RECA(frog.auGrid,nMatch,richness=TRUE,speciesThreshold=500,coresToUse=detectCores()-1)

	## extract env data - current hardcode window lengths
	ext_data <- obsPairs_out[1:10,2:9]

	env15_meanmean <- gen_windows(pairs = ext_data,
		variables = c("TNn_191101-201712", "TXx_191101-201712"),
		mstat = 'mean', 
		cstat = 'mean', 
		window = 15)
	env15_minmean <- gen_windows(pairs = ext_data,
		variables = c("Precip_191101-201712", "FWPT_191101-201712","FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
		mstat = 'min', 
		cstat = 'mean', 
		window = 15)
	env15_maxmean <- gen_windows(pairs = ext_data,
		variables = c("Precip_191101-201712", "FWPT_191101-201712","TNn_191101-201712","TXx_191101-201712"),
		mstat = 'max', 
		cstat = 'mean', 
		window = 15)

	env1_meanmean <- gen_windows(pairs = ext_data,
		variables = c("TNn_191101-201712", "TXx_191101-201712"),
		mstat = 'mean', 
		cstat = 'mean', 
		window = 1)
	env1_minmean <- gen_windows(pairs = ext_data,
		variables = c("Precip_191101-201712", "FWPT_191101-201712","FD_191101-201712","WRel1_191101-201712","TNn_191101-201712"),
		mstat = 'min', 
		cstat = 'mean', 
		window = 1)
	env1_maxmean <- gen_windows(pairs = ext_data,
		variables = c("Precip_191101-201712", "FWPT_191101-201712","TNn_191101-201712","TXx_191101-201712"),
		mstat = 'max', 
		cstat = 'mean', 
		window = 1)
		
	pnt1 <- SpatialPoints(data.frame(ext_data$Lon1,ext_data$Lat1))
	pnt2 <- SpatialPoints(data.frame(ext_data$Lon2,ext_data$Lat2))

	subs_brk <- brick("SUBS_brk.grd")
	env1_subs <- extract(subs_brk,pnt1)
	env2_subs <- extract(subs_brk,pnt2)

	
}