
## create obsPairs dataset

## load dependencies
library(ffbase)
library(raster)
library(SDMTools)
library(raster)
library(lubridate)
require(doSNOW)
require(foreach)
require(parallel)

source("//ces-10-cdc/OSM_CDC_MMRG_work/users/GPAAG_refuge/BIOL/G/DEV/_tools/ObsPairSampler/site-richness-extractor-bigData.R")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/obsPairSampler-bigData-RECA.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/siteAggregator.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/getSamples_AND_ENV.r")

## working directory
setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/")

######################################################################
##
## Aggregation bit. Can skip this and load saved aggregated files 
##
# which data file to use
file_name <- "//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/SOURCE/biol/plants/ala-nov-2018-processing/filtered_data_2018-11-05.csv"


## get grid parametres from env grid\\ces-10-cdc\OSM_CDC_GISDATA_work\_DEV\her134\SUBS\out\DES_mean.flt
ras <- "//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/AWAP_monthly/RECA/monthly/years1901_2017/Centered1960/FWPT_mean_Cmean_mean_1946_1975.flt" ##"//ces-10-cdc/OSM_CDC_GISDATA_work/_DEV/her134/SUBS/out/DES_mean.flt"
ras_sp <- raster(ras)
res <- res(ras_sp)[1] ## resolution
box <- extent(ras_sp) ## bounding box

## load raw species observations
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
write.csv(datRED,"PLANTS_aggregated_basicFilt.csv",row.names=FALSE)
save(datRED,file="PLANTS_aggregated_basicFilt.RData")

##
###########################################################################

yr <- 15 ## width of your climate window
biol_group <- "PLANTS" ## or 'AVES'

## input file - aggregated obs data
fn_in <- paste(biol_group,"_aggregated_basicFilt.RData",sep="") 

## output file
wd_out <- ## ADD YOUR OUTPUT WORKING DIRECTORY HERE
fn_out=paste("ObsPairsTable_RECA_",biol_group,"_nMatch_",nMatch,".RData",sep="")

## load aggregated data
setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/")
load(fn_in) ## "AVES_aggregated_basicFilt.RData"

## number of matches - 50% of desired pairs 
nMatch <- 185000 ## this is max for PLANTS. Max available for birds is around 1600000

## reduce to dates where we have env data
datRED <- datRED[!is.na(datRED$gen_spec),]
datRED <- datRED[datRED$eventDate != "",]
date_test <- as.Date(datRED$eventDate) > (as.Date("1911-01-01") %m+% years(yr))
datRED <- datRED[date_test,]
date_test <- as.Date(datRED$eventDate) < as.Date("2018-01-01")
datRED <- datRED[date_test,]
datRED <- droplevels(datRED)

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

## clean up a bit
registerDoSEQ()

## save
setwd(wd_out)
save(obsPairs_out,file=fn_out)