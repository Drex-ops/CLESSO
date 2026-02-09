#######
##
## Make aggregated sample from raw
##
#######

siteAggregator <- function(dat,res,box){

	require(ffbase)

	## set breaks
	latBRKS <- seq(box[3],box[4],by=res)
	lonBRKS <- seq(box[1],box[2],by=res)
	
	## calculate centroids
	os <- res/2
	latCENT <-  seq(box[3]+os,box[4]-os,by=res)
	lonCENT <- seq(box[1]+os,box[2]-os,by=res)

	## break data
	latCUT <- cut(dat$decimalLatitude,breaks=latBRKS)
	lonCUT <- cut(dat$decimalLongitude,breaks=lonBRKS)

	## reduce data.frame to wanted information
	datRED <- data.frame(RAW_latdec=dat$decimalLatitude,RAW_longdec=dat$decimalLongitude,crdUncertainty=dat$coordinateUncertaintyInMeters,gen_spec=as.character(dat$scientificName),eventDate=dat$eventDate)
	#rm(dat)
	gc()
	
	## assign centroids
	datRED$lonID <- lonCENT[as.numeric(lonCUT)]
	datRED$latID <- latCENT[as.numeric(latCUT)]
	
	
	## test plot - ALL GOOD
	#plot(ras_sp,xlim=c( 146, 147),ylim=c(-43,-42))
	#points(datRED$lonID[datRED$latID >= -43.95364 & datRED$latID <= -40.18248],datRED$latID[datRED$latID >= -43.95364 & datRED$latID <= -40.18248],pch="+")

	## garbage collection
	rm(list=c("latCUT","lonCUT","latCENT","lonCENT"))
	gc()

	## remove duplicate location-species records
	n1 <- nrow(datRED)
	dupID <- paste(datRED$RAW_latdec,datRED$RAW_longdec,datRED$gen_spec,dat$eventDate,sep=":")
	test <- duplicated(dupID)
	datRED <- datRED[!test,]
	print(n1 - nrow(datRED))

	## create site IDs
	datRED$ID <- paste(datRED$lonID,datRED$latID,sep=":")
	datRED$ID <- as.factor(datRED$ID)
	datRED <- datRED[order(datRED$ID),]

	## calculate number of unique location-species records per site
	ones <- rep(1,nrow(datRED))
	counts <- bySum(ones,datRED$ID)
	datRED$nRecords <- rep(counts,counts)
	rm(list=c("ones","test"))
	gc()

	## calculate number of unique location-species records per site - excluding unique month-year-lat-long 
	DateLocDups <- paste(datRED$RAW_latdec,datRED$RAW_longdec,datRED$eventDate,sep=":")
	test <- duplicated(DateLocDups)
	recs <- datRED$ID[!test]
	ones <- rep(1,length(recs))
	recCounts <- bySum(ones,recs)
	datRED$nRecords.exDateLocDups <- rep(recCounts,counts)
	rm(list="recs","test","DateLocDups","recCounts","ones")
	gc()

	## calculate number of unique locations records per site
	sites <- paste(datRED$ID,datRED$RAW_latdec,datRED$RAW_longdec,sep=":")
	id <- datRED$ID
	test <-  duplicated(sites)
	sites <- sites[!test]
	id <- id[!test]
	ones <- rep(1,length(id))
	siteCounts <- bySum(ones,id)
	datRED$nSiteVisits <- rep(siteCounts,counts)
	rm(list=c("ones","sites","id","test","siteCounts","counts"))
	gc()

	## reduce to unique site-species-eventDate records
	species <- paste(datRED$ID,datRED$gen_spec,datRED$eventDate,sep=":")
	test <- duplicated(species)
	datRED <- datRED[!test,]
	
	## calculate number of unique species per site
	LocDups <- paste(datRED$ID,datRED$gen_spec,sep=":")
	test <- duplicated(LocDups)
	recs <- datRED$ID[!test]
	ones <- rep(1,length(recs))
	recCounts <- bySum(ones,recs)	
	ones <- rep(1,nrow(datRED))
	counts <- bySum(ones,datRED$ID)
	datRED$richness <- rep(recCounts,counts)
	rm(list=c("ones","counts","test"))
	gc()

	return(datRED)
}
