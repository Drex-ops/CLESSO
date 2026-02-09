#######
##
## Make obs-pair sampled set
##
#######


getSamples_AND_Env_RECA <- function(datRED=datRED,nMatch=NULL,biol_group=NA,gen_windows2=gen_windows,exe2=exe){

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
		list(variables = c("Precip_191101-201712"),
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
		out <- gen_windows2(pairs=ext_data,
		variables = init_params[[x]]$variables,
		mstat =  init_params[[x]]$mstat, 
		cstat =  init_params[[x]]$cstat, 
		window =  init_params[[x]]$window,
		exe=exe2
		)
		colnames(out) <- paste(init_params[[x]]$prefix,colnames(out),sep="_")
		out[,9:ncol(out)]
	}
	stopCluster(cl)
	registerDoSEQ()
	
print("D")
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
							env_out1,env1_subs,Trng_1,
							env_out2,env2_subs,Trng_2)

	## clean up
	rm(ext_data)
	rm(frog.auGrid)
	rm(data)
	rm(m1)
	rm(list("env_out","env_out1","env_out2","env1_subs","env2_subs","Trng_1","Trng_2"))
	gc()
	
	## save
	fn=paste("ObsPairsTable_RECA_",biol_group,".RData",sep="")
	save(obsPairs_out,file=fn)
	#write.csv()
	return(obsPairs_out)
}

