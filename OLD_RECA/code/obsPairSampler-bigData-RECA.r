
obsPairSampler.bigData.RECA <- function(frog.auGrid,nMatch,m1,richness=TRUE,speciesThreshold=500,coresToUse=detectCores()-1){
	
	m1 <- m1
	
	msg <- data.frame(Type=NA,Acheived=NA,Wanted=NA,Ratio=NA)

	obsMatch <- data.frame(matrix(nrow=nMatch*2,ncol=18))
	names(obsMatch) <- c("observation.numbers","Lon1","Lat1","Lon2","Lat2","Match","Sorenson","Richness.S1","Richness.S2","EventDate1","EventDate2","SharedSpecies","nRecords1","nRecords2","nRecords.exDateLocDups1","nRecords.exDateLocDups2","nSiteVisits1","nSiteVisits2")

	binMatch <- 0
	binMiss <- 0
	attemptedPairs <- NA
	row.count <- 1
	
	## initial pair sampler
	nm <- nMatch
	

	sampIndex <- 1:nrow(frog.auGrid)
	tr <- sampIndex
	s1 <- sample(sampIndex,nm,replace=TRUE)
	samp1 <- tr[s1]
	s2 <- sample(sampIndex,nm,replace=TRUE)
	samp2 <- tr[s2]	


	## remove same site comparisons
#	test <- samp1 != samp2
#	samp1 <- samp1[test]
#	samp2 <- samp2[test]

	## test for and remove duplicates
	obPairs <- rep(NA,length(samp1))
	test <- samp1 - samp2
	obPairs[test >= 0] <- paste(samp2[test >= 0],samp1[test >= 0],sep="~")
	obPairs[test < 0] <- paste(samp1[test < 0],samp2[test < 0],sep="~")
	

	test <- duplicated(obPairs)	
	obPairs <- obPairs[test == FALSE]
	samp1 <- samp1[test == FALSE]
	samp2 <- samp2[test == FALSE]
	attemptedPairs <- obPairs
	

	## populate data.frame
	row.count <- length(obPairs)
	place <- 1:row.count
	obsMatch$observation.numbers[place] <- obPairs
	obsMatch$Lon1[place] <- frog.auGrid$Longitude[samp1]
	obsMatch$Lat1[place] <- frog.auGrid$Latitude[samp1]
	obsMatch$Lon2[place] <- frog.auGrid$Longitude[samp2]
	obsMatch$Lat2[place] <- frog.auGrid$Latitude[samp2]
	match <- frog.auGrid$species[samp1] == frog.auGrid$species[samp2]
	obsMatch$Match[place] <- as.numeric(!match)

	obsMatch$nRecords1[place] <- frog.auGrid$nRecords[samp1]
	obsMatch$nRecords2[place] <- frog.auGrid$nRecords[samp2]
	obsMatch$nRecords.exDateLocDups1[place] <- frog.auGrid$nRecords.exDateLocDups[samp1]
	obsMatch$nRecords.exDateLocDups2[place] <- frog.auGrid$nRecords.exDateLocDups[samp2]
	obsMatch$nSiteVisits1[place] <- frog.auGrid$nSiteVisits[samp1]
	obsMatch$nSiteVisits2[place] <- frog.auGrid$nSiteVisits[samp2]

	obsMatch$EventDate1[place] <- frog.auGrid$eventDate[samp1]
	obsMatch$EventDate2[place] <- frog.auGrid$eventDate[samp2]

#	if(!resample & target.biome){
#		back2tr <- as.numeric(unlist(strsplit(obsMatch$observation.numbers[obsMatch$Match == 0 & !is.na(obsMatch$Match)],split="~")))
#		tr <- c(tr,back2tr)
#	}

	## add richness
	if(richness == TRUE){
		obsMatch$Richness.S1[place] <- frog.auGrid$Site.Richness[samp1]
		obsMatch$Richness.S2[place] <- frog.auGrid$Site.Richness[samp2]
		test <- m1[samp1,] + m1[samp2,]
		test <- test == 2
		if(is.null(nrow(test))){obsMatch$SharedSpecies[place] <- sum(test)}
		if(!is.null(nrow(test))){obsMatch$SharedSpecies[place] <- rowSums(test)}
	}


	binMiss <- length(obsMatch$Match[obsMatch$Match == 1 & !is.na(obsMatch$Match)])

	for(zzz in 1:50){

		print(row.count)
		if(binMiss == nMatch){msg <- rbind(msg,data.frame(Type="Missmatch",Acheived=binMiss,Wanted=nMatch,Ratio=binMiss/nMatch));break}
		##if(length(tr$target.rows) <= 2){msg <- rbind(msg,data.frame(Type="Missmatch",Acheived=binMiss,Wanted=nMatch,Ratio=binMiss/nMatch));message(paste("Unable to fill missmatch quota. ",binMiss," of ",nMatch," observation missmatches found",sep=""));break}
		nm <- nMatch - binMiss


		## iterate until missmatch quota filled


		sampIndex <- 1:nrow(frog.auGrid)
		tr <- sampIndex
		s1 <- sample(sampIndex,nm,replace=TRUE)
		samp1 <- tr[s1]
		s2 <- sample(sampIndex,nm,replace=TRUE)
		samp2 <- tr[s2]	

	
		if(length(samp1) != 0 & length(samp2) != 0){

			## remove same site comparisons
			#test <- samp1 != samp2
			#samp1 <- samp1[test]
			#samp2 <- samp2[test]
	
			## test for and remove duplicates
			obPairs <- rep(NA,length(samp1))
	
			test <- samp1 - samp2
	
			obPairs[test >= 0] <- paste(samp2[test >= 0],samp1[test >= 0],sep="~")
			obPairs[test < 0] <- paste(samp1[test < 0],samp2[test < 0],sep="~")

			ln <- length(attemptedPairs)
			attemptedPairs <- c(attemptedPairs,obPairs)
			test <- duplicated(attemptedPairs)
			test <- test[(ln+1):length(attemptedPairs)]	
			obPairs <- obPairs[test == FALSE]
			samp1 <- samp1[test == FALSE]
			samp2 <- samp2[test == FALSE]
	
	
			len <- length(obPairs)

			if(any(is.na(samp1))){
				print(3)
				print(length(samp1))
				print(length(samp2))
				
				break
			}
			if(length(samp1) != 0 & length(samp2) != 0){
				## populate data.frame
				place <- (row.count+1):(row.count+len)
				obsMatch$observation.numbers[place] <- obPairs
				obsMatch$Lon1[place] <- frog.auGrid$Longitude[samp1]
				obsMatch$Lat1[place] <- frog.auGrid$Latitude[samp1]
				obsMatch$Lon2[place] <- frog.auGrid$Longitude[samp2]
				obsMatch$Lat2[place] <- frog.auGrid$Latitude[samp2]
				match <- frog.auGrid$species[samp1] == frog.auGrid$species[samp2]
				obsMatch$Match[place] <- as.numeric(!match)


				obsMatch$nRecords1[place] <- frog.auGrid$nRecords[samp1]
				obsMatch$nRecords2[place] <- frog.auGrid$nRecords[samp2]
				obsMatch$nRecords.exDateLocDups1[place] <- frog.auGrid$nRecords.exDateLocDups[samp1]
				obsMatch$nRecords.exDateLocDups2[place] <- frog.auGrid$nRecords.exDateLocDups[samp2]
				obsMatch$nSiteVisits1[place] <- frog.auGrid$nSiteVisits[samp1]
				obsMatch$nSiteVisits2[place] <- frog.auGrid$nSiteVisits[samp2]
	
				obsMatch$EventDate1[place] <- frog.auGrid$eventDate[samp1]
				obsMatch$EventDate2[place] <- frog.auGrid$eventDate[samp2]
				
				if(anyNA(obsMatch$EventDate1[place])){
					print(which(is.na(obsMatch$EventDate1[place]))[1])
					print(obsMatch[which(is.na(obsMatch$EventDate1[place]))[1],])
					break				
				}
	
				## add richness
				if(richness == TRUE){
					obsMatch$Richness.S1[place] <- frog.auGrid$Site.Richness[samp1]
					obsMatch$Richness.S2[place] <- frog.auGrid$Site.Richness[samp2]
					test <- m1[samp1,] + m1[samp2,]
					test <- test == 2
					if(length(nrow(test)) == 0){obsMatch$SharedSpecies[place] <- sum(test)
					}else{
					obsMatch$SharedSpecies[place] <- rowSums(test)}
				}
		

				binMiss <- length(obsMatch$Match[obsMatch$Match == 1 & !is.na(obsMatch$Match)])

				#if(!resample & target.biome){
				#	obM <- obsMatch[place,]
				#	back2tr <- as.numeric(unlist(strsplit(obM$observation.numbers[obM$Match == 0 & !is.na(obM$Match)],split="~")))
				#tr <- c(tr,back2tr)
				#	if(length(unique(frog.auGrid$species[tr])) <= 1){msg <- rbind(msg,data.frame(Type="Missmatch",Acheived=binMiss,Wanted=nMatch,Ratio=binMiss/nMatch));message(paste("Unable to fill missmatch quota. ",binMiss," of ",nMatch," observation missmatches found",sep=""));break}
				#}
	
				#if(!resample & !target.biome){
				#	back2tr <- as.numeric(unlist(strsplit(obsMatch$observation.numbers[obsMatch$Match == 0 & !is.na(obsMatch$Match)],split="~")))
				#	TF <- target.specifier[match(back2tr,target.rows)]
				#	back2tr.df <- data.frame(target.rows=back2tr,target.specifier=TF)
				#	tr <- rbind(tr,back2tr.df)
				#	if(length(unique(frog.auGrid$species[tr$target.rows])) <= 1){msg <- rbind(msg,data.frame(Type="Missmatch",Acheived=binMiss,Wanted=nMatch,Ratio=binMiss/nMatch));message(paste("Unable to fill missmatch quota. ",binMiss," of ",nMatch," observation missmatches found",sep=""));break}
	
#				}
	
			}
			row.count <- length(obsMatch$Match[!is.na(obsMatch$Match)])
		}

		#if(!resample & target.biome){
		#if(length(unique(frog.auGrid$species[tr])) <= 1){msg <- rbind(msg,data.frame(Type="Missmatch",Acheived=binMiss,Wanted=nMatch,Ratio=binMiss/nMatch));message(paste("Unable to fill missmatch quota. ",binMiss," of ",nMatch," observation missmatches found",sep=""));break}
		#}

		#if(!resample & !target.biome){
		#if(length(unique(frog.auGrid$species[tr$target.rows])) <= 1){msg <- rbind(msg,data.frame(Type="Missmatch",Acheived=binMiss,Wanted=nMatch,Ratio=binMiss/nMatch));message(paste("Unable to fill missmatch quota. ",binMiss," of ",nMatch," observation missmatches found",sep=""));break}
		#}
	}
	


	binMatch <-  length(obsMatch$Match[obsMatch$Match == 0 & !is.na(obsMatch$Match)])
	ones <- rep(1,nrow(frog.auGrid))
	speciesCounts <- bySum(ones,frog.auGrid$species)

	attemptedPairs <- c()
	message()
	gc()

	for(zzz in 1:50){

		row.count <-  length(obsMatch$Match[!is.na(obsMatch$Match)])
		print(paste("Matches: ",binMatch,sep=""))
		if(binMatch == nMatch){msg <- rbind(msg,data.frame(Type="Match",Acheived=binMatch,Wanted=nMatch,Ratio=binMatch/nMatch));break}
		if(max(speciesCounts) == 1){message(paste("Unable to fill match quota. ",binMatch," of ",nMatch," observation matches found",sep=""));break}
		remaining <- nMatch - binMatch

		#if(!resample & target.biome){
		tr <- sampIndex
		#used <- as.numeric(unlist(strsplit(obsMatch$observation.numbers[obsMatch$Match == 0 & !is.na(obsMatch$Match)],split="~")))
		#if(length(used) > 0){tr <- tr[-match(used,tr)]}
		sampIndex <- 1:length(tr)
		if(length(tr) < floor(remaining*2)){remaining <- floor(length(tr)/2)}
		if(length(sampIndex) > 1){s1 <- sample(sampIndex,remaining,replace=FALSE)}
		if(length(sampIndex) == 1){s1 <- sampIndex}
		if(remaining == 0){s1 <- sample(sampIndex,remaining,replace=FALSE)}

		samp1 <- tr[s1]
		tr <- tr[-s1]
		sampIndex <- 1:length(tr)

		species <- frog.auGrid$species[samp1]

		odr <- order(species)
		samp1 <- samp1[odr]
		species <- species[odr]

		redGrid <- data.table(row.count=tr,species=frog.auGrid$species[tr])
		setkey(redGrid,"species")
	
		unqSpecies <- unique(species)

		if(length(unqSpecies) <= speciesThreshold){
			samp2 <- foreach(x = unqSpecies,.packages="data.table",.combine="rbind") %dopar% {
				dat <- redGrid[J(x)]
				reps <- length(species[species==x])
				nrd <- nrow(dat)
				if(nrd == 1){
					a <- dat$row.count[1]
					if(is.na(a)){dat$row.count <- 0}
				}
				if(nrd > reps){
					a <- sample(dat$row.count,reps,replace=FALSE)
					b <- rep(TRUE,reps)
				}
				if(nrd <= reps){
					if(nrd == 1){a <- dat$row.count}
					if(nrd != 1){a <- sample(dat$row.count,nrd,replace=FALSE)}
					nrem <- reps - nrd
					a <- c(a,rep(-1,nrem))
					b <- rep(c(TRUE,FALSE),c(nrd,nrem))
					b[a == 0] <- FALSE
					a[a == 0] <- -1
				}

				cbind(a,b)


			}
			
			samp1 <- samp1[as.logical(samp2[,2])]
			samp2 <- samp2[as.logical(samp2[,2]),1]
			

		}

		if(length(unqSpecies) > speciesThreshold){
			s1 <- Sys.time()
			ctu <- length(unqSpecies) / coresToUse

			if(ctu >= speciesThreshold){
				CTU <- coresToUse
			}
			
			if(ctu < speciesThreshold){
				CTU <- ceiling(length(unqSpecies) / speciesThreshold)

			}
			
			cl <- makeCluster(CTU)
			registerDoSNOW(cl)

			cs <- ceiling(length(unqSpecies) / getDoParWorkers())
			chunks <- rep(1:getDoParWorkers(),each=cs)
			chunks <- chunks[1:length(unqSpecies)]

			samp2 <- foreach(y = unique(chunks),.packages=c("data.table","foreach"),.combine="rbind") %dopar% {
				
				uS <- unqSpecies[chunks == y]

				s2 <- foreach(x = uS,.packages="data.table",.combine="rbind") %do% {
				dat <- redGrid[J(x)]
				reps <- length(species[species==x])
				nrd <- nrow(dat)
				if(nrd == 1){
					a <- dat$row.count[1]
					if(is.na(a)){dat$row.count <- 0}
				}
				if(nrd > reps){
					a <- sample(dat$row.count,reps,replace=FALSE)
					b <- rep(TRUE,reps)
				}
				if(nrd <= reps){
					if(nrd == 1){a <- dat$row.count}
					if(nrd != 1){a <- sample(dat$row.count,nrd,replace=FALSE)}
					nrem <- reps - nrd
					a <- c(a,rep(-1,nrem))
					b <- rep(c(TRUE,FALSE),c(nrd,nrem))
					b[a == 0] <- FALSE
					a[a == 0] <- -1
				}
				cbind(a,b)

				}
				s2
			}
			Sys.time() - s1
			
			stopCluster(cl)
			
			registerDoSEQ()

			samp1 <- samp1[as.logical(samp2[,2])]
			samp2 <- samp2[as.logical(samp2[,2]),1]
		}
		#}
		
		#stopCluster(cl)
		registerDoSEQ()
	

		if(length(samp1) != 0 & length(samp2) != 0){
			## remove same obs comparisons
			test <- samp1 != samp2
			samp1 <- samp1[test]
			samp2 <- samp2[test]
			## remove duplicates
	
			## test for and remove duplicates
			obPairs <- rep(NA,length(samp1))
			test <- samp1 - samp2
			obPairs[test >= 0] <- paste(samp2[test >= 0],samp1[test >= 0],sep="~")
			obPairs[test < 0] <- paste(samp1[test < 0],samp2[test < 0],sep="~")
			
			ap <- obsMatch$observation.numbers[obsMatch$Match == 0 & !is.na(obsMatch$observation.numbers)]
			ln <- length(ap)
			attemptedPairs <- c(ap,obPairs)
			test <- duplicated(attemptedPairs)
			test <- test[(ln+1):length(attemptedPairs)]	
			obPairs <- obPairs[test == FALSE]
			samp1 <- samp1[test == FALSE]
			samp2 <- samp2[test == FALSE]
						

			## populate data.frame
			row.count <- length(obsMatch$Match[!is.na(obsMatch$Match)])
			if(length(samp1) != 0 & length(samp2) != 0){
				place <- (row.count+1):(row.count+length(obPairs))
				obsMatch$observation.numbers[place] <- obPairs
				obsMatch$Lon1[place] <- frog.auGrid$Longitude[samp1]
				obsMatch$Lat1[place] <- frog.auGrid$Latitude[samp1]
				obsMatch$Lon2[place] <- frog.auGrid$Longitude[samp2]
				obsMatch$Lat2[place] <- frog.auGrid$Latitude[samp2]
				match <- frog.auGrid$species[samp1] == frog.auGrid$species[samp2]
				obsMatch$Match[place] <- as.numeric(!match)

				obsMatch$nRecords1[place] <- frog.auGrid$nRecords[samp1]

				obsMatch$nRecords2[place] <- frog.auGrid$nRecords[samp2]
				obsMatch$nRecords.exDateLocDups1[place] <- frog.auGrid$nRecords.exDateLocDups[samp1]
				obsMatch$nRecords.exDateLocDups2[place] <- frog.auGrid$nRecords.exDateLocDups[samp2]
				obsMatch$nSiteVisits1[place] <- frog.auGrid$nSiteVisits[samp1]
				obsMatch$nSiteVisits2[place] <- frog.auGrid$nSiteVisits[samp2]
				
				obsMatch$EventDate1[place] <- frog.auGrid$eventDate[samp1]
				obsMatch$EventDate2[place] <- frog.auGrid$eventDate[samp2]
				
				if(anyNA(obsMatch$EventDate1[place])){
					print(which(is.na(obsMatch$EventDate1[place]))[1])
					print(obsMatch[which(is.na(obsMatch$EventDate1[place]))[1],])
					break				
				}

				## add richness
				if(richness == TRUE){
					obsMatch$Richness.S1[place] <- frog.auGrid$Site.Richness[samp1]
					obsMatch$Richness.S2[place] <- frog.auGrid$Site.Richness[samp2]
					test <- m1[samp1,] + m1[samp2,]
					test <- test == 2
					if(length(nrow(test)) == 0){
						obsMatch$SharedSpecies[place] <- sum(test)
					}else{
						obsMatch$SharedSpecies[place] <- rowSums(test)
					}

				}
			}
		}

		binMatch <- length(obsMatch$Match[obsMatch$Match == 0 & !is.na(obsMatch$Match)])

		#if(!resample & target.biome){
		#	tr <- target.rows
		#	used <- as.numeric(unlist(strsplit(obsMatch$observation.numbers[obsMatch$Match == 0 & !is.na(obsMatch$Match)],split="~")))
		#	tr <- tr[-match(used,tr)]
		#	ones <- rep(1,length(tr))
		#	speciesCounts <- bySum(ones,frog.auGrid$species[tr])
		#	assign("SC.test",speciesCounts,pos=.GlobalEnv)
		#}

	}

	gc()
 	obsMatch <- obsMatch[!is.na(obsMatch$Lat1),]
	m01 <- unlist(lapply(c(0,1),function(x){nrow(obsMatch[obsMatch$Match == x,])}))
	msg <- data.frame(Target=TRUE,Type=c("Missmatch","Match"),Wanted=nMatch,Achieved=m01,Ratio=m01/nMatch)
	assign("samplesAcheived",msg,pos=.GlobalEnv)
	message("n samples achieved diagnostics saved in samplesAcheived") 
	rm(attemptedPairs)
	
	## format for dynowindow
	year1 <- as.numeric(substr(obsMatch$EventDate1,1,4))
	month1 <- as.numeric(substr(obsMatch$EventDate1,6,7))
	year2 <- as.numeric(substr(obsMatch$EventDate2,1,4))
	month2 <- as.numeric(substr(obsMatch$EventDate2,6,7))
	obsMatch <- cbind(obsMatch[,1:3],year1,month1,obsMatch[,4:5],year2,month2,obsMatch[,6:ncol(obsMatch)])
	
	return(obsMatch)
}