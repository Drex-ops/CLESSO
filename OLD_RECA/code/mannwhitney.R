


MW_U <- function(x, y, ...){ 
  ## Calculate Mann Whitney u statistic
  ## Hanley and McNeil 1982 example data:
  ## x <- c(rep(1, 33), rep(2, 6), rep(3, 6), rep(4, 11), rep(5, 2))
  ## y <- c(rep(1, 3), rep(2, 2), rep(3, 2), rep(4, 11), rep(5, 33))
  ## Test data:
  ## x <- test_x; y <- test_y
  
  ## Configure data counts
  if(length(x) == length(y)){
    dat <- data.frame(x = x, y = y)
    match_mismatch <- table(dat)
    binX <- as.numeric(match_mismatch[,1])
    binY <- as.numeric(match_mismatch[,2])
  } else { # assumes x,y are counts
    binX <- as.numeric(table(x))
    binY <- as.numeric(table(y))
  }
  
  nN <- sum(binY)
  nA <- 0
  
  aX <- list()
  for (i in seq_along(binY)){
    aX[[i]] <- nN-binY[i]
    ## update nN
    nN <- aX[[i]]
  }
  aX <- unlist(aX)
  
  aY <- list(nA)
  for(i in 2:length(binX)){
    aY[[i]] <- aY[i-1][[1]]+binX[i-1]
  }
  aY <- unlist(aY)
  
  ## from Hanley and McNeil 1982
  ## (1) X (2) + 1/2 X (1) X (3) 
  statistic <- sum(binX * aX + 0.5*(as.numeric(binX) * as.numeric(binY)))
  W <- statistic / (sum(as.numeric(binX)) * sum(as.numeric(binY)))
  theta <- W 
  Q1_tot <- sum(binY * (aY^2 + aY * binX + (1/3) * binX^2))
  Q2_tot <- sum(binX * (aX^2 + aX * binY + (1/3) * binY^2))
  Q1 <- Q1_tot / (sum(binY) * sum(binX)^2)
  Q2 <- Q2_tot / (sum(binX) * sum(binY)^2)
  
  SE <- sqrt((theta * (1-theta) + (sum(binY) - 1) * (Q1 - theta^2) + (sum(binX) -1) * (Q2 - theta^2)) / 
               (sum(binY) * sum(binX)))
  
  out <- list()
  out$Statistic <- statistic
  out$W <- W
  out$SE <- SE
  out$explanation <- 'Where [1] is the Mann-Whitney U statistic, [2] is  the W/AUC, and [3] is +SE(W)'
  
  return(out)
}


setwd("R:\\DEV\\hos06j\\")

if(TRUE){

files <- list.files(pattern="fittedGDM")
files <- files[-grep("AVESg1995_",files)]
files <- files[-grep("biAverage",files)]
files <- files[-grep("run",files)]

splt <- unlist(strsplit(files,split="_"))

clim <- splt[seq(3,length(splt),by=6)]
weath <- splt[seq(4,length(splt),by=6)]

clim <- as.numeric(gsub("climYr","",clim))
weath <- as.numeric(gsub("weathYr","",weath))

mwU <- c()

for(i in 1:length(files)){

	load(files[i])
	load(gsub("_fittedGDM","ObsEnvTable",files[i]))
	## temporalily remove NAs - this to be fixed in getSamples before sampling
	test <- is.na(rowSums(obsPairs_out[,23:96]))
	obsPairs_out <- obsPairs_out[!test,]

	## there's NAflags present (-9999). Find and remove
	tst <- c()
	for(c in 23:96){
		if(any(obsPairs_out[,c] == -9999)){
		tst <- c(tst,c);print(c)}
		}
	test <- rep(0,nrow(obsPairs_out))
	for(c in 23:96){
		test <- test + (obsPairs_out[,c] == -9999)
	}
	test <- test == 0

	obsPairs_out <- obsPairs_out[test,]
	mw <- MW_U(fit$predicted,obsPairs_out$Match)$W
	mwU <- c(mwU,mw)

}


setwd("R:\\DEV\\hos06j\\")

files <- list.files(pattern="fittedGDM")
files <- files[-grep("AVESg1995_",files)]
files <- files[-grep("run",files)]
files <- files[grep("biAverage",files)]


splt <- unlist(strsplit(files,split="_"))

clim_ba <- splt[seq(3,length(splt),by=7)]
weath_ba <- splt[seq(4,length(splt),by=7)]

clim_ba <- as.numeric(gsub("climYr","",clim_ba))
weath_ba <- as.numeric(gsub("weathYr","",weath_ba))

mwU_ba <- c()

for(i in 1:length(files)){

	load(files[i])
	load(gsub("_fittedGDM","ObsEnvTable",files[i]))
	## temporalily remove NAs - this to be fixed in getSamples before sampling
	test <- is.na(rowSums(obsPairs_out[,23:96]))
	obsPairs_out <- obsPairs_out[!test,]

	## there's NAflags present (-9999). Find and remove
	tst <- c()
	for(c in 23:96){
		if(any(obsPairs_out[,c] == -9999)){
		tst <- c(tst,c);print(c)}
		}
	test <- rep(0,nrow(obsPairs_out))
	for(c in 23:96){
		test <- test + (obsPairs_out[,c] == -9999)
	}
	test <- test == 0

	obsPairs_out <- obsPairs_out[test,]
	mw <- MW_U(fit$predicted,obsPairs_out$Match)$W
	mwU_ba <- c(mwU_ba,mw)

}


}