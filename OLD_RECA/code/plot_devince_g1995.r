## Plot change in deviance

setwd("R:\\DEV\\hos06j\\")

files <- list.files(pattern="DevianceCalcs")
files <- files[grep("AVESg1995",files)]

splt <- unlist(strsplit(files,split="_"))

clim <- splt[seq(3,length(splt),by=5)]
weath <- splt[seq(4,length(splt),by=5)]

clim <- as.numeric(gsub("climYr","",clim))
weath <- as.numeric(gsub("weathYr","",weath))

dev <- c()

for(i in 1:length(files)){

	load(files[i])
	dev <- c(dev,gdm_dev$Nagelkerke)

}


plot(1,ylim=range(dev),xlim=c(10,65),type="n",xlab="Climate Window (yrs)",ylab="Performance")

for(w in unique(weath)){

	x <- clim[weath == w]
	y <- dev[weath == w]
	
	odr <- order(x)
	
	x <- x[odr]
	y <- y[odr]
	
	lines(x,y,col=w)
}

#legend('topleft',legend=c(1,2,3),col=unique(weath),lty=1,title="Weather window (yrs)")