## Plot change in deviance

setwd("R:\\DEV\\hos06j\\")

files <- list.files(pattern="DevianceCalcs")
files <- files[-grep("Spatial_ObsPair_1945_75_ DevianceCalcs.RData",files)]
files <- files[-grep("AVESg1995_",files)]
files <- files[-grep("biAverage",files)]

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


setwd("R:\\DEV\\hos06j\\")

files <- list.files(pattern="DevianceCalcs")
files <- files[-grep("Spatial_ObsPair_1945_75_ DevianceCalcs.RData",files)]
files <- files[-grep("AVESg1995_",files)]
files <- files[grep("biAverage",files)]

splt <- unlist(strsplit(files,split="_"))

clim_ba <- splt[seq(3,length(splt),by=6)]
weath_ba <- splt[seq(4,length(splt),by=6)]

clim_ba <- as.numeric(gsub("climYr","",clim_ba))
weath_ba <- as.numeric(gsub("weathYr","",weath_ba))

dev_ba <- c()

for(i in 1:length(files)){

	load(files[i])
	dev_ba <- c(dev_ba,gdm_dev$Nagelkerke)

}


plot(1,ylim=range(dev),xlim=c(5,67),type="n",xlab="Climate Window (yrs)",ylab="Performance")

	x <- clim[weath == 1]
	y <- dev[weath == 1]	
	odr <- order(x)	
	x <- x[odr]
	y <- y[odr]
	lines(x,y,col="red")

	x <- clim_ba
	y <- dev_ba
	odr <- order(x)	
	x <- x[odr]
	y <- y[odr]
	lines(x,y,col="blue")


for(w in unique(weath)){

	x <- clim[weath == w]
	y <- dev[weath == w]
	
	odr <- order(x)
	
	x <- x[odr]
	y <- y[odr]
	
	lines(x,y,col=w)
}

legend('topleft',legend=c(1,2,3),col=unique(weath),lty=1,title="Weather window (yrs)")