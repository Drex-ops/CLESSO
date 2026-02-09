## Plot change in deviance

setwd("R:\\DEV\\hos06j\\PLANTS")

files <- list.files(pattern="DevianceCalcs")
files <- files[grep("PLANTS_",files)]
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


setwd("R:\\DEV\\hos06j\\PLANTS")

files <- list.files(pattern="DevianceCalcs")
files <- files[grep("PLANTS_",files)]
files <- files[grep("biAverage",files)]

splt <- unlist(strsplit(files,split="_"))

clim_ba <- splt[seq(4,length(splt),by=6)]
weath_ba <- splt[seq(5,length(splt),by=6)]

clim_ba <- as.numeric(gsub("climYr","",clim_ba))
weath_ba <- as.numeric(gsub("weathYr","",weath_ba))

dev_ba <- c()

for(i in 1:length(files)){

	load(files[i])
	dev_ba <- c(dev_ba,gdm_dev$Nagelkerke)

}


plot(1,ylim=range(c(dev)),xlim=c(0,75),type="n",xlab="Climate Window (yrs)",ylab="Performance")

	x <- clim[weath == 1]
	y <- dev[weath == 1]	
	odr <- order(x)	
	x <- x[odr]
	y <- y[odr]
	lines(x,y,col="green")

	x <- clim_ba
	y <- dev_ba
	odr <- order(x)	
	x <- x[odr]
	y <- y[odr]
	lines(x,y,col="blue")
	
x_1 <- clim[weath == 1]
y_1 <- dev[weath == 1]	
odr <- order(x_1)	
x_1 <- x_1[odr]
y_1 <- y_1[odr]

x_2 <- clim_ba
y_2 <- dev_ba
odr <- order(x_2)	
x_2 <- x_2[odr]
y_2 <- y_2[odr]
	
data <- data.frame(x=c(x_1,x_2),y=c(y_1,y_2),ID=c(rep("Spatio-temporal",length(x_1)),rep("Spatial",length(x_2))))


spline_int_1 <- as.data.frame(spline(x_1, y_1))
spline_int_2 <- as.data.frame(spline(x_2, y_2))

data2 <- rbind(spline_int_1,spline_int_2)
data2$ID <- c(rep("Spatial",nrow(spline_int_1)),rep("Spatio-temporal",nrow(spline_int_1)))


png('PLANTS_performance.png',width=6,height=4,units='in',res=400)
cl <- c("forest green")
p<-ggplot(data=data2, aes(x=x, y=y,col=rev(ID)))  + geom_line(aes(linetype=rev(ID)),lwd=1.2) + scale_color_manual(values=c(cl,cl)) + 
 theme(
  axis.text.y = element_blank(),
  axis.ticks = element_blank()) + xlab('Climate window (yrs)') + ylab('Performance') #+
  ##scale_linetype_manual(values=c('solid','dotted'),labels=c("Spatio-temporal", "Spatial"))#+ theme(legend.position='top')
  p$labels$linetype <- 'Plant\nmodel type'
   p$labels$colour <- 'Plant\nmodel type'
  p
  ##scale_linetype_discrete(name='Plant\nmodel type') 
dev.off()
p

col=cl,lty=c('solid')
aes(linetype=rev(ID)),lwd=1.2

+ geom_line(data = spline_int_1, aes(x = x, y = y),col=cl,lwd=1.5) +
		geom_line(data = spline_int_2, aes(x = x, y = y),col=cl,lwd=1.5,lty='dashed')##+ geom_point()

p<-p + ylab('Performance') + xlab('Climate window (yrs)') + theme(axis.text.y = element_blank())
p
dev.off()

	
p <- ggplot(


for(w in unique(weath)){

	x <- clim[weath == w]
	y <- dev[weath == w]
	
	odr <- order(x)
	
	x <- x[odr]
	y <- y[odr]
	
	lines(x,y,col=w)
}

legend('topleft',legend=c(1,2,3),col=unique(weath),lty=1,title="Weather window (yrs)")