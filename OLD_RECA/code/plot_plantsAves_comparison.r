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

dev_p <- dev
clim_p <- clim

## Plot change in deviance

setwd("R:\\DEV\\hos06j\\")

files <- list.files(pattern="DevianceCalcs")
files <- files[grep('V3',files)]
files <- files[-grep("biAverage",files)]
files <- files[-grep("15weathYr",files)]


splt <- unlist(strsplit(files,split="_"))

clim <- splt[seq(4,length(splt),by=6)]
weath <- splt[seq(5,length(splt),by=6)]

clim <- as.numeric(gsub("climYr","",clim))
weath <- as.numeric(gsub("weathYr","",weath))

dev <- c()

for(i in 1:length(files)){

	load(files[i])
	dev <- c(dev,gdm_dev$Nagelkerke)

}

normalized = function(x){(x-min(x))/(max(x)-min(x))}


plot(1,ylim=range(0,1),xlim=c(5,75),type="n",xlab="Climate Window (yrs)",ylab="Performance")
	lines(x,normalized(y),col="green")
	lines(x,normalized(y),col="red")


	x_1 <- clim[weath == 1]
	y_1 <- dev[weath == 1]	
	odr <- order(x_1)	
	x_1 <- x_1[odr]
	y_1 <- y_1[odr]
	
	x_2 <- clim_p
	y_2 <- dev_p
	odr <- order(x_2)	
	x_2 <- x_2[odr]
	y_2 <- y_2[odr]
	y_2 <- y_2[x_2 < 61]
	x_2 <- x_2[x_2 < 61]
	
	
	
	
	
spline_int_1 <- as.data.frame(spline(x_1, normalized(y_1)))
spline_int_2 <- as.data.frame(spline(x_2, normalized(y_2)))

data2 <- rbind(spline_int_1,spline_int_2)
data2$ID <- c(rep("Birds",nrow(spline_int_1)),rep("Plants",nrow(spline_int_2)))


png('PLANTS_AVES_performance.png',width=6,height=4,units='in',res=400)
cl <- c('sky blue',"forest green")
p<-ggplot(data=data2, aes(x=x, y=y,col=ID))  + geom_line(lwd=1.2) + scale_color_manual(values=cl) + 
 theme(
  axis.text.y = element_blank(),
  axis.ticks = element_blank()) + xlab('Climate window (yrs)') + ylab('Performance') #+
  ##scale_linetype_manual(values=c('solid','dotted'),labels=c("Spatio-temporal", "Spatial"))#+ theme(legend.position='top')
  p$labels$linetype <- 'Biological\ngroup type'
   p$labels$colour <- 'Biological\ngroup type'
  p
  ##scale_linetype_discrete(name='Plant\nmodel type') 
dev.off()

data2 <- rbind(spline_int_1)
data2$ID <- c(rep("Birds",nrow(spline_int_1)))


png('AVES_performance.png',width=6,height=4,units='in',res=400)
cl <- c('sky blue')
p<-ggplot(data=data2, aes(x=x, y=y,col=ID))  + geom_line(lwd=1.2) + scale_color_manual(values=cl) + 
 theme(
  axis.text.y = element_blank(),
  axis.ticks = element_blank()) + xlab('Climate window (yrs)') + ylab('Performance') #+
  ##scale_linetype_manual(values=c('solid','dotted'),labels=c("Spatio-temporal", "Spatial"))#+ theme(legend.position='top')
  p$labels$linetype <- 'Biological\ngroup type'
   p$labels$colour <- 'Biological\ngroup type'
  p
  ##scale_linetype_discrete(name='Plant\nmodel type') 
dev.off()


data2 <- rbind(spline_int_2)
data2$ID <- c(rep("Plants",nrow(spline_int_2)))


png('Plants_performance2.png',width=6,height=4,units='in',res=400)
cl <- c('forest green')
p<-ggplot(data=data2, aes(x=x, y=y,col=ID))  + geom_line(lwd=1.2) + scale_color_manual(values=cl) + 
 theme(
  axis.text.y = element_blank(),
  axis.ticks = element_blank()) + xlab('Climate window (yrs)') + ylab('Performance') #+
  ##scale_linetype_manual(values=c('solid','dotted'),labels=c("Spatio-temporal", "Spatial"))#+ theme(legend.position='top')
  p$labels$linetype <- 'Biological\ngroup type'
   p$labels$colour <- 'Biological\ngroup type'
  p
  ##scale_linetype_discrete(name='Plant\nmodel type') 
dev.off()
	