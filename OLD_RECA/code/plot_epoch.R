library(ggplot2)

## plot 

setwd("R:\\DEV\\hos06j\\")
load("274845dissimilarities_epoch.RData")

mns <- c()
q25 <- c()
q97 <- c()

#dissimilarities_epoch2 <- dissimilarities_epoch[,-ncol(dissimilarities_epoch)]
dissimilarities_epoch2 <- dissimilarities_epoch

for(c in 1:ncol(dissimilarities_epoch2)){

	qs <- quantile(dissimilarities_epoch2[,c],c(0.025,0.5,0.975))
	mns <- c(mns,qs[2])
	q25 <- c(q25,qs[1])
	q97 <- c(q97,qs[3])

}

yrs <- 1966:2016

data <- data.frame(yrs,mns,q25,q97)

spline_int <- as.data.frame(spline(data$yrs, data$mns))


png('AVES_dissimilarity_epoch65.png',width=6,height=4,units='in',res=400)
cl <- "sky blue"
p<-ggplot(data=data, aes(x=yrs, y=mns))  + geom_line(data = spline_int, aes(x = x, y = y),col=cl,lwd=1.5) + geom_point()
p<-p+geom_ribbon(aes(ymin=data$q25, ymax=data$q97), linetype=2, alpha=0.3,fill=cl) + ylab('Dissimilarity') + xlab('') +  
		geom_line(data=data,aes(x=yrs,y=q25),lty='dashed',col=cl) + geom_line(data=data,aes(x=yrs,y=q97),lty='dashed',col=cl) 
p
dev.off()

+ geom_line(data=data,x=yrs,y=q97,lty='dashed',col=cl)
for(r in sample(1:nrow(dissimilarities_epoch),2500)){



}