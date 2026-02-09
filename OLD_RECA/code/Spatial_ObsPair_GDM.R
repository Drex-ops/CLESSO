library(ffbase)
library(raster)
library(dynowindow)
library(feather)
library(Gdm01)
library(SDMTools)
library(raster)
library(nnls)
library(lubridate)

source("//ces-10-cdc/OSM_CDC_MMRG_work/users/GPAAG_refuge/BIOL/G/DEV/_tools/ObsPairSampler/site-richness-extractor-bigData.R")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/obsPairSampler-bigData-RECA.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/siteAggregator.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/dynowindow.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/getSamples_AND_ENV.r")
source("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/code/obs.gdm.plot_logit.R")
source("Y:\\MOD\\G\\GDM\\_tools\\create_5splineOutput2.R")
source("Y:\\MOD\\G\\GDM\\_tools\\gdm.spline.plot.R")
source("U:\\users\\hos06j\\R scripts\\nnnpls.fit2_(good).R")
source("U:\\users\\hos06j\\R scripts\\negexp_GDMlink.R")
source("U:\\users\\hos06j\\R scripts\\fitGDM.R")

RsqGLM <- function(obs = NULL, pred = NULL, model = NULL) {
  # version 1.2 (3 Jan 2015)
 
  model.provided <- ifelse(is.null(model), FALSE, TRUE)
 
  if (model.provided) {
    if (!("glm" %in% class(model))) stop ("'model' must be of class 'glm'.")
    if (!is.null(pred)) message("Argument 'pred' ignored in favour of 'model'.")
    if (!is.null(obs)) message("Argument 'obs' ignored in favour of 'model'.")
    obs <- model$y
    pred <- model$fitted.values
 
  } else { # if model not provided
    if (is.null(obs) | is.null(pred)) stop ("You must provide either 'obs' and 'pred', or a 'model' object of class 'glm'")
    if (length(obs) != length(pred)) stop ("'obs' and 'pred' must be of the same length (and in the same order).")
    #if (!(obs %in% c(0, 1)) | pred < 0 | pred > 1) stop ("Sorry, 'obs' and 'pred' options currently only implemented for binomial GLMs (binary response variable with values 0 or 1) with logit link.")
    logit <- log(pred / (1 - pred))
    model <- glm(obs ~ logit, family = "binomial")
  }
 
  null.mod <- glm(obs ~ 1, family = family(model))
  loglike.M <- as.numeric(logLik(model))
  loglike.0 <- as.numeric(logLik(null.mod))
  N <- length(obs)
 
  # based on Nagelkerke 1991:
  CoxSnell <- 1 - exp(-(2 / N) * (loglike.M - loglike.0))
  Nagelkerke <- CoxSnell / (1 - exp((2 * N ^ (-1)) * loglike.0))
 
  # based on Allison 2014:
  McFadden <- 1 - (loglike.M / loglike.0)
  Tjur <- mean(pred[obs == 1]) - mean(pred[obs == 0])
  sqPearson <- cor(obs, pred) ^ 2
 
  return(list(CoxSnell = CoxSnell, Nagelkerke = Nagelkerke, McFadden = McFadden, Tjur = Tjur, sqPearson = sqPearson))
}


## inverse link 
inv.logit <- function(x){exp(x)/(1+exp(x))}

## transform obs to diss
ObsTrans <- function(p0,w,p){

	prw <- (p*w) / ((1-p) + (p*w))

	p0w <- (p0*w) / ((1-p0) + (p0*w))

	out <- 1 - ((1-prw) / (1-p0w))
	return(list(prw=prw,out=out))
}

## fit obs GDM
fitGDM <- function(formula=NULL,data=NULL){

	fit <- glm(formula,family=binomial(),data=data,control=list(maxit=500),method='nnls.fit')

	return(fit)
}

rasterOptions(tmpdir="S:\\rasterTemp")



## run straight spatial obs GDM

## make brick of env data
setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/FLT_1945_1975/")

files <- list.files(pattern=".flt")
env_idx <- c(20,17,28,55,24,35,53,15,64,57,1,26,11,14,27,21,49,45)
files <- files[env_idx]

stk <- stack()
for(f in files){
	ras <- raster(f)
	stk <- stack(stk,ras)
}
brk <- brick(stk,file="//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/SpatialObsGDM_env.grd")


setwd("//lw-osm-02-cdc/OSM_CBR_LW_BACKCAST_work/DEV/hos06j/")
biol_group="AVES"
fn=paste("ObsPairsTable_RECA_",biol_group,"WindowTestRuns.RData",sep="")
load(fn)
ext_data <- obsPairs_out[,2:9]

site1 <- extract(brk,ext_data[,1:2])

site2  <- extract(brk,ext_data[,5:6])


toSpline <- cbind(site1,site2)
splined <- splineData(toSpline)



