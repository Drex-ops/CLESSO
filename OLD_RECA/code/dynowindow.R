#' @title Obs-pair climate window builder thing 
#' 
#' @description Generate climate window summary variables for obs-pair input
#' 
#' @param pairs str(filepath) or data.frame like object. Obs pair table with year and month cols (x1, y1, year1, mon1, x2, y2 etc).
#' @param variables vector of variable names. Variable names must be file names without an extension.
#' @param mstat str, statistic to summarize monthly variables to a notional year.
#' @param cstat str, statistic to summarize yearly variables over window period.
#' @param window int, length of climate window in years.
#' @return data.frame
#' 
#'@importFrom feather write_feather read_feather
#'@export
gen_windows = function(pairs, variables, mstat, cstat, window, 
                       pairs_dst = NULL, npy_src = NULL, start_year = 1911){
  
  if(!exists('exe')){
    stop(sprintf('%s %s', 
                 'Cannot locate a version of python.', 
                 'A variable exe needs to be available pointing to a python .exe'))
  }
  
  type_pairs = class(pairs)
  if(type_pairs != 'character'){
    
    # data.frame like
    
    if(is.null(pairs_dst)){
      pairs_dst = tempfile(tmpdir="\\\\ces-10-cdc.it.csiro.au\\OSM_CDC_LW_GPAAG_scratch\\rtmp",fileext = '.feather')
	  print(pairs_dst)
    }  
    
    col_class = rep(c('numeric', 'numeric', 'integer', 'integer'), 2)
    for (c in 1:8){
      class(pairs[, c]) = col_class[c]
    }
    
    min_year = min(c(pairs[, 3], pairs[, 7]))
    if ((min_year - window) < start_year){
      
      # 
      stop(sprintf('Found year: %s. Cannot build climate windows for period before %s', 
                   min_year, start_year))
      
    }
    
    write_feather(pairs, pairs_dst)
    
  } else {
    
    pairs_dst = pairs
  }
  
  pyfile =  "\\\\lw-osm-02-cdc\\OSM_CBR_LW_BACKCAST_work\\DEV\\hos06j\\code\\pyper.py"##paste(.libPaths(), 'dynowindow/exec/pyper.py', sep = '/') ##
  if(Sys.info()['sysname'] == 'Windows'){
    pyfile = gsub('/', '\\\\', pyfile)
  }
  
  variables = paste(variables, collapse = ' ')
  
  call = sprintf('%s "%s" -f %s -s %s -m %s -e %s -w %s', 
                 exe, pyfile, pairs_dst, cstat, mstat, variables, window)
  
  if(!is.null(npy_src)){
    call = sprintf('%s -src %s', call, npy_src)
  }
  
  # pass to python
  output_fp = system(call, intern = TRUE)
  
  # ...?
  output = tryCatch({
    read_feather(output_fp)
  }, error = function(e){e}
  )
  
  if(!length(grep('data.frame', class(output)))){
	print(class(output))
    stop(sprintf('Could not read output file. Error: %s', output_fp))
    
  } else {
  
	file.remove(output_fp)
	file.remove(pairs_dst)
    return(as.data.frame(output))
  }
  
}
