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
      pairs_dst = tempfile(fileext = '.feather')
    }  
    
    col_class = rep(c('numeric', 'numeric', 'integer', 'integer'), 2)
    for (i in 1:8){
      class(pairs[, i]) = col_class[i]
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
  
  pyfile = paste(.libPaths(), 'dynowindow/exec/pyper.py', sep = '/')
  if(Sys.info()['sysname'] == 'Windows'){
    pyfile = gsub('/', '\\\\', pyfile)
  }
  
  variables = paste(variables, collapse = ' ')
  
  call = sprintf('%s "%s" -f %s -s %s -m %s -e %s -w %s', 
                 exe, pyfile, pairs_dst, mstat, cstat, variables, window)
  
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
    stop(sprintf('Could not read output file. Error: %s', output_fp))
    
  } else {
    
    return(as.data.frame(output))
  }
  
}

# exe = 'C:\\Users\\war42q\\AppData\\Local\\Continuum\\Anaconda3\\pyt