make_jobs <- function(path_training_controls, overwrite_batch = TRUE) {

  # read in study specific controls
  source(path_training_controls)
  
  # relative paths should work from any repo project if a local copy of lab_support exists
  path_chtc <- "../lab_support/chtc/static_files"
  
  
  # Get split indices from cv resample parameters
  if (cv_resample_type == "boot") {
    split_num <- 1:cv_resample 
    # set nested split_num parameters to NA
    outer_split_num <- NA
    inner_split_num <- NA
  }
  
  if (cv_resample_type == "kfold") {
    n_repeats <- as.numeric(str_remove(cv_resample, "_x_\\d{1,2}"))
    n_folds <- as.numeric(str_remove(cv_resample, "\\d{1,3}_x_"))
    
    split_num <- 1:(n_repeats * n_folds)
    
    # set nested split_num parameters to NA
    outer_split_num <- NA
    inner_split_num <- NA
  }
  
  if (cv_resample_type == "nested") {
    # set split_num to NA and use outer_split_num and inner_split_num
    split_num <- NA
    
    # outer cv loop - always will be kfold
    outer_n_repeats <- as.numeric(str_remove(cv_outer_resample, "_x_\\d{1,2}"))
    outer_n_folds <- as.numeric(str_remove(cv_outer_resample, "\\d{1,3}_x_"))
    
    outer_split_num <- 1:(outer_n_repeats * outer_n_folds)
    
    # inner cv loop - can be kfold or bootstrap
    if (str_detect(cv_inner_resample, "_x_")) {
      inner_n_repeats <- as.numeric(str_remove(cv_inner_resample, "_x_\\d{1,2}"))
      inner_n_folds <- as.numeric(str_remove(cv_inner_resample, "\\d{1,3}_x_"))
      
      inner_split_num <- 1:(inner_n_repeats * inner_n_folds)
    } 
    
    if (!str_detect(cv_inner_resample, "_x_")) {
      inner_split_num <- 1:cv_inner_resample
    }
  }
  
  configs <- tibble(split_num, outer_split_num, inner_split_num)
  
  # create new batch directory (if it does not already exist) 
  if (!dir.exists(file.path(path_batch))) {
    dir.create(file.path(path_batch))
    dir.create(file.path(path_batch, "input"))
    dir.create(file.path(path_batch, "output"))
  } else {
    stop("Batch folder already exists. No new folders created. Set overwrite_batch = TRUE to write over existing batch.")
  }
  
  # write jobs file to input folder
  configs %>% 
    write_csv(file.path(path_batch, "input", "configs.csv"))
  
  
  # create job_nums.csv file  
  configs |> 
    tibble::rownames_to_column("job_num") |> 
    mutate(job_num = as.numeric(job_num)) |> 
    select(job_num) |> 
    write_csv(file.path(path_batch, "input", "job_nums.csv"), 
              col_names = FALSE)
  
  # copy data to input folder as data_trn 
  chunks <- str_split_fixed(data_trn, "\\.", n = Inf) # parse name from extensions
  if (length(chunks) == 2) {
    fn <- str_c("data_trn.", chunks[[2]])
  } else {
    fn <- str_c("data_trn.", chunks[[2]], ".", chunks[[3]])
  }
  check_copy <- file.copy(from = file.path(path_data, data_trn),
                          to = file.path(path_batch, "input", fn),
                          overwrite = overwrite_batch)
  if (!check_copy) {
    stop("data_trn not copied to input folder. Check path_data and data_trn (file name) in training controls.")
  }


  # copy study specific training_controls to input folder 
  check_copy <- file.copy(from = file.path(path_training_controls),
                          to = file.path(path_batch, "input", "training_controls.R"),
                          overwrite = overwrite_batch) 
  if (!check_copy) {
    stop("Training controls not copied to input folder. Check path_training_controls in mak_jobs.")
  }
  
  # copy train.sh to input folder 
  check_copy <- file.copy(from = here::here("./_chtc/lasso/train.sh"),
                          to = file.path(path_batch, "input", "train.sh"),
                          overwrite = overwrite_batch) 
  if (!check_copy) {
    stop("train.sh not copied to input folder.")
  }

  # copy fun_chtc.R to input folder 
  check_copy <- file.copy(from = here::here(path_chtc, "fun_chtc.R"),
                          to = file.path(path_batch, "input", "fun_chtc.R"),
                          overwrite = overwrite_batch) 
  if (!check_copy) {
    stop("fun_cthc.R not copied to input folder.")
  }

  # copy fit_chtc.R to input folder 
  check_copy <- file.copy(from = here::here("./_chtc/lasso/fit_chtc.R"),
                          to = file.path(path_batch, "input", "fit_chtc.R"),
                          overwrite = overwrite_batch) 
  if (!check_copy) {
    stop("fit_chtc.R not copied to input folder.")
  }
  
  # create submit file from training controls -----------------
  write("#train.sub", 
        file.path(path_batch, "input", "train.sub"))
  
  # set staging directory
  singularity <- str_c('+SingularityImage = "osdf:///chtc/staging/', username, '/train.sif"')
  container <- str_c("container_image = osdf:///chtc/staging/", username, "/train.sif")
  
  write(c(singularity, 
          container,
          "executable = train.sh",
          "arguments = job_nums.csv $(job_num)",
          "  ",
          "log = $(Cluster).log",
          "error = error/error_$(job_num).err", 
          "  ",
          "should_transfer_files = YES",
          "when_to_transfer_output = ON_EXIT",
          'transfer_output_remaps = "results_$(job_num).csv = results/results_$(job_num).csv"',
          "on_exit_hold = exitcode != 0",
          "max_retries = 1"), 
        file.path(path_batch, "input", "train.sub"), append = TRUE)
  
  # add files to transfer
  if(stage_data == FALSE) {
    transfer_files_str <- str_c("transfer_input_files = fun_chtc.R, fit_chtc.R, training_controls.R, configs.csv, job_nums.csv, osdf:///chtc/staging/", username, "/train.sif,", fn)
  } 
  
  if(stage_data == TRUE) {
    transfer_files_str <- str_c("transfer_input_files = fun_chtc.R, fit_chtc.R, training_controls.R, configs.csv, job_nums.csv, osdf:///chtc/staging/", username, "/train.sif, osdf:///chtc/staging/", username, "/", fn)
  }
  
  write(transfer_files_str, file.path(path_batch, "input", "train.sub"), append = TRUE)
  
  # add max idle jobs
  max_idle_str <- str_c("max_idle = ", max_idle)
  write(max_idle_str, file.path(path_batch, "input", "train.sub"), append = TRUE)
  
  # add cpus requested
  cpus_str <- str_c("request_cpus = ", request_cpus)
  write(cpus_str, file.path(path_batch, "input", "train.sub"), append = TRUE)
  
  # add memory requested
  memory_str <- str_c("request_memory = ", request_memory)
  write(memory_str, file.path(path_batch, "input", "train.sub"), append = TRUE)
  
  # add disk space requested
  disk_str <- str_c("request_disk = ", request_disk)
  write(disk_str, file.path(path_batch, "input", "train.sub"), append = TRUE)
  
  # add flock
  flock_str <- str_c("want_campus_pools = ", want_campus_pools)
  write(flock_str, file.path(path_batch, "input", "train.sub"), append = TRUE)
  
  # add glide
  glide_str <- str_c("want_ospool = ", want_ospool)
  write(glide_str, file.path(path_batch, "input", "train.sub"), append = TRUE)
  
  # add queue
  queue_str <- str_c("queue job_num from job_nums.csv")
  write(queue_str, file.path(path_batch, "input", "train.sub"), append = TRUE)
}
