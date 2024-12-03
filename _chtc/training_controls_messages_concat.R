# Training controls for messages study

# NOTES------------------------------

# v1: initial version with messages as separate rows per label (vs concatenated)
# v2: concatenated messages

# Batches done:

# Batches to do:


# SET GLOBAL PARAMETERS--------------------
study <- "messages"
window <- "1day"
lead <- 0
version <- "v2" 
algorithm <- "xgboost"
model <- "main"

feature_set <- c("all_raw", "3day_raw", "1week_raw", "all_norm",
                 "3day_norm", "1week_norm") # messages feature set name
data_trn <- str_c("liwc_features_cat.csv")

seed_splits <- 102030

ml_mode <- "classification"   # regression or classification
configs_per_job <- 200 # number of model configurations that will be fit/evaluated within each CHTC

# RESAMPLING FOR OUTCOME-----------------------------------
# note that ratio is under_ratio, which is used by downsampling as is
# It is converted to  overratio (1/ratio) for up and smote
# memory constraints with SMOTE
# daily lapse base rate ~ 7% (~ 13:1 majority to minority cases)
resample <- c("none", "up_1", "up_2", "up_3", "up_4", "up_5", "down_1", "down_2", "down_3", "down_4", "down_5")

# CHTC SPECIFIC CONTROLS------ ---------------------
# tar <- c("train.tar.gz") # name of tar packages for submit file - does not transfer these anywhere 
max_idle <- 1000
request_cpus <- 1 
request_memory <- "5000MB"
request_disk <- "5000MB"
flock <- FALSE
glide <- FALSE


# OUTCOME-------------------------------------
y_col_name <- "lapse" 
y_level_pos <- "lapse" 
y_level_neg <- "no lapse"


# CV SETTINGS---------------------------------
cv_resample_type <- "nested" # can be boot, kfold, or nested
cv_resample = NULL # can be repeats_x_folds (e.g., 1_x_10, 10_x_10) or number of bootstraps (e.g., 100)
cv_inner_resample <- "1_x_10" # can also be a single number for bootstrapping (i.e., 100)
cv_outer_resample <- "3_x_10" # outer resample will always be kfold
cv_group <- "subid" # set to NULL if not grouping

cv_name <- if_else(cv_resample_type == "nested",
                   str_c(cv_resample_type, "_", cv_inner_resample, "_",
                         cv_outer_resample),
                   str_c(cv_resample_type, "_", cv_resample))

# STUDY PATHS----------------------------
# the name of the batch of jobs to set folder name
name_batch <- str_c("train_", algorithm, "_", cv_name, "_", version, "_", model) 
# the path to the batch of jobs to put the folder name
path_batch <- str_c("studydata/risk/chtc/", study, "/", name_batch) 
# location of data set
path_data <- str_c("studydata/risk/data_processed/", study, "/liwc") 

# ALGORITHM-SPECIFIC HYPERPARAMETERS-----------
#hp1_glmnet <- c(0.05, seq(.1, 1, length.out = 10)) # alpha (mixture)
#hp2_glmnet_min <- -8 # min for penalty grid - will be passed into exp(seq(min, max, length.out = out))
#hp2_glmnet_max <- 2 # max for penalty grid
#hp2_glmnet_out <- 200 # length of penalty grid

#hp1_knn <- seq(5, 255, length.out = 26) # neighbors (must be integer)

#hp1_rf <- c(2, 10, 20, 30, 40) # mtry (p/3 for reg or square root of p for class)
#hp2_rf <- c(2, 15, 30) # min_n
#hp3_rf <- 1500 # trees (10 x's number of predictors)

hp1_xgboost <- c(0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, .4)  # learn_rate, how fast model fits residual error; high: faster, but may overshoot, low: slower, may get stuck on less optimal solutions
hp2_xgboost <- c(1, 2, 3, 4) # tree_depth, complexity of tree structure (larger no. = more likely to overfit)
hp3_xgboost <- c(10, 15, 20, 30, 40, 50)  # mtry, no. feats. to split on at each split
# trees = 500
# early stopping = 20

#hp1_rda <- seq(0.1, 1, length.out = 10)  # frac_common_cov: Fraction of the Common Covariance Matrix (0-1; 1 = LDA, 0 = QDA)
#hp2_rda <- seq(0.1, 1, length.out = 10) # frac_identity: Fraction of the Identity Matrix (0-1)
 
#hp1_nnet <- seq(10, 50, length.out = 5)  # epochs
#hp2_nnet <- seq(0, 0.1, length.out = 15) # penalty
#hp3_nnet <- seq(5, 30, length.out = 5) # hidden units

# FORMAT DATA-----------------------------------------
format_data <- function (df){
  
  df |> 
    rename(y = !!y_col_name) |> 
    # set pos class first
    mutate(y = factor(y, levels = c(!!y_level_pos, !!y_level_neg)), 
           across(where(is.character), factor)) |>
    select(-day_start)  
  # Now include additional mutates to change classes for columns as needed
  # see https://jjcurtin.github.io/dwt/file_and_path_management.html#using-a-separate-mutate
}


# BUILD RECIPE---------------------------------------
# Script should have a single build_recipe function to be compatible with fit script. 
build_recipe <- function(d, config) {
  # d: (training) dataset from which to build recipe
  # job: single-row job-specific tibble
  
  # get relevant info from job (algorithm, feature_set, resample, under_ratio)
  algorithm <- config$algorithm
  
  if (config$resample == "none") {
    resample <- config$resample
  } else {
    resample <- str_split(config$resample, "_")[[1]][1]
    ratio <- as.numeric(str_split(config$resample, "_")[[1]][2])
  }
  
  # Set recipe steps generalizable to all model configurations
  rec <- recipe(y ~ ., data = d) |>
    step_rm(subid, label_num) |>  # needed to retain until now for grouped CV in splits
    step_impute_median(all_numeric_predictors()) |> 
    step_impute_mode(all_nominal_predictors()) |> 
    step_dummy(all_factor_predictors()) |> 
    step_select(where(~ !any(is.na(.)))) |>
    step_nzv(all_predictors())
 
  
  # resampling options for unbalanced outcome variable
  if (resample == "down") {
    rec <- rec |> 
      # ratio is equivalent to tidymodels under_ratio
      themis::step_downsample(y, under_ratio = ratio, seed = 10) 
  }
  
  
  if (resample == "smote") {
    ratio <- 1 / ratio # correct ratio to over_ratio
    rec <- rec |> 
      themis::step_smote(y, over_ratio = ratio, seed = 10) 
  }
  
  if (resample == "up") {
    ratio <- 1 / ratio # correct ratio to over_ratio
    rec <- rec |> 
      themis::step_upsample(y, over_ratio = ratio, seed = 10)
  }
  
  # select down to features for feature_set
  # no need to select down if feature_set is all 

  if(str_detect(config$feature_set, "raw")){
    rec <- rec |>
      step_select(-contains("norm"))
    if(str_starts(config$feature_set, "3day")){
      rec <- rec |>
        step_select(-ends_with("1week"))
    }
    if(str_starts(config$feature_set, "1week")){
      rec <- rec |>
        step_select(-ends_with("3day"))
    }
  }
  if(str_detect(config$feature_set, "norm")){
    rec <- rec |>
      step_select(-contains("raw"))
    if(str_starts(config$feature_set, "3day")){
      rec <- rec |>
        step_select(-ends_with("1week"))
    }
    if(str_starts(config$feature_set, "1week")){
      rec <- rec |>
        step_select(-ends_with("3day"))
    }
  }

  return(rec)
}


# Update paths for OS--------------------------------
# This does NOT need to be edited.  This will work for Windows, Mac and Linux OSs
path_batch <- case_when(Sys.info()[["sysname"]] == "Windows" ~str_c("P:/", path_batch),
                        Sys.info()[["sysname"]] == "Linux" ~str_c("~/mnt/private/", path_batch),
                        .default = str_c("/Volumes/private/", path_batch))

path_data <- case_when(Sys.info()[["sysname"]] == "Windows" ~str_c("P:/", path_data),
                       Sys.info()[["sysname"]] == "Linux" ~str_c("~/mnt/private/", path_data),
                       .default = str_c("/Volumes/private/", path_data))
