# Training controls for phone study

# SET GLOBAL PARAMETERS--------------------
study <- "messages"
window <- "day"
lead <- 0
version <- "v1"
algorithm <- "random_forest"
batch <- "combined"
roll_dur <- 24

configs_per_job <- 50  # number of model configurations that will be fit/evaluated within each CHTC

# Source function
source("https://github.com/jjcurtin/lab_support/blob/main/format_path.R?raw=true")

# RESAMPLING FOR OUTCOME-----------------------------------
# note that ratio is under_ratio, which is used by downsampling as is
# It is converted to  overratio (1/ratio) for up and smote
resample <- c("up_1", "up_2", "up_3", "up_4", "up_5", 
              "down_1", "down_2", "down_3", "down_4", "down_5") 


# DATA, SPLITS AND OUTCOME-------------------------------------
feature_set <- c("ind_all", "ind_3day", "ind_7day") 
data_trn <- str_c("features_", batch, "_", version, ".csv") 
seed_splits <- 102030

ml_mode <- "classification"   # regression or classification
y_col_name <- "lapse" 
y_level_pos <- "yes" 
y_level_neg <- "no"


# CV SETTINGS---------------------------------
cv_resample_type <- "kfold" # can be boot, kfold, or nested
cv_resample = "6_x_5" # can be repeats_x_folds (e.g., 1_x_10, 10_x_10) or number of bootstraps (e.g., 100)
cv_inner_resample <- NULL # can also be a single number for bootstrapping (i.e., 100)
cv_outer_resample <- NULL # outer resample will always be kfold
cv_group <- "subid" # set to NULL if not grouping
cv_strat <- TRUE # set to FALSE if not stratifying - If TRUE you must have a strat variable in your data

cv_name <- if_else(cv_resample_type == "nested",
                   str_c(cv_resample_type, "_", cv_inner_resample, "_",
                         cv_outer_resample),
                   str_c(cv_resample_type, "_", cv_resample))

# STUDY PATHS----------------------------
# the name of the batch of jobs to set folder name
name_batch <- str_c("train_", algorithm, "_", cv_name, "_", version, "_", batch) 
# the path to the batch of jobs
path_batch <- format_path(str_c("risk/chtc/", study, "/", name_batch))
# location of data set
path_data <- format_path("risk/data_processed/messages") 


# ALGORITHM-SPECIFIC HYPERPARAMETERS-----------
hp1_glmnet <- c(0.05, seq(.1, 1, length.out = 10)) # alpha (mixture)
hp2_glmnet_min <- -8 # min for penalty grid - will be passed into exp(seq(min, max, length.out = out))
hp2_glmnet_max <- 2 # max for penalty grid
hp2_glmnet_out <- 200 # length of penalty grid

hp1_knn <- seq(5, 255, length.out = 26) # neighbors (must be integer)


hp1_rf <- c(5, 10, 20, 50, 75) # mtry (p/3 for reg or square root of p for class)
hp2_rf <- c(2, 10, 20, 50) # min_n
hp3_rf <- 1500 # trees (10 x's number of predictors)

hp1_xgboost <- c(0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, .4)  # learn_rate
hp2_xgboost <- c(1, 2, 3, 4, 5) # tree_depth
hp3_xgboost <- seq(100, 900, 100)  # mtry
# trees = 500
# early stopping = 20

hp1_rda <- seq(.1, 1, length.out = 10)  # frac_common_cov: Fraction of the Common Covariance Matrix (0-1; 1 = LDA, 0 = QDA)
hp2_rda <- seq(.1, 1, length.out = 10) # frac_identity: Fraction of the Identity Matrix (0-1)
 
hp1_nnet <- seq(10, 100, length.out = 10)  # epochs
hp2_nnet <- seq(0, 0.1, length.out = 100) # penalty
hp3_nnet <- seq(5, 30, length.out = 5) # hidden units

# CHTC SPECIFIC CONTROLS----------------------------
username <- "jyu274" # for setting staging directory (until we have group staging folder)
stage_data <- FALSE # If FALSE .sif will still be staged, just not data_trn
max_idle <- 1000
request_cpus <- 1 
request_memory <- "90000MB"
request_disk <- "3000MB"
want_campus_pools <- TRUE # previously flock
want_ospool <- TRUE # previously glide


# FORMAT DATA-----------------------------------------
format_data <- function (df){
  
  df |>  
    rename(y = !!y_col_name) |>  
    mutate(y = factor(y, levels = c(!!y_level_pos, !!y_level_neg)), # set pos class first
           across(where(is.character), factor)) |>  
    select(-dttm_label)
}


# BUILD RECIPE---------------------------------------
# Script should have a single build_recipe function to be compatible with fit script. 
build_recipe <- function(d, config) {
  # d: (training) dataset from which to build recipe
  # config: single-row config-specific tibble
  
  # get relevant info from job (algorithm, feature_set, resample, under_ratio)
  algorithm <- config$algorithm
  feature_set <- config$feature_set
  
  if (config$resample == "none") {
    resample <- config$resample
  } else {
    resample <- str_split(config$resample, "_")[[1]][1]
    ratio <- as.numeric(str_split(config$resample, "_")[[1]][2])
  }
  
  # Set recipe steps generalizable to all model configurations
  rec <- recipe(y ~ ., data = d) %>%
    step_rm(subid)

  if(cv_strat) {
    rec <- rec |> 
      step_rm(strat) # remove strat variable
  }
  
  rec <- rec |>
    step_zv(all_predictors()) |> 
    step_impute_median(all_numeric_predictors()) |> 
    step_impute_mode(all_nominal_predictors()) 
  
  # resampling options for unbalanced outcome variable
  if (resample == "down") {
    # under_ratio = ratio.  No conversion needed
    rec <- rec %>% 
      themis::step_downsample(y, under_ratio = ratio, seed = 10) 
  }
  
  if (resample == "smote") {
    # correct ratio to over_ratio
    rec <- rec %>% 
      themis::step_smote(y, over_ratio = 1 / ratio, seed = 10) 
  }
  
  if (resample == "up") {
    # correct ratio to over_ratio
    rec <- rec %>% 
      themis::step_upsample(y, over_ratio = 1 / ratio, seed = 10)
  }

  # select down to features for feature_set
  # no need to select down if feature_set is all
  
  if (str_detect(config$feature_set, "3day")){
      rec <- rec |> step_rm(contains("7day"))
  }
  if (str_detect(config$feature_set, "7day")){
      rec <- rec |> step_rm(contains("3day"))
  }
  

  
  # algorithm specific steps
  if (algorithm == "glmnet") {
    rec <- rec  %>%
      step_dummy(all_nominal_predictors()) %>%
      step_normalize(all_predictors())
  } 
  
  if (algorithm == "random_forest") {
    # no algorithm specific steps
  } 
  
  if (algorithm == "xgboost") {
    rec <- rec  %>% 
      step_dummy(all_nominal_predictors())
  } 
  
  # final steps for all algorithms
  rec <- rec %>%
    # drop columns with NA values after imputation (100% NA)
    step_rm(where(~ any(is.na(.)))) %>%
    step_nzv()
  
  return(rec)
}
