# Training controls

# FORMAT PATH FUNCTION------
library(stringr)
library(dplyr)
source("https://github.com/jjcurtin/lab_support/blob/main/format_path.R?raw=true")

# SET GLOBAL PARAMETERS------
study <- "messages"
window <- "day"
lead <- 0
version <- "v10"
algorithm <- "lasso"
batch <- "mak_features"


# DATA, SPLITS AND OUTCOME------
feature_set <- c("all") 
data_trn <- str_c("features_meta_", window, "_24h_", version, ".csv") 
seed_splits <- 102030

ml_mode <- "classification"   # regression or classification
y_col_name <- "lapse" 
y_level_pos <- "yes" 
y_level_neg <- "no"


# CV SETTINGS------
cv_resample_type <- "kfold" # can be boot, kfold, or nested
cv_resample = "20_x_5" # can be repeats_x_folds (e.g., 1_x_10, 10_x_10) or number of bootstraps (e.g., 100)
cv_inner_resample <- NULL # can also be a single number for bootstrapping (i.e., 100)
cv_outer_resample <- NULL # outer resample will always be kfold
cv_group <- "subid" # set to NULL if not grouping
cv_strat <- TRUE # set to FALSE if not stratifying - If TRUE you must have a strat variable in your data
# IMPORTANT - NEED TO REMOVE STRATIFY VARIABLE FROM DATA IN RECIPE - See Recipe below for example code


cv_name <- if_else(cv_resample_type == "nested",
                   str_c(cv_resample_type, "_", cv_inner_resample, "_",
                         cv_outer_resample),
                   str_c(cv_resample_type, "_", cv_resample))

# STUDY PATHS------
# the name of the batch of jobs to set folder name
name_batch <- str_c("train_", algorithm, "_", cv_name, "_", version, "_", batch) 
# the path to the batch of jobs
path_batch <- format_path(str_c("risk/chtc/", study, "/", name_batch)) 
# location of data set
path_data <- format_path("risk/data_processed/shared") 


# CHTC SPECIFIC CONTROLS------
username <- "kpaquette2"
stage_data = FALSE
max_idle <- 1000
request_cpus <- 1 
request_memory <- "90000MB"
request_disk <- "3000MB"
want_campus_pools <- TRUE # previously flock
want_ospool <- TRUE # previously glide


# FORMAT DATA------
format_data <- function (df, lapse_strat = NULL){
  
  df <- df |> 
    rename(y = !!y_col_name) |> 
    # set pos class first
    mutate(y = factor(y, levels = c(!!y_level_pos, !!y_level_neg)), 
           across(where(is.character), factor)) |>
    select(-c(dttm_label)) 
  
  return(df)
}




# BUILD RECIPE------
# Script should have a single build_recipe function to be compatible with fit script. 
build_recipe <- function(d) {
  # d: (training) dataset from which to build recipe
  
  rec <- recipe(y ~ ., data = d)  |> 
    step_rm(label_num, subid, strat) |>
    step_zv(all_predictors()) |> 
    step_impute_median(all_numeric_predictors()) |> 
    step_impute_mode(all_nominal_predictors()) |> 
    step_dummy(all_nominal_predictors()) |>  # dummy here for lasso
    step_select(where(~ !any(is.na(.)))) |>
    step_zv(all_predictors())  |> 
    step_nzv(all_predictors()) |> 
    step_normalize(all_predictors())
  
  
  return(rec)
}




