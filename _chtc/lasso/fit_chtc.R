# Fit model at chtc

# libraries & source functions file ----------------
suppressWarnings(suppressPackageStartupMessages({
  require(ranger)
  require(glmnet)
  require(xgboost)
  require(discrim)
  require(nnet)
  # require(kknn)
  require(dplyr)
  require(tidyr)
  require(stringr)
  require(readr)
  require(purrr)
  require(parsnip)
  require(recipes)
  require(themis)
  require(tune)
  require(yardstick)
  require(rsample)
})) 
source("fun_chtc.R")
source("training_controls.R")  

# set up job ---------
# for testing:
#   job_num_arg <- 2
#   config_start_arg <- 2
#   config_end_arg <- 2
args <- commandArgs(trailingOnly = TRUE) 
job_num_arg <- args[1]

# Read in data train --------------- 
fn <- str_subset(list.files(), "^data_trn")
if (str_detect(fn, ".rds")) {
  d <- read_rds(fn)
} else {
  d <- read_csv(fn, show_col_types = FALSE) # supports both csv and tsv formats
}


# Format data-----------------------
# change column classes, rename Y, etc
# This is a custom/study specific function that exists in training_controls
d <- format_data(d)  


# Create splits object ---------------
splits <- d %>% 
  make_splits(cv_resample_type, cv_resample, cv_outer_resample, 
              cv_inner_resample, cv_group, cv_strat = cv_strat,
              the_seed = seed_splits)


# Get held in data ---------------

d_in <- training(splits$splits[[job_num_arg]])

# set controls ---------------
penalty_grid <- tibble(penalty = 10^seq(-6, 2, length = 50))

ctrl <- control_grid(save_pred = TRUE,
                     extract = extract_fit_parsnip)


# Lasso on Meta features ---------------
d_in_meta <- d_in |>
  select(-starts_with("label_day"), -starts_with("baseline_"))


# Build recipe
rec_meta <- build_recipe(d_in_meta)

# Split training data for penalty tuning
set.seed(102030)
splits_meta <- group_vfold_cv(d_in_meta, v = 5, repeats = 2, 
                              group = "subid", strata = "strat")


# Fit models
models_meta <- logistic_reg(penalty = tune(),
                            mixture = 1) |>
  set_engine("glmnet") |>
  set_mode("classification") |> 
  tune_grid(
    resamples = splits_meta,
    preprocessor = rec_meta,
    grid = penalty_grid,
    metrics = metric_set(roc_auc),
    control = ctrl
  )

# Get best penalties
best_penalties_meta <- models_meta |>
  collect_metrics(summarize = FALSE) |>
  group_by(id, id2) |>
  slice_max(.estimate, with_ties = FALSE) |>
  select(id, id2, penalty)

# tibble to save retained features
feats_meta <- tibble()


# loop over splits to get retained features for each of the 10 models
for (i in 1:nrow(best_penalties_meta)){
  penalty_val <- best_penalties_meta$penalty[i]
  
  # Get parsnip object
  parsnip_model <- models_meta$.extracts[[i]]$.extracts[[i]]
  
  # Extract glmnet fit and coefficients
  glmnet_fit <- parsnip_model$fit
  
  coefs <- coef(glmnet_fit, s = penalty_val) 
  
  
  # save retained coefficients
  retained_coefs <- as.matrix(coefs)
  
  feats_meta_retained <- tibble(
    feature = rownames(retained_coefs),
    coefficient = retained_coefs[,1]
  ) |>  
    filter(coefficient != 0 & feature!= "(Intercept)") |> 
    pull(feature)
  
  feats_meta <- feats_meta |> 
    bind_rows(tibble(feats_meta = feats_meta_retained))
}


# Retain features retained in at least 25% of splits

feats_meta <- feats_meta |> 
  count(feats_meta) |> 
  mutate(prop = n/nrow(splits_meta)) |> 
  filter(prop >= .25) |> 
  summarise(meta = str_c(feats_meta, collapse = ", ")) |> 
  mutate(split = job_num_arg) |> 
  select(split, meta)



# Lasso on baseline features ---------------
d_in_base <- d_in |>
  select(-starts_with("label_day"), -starts_with("meta_"))


# Build recipe
rec_base <- build_recipe(d_in_base)

# Split training data for penalty tuning
set.seed(102030)
splits_base <- group_vfold_cv(d_in_base, v = 5, repeats = 2, 
                              group = "subid", strata = "strat")


# Fit models
models_base <- logistic_reg(penalty = tune(),
                            mixture = 1) |>
  set_engine("glmnet") |>
  set_mode("classification") |> 
  tune_grid(
    resamples = splits_base,
    preprocessor = rec_base,
    grid = penalty_grid,
    metrics = metric_set(roc_auc),
    control = ctrl
  )

# Get best penalties
best_penalties_base <- models_base |>
  collect_metrics(summarize = FALSE) |>
  group_by(id, id2) |>
  slice_max(.estimate, with_ties = FALSE) |>
  select(id, id2, penalty)

# tibble to save retained features
feats_base <- tibble()


# loop over splits to get retained features for each of the 10 models
for (i in 1:nrow(best_penalties_base)){
  penalty_val <- best_penalties_base$penalty[i]
  
  # Get parsnip object
  parsnip_model <- models_base$.extracts[[i]]$.extracts[[i]]
  
  # Extract glmnet fit and coefficients
  glmnet_fit <- parsnip_model$fit
  
  coefs <- coef(glmnet_fit, s = penalty_val) 
  
  
  # save retained coefficients
  retained_coefs <- as.matrix(coefs)
  
  feats_base_retained <- tibble(
    feature = rownames(retained_coefs),
    coefficient = retained_coefs[,1]
  ) |>  
    filter(coefficient != 0 & feature!= "(Intercept)") |> 
    pull(feature)
  
  feats_base <- feats_base |> 
    bind_rows(tibble(feats_base = feats_base_retained))
}


# Retain features retained in at least 50% of splits
original_base_names <-  d_in_base |> names()

feats_base <- feats_base |> 
  count(feats_base) |> 
  mutate(prop = n/nrow(splits_base)) |> 
  filter(prop >= .5) |> 
  mutate(feats_base = if_else(!feats_base %in% original_base_names,
                              str_replace(feats_base, "_[^_]+$", ""),
                              feats_base)) |> 
  summarise(baseline = str_c(feats_base, collapse = ", ")) |> 
  mutate(split = job_num_arg) |> 
  select(split, baseline)

# Combine features and save---------------

feats_meta |> 
  left_join(feats_base, by = "split") |> 
  write_csv(str_c("results_", job_num_arg, ".csv"))
