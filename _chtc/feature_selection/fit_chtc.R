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
#   job_num_arg <- 12

args <- commandArgs(trailingOnly = TRUE) 
job_num_arg <- args[1] 
job_num_arg <- as.numeric(args[1]) + 1

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
penalty_grid <- expand.grid(penalty = 10^seq(-6, 0, length = 20),
                       mixture = seq(.3, .8, .1))

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
                            mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("classification") |> 
  tune_grid(
    resamples = splits_meta,
    preprocessor = rec_meta,
    grid = penalty_grid,
    metrics = metric_set(roc_auc)
  )

# Get best penalties
best_params <- models_meta |> 
  collect_metrics(summarize = FALSE) |> 
  group_by(id, id2, mixture, penalty) |> 
  summarise(mean = mean(.estimate), .groups = "drop_last") |> 
  group_by(id, id2) |> 
  slice_max(mean, n = 1, with_ties = FALSE) |> 
  ungroup()


# Refit best models on each analysis split to get coefficients
all_coefs <- map2_dfr(best_params$id, best_params$id2, function(id, id2) {
  
  row <- best_params |>  
    filter(id == !!id, id2 == !!id2)
  
  split_obj <- splits_meta$splits[[which(splits_meta$id == id & splits_meta$id2 == id2)]]
  
  split_d_in <- analysis(split_obj)
  
  rec_prepped <- rec_meta |> 
    prep(training = split_d_in)
  
  feat_split_d_in <- rec_prepped |> 
    bake(new_data = NULL)
  
  final_model <- logistic_reg(penalty = row$penalty,
                       mixture = row$mixture) |> 
    set_engine("glmnet") |> 
    set_mode("classification") |> 
    fit(formula = y ~ ., data = feat_split_d_in)
  
  coefs <- coef(extract_fit_engine(final_model), s = row$penalty)
  
  tibble(term = rownames(coefs),
         estimate = as.numeric(coefs)) |> 
    filter(term != "(Intercept)") |> 
    mutate(
      selected = estimate != 0,
      split_id = paste(id, id2, sep = "_"))
})


# Aggregate features retained to assess stability of important features
stability_meta <- all_coefs |> 
  group_by(term) |> 
  summarise(times_present = sum(selected),
            times_available = n(),
            prop = times_present / times_available,
            mean = mean(abs(estimate)), 
            .groups = "drop") 

# retain by proportion cutoff .5 (consider different proportions) - keep at least 40 feats
prop_meta <- stability_meta |>  
  filter(prop >= .5)

if (nrow(prop_meta) < 40) {
  prop_meta <- prop_meta |> 
    bind_rows(stability_meta |> 
                filter(prop >= .1 & prop < .5) |> 
                arrange(desc(prop), desc(mean)) |>
                slice_head(n = 40 - nrow(prop_meta)))
}

feats_meta <- prop_meta |> 
  summarise(meta = str_c(term, collapse = ", ")) |> 
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
                            mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("classification") |> 
  tune_grid(
    resamples = splits_base,
    preprocessor = rec_base,
    grid = penalty_grid,
    metrics = metric_set(roc_auc)
  )

# Get best penalties
best_params_base <- models_base |> 
  collect_metrics(summarize = FALSE) |> 
  group_by(id, id2, mixture, penalty) |> 
  summarise(mean = mean(.estimate), .groups = "drop_last") |> 
  group_by(id, id2) |> 
  slice_max(mean, n = 1, with_ties = FALSE) |> 
  ungroup()


# Refit best models on each analysis split to get coefficients
all_coefs <- map2_dfr(best_params_base$id, best_params_base$id2, function(id, id2) {
  
  row <- best_params_base |>  
    filter(id == !!id, id2 == !!id2)
  
  split_obj <- splits_base$splits[[which(splits_base$id == id & splits_base$id2 == id2)]]
  
  split_d_in <- analysis(split_obj)
  
  rec_prepped <- rec_base |> 
    prep(training = split_d_in)
  
  feat_split_d_in <- rec_prepped |> 
    bake(new_data = NULL)
  
  final_model <- logistic_reg(penalty = row$penalty,
                              mixture = row$mixture) |> 
    set_engine("glmnet") |> 
    set_mode("classification") |> 
    fit(formula = y ~ ., data = feat_split_d_in)
  
  coefs <- coef(extract_fit_engine(final_model), s = row$penalty)
  
  tibble(term = rownames(coefs),
         estimate = as.numeric(coefs)) |> 
    filter(term != "(Intercept)") |> 
    mutate(
      selected = estimate != 0,
      split_id = paste(id, id2, sep = "_"))
})


# Aggregate features retained to assess stability of important features
stability_base <- all_coefs |> 
  group_by(term) |> 
  summarise(times_present = sum(selected),
            times_available = n(),
            prop = times_present / times_available,
            mean = mean(abs(estimate)), 
            .groups = "drop") 

# Retain top 5-10 features (by prop of splits variables retained in and coef)
prop_base <- stability_base |>  
  filter(prop >= .8)

if (nrow(prop_base) < 5) {
  prop_base <- prop_base |> 
    bind_rows(stability_base |> 
                filter(prop >= .7 & prop < .8) |> 
                arrange(desc(mean)) |> 
                slice_head(n = 10 - nrow(prop_base)))
}

if (nrow(prop_base) > 10) {
  prop_base <- prop_base |> 
    arrange(desc(mean)) |> 
    slice_head(n = 10)
}

original_base_names <-  d_in_base |> names()

feats_base <- prop_base |> 
  mutate(feats_base = if_else(!term %in% original_base_names,
                              str_replace(term, "_[^_]+$", ""),
                              term)) |> 
  mutate(split = job_num_arg) |> 
  select(split, baseline = feats_base) |> 
  unique() |> 
  summarise(baseline = str_c(baseline, collapse = ", ")) 
  

# Combine features and save---------------

feats_meta |> 
  left_join(feats_base, by = "split") |> 
  write_csv(str_c("results_", job_num_arg, ".csv"))
