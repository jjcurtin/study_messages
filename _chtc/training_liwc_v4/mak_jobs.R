# setup chtc jobs & associated files/folders

# NOTE: this script should be run from a project on the repo (e.g., RISK, MATCH, RISK2) 
# for the relative paths to access local copy of lab_support to work

# CHANGE ME  -------------------- 
# Set path for training_controls.R
# location of study specific training_controls.R (can use relative path if in repo) 
path_training_controls <- "./_chtc/training_liwc_v4/training_controls.R" 

# DON'T CHANGE -------------------- 
library(tidyverse)
source("https://github.com/jjcurtin/lab_support/blob/main/chtc/fun_make_jobs.R?raw=true")

# set overwrite_jobs to TRUE if you want to overwrite existing files.  Be careful!!
make_jobs(path_training_controls, overwrite_batch = TRUE)
