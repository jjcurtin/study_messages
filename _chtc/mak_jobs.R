# setup chtc jobs & associated files/folders

library(tidyverse) 
devtools::source_url("https://github.com/jjcurtin/lab_support/blob/main/chtc/fun_make_jobs.R?raw=true")

path_training_controls <- file.path("_chtc/training_controls_messages_concat.R") 
make_jobs(path_training_controls, overwrite_batch = FALSE)


