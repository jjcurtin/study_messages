#!/bin/bash   
 
# NOTES
# This scripts assumes that you will call it from the root directory of your study repository.  The script is NOT run from the reports repo.  An up-to-date copy of the script is saved in reports but moved to each study repo when we set up the study repo.

# Call the script with the relative path to qmd file
# . render_report.sh subfolder1/subfolder2/report.qmd
# The rendered html file will be created and committed to reports/<study>/subfolder1/subfolder2/report.html

FILENAME=$1
FILEPATH=$(dirname $FILENAME)
STUDY=$(basename $PWD)

# Check for reports repo and clone it if it doesn't exist
if [ ! -d "../reports/" ]; then
  echo ""
  echo -e "\033[31mreport repo does not exist.  Cloning it now.\033[0m"
  git clone git@github.com:jjcurtin/reports.git ../reports/
fi 
  
# Check for study folder and subfolders in reports repo and create if it it doesn't exist
if [ ! -d "../reports/$STUDY/$FILEPATH" ]; then 
  echo ""
  echo -e "\033[31mMaking $STUDY/$FILEPATH folder\033[0m"
  mkdir "../reports/$STUDY/$FILEPATH" -p
fi
 
# Render report in study repo then move html to reports repo
echo "" 
echo -e "\033[31mRendering $FILENAME\033[0m"
quarto render "$FILENAME"
mv "${FILENAME%.*}.html" "../reports/$STUDY/$FILEPATH" 

 
# Commit report to the report repo 
echo -e "\033[31mCommmiting report for $FILENAME\033[0m"
cd ../reports
git pull
git add .
git commit -m "added report from $FILENAME"
git push
cd -
