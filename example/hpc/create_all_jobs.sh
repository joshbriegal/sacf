#!/bin/bash

array=($(cat ngtshead_cmds.txt | grep -oh "NG[0-9][0-9][0-9][0-9][+|-][0-9][0-9][0-9][0-9]"))
for each in "${array[@]}"
do
  files=$(ls /home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/$each/*.* 2> /dev/null | wc -l)
  # echo "$each $files"
  if (( $files < 11 ))
  then
    size=$(du -s /home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/$each | awk '{print $1}')
    if (( $size < 1000000 )) # less than 1GB implies data not there
    then
      # echo "$each ($files) $(du -sh /home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/$each)"
      # rm -r "/home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/$each/*"
      bash /home/jtb34/GitHub/GACF/example/hpc/create_job.sh "$each" "y" "y" "y" # copy all files
    fi
  #     echo "/home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/$each" 
  #     continue
  else
      bash /home/jtb34/GitHub/GACF/example/hpc/create_job.sh "$each" "y" "n" "n" # don't copy files, just send sbatch
  fi
done
