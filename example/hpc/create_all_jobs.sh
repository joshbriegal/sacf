#!/bin/bash

array=($(cat ngtshead_cmds.txt | grep -oh "NG[0-9][0-9][0-9][0-9][+|-][0-9][0-9][0-9][0-9]"))
#array=($(ls /home/jtb34/rds/hpc-work/GACF_OUTPUTS | grep NG))
for each in "${array[@]}"
do
  # files=$(ls /home/jtb34/rds/hpc-work/GACF_OUTPUTS/$each/*.fits 2> /dev/null | wc -l)
  files=$(ls /home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/$each/*.fits 2> /dev/null | wc -l)
  echo $files
  if (( $files != 0 ))
  then
      # echo "/home/jtb34/rds/hpc-work/GACF_OUTPUTS/$each"
      echo "/home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/$each" 
      continue
  else
      bash /home/jtb34/GitHub/GACF/example/hpc/create_job.sh "$each" "y" "y" "y"
  fi
done
