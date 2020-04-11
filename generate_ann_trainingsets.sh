#!/usr/bin/env bash
start_str="20190318"
end_str="20190318"
t_trn="15.0"
t_skp="15.0" # change in loop
kind="HSI"

dt_start=$(date -d "$start_str 0000" +"%Y%m%d %H%M")
dt_end=$(date -d "$end_str 2359" +"%Y%m%d %H%M")
echo "dt_begin = " $dt_start
echo "dt_end = " $dt_end

#python generate_toml.py dt_start=$start_str dt_end=$end_str t_train=$t_trn t_skip=$t_skp

runList=''

while [ "$dt_start" '<' "$dt_end" ]
	do runList=$runList' '${dt_start:0:8}"-"${dt_start:9:11}
	dt_start=$(date -d "$dt_start + 15 minute" +"%Y%m%d %H%M")
	echo "CREATE JOB: "${dt_start:0:8}"-"${dt_start:9:11}
done

echo $runList

for run in $runList; do python generate_trainingset.py dt_start="$run" t_train="$t_trn" kind="$kind" & done

