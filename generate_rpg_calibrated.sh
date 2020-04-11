#!/usr/bin/env bash
dt_start=20190201
dt_end=20190301

runList=''

while [ "$dt_start" != "$dt_end" ]
	do runList=$runList' '$dt_start
	dt_start=$(date -d "$dt_start + 1 day" '+%Y%m%d')
done

for run in $runList; do python LIMRAD94_to_Cloudnet_v3.py date="$run" & done
