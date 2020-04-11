#!/usr/bin/env bash
dt_start=20190101
dt_end=20190201

runList=''

while [ "$dt_start" != "$dt_end" ]
	do runList=$runList' '$dt_start
	dt_start=$(date -d "$dt_start + 1 day" '+%Y%m%d')
done

for run in $runList; do python process_cloudnet.py date="$run" & done
