#!/usr/bin/env bash

# default values
t_train=60.0
kind="HSI"
site='lacros_dacapo_gpu'
cloudnet="CLOUDNETpy94"

if [[ $1 != "" ]]; then
    while [ "$1" != "" ]; do
        case $1 in
            -t0 | --dt-start )   	shift
                    start_day="$1"
                                ;;
            -t1 | --dt-end )     	shift
                    end_day="$1"
                                ;;
            -h0 | --h-start )   	shift
                    start_hour="$1"
                                ;;
            -h1 | --h-end )     	shift
                    end_hour="$1"
                                ;;
            -s | --system )     	shift
                    system="$1"
                                ;;
            -cn | --cloudnet )     	shift
                    cloudnet="$1"
                                ;;
            -ttr | --t-train )     	shift
                    t_train="$1"
                                ;;
            -site | --site )     	shift
                    site="$1"
                                ;;
            -h | --help )      	echo "help"
                              	echo "no help available"
                                ;;
            * )                	echo "unknown"
        esac
        shift
    done
else
    echo "USE PRE-DEFINED DATE"

    start_day="20190801"; end_day="20190801"; start_hour="0000"; end_hour="2359";

fi


dt_start=$(date -d "$start_day $start_hour" +"%Y%m%d %H%M")
dt_end=$(date -d "$end_day $end_hour" +"%Y%m%d %H%M")

# generate le toml file contianing the intervals
#python ./generate_toml.py dt_start="$dt_start" dt_end="$dt_end" t_train="$t_train" t_skip="$t_train"

pids=""
RESULT=0
runList=''

while [ "$dt_start" '<' "$dt_end" ]
	do runList=$runList' '${dt_start:0:8}"-"${dt_start:9:11}
	dt_start=$(date -d "$dt_start + ${t_train%.*} minute" +"%Y%m%d %H%M")
done

echo "CREATE JOB: "$runList

for run in $runList; do
    python generate_trainingset.py dt_start="$run" t_train="$t_train" kind="$kind" cnet="$cloudnet" site="$site" &
    pids="$pids $!";
done

for pid in $pids; do
    wait $pid || let "RESULT=1"
done

if [ "$RESULT" == "1" ];
    then
       exit 1
fi

echo "------DONE-----"
