#!/usr/bin/env bash

# default values
t_trn="15.0"
t_skp="15.0" # change in loop

kind="HSI"

if [[ $1 != "" ]]
    then
    echo "USE COMMAND LINE INPUT ARGS:  $1"

    while [ "$1" != "" ]; do
        case $1 in
            -t0 | --dt-start )   	shift
                    echo "start_day = $1"
                    start_day="$1"
                                ;;
            -t1 | --dt-end )     	shift
                    echo "end_day = $1"
                    end_day="$1"
                                ;;
            -h0 | --h-start )   	shift
                    echo "start_hour = $1"
                    start_hour="$1"
                                ;;
            -h1 | --h-end )     	shift
                    echo "end_hour = $1"
                    end_hour="$1"
                                ;;
            -s | --system )     	shift
                    echo "system = $1"
                    system="$1"
                                ;;
            -cn | --cloudnet )     	shift
                    echo "cloudnet = $1"
                    cloudnet="$1"
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

    start_day="20190313"; end_day="20190313"; start_hour="0600"; end_hour="1945"

fi


dt_start=$(date -d "$start_day $start_hour" +"%Y%m%d %H%M")
dt_end=$(date -d "$end_day $end_hour" +"%Y%m%d %H%M")
echo "dt_begin = " $dt_start
echo "dt_end = " $dt_end

#python generate_toml.py dt_start="$dt_start" dt_end="$dt_end" t_train="$t_trn" t_skip="$t_skp"
pids=""
RESULT=0
runList=''

while [ "$dt_start" '<' "$dt_end" ]
	do runList=$runList' '${dt_start:0:8}"-"${dt_start:9:11}
	dt_start=$(date -d "$dt_start + 15 minute" +"%Y%m%d %H%M")
done

echo "CREATE JOB: "$runList

for run in $runList; do
    python generate_trainingset.py dt_start="$run" t_train="$t_trn" kind="$kind" cnet="$cloudnet" &
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
