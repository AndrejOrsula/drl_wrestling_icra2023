#!/usr/bin/env bash

## The command to run inside the container
PROCESS_CMD=(
    webots
    /usr/local/webots-project/worlds/wrestling.wbt
    --no-rendering
    "${@}"
)

## The minimum temperature and power draw above which the process is assumed to be running
GPU_MIN_TEMPERATURE=45
GPU_MIN_POWER_DRAW=50
GPU_QUERY_COUNT=5
GPU_QUERY_PERIOD=1

## Sleep times (in seconds) between various checks
SLEEP_TIME_INITIAL=600
SLEEP_TIME_RECURRENT=120
SLEEP_TIME_POST_KILL=5

IS_PROCESS_RUNNING="false"
PROCESS_PID=""
PROCESS_COUNT=0
while true; do
    if [ "${IS_PROCESS_RUNNING}" = "false" ]; then
        PROCESS_COUNT=$((PROCESS_COUNT+1))
        echo -e "Starting process #${PROCESS_COUNT}..."
        # shellcheck disable=SC2048
        ${PROCESS_CMD[*]} &
        PROCESS_PID="$!"
        IS_PROCESS_RUNNING="true"
        sleep "${SLEEP_TIME_INITIAL}"
    fi

    SHOULD_KILL="true"
    for _ in $(seq 1 "${GPU_QUERY_COUNT}"); do
        GPU_TEMPERATURE="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)"
        GPU_POWER_DRAW="$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)"
        GPU_POWER_DRAW="${GPU_POWER_DRAW%.*}"
        if [ "${GPU_TEMPERATURE}" -gt "${GPU_MIN_TEMPERATURE}" ] \
            || [ "${GPU_POWER_DRAW}" -gt "${GPU_MIN_POWER_DRAW}" ]; then
            SHOULD_KILL="false"
            break
        fi
        sleep "${GPU_QUERY_PERIOD}"
    done
    if [ "${SHOULD_KILL}" = "true" ]; then
        echo -e "Killing the process..."
        kill -- -$(ps -o pgid= $PID | grep -o [0-9]*) > /dev/null 2>&1
        sleep "${SLEEP_TIME_POST_KILL}"
    fi

    if ! ps -p "${PROCESS_PID}" > /dev/null
    then
        echo -e "The process is not running anymore. Restarting...\n"
        IS_PROCESS_RUNNING="false"
    fi

    if [ "${IS_PROCESS_RUNNING}" = "true" ]; then
        echo -n "|"
        sleep "${SLEEP_TIME_RECURRENT}"
    fi
done
