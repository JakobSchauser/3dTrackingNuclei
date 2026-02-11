#!/bin/bash
#
# ===== Default Configuration =====
PARTITION="main"
GPUS="1"
CPUS="16"
MEM="64G"
TIME="04:00:00"
PORT="8888"
ENV="ultrack_pipeline"

# ===== Parse arguments (e.g., gpu:2 cpu:16 mem:64G env:my_env) =====
for arg in "$@"; do
    # Each argument must contain a colon osco.py
    if [[ "$arg" != *:* ]]; then
        echo "❌ Invalid argument format: '$arg'"
        echo "   Expected format: key:value (e.g., gpu:2 cpu:16 env:my_env)"
        exit 1
    fi

    key="${arg%%:*}"   # part before :
    val="${arg#*:}"    # part after :

    case "$key" in
        gpu|gpus)          GPUS="$val" ;;
        cpu|cpus)          CPUS="$val" ;;
        mem|memory)        MEM="$val" ;;
        time)              TIME="$val" ;;
        port)              PORT="$val" ;;
        env|environment)   ENV="$val" ;;
        part|partition)    PARTITION="$val" ;;
        *)
            echo "❌ Error: Unknown argument '$key'"
            echo "   Allowed keys: gpu, cpu, mem, time, port, env, partition"
            exit 1
            ;;
    esac
done

# ===== Display configuration =====
echo "===== Interactive session configuration ====="
echo "Partition: $PARTITION"
echo "GPUs:      $GPUS"
echo "CPUs:      $CPUS"
echo "Memory:    $MEM"
echo "Time:      $TIME"
echo "Port:      $PORT"
echo "Env:       $ENV"
echo "============================================="

# ===== Load conda properly =====
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"

# ===== Start interactive GPU session and launch Jupyter =====
echo "Starting interactive session with GPU..."
srun --partition="$PARTITION" --gres=gpu:"$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --pty \
    bash -c "
        echo 'Loading environment...'
        source ~/.bashrc
        echo 'Starting Jupyter on port $PORT...'
        jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
    "

