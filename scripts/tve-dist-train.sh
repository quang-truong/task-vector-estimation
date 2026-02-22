#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,5" 
export NUM_CUDA_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export OMP_NUM_THREADS=1

CFG_FILE="$1"
SWEEP_FILE="$2"

# Check if SWEEP_FILE is provided and not an option starting with "-"
if [ -n "$SWEEP_FILE" ] && [[ "$SWEEP_FILE" != -* ]]; then
    shift; shift
else
    SWEEP_FILE=""
    shift
fi

# Construct the base command with CFG_FILE
CMD="torchrun --master_port=29502 --nproc_per_node=$NUM_CUDA_DEVICES src/train_tve.py \
-c \"$CFG_FILE\" \
--num_runs 1 \
--num_workers 4"

# Add SWEEP_FILE to command if provided
if [ -n "$SWEEP_FILE" ]; then
    CMD="$CMD -s \"$SWEEP_FILE\""
fi

# Append any additional arguments passed to the script
CMD="$CMD $@"

# Run the command
eval $CMD