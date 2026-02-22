#!/bin/bash

CFG_FILE="$1"
SWEEP_FILE="$2"

# Shift arguments if both CFG_FILE and SWEEP_FILE are provided
if [ -n "$SWEEP_FILE" ] && [[ "$SWEEP_FILE" != -* ]]; then
    shift; shift
else
    SWEEP_FILE=""
    shift
fi

# Base command with CFG_FILE
CMD="python src/train_tve.py \
-c \"$CFG_FILE\" \
--num_runs 1 \
--num_workers 2 \
--gpus [0]"

# Add SWEEP_FILE to command if provided
if [ -n "$SWEEP_FILE" ]; then
    CMD="$CMD -s \"$SWEEP_FILE\""
fi

# Append any additional arguments passed to the script
CMD="$CMD $@"

# Run the command
eval $CMD