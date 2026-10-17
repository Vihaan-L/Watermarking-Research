#!/bin/bash

# Default values
TASK_NAME="humaneval"
MODEL="infly/OpenCoder-1.5B-Instruct"
TRAIN_BATCH_SIZE=64
NUM_EPOCHS=3
OUTPUT_DIR="outputs/sft-humaneval"
GPUS="4"
PORT=29523
AUGMENTATION="false"
LIMIT=0
ALPHA_DISTILL=0.6
ALPHA_CE=0.2
ALPHA_SWITCH=0.2
CONTEXT_WIDTH=2
USE_CACHE="false"

# Environment variables
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=$GPUS
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"

# Run the training script with accelerate
echo "Running training script with GPU: $GPUS"

accelerate launch \
    --num_processes 1 \
    --main_process_port $PORT \
    scripts/run_sft.py \
    --task_name $TASK_NAME \
    --model $MODEL \
    --output_dir $OUTPUT_DIR \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --alpha_ce $ALPHA_CE \
    --alpha_switch $ALPHA_SWITCH \
    --context_width $CONTEXT_WIDTH 