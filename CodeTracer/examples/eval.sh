#!/bin/bash

# Default values
MODEL="infly/OpenCoder-1.5B-Instruct"
TASK="humaneval"
WM="coderl"
GPUS="3"
CODE_MODEL="outputs/orm-masked-adv5/checkpoint-300/"
# CODE_MODEL="none"
BEST=false
SWITCH_THRESHOLD="0.5"
SEED=42
DELTA=2.0
CONTEXT_WIDTH=2
ENTROPY_THRESHOLD=1.2
LIMIT=20

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export HF_ALLOW_CODE_EVAL=1
export CPATH="$CONDA_PREFIX/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Run the main script with HumanEval configuration
python scripts/run_eval.py \
    --model ${MODEL} \
    --task_name ${TASK} \
    --output_dir ./outputs/eval \
    --batch_size 20 \
    --max_length 2048 \
    --precision "bf16" \
    --temperature 0.2 \
    --top_p 0.95 \
    --n_samples 20 \
    --wm ${WM} \
    --gamma 0.5 \
    --delta ${DELTA} \
    --postprocess true \
    --allow_code_execution true \
    --detection_z_threshold 4.0 \
    --context_width ${CONTEXT_WIDTH} \
    --switch_threshold ${SWITCH_THRESHOLD} \
    --entropy_threshold ${ENTROPY_THRESHOLD} \
    --seed ${SEED} \
    --code_model ${CODE_MODEL} \
    --limit ${LIMIT}