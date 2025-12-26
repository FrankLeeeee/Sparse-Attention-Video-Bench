#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/..

DIMENSION=$1
START=${2:-"-1"}
END=${3:-"-1"}
SEED=${4:-1024}

# ensure DIMENSION is given
if [ -z "$DIMENSION" ]; then
    echo "DIMENSION is not set"
    exit 1
fi

python $ROOT_DIR/scripts/wan_t2v_inference.py  \
    --strategy svg \
    --sparsity 0.75 \
    --height 720 \
    --width 1280 \
    --prompt_file $SCRIPT_DIR/prompts/optimized_${DIMENSION}.txt \
    --start-index $START \
    --end-index $END \
    --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --warmup_step 1 \
    --num_inference_steps 50 \
    --num_frames 81 \
    --seed $SEED \
    --output_dir $SCRIPT_DIR/results/wan1.3b/svg/$DIMENSION/SEED_$SEED
