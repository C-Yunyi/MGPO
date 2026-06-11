#!/usr/bin/env bash
STEP=${1:-1050}
IPC=${2:-10}


SAVE_DIR="/root/autodl-tmp/imagenet1k_group_ipc${IPC}/train"
mkdir -p logs
mkdir -p ${SAVE_DIR}

for GROUP_ID in $(seq 0 19); do
    CKPT="/root/autodl-tmp/fine-tuned-model-1k-group/group_${GROUP_ID}/ckpt_step_${STEP}.pt"
    
    echo "========================================"
    echo "Sample 1K Group ${GROUP_ID} (classes $((GROUP_ID*50))-$((GROUP_ID*50+49)))"
    echo "CKPT: ${CKPT}, IPC: ${IPC}"
    echo "Start: $(date)"
    echo "========================================"
    
    python sample.py \
        --model DiT-XL/2 \
        --image-size 256 \
        --ckpt ${CKPT} \
        --spec 1k \
        --num-samples ${IPC} \
        --cfg-scale 4.0 \
        --num-sampling-steps 50 \
        --nclass 50 \
        --phase ${GROUP_ID} \
        --save-dir ${SAVE_DIR}
    
    echo "Group ${GROUP_ID} Done: $(date)"
    echo "========================================"
done

echo "All groups finished!"