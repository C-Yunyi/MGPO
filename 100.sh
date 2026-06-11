#!/usr/bin/env bash

STEP=550
IPC=10

for GROUP_ID in 0 1 2 3 4
do
    echo "========================================"
    echo "Running GROUP_ID: ${GROUP_ID}"
    echo "Classes: $((GROUP_ID*20))-$((GROUP_ID*20+19))"
    echo "Start time: $(date)"
    echo "========================================"

    CKPT="/root/autodl-tmp/fine-tuned-model-100-group20/group_${GROUP_ID}/ckpt_step_${STEP}.pt"
    SAVE_DIR="/root/autodl-tmp/logs/imagenet100_group_ipc${IPC}/train/group_${GROUP_ID}"

    mkdir -p ${SAVE_DIR}

    python sample.py \
        --model DiT-XL/2 \
        --image-size 256 \
        --ckpt ${CKPT} \
        --spec 100 \
        --num-samples ${IPC} \
        --cfg-scale 4.0 \
        --num-sampling-steps 50 \
        --nclass 20 \
        --phase ${GROUP_ID} \
        --save-dir ${SAVE_DIR}

    echo "Finished GROUP_ID: ${GROUP_ID}"
    echo "Finish time: $(date)"
    echo ""
done