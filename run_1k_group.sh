#!/usr/bin/env bash
# 用法: bash run_1k_group.sh
# 串行训练 ImageNet-1K 的 20 组（每组 50 类），每组保存独立 checkpoint

set -e  # 任意一组失败则停止

BASE_SAVE_DIR="/root/autodl-tmp/fine-tuned-model-1k-group"
SAMPLE_DIR="/root/autodl-tmp/grpo_1k_samples"

for GROUP_ID in $(seq 14 19); do
    echo "========================================"
    echo "Training Group ${GROUP_ID} (classes $((GROUP_ID*50))-$((GROUP_ID*50+49)))"
    echo "Start: $(date)"
    echo "========================================"

    python train_multi_reward_1k.py \
        --spec 1k_group \
        --phase ${GROUP_ID} \
        --nclass 50 \
        --save-model-dir ${BASE_SAVE_DIR}/group_${GROUP_ID} \
        --save-dir ${SAMPLE_DIR}/group_${GROUP_ID} \
        --total-iters 1200 \
        --checkpoint-interval 1050 \
        --log-interval 100 \
        --batch-size 4 \
        --group-k 8 \
        --lr 1e-5 \
        --cfg-scale 4.0 \
        --alpha-cls 1.0 \
        --alpha-mst 1.0 \
        --reward-model resnet50

    echo "Group ${GROUP_ID} Done: $(date)"
    echo "========================================"
done

echo "All 20 groups finished!"
