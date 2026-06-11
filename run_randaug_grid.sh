#!/bin/bash

mkdir -p logs_lr_grid

n=2
m=8

for lr in $(seq 0.015 0.001 0.035); do
  lr_tag=$(printf "%.3f" "$lr")
  echo "Running with randaug_n=$n, randaug_m=$m, lr=$lr_tag"

  python train2.py \
    -d imagenet \
    --imagenet_dir ../autodl-tmp/logs/imagenet100_group_ipc10/train ../autodl-tmp/imagenet \
    -n resnet_ap \
    --nclass 100 \
    --norm_type instance \
    --ipc 10 \
    --tag test_n${n}_m${m}_lr${lr_tag} \
    --slct_type random \
    --spec 100 \
    --lr $lr_tag \
    --randaug True \
    --randaug_n $n \
    --randaug_m $m \
    > logs_lr_grid/train_n${n}_m${m}_lr${lr_tag}.log 2>&1
done