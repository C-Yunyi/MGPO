#!/bin/bash
set -e

for f in \
    run_kl_beta_sweep.sh \
    run_lambda_mst_dist_sweep.sh \
    run_lambda_mst_dir_sweep.sh \
    run_reward_model_sweep.sh
do
    if [ ! -f "$f" ]; then
        echo "Error: $f not found."
        exit 1
    fi
done

#echo "=============================="
#echo "[1/4] KL beta sweep"
#echo "=============================="
#bash run_kl_beta_sweep.sh

echo "=============================="
echo "[2/4] lambda-mst-dist sweep"
echo "=============================="
bash run_lambda_mst_dist_sweep.sh

echo "=============================="
echo "[3/4] lambda-mst-dir sweep"
echo "=============================="
bash run_lambda_mst_dir_sweep.sh

echo "=============================="
echo "[4/4] reward-model sweep"
echo "=============================="
bash run_reward_model_sweep.sh

echo "All fine-tuning sweeps finished."