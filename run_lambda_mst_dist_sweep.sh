#!/bin/bash
set -e

PYTHON_BIN="python"
TRAIN_SCRIPT="train_multi_reward.py"

DATA_SPEC="woof"
MST_ROOT="./mst"

BASE_SAMPLE_ROOT="/root/autodl-tmp/grpo_multi_samples"
BASE_CKPT_ROOT="/root/autodl-tmp/fine-tuned-model-multi"

# default config
MODEL_NAME="DiT-XL/2"
IMAGE_SIZE=256
NUM_CLASSES=1000
LATENT_CHANNELS=4
DEVICE="cuda"
SEED=0

T_TOTAL=1000
ETA_SDE=1.0
CFG_SCALE=4.0
NUM_SAMPLING_STEPS=50

BATCH_SIZE=4
GROUP_K=8
LR=1e-5
TOTAL_ITERS=600
CLIP_RANGE=0.1
ADV_EPS=1e-8
KL_BETA=0.1
LAMBDA_ODE=0.0

ALPHA_CLS=1.0
ALPHA_MST=1.0
COMBINE_MODE="weighted_sum"

# controlled variable: lambda-mst-dist
LAMBDA_MST_DIR=0.0
DIST_LIST=(0.2 0.4 0.6 0.8 1.0)

REWARD_MODEL="resnet50"
VAE="mse"

LOG_INTERVAL=100
CHECKPOINT_INTERVAL=550

mkdir -p "${BASE_SAMPLE_ROOT}"
mkdir -p "${BASE_CKPT_ROOT}"

for LAMBDA_MST_DIST in "${DIST_LIST[@]}"; do
    EXP_NAME="lambda_mst_dist_${LAMBDA_MST_DIST}"
    SAVE_DIR="${BASE_SAMPLE_ROOT}/${EXP_NAME}"
    SAVE_MODEL_DIR="${BASE_CKPT_ROOT}/${EXP_NAME}"
    LOG_FILE="${SAVE_MODEL_DIR}/train.log"

    mkdir -p "${SAVE_DIR}"
    mkdir -p "${SAVE_MODEL_DIR}"

    echo "======================================"
    echo "Running experiment: ${EXP_NAME}"
    echo "save_dir       = ${SAVE_DIR}"
    echo "save_model_dir = ${SAVE_MODEL_DIR}"
    echo "======================================"

    ${PYTHON_BIN} ${TRAIN_SCRIPT} \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --model-name "${MODEL_NAME}" \
        --image-size "${IMAGE_SIZE}" \
        --num-classes "${NUM_CLASSES}" \
        --latent-channels "${LATENT_CHANNELS}" \
        --t-total "${T_TOTAL}" \
        --eta-sde "${ETA_SDE}" \
        --cfg-scale "${CFG_SCALE}" \
        --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --group-k "${GROUP_K}" \
        --lr "${LR}" \
        --total-iters "${TOTAL_ITERS}" \
        --clip-range "${CLIP_RANGE}" \
        --adv-eps "${ADV_EPS}" \
        --kl-beta "${KL_BETA}" \
        --lambda-ode "${LAMBDA_ODE}" \
        --alpha-cls "${ALPHA_CLS}" \
        --alpha-mst "${ALPHA_MST}" \
        --combine-mode "${COMBINE_MODE}" \
        --mst-root "${MST_ROOT}" \
        --lambda-mst-dist "${LAMBDA_MST_DIST}" \
        --lambda-mst-dir "${LAMBDA_MST_DIR}" \
        --spec "${DATA_SPEC}" \
        --reward-model "${REWARD_MODEL}" \
        --vae "${VAE}" \
        --save-dir "${SAVE_DIR}" \
        --save-model-dir "${SAVE_MODEL_DIR}" \
        --log-interval "${LOG_INTERVAL}" \
        --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
        2>&1 | tee "${LOG_FILE}"
done

echo "All lambda-mst-dist sweep experiments finished."