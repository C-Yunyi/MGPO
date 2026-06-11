#!/bin/bash
set -e

# =========================================================
# Usage:
#   bash run_eval_sweep_unified_fixed.sh
#
# Change this to choose which sweep to run:
#   kl / dist / dir / reward
# =========================================================
SWEEP_TYPE="dist"

PYTHON_BIN="python"
SAMPLE_SCRIPT="sample.py"
TRAIN_SCRIPT="train2.py"

# =========================
# sample.py fixed config
# =========================
MODEL="DiT-XL/2"
IMAGE_SIZE=256
SPEC="woof"
NUM_SAMPLES=10

# =========================
# train2.py fixed config
# =========================
DATASET="imagenet"
REAL_DIR="/root/autodl-tmp/imagewoof2/"
NET_TYPE="resnet_ap"
NCLASS=10
NORM_TYPE="instance"
IPC=10
SLCT_TYPE="random"

# =========================
# checkpoint config
# =========================
# change this if your checkpoint file has another name
CKPT_NAME="ckpt_step_550.pt"

# =========================
# roots
# =========================
CKPT_BASE="/root/autodl-tmp/fine-tuned-model-multi"
SYN_BASE="/root/autodl-tmp/results_hparam_eval"
LOG_BASE="/root/autodl-tmp/student_hparam_eval_logs"

mkdir -p "${SYN_BASE}"
mkdir -p "${LOG_BASE}"

# =========================
# choose sweep
# =========================
EXP_LIST=()
SUMMARY_CSV=""

case "${SWEEP_TYPE}" in
    kl)
        EXP_LIST=(
            "klbeta_0.02"
            "klbeta_0.04"
            "klbeta_0.06"
            "klbeta_0.08"
            "klbeta_0.1"
            "klbeta_0.12"
            "klbeta_0.14"
            "klbeta_0.16"
            "klbeta_0.18"
            "klbeta_0.2"
        )
        SUMMARY_CSV="${LOG_BASE}/summary_kl.csv"
        ;;
    dist)
        EXP_LIST=(
            "lambda_mst_dist_0.2"
            "lambda_mst_dist_0.4"
            "lambda_mst_dist_0.6"
            "lambda_mst_dist_0.8"
            "lambda_mst_dist_1.0"
        )
        SUMMARY_CSV="${LOG_BASE}/summary_lambda_mst_dist.csv"
        ;;
    dir)
        EXP_LIST=(
            "lambda_mst_dir_0.2"
            "lambda_mst_dir_0.4"
            "lambda_mst_dir_0.6"
            "lambda_mst_dir_0.8"
            "lambda_mst_dir_1.0"
        )
        SUMMARY_CSV="${LOG_BASE}/summary_lambda_mst_dir.csv"
        ;;
    reward)
        EXP_LIST=(
            "reward_model_resnet18"
            "reward_model_resnet50"
            "reward_model_resnet101"
        )
        SUMMARY_CSV="${LOG_BASE}/summary_reward_model.csv"
        ;;
    *)
        echo "Error: unsupported SWEEP_TYPE=${SWEEP_TYPE}"
        echo "Use one of: kl / dist / dir / reward"
        exit 1
        ;;
esac

echo "=================================================="
echo "Running unified evaluation sweep"
echo "SWEEP_TYPE = ${SWEEP_TYPE}"
echo "SUMMARY    = ${SUMMARY_CSV}"
echo "=================================================="

echo "exp_name,student_tag,best_acc_mean,best_acc_std,last_acc_mean,last_acc_std,ckpt_path,syn_data_dir,log_dir,student_save_dir_guess" > "${SUMMARY_CSV}"

for EXP_NAME in "${EXP_LIST[@]}"; do
    CKPT_PATH="${CKPT_BASE}/${EXP_NAME}/${CKPT_NAME}"
    SYN_DIR="${SYN_BASE}/${EXP_NAME}"
    EXP_LOG_DIR="${LOG_BASE}/${EXP_NAME}"

    SAMPLE_LOG="${EXP_LOG_DIR}/sample.log"
    TRAIN_LOG="${EXP_LOG_DIR}/train.log"

    # train2.py uses tag to name save folder
    STUDENT_TAG="student_${EXP_NAME}"

    # common guessed save path pattern used by many repos
    STUDENT_SAVE_DIR_GUESS="./save/${DATASET}/${NET_TYPE}/${STUDENT_TAG}"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "Warning: checkpoint not found: ${CKPT_PATH}"
        continue
    fi

    mkdir -p "${SYN_DIR}"
    mkdir -p "${EXP_LOG_DIR}"

    echo ""
    echo "=================================================="
    echo "Experiment        : ${EXP_NAME}"
    echo "Checkpoint        : ${CKPT_PATH}"
    echo "Synthetic data dir: ${SYN_DIR}"
    echo "Log dir           : ${EXP_LOG_DIR}"
    echo "Student tag       : ${STUDENT_TAG}"
    echo "Student save guess: ${STUDENT_SAVE_DIR_GUESS}"
    echo "=================================================="

    # -------------------------------------------------
    # 1) sample
    # -------------------------------------------------
    ${PYTHON_BIN} ${SAMPLE_SCRIPT} \
        --model "${MODEL}" \
        --image-size "${IMAGE_SIZE}" \
        --ckpt "${CKPT_PATH}" \
        --save-dir "${SYN_DIR}" \
        --spec "${SPEC}" \
        --num-samples "${NUM_SAMPLES}" \
        2>&1 | tee "${SAMPLE_LOG}"

    # -------------------------------------------------
    # 2) train student network
    # -------------------------------------------------
    ${PYTHON_BIN} ${TRAIN_SCRIPT} \
        -d "${DATASET}" \
        --imagenet_dir "${SYN_DIR}" "${REAL_DIR}" \
        -n "${NET_TYPE}" \
        --nclass "${NCLASS}" \
        --norm_type "${NORM_TYPE}" \
        -i "${IPC}" \
        --tag "${STUDENT_TAG}" \
        -s "${SLCT_TYPE}" \
        --spec "${SPEC}" \
        2>&1 | tee "${TRAIN_LOG}"

    # -------------------------------------------------
    # 3) parse accuracy
    # -------------------------------------------------
    ACC_LINE=$(grep "Best, last acc:" "${TRAIN_LOG}" | tail -n 1 || true)

    if [ -n "${ACC_LINE}" ]; then
        BEST_MEAN=$(echo "${ACC_LINE}" | awk -F'Best, last acc: ' '{print $2}' | awk '{print $1}')
        BEST_STD=$(echo "${ACC_LINE}" | awk -F'Best, last acc: ' '{print $2}' | awk '{print $2}')
        LAST_MEAN=$(echo "${ACC_LINE}" | awk -F'Best, last acc: ' '{print $2}' | awk '{print $3}')
        LAST_STD=$(echo "${ACC_LINE}" | awk -F'Best, last acc: ' '{print $2}' | awk '{print $4}')
    else
        BEST_MEAN="NA"
        BEST_STD="NA"
        LAST_MEAN="NA"
        LAST_STD="NA"
    fi

    echo "${EXP_NAME},${STUDENT_TAG},${BEST_MEAN},${BEST_STD},${LAST_MEAN},${LAST_STD},${CKPT_PATH},${SYN_DIR},${EXP_LOG_DIR},${STUDENT_SAVE_DIR_GUESS}" >> "${SUMMARY_CSV}"
done

echo ""
echo "=================================================="
echo "Finished SWEEP_TYPE=${SWEEP_TYPE}"
echo "Summary saved to: ${SUMMARY_CSV}"
echo "=================================================="