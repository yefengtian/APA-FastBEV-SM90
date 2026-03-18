#!/usr/bin/env bash
set -euo pipefail

# ====== 可改参数 ======
GPUS=4
CONFIG="/vePFS/Daniel/Fast-BEV-dev/configs/fastbev/exp/paper/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_psd_new.py"
WORK_DIR="/vePFS/Daniel/Fast-BEV-dev/work_dir400_hook/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_psd_new"
LOG_FILE="${WORK_DIR}/log.txt"
PROJECT_DIR="/vePFS/Daniel/Fast-BEV-dev"

# 如果要固定具体卡，打开这一行
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# ====== 准备目录 ======
mkdir -p "${WORK_DIR}"

echo "[INFO] CONFIG      = ${CONFIG}"
echo "[INFO] WORK_DIR    = ${WORK_DIR}"
echo "[INFO] LOG_FILE    = ${LOG_FILE}"
echo "[INFO] GPUS        = ${GPUS}"

# ====== 后台启动 ======
nohup bash -lc "
  trap '' HUP
  cd ${PROJECT_DIR}
  torchrun --nproc_per_node=${GPUS} tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    --work-dir ${WORK_DIR}
" > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
disown || true

echo "[INFO] Started. PID=${PID}"
echo "[INFO] Tail log: tail -f ${LOG_FILE}"
echo "[INFO] Check:    ps -p ${PID} -o pid,ppid,tty,stat,cmd"