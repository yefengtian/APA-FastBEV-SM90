#!/bin/bash
EXPORT PYTHONPATH=/vePFS/Daniel/Fast-BEV-dev
JSON_DIR="/vePFS/val_data/2026_2_1_full_size/jsons/"
CONFIG="/vePFS/Daniel/Fast-BEV-dev/configs/fastbev/exp/paper/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_psd_type.py"
CHECKPOINT="/vePFS/Daniel/Fast-BEV-dev/work_dir400_hook/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_psd_new/epoch_300.pth"
DATA_DIR="/vePFS/data/zhixing_data/"
OUT_DIR="/vePFS/Daniel/Fast-BEV-dev/metrics_ps_ws/output"

NUM_WORKERS=4
GPUS="0,1,2,3"

IOU_THRESH=0.8
KP_THRESH=0.2
MAX_R=5.0

# 生成 tag：0.8 -> 80, 0.1 -> 10
IOU_TAG=$(python -c "print(int(round(float('$IOU_THRESH')*100)))")
KP_TAG=$(python -c "print(int(round(float('$KP_THRESH')*100)))")
R_TAG="${MAX_R}m"

OUT_FILE="${OUT_DIR}/report_full_size_${R_TAG}_iou${IOU_TAG}_kp${KP_TAG}_mp.json"

echo "Output file: $OUT_FILE"
echo "Start multi-process evaluation..."

python /vePFS/Daniel/Fast-BEV-dev/metrics_ps_ws/main_mul.py \
  --json_dir "$JSON_DIR" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data_dir "$DATA_DIR" \
  --out "$OUT_FILE" \
  --num_workers "$NUM_WORKERS" \
  --gpus "$GPUS" \
  --iou_thresh "$IOU_THRESH" \
  --kp_thresh "$KP_THRESH" \
  --max_r "$MAX_R"

echo "Done."
