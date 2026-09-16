#!/usr/bin/env bash
# NegoCollab Stage 1 (nego) then Stage 2 (receiver ft).
set -euo pipefail

VEH_DIR=${VEH_DIR:-opencood/logs/airv2x_homo_vehicle_camera/homo_vehicle}
RSU_DIR=${RSU_DIR:-opencood/logs/airv2x_homo_rsu_camera/homo_rsu}
DRONE_DIR=${DRONE_DIR:-opencood/logs/airv2x_homo_drone_camera/homo_drone}

python opencood/tools/train_airv2x_heter.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_negocollab/airv2x_negocollab_stage1_nego.yaml \
  --vehicle_dir "${VEH_DIR}" \
  --rsu_dir "${RSU_DIR}" \
  --drone_dir "${DRONE_DIR}" \
  --tag nego_stage1

# Stage 2 continues from the Stage-1 run directory via --model_dir after
# Stage 1 finishes. Example:
# python opencood/tools/train_airv2x_heter.py \
#   -y opencood/hypes_yaml/airv2x/camera/det/airv2x_negocollab/airv2x_negocollab_stage2_ft.yaml \
#   --vehicle_dir "${VEH_DIR}" --rsu_dir "${RSU_DIR}" --drone_dir "${DRONE_DIR}" \
#   --model_dir <stage1_log_dir> \
#   --tag nego_stage2
